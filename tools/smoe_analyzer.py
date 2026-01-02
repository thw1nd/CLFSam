# tools/smoe_analyzer.py
# Paper-ready analysis for the SMoE refinement head (FRN/p1)
#
# What you get after running analyzer.save(out_dir):
#   - smoe_stats.json              (full stats for reproducibility)
#   - smoe_global.csv              (all/boundary/interior summary; copy into Excel/LaTeX)
#   - smoe_per_class.csv           (per-class routing + uncertainty)
#   - smoe_per_class_boundary.csv  (per-class routing on boundary band)
#   - smoe_per_class_interior.csv  (per-class routing on interior)
#   - smoe_global_table.tex        (LaTeX snippet for global table)
#   - smoe_per_class_table.tex     (LaTeX snippet for per-class table)
#   - (optional) quick_plots/*.png (if matplotlib is available)
#
# Hooks to attach (recommended):
#   - p1.gate -> logits/mix
#   - p1.exp_local / exp_context / exp_bottl -> expert outputs (for contribution stats)
#
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


@torch.no_grad()
def boundary_mask_from_gt(gt: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """
    gt: [B,H,W] (long)
    return: [B,H,W] bool, boundary band (dilated)
    """
    B, H, W = gt.shape
    g = gt.float().unsqueeze(1)  # [B,1,H,W]
    gpad = F.pad(g, (1, 1, 1, 1), mode="replicate").squeeze(1)  # [B,H+2,W+2]
    center = gt

    # 4-neighborhood diff
    up = gpad[:, 0:H, 1:W + 1].long()
    down = gpad[:, 2:H + 2, 1:W + 1].long()
    left = gpad[:, 1:H + 1, 0:W].long()
    right = gpad[:, 1:H + 1, 2:W + 2].long()

    b = (center != up) | (center != down) | (center != left) | (center != right)  # [B,H,W]

    if radius > 0:
        b = F.max_pool2d(
            b.float().unsqueeze(1),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        ).squeeze(1) > 0
    return b


def _safe_div(a: float, b: float) -> float:
    return float(a / max(1.0, b))


def _mean_std(sum1: float, sum2: float, n: float) -> Tuple[float, float]:
    """Return (mean, std) from sum and sum of squares."""
    if n <= 1:
        m = _safe_div(sum1, n)
        return m, 0.0
    m = sum1 / n
    var = max(0.0, (sum2 / n) - m * m)
    return float(m), float(math.sqrt(var))


def _hist_update(hist: torch.Tensor, x: torch.Tensor, edges: torch.Tensor) -> None:
    """
    hist: [nbins] float64 CPU tensor
    x: arbitrary shape on any device
    edges: [nbins+1] float tensor on CPU
    """
    # clamp to edges range
    x = x.detach()
    x = torch.clamp(x, float(edges[0].item()), float(edges[-1].item()))
    # bucketize returns index in [0..nbins], subtract 1 -> [0..nbins-1]
    idx = torch.bucketize(x.reshape(-1).cpu(), edges, right=False) - 1
    idx = torch.clamp(idx, 0, hist.numel() - 1)
    binc = torch.bincount(idx, minlength=hist.numel()).double()
    hist += binc


@dataclass
class RegionStats:
    # pixel counts
    pix: float = 0.0

    # mix sums: sum over pixels of mix (per expert)
    mix_sum: Optional[torch.Tensor] = None  # [K] float64 CPU

    # top1 counts (per expert)
    top1_cnt: Optional[torch.Tensor] = None  # [K] float64 CPU

    # entropy sums
    ent_sum: float = 0.0
    ent_sq_sum: float = 0.0

    # margin sums (top1 - top2)
    margin_sum: float = 0.0
    margin_sq_sum: float = 0.0

    # dominance counts: for each thr, for each expert count(mix_i >= thr)
    dom_cnt: Optional[torch.Tensor] = None  # [T,K] float64 CPU

    # histograms
    ent_hist: Optional[torch.Tensor] = None  # [E] float64 CPU
    margin_hist: Optional[torch.Tensor] = None  # [M] float64 CPU

    # contribution sums: sum over pixels of mean(|mix_i*e_i|)  (per expert)
    contrib_sum: Optional[torch.Tensor] = None  # [K] float64 CPU

    def init(self, K: int, T: int, E: int, M: int):
        self.mix_sum = torch.zeros(K, dtype=torch.float64)
        self.top1_cnt = torch.zeros(K, dtype=torch.float64)
        self.dom_cnt = torch.zeros(T, K, dtype=torch.float64)
        self.ent_hist = torch.zeros(E, dtype=torch.float64)
        self.margin_hist = torch.zeros(M, dtype=torch.float64)
        self.contrib_sum = torch.zeros(K, dtype=torch.float64)


class SMoEAnalyzer:
    """
    Paper-ready SMoE routing analysis.

    Compared to your original version, this extends:
      - entropy std + histogram (all / boundary / interior)
      - top1 margin (top1-top2) mean/std + histogram (a measure of decisiveness)
      - dominance ratio at thresholds (e.g., mix>=0.5 / 0.7) per expert
      - per-class: mix_mean, top1_ratio, entropy_mean/std, margin_mean/std, dominance ratios
      - optional correctness-conditioned stats (if you pass pred to step)

    Usage (same as before):
      - hooks write logits/mix (and optional expert outputs) into analyzer._cache
      - call analyzer.step(gt, pred=pred_optional, img_id=optional)
      - after loop: analyzer.save(out_dir)
    """
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        eps: float = 1e-8,
        expert_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
        boundary_radius: int = 2,
        dominance_thresholds: Sequence[float] = (0.5, 0.7),
        entropy_bins: int = 30,
        margin_bins: int = 30,
        store_per_image: bool = False,
    ):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

        self.K = 3
        self.expert_names = list(expert_names) if expert_names is not None else ["local", "context", "bottleneck"]
        assert len(self.expert_names) == self.K, "expert_names must have length 3"

        self.class_names = list(class_names) if class_names is not None else None
        if self.class_names is not None:
            assert len(self.class_names) == self.num_classes, "class_names length must match num_classes"

        self.boundary_radius = int(boundary_radius)

        self.dominance_thresholds = [float(x) for x in dominance_thresholds]
        self.T = len(self.dominance_thresholds)

        # hist edges
        # entropy range: [0, log(K)]
        self.ent_edges = torch.linspace(0.0, float(math.log(self.K) + 1e-6), steps=int(entropy_bins) + 1)
        self.margin_edges = torch.linspace(0.0, 1.0, steps=int(margin_bins) + 1)

        self.store_per_image = bool(store_per_image)
        self._per_image: List[dict] = []

        self.reset()
        self._cache: Dict[str, torch.Tensor] = {}  # hooks write here

    def reset(self):
        E = self.ent_edges.numel() - 1
        M = self.margin_edges.numel() - 1
        self.region_all = RegionStats()
        self.region_b = RegionStats()
        self.region_i = RegionStats()
        for r in (self.region_all, self.region_b, self.region_i):
            r.init(self.K, self.T, E, M)

        # per-class regions
        self.cls_pix = torch.zeros(self.num_classes, dtype=torch.float64)
        self.cls_stats_all = [RegionStats() for _ in range(self.num_classes)]
        self.cls_stats_b = [RegionStats() for _ in range(self.num_classes)]
        self.cls_stats_i = [RegionStats() for _ in range(self.num_classes)]
        for c in range(self.num_classes):
            self.cls_stats_all[c].init(self.K, self.T, E, M)
            self.cls_stats_b[c].init(self.K, self.T, E, M)
            self.cls_stats_i[c].init(self.K, self.T, E, M)

        # correctness-conditioned (optional): correct vs wrong pixels
        self.enable_correctness = False
        self.region_corr = RegionStats()
        self.region_wrong = RegionStats()
        self.region_corr.init(self.K, self.T, E, M)
        self.region_wrong.init(self.K, self.T, E, M)

        self.n_images = 0

        self._per_image = []

    # ---------- hook interface ----------
    def cache(self, name: str, tensor: torch.Tensor):
        self._cache[name] = tensor.detach()

    def cache_logits(self, logits: torch.Tensor):
        self._cache["logits"] = logits.detach()
        self._cache["mix"] = F.softmax(logits.detach(), dim=1)

    # ---------- internal update helpers ----------
    @torch.no_grad()
    def _update_region(
        self,
        reg: RegionStats,
        mix: torch.Tensor,          # [B,K,h,w]
        valid_mask: torch.Tensor,   # [B,h,w] bool
        ent: torch.Tensor,          # [B,h,w] float
        margin: torch.Tensor,       # [B,h,w] float
        top1: torch.Tensor,         # [B,h,w] long
        top1v: torch.Tensor,        # [B,h,w] float
        experts: Optional[List[torch.Tensor]] = None,  # list of [B,C,h,w]
    ):
        vc = float(valid_mask.sum().item())
        if vc <= 0:
            return

        reg.pix += vc

        # mix sums
        mv = (mix * valid_mask.unsqueeze(1).float()).sum(dim=(0, 2, 3))  # [K]
        reg.mix_sum += mv.double().cpu()

        # top1 counts
        tc = torch.bincount(top1[valid_mask].reshape(-1), minlength=self.K).double().cpu()
        reg.top1_cnt += tc

        # entropy sums and sumsqs
        ev = ent[valid_mask]
        reg.ent_sum += float(ev.sum().item())
        reg.ent_sq_sum += float((ev * ev).sum().item())

        # margin (top1-top2) sums and sumsqs
        mv2 = margin[valid_mask]
        reg.margin_sum += float(mv2.sum().item())
        reg.margin_sq_sum += float((mv2 * mv2).sum().item())

        # dominance ratios at thresholds
        for ti, thr in enumerate(self.dominance_thresholds):
            for k in range(self.K):
                cnt = float((mix[:, k][valid_mask] >= thr).sum().item())
                reg.dom_cnt[ti, k] += cnt

        # histograms
        _hist_update(reg.ent_hist, ev, self.ent_edges)
        _hist_update(reg.margin_hist, mv2, self.margin_edges)

        # contribution energy (optional)
        if experts is not None:
            # mean over channel of |mix_i * e_i| -> [B,h,w]
            for k in range(self.K):
                cm = (mix[:, k:k + 1] * experts[k]).abs().mean(dim=1)  # [B,h,w]
                reg.contrib_sum[k] += float(cm[valid_mask].sum().item())

    @torch.no_grad()
    def step(
        self,
        gt: torch.Tensor,
        pred: Optional[torch.Tensor] = None,
        img_id: Optional[Union[str, int]] = None,
    ):
        """
        gt:   [B,H,W] long (auto resized to mix resolution)
        pred: [B,H,W] long (optional; used to compute correct vs wrong routing behavior)
        img_id: optional identifier (if store_per_image=True)
        """
        assert "mix" in self._cache, "mix not found. Did you attach hooks on p1.gate?"
        mix = self._cache["mix"]  # [B,K,h,w]
        B, K, h, w = mix.shape

        if gt.dim() == 4:
            gt = gt.squeeze(1)
        gt = gt.long()

        # resize GT to mix resolution
        if (gt.shape[-2], gt.shape[-1]) != (h, w):
            gt = F.interpolate(gt.unsqueeze(1).float(), size=(h, w), mode="nearest").long().squeeze(1)

        valid = (gt != self.ignore_index)
        vcount = int(valid.sum().item())
        if vcount == 0:
            self._cache.clear()
            return

        self.n_images += B

        # boundary / interior masks
        bmask = boundary_mask_from_gt(gt, radius=self.boundary_radius) & valid
        imask = (~bmask) & valid

        # entropy per pixel
        ent = -(mix * (mix + self.eps).log()).sum(dim=1)  # [B,h,w]

        # top1, top2, margin
        topv, topi = torch.topk(mix, k=2, dim=1)  # [B,2,h,w]
        top1v = topv[:, 0]
        top2v = topv[:, 1]
        margin = (top1v - top2v).clamp(min=0.0, max=1.0)
        top1 = mix.argmax(dim=1)  # [B,h,w]

        # expert outputs (optional)
        experts = None
        if all(k in self._cache for k in ["e_local", "e_context", "e_bottl"]):
            experts = [self._cache["e_local"], self._cache["e_context"], self._cache["e_bottl"]]

        # update regions
        self._update_region(self.region_all, mix, valid, ent, margin, top1, top1v, experts=experts)
        self._update_region(self.region_b, mix, bmask, ent, margin, top1, top1v, experts=experts)
        self._update_region(self.region_i, mix, imask, ent, margin, top1, top1v, experts=experts)

        # correctness-conditioned stats (optional)
        if pred is not None:
            self.enable_correctness = True
            if pred.dim() == 4:
                pred = pred.squeeze(1)
            pred = pred.long()
            if (pred.shape[-2], pred.shape[-1]) != (h, w):
                pred = F.interpolate(pred.unsqueeze(1).float(), size=(h, w), mode="nearest").long().squeeze(1)
            corr = (pred == gt) & valid
            wrong = (pred != gt) & valid
            self._update_region(self.region_corr, mix, corr, ent, margin, top1, top1v, experts=experts)
            self._update_region(self.region_wrong, mix, wrong, ent, margin, top1, top1v, experts=experts)

        # per-class stats (all/boundary/interior)
        # NOTE: for paper tables, this is usually enough (avoid heavy per-image storage)
        for c in range(self.num_classes):
            cmask = (gt == c) & valid
            ccnt = int(cmask.sum().item())
            if ccnt == 0:
                continue
            self.cls_pix[c] += ccnt
            self._update_region(self.cls_stats_all[c], mix, cmask, ent, margin, top1, top1v, experts=experts)
            self._update_region(self.cls_stats_b[c], mix, (cmask & bmask), ent, margin, top1, top1v, experts=experts)
            self._update_region(self.cls_stats_i[c], mix, (cmask & imask), ent, margin, top1, top1v, experts=experts)

        # optional per-image summary (tiny memory)
        if self.store_per_image:
            # per-image mean entropy and mean mix (valid pixels)
            # (use the "all" region of this batch only, not global)
            ent_img = []
            mix_img = []
            for bi in range(B):
                vm = valid[bi]
                if int(vm.sum().item()) == 0:
                    continue
                ent_img.append(float(ent[bi][vm].mean().item()))
                mix_img.append([float(mix[bi, k][vm].mean().item()) for k in range(self.K)])
            if len(ent_img) > 0:
                self._per_image.append({
                    "img_id": None if img_id is None else str(img_id),
                    "entropy_mean": float(sum(ent_img) / len(ent_img)),
                    "mix_mean": [float(sum(x[k] for x in mix_img) / len(mix_img)) for k in range(self.K)],
                })

        self._cache.clear()

    # ---------- export helpers ----------
    def _region_to_dict(self, reg: RegionStats, prefix: str = "") -> Dict:
        d: Dict[str, object] = {}
        pix = float(reg.pix)

        d[prefix + "pixels"] = int(pix)
        d[prefix + "mix_mean"] = [_safe_div(float(reg.mix_sum[k].item()), pix) for k in range(self.K)]
        d[prefix + "top1_ratio"] = [_safe_div(float(reg.top1_cnt[k].item()), pix) for k in range(self.K)]

        ent_mean, ent_std = _mean_std(reg.ent_sum, reg.ent_sq_sum, pix)
        d[prefix + "entropy_mean"] = ent_mean
        d[prefix + "entropy_std"] = ent_std

        mar_mean, mar_std = _mean_std(reg.margin_sum, reg.margin_sq_sum, pix)
        d[prefix + "margin_mean"] = mar_mean
        d[prefix + "margin_std"] = mar_std

        # dominance ratios
        dom = []
        for ti, thr in enumerate(self.dominance_thresholds):
            dom.append({
                "thr": thr,
                "ratio": [_safe_div(float(reg.dom_cnt[ti, k].item()), pix) for k in range(self.K)],
            })
        d[prefix + "dominance"] = dom

        # contribution means (if experts were hooked)
        d[prefix + "contrib_mean"] = [_safe_div(float(reg.contrib_sum[k].item()), pix) for k in range(self.K)]

        # histograms
        d[prefix + "entropy_hist"] = {
            "edges": [float(x) for x in self.ent_edges.tolist()],
            "counts": [int(x) for x in reg.ent_hist.tolist()],
        }
        d[prefix + "margin_hist"] = {
            "edges": [float(x) for x in self.margin_edges.tolist()],
            "counts": [int(x) for x in reg.margin_hist.tolist()],
        }
        return d

    def finalize(self) -> Dict:
        out: Dict[str, object] = {}
        out["experts"] = self.expert_names
        out["num_classes"] = self.num_classes
        out["class_names"] = self.class_names
        out["ignore_index"] = self.ignore_index
        out["boundary_radius"] = self.boundary_radius
        out["dominance_thresholds"] = self.dominance_thresholds
        out["n_images"] = int(self.n_images)

        # regions
        out["all"] = self._region_to_dict(self.region_all)
        out["boundary"] = self._region_to_dict(self.region_b)
        out["interior"] = self._region_to_dict(self.region_i)

        # a very paper-friendly indicator: boundary bias (boundary - interior)
        bias = [out["boundary"]["mix_mean"][k] - out["interior"]["mix_mean"][k] for k in range(self.K)]
        out["boundary_bias_mix_mean"] = bias

        # correctness-conditioned
        if self.enable_correctness:
            out["correct"] = self._region_to_dict(self.region_corr)
            out["wrong"] = self._region_to_dict(self.region_wrong)

        # per-class
        per_class_all = []
        per_class_b = []
        per_class_i = []
        for c in range(self.num_classes):
            name = str(c) if self.class_names is None else self.class_names[c]
            per_class_all.append({
                "class": c,
                "name": name,
                **self._region_to_dict(self.cls_stats_all[c]),
            })
            per_class_b.append({
                "class": c,
                "name": name,
                **self._region_to_dict(self.cls_stats_b[c]),
            })
            per_class_i.append({
                "class": c,
                "name": name,
                **self._region_to_dict(self.cls_stats_i[c]),
            })

        out["per_class"] = per_class_all
        out["per_class_boundary"] = per_class_b
        out["per_class_interior"] = per_class_i

        # optional per-image
        if self.store_per_image:
            out["per_image"] = self._per_image

        return out

    # ---------- saving (json + csv + latex + quick plots) ----------
    def _write_csv_global(self, out_dir: Path, stats: Dict) -> None:
        path = out_dir / "smoe_global.csv"
        headers = [
            "region", "pixels",
            *[f"mix_mean_{n}" for n in self.expert_names],
            *[f"top1_ratio_{n}" for n in self.expert_names],
            "entropy_mean", "entropy_std",
            "margin_mean", "margin_std",
        ]
        # dominance columns
        for thr in self.dominance_thresholds:
            headers += [f"dom@{thr}_{n}" for n in self.expert_names]
        # contribution
        headers += [f"contrib_mean_{n}" for n in self.expert_names]

        rows = []
        for region_key in ["all", "boundary", "interior"]:
            r = stats[region_key]
            row = {
                "region": region_key,
                "pixels": r["pixels"],
                "entropy_mean": r["entropy_mean"],
                "entropy_std": r["entropy_std"],
                "margin_mean": r["margin_mean"],
                "margin_std": r["margin_std"],
            }
            for k, n in enumerate(self.expert_names):
                row[f"mix_mean_{n}"] = r["mix_mean"][k]
                row[f"top1_ratio_{n}"] = r["top1_ratio"][k]
                row[f"contrib_mean_{n}"] = r["contrib_mean"][k]
            # dominance
            dom = r["dominance"]
            for di, thr in enumerate(self.dominance_thresholds):
                ratios = dom[di]["ratio"]
                for k, n in enumerate(self.expert_names):
                    row[f"dom@{thr}_{n}"] = ratios[k]
            rows.append(row)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_csv_per_class(self, out_dir: Path, key: str, filename: str) -> None:
        stats = self.finalize()
        items = stats[key]
        path = out_dir / filename
        headers = [
            "class", "name", "pixels",
            *[f"mix_mean_{n}" for n in self.expert_names],
            *[f"top1_ratio_{n}" for n in self.expert_names],
            "entropy_mean", "entropy_std",
            "margin_mean", "margin_std",
        ]
        for thr in self.dominance_thresholds:
            headers += [f"dom@{thr}_{n}" for n in self.expert_names]
        headers += [f"contrib_mean_{n}" for n in self.expert_names]

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in items:
                row = {
                    "class": r["class"],
                    "name": r["name"],
                    "pixels": r["pixels"],
                    "entropy_mean": r["entropy_mean"],
                    "entropy_std": r["entropy_std"],
                    "margin_mean": r["margin_mean"],
                    "margin_std": r["margin_std"],
                }
                for k, n in enumerate(self.expert_names):
                    row[f"mix_mean_{n}"] = r["mix_mean"][k]
                    row[f"top1_ratio_{n}"] = r["top1_ratio"][k]
                    row[f"contrib_mean_{n}"] = r["contrib_mean"][k]
                dom = r["dominance"]
                for di, thr in enumerate(self.dominance_thresholds):
                    ratios = dom[di]["ratio"]
                    for k, n in enumerate(self.expert_names):
                        row[f"dom@{thr}_{n}"] = ratios[k]
                writer.writerow(row)

    def _latex_escape(self, s: str) -> str:
        return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    def _write_latex_tables(self, out_dir: Path, stats: Dict) -> None:
        # Global table: all/boundary/interior
        lines = []
        lines.append(r"% Auto-generated by tools/smoe_analyzer.py")
        lines.append(r"\begin{tabular}{lrrrrrr}")
        lines.append(r"\toprule")
        # Columns: region, mix_local, mix_context, mix_bottleneck, entropy, margin, boundary_bias(optional)
        lines.append(
            r"Region & $w_\mathrm{local}$ & $w_\mathrm{context}$ & $w_\mathrm{bottl}$ & Entropy$\downarrow$ & Margin$\uparrow$ & $\Delta w_\mathrm{local}$ \\")
        lines.append(r"\midrule")
        b_bias_local = stats["boundary_bias_mix_mean"][0]
        for region_key in ["all", "boundary", "interior"]:
            r = stats[region_key]
            w0, w1, w2 = r["mix_mean"]
            ent = r["entropy_mean"]
            mar = r["margin_mean"]
            delta_local = b_bias_local if region_key == "boundary" else (0.0 if region_key != "interior" else -b_bias_local)
            lines.append(f"{region_key} & {w0:.3f} & {w1:.3f} & {w2:.3f} & {ent:.3f} & {mar:.3f} & {delta_local:.3f} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        (out_dir / "smoe_global_table.tex").write_text("\n".join(lines), encoding="utf-8")

        # Per-class table: keep it compact (top1 ratios)
        items = stats["per_class"]
        lines = []
        lines.append(r"% Auto-generated by tools/smoe_analyzer.py")
        lines.append(r"\begin{tabular}{lrrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Class & Pixels & Top1(local) & Top1(context) & Top1(bottl) & Entropy$\downarrow$ \\")
        lines.append(r"\midrule")
        for r in items:
            if r["pixels"] == 0:
                continue
            name = self._latex_escape(str(r["name"]))
            t0, t1, t2 = r["top1_ratio"]
            ent = r["entropy_mean"]
            lines.append(f"{name} & {int(r['pixels'])} & {t0:.3f} & {t1:.3f} & {t2:.3f} & {ent:.3f} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        (out_dir / "smoe_per_class_table.tex").write_text("\n".join(lines), encoding="utf-8")

    def _quick_plots(self, out_dir: Path, stats: Dict) -> None:
        # optional: generate a few quick plots for paper draft
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        qdir = out_dir / "quick_plots"
        qdir.mkdir(parents=True, exist_ok=True)

        # 1) mix_mean bars (all/boundary/interior)
        regions = ["all", "boundary", "interior"]
        x = torch.arange(len(regions))
        mix = torch.tensor([stats[r]["mix_mean"] for r in regions], dtype=torch.float32)  # [3,3]

        fig = plt.figure(figsize=(6.2, 3.2), dpi=180)
        ax = fig.add_subplot(111)
        width = 0.22
        for k, name in enumerate(self.expert_names):
            ax.bar(x + (k - 1) * width, mix[:, k].numpy(), width=width, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean routing weight")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(qdir / "mix_mean_regions.png", bbox_inches="tight")
        plt.close(fig)

        # 2) entropy histogram (all)
        ent_counts = torch.tensor(stats["all"]["entropy_hist"]["counts"], dtype=torch.float32)
        ent_edges = torch.tensor(stats["all"]["entropy_hist"]["edges"], dtype=torch.float32)
        ent_centers = (ent_edges[:-1] + ent_edges[1:]) / 2

        fig = plt.figure(figsize=(6.2, 3.2), dpi=180)
        ax = fig.add_subplot(111)
        ax.plot(ent_centers.numpy(), ent_counts.numpy())
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(qdir / "entropy_hist_all.png", bbox_inches="tight")
        plt.close(fig)

        # 3) margin histogram (all)
        mar_counts = torch.tensor(stats["all"]["margin_hist"]["counts"], dtype=torch.float32)
        mar_edges = torch.tensor(stats["all"]["margin_hist"]["edges"], dtype=torch.float32)
        mar_centers = (mar_edges[:-1] + mar_edges[1:]) / 2

        fig = plt.figure(figsize=(6.2, 3.2), dpi=180)
        ax = fig.add_subplot(111)
        ax.plot(mar_centers.numpy(), mar_counts.numpy())
        ax.set_xlabel("Top1-Top2 margin")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(qdir / "margin_hist_all.png", bbox_inches="tight")
        plt.close(fig)

    def save(self, out_dir: str, make_plots: bool = True) -> Dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stats = self.finalize()

        # 1) json
        (out_dir / "smoe_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 2) csv exports
        self._write_csv_global(out_dir, stats)
        # per-class (all/boundary/interior)
        # use the cached stats for these keys
        self._write_csv_per_class(out_dir, "per_class", "smoe_per_class.csv")
        self._write_csv_per_class(out_dir, "per_class_boundary", "smoe_per_class_boundary.csv")
        self._write_csv_per_class(out_dir, "per_class_interior", "smoe_per_class_interior.csv")

        # 3) latex tables
        self._write_latex_tables(out_dir, stats)

        # 4) quick plots (optional)
        if make_plots:
            self._quick_plots(out_dir, stats)

        return stats


def attach_smoe_hooks(p1_module, analyzer: SMoEAnalyzer):
    """
    p1_module: BaseSMoERefinementHead instance
    analyzer: SMoEAnalyzer
    """
    handles = []

    # gate -> logits/mix
    handles.append(p1_module.gate.register_forward_hook(
        lambda m, inp, out: analyzer.cache_logits(out)
    ))

    # experts (optional but recommended)
    handles.append(p1_module.exp_local.register_forward_hook(
        lambda m, inp, out: analyzer.cache("e_local", out)
    ))
    handles.append(p1_module.exp_context.register_forward_hook(
        lambda m, inp, out: analyzer.cache("e_context", out)
    ))
    handles.append(p1_module.exp_bottl.register_forward_hook(
        lambda m, inp, out: analyzer.cache("e_bottl", out)
    ))

    return handles
