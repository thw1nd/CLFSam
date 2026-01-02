import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss


class FocalLoss(nn.Module):
    """
    多分类语义分割版 Focal Loss
    - logits: [B, C, H, W]
    - labels: [B, H, W]，其中 ignore_index 的像素会被忽略
    - alpha: 可为 float（统一系数）或 shape=[C] 的 per-class 向量
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, reduction="mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.reduction = reduction

        # alpha 可以是标量或 per-class 向量
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32)
                             if not isinstance(alpha, torch.Tensor) else alpha.float())

    def forward(self, logits, labels):
        # logits: [B,C,H,W], labels: [B,H,W]
        B, C, H, W = logits.shape
        assert labels.shape == (B, H, W), f"labels应为[B,H,W]，当前是{labels.shape}"

        # 有效像素
        valid = (labels != self.ignore_index)
        if valid.sum() == 0:
            # 没有有效像素就返回0
            return logits.new_zeros([])

        # 先把无效标签夹到0～C-1，避免 gather 报错，随后用 valid 掩蔽
        labels_clamped = labels.clamp(min=0, max=C-1)

        log_p = F.log_softmax(logits, dim=1)          # [B,C,H,W]
        logpt = log_p.gather(1, labels_clamped.unsqueeze(1)).squeeze(1)  # [B,H,W]
        pt = logpt.exp()                               # [B,H,W]

        # 计算 alpha_t：支持标量或向量
        if self.alpha.ndim == 0:
            alpha_t = torch.full_like(pt, float(self.alpha))
        else:
            # per-class alpha
            alpha_t = self.alpha.to(logits.device)[labels_clamped]

        # Focal
        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * logpt

        # 只在有效像素上求平均/求和
        loss = loss * valid.float()
        if self.reduction == "mean":
            return loss.sum() / (valid.sum() + 1e-6)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class EdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, edge_factor=1.0):
        super(EdgeLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.edge_factor = edge_factor

    def get_boundary(self, x):
        laplacian_kernel_target = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).cuda(device=x.device)
        x = x.unsqueeze(1).float()
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0

        return x

    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        # print(boundary_targets.shape)
        logits = F.softmax(logits, dim=1).argmax(dim=1).squeeze(dim=1)
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        # print(boundary_pre)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        # print(boundary_pre)
        # dice_loss = 1 - ((2. * (boundary_pre * boundary_targets).sum(1) + 1.0) /
        #                  (boundary_pre.sum(1) + boundary_targets.sum(1) + 1.0))
        # dice_loss = dice_loss.mean()
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)

        return edge_loss

    def forward(self, logits, targets):
        loss = (self.main_loss(logits, targets) + self.compute_edge_loss(logits, targets) * self.edge_factor) / (self.edge_factor+1)
        return loss


class OHEM_CELoss(nn.Module):
    def __init__(self, thresh=0.7, ignore_index=255):
        super(OHEM_CELoss, self).__init__()
        # 只存一个纯数，别在这里 .cuda()
        self.thresh_val = float(thresh)
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction='none'
        )

    def forward(self, logits, labels):
        # 把阈值放到当前logits所在的device上
        thresh = -torch.log(
            torch.tensor(self.thresh_val, device=logits.device, dtype=torch.float32)
        )
        # 有效像素里至少保留 1/16
        valid_mask = (labels != self.ignore_index)
        n_valid = valid_mask.sum()
        # 防止极端情况除0
        n_min = max(n_valid.item() // 16, 1)

        loss = self.criteria(logits, labels).view(-1)  # [N]
        loss_hard = loss[loss > thresh]

        if loss_hard.numel() < n_min:
            # 不够就再从所有像素里topk
            loss_hard, _ = loss.topk(n_min)

        return loss_hard.mean()


class UnetFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 0.7, 0.3)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux, labels)
        else:
            loss = self.main_loss(logits, labels)

        return loss


class FreASamLoss(nn.Module):
    """
    FreASam 的 loss，基于你现在的版本做了三点加强：
      1) 真正把 class_weights 用到 CE 上
      2) 训练阶段给主输出加一层 OHEM，专门抓小目标/难像素
      3) 边缘分支保持不变
    """
    def __init__(self,
                 ignore_index=255,
                 aux_weight=0.25,
                 edge_factor=0.5,
                 band_width=1,
                 use_bce=True,
                 class_weights = None,
                 ohem_on=True,           # NEW: 默认开OHEM，也可以关
                 ohem_thresh=0.5):       # NEW: OHEM阈值
        super().__init__()
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight
        self.edge_factor = edge_factor
        self.band_width = band_width
        self.use_bce = use_bce
        self.ohem_on = ohem_on
        self.ohem_thresh = ohem_thresh

        # --- class weights 注册到buffer里，跟模型一起to(device) ---
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights",
                             class_weights if class_weights is not None else None)

        # --- 构建 main/aux 的 CE，这里要兼容你的 SoftCrossEntropyLoss ---
        ce_kwargs = dict(smooth_factor=0.05, ignore_index=ignore_index)
        self._manual_class_weight = False  # 如果 soft_ce 不支持 weight，就手动做一遍
        if self.class_weights is not None:
            ce_kwargs["weight"] = self.class_weights

        try:
            ce_main = SoftCrossEntropyLoss(**ce_kwargs)
            ce_aux = SoftCrossEntropyLoss(**ce_kwargs)
        except TypeError:
            # 说明你的 SoftCrossEntropyLoss 没有 weight 这个参数，就退回去
            ce_main = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
            ce_aux = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)
            self._manual_class_weight = self.class_weights is not None

        # --- 主 loss: CE + Dice，权重还是(1,1)，不改你的端口 ---
        self.main_loss = JointLoss(
            ce_main,
            DiceLoss(smooth=0.05, ignore_index=ignore_index),
            1, 1
        )
        # --- 辅助头还是单CE ---
        self.aux_loss = ce_aux

        # --- 边缘监督所需的卷积核，沿用你原来的写法 ---
        k_sx  = torch.tensor([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=torch.float32)[None, None]
        k_sy  = torch.tensor([[-1,-2,-1],
                              [ 0, 0, 0],
                              [ 1, 2, 1]], dtype=torch.float32)[None, None]
        k_lap = torch.tensor([[-1,-1,-1],
                              [-1, 8,-1],
                              [-1,-1,-1]], dtype=torch.float32)[None, None]
        self.register_buffer("sobel_x", k_sx)
        self.register_buffer("sobel_y", k_sy)
        self.register_buffer("lap_kernel", k_lap)

        # --- OHEM 分支 ---
        if self.ohem_on:
            self.ohem_ce = OHEM_CELoss(thresh=ohem_thresh, ignore_index=ignore_index)

    # ---------- 下面是你原来就有的工具函数，保持不变 ----------
    @staticmethod
    def _extract_from_container(x):
        if isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        if isinstance(x, dict):
            for k in ("gt_semantic_seg", "mask", "label", "labels", "gt", "y"):
                if k in x:
                    x = x[k]; break
        return x

    def _normalize_labels(self, labels, device):
        labels = self._extract_from_container(labels)
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.ndim == 0:
            labels = labels.view(1, 1, 1)
        if labels.ndim == 4 and labels.size(1) == 1:
            labels = labels[:, 0]
        elif labels.ndim == 2:
            labels = labels.unsqueeze(0)
        labels = labels.to(device=device, dtype=torch.long)
        assert labels.ndim == 3, f"labels must be [B,H,W], got {labels.shape}"
        return labels

    @torch.no_grad()
    def _get_boundary_target(self, targets):
        targets = self._normalize_labels(targets, device=self.lap_kernel.device)
        B, H, W = targets.shape
        valid = (targets != self.ignore_index).float().unsqueeze(1)
        t = targets.clone()
        t[t == self.ignore_index] = 0
        t = t.float().unsqueeze(1)
        e = F.conv2d(t, self.lap_kernel, padding=1).abs()
        edge_gt = (e > 0).float()
        if self.band_width > 0:
            k = 2 * self.band_width + 1
            edge_gt = F.max_pool2d(edge_gt, kernel_size=k, stride=1, padding=self.band_width)
        edge_gt = edge_gt * valid
        return edge_gt, valid

    def _edge_from_probs(self, p):
        B, C, H, W = p.shape
        gx = F.conv2d(p, self.sobel_x.expand(C,1,3,3), padding=1, groups=C)
        gy = F.conv2d(p, self.sobel_y.expand(C,1,3,3), padding=1, groups=C)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        edge_pred = grad_mag.mean(dim=1, keepdim=True)
        edge_pred = torch.tanh(2.0 * edge_pred)
        return edge_pred

    def _edge_loss(self, logits, labels):
        edge_gt, valid = self._get_boundary_target(labels)
        if valid.sum() < 1:
            return logits.new_zeros([])
        p = F.softmax(logits, dim=1)
        edge_logit = self._edge_from_probs(p)
        loss_map = F.binary_cross_entropy_with_logits(edge_logit, edge_gt, reduction='none')
        loss = (loss_map * valid).sum() / (valid.sum() + 1e-6)
        return loss

    def _main_with_edge(self, logits, labels):
        main = self.main_loss(logits, labels)
        edge = self._edge_loss(logits, labels)
        return (main + self.edge_factor * edge) / (1.0 + self.edge_factor)

    # NEW: 如果 SoftCE 不支持 weight，就手动做一遍 weighted CE
    def _manual_weighted_ce(self, logits, labels):
        return F.cross_entropy(
            logits, labels,
            weight=self.class_weights,
            ignore_index=self.ignore_index
        )

    def forward(self, logits, labels):
        device = logits[0].device if isinstance(logits, (list, tuple)) else logits.device
        labels = self._normalize_labels(labels, device=device)

        # --- 有主有辅 ---
        if self.training and isinstance(logits, (list, tuple)) and len(logits) == 2:
            logit_main, logit_aux = logits

            main = self._main_with_edge(logit_main, labels)

            # NEW: 只对主输出做OHEM，让小目标被抓上来
            if self.ohem_on:
                hard_ce = self.ohem_ce(logit_main, labels)
                main = 0.85 * main + 0.15 * hard_ce # if loss have stronger change, 0.2 can change to 0.15

            # NEW: 如果SoftCE没吃到weight，这里手动补一遍
            if self._manual_class_weight:
                cw_ce = self._manual_weighted_ce(logit_main, labels)
                main = 0.7 * main + 0.3 * cw_ce

            loss = main + self.aux_weight * self.aux_loss(logit_aux, labels)

        else:
            # --- 只有主输出 ---
            main = self._main_with_edge(logits, labels)

            if self.ohem_on and self.training:
                hard_ce = self.ohem_ce(logits, labels)
                main = 0.85 * main + 0.15 * hard_ce

            if self._manual_class_weight:
                cw_ce = self._manual_weighted_ce(logits, labels)
                main = 0.7 * main + 0.3 * cw_ce

            loss = main

        return loss


class FreASamLossv2(nn.Module):
    """
    简化后的 FreASamLoss，加入了 Focal Loss 和 Dice Loss 的组合，权重根据任务平衡小类和大类的精度。
    """
    def __init__(self,
                 ignore_index=255,
                 aux_weight=0.25,
                 class_weights=None,
                 focal_alpha=0.25,       # Focal Loss的alpha值
                 focal_gamma=2.0):       # Focal Loss的gamma值
        super().__init__()
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight

        # --- class weights 注册到buffer里，跟模型一起to(device) ---
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

        # --- 构建 FocalLoss ---
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index)

        # --- 构建主损失：Focal Loss + Dice Loss ---
        ce_kwargs = dict(smooth_factor=0.05, ignore_index=ignore_index)
        self.main_loss = JointLoss(
            self.focal_loss,  # 使用 Focal Loss 替代传统的 CE
            DiceLoss(smooth=0.05, ignore_index=ignore_index),
            0.7,  # Focal Loss 的权重
            0.3   # Dice Loss 的权重
        )
        # --- 辅助损失仍然是 CE ---
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    @staticmethod
    def _extract_from_container(x):
        if isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        if isinstance(x, dict):
            for k in ("gt_semantic_seg", "mask", "label", "labels", "gt", "y"):
                if k in x:
                    x = x[k];
                    break
        return x

    def _normalize_labels(self, labels, device):
        labels = self._extract_from_container(labels)
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.ndim == 0:
            labels = labels.view(1, 1, 1)
        if labels.ndim == 4 and labels.size(1) == 1:
            labels = labels[:, 0]
        elif labels.ndim == 2:
            labels = labels.unsqueeze(0)
        labels = labels.to(device=device, dtype=torch.long)
        assert labels.ndim == 3, f"labels must be [B,H,W], got {labels.shape}"
        return labels

    def forward(self, logits, labels):
        device = logits[0].device if isinstance(logits, (list, tuple)) else logits.device
        labels = self._normalize_labels(labels, device=device)

        # --- 有主有辅 ---
        if self.training and isinstance(logits, (list, tuple)) and len(logits) == 2:
            logit_main, logit_aux = logits

            main = self.main_loss(logit_main, labels)

            loss = main + self.aux_weight * self.aux_loss(logit_aux, labels)

        else:
            # 只有主输出
            main = self.main_loss(logits, labels)

            loss = main

        return loss

if __name__ == '__main__':
    targets = torch.randint(low=0, high=2, size=(2, 16, 16))
    logits = torch.randn((2, 2, 16, 16))
    # print(targets)
    model = EdgeLoss()
    loss = model.compute_edge_loss(logits, targets)

    print(loss)