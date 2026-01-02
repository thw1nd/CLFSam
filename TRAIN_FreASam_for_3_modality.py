# FreAsam_train_for_2_modality.py —— 直接覆盖使用
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import numpy as np
import argparse
from pathlib import Path
from tools.New_metric import Evaluator
import random

def _cfg_get(cfg, key, default):
    return getattr(cfg, key, default)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=Path, required=True, help='Path to config .py')
    return parser.parse_args()

# =============== 逐步解冻（读取 config 中开关） ===============
def apply_unfreeze_from_config(model, epoch, cfg, logger=None):
    """
    根据 cfg.UNFREEZE_SCHEDULE 在指定 epoch 进行分阶段解冻。
    - 仅解冻 encoder.blocks.* 内的参数；
    - 非 block 层（pos_embed/patch_embed/...）保持冻结；
    - 多次调用是幂等的：已经 requires_grad=True 的参数不会重复计数。
    返回：本轮新解冻的参数列表（用于后续给梯度乘以 LR multiplier）
    """
    schedule = getattr(cfg, "UNFREEZE_SCHEDULE", None)
    if not schedule:
        return []

    # 获取 block 数
    try:
        total_blocks = len(model.encoder.blocks)
    except Exception:
        total_blocks = 0
        for name, _ in model.named_parameters():
            if "encoder.blocks." in name:
                try:
                    idx = int(name.split("encoder.blocks.")[1].split(".")[0])
                    total_blocks = max(total_blocks, idx + 1)
                except Exception:
                    pass

    def indices_for_rule(rule):
        bspec = rule.get("blocks", {"type": "all"})
        if bspec.get("type") == "all":
            return list(range(total_blocks))
        if bspec.get("type") == "last":
            n = int(bspec.get("n", 1))
            n = max(1, min(n, total_blocks))
            return list(range(total_blocks - n, total_blocks))
        if bspec.get("type") == "range":
            start = int(bspec.get("start", 0))
            end = int(bspec.get("end", total_blocks))
            start = max(0, min(start, total_blocks))
            end = max(0, min(end, total_blocks))
            return list(range(start, end))
        return list(range(total_blocks))

    # 名称匹配器
    def is_norm(name: str) -> bool:
        return (".norm" in name) or ("_norm" in name) or (".ln" in name) or (".gn" in name)

    def is_mlp(name: str) -> bool:
        return (".mlp." in name) or (".aux1_mlp." in name)

    def is_attn(name: str) -> bool:
        return (".attn." in name)

    part_pred = {"norm": is_norm, "mlp":  is_mlp, "attn": is_attn}

    total_changed = 0
    applied_rules = []
    newly_unfrozen = []  # <--- 新增：记录本轮新解冻的参数对象

    for rule in schedule:
        if epoch < int(rule.get("epoch", 0)):
            continue  # 时机未到
        parts = rule.get("parts", [])
        idx_list = indices_for_rule(rule)
        want = [part_pred[p] for p in parts if p in part_pred]
        if not want or not idx_list:
            continue

        changed = 0
        for name, p in model.named_parameters():
            if "encoder.blocks." not in name:
                continue
            try:
                k = int(name.split("encoder.blocks.")[1].split(".")[0])
            except Exception:
                continue
            if k not in idx_list:
                continue
            if not any(pred(name) for pred in want):
                continue

            if not p.requires_grad:
                p.requires_grad = True
                changed += 1
                newly_unfrozen.append(p)  # 记录下来

        total_changed += changed
        if changed > 0:
            applied_rules.append((rule.get("epoch"), parts, idx_list, changed))

    if logger is not None and applied_rules:
        for ep, parts, idxs, changed in applied_rules:
            logger(f"[Unfreeze][epoch={epoch}] parts={parts} blocks={idxs} -> params +{changed}")

    return newly_unfrozen

# =============== Lightning 模块 ===============
class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val   = Evaluator(num_class=config.num_classes)

        # ---- 解冻后学习率暖启动（梯度缩放法）所需的注册表 ----
        # 记录 {param_id: {"p": tensor, "start_epoch": int}}
        self._unfreeze_registry = {}
        # 可从 config 读取（有默认）
        self._warmup_epochs = _cfg_get(self.config, 'UNFREEZE_WARMUP_EPOCHS', 4)
        self._init_mult = _cfg_get(self.config, 'UNFREEZE_LR_MULT', 0.3)

        # ---- 记录 epoch 指标到 txt 的相关配置 ----
        # 默认写到当前目录的 epoch_metrics.txt，可以在 config 里加 metrics_txt_path 自定义
        self.metrics_txt_path = _cfg_get(self.config, 'metrics_txt_path', './epoch_metrics.txt')
        # 用来缓存每个 epoch 的 train 指标，方便在 val 结束时一起写入
        self._last_train_stats = {}

    # 兼容老 ckpt（避免 strict 报错）
    def on_load_checkpoint(self, checkpoint) -> None:
        sd = checkpoint.get("state_dict", {})
        drop_keys = [k for k in list(sd.keys())
                     if k == "loss.class_weights" or k.startswith("loss.class_weights.")]
        for k in drop_keys:
            sd.pop(k, None)
        if drop_keys:
            print(f"[compat] dropped unexpected keys from checkpoint: {drop_keys[:5]}{' ...' if len(drop_keys) > 5 else ''}")

    # 解决 SWA deepcopy config 含模块对象的序列化问题
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'config' in state:
            state['config'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def forward(self, x, aux1, aux2):
        return self.net(x, aux1, aux2)

    # 训练一开始创建/清空 txt，并写表头
    def on_fit_start(self):
        # 多卡情况下只在 global_zero 上写文件
        is_main = (self.trainer is None) or getattr(self.trainer, "is_global_zero", True)
        if is_main:
            with open(self.metrics_txt_path, "w", encoding="utf-8") as f:
                f.write("epoch\ttrain_mIoU\ttrain_OA\tval_mIoU\tval_OA\n")
            print(f"[MetricsLog] init file at: {self.metrics_txt_path}")

    # 类别权重线性渐变
    def _update_class_weights_with_ramp(self):
        base = _cfg_get(self.config, 'base_class_weights', None)
        T    = _cfg_get(self.config, 'ramp_epochs_for_class_weights', 0)
        if base is None:
            return
        if T is None or T <= 0:
            alpha = 1.0
        else:
            alpha = min(1.0, float(self.current_epoch + 1) / float(T))
        w0 = torch.ones_like(base)
        w_cur = (1.0 - alpha) * w0 + alpha * base
        if hasattr(self.loss, 'class_weights'):
            if self.loss.class_weights is None:
                self.loss.register_buffer('class_weights', w_cur.to(dtype=torch.float32).clone())
            else:
                self.loss.class_weights.data.copy_(w_cur.to(self.loss.class_weights.device))
        try:
            arr = w_cur.detach().cpu().numpy().round(3).tolist()
        except Exception:
            arr = str(w_cur)
        print(f"[ClassWeights] epoch={self.current_epoch:02d} alpha={alpha:.2f} -> {arr}")

    # 计算给定参数在当前 epoch 的梯度缩放因子（等价 lr multiplier）
    def _warmup_mult_for_param(self, start_epoch: int) -> float:
        if self._warmup_epochs is None or self._warmup_epochs <= 0:
            return 1.0
        # 第一个生效 epoch 使用 init_mult，之后线性升到 1.0
        e0 = start_epoch
        e1 = e0 + self._warmup_epochs
        e  = self.current_epoch
        if e <= e0:
            return float(self._init_mult)
        if e >= e1:
            return 1.0
        ratio = (e - e0) / float(self._warmup_epochs)
        return float(self._init_mult + (1.0 - self._init_mult) * ratio)

    # 注册新解冻参数
    def _register_newly_unfrozen(self, params_list):
        added = 0
        for p in params_list:
            pid = id(p)
            if pid not in self._unfreeze_registry:
                self._unfreeze_registry[pid] = {"p": p, "start_epoch": int(self.current_epoch)}
                added += 1
        if added > 0:
            print(f"[UnfreezeWarmup] epoch={self.current_epoch} newly_unfrozen={added} "
                  f"init_mult={self._init_mult} warmup_epochs={self._warmup_epochs}")

    def on_train_epoch_start(self):
        newly = apply_unfreeze_from_config(self.net, epoch=self.current_epoch, cfg=self.config, logger=print)
        if newly:
            self._register_newly_unfrozen(newly)
        self._update_class_weights_with_ramp()

        # ★ 新增：把当前 epoch 同步给 fusion 模块，用于 freeze_hp_epochs
        if hasattr(self.net, "fusion_module") and hasattr(self.net.fusion_module, "set_epoch"):
            self.net.fusion_module.set_epoch(self.current_epoch)

    # ---- 关键：在每次 optimizer.step 之前，给“刚解冻的参数”乘以梯度缩放系数 ----
    def on_before_optimizer_step(self, optimizer):
        if not self._unfreeze_registry:
            return
        # 对注册过的参数，按各自的 start_epoch 计算 multiplier，并缩放梯度
        scaled, count = 0, 0
        for info in self._unfreeze_registry.values():
            p = info["p"]
            if (p.grad is None) or (not p.requires_grad):
                continue
            mult = self._warmup_mult_for_param(info["start_epoch"])
            # 避免重复 in-place 放大缩小：这里只做乘法（<=1 到 1）
            p.grad.data.mul_(mult)
            scaled += mult
            count  += 1
        if count > 0:
            avg_mult = scaled / float(count)
            # 打印一次即可（Lightning 里该钩子每步都会调，这里做个简短日志）
            if (self.global_step % 100) == 0:
                print(f"[UnfreezeWarmup] step={self.global_step} epoch={self.current_epoch} "
                      f"applied_to={count} avg_mult={avg_mult:.3f}")

    def training_step(self, batch, batch_idx):
        img, aux1, aux2, mask = batch['img'], batch['aux1'], batch['aux2'], batch['gt_semantic_seg']

        # ------- 前向 -------
        prediction = self.net(img, aux1, aux2)

        # ------- 主分割 loss -------
        seg_loss = self.loss(prediction, mask)

        # ------- fusion 正则 loss（来自 MultiScaleFreqSEFusionV2） -------
        if hasattr(self.net, "get_fusion_aux_loss"):
            fusion_aux = self.net.fusion_module.get_aux_loss()  # 标量 tensor，带梯度
            print('use fusion_aux_loss ')
        else:
            # 兼容旧模型：如果没有，就给个 0
            fusion_aux = seg_loss.new_zeros(())

        # 从 config 里读一个系数（没有就默认为 1.0）
        lambda_fuse = _cfg_get(self.config, 'fusion_aux_lambda', 1.0)

        loss = seg_loss + lambda_fuse * fusion_aux

        # ------- 指标计算仍然只用主 logits -------
        pred_main = prediction[0] if (_cfg_get(self.config, 'use_aux_loss', False)
                                      and isinstance(prediction, (list, tuple))) else prediction
        pre_mask = pred_main.softmax(dim=1).argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # 记录 loss
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_seg_loss', seg_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log('train_fusion_aux_loss', fusion_aux, prog_bar=False, on_step=False, on_epoch=True)

        return {"loss": loss}

    def on_train_epoch_end(self):
        iou_per_class = self.metrics_train.Intersection_over_Union()
        f1_per_class = self.metrics_train.F1()

        # 忽略最后一个类别
        iou_per_class = iou_per_class
        f1_per_class = f1_per_class

        mIoU_macro = float(np.nanmean(iou_per_class))
        mF1_macro = float(np.nanmean(f1_per_class))
        OA = float(self.metrics_train.OA())
        mKappa = float(self.metrics_train.mKappa())

        print('train:', {'mIoU_macro': mIoU_macro, 'mF1_macro': mF1_macro, 'OA': OA, 'mKappa': mKappa})
        print({cls: float(iou) for cls, iou in zip(self.config.classes, iou_per_class)})  # 忽略最后一个类别
        self.log_dict({
            'train_mIoU': mIoU_macro, 'train_F1': mF1_macro, 'train_OA': OA, 'train_mKappa': mKappa,
            'train_mIoU_macro': mIoU_macro, 'train_mF1_macro': mF1_macro
        }, prog_bar=True)

        # 记录训练指标
        self._last_train_stats = {
            "mIoU_macro": mIoU_macro,
            "OA": OA
        }
        self.metrics_train.reset()

    def on_validation_epoch_end(self):
        iou_per_class = self.metrics_val.Intersection_over_Union()
        f1_per_class = self.metrics_val.F1()

        # 忽略最后一个类别
        iou_per_class = iou_per_class
        f1_per_class = f1_per_class

        mIoU_macro = float(np.nanmean(iou_per_class))
        mF1_macro = float(np.nanmean(f1_per_class))
        OA = float(self.metrics_val.OA())
        mKappa = float(self.metrics_val.mKappa())

        print('val:', {'mIoU_macro': mIoU_macro, 'mF1_macro': mF1_macro, 'OA': OA, 'mKappa': mKappa})
        print({cls: float(iou) for cls, iou in zip(self.config.classes, iou_per_class)})  # 忽略最后一个类别
        self.log_dict({
            'val_mIoU': mIoU_macro, 'val_F1': mF1_macro, 'val_OA': OA, 'val_mKappa': mKappa,
            'val_mIoU_macro': mIoU_macro, 'val_mF1_macro': mF1_macro
        }, prog_bar=True)

        self.metrics_val.reset()

    def validation_step(self, batch, batch_idx):
        img, aux1, aux2, mask = batch['img'], batch['aux1'], batch['aux2'], batch['gt_semantic_seg']
        prediction = self.forward(img, aux1, aux2)
        pred_main = prediction[0] if (_cfg_get(self.config, 'use_aux_loss', False) and isinstance(prediction, (list, tuple))) else prediction
        pre_mask = pred_main.softmax(dim=1).argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
        loss_val = self.loss(prediction, mask)
        self.log('val_loss', loss_val, prog_bar=False, on_step=False, on_epoch=True)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        iou_per_class = self.metrics_val.Intersection_over_Union()
        f1_per_class = self.metrics_val.F1()

        # 忽略最后一个类别
        iou_per_class = iou_per_class
        f1_per_class = f1_per_class

        mIoU_macro = float(np.nanmean(iou_per_class))
        mF1_macro = float(np.nanmean(f1_per_class))
        OA = float(self.metrics_val.OA())
        mKappa = float(self.metrics_val.mKappa())

        print('val:', {'mIoU_macro': mIoU_macro, 'mF1_macro': mF1_macro, 'OA': OA, 'mKappa': mKappa})
        print({cls: float(iou) for cls, iou in zip(self.config.classes, iou_per_class)})  # 忽略最后一个类别
        self.log_dict({
            'val_mIoU': mIoU_macro, 'val_F1': mF1_macro, 'val_OA': OA, 'val_mKappa': mKappa,
            'val_mIoU_macro': mIoU_macro, 'val_mF1_macro': mF1_macro
        }, prog_bar=True)

        self.metrics_val.reset()

    def configure_optimizers(self):
        return [self.config.optimizer], [self.config.lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader

def main():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=_cfg_get(config, 'save_top_k', 3),
        monitor=_cfg_get(config, 'monitor', 'val_mIoU'),
        save_last=_cfg_get(config, 'save_last', True),
        mode=_cfg_get(config, 'monitor_mode', 'max'),
        dirpath=_cfg_get(config, 'weights_path', './weights'),
        filename=_cfg_get(config, 'weights_name', 'epoch{epoch:03d}-val{val_mIoU:.3f}')
    )
    logger = CSVLogger('../lightning_logs', name=_cfg_get(config, 'log_name', 'freASam'))

    model = Supervision_Train(config)

    patience = _cfg_get(config, 'earlystop_patience', 20)
    early_stop = EarlyStopping(monitor=_cfg_get(config, 'monitor', 'val_mIoU'),
                               mode=_cfg_get(config, 'monitor_mode', 'max'),
                               patience=patience)
    swa_start_ratio = _cfg_get(config, 'swa_start_ratio', 0.6)
    swa = StochasticWeightAveraging(
        swa_lrs=_cfg_get(config, 'swa_lrs', 1.0e-5),
        swa_epoch_start=int(swa_start_ratio * _cfg_get(config, 'max_epoch', 110))
    )

    if _cfg_get(config, 'pretrained_ckpt_path', None):
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    grad_clip_val  = _cfg_get(config, 'gradient_clip_val', 1.0)
    grad_clip_algo = _cfg_get(config, 'gradient_clip_algorithm', 'norm')

    trainer = pl.Trainer(
        devices=_cfg_get(config, 'gpus', 1),
        max_epochs=_cfg_get(config, 'max_epoch', 110),
        accelerator='auto',
        precision=_cfg_get(config, 'precision', '32-true'),
        gradient_clip_val=grad_clip_val,
        gradient_clip_algorithm=grad_clip_algo,
        accumulate_grad_batches=_cfg_get(config, 'accumulate_n', 1),
        check_val_every_n_epoch=_cfg_get(config, 'check_val_every_n_epoch', 1),
        callbacks=[checkpoint_callback, early_stop, swa],
        strategy=_cfg_get(config, 'strategy', 'auto'),
        logger=logger
    )
    trainer.fit(model=model, ckpt_path=_cfg_get(config, 'resume_ckpt_path', None))

if __name__ == '__main__':
    main()
