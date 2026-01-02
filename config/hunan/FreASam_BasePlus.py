from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.Hunan_dataset import *
from geoseg.models.FreASam.FreASam_BasePlus import FreASam
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


# =============== 逐步解冻的开关 ===============
INFO_BLOCK_IDX = -1          # -1 = 最后一个 block（用于调试打印）
UNFREEZE_EPOCH = 15          # 第 15 轮开始微解冻
UNFREEZE_LAST_N_BLOCKS = 4   # 解冻最后 n 个 blocks
UNFREEZE_MATCH_ALL = True    # True=解冻 block 的所有参数；False=仅解冻 attn/qkv/proj/LoRA/MLP

UNFREEZE_SCHEDULE = [
    # 先只动最后2个block的归一化；给模型“热身”
    {"epoch": 0, "blocks": {"type": "all"}, "parts": ["norm"]},
    # {"epoch": 2, "blocks": {"type": "last", "n": 8}, "parts": ["mlp"]},
    {"epoch": 1, "blocks": {"type": "all"}, "parts": ["mlp","proj"]},


]

# =============== 辅助：BN -> GN（小 batch 更稳） ===============
def _pick_group_count(C: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if C % g == 0:
            return g
    return 1

def bn_to_gn(module: nn.Module, eps: float = 1e-5, affine: bool = True):
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            C = child.num_features
            groups = _pick_group_count(C)
            setattr(module, name, nn.GroupNorm(groups, C, eps=eps, affine=affine))
        else:
            bn_to_gn(child, eps=eps, affine=affine)

# =============== 分组与正则判断 ===============
def _is_norm_like(name: str) -> bool:
    n = name.lower()
    if n.endswith(".bias"):
        return True
    if ("norm" in n) or ("bn" in n) or ("ln" in n) or ("gn" in n):
        return True
    return False

def _is_mlp_or_lora(name: str) -> bool:
    n = name.lower()
    if "mlp" in n:
        return True
    if "lora_a" in n or "lora_b" in n or "lora_down" in n or "lora_up" in n:
        return True
    return False

# =============== 兼容多种 ckpt 的取 state_dict ===============
def _get_sam2_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if 'model' in ckpt_obj and isinstance(ckpt_obj['model'], dict):
            return ckpt_obj['model']
        if 'state_dict' in ckpt_obj and isinstance(ckpt_obj['state_dict'], dict):
            return ckpt_obj['state_dict']
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise RuntimeError("SAM2 ckpt 不含可用的 state_dict（既无 'model' 也无 'state_dict'）。")

def _strip_trunk_prefix(name: str, trunk_prefix="image_encoder.trunk."):
    return name[len(trunk_prefix):] if name.startswith(trunk_prefix) else name

def _shape_compatible(t_src: torch.Tensor, t_dst: torch.Tensor) -> bool:
    return tuple(t_src.shape) == tuple(t_dst.shape)

# =============== 权重加载 + 冻结 ===============
def load_sam2_weights_to_multimodal_model(
    model,
    sam2_ckpt_path: str,
    freeze_loaded: bool = True,
    verbose: bool = True,
    trunk_prefix: str = "image_encoder.trunk.",
    encoder_prefix: str = "encoder.",
    freeze_patterns_exclude = ("lora",),
):
    assert os.path.isfile(sam2_ckpt_path), f"找不到权重：{sam2_ckpt_path}"
    ckpt = torch.load(sam2_ckpt_path, map_location="cpu")
    print(f"==> 读取 SAM2 权重: {sam2_ckpt_path} | 顶层键: {list(ckpt.keys())}")
    sam2_sd_raw = _get_sam2_state_dict(ckpt)

    sam2_trunk = OrderedDict()
    for k, v in sam2_sd_raw.items():
        if k.startswith(trunk_prefix):
            sam2_trunk[_strip_trunk_prefix(k, trunk_prefix)] = v
        elif k.startswith(("blocks.", "patch_embed.", "pos_embed", "pos_embed_window", "norm", "x_norm")):
            sam2_trunk[k] = v

    model_sd = model.state_dict()
    loaded, skipped, shape_mismatch = [], [], []

    def _copy_if_ok(src_key, dst_key, src_t):
        if dst_key in model_sd:
            if _shape_compatible(src_t, model_sd[dst_key]):
                model_sd[dst_key].copy_(src_t)
                loaded.append(dst_key)
            else:
                shape_mismatch.append((src_key, dst_key, tuple(src_t.shape), tuple(model_sd[dst_key].shape)))
        else:
            pass

    for src_key, src_tensor in sam2_trunk.items():
        subkey = src_key

        # 1) 主干直接尝试
        dst_x = f"{encoder_prefix}{subkey}"
        _copy_if_ok(src_key, dst_x, src_tensor)

        # 2) patch_embed → aux1_patch_embed（若存在）
        if subkey.startswith("patch_embed."):
            dst_aux1 = f"{encoder_prefix}aux1_{subkey}"
            _copy_if_ok(src_key, dst_aux1, src_tensor)
            continue

        # 3) blocks.* 内的细分映射
        if subkey.startswith("blocks."):

            # 3.1 norm1 / norm2 → aux1_norm1 / aux1_norm2
            if (".norm1" in subkey) or (".x_norm1" in subkey):
                aux1_subkey = subkey.replace(".norm1", ".aux1_norm1").replace(".x_norm1", ".aux1_norm1")
                _copy_if_ok(src_key, f"{encoder_prefix}{aux1_subkey}", src_tensor)

                # 同理加入 aux2 的冻结
                aux2_subkey = subkey.replace(".norm1", ".aux2_norm1").replace(".x_norm1", ".aux2_norm1")
                _copy_if_ok(src_key, f"{encoder_prefix}{aux2_subkey}", src_tensor)
                continue

            if (".norm2" in subkey) or (".x_norm2" in subkey):
                aux1_subkey = subkey.replace(".norm2", ".aux1_norm2").replace(".x_norm2", ".aux1_norm2")
                _copy_if_ok(src_key, f"{encoder_prefix}{aux1_subkey}", src_tensor)

                # 同理加入 aux2 的冻结
                aux2_subkey = subkey.replace(".norm2", ".aux2_norm2").replace(".x_norm2", ".aux2_norm2")
                _copy_if_ok(src_key, f"{encoder_prefix}{aux2_subkey}", src_tensor)
                continue

            # 3.2 attn.proj.* → 同时写到 aux1_proj.* 和 aux2_proj.*
            if ".attn.proj." in subkey:
                _copy_if_ok(src_key, f"{encoder_prefix}{subkey}", src_tensor)  # x
                _copy_if_ok(src_key, f"{encoder_prefix}{subkey.replace('.attn.proj.', '.attn.aux1_proj.')}", src_tensor)  # aux1
                _copy_if_ok(src_key, f"{encoder_prefix}{subkey.replace('.attn.proj.', '.attn.aux2_proj.')}", src_tensor)  # aux2
                continue

            # 3.3 mlp.layers.* → 同时写到 aux1_mlp.layers.* 和 aux2_mlp.layers.*
            if ".mlp.layers." in subkey:
                _copy_if_ok(src_key, f"{encoder_prefix}{subkey}", src_tensor)  # x
                _copy_if_ok(src_key, f"{encoder_prefix}{subkey.replace('.mlp.', '.aux1_mlp.')}", src_tensor)  # aux1
                _copy_if_ok(src_key, f"{encoder_prefix}{subkey.replace('.mlp.', '.aux2_mlp.')}", src_tensor)  # aux2
                continue

            # 3.4 qkv / 其他：处理 aux2 分支的权重
            if ".attn.aux2_proj." in subkey:  # 修改为处理 aux2 的权重
                _copy_if_ok(src_key, f"{encoder_prefix}{subkey.replace('.attn.aux2_proj.', '.attn.aux2_proj.')}", src_tensor)  # aux2
                continue

            if ".attn.aux2_qkv." in subkey:  # 新增处理 aux2_qkv 的权重
                _copy_if_ok(src_key, f"{encoder_prefix}{subkey.replace('.attn.aux2_qkv.', '.attn.aux2_qkv.')}", src_tensor)
                continue

        # 4) 顶层 norm / pos_embed / pos_embed_window，如模型存在 aux1_* 才复制
        if subkey.startswith(("norm", "x_norm", "pos_embed", "pos_embed_window")):
            _copy_if_ok(src_key, f"{encoder_prefix}aux2_{subkey}", src_tensor)

    # 加载到模型中
    model.load_state_dict(model_sd, strict=False)

    if freeze_loaded:
        frozen, total = 0, 0
        for n, p in model.named_parameters():
            total += 1
            if any(x in n for x in freeze_patterns_exclude):
                continue
            if n in loaded:
                p.requires_grad_(False)
                frozen += 1
        if verbose:
            print(f"[SAM2->Encoder] 已加载(并冻结) {len(loaded)} 层 / 冻结 {frozen}/{total}")

    return {"loaded": loaded, "shape_mismatch": shape_mismatch, "skipped": skipped}

# ----------------------------- 训练超参 -----------------------------
max_epoch = 120
warmup_epochs = 12
eta_min = 1e-5
ignore_index = 255
background_index = None
train_batch_size = 14
val_batch_size = 14

# 分组学习率
lr = 2e-3             # 头部
backbone_lr = 1e-4      # encoder 主干
lr_enc_mlp = 2e-4     # encoder 的 MLP/LoRA

# 正则
weight_decay = 1e-2
backbone_weight_decay = 1e-2

# 类别权重平滑
# base_class_weights = torch.tensor([1, 1, 2, 2, 1, 2, 1], dtype=torch.float32)
# ramp_epochs_for_class_weights = 15

accumulate_n = 2
num_classes = 7
classes = CLASSES

test_time_aug = 'lr'
output_mask_dir, output_mask_rgb_dir = None, None
weights_name = "FreASam-Base-New12.7"
weights_path = f"model_weights/hunan/{weights_name}"
metrics_txt_path = f"model_weights/hunan/text/{weights_name}"
test_weights_name = "FreASam-Base-New12.7"
log_name = f'potsdam/{weights_name}'
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = 'auto'
pretrained_ckpt_path = None
resume_ckpt_path = None

# ----------------------------- 构建模型 + 读 SAM2 + 冻结 -----------------------------
net = FreASam(n_class=7)
sam2_weights_path = r"D:\\ESUWork\\sam2.1_hiera_base_plus.pt"
summary = load_sam2_weights_to_multimodal_model(
    model=net,
    sam2_ckpt_path=sam2_weights_path,
    freeze_loaded=True,   # 加载后先冻结（LoRA 不冻结）
    verbose=True
)

# BN->GN
# bn_to_gn(net)

# 打印冻结层
frozen_params = [n for n, p in net.named_parameters() if not p.requires_grad]
print("冻结的权重层：")
for name in frozen_params:
    print(name)

# ----------------------------- Loss -----------------------------
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# ----------------------------- 数据 -----------------------------
train_set = HunanDataset(
data_root=r'H:\Datasets\Hunan_Dataset\Hunan_Dataset\train',
rgb_dir='s1', aux1_dir='s2', aux2_dir='topo', mask_dir='lc',
suffix='.tif',
band_indices_s2=None,           # None = 使用 S2 全部 13 个波段
img_size=(256, 256),
train=True,
transform=None,                 # 或 transform=build_geo_transform((512,512), is_train=True)
prefix_map={'s1':'s1_', 's2':'s2_', 'topo':'topo_', 'lc':'lc_'},
)

val_set = HunanDataset(
data_root=r'H:\Datasets\Hunan_Dataset\Hunan_Dataset\test',
rgb_dir='s1', aux1_dir='s2', aux2_dir='topo', mask_dir='lc',
suffix='.tif',
band_indices_s2=None,           # None = 使用 S2 全部 13 个波段
img_size=(256, 256),
train=False,
transform=None,                 # 或 transform=build_geo_transform((512,512), is_train=True)
prefix_map={'s2':'s2_', 's1':'s1_', 'topo':'topo_', 'lc':'lc_'},
)
train_loader = DataLoader(train_set, batch_size=train_batch_size,
                          num_workers=4, pin_memory=True, shuffle=True,
                          drop_last=True, persistent_workers=True)
val_loader = DataLoader(val_set, batch_size=val_batch_size,
                        num_workers=4, shuffle=False, pin_memory=True,
                        drop_last=False, persistent_workers=True)

# ----------------------------- 优化器分组 -----------------------------
loaded_set = set(summary.get("loaded", []))

def _is_sam2_loaded_encoder_param(name_in_encoder: str) -> bool:
    """
    name_in_encoder 形如 'blocks.0.attn.qkv.weight'
    summary["loaded"] 里存的是 'encoder.blocks.0.attn.qkv.weight'
    """
    full_name = f"encoder.{name_in_encoder}"
    return full_name in loaded_set

enc_backbone_decay, enc_backbone_nodecay = [], []   # SAM2 主干上被解冻的那些参数（小 LR）
enc_mlp_decay, enc_mlp_nodecay = [], []             # encoder 里的 MLP / LoRA（中 LR）
enc_new_decay, enc_new_nodecay = [], []             # encoder 里“新加的模块”（非 SAM2 & 非 LoRA/MLP），用大 LR
head_decay, head_nodecay = [], []                   # encoder 之外的所有模块（decoder, seg-head, fusion 等）

# 先处理 encoder 内部
for n, p in net.encoder.named_parameters():
    # 冻结的参数先不管 requires_grad，仍然放进 group 里，后面解冻时可以直接用
    is_mlp_lora = _is_mlp_or_lora(n)
    no_decay = _is_norm_like(n) or is_mlp_lora
    is_loaded_backbone = _is_sam2_loaded_encoder_param(n)

    if is_mlp_lora:
        # 所有 LoRA / MLP 统一放在 enc_mlp_* 组，用 lr_enc_mlp
        (enc_mlp_nodecay if no_decay else enc_mlp_decay).append(p)
    else:
        # 非 LoRA / MLP：再区分「来自 SAM2」还是「你新写的模块」
        if is_loaded_backbone:
            # 这是 SAM2 主干上被解冻的那部分（例如按 schedule 解冻的 norm / 最后 2 个 block 的 qkv/proj）
            (enc_backbone_nodecay if no_decay else enc_backbone_decay).append(p)
        else:
            # 这是你在 encoder 里新写的结构（例如 CrossModalFuseModule 的卷积、cross_gamma 等）
            # 训练策略上更像“头部”，用较大的 LR 更合适
            (enc_new_nodecay if no_decay else enc_new_decay).append(p)

# 再处理 encoder 之外的所有参数：统一当作 head（decoder / seg-head / fusion 等）
for n, p in net.named_parameters():
    if n.startswith("encoder."):
        continue
    (head_nodecay if _is_norm_like(n) else head_decay).append(p)

optimizer_groups = [
    # 1) SAM2 主干上被解冻的那一小撮参数（例如你按 UNFREEZE_SCHEDULE 放开的那几层）
    {"params": enc_backbone_decay,    "lr": backbone_lr, "weight_decay": backbone_weight_decay, "name": "enc_backbone_decay"},
    {"params": enc_backbone_nodecay,  "lr": backbone_lr, "weight_decay": 0.0,                   "name": "enc_backbone_nodecay"},

    # 2) encoder 里的 MLP / LoRA（包括 qkv 里的 LoRA、CrossModalFuse 里带 lora_xxx 的线性层）
    {"params": enc_mlp_decay,         "lr": lr_enc_mlp,  "weight_decay": backbone_weight_decay, "name": "enc_mlp_decay"},
    {"params": enc_mlp_nodecay,       "lr": lr_enc_mlp,  "weight_decay": 0.0,                   "name": "enc_mlp_nodecay"},

    # 3) 你在 encoder 里新写的模块（非 SAM2 & 非 LoRA/MLP），用和 head 一样的 LR
    {"params": enc_new_decay,         "lr": lr,          "weight_decay": weight_decay,          "name": "enc_new_decay"},
    {"params": enc_new_nodecay,       "lr": lr,          "weight_decay": 0.0,                   "name": "enc_new_nodecay"},

    # 4) encoder 之外的所有层（decoder / seg-head / fusion 等），传统意义上的“头部”
    {"params": head_decay,            "lr": lr,          "weight_decay": weight_decay,          "name": "head_decay"},
    {"params": head_nodecay,          "lr": lr,          "weight_decay": 0.0,                   "name": "head_nodecay"},
]

# 去掉空的 param group，避免警告
optimizer_groups = [g for g in optimizer_groups if len(g["params"]) > 0]

print("📊 Optimizer groups:")
for g in optimizer_groups:
    print(f"  - {g['name']}: {len(g['params'])} params, lr={g['lr']}, wd={g['weight_decay']}")

optimizer = torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.99), eps=1e-5, amsgrad=False)

# ----------------------------- LR Schedulers -----------------------------
def _warmup_lambda(cur_epoch: int):
    return (cur_epoch + 1) / max(1, warmup_epochs)

warmup_scheduler  = LambdaLR(optimizer, lr_lambda=_warmup_lambda)
cosine_scheduler  = CosineAnnealingLR(optimizer, T_max=max(1, max_epoch - warmup_epochs), eta_min=eta_min)
lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])