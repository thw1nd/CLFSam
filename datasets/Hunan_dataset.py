# -*- coding: utf-8 -*-
"""
Hunan_dataset.py  —— 综合改进版（多通道 + 模态自适应归一化 + 训练增强）

主要特性：
1) 使用 rasterio 读取 GeoTIFF，稳健支持 S1/S2/TOPO 等多通道栅格；
2) 不再强制把各模态改成 3 通道，完整保留原始通道数（例如 S2=13, S1=2, TOPO=2）；
3) 在 Dataset 内部对不同模态做“几何对齐 + 强度归一化”，训练阶段同时启用适度的数据增强；
4) transform 仍兼容两种风格：
   - Albumentations.Compose（只建议放几何变换）；
   - 普通函数 transform(img, aux1, aux2, mask) -> (img, aux1, aux2, mask)。

返回字典：
{
    'img':  Tensor[C_s2,H,W],   # S2 (原始 13 通道或你设置的 band_indices)
    'aux1': Tensor[C_s1,H,W],   # S1 (原始 2 通道)
    'aux2': Tensor[C_topo,H,W], # TOPO/DEM/TRI (原始 2 通道)
    'gt_semantic_seg': LongTensor[H,W],
    'img_id': str
}
"""

CLASSES = ('cropland', 'forest', 'grassland', 'wetland', 'water', 'unused land', 'built-up area')

PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
           [255, 255, 0], [255, 0, 0], [255, 0, 255]]

import os
import os.path as osp
from typing import List, Optional, Tuple, Sequence, Dict
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

import rasterio
import albumentations as A

# 可选：屏蔽 rasterio 的未地理参考提醒（对训练无影响）
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


IGBP2HUNAN = np.array([255, 0, 1, 2, 1, 3, 4, 6, 6, 5, 6, 7, 255], dtype=np.int64)
ALLOWED_LABEL_SET = set(range(7)) | {255}  # {0..6, 255}

def remap_label_to_0_6_255(lc: np.ndarray) -> np.ndarray:
    """
    将原始 IGBP 标签重映射为 {0..6, 255}：
      1) 先把 255 替换成 12；2) 用 IGBP2HUNAN 查表映射；
      3) 把 7 合并到 6；最终仅保留 {0..6,255}
    """
    lc = lc.copy()
    lc[lc == 255] = 12
    lc = IGBP2HUNAN[lc]
    # 兜底：有些瓦片会出现 7（作者映射里保留了 7），合并为 6，确保只有 0..6/255
    lc[lc == 7] = 6
    return lc.astype(np.int64, copy=False)


# =========================
# 工具函数
# =========================

def read_geotiff_as_hwc(path: str, band_indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    读取 GeoTIFF -> [H, W, C] np.float32
    band_indices: 选择波段（1-based）；None = 读全部波段
    """
    with rasterio.open(path) as src:
        if band_indices is None:
            arr = src.read()  # [C, H, W]
        else:
            arr = src.read(band_indices)  # [C, H, W]
    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)  # -> [H, W, C]
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def read_mask_as_hw(path: str) -> np.ndarray:
    """读取语义标签 -> [H, W] np.int64（保持原类别编号）"""
    with rasterio.open(path) as src:
        mask = src.read(1)

    # 若是浮点，先取最近整数；然后转 int64
    if mask.dtype.kind == "f":
        mask = np.rint(mask).astype(np.int64)
    else:
        mask = mask.astype(np.int64)

    # ★ 关键：做 IGBP -> Hunan7 映射，并把 7 并入 6
    mask = remap_label_to_0_6_255(mask)
    return mask


def ensure_3ch(arr: np.ndarray, select_idx: Tuple[int, int, int] = (0, 1, 2), fill_value: float = 0.0) -> np.ndarray:
    """
    （旧实现保留作参考，当前不再在 Dataset 中调用）
    把 [H,W,C] 强制转成 [H,W,3]
    - C==1: 复制三份
    - C==2: 补一个常数通道
    - C>=3: 选取指定的三个通道（默认取前 3）
    - 若输入为 [H,W]，先扩成 [H,W,1]
    """
    if arr.ndim == 2:
        arr = arr[..., None]
    C = arr.shape[2]
    if C == 3:
        return arr
    if C == 1:
        return np.repeat(arr, 3, axis=2)
    if C == 2:
        pad = np.full_like(arr[..., :1], fill_value)
        return np.concatenate([arr, pad], axis=2)
    idx = np.clip(np.array(select_idx, dtype=int), 0, C - 1)
    return arr[..., idx]


def robust_minmax_normalize_per_channel(
    arr: np.ndarray,
    lower: float = 2.0,
    upper: float = 98.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    对每个通道做鲁棒 Min-Max 归一化：
      1) 按通道统计 [lower, upper] 分位数；
      2) 线性映射到 [0,1]，再裁剪到 [0,1]；
    这样能适配 S2/S1/TOPO 不同动态范围，而无需事先知道全局统计量。
    """
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w, c = arr.shape
    flat = arr.reshape(-1, c).astype(np.float32)

    q_low = np.percentile(flat, lower, axis=0)
    q_high = np.percentile(flat, upper, axis=0)

    q_low = q_low.reshape(1, 1, -1)
    q_high = q_high.reshape(1, 1, -1)

    scale = q_high - q_low
    scale = np.where(scale < eps, 1.0, scale)

    out = (arr - q_low) / scale
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def augment_s2_intensity(arr: np.ndarray) -> np.ndarray:
    """
    针对 S2（多光谱）做轻量强度增强：随机亮度/对比度、少量高斯噪声。
    """
    x = arr.astype(np.float32)
    # 随机整体亮度缩放
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        x = x * scale
    # 随机加一点偏移
    if np.random.rand() < 0.5:
        span = float(np.percentile(x, 90) - np.percentile(x, 10) + 1e-6)
        bias = np.random.uniform(-0.05, 0.05) * span
        x = x + bias
    # 轻微高斯噪声
    if np.random.rand() < 0.3:
        noise_std = 0.02 * (float(np.std(x)) + 1e-6)
        noise = np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        x = x + noise
    return x


def augment_s1_intensity(arr: np.ndarray) -> np.ndarray:
    """
    针对 S1（SAR）做轻量乘性噪声 + 加性噪声，避免过强颜色增强。
    """
    x = arr.astype(np.float32)
    # 乘性噪声（类似 speckle 的整体扰动）
    if np.random.rand() < 0.7:
        mul = np.random.normal(loc=1.0, scale=0.05)
        mul = float(np.clip(mul, 0.85, 1.15))
        x = x * mul
    # 轻微加性噪声
    if np.random.rand() < 0.3:
        base = float(np.mean(np.abs(x)) + 1e-6)
        noise_std = 0.02 * base
        noise = np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        x = x + noise
    return x


def augment_topo_intensity(arr: np.ndarray) -> np.ndarray:
    """
    针对 TOPO/DEM/TRI：只做非常轻微的加性噪声，保持整体结构。
    """
    x = arr.astype(np.float32)
    if np.random.rand() < 0.2:
        noise_std = 0.01 * (float(np.std(x)) + 1e-6)
        noise = np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        x = x + noise
    return x


# =========================
# Albumentations 几何增强（只做几何，强制对齐）
# =========================

def build_geo_transform(img_size: Optional[Tuple[int, int]] = None,
                        is_train: bool = True) -> Optional[A.Compose]:
    """
    只包含几何变换（翻转、旋转、缩放/重采样），保证多模态和 mask 严格对齐。
    归一化和模态相关的强度增强在 Dataset.__getitem__ 中统一处理。
    """
    tfms = []
    if is_train:
        # 适度几何增强：翻转 + 90° 旋转 + 小角度旋转
        tfms.append(A.HorizontalFlip(p=0.0))
        tfms.append(A.VerticalFlip(p=0.0))
        tfms.append(A.RandomRotate90(p=0.0))
        tfms.append(
            A.ShiftScaleRotate(
                shift_limit=0.0,
                scale_limit=0.0,
                rotate_limit=15,
                border_mode=0,      # CONSTANT
                value=0,
                mask_value=255,
                p=0.0,
            )
        )

    if img_size is not None:
        h, w = img_size
        tfms.append(A.Resize(height=h, width=w, interpolation=1))  # 1=LINEAR

    if not tfms:
        return None

    return A.Compose(tfms, additional_targets={
        'aux1': 'image',
        'aux2': 'image'
    })


# 旧接口保留（现在仅做“恒等变换”），避免外部代码直接调用时报错。
def get_training_transform():
    return A.Compose([])


def train_aug(img, aux1, aux2, mask):
    img, aux1, aux2, mask = np.array(img), np.array(aux1), np.array(aux2), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    aug_aux1 = get_training_transform()(image=aux1.copy())
    aux1 = aug_aux1['image']
    aug_aux2 = get_training_transform()(image=aux2.copy())
    aux2 = aug_aux2['image']
    return img, aux1, aux2, mask


def get_val_transform():
    return A.Compose([])


def val_aug(img, aux1, aux2, mask):
    img, aux1, aux2, mask = np.array(img), np.array(aux1), np.array(aux2), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    aug_aux1 = get_val_transform()(image=aux1.copy())
    aux1 = aug_aux1['image']
    aug_aux2 = get_val_transform()(image=aux2.copy())
    aux2 = aug_aux2['image']
    return img, aux1, aux2, mask


# =========================
# Dataset 实现
# =========================

class HunanDataset(Dataset):
    """
    目录结构示例：
        data_root/
            s2/   s2_XXXXX.tif
            s1/   s1_XXXXX.tif
            topo/ topo_XXXXX.tif   或 tri/ tri_XXXXX.tif
            lc/   lc_XXXXX.tif

    默认按“目录名_”作为各自文件名前缀；若某目录无前缀，可用 prefix_map 覆盖为 ""。
    """

    def __init__(
        self,
        data_root: str,
        rgb_dir: str = 's2',
        aux1_dir: str = 's1',
        aux2_dir: str = 'topo',            # 或 'tri'
        mask_dir: str = 'lc',
        suffix: str = '.tif',
        band_indices_s2: Optional[Sequence[int]] = None,  # None = 保留 S2 全部波段（推荐）
        img_size: Optional[Tuple[int, int]] = None,       # 若需统一尺寸，传 (H,W)
        train: bool = True,
        transform: Optional[object] = None,               # 建议只放几何变换
        id_list: Optional[List[str]] = None,              # 外部指定 id 列表；否则自动扫描
        prefix_map: Optional[Dict[str, str]] = None,      # 各目录的文件名前缀；默认 "{dir}_"
        strip_rgb_prefix: bool = True,                    # 扫描 rgb_dir 时是否剥除前缀
    ):
        super().__init__()
        self.data_root = data_root
        self.rgb_dir = rgb_dir
        self.aux1_dir = aux1_dir
        self.aux2_dir = aux2_dir
        self.mask_dir = mask_dir
        self.suffix = suffix
        self.band_indices_s2 = band_indices_s2
        self.img_size = img_size
        self.train = train
        self.strip_rgb_prefix = strip_rgb_prefix

        # 前缀映射
        if prefix_map is None:
            self.prefix_map = {
                rgb_dir:  f"{rgb_dir}_",
                aux1_dir: f"{aux1_dir}_",
                aux2_dir: f"{aux2_dir}_",
                mask_dir: f"{mask_dir}_",
            }
        else:
            self.prefix_map = prefix_map

        # 扫描 id
        self.img_ids = id_list if id_list is not None else self._scan_ids()
        if not isinstance(self.img_ids, list) or len(self.img_ids) == 0:
            raise RuntimeError(f"No samples found under {osp.join(self.data_root, self.rgb_dir)} with suffix {self.suffix}")

        # transform：若未提供，则只做几何对齐（可选）
        self.geo_transform = transform if transform is not None else build_geo_transform(img_size, is_train=train)
        # 兼容旧代码中直接访问 self.transform
        self.transform = self.geo_transform

    # ---------- 扫描与长度 ----------
    def _scan_ids(self) -> List[str]:
        img_dir = osp.join(self.data_root, self.rgb_dir)
        if not osp.isdir(img_dir):
            raise FileNotFoundError(f"Not a directory: {img_dir}")
        rgb_prefix = self.prefix_map.get(self.rgb_dir, "")
        ids = []
        for fn in os.listdir(img_dir):
            if not fn.lower().endswith(self.suffix.lower()):
                continue
            stem = osp.splitext(fn)[0]  # 例如 "s2_11624"
            if self.strip_rgb_prefix and rgb_prefix and stem.startswith(rgb_prefix):
                stem = stem[len(rgb_prefix):]  # -> "11624"
            ids.append(stem)
        ids.sort()
        return ids

    def __len__(self) -> int:
        return len(self.img_ids)

    # ---------- 读取单样本 ----------
    def load_img_and_mask(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        直接保留各模态的原始通道数：
          - S2: 由 band_indices_s2 控制（None = 全部波段）；
          - S1: 全部波段（当前为 2 通道）；
          - TOPO: 全部波段（当前为 2 通道）。
        """
        img_id = self.img_ids[index]
        p_rgb = self.prefix_map.get(self.rgb_dir, "")
        p_a1  = self.prefix_map.get(self.aux1_dir, "")
        p_a2  = self.prefix_map.get(self.aux2_dir, "")
        p_msk = self.prefix_map.get(self.mask_dir, "")

        img_name = osp.join(self.data_root, self.rgb_dir,  p_rgb + img_id + self.suffix)
        a1_name  = osp.join(self.data_root, self.aux1_dir, p_a1  + img_id + self.suffix)
        a2_name  = osp.join(self.data_root, self.aux2_dir, p_a2  + img_id + self.suffix)
        msk_name = osp.join(self.data_root, self.mask_dir, p_msk + img_id + self.suffix)

        # 读取原始数组（H,W,C）
        img  = read_geotiff_as_hwc(img_name)  # S2 多波段
        aux1 = read_geotiff_as_hwc(a1_name, band_indices=self.band_indices_s2)                                      # S1（2 通道）
        aux2 = read_geotiff_as_hwc(a2_name)                                      # TOPO/DEM/TRI（2 通道）
        mask = read_mask_as_hw(msk_name)                                         # [H,W] int64

        # 不再强制 reshape 为 3 通道，完整保留原始通道数
        return img, aux1, aux2, mask

    # ---------- 强度增强 & 归一化 ----------
    def _apply_train_intensity(self,
                               img: np.ndarray,
                               aux1: np.ndarray,
                               aux2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        训练阶段：几何变换之后，对各模态分别做轻量强度增强 + 鲁棒归一化。
        """
        # 模态相关强度增强
        img_aug = augment_s2_intensity(img)
        aux1_aug = augment_s1_intensity(aux1)
        aux2_aug = augment_topo_intensity(aux2)

        # 各模态独立做鲁棒 Min-Max 归一化到 [0,1]
        img_norm = robust_minmax_normalize_per_channel(img_aug)
        aux1_norm = robust_minmax_normalize_per_channel(aux1_aug)
        aux2_norm = robust_minmax_normalize_per_channel(aux2_aug)
        return img_norm, aux1_norm, aux2_norm

    def _apply_eval_intensity(self,
                              img: np.ndarray,
                              aux1: np.ndarray,
                              aux2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        验证/测试阶段：不做随机增强，只做鲁棒 Min-Max 归一化。
        """
        img_norm = robust_minmax_normalize_per_channel(img)
        aux1_norm = robust_minmax_normalize_per_channel(aux1)
        aux2_norm = robust_minmax_normalize_per_channel(aux2)
        return img_norm, aux1_norm, aux2_norm

    # ---------- __getitem__ ----------
    def __getitem__(self, index: int):
        img, aux1, aux2, mask = self.load_img_and_mask(index)

        # 先做几何变换（Albumentations 或普通函数），保证多模态 & mask 对齐
        if self.geo_transform is not None:
            if isinstance(self.geo_transform, A.Compose):
                data = self.geo_transform(image=img, aux1=aux1, aux2=aux2, mask=mask)
                img, aux1, aux2, mask = data['image'], data['aux1'], data['aux2'], data['mask']
            else:
                # 普通函数：f(img, aux1, aux2, mask) -> (img, aux1, aux2, mask)
                img, aux1, aux2, mask = self.geo_transform(img, aux1, aux2, mask)

        # 再做模态相关的强度增强 & 归一化
        if self.train:
            img, aux1, aux2 = self._apply_eval_intensity(img, aux1, aux2)
        else:
            img, aux1, aux2 = self._apply_eval_intensity(img, aux1, aux2)

        # np(HWC/HW) -> torch(CHW/HW)
        if img.ndim == 2:
            img = img[..., None]
        if aux1.ndim == 2:
            aux1 = aux1[..., None]
        if aux2.ndim == 2:
            aux2 = aux2[..., None]

        img_t  = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        aux1_t = torch.from_numpy(aux1).permute(2, 0, 1).contiguous().float()
        aux2_t = torch.from_numpy(aux2).permute(2, 0, 1).contiguous().float()
        mask_t = torch.from_numpy(mask).long()

        return {
            'img': img_t,
            'aux1': aux1_t,
            'aux2': aux2_t,
            'gt_semantic_seg': mask_t,
            'img_id': self.img_ids[index],
        }


# =========================
# 用法示例（按需放到你的 config/脚本中）：
# =========================
# 推荐做法：直接让 Dataset 负责归一化 + 训练增强，外部 transform 只负责几何（可省略）。
#
# train_set = HunanDataset(
#     data_root=r'H:\Datasets\Hunan_Dataset\Hunan_Dataset\train',
#     rgb_dir='s2', aux1_dir='s1', aux2_dir='topo', mask_dir='lc',
#     suffix='.tif',
#     band_indices_s2=None,           # None = 使用 S2 全部 13 个波段
#     img_size=(512, 512),
#     train=True,
#     transform=None,                 # 或 transform=build_geo_transform((512,512), is_train=True)
#     # 若某目录无前缀，例如 lc/11624.tif，使用：
#     # prefix_map={'s2':'s2_', 's1':'s1_', 'topo':'topo_', 'lc':''},
# )
#
# val_set = HunanDataset(
#     data_root=r'H:\Datasets\Hunan_Dataset\Hunan_Dataset\test',
#     rgb_dir='s2', aux1_dir='s1', aux2_dir='topo', mask_dir='lc',
#     suffix='.tif',
#     band_indices_s2=None,
#     img_size=(512, 512),
#     train=False,                    # 会自动关闭随机增强，只保留鲁棒归一化
#     transform=None,                 # 或 transform=build_geo_transform((512,512), is_train=False)
# )
