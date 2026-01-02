# -*- coding: utf-8 -*-
"""
Vaihingen Dataset - 在线滑动窗口 + 简单数据增强版本（IRRG + DSM 一个辅助模态）

命名规则（以 area2 为例）：
    IRRG : top_mosaic_09cm_area2.tif
    DSM  : dsm_09cm_matching_area2.tif
    Label: top_mosaic_09cm_area2.tif  （这里已是索引标签：0~4，clutter=255）

本文件中 IRRG / DSM 的预处理严格对齐 MFNet（SSRS/MFNet/utils.py）中 ISPRS_dataset 对
Vaihingen 的处理方式：
    - IRRG: 读取原始 IRRG，按 1/255 线性缩放到 [0,1]
    - DSM : 读取原始 DSM，高度值按 per-image min-max 归一化到 [0,1]

最终输出（__getitem__）中：
    - img  : torch.float32，[3, H, W]   —— IRRG 三通道
    - aux2 : torch.float32，[1, H, W]   —— DSM 单通道
    - mask : torch.int64，[H, W]        —— 像素取值 {0,1,2,3,4,255}
"""

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from PIL import Image

# ===================== 一些常量定义 =====================

CLASSES = ('impervious', 'building', 'low_veg', 'tree', 'car')
PALETTE = [
    [255, 255, 255],   # impervious
    [0, 0, 255],       # building
    [0, 255, 255],     # low vegetation
    [0, 255, 0],       # tree
    [255, 255, 0],     # car
]

# 原始大图尺寸 (ISPRS Vaihingen IRRG / DSM 09cm)
ORIGIN_IMG_SIZE = (512, 512)   # 这里只是默认 patch 大小，不是整幅图尺寸
ORIGIN_STRIDE = (512, 512)

# 文件名前缀 / 后缀（按 ISPRS Vaihingen 命名）
RGB_PREFIX  = 'top_mosaic_09cm_'
RGB_SUFFIX  = '.tif'
AUX2_PREFIX = 'dsm_09cm_matching_'
AUX2_SUFFIX = '.tif'
MASK_PREFIX = 'top_mosaic_09cm_'
MASK_SUFFIX = '.tif'


# ===================== DSM 图像归一化（MFNet 风格） =====================

def minmax_normalize_dsm(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    对 DSM（Digital Surface Model）做 per-image min-max 归一化，严格模仿 MFNet:

        dsm = io.imread(...)
        dsm = 1.0 * dsm
        min_v = dsm.min()
        max_v = dsm.max()
        dsm = (dsm - min_v) / (max_v - min_v)

    本实现支持 2D 或 3D 输入，返回 float32、[H, W, 1]。
    """
    if arr.ndim == 3:
        # Vaihingen DSM 一般是单通道，如果是 [H,W,1] 或 [H,W,3]，这里取第 1 通道
        arr = arr[..., 0]
    arr = arr.astype(np.float32)

    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < eps:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = (arr - min_v) / (max_v - min_v)

    # [H, W] -> [H, W, 1]
    norm = norm[..., None]
    return norm


# ===================== Albumentations 增强（仅几何，不做 Normalize） =====================

def get_training_transform():
    """
    训练增强：
    - 随机水平翻转
    - 随机垂直翻转
    - 随机 90° 旋转

    注意：不再使用 albu.Normalize()，因为 IRRG / DSM 已在读取阶段缩放到 [0,1]。
    """
    return albu.Compose(
        [
            albu.HorizontalFlip(p=0.1),
            albu.VerticalFlip(p=0.1),
            albu.RandomRotate90(p=0.1),
        ],
        additional_targets={
            'aux2': 'image',
        }
    )


def train_aug(img, aux2, mask):
    """
    img, aux2, mask 均为 np.array
    img  : [H, W, 3]，0~1
    aux2 : [H, W, 1]，0~1
    mask : [H, W]，int（0~4,255）
    """
    img = np.array(img)
    aux2 = np.array(aux2)
    mask = np.array(mask)

    aug = get_training_transform()(
        image=img,
        aux2=aux2,
        mask=mask,
    )
    img = aug['image']
    aux2 = aug['aux2']
    mask = aug['mask']

    return img, aux2, mask


def get_val_transform():
    """
    验证 / 测试阶段：不做任何数值归一化，只保持几何一致性。
    这里给一个“空”的 Compose，便于与 train_aug 统一接口。
    """
    return albu.Compose(
        [],
        additional_targets={
            'aux2': 'image',
        }
    )


def val_aug(img, aux2, mask):
    img = np.array(img)
    aux2 = np.array(aux2)
    mask = np.array(mask)

    aug = get_val_transform()(
        image=img,
        aux2=aux2,
        mask=mask,
    )
    img = aug['image']
    aux2 = aug['aux2']
    mask = aug['mask']

    return img, aux2, mask


# ===================== 在线滑动窗口 Dataset =====================

class VaihingenDataset(Dataset):
    """
    Vaihingen Dataset（IRRG + DSM），在线滑动窗口版本。

    - IRRG: 3 通道，按 MFNet 方式线性缩放到 [0,1]
    - DSM : 1 通道，按 MFNet 方式做 per-image min-max 到 [0,1]
    - Label: 你已经离线映射为 {0,1,2,3,4,255}，这里不再做颜色映射，直接读取灰度标签。
    """

    def __init__(
        self,
        data_root=r'H:\\Datasets\\Vaihingen\\Data',
        rgb_dir='rgb',
        aux2_dir='dsm',
        mask_dir='label',
        transform=train_aug,
        img_size=ORIGIN_IMG_SIZE,
        stride=None,
        cache=True,
    ):
        super().__init__()

        self.data_root = data_root
        self.rgb_dir = rgb_dir
        self.aux2_dir = aux2_dir  # DSM 文件夹
        self.mask_dir = mask_dir
        self.transform = transform

        self.img_size = img_size
        self.stride = stride if stride is not None else img_size
        self.cache = cache

        # 简单的 in-memory cache（按大图 ID 缓存）
        self.rgb_cache = {} if cache else None
        self.aux2_cache = {} if cache else None
        self.mask_cache = {} if cache else None

        # 获取所有大图 ID（如 'area2', 'area3', ...）
        self.img_ids = self.get_img_ids(self.data_root, self.rgb_dir)

        # 构建滑动窗口索引
        self.sliding_index = self.build_sliding_index()

    # ---------- 读取整幅 IRRG / DSM / Label，并完成预处理 ----------

    def _load_full_img_and_mask(self, img_id):
        """
        读取完整的 IRRG、DSM、Mask（整幅大图），并完成：
            - IRRG: 0~255 -> 0~1, float32，[H,W,3]
            - DSM : per-image min-max -> 0~1, float32，[H,W,1]
            - Mask: 灰度索引标签，[H,W]，int64
        """
        if self.cache and img_id in self.rgb_cache:
            img = self.rgb_cache[img_id]
            aux2 = self.aux2_cache[img_id]
            mask = self.mask_cache[img_id]
            return img, aux2, mask

        img_name = self._build_path('rgb', img_id)
        aux2_name = self._build_path('aux2', img_id)
        mask_name = self._build_path('mask', img_id)

        # ----- IRRG -----
        img = np.array(Image.open(img_name))
        # Vaihingen IRRG 一般为 3 通道，如果意外是单通道，简单重复三次
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        img = img.astype(np.float32) / 255.0  # MFNet: 1/255 缩放到 [0,1]

        # ----- DSM -----
        dsm_raw = np.array(Image.open(aux2_name))
        aux2 = minmax_normalize_dsm(dsm_raw)  # [H,W,1], float32, [0,1]

        # ----- Label -----
        # 这里假设你已经离线把颜色标签转换成了索引标签 (0~4, clutter=255)
        # 因此直接读取灰度即可，不再做 convert_from_color。
        mask = np.array(Image.open(mask_name).convert('L'), dtype=np.uint8)  # [H,W]
        mask = mask.astype(np.int64)

        if self.cache:
            self.rgb_cache[img_id] = img
            self.aux2_cache[img_id] = aux2
            self.mask_cache[img_id] = mask

        return img, aux2, mask

    # ---------- Dataset 标准接口 ----------

    def __len__(self):
        return len(self.sliding_index)

    def __getitem__(self, index):
        """
        返回一个 patch：
            img  : [3, H, W], float32
            aux2 : [1, H, W], float32
            mask : [H, W],    int64
        """
        img_idx, y, x = self.sliding_index[index]
        img_id = self.img_ids[img_idx]

        img_full, aux2_full, mask_full = self._load_full_img_and_mask(img_id)

        patch_h, patch_w = self.img_size
        # 切 patch
        img = img_full[y:y + patch_h, x:x + patch_w, :]       # [H,W,3]
        aux2 = aux2_full[y:y + patch_h, x:x + patch_w, :]     # [H,W,1]
        mask = mask_full[y:y + patch_h, x:x + patch_w]        # [H,W]

        # 数据增强（几何一致）
        if self.transform is not None:
            img, aux2, mask = self.transform(img, aux2, mask)

        # 转 tensor: [H,W,C] -> [C,H,W]
        img = torch.from_numpy(img).permute(2, 0, 1).float()     # [3,H,W]
        aux2 = torch.from_numpy(aux2).permute(2, 0, 1).float()   # [1,H,W]
        mask = torch.from_numpy(mask).long()                     # [H,W]

        patch_id = f"{img_id}_y{y}_x{x}"

        return {
            'img': img,                  # IRRG 主模态
            'aux1': aux2,                # DSM 模态
            'gt_semantic_seg': mask,
            'img_id': patch_id,
        }

    # ---------- 其他辅助方法 ----------

    def get_img_ids(self, data_root, img_dir):
        """
        在 rgb_dir 中扫描所有 top_mosaic_09cm_*.tif，
        把后面的 ID 提取出来（如 area2）。
        """
        filenames = sorted(os.listdir(osp.join(data_root, img_dir)))
        ids = []
        for name in filenames:
            if not name.startswith(RGB_PREFIX):
                continue
            if not name.endswith(RGB_SUFFIX):
                continue
            # 去掉前缀和后缀，中间就是 ID
            stem = name[len(RGB_PREFIX):-len(RGB_SUFFIX)]
            if stem:
                ids.append(stem)

        print(f'Found {len(ids)} images in {osp.join(data_root, img_dir)}')
        return ids

    def _build_path(self, kind, img_id):
        """
        根据 kind='rgb'/'aux2'/'mask' 和 img_id='area2' 构造完整路径。
        """
        if kind == 'rgb':
            fname = RGB_PREFIX + img_id + RGB_SUFFIX
            return osp.join(self.data_root, self.rgb_dir, fname)
        elif kind == 'aux2':
            fname = AUX2_PREFIX + img_id + AUX2_SUFFIX
            return osp.join(self.data_root, self.aux2_dir, fname)
        elif kind == 'mask':
            fname = MASK_PREFIX + img_id + MASK_SUFFIX
            return osp.join(self.data_root, self.mask_dir, fname)
        else:
            raise KeyError(f"Unknown kind={kind}")

    def build_sliding_index(self):
        """
        对每张大图，按 img_size 和 stride 生成所有 (img_idx, y, x)。
        """
        index = []
        patch_h, patch_w = self.img_size
        stride_h, stride_w = self.stride

        for img_idx, img_id in enumerate(self.img_ids):
            rgb_path = self._build_path('rgb', img_id)
            aux2_path = self._build_path('aux2', img_id)
            mask_path = self._build_path('mask', img_id)

            # 取三种模态的公共有效区域大小
            with Image.open(rgb_path) as im_rgb:
                w_rgb, h_rgb = im_rgb.size
            with Image.open(aux2_path) as im_aux2:
                w_aux2, h_aux2 = im_aux2.size
            with Image.open(mask_path) as im_mask:
                w_mask, h_mask = im_mask.size

            w = min(w_rgb, w_aux2, w_mask)
            h = min(h_rgb, h_aux2, h_mask)

            if h < patch_h or w < patch_w:
                print(f"[Warn] image {img_id} effective area ({h},{w}) "
                      f"smaller than patch size {self.img_size}, skip.")
                continue

            ys = list(range(0, h - patch_h + 1, stride_h))
            if ys[-1] != h - patch_h:
                ys.append(h - patch_h)

            xs = list(range(0, w - patch_w + 1, stride_w))
            if xs[-1] != w - patch_w:
                xs.append(w - patch_w)

            for y in ys:
                for x in xs:
                    index.append((img_idx, y, x))

        print('Total sliding-window patches:', len(index))
        return index
