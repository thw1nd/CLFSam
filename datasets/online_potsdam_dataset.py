# -*- coding: utf-8 -*-
"""
Potsdam Dataset - 在线滑动窗口 + 简单数据增强版本（只保留 RGB + DSM 一个辅助模态）

使用方式：
- data_root/
    rgb/    # 整幅 RGB 大图
    dsm/    # 整幅 DSM（或其他模态），在 batch 里会以 'aux2' 这个 key 返回
    label/  # 整幅标签（建议是单通道类别索引图）

- Dataset 会自动对每张大图用 sliding-window 切成 patch，
  __len__ = 所有 patch 数量。

说明：
- 为了兼容你现有的训练 / 测试脚本（用的是 batch["img"] 和 batch["aux2"]），
  这里物理上只读取一份 DSM，但在返回的 dict 里仍然叫 'aux2'。
- 不再需要 aux1 / 第三模态。
"""

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from PIL import Image
import random

# ===================== 基本信息 =====================

CLASSES = ('Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car')

PALETTE = [
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
]

# patch 尺寸（网络输入）
ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)

# ========= 文件后缀（根据你自己的大图格式改） =========
RGB_SUFFIX = '.tif'     # RGB 大图后缀，比如 .tif / .png
AUX2_SUFFIX = '.jpg'    # DSM 或其他模态
MASK_SUFFIX = '.tif'    # label 后缀；建议是单通道类别索引图


# ===================== Albumentations 增强 =====================

def get_training_transform():
    """
    训练增强：
    - 随机水平翻转
    - 随机垂直翻转
    - 随机 90° 旋转
    - Normalize

    通过 additional_targets 保证 RGB / aux2 用同一组几何变换。
    """
    return albu.Compose(
        [
            albu.HorizontalFlip(p=0.1),
            albu.VerticalFlip(p=0.1),
            albu.RandomRotate90(p=0.1),
            albu.Normalize()
        ],
        additional_targets={
            'aux2': 'image',
        }
    )


def train_aug(img, aux2, mask):
    """
    img, aux2, mask 为 np.array
    """
    img = np.array(img)
    aux2 = np.array(aux2)
    mask = np.array(mask)

    aug = get_training_transform()(
        image=img,
        aux2=aux2,
        mask=mask
    )
    img = aug['image']
    aux2 = aug['aux2']
    mask = aug['mask']

    return img, aux2, mask


def get_val_transform():
    """
    验证 / 测试：只做 Normalize
    """
    return albu.Compose(
        [
            albu.Normalize()
        ],
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
        mask=mask
    )
    img = aug['image']
    aux2 = aug['aux2']
    mask = aug['mask']

    return img, aux2, mask


# ===================== 在线 sliding-window Dataset =====================

class PotsdamDataset(Dataset):
    """
    在线 sliding-window 版 Potsdam Dataset（RGB + DSM）

    data_root 目录结构示例：
        data_root/
          rgb/
            2_10.tif, 2_11.tif, ...
          dsm/
            2_10.jpg, 2_11.jpg, ...
          label/
            2_10.tif, 2_11.tif, ...

    要求：三个目录中的文件名（不含扩展名）一一对应。
    """

    def __init__(self,
                 data_root=r'H:\\Datasets\\Potsdam\\Data',
                 rgb_dir='rgb',
                 aux2_dir='dsm',          # DSM / 高程等第二模态
                 mosaic_ratio=0.0,        # 保留参数，当前不使用
                 mask_dir='label_no_broundary_no_clutter',
                 suffix='.png',           # 保留参数，当前不使用
                 transform=train_aug,     # 训练传 train_aug，验证传 val_aug
                 img_size=ORIGIN_IMG_SIZE, # patch 大小 (H, W)
                 stride=None,             # 滑窗步长 (H_step, W_step)，默认等于 img_size（不重叠）
                 cache=True):             # 是否缓存整图到内存
        self.data_root = data_root
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.aux2_dir = aux2_dir
        self.mosaic_ratio = mosaic_ratio
        self.transform = transform

        self.img_size = img_size  # (H, W)
        if stride is None:
            self.stride = img_size
        else:
            self.stride = stride

        self.rgb_suffix = RGB_SUFFIX
        self.aux2_suffix = AUX2_SUFFIX
        self.mask_suffix = MASK_SUFFIX

        self.cache = cache
        self.rgb_cache = {} if cache else None
        self.aux2_cache = {} if cache else None
        self.mask_cache = {} if cache else None

        # 获取所有大图 ID
        self.img_ids = self.get_img_ids(self.data_root, self.rgb_dir,
                                        self.aux2_dir, self.mask_dir)

        # 为每张大图构建 sliding-window 索引（注意考虑三模态尺寸交集）
        self.sliding_index = self.build_sliding_index()

    # ---------- 工具函数 ----------

    def get_img_ids(self, data_root, img_dir, aux2_dir, mask_dir):
        opt_filename_list = sorted(os.listdir(osp.join(data_root, img_dir)))
        aux2_filename_list = sorted(os.listdir(osp.join(data_root, aux2_dir)))
        mask_filename_list = sorted(os.listdir(osp.join(data_root, mask_dir)))

        print('Found {} RGB, {} aux2, {} masks in {}'.format(
            len(opt_filename_list), len(aux2_filename_list),
            len(mask_filename_list), data_root))

        assert len(opt_filename_list) == len(mask_filename_list) == len(aux2_filename_list)

        img_ids = [osp.splitext(name)[0] for name in opt_filename_list]
        return img_ids

    def _load_full_img_and_mask(self, img_id):
        """
        读取一整幅大图，返回 np.array：
        img:  H×W×3
        aux2: H×W×3
        mask: H×W  (单通道)
        """
        if self.cache and img_id in self.rgb_cache:
            img = self.rgb_cache[img_id]
            aux2 = self.aux2_cache[img_id]
            mask = self.mask_cache[img_id]
            return img, aux2, mask

        img_name = osp.join(self.data_root, self.rgb_dir, img_id + self.rgb_suffix)
        aux2_name = osp.join(self.data_root, self.aux2_dir, img_id + self.aux2_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)

        img = np.array(Image.open(img_name).convert('RGB'))
        aux2 = np.array(Image.open(aux2_name).convert('RGB'))
        # 注意：这里假设 label 已经是“单通道类别索引图”
        mask = np.array(Image.open(mask_name).convert('L'))

        if self.cache:
            self.rgb_cache[img_id] = img
            self.aux2_cache[img_id] = aux2
            self.mask_cache[img_id] = mask

        return img, aux2, mask

    def build_sliding_index(self):
        """
        对每张大图，按 img_size 和 stride 生成所有 (img_idx, y, x)。

        为了避免 Albumentations 报 H/W 不一致的错：
        - 这里用 RGB / aux2 / mask 三者 "公共的有效区域"（各自尺寸的最小值）来生成滑窗网格。
        """
        index = []
        patch_h, patch_w = self.img_size
        stride_h, stride_w = self.stride

        for img_idx, img_id in enumerate(self.img_ids):
            rgb_path = osp.join(self.data_root, self.rgb_dir, img_id + self.rgb_suffix)
            aux2_path = osp.join(self.data_root, self.aux2_dir, img_id + self.aux2_suffix)
            mask_path = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)

            with Image.open(rgb_path) as im_rgb:
                w_rgb, h_rgb = im_rgb.size
            with Image.open(aux2_path) as im_aux2:
                w_aux2, h_aux2 = im_aux2.size
            with Image.open(mask_path) as im_mask:
                w_mask, h_mask = im_mask.size

            # 取三模态的交集区域
            w = min(w_rgb, w_aux2, w_mask)
            h = min(h_rgb, h_aux2, h_mask)

            if h < patch_h or w < patch_w:
                # 如果交集区域比 patch 还小，就跳过这一张图
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

        print('Total sliding-window patches: {}'.format(len(index)))
        return index

    # ---------- Dataset 接口 ----------

    def __len__(self):
        return len(self.sliding_index)

    def __getitem__(self, index):
        """
        index: 全局 patch 索引 -> (img_idx, y, x) -> 裁出 patch
        """
        img_idx, y, x = self.sliding_index[index]
        img_id = self.img_ids[img_idx]

        img_full, aux2_full, mask_full = self._load_full_img_and_mask(img_id)

        patch_h, patch_w = self.img_size

        img = img_full[y:y + patch_h, x:x + patch_w, :]
        aux2 = aux2_full[y:y + patch_h, x:x + patch_w, :]
        mask = mask_full[y:y + patch_h, x:x + patch_w]

        # 数据增强（训练时传 train_aug，验证时传 val_aug）
        if self.transform is not None:
            img, aux2, mask = self.transform(img, aux2, mask)

        # 转 tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        aux2 = torch.from_numpy(aux2).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        patch_id = f"{img_id}_y{y}_x{x}"

        results = {
            'img': img,
            'aux2': aux2,                 # 这里仍然叫 aux2，兼容你现有的训练 / 测试代码
            'gt_semantic_seg': mask,
            'img_id': patch_id
        }
        return results

    # ============ mosaic 逻辑（暂时用不到，仅保留） ============

    def load_mosaic_img_and_mask(self, index):
        """
        如果以后你想在整图基础上做 mosaic，可以再改这个函数。
        当前没有在 __getitem__ 里使用 mosaic。
        """
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, aux2_a, mask_a = self._load_full_img_and_mask(self.img_ids[indexes[0]])
        img_b, aux2_b, mask_b = self._load_full_img_and_mask(self.img_ids[indexes[1]])
        img_c, aux2_c, mask_c = self._load_full_img_and_mask(self.img_ids[indexes[2]])
        img_d, aux2_d, mask_d = self._load_full_img_and_mask(self.img_ids[indexes[3]])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        start_y = h // 4
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(start_y, (h - start_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        img_top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        img_bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((img_top, img_bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)

        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return img, mask
