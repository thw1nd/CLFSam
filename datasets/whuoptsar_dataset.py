import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from PIL import Image
import random


CLASSES = ('farmland', 'city', 'village', 'water', 'forest', 'road', 'others')

PALETTE = [[0, 0, 255], [0, 255, 255], [0, 255, 0],
           [255, 255, 0], [255, 0, 0], [0, 0, 255], [255, 125, 225]]

def ensure_rgb3(arr, band_indices=(0, 1, 2)):
    '''
    将任意 shape 的影像变成 HxWx3：
      - 多波段(>=3)：按 band_indices 取 3 个通道（默认前三个）
      - 单通道：复制为 3 通道
      - 2D：也复制为 3 通道
    注意：这里不会进行任何颜色空间转换，仅做通道切片/复制，
         可避免 PIL.convert('RGB') 在 RGBA 情况下的隐式处理。
    '''
    import numpy as _np
    arr = _np.asarray(arr)
    if arr.ndim == 2:  # HxW
        return _np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3:
        h, w, c = arr.shape
        if c >= 3:
            b0, b1, b2 = band_indices
            return _np.stack([arr[..., b0], arr[..., b1], arr[..., b2]], axis=-1)
        elif c == 1:
            return _np.concatenate([arr, arr, arr], axis=-1)
        else:
            pad = arr[..., :1]
            return _np.concatenate([arr, pad], axis=-1)
    raise ValueError()

ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)
def get_training_transform():
    train_transform = [
        # albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.15),
        # albu.RandomRotate90(p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


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
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, aux1, aux2, mask):
    img, aux1, aux2, mask = np.array(img), np.array(aux1), np.array(aux2), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    aug_aux1 = get_val_transform()(image=aux1.copy())
    aux1 = aug_aux1['image']
    aug_aux2 = get_val_transform()(image=aux2.copy())
    aux2 = aug_aux2['image']
    return img, aux1, aux2, mask


class WhuoptsarDataset(Dataset):
    def __init__(self, data_root=r'H:\Datasets\WHUOPTSAR\data\train',
                 rgb_dir='opt',
                 aux1_dir='opt',
                 aux2_dir='sar',
                 mosaic_ratio=0.0,
                 mask_dir='label',
                 suffix='.png',
                 transform=train_aug, img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.aux1_dir = aux1_dir
        self.aux2_dir = aux2_dir
        self.mosaic_ratio = mosaic_ratio

        self.suffix = suffix
        self.transform = transform
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.rgb_dir, self.aux1_dir, self.aux2_dir, self.mask_dir)

    def __getitem__(self, index):
        # p_ratio = random.random()
        img, aux1, aux2, mask = self.load_img_and_mask(index)
        # if p_ratio < self.mosaic_ratio:
            # img, aux1, aux2, mask = self.load_mosaic_img_and_mask(index)
        # if self.transform:
            # img, aux1, aux2, mask = self.transform(img, aux1, aux2, mask)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        aux1 = torch.from_numpy(aux1).permute(2, 0, 1).float()
        aux2 = torch.from_numpy(aux2).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = {'img': img, 'aux1': aux1, 'aux2': aux2, 'gt_semantic_seg': mask, 'img_id': img_id}

        return results

    def __len__(self):
        length = len(self.img_ids)
        return length

    def get_img_ids(self, data_root, img_dir, aux1_dir, aux2_dir, mask_dir):
        opt_filename_list = os.listdir(osp.join(data_root, img_dir))
        aux1_filename_list = os.listdir(osp.join(data_root, aux1_dir))
        aux2_filename_list = os.listdir(osp.join(data_root, aux2_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        print(len(opt_filename_list))
        print(len(aux1_filename_list))
        print(len(aux2_filename_list))
        print(len(mask_filename_list))

        assert len(opt_filename_list) == len(mask_filename_list) == len(aux1_filename_list) == len(aux2_filename_list)
        img_ids = [(str(id.split('.')[0])) for id in opt_filename_list]
        img_ids = img_ids

        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.rgb_dir, img_id + self.suffix)
        aux1_name = osp.join(self.data_root, self.aux1_dir, img_id + self.suffix)
        aux2_name = osp.join(self.data_root, self.aux2_dir, img_id + self.suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.suffix)
        img = np.array(Image.open(img_name))  # 保留原始通道顺序（可能是 RGBA 或更多带）
        img = ensure_rgb3(img, band_indices=(0, 1, 2))  # 仅取前三通道作为 RGB

        aux1 = np.array(Image.open(aux1_name))
        aux1 = ensure_rgb3(aux1, band_indices=(0, 1, 2))

        aux2 = np.array(Image.open(aux2_name))  # SAR 若为单通道会被复制成3通道
        aux2 = ensure_rgb3(aux2, band_indices=(0, 1, 2))
        mask = Image.open(mask_name).convert('L')
        mask = np.array(mask)

        return img, aux1, aux2, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, aux1_a, aux2_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, aux1_b, aux2_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, aux1_c, aux2_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, aux1_d, aux2_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, aux1_a, aux2_a, mask_a = np.array(img_a), np.array(aux1_a), np.array(aux2_a), np.array(mask_a)
        img_b, aux1_b, aux2_b, mask_b = np.array(img_b), np.array(aux1_b), np.array(aux2_b), np.array(mask_b)
        img_c, aux1_c, aux2_c, mask_c = np.array(img_c), np.array(aux1_c), np.array(aux2_c), np.array(mask_c)
        img_d, aux1_d, aux2_d, mask_d = np.array(img_d), np.array(aux1_d), np.array(aux2_d), np.array(mask_d)

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), aux1=aux1_a.copy(), aux2=aux2_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), aux1=aux1_b.copy(), aux2=aux2_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), aux1=aux1_c.copy(), aux2=aux2_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), aux1=aux1_d.copy(), aux2=aux2_d.copy(), mask=mask_d.copy())

        img_crop_a, aux1_crop_a, aux2_crop_a, mask_crop_a = croped_a['image'], croped_a['aux1'], croped_a['aux2'], croped_a['mask']
        img_crop_b, aux1_crop_b, aux2_crop_b, mask_crop_b = croped_b['image'], croped_b['aux1'], croped_b['aux2'], croped_b['mask']
        img_crop_c, aux1_crop_c, aux2_crop_c, mask_crop_c = croped_c['image'], croped_c['aux1'], croped_c['aux2'], croped_c['mask']
        img_crop_d, aux1_crop_d, aux2_crop_d, mask_crop_d = croped_d['image'], croped_d['aux1'], croped_d['aux2'], croped_d['mask']

        img_top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        img_bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((img_top, img_bottom), axis=0)

        aux1_top = np.concatenate((aux1_crop_a, aux1_crop_b), axis=1)
        aux1_bottom = np.concatenate((aux1_crop_c, aux1_crop_d), axis=1)
        aux1 = np.concatenate((aux1_top, aux1_bottom), axis=0)

        aux2_top = np.concatenate((aux2_crop_a, aux2_crop_b), axis=1)
        aux2_bottom = np.concatenate((aux2_crop_c, aux2_crop_d), axis=1)
        aux2 = np.concatenate((aux2_top, aux2_bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        aux1 = np.ascontiguousarray(aux1)
        aux2 = np.ascontiguousarray(aux2)

        img = Image.fromarray(img)
        aux1 = Image.fromarray(aux1)
        aux2 = Image.fromarray(aux2)
        mask = Image.fromarray(mask)

        return img, aux1, aux2, mask

'''
Potsdam_val_dataset = PotsdamDataset(data_root=r'/home/a302/tonghw/GeoSeg-main/GeoSeg-main/data/potsdam_dsm/test',
                                        mosaic_ratio=0.0,
                                        transform=val_aug)
                                        '''
