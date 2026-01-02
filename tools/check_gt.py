import numpy as np
from PIL import Image
from pathlib import Path

# RGB2PIXEL 对应的逆映射规则
PIXEL2RGB = {
    0: (255, 255, 255),  # impervious
    1: (0, 0, 255),  # building
    2: (0, 255, 255),  # low vegetation
    3: (0, 255, 0),  # tree
    4: (255, 255, 0),  # car
    # 5: (255, 0, 0),  # clutter
}


IGNORE_VALUE = 255


def pixel_to_rgb(arr_ids: np.ndarray) -> np.ndarray:
    """
    将像素值图（0..5, 255）映射回 RGB 图像。
    :param arr_ids: 输入单通道像素值图（0..5, 255）
    :return: 对应的 RGB 图像
    """
    H, W = arr_ids.shape
    arr_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for pixel_value, rgb_value in PIXEL2RGB.items():
        arr_rgb[arr_ids == pixel_value] = rgb_value

    # 忽略值映射为白色（255,255,255）
    arr_rgb[arr_ids == IGNORE_VALUE] = (0, 0, 0)

    return arr_rgb


def save_rgb_from_pixel(in_dir, out_dir, suffixes=(".png", ".tif", ".tiff", ".jpg", ".jpeg"), recursive=True):
    """
    批量将单通道像素值图（0..5, 255）转换回 RGB 图像并保存
    :param in_dir: 输入文件夹
    :param out_dir: 输出文件夹
    :param suffixes: 要处理的文件扩展名
    :param recursive: 是否递归子目录
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = in_dir.rglob("*") if recursive else in_dir.iterdir()
    for p in files:
        if p.is_file() and p.suffix.lower() in suffixes:
            # 读取图像并转换为 numpy 数组
            img = Image.open(p)
            arr = np.array(img, dtype=np.uint8)

            # 将像素值图转换为 RGB 图像
            rgb_arr = pixel_to_rgb(arr)

            # 保持子目录结构，保存 RGB 图像
            out_path = out_dir / p.relative_to(in_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb_arr).save(out_path)

            print(f"已处理并保存图像: {out_path}")


# 使用示例：
input_folder = r"H:\Datasets\Vaihingen\ISPRS_semantic_labeling_Vaihingen\data\train_set\label_no_clutter_no_broundary"  # 输入单通道像素值图文件夹路径
output_folder = r"H:\Datasets\Vaihingen\ISPRS_semantic_labeling_Vaihingen\data\train_set\label_no_clutter_no_broundary_rgb"  # 输出 RGB 图像文件夹路径

save_rgb_from_pixel(input_folder, output_folder)
