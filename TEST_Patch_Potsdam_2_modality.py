import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
from TRAIN_FreASam_for_2_modality import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import gc
import psutil
import os
import random

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def label2rgb(mask):
    """
    将 0..4 的类别映射为彩色，其余所有未知类别统一映射为红色 (255, 0, 0)。
    0: Impervious surfaces -> 白色
    1: Building            -> 红色
    2: Low vegetation      -> 黄色
    3: Tree                -> 绿色
    4: Car                 -> 青色
    其它标签(包括 5、255 等) -> 红色（方便发现异常区域）
    """
    # 保证是 2D
    mask = np.asarray(mask).astype(np.int32)
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    known = np.zeros((h, w), dtype=bool)

    idx = (mask == 0)
    mask_rgb[idx] = (255, 255, 255)   # Impervious
    known |= idx

    idx = (mask == 1)
    mask_rgb[idx] = (255,   0,   0)   # Building
    known |= idx

    idx = (mask == 2)
    mask_rgb[idx] = (255, 255,   0)   # Low vegetation
    known |= idx

    idx = (mask == 3)
    mask_rgb[idx] = (0, 255,   0)     # Tree
    known |= idx

    idx = (mask == 4)
    mask_rgb[idx] = (0, 255, 255)     # Car
    known |= idx

    idx = (mask == 5)
    mask_rgb[idx] = (0, 0, 255)  # Car
    known |= idx

    # 其它所有 label 统一画成红色，便于肉眼区分“异常/未用类别”
    mask_rgb[~known] = (255, 255, 255)

    return mask_rgb


def tensor_to_vis_image(t: torch.Tensor) -> np.ndarray:
    """
    将网络输入的 tensor[C,H,W] 转成 0~255 的 HWC uint8 图像，方便保存可视化。
    简单做 per-channel min-max 归一化。
    """
    if t is None:
        return None
    # [C,H,W] -> numpy
    arr = t.detach().cpu().numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expect 3D tensor [C,H,W], got shape={arr.shape}")
    c, h, w = arr.shape
    if c == 1:
        ch = arr[0]
        vmin, vmax = ch.min(), ch.max()
        if vmax <= vmin:
            out = np.zeros_like(ch, dtype=np.uint8)
        else:
            out = ((ch - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        img = np.stack([out, out, out], axis=-1)  # HWC
        return img
    else:
        # 取前 3 个通道
        c_use = min(3, c)
        arr = arr[:c_use]
        out_ch = []
        for i in range(c_use):
            ch = arr[i]
            vmin, vmax = ch.min(), ch.max()
            if vmax <= vmin:
                ch_out = np.zeros_like(ch, dtype=np.uint8)
            else:
                ch_out = ((ch - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            out_ch.append(ch_out)
        # 如果不足 3 通道，用最后一通道补齐
        while len(out_ch) < 3:
            out_ch.append(out_ch[-1])
        img = np.stack(out_ch, axis=-1)  # HWC
        return img


def img_writer(inp):
    """
    写出预测结果、GT 以及输入的两个模态 patch。
    inp: (
        mask, gt_mask,
        pred_base, gt_base,
        rgb_flag,
        img1_vis, img2_vis,
        img1_base, img2_base
    )
    """
    (mask, gt_mask,
     pred_base, gt_base,
     rgb,
     img1_vis, img2_vis,
     img1_base, img2_base) = inp
    try:
        # 预测 & GT
        if rgb:
            pred_rgb = label2rgb(mask)
            gt_rgb = label2rgb(gt_mask)

            pred_path = pred_base + '.png'
            gt_path = gt_base + '.png'

            cv2.imwrite(pred_path, pred_rgb)
            cv2.imwrite(gt_path, gt_rgb)
        else:
            mask_png = mask.astype(np.uint8)
            gt_png = gt_mask.astype(np.uint8)

            pred_path = pred_base + '.png'
            gt_path = gt_base + '.png'

            cv2.imwrite(pred_path, mask_png)
            cv2.imwrite(gt_path, gt_png)

        # 输入1(img) 保存
        if img1_vis is not None and img1_base is not None:
            in1_path = img1_base + '.png'
            # img1_vis 是 RGB，要先转成 BGR 再给 cv2
            img1_bgr = cv2.cvtColor(img1_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(in1_path, img1_bgr)

            # 输入2(aux2) 保存（DSM 一般是灰度，这里 tensor_to_vis_image 已经给了 3 通道灰度，可以同样处理）
        if img2_vis is not None and img2_base is not None:
            in2_path = img2_base + '.png'
            img2_bgr = cv2.cvtColor(img2_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(in2_path, img2_bgr)

    except Exception as e:
        print(f"Error writing image {pred_base}: {e}")


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
    arg("--gt_output_path", type=Path, help="Path where to save GT masks.", required=False)
    arg("--input1_output_path", type=Path, help="Path to save input1(img) patches.", required=False)
    arg("--input2_output_path", type=Path, help="Path to save input2(aux2) patches.", required=False)
    return parser.parse_args()


def check_memory_usage(threshold=85):
    """检查内存使用情况，超过阈值时进行清理"""
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > threshold:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Memory usage high: {memory_percent}%, cleaned cache")
        return True
    return False


def process_single_batch(batch_input, model, args, evaluator):
    """处理单个batch的数据，返回需要保存的结果"""
    with torch.no_grad():
        # 将数据移动到GPU
        img = batch_input['img'].cuda(non_blocking=True)
        aux2 = batch_input['aux2'].cuda(non_blocking=True) if 'aux2' in batch_input else None

        # 模型预测
        raw_predictions = model(img, aux2)

        image_ids = batch_input["img_id"]
        masks_true = batch_input['gt_semantic_seg']

        # 应用softmax并获取预测结果
        raw_predictions = nn.Softmax(dim=1)(raw_predictions)
        predictions = raw_predictions.argmax(dim=1)

        batch_results = []
        for i in range(raw_predictions.shape[0]):
            # 预测结果
            mask = predictions[i].cpu().numpy().astype(np.uint8)

            # GT
            gt_np = masks_true[i].cpu().numpy()  # 形状可能是 (1,H,W) 或 (H,W)
            gt_np = np.squeeze(gt_np)  # 保证是 (H,W)

            # 输入1 / 输入2 可视化
            img1_vis = tensor_to_vis_image(img[i])
            img2_vis = tensor_to_vis_image(aux2[i]) if aux2 is not None else None

            # 评估
            evaluator.add_batch(pre_image=mask, gt_image=gt_np)

            mask_name = image_ids[i]

            pred_base = str(args.output_path / mask_name)
            gt_base = str(args.gt_output_path / mask_name)
            img1_base = str(args.input1_output_path / mask_name) if args.input1_output_path is not None else None
            img2_base = str(args.input2_output_path / mask_name) if args.input2_output_path is not None else None

            batch_results.append(
                (mask, gt_np,
                 pred_base, gt_base,
                 args.rgb,
                 img1_vis, img2_vis,
                 img1_base, img2_base)
            )

        # 立即清理GPU上的张量
        del img, aux2, raw_predictions, predictions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return batch_results


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)

    # 预测输出目录
    args.output_path.mkdir(exist_ok=True, parents=True)

    # GT 输出目录
    if args.gt_output_path is None:
        args.gt_output_path = args.output_path / "GT"
    args.gt_output_path.mkdir(exist_ok=True, parents=True)

    # 输入1(img) 输出目录
    if args.input1_output_path is None:
        args.input1_output_path = args.output_path / "Input1"
    args.input1_output_path.mkdir(exist_ok=True, parents=True)

    # 输入2(aux2) 输出目录
    if args.input2_output_path is None:
        args.input2_output_path = args.output_path / "Input2"
    args.input2_output_path.mkdir(exist_ok=True, parents=True)

    ckpt_path = os.path.join(config.weights_path, config.test_weights_name + '.ckpt')

    # 使用 torch.load 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 初始化模型
    model = Supervision_Train(config=config)

    # 加载模型的 state_dict
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.cuda()
    model.eval()
    model.freeze()

    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    # TTA设置
    if args.tta == "lr":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip()
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[90]),
            tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5],
                      interpolation='bicubic', align_corners=False)
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.val_dataset

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    max_processes = min(4, mp.cpu_count())
    print(f"Using {max_processes} processes for image writing")

    try:
        with mpp.Pool(processes=max_processes) as pool:
            for batch_idx, batch_input in enumerate(tqdm(test_loader, desc="Processing batches")):
                if batch_idx % 10 == 0:
                    check_memory_usage(85)

                batch_results = process_single_batch(batch_input, model, args, evaluator)

                if batch_results:
                    pool.map(img_writer, batch_results)

                del batch_input, batch_results
                gc.collect()

        # 计算评估指标
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()

        pa_per_class = evaluator.Pixel_Accuracy_Class()
        oa_cls_binary = evaluator.OA_per_class()

        OA = evaluator.OA()
        mIoU_macro = evaluator.mIoU_macro()
        mF1_macro = evaluator.mF1_macro()
        mKappa = evaluator.mKappa()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        for cls_name, cls_iou, cls_f1, cls_pa, cls_oabin in zip(
                config.classes, iou_per_class, f1_per_class, pa_per_class, oa_cls_binary):
            print(f'F1_{cls_name}:{cls_f1:.6f}, IOU_{cls_name}:{cls_iou:.6f}, '
                  f'PA_{cls_name}:{cls_pa:.6f}, OAcls_{cls_name}:{cls_oabin:.6f}')

        print(f'mF1_macro:{mF1_macro:.6f}, mIoU_macro:{mIoU_macro:.6f}, '
              f'OA:{OA:.6f}, mKappa:{mKappa:.6f}, FWIoU:{FWIoU:.6f}')

    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Processing completed and memory cleaned.")


if __name__ == "__main__":
    main()
