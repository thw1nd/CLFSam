import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
from TRAIN_for_2_modality import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import gc
import psutil

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
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    # B, G, R
    mask_rgb[np.all(mask_convert == 0, axis=0)] = (17, 91, 197)  # cropland
    mask_rgb[np.all(mask_convert == 1, axis=0)] = (88,   130,   52)  # forest
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [62, 205, 178]  # grassland
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [97, 84, 230]  # wetland
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [216, 155, 91]  # water
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [112, 174, 228]  # unused land
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [0, 160, 238]  # built-up area

    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    try:
        if rgb:
            mask_name_tif = mask_id + '.png'
            mask_tif = label2rgb(mask)
            cv2.imwrite(mask_name_tif, mask_tif)
        else:
            mask_png = mask.astype(np.uint8)
            mask_name_png = mask_id + '.png'
            cv2.imwrite(mask_name_png, mask_png)
    except Exception as e:
        print(f"Error writing image {mask_id}: {e}")


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
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
        aux1 = batch_input['aux1'].cuda(non_blocking=True) if 'aux1' in batch_input else None
        # aux2 = batch_input['aux2'].cuda(non_blocking=True) if 'aux2' in batch_input else None

        # 模型预测
        raw_predictions = model(img, aux1)

        image_ids = batch_input["img_id"]
        masks_true = batch_input['gt_semantic_seg']

        # 应用softmax并获取预测结果
        raw_predictions = nn.Softmax(dim=1)(raw_predictions)
        predictions = raw_predictions.argmax(dim=1)

        batch_results = []
        for i in range(raw_predictions.shape[0]):
            mask = predictions[i].cpu().numpy()
            evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
            mask_name = image_ids[i]
            batch_results.append((mask, str(args.output_path / mask_name), args.rgb))

        # 立即清理GPU上的张量
        del img, aux1, raw_predictions, predictions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return batch_results


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    ckpt_path = os.path.join(config.weights_path, config.test_weights_name + '.ckpt')

    # 使用 torch.load 加载 checkpoint，确保 weights_only=False
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 初始化模型
    model = Supervision_Train(config=config)

    # 加载模型的 state_dict
    model.load_state_dict(checkpoint['state_dict'], strict=False)  # strict=False 允许加载时忽略某些不匹配的层

    model.cuda()
    model.eval()

    # 设置模型为评估模式
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
            tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.val_set

    # 创建数据加载器，使用较小的batch_size
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 减小batch_size避免内存压力
        num_workers=2,  # 减少worker数量
        pin_memory=True,
        drop_last=False,
    )

    # 使用进程池，但限制进程数量
    max_processes = min(4, mp.cpu_count())  # 限制最大进程数
    print(f"Using {max_processes} processes for image writing")

    try:
        with mpp.Pool(processes=max_processes) as pool:
            for batch_idx, batch_input in enumerate(tqdm(test_loader, desc="Processing batches")):
                # 检查内存使用情况
                if batch_idx % 10 == 0:  # 每10个batch检查一次
                    check_memory_usage(85)

                # 处理当前batch
                batch_results = process_single_batch(batch_input, model, args, evaluator)

                # 立即写入当前batch的结果
                if batch_results:
                    pool.map(img_writer, batch_results)

                # 清理当前batch的相关变量
                del batch_input, batch_results
                gc.collect()

        # 计算评估指标
        # 计算评估指标（基于整套测试集的混淆矩阵）
        iou_per_class = evaluator.Intersection_over_Union()  # (C,)
        f1_per_class = evaluator.F1()

        # NEW: 每类 PA（行归一：TP/(TP+FN)）与 每类 OA_bin（二分类口径）
        pa_per_class = evaluator.Pixel_Accuracy_Class()
        oa_cls_binary = evaluator.OA_per_class()

        OA = evaluator.OA()
        mIoU_macro = evaluator.mIoU_macro()
        mF1_macro = evaluator.mF1_macro()
        mKappa = evaluator.mKappa()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        # 输出逐类与汇总
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
        # 最终清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Processing completed and memory cleaned.")


if __name__ == "__main__":
    main()