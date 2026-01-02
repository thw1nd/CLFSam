import numpy as np

class Evaluator(object):
    def __init__(self, num_class: int, eps: float = 1e-8):
        self.num_class = int(num_class)
        self.confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int64)
        self.eps = float(eps)

    # ---------- 核心：累计混淆矩阵 ----------
    def _generate_matrix(self, gt_image, pre_image):
        # 忽略背景：要求调用方已把背景设为 255；这里仅统计 [0, C-1]
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask].astype('int64')
        count = np.bincount(label, minlength=self.num_class ** 2)
        return count.reshape(self.num_class, self.num_class)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, f'pre_image shape {pre_image.shape}, gt_image shape {gt_image.shape}'
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int64)

    # ---------- 基本量 ----------
    def get_tp_fp_tn_fn(self):
        cm = self.confusion_matrix.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)  # 正确 TN 公式
        return tp, fp, tn, fn

    # ---------- 每类指标 ----------
    def Precision(self):
        tp, fp, *_ = self.get_tp_fp_tn_fn()
        return tp / np.clip(tp + fp, self.eps, None)

    def Recall(self):
        tp, *_ , fn = self.get_tp_fp_tn_fn()
        return tp / np.clip(tp + fn, self.eps, None)

    def F1(self):  # per-class F1
        p = self.Precision()
        r = self.Recall()
        return (2.0 * p * r) / np.clip(p + r, self.eps, None)

    def Intersection_over_Union(self):  # per-class IoU
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        return tp / np.clip(tp + fp + fn, self.eps, None)

    def Dice(self):  # per-class Dice = F1
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        return 2 * tp / np.clip((tp + fp) + (tp + fn), self.eps, None)

    def Pixel_Accuracy_Class(self):  # per-class Accuracy (PA) = TP / (TP + FN)
        cm = self.confusion_matrix.astype(np.float64)
        return np.diag(cm) / np.clip(cm.sum(axis=1), self.eps, None)

    # ---------- 宏/微/加权 ----------
    def mIoU_macro(self):
        return float(np.nanmean(self.Intersection_over_Union()))

    def mF1_macro(self):
        return float(np.nanmean(self.F1()))

    def OA(self):
        cm = self.confusion_matrix.astype(np.float64)
        return float(np.diag(cm).sum() / np.clip(cm.sum(), self.eps, None))

    def Frequency_Weighted_Intersection_over_Union(self):
        cm = self.confusion_matrix.astype(np.float64)
        freq = cm.sum(axis=1) / np.clip(cm.sum(), self.eps, None)
        iou = self.Intersection_over_Union()
        return float((freq[freq > 0] * iou[freq > 0]).sum())

    # ---------- mKappa（Cohen's Kappa，多分类） ----------
    def mKappa(self):
        cm = self.confusion_matrix.astype(np.float64)
        total = cm.sum()
        if total < self.eps:
            return 0.0
        po = np.diag(cm).sum() / total  # 观测一致率
        pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total * total)  # 期望一致率
        denom = max(1.0 - pe, self.eps)
        return float((po - pe) / denom)

    def OA_per_class(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        total = np.clip(tp + fp + tn + fn, self.eps, None)
        return (tp + tn) / total

    # ---------- 友好别名 ----------
    def IoU_per_class(self):  return self.Intersection_over_Union()
    def F1_per_class(self):   return self.F1()
    def Kappa(self):          return self.mKappa()

    def Accuracy_per_class(self):  # 等价于 Pixel_Accuracy_Class（PA）
        return self.Pixel_Accuracy_Class()
