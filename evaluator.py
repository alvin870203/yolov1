import numpy as np
import torch
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision

class DetEvaluator():
    def __init__(self,
                 metric_iou_thresh=torch.linspace(0.5, 0.95, 10),  # iou vector of mAP@0.5:0.95
                 eps=1e-16,
                 smooth_f=0.1):
        self.metric_iou_thresh = metric_iou_thresh
        self.n_iou = len(metric_iou_thresh)
        self.eps = eps
        self.smooth_f = smooth_f
        self.stats = dict(tp=[], score=[], pred_cls=[], target_cls=[])


    @torch.inference_mode()
    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            n_pred, n_target = len(pred['boxes']), len(target['boxes'])
            device = pred['boxes'].device
            stat = dict(
                score=torch.zeros(0, device=device),
                pred_cls=torch.zeros(0, device=device),
                tp=torch.zeros(n_pred, self.n_iou, dtype=torch.bool, device=device)
            )
            stat['target_cls'] = target['labels']
            if n_pred == 0:
                if n_target > 0:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                continue
            stat['score'] = pred['scores']
            stat['pred_cls'] = pred['labels']
            if n_target > 0:
                iou = box_iou(target['boxes'], pred['boxes'])  # shape(n_target, n_pred)
                # Match predictions to targets
                correct = np.zeros((n_pred, self.n_iou)).astype(bool)  # shape(n_pred, n_iou)
                correct_class = target['labels'][:, None] == pred['labels']   # shape(n_target, n_pred)
                iou = iou * correct_class  # zero out the wrong class
                iou = iou.cpu().numpy()
                for idx_thresh, thresh in enumerate(self.metric_iou_thresh.cpu().tolist()):
                    matches = np.nonzero(iou >= thresh)
                    matches = np.array(matches).T  # shape(n_matches, 2), where each row is (idx_target, idx_pred)
                    if matches.shape[0] > 0:
                        if matches.shape[0] > 1:
                            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]  # sort by descending iou
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # one-target to multi-pred
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # one-target to one-pred
                        correct[matches[:, 1].astype(int), idx_thresh] = True
                stat['tp'] = torch.tensor(correct, dtype=torch.bool, device=device)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])


    @torch.inference_mode()
    def compute(self):
        """
        Compute metrics for object detection. Simpler and faster than COCOeval in torchmetrics.
        """
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        unique_classes = []  # (np.ndarray, list): Array of shape (nc,) with unique class indices, or an empty list if not available.
        p = []  # (np.ndarray, list): Array of shape (nc,) with precision values per class, or an empty list if not available.
        r = []  # (np.ndarray, list): Array of shape (nc,) with recall values per class, or an empty list if not available.
        f1 = []  # (np.ndarray, list): Array of shape (nc,) with F1 values per class, or an empty list if not available.
        all_ap = []  #  (np.ndarray, list): Array of shape (nc, 10) with AP per class and per IoU threshold, or an empty list if not available.
        curves = []  # (list): List of curves, each represented as a list [x, y, xlabel, ylabel], or an empty list if not available.
        if len(stats) and stats['tp'].any():
            idx_desc_score = np.argsort(-stats["score"])
            tp, score, pred_cls = stats['tp'][idx_desc_score], stats['score'][idx_desc_score], stats['pred_cls'][idx_desc_score]
            unique_classes, n_gt_per_cls = np.unique(stats['target_cls'], return_counts=True)
            n_unique_classes = unique_classes.shape[0]
            # Create Precision-Recall curve and compute AP for each class
            x_curve = np.linspace(0, 1, 1000)
            # Average precision, precision and recall curves
            all_ap = np.zeros((n_unique_classes, self.n_iou))  # @0.5:0.95
            p_curve = np.zeros((n_unique_classes, 1000))  # @0.5
            r_curve = np.zeros((n_unique_classes, 1000))  # @0.5
            pr_curve = np.zeros((n_unique_classes, 1000))  # @0.5
            for idx_cls, cls in enumerate(unique_classes):
                idx = pred_cls == cls
                n_gt = n_gt_per_cls[idx_cls]
                n_pred = idx.sum()
                if n_pred == 0 or n_gt == 0:
                    continue
                # Accumulate FPs and TPs
                fp_cum = (1 - tp[idx]).cumsum(0)
                tp_cum = tp[idx].cumsum(0)
                # Recall
                recall = tp_cum / (n_gt + self.eps)  # 0~max_r, @0.5:0.95
                r_curve[idx_cls] = np.interp(-x_curve, -score[idx], recall[:, 0], left=0)  # negative x, xp because xp decreases. X: 0~-1, Y: max_r~0
                # Precision
                precision = tp_cum / (tp_cum + fp_cum)  # 1~min_p, @0.5:0.95
                p_curve[idx_cls] = np.interp(-x_curve, -score[idx], precision[:, 0], left=1)  # X: 0~-1, Y: min_p~1
                # AP from recall-precision curve
                for idx_thresh in range(self.n_iou):  # every iou threshold
                    # Append sentinel values to beginning and end
                    mrec = np.concatenate(([0.0], recall[:, idx_thresh], [1.0]))
                    mpre = np.concatenate(([1.0], precision[:, idx_thresh], [0.0]))
                    # Compute the precision envelope
                    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
                    # Integrate area under curve
                    x_auc = np.linspace(0, 1, 101)  # 101-point interp (COCO)
                    all_ap[idx_cls, idx_thresh] = np.trapz(np.interp(x_auc, mrec, mpre), x_auc)  # integrate
                    if idx_thresh == 0:
                        pr_curve[idx_cls] = np.interp(x_curve, mrec, mpre)  # X: 0~1, Y: 1~0
            # Compute F1 score (harmonic mean of precision and recall)
            f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + self.eps)  # @0.5, X: 0~-1, Y: right_f1~left_f1, shape(n_unique_classes, 1000)
            curves = [
                [x_curve, pr_curve, 'Recall', 'Precision'],
                [x_curve, f1_curve, 'Score', 'F1'],
                [x_curve, p_curve, 'Score', 'Precision'],
                [x_curve, r_curve, 'Score', 'Recall'],
            ]
            # Box filter of fraction smooth_f.
            f1_curve_mean = f1_curve.mean(0)  # mean over all classes, shape(1000,)
            n_f = round(len(f1_curve_mean) * self.smooth_f * 2) // 2 + 1  # number of filter elements (must be odd)
            pad = np.ones(n_f // 2)  # ones padding
            f1_curve_padded = np.concatenate((pad * f1_curve_mean[0], f1_curve_mean, pad * f1_curve_mean[-1]), 0)
            f1_curve_smoothed = np.convolve(f1_curve_padded, np.ones(n_f) / n_f, mode='valid')
            idx_max_f1 = f1_curve_smoothed.argmax()
            p, r, f1 = p_curve[:, idx_max_f1], r_curve[:, idx_max_f1], f1_curve[:, idx_max_f1]  # max-F1 precision, recall, F1 values
            tp = (r * n_gt_per_cls).round()  # max-F1 true positives
            fp = (tp / (p + self.eps) - tp).round()  # max-F1 false positives

            # print(f"{tp=}\n{fp=}\n{p=}\n{r=}\n{f1=}\n{all_ap=}\n{unique_classes=}\n")
            # print(f"{p_curve.shape=}\n{r_curve.shape=}\n{f1_curve.shape=}\n{x_curve.shape=}\n{pr_curve.shape=}")

        # (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        ap_50 = all_ap[:, 0] if len(all_ap) > 0 else []
        # (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        ap = all_ap.mean(1) if len(all_ap) > 0 else []
        # (float): The mean precision of all classes at max F1.
        mp = p.mean() if len(p) > 0 else 0.0
        # (float): The mean recall of all classes at max F1.
        mr = r.mean() if len(r) > 0 else 0.0
        # (float): The mAP at an IoU threshold of 0.5.
        map_50 = all_ap[:, 0].mean() if len(all_ap) > 0 else 0.0
        # (float): The mAP at an IoU threshold of 0.75.
        map_75 = all_ap[:, 5].mean() if len(all_ap) > 0 else 0.0
        # (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        map = all_ap.mean() if len(all_ap) > 0 else 0.0

        return dict(mp=mp, mr=mr, map_50=map_50, map_75=map_75, map=map,
                    unique_classes=unique_classes, p=p, r=r, f1=f1, ap_50=ap_50, ap=ap,
                    curves=curves)
