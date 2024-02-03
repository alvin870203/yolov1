"""
Full definition of a YOLOv1 model, all of it in this single file.
Ref:
1) the official Darknet implementation:
https://github.com/pjreddie/darknet/blob/master/examples/yolo.c
https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg
"""

import math
import inspect
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
import random

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import box_iou, box_convert, clip_boxes_to_image, nms, batched_nms
from model.extraction import ExtractionBackbone, ExtractionConfig, ExtractionConv2d


@dataclass
class Yolov1Config:
    img_h: int = 448
    img_w: int = 448
    n_class: int = 20
    n_bbox_per_cell: int = 2  # B in the paper
    n_grid_h: int = 7  # S in the paper
    n_grid_w: int = 7  # S in the paper
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5
    prob_thresh: float = 0.001
    iou_thresh: float = 0.5
    iou_type: str = 'default'  # 'default' or 'distance'
    rescore: bool = False
    reduce_head_stride: bool = False  # only stride 1 in the head, as apposed to stride 2 in the paper


class Yolov1(nn.Module):
    def __init__(self, config: Yolov1Config) -> None:
        super().__init__()
        self.config = config
        assert config.img_h == 448 and config.img_w == 448, "Currently, Yolov1 only support 448x448 input images"

        self.backbone = ExtractionBackbone(ExtractionConfig())

        self.head = nn.Sequential(
            ExtractionConv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            ExtractionConv2d(1024, 1024, kernel_size=3, stride=2 if not self.config.reduce_head_stride else 1, padding=1),
            ExtractionConv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            ExtractionConv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear((1024 * 7 * 7) if not self.config.reduce_head_stride else (1024 * 14 * 14), 4096),  # hardcoded in_features is the reason for required 448x448 input image size
            nn.Dropout(0.5),
            nn.Linear(4096, config.n_grid_h * config.n_grid_w * (config.n_class + config.n_bbox_per_cell * 5))
        )


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


    def _init_weights(self, module):
        raise NotImplementedError("FUTURE: init weights for Yolov1 model")


    def _batched_box_iou(self, boxes1: Tensor, boxes2: Tensor) -> Tensor:
        """
        Return intersection-over-union (Jaccard index) between a batch of two sets of boxes.
        Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        Args:
            boxes1 (Tensor[..., N, 4]): batch of first set of boxes
            boxes2 (Tensor[..., M, 4]): batch of second set of boxes
        Returns:
            Tensor[..., N, M]: each NxM matrix containing the pairwise IoU values for every element in boxes1 & boxes2 pair
        """
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        lt = torch.max(boxes1[..., None, :2], boxes2[..., None, :, :2])
        rb = torch.min(boxes1[..., None, 2:], boxes2[..., None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area1[..., None] + area2[..., None, :] - inter
        iou = inter / union
        return iou


    def _batched_distance_box_iou(self, boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
        """
        Return distance intersection-over-union (Jaccard index) between a batch of two sets of boxes.
        Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        Args:
            boxes1 (Tensor[..., N, 4]): batch of first set of boxes
            boxes2 (Tensor[..., M, 4]): batch of second set of boxes
        Returns:
            Tensor[..., N, M]: each NxM matrix containing the pairwise IoU values for every element in boxes1 & boxes2 pair
        """
        iou = self._batched_box_iou(boxes1, boxes2)
        lti = torch.min(boxes1[..., None, :2], boxes2[..., None, :, :2])
        rbi = torch.max(boxes1[..., None, 2:], boxes2[..., None, :, 2:])
        whi = (rbi - lti).clamp(min=0)
        diagonal_distance_squared = (whi[..., 0] ** 2) + (whi[..., 1] ** 2) + eps
        # Centers of boxes
        cx_1 = (boxes1[..., 0] + boxes1[..., 2]) / 2
        cy_1 = (boxes1[..., 1] + boxes1[..., 3]) / 2
        cx_2 = (boxes2[..., 0] + boxes2[..., 2]) / 2
        cy_2 = (boxes2[..., 1] + boxes2[..., 3]) / 2
        # Distance between boxes' centers squared
        centers_distance_squared = ((cx_1[..., None] - cx_2[..., None, :]) ** 2) + ((cy_1[..., None] - cy_2[..., None, :]) ** 2)
        return iou - (centers_distance_squared / diagonal_distance_squared), iou


    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits (Tensor): size(N, n_grid_h, n_grid_w, (n_class + n_bbox_per_cell * 5))
            targets (Tensor): size(N, n_grid_h, n_grid_w, (1 + 1 + 4))
        Returns:
            loss (Tensor): size(,)
        """
        loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
        for logits_per_img, targets_per_img in zip(logits, targets):
            # Compute no-object confidence loss
            noobj_idx = (targets_per_img[:, :, 0] == 0.0)
            if noobj_idx.numel() != 0:
                noobj_bbox_logits = logits_per_img[:, :, self.config.n_class :][noobj_idx]
                noobj_conf_logits = noobj_bbox_logits.view(noobj_idx.sum() * self.config.n_bbox_per_cell, 5)[:, 0]
                loss += (self.config.lambda_noobj * F.mse_loss(noobj_conf_logits,
                                                               torch.zeros_like(noobj_conf_logits),
                                                               reduction='sum'))
            # Compute object classification loss
            obj_idx = (targets_per_img[:, :, 0] == 1.0)
            if obj_idx.numel() == 0:
                continue  # skip following computation if there is no object in this image
            obj_class_logits = logits_per_img[:, :, : self.config.n_class][obj_idx]
            obj_class_targets = targets_per_img[:, :, 1][obj_idx].to(torch.int64)
            obj_class_targets_one_hot = F.one_hot(obj_class_targets, num_classes=self.config.n_class)
            loss += F.mse_loss(obj_class_logits, obj_class_targets_one_hot.to(logits.dtype), reduction='sum')  # FUTURE: try BCELoss for improvement
            # Compute responsible grid cell
            obj_coord_targets = targets_per_img[:, :, 2 :][obj_idx]
            obj_bbox_logits = logits_per_img[:, :, self.config.n_class :][obj_idx]
            obj_conf_logits = obj_bbox_logits.view(obj_idx.sum(), self.config.n_bbox_per_cell, 5)[:, :, 0]
            obj_coord_logits = obj_bbox_logits.view(obj_idx.sum(), self.config.n_bbox_per_cell, 5)[:, :, 1:]
            obj_x1y1x2y2_logits = torch.stack([
                obj_coord_logits[:, :, 0] / self.config.n_grid_w - torch.pow(obj_coord_logits[:, :, 2], 2) / 2,
                obj_coord_logits[:, :, 1] / self.config.n_grid_h - torch.pow(obj_coord_logits[:, :, 3], 2) / 2,
                obj_coord_logits[:, :, 0] / self.config.n_grid_w + torch.pow(obj_coord_logits[:, :, 2], 2) / 2,
                obj_coord_logits[:, :, 1] / self.config.n_grid_h + torch.pow(obj_coord_logits[:, :, 3], 2) / 2], dim=-1)
            obj_x1y1x2y2_targets = torch.stack([
                obj_coord_targets[:, 0] / self.config.n_grid_w - torch.pow(obj_coord_targets[:, 2], 2) / 2,
                obj_coord_targets[:, 1] / self.config.n_grid_h - torch.pow(obj_coord_targets[:, 3], 2) / 2,
                obj_coord_targets[:, 0] / self.config.n_grid_w + torch.pow(obj_coord_targets[:, 2], 2) / 2,
                obj_coord_targets[:, 1] / self.config.n_grid_h + torch.pow(obj_coord_targets[:, 3], 2) / 2], dim=-1).unsqueeze(-2)
            if self.config.iou_type == 'default':
                match_iou_matrix = self._batched_box_iou(obj_x1y1x2y2_logits, obj_x1y1x2y2_targets)
            elif self.config.iou_type == 'distance':
                match_iou_matrix, default_iou_matrix = self._batched_distance_box_iou(obj_x1y1x2y2_logits, obj_x1y1x2y2_targets)
            sorted_iou, sorted_idx = match_iou_matrix.sort(dim=-2, descending=True)
            matched_idx = sorted_idx[:, 0]
            unmatched_idx = sorted_idx[:, 1:].squeeze(-1)
            # FUTURE: many ways to improve as shown in the https://github.com/pjreddie/darknet/blob/master/src/detection_layer.c#L50
            # Compute responsible object confidence loss
            matched_conf_logits = torch.take_along_dim(obj_conf_logits, matched_idx, dim=-1)
            if self.config.rescore:
                default_iou = torch.take_along_dim(default_iou_matrix, matched_idx, dim=-1)
                loss += F.mse_loss(matched_conf_logits, default_iou[:, 0], reduction='sum')
            else:
                loss += F.mse_loss(matched_conf_logits, torch.ones_like(matched_conf_logits), reduction='sum')
            # Compute no-responsible object confidence loss
            unmatched_conf_logits = torch.take_along_dim(obj_conf_logits, unmatched_idx, dim=-1)
            loss += F.mse_loss(unmatched_conf_logits, torch.zeros_like(unmatched_conf_logits), reduction='sum')
            # Compute responsible object bbox x,y,sqrt(w),sqrt(h) loss
            matched_coord_logits = torch.take_along_dim(obj_coord_logits, matched_idx.unsqueeze(-1), dim=-2).squeeze(-2)
            loss += self.config.lambda_coord * F.mse_loss(matched_coord_logits, obj_coord_targets, reduction='sum')

        return loss / logits.shape[0]


    def forward(self, imgs: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            imgs (Tensor): size(N, 3, img_h, img_w)
            targets (Tensor): size(N, n_grid_h, n_grid_w, (1 + 1 + 4))
                targets[i, j, k, 0] is the is_obj for the j,k-th grid cell, 1.0 if there is an object, 0.0 otherwise
                targets[i, j, k, 1] is the class index for the j,k-th grid cell, 0.0~float(n_class-1)
                targets[i, j, k, 2:6] is the bbox coordinates for the j,k-th grid cell, 0.0~1.0
                    targets[i, j, k, 2] is the x coordinate of the bbox center normalized by the "grid cell" width
                    targets[i, j, k, 3] is the y coordinate of the bbox center normalized by the "grid cell" height
                    targets[i, j, k, 4] is the sqrt(w) of the bbox normalized by the "img" width
                    targets[i, j, k, 5] is the sqrt(h) of the bbox normalized by the "img" height
                    (same for logits)
        Returns:
            logits (Tensor): size(N, n_grid_h, n_grid_w, (n_class + n_bbox_per_cell * 5))
                logits[i, j, k, 0:n_class] is the class logits for the j,k-th grid cell
                logits[i, j, k, n_class] is the objectness confidence score for the 1'st bbox in the j,k-th grid cell
                logits[i, j, k, n_class+1:n_class+5] is the bbox coordinates for the 1'st bbox in the j,k-th grid cell
                logits[i, j, k, n_class+5] is the objectness confidence score for the 2'nd bbox in the j,k-th grid cell
                logits[i, j, k, n_class+6:n_class+10] is the bbox coordinates for the 2'nd bbox in the j,k-th grid cell
                (and so on, where bbox coordinates have the same definition as targets)
            loss (Tensor): size(,)
        """
        device = imgs.device

        # Forward the Yolov1 model itself
        # N x 3 x 448 x 448
        x = self.backbone(imgs)
        # N x 1024 x 14 x 14
        logits = self.head(x)
        # N x 1470
        logits = logits.view(-1, self.config.n_grid_h, self.config.n_grid_w,
                             self.config.n_class + self.config.n_bbox_per_cell * 5)
        # N x 7 x 7 x 30

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            loss = self._compute_loss(logits, targets)
        else:
            loss = None

        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO
        raise NotImplementedError("FUTURE: init from pretrained model")


    def configure_optimizers(self, optimizer_type, weight_decay, learning_rate, betas, device_type, use_fused):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls decay, all biases and norms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        if optimizer_type == 'adamw':
            # Create AdamW optimizer and use the fused version if it is available
            if use_fused:
                fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
                use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")
        elif optimizer_type == 'sgd':
            # Create SGD optimizer
            optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=betas[0])
            print(f"using SGD")

        return optimizer


    def estimate_tops(self):
        """
        Estimate the number of TOPS and parameters in the model.
        """
        raise NotImplementedError("FUTURE: estimate TOPS for Yolov1 model")


    @torch.no_grad()
    def generate(self, imgs, top_k=None):
        """
        Predict on test imgs and return the top_k predictions.
        """
        # Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        self.eval()
        raise NotImplementedError("FUTURE: generate for Yolov1 model")
        self.train()


    @torch.inference_mode()
    def postprocess_for_map(self, logits, Y_supp):
        """
        Postprocess the logits and the targets for mAP calculation.
        """
        # Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        self.eval()
        device = logits.device
        preds_for_map, targets_for_map = [], []
        for logits_per_img, y_supp in zip(logits, Y_supp):
            grid_y, grid_x, idx_class, idx_bbox = torch.meshgrid(torch.arange(self.config.n_grid_h, device=device),
                                                                 torch.arange(self.config.n_grid_w, device=device),
                                                                 torch.arange(self.config.n_class, device=device),
                                                                 torch.arange(self.config.n_bbox_per_cell, device=device),
                                                                 indexing='ij')
            prob_class = logits_per_img[grid_y, grid_x, idx_class]
            conf = logits_per_img[grid_y, grid_x, self.config.n_class + idx_bbox * 5]
            prob = prob_class * conf
            mask = (prob > self.config.prob_thresh) & (conf > 0.0)
            grid_y = grid_y[mask]
            grid_x = grid_x[mask]
            idx_class = idx_class[mask]
            idx_bbox = idx_bbox[mask]
            prob = prob[mask]
            cx = logits_per_img[grid_y, grid_x, self.config.n_class + idx_bbox * 5 + 1]
            cy = logits_per_img[grid_y, grid_x, self.config.n_class + idx_bbox * 5 + 2]
            w = logits_per_img[grid_y, grid_x, self.config.n_class + idx_bbox * 5 + 3]
            h = logits_per_img[grid_y, grid_x, self.config.n_class + idx_bbox * 5 + 4]
            cx = (cx + grid_x) / self.config.n_grid_w * self.config.img_w
            cy = (cy + grid_y) / self.config.n_grid_h * self.config.img_h
            w = (w ** 2) * self.config.img_w
            h = (h ** 2) * self.config.img_h
            coord = torch.stack((cx, cy, w, h), dim=-1)

            boxes_for_nms = clip_boxes_to_image(box_convert(coord, in_fmt='cxcywh', out_fmt='xyxy'),
                                                size=(self.config.img_h, self.config.img_w))
            scores_for_nms = prob
            classes_for_nms = idx_class

            boxes_for_nms, scores_for_nms = boxes_for_nms.to(torch.float32), scores_for_nms.to(torch.float32)
            keep = batched_nms(boxes_for_nms, scores_for_nms, classes_for_nms, self.config.iou_thresh)  # don't work for BFloat16

            boxes_for_map = boxes_for_nms[keep]
            scores_for_map = scores_for_nms[keep]
            classes_for_map = classes_for_nms[keep]

            preds_for_map.append(dict(boxes=boxes_for_map, scores=scores_for_map, labels=classes_for_map))
            targets_for_map.append(dict(boxes=y_supp['boxes'].to(device), labels=y_supp['labels'].to(device)))
        return preds_for_map, targets_for_map


if __name__ == '__main__':
    # Test the model by `python -m model.yolov1` from the workspace directory
    config = Yolov1Config()
    model = Yolov1(config)
    print(model)
    print(f"num params: {model.get_num_params():,}")

    imgs = torch.randn(2, 3, config.img_h, config.img_w)
    targets = []
    for _ in range(2 * config.n_grid_w * config.n_grid_h):
        targets.extend([random.randint(0, 1), random.randint(0, config.n_class - 1),
                        random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
    targets = torch.tensor(targets, dtype=torch.float32).view(2, config.n_grid_h, config.n_grid_w, 1 + 1 + 4)
    logits, loss = model(imgs, targets)
    print(f"logits shape: {logits.shape}")
    if loss is not None:
        print(f"loss shape: {loss.shape}")
