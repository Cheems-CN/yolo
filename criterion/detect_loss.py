from __future__ import annotations

import math
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from components.modules.head import dist2bbox, make_anchors


# --------------------------
# Basic bbox utilities (xyxy) + CIoU
# --------------------------
def _xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    xy = xywh[..., :2]
    wh = xywh[..., 2:].clamp_(min=0)
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    return torch.cat([xy1, xy2], dim=-1)


def bbox_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, ciou: bool = True, eps: float = 1e-7) -> torch.Tensor:
    """
    IoU/CIoU between boxes in xyxy format.
    Shapes:
      - box1: (*, 4)
      - box2: (*, 4)
    Returns:
      IoU (if ciou=False) or CIoU (if ciou=True) as (*, 1)
    """
    # Intersection
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    # Areas
    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter + eps
    iou = (inter + eps) / union

    if not ciou:
        return iou.unsqueeze(-1)

    # CIoU terms (https://arxiv.org/abs/1911.08287)
    # center distance
    c1x = (box1[..., 0] + box1[..., 2]) * 0.5
    c1y = (box1[..., 1] + box1[..., 3]) * 0.5
    c2x = (box2[..., 0] + box2[..., 2]) * 0.5
    c2y = (box2[..., 1] + box2[..., 3]) * 0.5
    rho2 = (c2x - c1x) ** 2 + (c2y - c1y) ** 2

    # enclosing diagonal
    x_c1 = torch.min(box1[..., 0], box2[..., 0])
    y_c1 = torch.min(box1[..., 1], box2[..., 1])
    x_c2 = torch.max(box1[..., 2], box2[..., 2])
    y_c2 = torch.max(box1[..., 3], box2[..., 3])
    c2 = ((x_c2 - x_c1) ** 2 + (y_c2 - y_c1) ** 2) + eps

    # aspect ratio term
    w1 = (box1[..., 2] - box1[..., 0]).clamp(min=eps)
    h1 = (box1[..., 3] - box1[..., 1]).clamp(min=eps)
    w2 = (box2[..., 2] - box2[..., 0]).clamp(min=eps)
    h2 = (box2[..., 3] - box2[..., 1]).clamp(min=eps)
    v = (4 / (math.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou_term = (rho2 / c2) + alpha * v
    ciou = iou - ciou_term
    return ciou.unsqueeze(-1)


# --------------------------
# DFL (Distribution Focal Loss) for bounding box distances
# --------------------------
class DFLoss(nn.Module):
    """Distribution Focal Loss used in YOLOv8 bbox regression bins."""
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_dist: (N, reg_max) raw logits per side
            target: (N,) float target in [0, reg_max - 1]
        Returns: (N, 1) averaged DFL value
        """
        target = target.clamp_(0, self.reg_max - 1 - 1e-4)
        tl = target.long()
        tr = (tl + 1).clamp(max=self.reg_max - 1)
        wl = tr - target
        wr = 1 - wl
        loss = (F.cross_entropy(pred_dist, tl, reduction="none") * wl +
                F.cross_entropy(pred_dist, tr, reduction="none") * wr)
        return loss.mean(-1, keepdim=True)


# --------------------------
# Helper to convert xyxy target boxes to distances (ltrb) from anchor point
# --------------------------
def bbox2dist(anchor_points: torch.Tensor, boxes_xyxy: torch.Tensor, reg_max: int) -> torch.Tensor:
    """
    Compute target l, t, r, b distances from anchor points.
    Args:
        anchor_points: (A, 2) [x, y] anchor centers in grid coords
        boxes_xyxy: (B, A, 4) target boxes aligned with anchors, in xyxy grid coords
        reg_max: number of bins
    Returns:
        (B, A, 4) continuous distances in [0, reg_max)
    """
    # distances in grid units
    left   = (anchor_points[:, 0][None, :] - boxes_xyxy[..., 0]).clamp(min=0)
    top    = (anchor_points[:, 1][None, :] - boxes_xyxy[..., 1]).clamp(min=0)
    right  = (boxes_xyxy[..., 2] - anchor_points[:, 0][None, :]).clamp(min=0)
    bottom = (boxes_xyxy[..., 3] - anchor_points[:, 1][None, :]).clamp(min=0)
    ltrb = torch.stack([left, top, right, bottom], dim=-1)
    # clamp to valid range for DFL bins
    return ltrb.clamp(max=reg_max - 1 - 1e-4)


# --------------------------
# BboxLoss = CIoU + DFL (as in Ultralytics)
# --------------------------
class BboxLoss(nn.Module):
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,      # (B, A, 4*reg_max) raw logits
        pred_bboxes: torch.Tensor,    # (B, A, 4) xyxy in grid units
        anchor_points: torch.Tensor,  # (A, 2) grid coords
        target_bboxes: torch.Tensor,  # (B, A, 4) xyxy in grid units
        target_scores: torch.Tensor,  # (B, A, C) one-hot weighted scores
        target_scores_sum: torch.Tensor,  # scalar or (B,1) sum of target_scores
        fg_mask: torch.Tensor,        # (B, A) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # (N_fg, 1)

        # CIoU loss on xyxy (grid units)
        iou = bbox_iou_xyxy(pred_bboxes[fg_mask], target_bboxes[fg_mask], ciou=True)  # (N_fg, 1)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL on ltrb bins
        if self.dfl_loss:
            # continuous target ltrb in [0, reg_max)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.reg_max),  # (N_fg*4, reg_max)
                target_ltrb[fg_mask].view(-1)               # (N_fg*4,)
            ) * weight.repeat_interleave(4, dim=0)
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = pred_bboxes.sum() * 0.0

        return loss_iou, loss_dfl


# --------------------------
# Task-Aligned Assigner (simplified HBB version, aligned with Ultralytics)
# --------------------------

class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk: int, num_classes: int, alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        # runtime attributes
        self.bs = 0
        self.n_max_boxes = 0

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,    # (B, A, C), sigmoid-ed probabilities
        pd_bboxes: torch.Tensor,    # (B, A, 4), xyxy in pixels
        anc_points: torch.Tensor,   # (A, 2), in pixels
        gt_labels: torch.Tensor,    # (B, M, 1)
        gt_bboxes: torch.Tensor,    # (B, M, 4), xyxy in pixels
        mask_gt: torch.Tensor,      # (B, M, 1) bool
    ):
        device = pd_scores.device
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0 or mask_gt.sum() == 0:
            empty = (torch.zeros(self.bs, pd_scores.shape[1], device=device, dtype=torch.long),
                     torch.zeros(self.bs, pd_scores.shape[1], 4, device=device),
                     torch.zeros(self.bs, pd_scores.shape[1], self.num_classes, device=device),
                     torch.zeros(self.bs, pd_scores.shape[1], device=device, dtype=torch.bool),
                     torch.zeros(self.bs, pd_scores.shape[1], device=device, dtype=torch.long))
            return empty

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target (one-hot initial scores)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize and weight scores with alignment metric (Ultralytics behavior)
        align_metric = align_metric * mask_pos  # (B, M, A)
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)         # (B, M, 1)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)     # (B, M, 1)
        # Avoid division by zero; broadcast over anchors
        norm = (pos_align_metrics + self.eps)
        norm_align_metric = (align_metric * pos_overlaps / norm).amax(-2).unsqueeze(-1)  # (B, A, 1)
        target_scores = target_scores * norm_align_metric  # (B, A, C)

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Build per-anchor targets like Ultralytics."""
        # Flatten batch-wise gt indexing
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        mapped = target_gt_idx + batch_ind * self.n_max_boxes  # (B, A)
        # Labels: (B, A)
        flat_labels = gt_labels.long().flatten()
        target_labels = flat_labels[mapped]  # (B, A)
        target_labels.clamp_(0)

        # Boxes: (B, A, 4)
        flat_boxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])
        target_bboxes = flat_boxes[mapped]

        # One-hot scores initialized to 0/1 for fg anchors
        target_scores = torch.zeros((self.bs, target_labels.shape[1], self.num_classes),
                                    dtype=torch.float32, device=gt_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1.0)
        # Mask out background anchors
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask, target_scores, torch.zeros_like(target_scores))
        return target_labels, target_bboxes, target_scores

    def get_pos_mask(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        mask_gt: torch.Tensor,
    ):
        B, A, C = pd_scores.shape
        M = gt_labels.shape[1]
        device = pd_scores.device

        # classification prob for each gt class per anchor: (B, M, A)
        bbox_scores = torch.zeros(B, M, A, device=device)
        b_idx = torch.arange(B, device=device)[:, None].expand(B, M)
        c_idx = gt_labels.squeeze(-1).long()
        bbox_scores[mask_gt.squeeze(-1)] = pd_scores[b_idx, :, c_idx][mask_gt.squeeze(-1)]

        # IoU overlaps on pixels: (B, M, A)
        overlaps = torch.zeros(B, M, A, device=device)
        for b in range(B):
            if mask_gt[b].any():
                pb = pd_bboxes[b].unsqueeze(0).expand(M, -1, -1)   # (M, A, 4)
                gb = gt_bboxes[b].unsqueeze(1).expand(-1, A, -1)   # (M, A, 4)
                overlaps[b, mask_gt[b].squeeze(-1)] = bbox_iou_xyxy(pb[mask_gt[b].squeeze(-1)], gb[mask_gt[b].squeeze(-1)], ciou=True).squeeze(-1)

        # alignment metric
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        # candidate mask: centers within gt box
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes) & mask_gt
        # select top-k anchors per-gt
        topk_mask = self.select_topk_candidates(align_metric)
        mask_pos = topk_mask & mask_in_gts
        return mask_pos, align_metric, overlaps

    @staticmethod
    def select_candidates_in_gts(xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9):
        # xy_centers: (A, 2); gt_bboxes: (B, M, 4) in xyxy pixels
        B, M, _ = gt_bboxes.shape
        A = xy_centers.shape[0]
        lt = xy_centers[None, None, :, :] - gt_bboxes[:, :, None, :2]   # (B,M,A,2)
        rb = gt_bboxes[:, :, None, 2:] - xy_centers[None, None, :, :]   # (B,M,A,2)
        deltas = torch.cat([lt, rb], dim=-1)  # (B,M,A,4)
        return deltas.amin(-1).gt_(eps)       # (B,M,A) bool

    def select_topk_candidates(self, metrics: torch.Tensor):
        # metrics: (B, M, A)
        B, M, A = metrics.shape
        topk = min(self.topk, A)
        topk_scores, topk_idx = torch.topk(metrics, topk, dim=-1)  # (B,M,topk)
        mask = torch.zeros_like(metrics, dtype=torch.bool)
        ar = torch.arange(B, device=metrics.device)[:, None, None]
        mr = torch.arange(M, device=metrics.device)[None, :, None]
        mask[ar, mr, topk_idx] = True
        return mask

    @staticmethod
    def select_highest_overlaps(mask_pos: torch.Tensor, overlaps: torch.Tensor, n_max_boxes: int):
        # Resolve multiple gts assigned to same anchor: keep the gt with max IoU
        B, M, A = mask_pos.shape
        fg_mask = mask_pos.sum(dim=1)  # (B, A)
        if fg_mask.max() > 1:
            max_overlaps_idx = overlaps.argmax(dim=1)  # (B, A)
            is_max = torch.zeros_like(mask_pos, dtype=mask_pos.dtype)
            br = torch.arange(B, device=mask_pos.device)[:, None]
            ar = torch.arange(A, device=mask_pos.device)[None, :]
            is_max[br, max_overlaps_idx, ar] = 1
            mask_pos = is_max
            fg_mask = mask_pos.sum(dim=1)
        target_gt_idx = mask_pos.argmax(dim=1)
        return target_gt_idx, fg_mask, mask_pos


class YoloV11DetectionLoss:
    def __init__(self, model, tal_topk: int = 10):
        """
        Args:
            model: your YOLOv11 nn.Module instance, used to read nc, reg_max.
            tal_topk: top-k for TaskAlignedAssigner (default 10, or 1 for one-to-one branch).
        """
        device = next(model.parameters()).device
        m = model.detect  # Detect head
        self.device = device
        self.nc = m.nc
        self.reg_max = m.reg_max
        self.no = self.nc + self.reg_max * 4
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max).to(device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        # Loss gains (tunable)
        self.hyp = type("H", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5})()  # sensible defaults

    @staticmethod
    def _preprocess_targets(targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor, device: torch.device):
        """
        Convert [batch_idx, cls, xywh_norm] to per-image tensors and scale xywh to xyxy pixels.
        Returns: (B, n_max, 1), (B, n_max, 4), mask_gt (B, n_max, 1)
        """
        nl = targets.shape[0]
        if nl == 0:
            gt_labels = torch.zeros(batch_size, 0, 1, device=device)
            gt_bboxes = torch.zeros(batch_size, 0, 4, device=device)
            mask_gt = torch.zeros(batch_size, 0, 1, dtype=torch.bool, device=device)
            return gt_labels, gt_bboxes, mask_gt

        i = targets[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        counts = counts.to(torch.int32)
        maxn = int(counts.max())
        out = torch.zeros(batch_size, maxn, 5, device=device)
        for j in range(batch_size):
            matches = i == j
            if matches.any():
                out[j, :matches.sum()] = targets[matches, 1:6]  # [cls, x, y, w, h]
        # scale to pixels and convert to xyxy
        bxywh = out[..., 1:5]
        bxywh[..., [0, 2]] *= scale_tensor[0]  # x,w scaled by width
        bxywh[..., [1, 3]] *= scale_tensor[1]  # y,h scaled by height
        bxyxy = _xywh_to_xyxy(bxywh)
        gt_labels = out[..., :1]
        gt_bboxes = bxyxy
        mask_gt = gt_bboxes.sum(-1, keepdim=True).gt(0)
        return gt_labels, gt_bboxes, mask_gt

    def _decode_boxes(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        # pred_dist: (B, A, 4*reg_max) raw logits -> softmax over reg_max bins, expectation -> distances
        B, A, C = pred_dist.shape
        if self.reg_max > 1:
            pred = pred_dist.view(B, A, 4, self.reg_max).softmax(dim=-1).matmul(self.proj.type(pred_dist.dtype).view(1, 1, 1, -1))
            pred = pred.squeeze(-2)  # (B, A, 4)
        else:
            pred = pred_dist.view(B, A, 4)
        return dist2bbox(pred, anchor_points, xywh=False)  # xyxy in grid units

    def __call__(self, feats: Any, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute [box, cls, dfl] losses (summed) like Ultralytics v8.
        Args:
            feats: list of 3 tensors [B, no, H, W] OR dict {'one2many': [...], 'one2one': [...]}
            batch: {'imgs': BxCxHxW, 'batch_idx': (N_gt,), 'cls': (N_gt,1), 'bboxes': (N_gt,4) [xywh_norm]}
        Returns:
            total_loss (scalar tensor), detached per-term tensor (box, cls, dfl)
        """
        if isinstance(feats, dict):  # E2E: sum one-to-many and one-to-one
            l_many, v_many = self._loss_single(feats["one2many"], batch)
            l_one, v_one = self._loss_single(feats["one2one"], batch, tal_topk_override=1)
            return l_many + l_one, v_many + v_one
        else:
            return self._loss_single(feats, batch)

    def _loss_single(self, feats: list[torch.Tensor], batch: dict, tal_topk_override: int | None = None):
        device = self.device
        B = feats[0].shape[0]
        # Build stride list from image size and feature map sizes
        img_h, img_w = batch["imgs"].shape[-2:]
        strides = []
        for xi in feats:
            h, w = xi.shape[-2:]
            # assume integer strides
            stride_h = img_h // h
            stride_w = img_w // w
            assert stride_h == stride_w, "Non-square strides are not supported in this minimal loss."
            strides.append(stride_h)
        strides = torch.tensor(strides, device=device, dtype=torch.float)

        # Flatten predictions: (B, no, A) -> split to dist/scores then (B, A, *)
        no = self.no
        pred_cat = torch.cat([xi.view(B, no, -1) for xi in feats], dim=2)
        pred_dist, pred_scores = pred_cat.split((self.reg_max * 4, self.nc), dim=1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (B, A, C)
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()      # (B, A, 4*reg_max)

        # Anchors and stride tensors in grid units & pixels
        anchor_points, stride_tensor = make_anchors(feats, strides, 0.5)  # (A,2), (A,1)
        anchor_points = anchor_points.to(device=device, dtype=pred_scores.dtype)
        stride_tensor = stride_tensor.to(device=device, dtype=pred_scores.dtype)

        # Targets
        # Build targets tensor [batch_idx, cls, x, y, w, h] normalized in [0,1]
        targets_merged = torch.cat([batch["batch_idx"].view(-1, 1),
                                    batch["cls"].view(-1, 1),
                                    batch["bboxes"].view(-1, 4)], dim=1).to(device)
        gt_labels, gt_bboxes, mask_gt = self._preprocess_targets(
            targets_merged, B, scale_tensor=torch.tensor([img_w, img_h], device=device, dtype=pred_scores.dtype), device=device
        )

        # Predicted boxes in grid units -> xyxy
        pred_bboxes = self._decode_boxes(anchor_points, pred_dist)  # (B, A, 4)

        # Assignment (use pixels)
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), torch.tensor(1.0, device=device))

        # Classification loss (BCE)
        cls_loss = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() / target_scores_sum

        # Bbox + DFL
        box_loss = pred_scores.sum() * 0.0
        dfl_loss = pred_scores.sum() * 0.0
        if fg_mask.sum() > 0:
            box_loss, dfl_loss = self.bbox_loss(
                pred_dist,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,  # back to grid units
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        box_loss = box_loss * self.hyp.box
        cls_loss = cls_loss * self.hyp.cls
        dfl_loss = dfl_loss * self.hyp.dfl
        total = (box_loss + cls_loss + dfl_loss) * B
        return total, torch.stack((box_loss.detach(), cls_loss.detach(), dfl_loss.detach()))
