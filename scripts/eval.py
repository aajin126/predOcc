import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_iou(pred, gt, occ_thr=0.2):
    pred_occ = (pred > occ_thr)
    gt_occ   = (gt > occ_thr)

    inter = (pred_occ & gt_occ).sum().float()
    union = (pred_occ | gt_occ).sum().float()

    iou = inter / (union + 1e-6)

    return iou

def compute_miou(pred, gt, occ_thr=0.5):
    pred_occ = (pred > occ_thr)
    gt_occ   = (gt > occ_thr)

    pred_free = ~pred_occ
    gt_free   = ~gt_occ

    inter_occ = (pred_occ & gt_occ).sum().float()
    union_occ = (pred_occ | gt_occ).sum().float()
    iou_occ = inter_occ / (union_occ + 1e-6)

    inter_free = (pred_free & gt_free).sum().float()
    union_free = (pred_free | gt_free).sum().float()
    iou_free = inter_free / (union_free + 1e-6)

    miou = (iou_occ + iou_free) / 2.0

    return miou

def compute_ssim_metric(pred, gt):
    """Compute SSIM between prediction and ground truth.
    
    Args:
        pred: prediction tensor (C, H, W)
        gt: ground truth tensor (C, H, W)
    
    Returns:
        ssim score (float)
    """
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    gt_np = gt.detach().cpu().numpy().astype(np.float32)
    
    # Compute SSIM (data_range should be 1.0 for normalized values)
    score = ssim(pred_np, gt_np, data_range=1.0, channel_axis=0)
    
    return float(score)