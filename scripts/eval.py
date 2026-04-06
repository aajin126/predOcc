import torch


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