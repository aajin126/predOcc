def compute_iou(pred, gt, valid_mask=None, thr=0.5, eps=1e-6):
    """
    pred, gt: torch.Tensor (same shape)
    valid_mask: torch.Tensor (same shape) or None
    thr: threshold for binarization
    
    returns:
        iou (torch scalar)
    """

    pred_bin = (pred > thr).float()
    gt_bin   = (gt > thr).float()

    if valid_mask is None:
        inter = (pred_bin * gt_bin).sum()
        union = ((pred_bin + gt_bin) > 0).float().sum()
    else:
        v = (valid_mask > 0.5).float()
        inter = (pred_bin * gt_bin * v).sum()
        union = (((pred_bin + gt_bin) > 0).float() * v).sum()

    iou = (inter + eps) / (union + eps)
    return iou
