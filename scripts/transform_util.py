import torch
import torch.nn.functional as F


def calc_valid_map(source_map, dx, dy, dtheta, x_lim, y_lim):
    # source_map: (B,1,H,W), a prediction map for time t+1 expressed in the t frame
    B, C, H, W = source_map.shape
    device = source_map.device

    x_min, x_max = x_lim
    y_min, y_max = y_lim
    map_w = (x_max - x_min) 
    map_h = (y_max - y_min)

    # create base grid:
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B,1,1,1)  # (B,H,W,2)

    # map to world coordinates
    u, v = base[...,0], base[...,1]
    x_tg = x_min + (u + 1) * 0.5 * map_w
    y_tg = y_min + (v + 1) * 0.5 * map_h

    ct = torch.cos(dtheta).view(B,1,1)
    st = torch.sin(dtheta).view(B,1,1)
    dxv = dx.view(B,1,1)
    dyv = dy.view(B,1,1)

    x_shift = x_tg - dxv
    y_shift = y_tg - dyv

    # inverse rotate (target -> source)
    x_src =  ct * x_shift + st * y_shift
    y_src = -st * x_shift + ct * y_shift

    u_src = 2.0 * (x_src - x_min) / map_w - 1.0
    v_src = 2.0 * (y_src - y_min) / map_h - 1.0
    grid = torch.stack([u_src, v_src], dim=-1)  # (B,H,W,2)

    # prediction map aligned to the t+1 frame
    fin_pred_map = F.grid_sample(source_map, grid, mode='bilinear',
                           padding_mode='zeros', align_corners=True)

    # validity mask over the target frame, marking pixels that fall within the source map bounds
    valid_mask = (u_src >= -1) & (u_src <= 1) & (v_src >= -1) & (v_src <= 1)
    valid_mask = valid_mask.float().unsqueeze(1)  # (B,1,H,W)

    return fin_pred_map, valid_mask


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
