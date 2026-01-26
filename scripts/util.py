import torch
import torch.nn.functional as F
import math

eps = 1e-4

def reprojection(source_map, dx, dy, dtheta, x_lim, y_lim):
    B, C, H, W = source_map.shape
    device, dtype = source_map.device, source_map.dtype

    dx = dx.expand(B)
    dy = dy.expand(B)
    dtheta = dtheta.expand(B)

    x_min, x_max = x_lim
    y_min, y_max = y_lim

    # 1) prob -> logit
    source_logit = torch.logit(source_map.clamp(eps, 1 - eps))

    # 2) target grid centers (t+1 frame) in world coords
    res_x = (x_max - x_min) / H
    res_y = (y_max - y_min) / W

    ix = torch.arange(H, device=device)
    iy = torch.arange(W, device=device)
    ix_grid, iy_grid = torch.meshgrid(ix, iy, indexing='ij')  # (H,W)

    x_tg = x_min + (ix_grid.to(dtype) + 0.5) * res_x
    y_tg = y_min + (iy_grid.to(dtype) + 0.5) * res_y

    x_tg = x_tg.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)
    y_tg = y_tg.unsqueeze(0).expand(B, -1, -1)

    # 3) reprojection : inverse transform to find source sampling locations
    ct = torch.cos(dtheta).view(B, 1, 1).to(dtype)
    st = torch.sin(dtheta).view(B, 1, 1).to(dtype)
    dxv = dx.view(B, 1, 1).to(dtype)
    dyv = dy.view(B, 1, 1).to(dtype)

    x_shift = x_tg + dxv
    y_shift = y_tg + dyv

    x_src =  ct * x_shift - st * y_shift
    y_src =  st * x_shift + ct * y_shift

    # --- 4) world -> normalized coords for grid_sample
    grid_x = 2.0 * (y_src - y_min) / (y_max - y_min) - 1.0  # -> W-axis
    grid_y = 2.0 * (x_src - x_min) / (x_max - x_min) - 1.0  # -> H-axis

    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B,H,W,2)

    # valid mask (normalized range)
    valid = (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)
    valid_mask = valid.to(dtype).unsqueeze(1)  # (B,1,H,W)

    # 5) bilinear warp in logit space, then sigmoid back to prob
    warped_logit = F.grid_sample(source_logit, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    warped_prob = torch.sigmoid(warped_logit/0.2)

    warped_prob = warped_prob * valid_mask

    return warped_prob, valid_mask


def reprojection_logits(source_logits, dx, dy, dtheta, x_lim, y_lim):
    B, C, H, W = source_logits.shape
    device, dtype = source_logits.device, source_logits.dtype

    dx = dx.expand(B); dy = dy.expand(B); dtheta = dtheta.expand(B)

    x_min, x_max = x_lim
    y_min, y_max = y_lim

    res_x = (x_max - x_min) / H
    res_y = (y_max - y_min) / W

    ix = torch.arange(H, device=device)
    iy = torch.arange(W, device=device)
    ix_grid, iy_grid = torch.meshgrid(ix, iy, indexing='ij')

    x_tg = x_min + (ix_grid.to(dtype) + 0.5) * res_x
    y_tg = y_min + (iy_grid.to(dtype) + 0.5) * res_y
    x_tg = x_tg.unsqueeze(0).expand(B, -1, -1)
    y_tg = y_tg.unsqueeze(0).expand(B, -1, -1)

    ct = torch.cos(dtheta).view(B,1,1).to(dtype)
    st = torch.sin(dtheta).view(B,1,1).to(dtype)
    dxv = dx.view(B,1,1).to(dtype)
    dyv = dy.view(B,1,1).to(dtype)

    x_shift = x_tg + dxv
    y_shift = y_tg + dyv
    x_src =  ct * x_shift - st * y_shift
    y_src =  st * x_shift + ct * y_shift

    grid_x = 2.0 * (y_src - y_min) / (y_max - y_min) - 1.0
    grid_y = 2.0 * (x_src - x_min) / (x_max - x_min) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    valid = (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)
    valid_mask = valid.to(dtype).unsqueeze(1)

    warped_logits = F.grid_sample(source_logits, grid, mode="bilinear",
                                  padding_mode="zeros", align_corners=False)
    
    warped_logits = warped_logits * valid_mask

    return warped_logits, valid_mask
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # 1) t frame 기준 로봇 [0,0,0]일 때 맵 
# map1_t = torch.tensor([
#     [1,0,0,0,0],
#     [1,0,0,0,1],
#     [1,0,0,0,1],
#     [0,0,1,0,1],
#     [0,0,1,0,0],
# ], dtype=torch.float32, device=device).view(1,1,5,5)

# # 2) t frame 기준인데 t+1 time step일 때 맵 = source_map 
# source_map = torch.tensor([
#     [1,0,0,0,0],
#     [1,0,0,0,1],
#     [1,0,1,0,1],
#     [0,0,1,0,1],
#     [0,0,0,0,0], 
# ], dtype=torch.float32, device=device).view(1,1,5,5)

# # 3) t+1 frame 기준 t+1 time step GT 
# gt_map_t1 = torch.tensor([
#     [1,0,0,0,1],
#     [1,0,1,0,1],
#     [0,0,1,0,1],
#     [0,0,0,0,0],
#     [0,0,0,0,0],
# ], dtype=torch.float32, device=device).view(1,1,5,5)

# pi_ = math.pi
# # 4) ---- motion: t -> t+1 (dx=0, dy=1, dtheta=0) ---- (네가 준 4번)
# dx = torch.tensor([1.0], dtype=torch.float32, device=device)
# dy = torch.tensor([0.0], dtype=torch.float32, device=device)
# dtheta = torch.tensor([0.0], dtype=torch.float32, device=device)

# x_lim = [0.0, 5.0]
# y_lim = [-2.5, 2.5]

# fin_pred_map,valid_mask= reprojection(source_map, dx, dy, dtheta, x_lim, y_lim)


# print("map1_t:\n", map1_t[0,0].double().cpu())
# print("source_map:\n", source_map[0,0].double().cpu())
# print("fin_pred_map:\n", fin_pred_map[0,0].double().cpu())
# print("valid_mask:\n", valid_mask[0,0].double().cpu())
# # print("interp_pred_map:\n", interp_pred_map[0,0].double().cpu())
# # print("merged map:\n", merged_mask[0,0].double().cpu())
# # print("mask_interp:\n", mask_interp[0,0].double().cpu())
# # print("gt_map_t1:\n", gt_map_t1[0,0].double().cpu())
# # print("L1 diff(fin_pred vs GT):", (fin_pred_map - gt_map_t1).abs().sum().item())

# fin_pred_map,valid_mask=  transform_map(source_map, dx, dy, dtheta, x_lim, y_lim)

# print("fin_pred_map:\n", fin_pred_map[0,0].double().cpu())
# print("valid_mask:\n", valid_mask[0,0].double().cpu())