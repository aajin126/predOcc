import torch
import torch.nn.functional as F
import math

eps = 1e-4

def occ_interp(source_map, target_map, dx, dy, dtheta, x_lim, y_lim, occ_th=0.1):

    B, C, H, W = target_map.shape
    device, dtype = target_map.device, target_map.dtype

    x_min, x_max = x_lim
    y_min, y_max = y_lim
    res_x = (x_max - x_min) / H
    res_y = (y_max - y_min) / W

    out = target_map.clone()

    # --- existing probs/occ in target (do not overwrite these)
    existing_prob = out[:, 0]    
    existing_occ  = existing_prob > 0    

    # source occupied grids
    src_prob = source_map[:, 0]             
    src_occ  = (src_prob >= occ_th)          

    ct = torch.cos(dtheta).to(dtype)        
    st = torch.sin(dtheta).to(dtype)

    mask_new = torch.zeros((B, H, W), device=device, dtype=torch.bool)

    for b in range(B):
        occ_idx = torch.nonzero(src_occ[b], as_tuple=False) # src occupied grid indices
        if occ_idx.numel() == 0:
            continue

        ix0 = occ_idx[:, 0]
        iy0 = occ_idx[:, 1]

        # 1) 4 corners as vertex grid indices
        vx = torch.cat([ix0, ix0 + 1, ix0, ix0 + 1], dim=0)
        vy = torch.cat([iy0, iy0, iy0 + 1, iy0 + 1], dim=0)

        # unique via 1D key
        key = vx * (W + 1) + vy
        key = torch.unique(key)
        vx = key // (W + 1)
        vy = key %  (W + 1)

        cand = [(vx - 1, vy - 1), (vx - 1, vy), (vx,vy - 1), (vx, vy)]
        sum_p = torch.zeros_like(vx, dtype=dtype, device=device)
        cnt_p = torch.zeros_like(vx, dtype=dtype, device=device)

        for cx, cy in cand:
            v = (cx >= 0) & (cx < H) & (cy >= 0) & (cy < W)
            if v.any():
                p = src_prob[b, cx[v], cy[v]].to(dtype)
                o = (p >= occ_th).to(dtype)
                sum_p[v] += p * o
                cnt_p[v] += o

        v_has = cnt_p > 0
        if not v_has.any():
            continue

        vx = vx[v_has]
        vy = vy[v_has]
        corner_p = (sum_p[v_has] / cnt_p[v_has]) # mean prob for each corner
        corner_p = corner_p * 0.7

        #  grid -> world in frame t
        x_v_t = x_min + vx.to(dtype) * res_x
        y_v_t = y_min + vy.to(dtype) * res_y

        # 2) transform to frame t+1
        dx_b = dx[b].to(dtype)
        dy_b = dy[b].to(dtype)
        x_rel = x_v_t - dx_b
        y_rel = y_v_t - dy_b

        x_v_t1 =  ct[b] * x_rel + st[b] * y_rel
        y_v_t1 = -st[b] * x_rel + ct[b] * y_rel

        # 3)t+1 world -> t+1 grid indices
        ix_t = torch.floor((x_v_t1 - x_min) / res_x).to(torch.long)
        iy_t = torch.floor((y_v_t1 - y_min) / res_y).to(torch.long)

        valid = (ix_t >= 0) & (ix_t < H) & (iy_t >= 0) & (iy_t < W)
        ix_t = ix_t[valid]
        iy_t = iy_t[valid]
        corner_p = corner_p[valid]

        # skip grids that already have prob
        already = existing_occ[b, ix_t, iy_t]
        if (~already).sum() == 0:
            continue        
        ix_t = ix_t[~already]
        iy_t = iy_t[~already]
        corner_p = corner_p[~already]

        # if some corner samples overlap on the same target grid -> using mean
        flat = ix_t * W + iy_t
        ngrid = H * W

        sum_t = torch.zeros((ngrid,), device=device, dtype=dtype)
        cnt_t = torch.zeros((ngrid,), device=device, dtype=dtype)

        sum_t.scatter_add_(0, flat, corner_p)
        cnt_t.scatter_add_(0, flat, torch.ones_like(corner_p, dtype=dtype))

        sel = cnt_t > 0
        if not sel.any():
            continue

        # mean values per grid
        mean_t = sum_t[sel] / cnt_t[sel]
        idx = torch.nonzero(sel, as_tuple=False).squeeze(1)

        ixw = idx // W
        iyw = idx % W

        if ixw.numel() == 0:
            continue
        out[b, 0, ixw, iyw] = mean_t
        mask_new[b, ixw, iyw] = True

    return out, mask_new.unsqueeze(1).to(dtype)

def transform_map(source_map, dx, dy, dtheta, x_lim, y_lim):

    B, C, H, W = source_map.shape
    device = source_map.device
    dtype  = source_map.dtype
    
    dx = dx.expand(B)
    dy = dy.expand(B)
    dtheta = dtheta.expand(B)  

    x_min, x_max = x_lim
    y_min, y_max = y_lim

    # grid sizes (world units per grid)
    res_x = (x_max - x_min) / H
    res_y = (y_max - y_min) / W

    # 1) build target grid indices
    ix = torch.arange(H, device=device)  # 0..H-1 (x-index)
    iy = torch.arange(W, device=device)  # 0..W-1 (y-index)
    ix_grid, iy_grid = torch.meshgrid(ix, iy, indexing='ij')  # (H,W)

    # 2) target grid centers -> world coords (t+1 frame)
    x_tg = x_min + (ix_grid.to(dtype) + 0.5) * res_x 
    y_tg = y_min + (iy_grid.to(dtype) + 0.5) * res_y 

    # expand to batch: (B,H,W)
    x_tg = x_tg.unsqueeze(0).expand(B, -1, -1)
    y_tg = y_tg.unsqueeze(0).expand(B, -1, -1)

    # 3) inverse transform: p_src = R(-θ) (p_tg - t)
    ct = torch.cos(dtheta).view(B, 1, 1).to(dtype)
    st = torch.sin(dtheta).view(B, 1, 1).to(dtype)
    dxv = dx.view(B, 1, 1).to(dtype)
    dyv = dy.view(B, 1, 1).to(dtype)

    x_shift = x_tg + dxv
    y_shift = y_tg + dyv

    x_src =  ct * x_shift - st * y_shift
    y_src = st * x_shift + ct * y_shift

    # 4) source world coords -> source grid indices (nearest via floor)
    ix_src = torch.floor((x_src - x_min) / res_x).to(torch.long)
    iy_src = torch.floor((y_src - y_min) / res_y).to(torch.long)

    valid = (ix_src >= 0) & (ix_src < H) & (iy_src >= 0) & (iy_src < W)  # (B,H,W)

    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, W)

    fin = torch.zeros((B, H, W), device=device, dtype=dtype) 
    fin[valid] = source_map[b_idx[valid], 0, ix_src[valid], iy_src[valid]]
    fin_pred_map = fin.unsqueeze(1)                       # (B,1,H,W)

    valid_mask = valid.to(dtype).unsqueeze(1)             # (B,1,H,W)

    # interp_pred_map, mask_interp = occ_interp(source_map, fin_pred_map, dx, dy, dtheta, x_lim, y_lim)
    
    # merged_mask = (mask_interp > 0) | (valid_mask > 0)
    # merged_mask = merged_mask.to(dtype) 

    return fin_pred_map, valid_mask #interp_pred_map, merged_mask

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