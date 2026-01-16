import torch
import torch.nn.functional as F
import math

# def transform_map(source_map, dx, dy, dtheta, x_lim, y_lim):
#     # source_map: (B,1,H,W), a prediction map for time t+1 expressed in the t frame
#     B, C, H, W = source_map.shape
#     device = source_map.device
#     dtype  = source_map.dtype

#     x_min, x_max = x_lim
#     y_min, y_max = y_lim
#     map_w = (y_max - y_min)
#     map_h = (x_max - x_min) 

#     # create base grid:
#     j = torch.arange(W, device=device, dtype=dtype)
#     i = torch.arange(H, device=device, dtype=dtype)
#     ii, jj = torch.meshgrid(i, j, indexing='ij')  # (H,W)

#     u = (2.0 * (jj + 0.5) / W) - 1.0  # width-normalized
#     v = (2.0 * (ii + 0.5) / H) - 1.0  # height-normalized

#     u = u.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)
#     v = v.unsqueeze(0).expand(B, -1, -1)

#     x_tg = x_min + (v + 1) * 0.5 * map_h
#     y_tg = y_min + (u + 1) * 0.5 * map_w

#     ct = torch.cos(dtheta).view(B,1,1)
#     st = torch.sin(dtheta).view(B,1,1)
#     dxv = dx.view(B,1,1)
#     dyv = dy.view(B,1,1)

#     x_shift = x_tg + dxv
#     y_shift = y_tg + dyv

#     # inverse rotate (target -> source)
#     x_src =  ct * x_shift + st * y_shift
#     y_src = -st * x_shift + ct * y_shift

#     u_src = 2.0 * (y_src - y_min) / map_w - 1.0
#     v_src = 2.0 * (x_src - x_min) / map_h - 1.0
#     grid = torch.stack([u_src, v_src], dim=-1)  # (B,H,W,2)

#     # prediction map aligned to the t+1 frame
#     fin_pred_map = F.grid_sample(source_map, grid, mode='nearest',
#                            padding_mode='zeros', align_corners=False)

#     # validity mask over the target frame, marking pixels that fall within the source map bounds
#     valid_mask = (u_src >= -1) & (u_src <= 1) & (v_src >= -1) & (v_src <= 1)
#     valid_mask = valid_mask.float().unsqueeze(1)  # (B,1,H,W)

#     return fin_pred_map, valid_mask

def transform_map(source_map, dx, dy, dtheta, x_lim, y_lim):

    B, C, H, W = source_map.shape
    device = source_map.device
    dtype  = source_map.dtype

    x_min, x_max = x_lim
    y_min, y_max = y_lim

    # cell sizes (world units per cell)
    res_x = (x_max - x_min) / H
    res_y = (y_max - y_min) / W

    # 1) build target grid indices
    ix = torch.arange(H, device=device)  # 0..H-1 (x-index)
    iy = torch.arange(W, device=device)  # 0..W-1 (y-index)
    ix_grid, iy_grid = torch.meshgrid(ix, iy, indexing='ij')  # (H,W)

    # 2) target cell centers -> world coords (t+1 frame)
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
    return fin_pred_map, valid_mask

# import torch
# import torch.nn.functional as F


# def calc_valid_mask(source_map, dx, dy, dtheta, x_lim, y_lim, pivot=(0.0, 0.0)):
#     # returns (B,1,H,W) where 1 means target pixel maps to a valid source coordinate
#     B, _, H, W = source_map.shape
#     dev, dt = source_map.device, source_map.dtype

#     x_min, x_max = float(x_lim[0]), float(x_lim[1])
#     y_min, y_max = float(y_lim[0]), float(y_lim[1])
#     map_w = x_max - x_min
#     map_h = y_max - y_min

#     # target pixel centers grid (H,W) in world coords
#     ys = torch.linspace(0, H - 1, H, device=dev, dtype=dt)
#     xs = torch.linspace(0, W - 1, W, device=dev, dtype=dt)
#     vv, uu = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)

#     x_t = x_min + ((H - (vv + 0.5)) / H) * map_w
#     y_t = y_min + ((uu + 0.5) / W) * map_h

#     x_t = x_t[None].repeat(B, 1, 1)  # (B,H,W)
#     y_t = y_t[None].repeat(B, 1, 1)

#     dxv = dx.to(device=dev, dtype=dt).view(B, 1, 1)
#     dyv = dy.to(device=dev, dtype=dt).view(B, 1, 1)
#     th  = dtheta.to(device=dev, dtype=dt).view(B, 1, 1)

#     c = torch.cos(th)
#     s = torch.sin(th)

#     x0, y0 = float(pivot[0]), float(pivot[1])

#     x_m = x_t - dxv - x0
#     y_m = y_t - dyv - y0

#     # R(-th)
#     x_s =  c * x_m + s * y_m + x0
#     y_s = -s * x_m + c * y_m + y0

#     # inside source bounds?
#     valid = (x_s >= x_min) & (x_s <= x_max) & (y_s >= y_min) & (y_s <= y_max)
#     return valid.float().unsqueeze(1)


# def transform_map(source_map, dx, dy, dtheta, x_lim, y_lim):
#     B, _, H, W = source_map.shape
#     dev, dt = source_map.device, source_map.dtype

#     dx = dx.expand(B)    
#     dy = dy.expand(B)    
#     dtheta = dtheta.expand(B)  

#     x_min, x_max = x_lim
#     y_min, y_max = y_lim
#     map_w = (x_max - x_min)
#     map_h = (y_max - y_min)

# 		#result map init
#     out = torch.zeros((B, 1, H, W), device=dev, dtype=dt)

#     src = source_map[:, 0]                 # (B,H,W)
#     pad = F.pad(src, (1,1,1,1), value=0.0) # padding
#     vW = 2 * W #to obtain center coord as int

#     for b in range(B):
#         occ = (src[b] != 0)  # masking occupied cell 
#         if not torch.any(occ):
#             continue

#         ij = torch.nonzero(occ, as_tuple=False)
#         h = ij[:, 0].long()
#         w = ij[:, 1].long()

#         # 1) sampling
        
#         # centers 
#         gx_c, gy_c = 2*w + 1, 2*h + 1 
#         val_c = src[b, h, w]
				
# 				# corners
#         gx00, gy00 = 2*w,2*h
#         gx10, gy10 = 2*(w+1),2*h
#         gx01, gy01 = 2*w,2*(h+1)
#         gx11, gy11 = 2*(w+1),2*(h+1)

#         hp, wp = h + 1, w + 1 # for padding

#         # calc corners occupancy prob
#         v00 = torch.stack([pad[b,hp-1,wp-1], pad[b,hp-1,wp], pad[b,hp,wp-1], pad[b,hp,wp]], 0).amax(0)
#         v10 = torch.stack([pad[b,hp-1,wp], pad[b,hp-1,wp+1], pad[b,hp,wp], pad[b,hp,wp+1]], 0).amax(0)
#         v01 = torch.stack([pad[b,hp,wp-1], pad[b,hp,wp], pad[b,hp+1,wp-1], pad[b,hp+1,wp]], 0).amax(0)
#         v11 = torch.stack([pad[b,hp,wp], pad[b,hp,wp+1], pad[b,hp+1,wp], pad[b,hp+1,wp+1]], 0).amax(0)

# 		# total occupancy prob of samples
#         gx = torch.cat([gx_c, gx00, gx10, gx01, gx11], 0)
#         gy = torch.cat([gy_c, gy00, gy10, gy01, gy11], 0)
#         val = torch.cat([val_c, v00,  v10,  v01,  v11 ], 0)

#         # remove duplicates
#         v_id = gy * (vW + 1) + gx
#         v_id_u, inv = torch.unique(v_id, return_inverse=True)
#         val_u = torch.full((v_id_u.numel(),), -torch.inf, device=dev, dtype=dt)
#         val_u.scatter_reduce_(0, inv, val, reduce="amax", include_self=True)

# 		# unique gx, gy
#         gx_u = (v_id_u % (vW + 1)).long()
#         gy_u = (v_id_u // (vW + 1)).long()

#         # 2) transform to t+1
#         y = y_min + (gx_u.to(dt) / (2.0 * W)) * map_h
#         x = x_min + ((2.0 * H - gy_u.to(dt)) / (2.0 * H)) * map_w

#         c = torch.cos(dtheta[b])
#         s = torch.sin(dtheta[b])

#         x0 = 0.0
#         y0 = 0.0

#         x_rel = x - x0
#         y_rel = y - y0

#         x_rot = c * x_rel - s * y_rel
#         y_rot = s * x_rel + c * y_rel

#         x1 = x_rot + x0 + dx[b]
#         y1 = y_rot + y0 + dy[b]

#         # 3) fill the final predicted map (cell-wise max)
#         u = (y1 - y_min) / map_h * W
#         v = (x_max - x1) / map_w * H
#         ww = torch.floor(u).long()
#         hh = torch.floor(v).long()

#         eps = 1e-6
#         inb = (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W) \
#             & ((u - torch.floor(u)) > eps) & ((u - torch.floor(u)) < 1 - eps) \
#             & ((v - torch.floor(v)) > eps) & ((v - torch.floor(v)) < 1 - eps)
#         if not torch.any(inb):
#             continue

#         lin = hh[inb] * W + ww[inb]
#         out_flat = out[b, 0].view(-1)
#         out_flat.scatter_reduce_(0, lin, val_u[inb], reduce="amax", include_self=True)

#     valid_mask = calc_valid_mask(source_map, dx, dy, dtheta, x_lim, y_lim)
#     return out, valid_mask

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) t frame 기준 로봇 [0,0,0]일 때 맵 (네가 준 1번)
map1_t = torch.tensor([
    [1,0,0,0,0],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [0,0,1,0,1],
    [0,0,1,0,0],
], dtype=torch.float32, device=device).view(1,1,5,5)

# 2) t frame 기준인데 t+1 time step일 때 맵 = source_map (네가 준 2번)
source_map = torch.tensor([
    [1,0,0,0,0],
    [1,0,0,0,1],
    [1,0,1,0,1],
    [0,0,1,0,1],
    [0,0,0,0,0], 
], dtype=torch.float32, device=device).view(1,1,5,5)

# 3) t+1 frame 기준 t+1 time step GT (네가 준 3번)
gt_map_t1 = torch.tensor([
    [1,0,0,0,1],
    [1,0,1,0,1],
    [0,0,1,0,1],
    [0,0,0,0,0],
    [0,0,0,0,0],
], dtype=torch.float32, device=device).view(1,1,5,5)

pi_ = math.pi
# 4) ---- motion: t -> t+1 (dx=0, dy=1, dtheta=0) ---- (네가 준 4번)
dx = torch.tensor([0.0], dtype=torch.float32, device=device)
dy = torch.tensor([0.0], dtype=torch.float32, device=device)
dtheta = torch.tensor([pi_/6], dtype=torch.float32, device=device)

x_lim = [0.0, 5.0]
y_lim = [-2.5, 2.5]

fin_pred_map, valid_mask = transform_map(source_map, dx, dy, dtheta, x_lim, y_lim)

print("map1_t:\n", map1_t[0,0].double().cpu())
print("source_map:\n", source_map[0,0].double().cpu())
print("fin_pred_map:\n", fin_pred_map[0,0].double().cpu())
print("valid_mask:\n", valid_mask[0,0].double().cpu())
#print("gt_map_t1:\n", gt_map_t1[0,0].double().cpu())
#print("L1 diff(fin_pred vs GT):", (fin_pred_map - gt_map_t1).abs().sum().item())