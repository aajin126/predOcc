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
    fin_pred_map = F.grid_sample(source_map, grid, mode='nearest',
                           padding_mode='zeros', align_corners=True)

    # validity mask over the target frame, marking pixels that fall within the source map bounds
    valid_mask = (u_src >= -1) & (u_src <= 1) & (v_src >= -1) & (v_src <= 1)
    valid_mask = valid_mask.float().unsqueeze(1)  # (B,1,H,W)

    return fin_pred_map, valid_mask


# ---- 여기부터 "맵 만들기" ----
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) t frame 기준 로봇 [0,0,0]일 때 맵 (네가 준 1번)
map1_t = torch.tensor([
    [1,0,1,0,0],
    [1,0,1,0,1],
    [1,0,0,0,1],
    [0,0,0,0,1],
    [0,0,0,0,0],  # <- (0,0,0) 표시한 셀은 값 0이라고 했으니 0으로 둠
], dtype=torch.float32, device=device).view(1,1,5,5)

# 2) t frame 기준인데 t+1 time step일 때 맵 = source_map (네가 준 2번)
source_map = torch.tensor([
    [1,0,0,0,0],
    [1,0,1,0,1],
    [1,0,1,0,1],
    [0,0,0,0,1],
    [0,0,0,0,0],
], dtype=torch.float32, device=device).view(1,1,5,5)

# 3) t+1 frame 기준 t+1 time step GT (네가 준 3번)
gt_map_t1 = torch.tensor([
    [1,0,0,0,0],
    [1,0,0,0,0],
    [1,0,1,0,1],
    [1,0,1,0,1],
    [0,0,0,0,1],
], dtype=torch.float32, device=device).view(1,1,5,5)

# ---- motion: t -> t+1 (dx=0, dy=1, dtheta=0) ----
dx = torch.tensor([0.0], dtype=torch.float32, device=device)
dy = torch.tensor([1.0], dtype=torch.float32, device=device)
dtheta = torch.tensor([0.0], dtype=torch.float32, device=device)

x_lim = (0.0, 4.0)
y_lim = (-2.0, 2.0)

fin_pred_map, valid_mask = calc_valid_map(source_map, -dx, -dy, -dtheta, x_lim, y_lim)

print("map1_t:\n", map1_t[0,0].int().cpu())
print("source_map:\n", source_map[0,0].int().cpu())
print("fin_pred_map:\n", fin_pred_map[0,0].int().cpu())
print("gt_map_t1:\n", gt_map_t1[0,0].int().cpu())
print("valid_mask:\n", valid_mask[0,0].int().cpu())
print("L1 diff(fin_pred vs GT):", (fin_pred_map - gt_map_t1).abs().sum().item())