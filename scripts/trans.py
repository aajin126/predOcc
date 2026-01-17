from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import math


# ----------------------------
# 1) transform t -> t+1
# ----------------------------
def coord_transform(source_xy, dx, dy, dtheta):

    ct = np.cos(dtheta)
    st = np.sin(dtheta)

    x, y = source_xy[:, 0], source_xy[:, 1]
    x_rel = x - dx
    y_rel = y - dy

    x1 =  ct * x_rel + st * y_rel
    y1 = -st * x_rel + ct * y_rel
    return np.stack([x1, y1], axis=1).astype(np.float32)

# ----------------------------
# 2) grid to world
# ----------------------------
def grid_to_world(grid, origin, res):

    grid = np.asarray(grid, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)

    i = grid[:, 0]
    j = grid[:, 1]

    x = origin[0] + (j + 0.5) * res
    y = origin[1] + (i + 0.5) * res

    return np.stack([x, y], axis=1).astype(np.float32)

# ----------------------------
# 2) world to grid
# ----------------------------
def world_to_grid(world, origin, res):

    world = np.asarray(world, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)

    x = world[:, 0]
    y = world[:, 1]

    j = np.floor((x - origin[0]) / res).astype(np.int32)
    i = np.floor((y - origin[1]) / res).astype(np.int32)

    return np.stack([i, j], axis=1).astype(np.int32)

# ----------------------------
# 1) 8-neighbor connected components
# ----------------------------
def get_neighbor_components(occ_mask):
    B, H, W = occ_mask.shape
    vis = np.zeros((H, W), dtype=bool)
    comps = []
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for i in range(H):
        for j in range(W):
            if occ_mask[i, j] and not vis[i, j]:
                q = deque([(i, j)])
                vis[i, j] = True
                comp = []
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in nbrs:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and occ_mask[nx, ny] and not vis[nx, ny]:
                            vis[nx, ny] = True
                            q.append((nx, ny))
                comps.append(comp)
    return comps


# ----------------------------
# 2) boundary cells of a component (8-neighbor)
# ----------------------------
def get_boundary_cells(comp, occ_mask):
    B, H, W = occ_mask.shape
    comp_set = set(comp)
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    boundary = []
    for (i, j) in comp:
        is_boundary = False
        for dx, dy in nbrs:
            ni, nj = i + dx, j + dy
            if not (0 <= ni < H and 0 <= nj < W) or ((ni, nj) not in comp_set):
                is_boundary = True
                break
        if is_boundary:
            boundary.append((i, j))
    return boundary


# ----------------------------
# 3) convex hull from points
# ----------------------------
def convex_hull(points_xy):

    if points_xy.shape[0] < 3:
        return points_xy

    pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts, returnPoints=True).reshape(-1, 2)
    return hull


# ----------------------------
# 5) calculate candidate grid index range by AABB (center)
# ----------------------------
def calc_candidate_grid(xy, H, W, x_lim, y_lim):
    """
    return: (i0,i1,j0,j1) inclusive indices for candidate centers
    """
    x_min, x_max = x_lim
    y_min, y_max = y_lim
    res_x = (x_max - x_min) / H
    res_y = (y_max - y_min) / W

    minx, maxx = float(xy[:, 0].min()), float(xy[:, 0].max())
    miny, maxy = float(xy[:, 1].min()), float(xy[:, 1].max())

    # center_x(i) = x_min + (i+0.5)*res_x
    i0 = int(np.floor((minx - x_min)/res_x - 0.5))
    i1 = int(np.ceil ((maxx - x_min)/res_x - 0.5))
    j0 = int(np.floor((miny - y_min)/res_y - 0.5))
    j1 = int(np.ceil ((maxy - y_min)/res_y - 0.5))

    # clamp
    i0 = max(0, min(H-1, i0))
    i1 = max(0, min(H-1, i1))
    j0 = max(0, min(W-1, j0))
    j1 = max(0, min(W-1, j1))
    return i0, i1, j0, j1


# ----------------------------
# 6) Test inside or outside convex hull by rasterization
# ----------------------------
def rasterize(hull_xy, H, W, x_lim, y_lim, i0, i1, j0, j1, eps=0.0):
    """
    return: (H,W) bool mask
    """
    out = np.zeros((H, W), dtype=bool)
    if hull_xy.shape[0] < 3:
        return out

    x_min, x_max = x_lim
    y_min, y_max = y_lim
    res_x = (x_max - x_min) / H
    res_y = (y_max - y_min) / W

    ii = np.arange(i0, i1+1)
    jj = np.arange(j0, j1+1)
    I, J = np.meshgrid(ii, jj, indexing="ij")

    cx = x_min + (I + 0.5) * res_x
    cy = y_min + (J + 0.5) * res_y
    P = np.stack([cx, cy], axis=-1).reshape(-1, 2)  # (N,2)

    A = hull_xy
    B = np.roll(hull_xy, -1, axis=0)
    E = B - A  # (M,2)

    PA = P[:, None, :] - A[None, :, :]  # (N,M,2)
    cross = E[None, :, 0] * PA[:, :, 1] - E[None, :, 1] * PA[:, :, 0]  # (N,M)

    inside_ccw = np.all(cross >= -eps, axis=1)
    inside_cw  = np.all(cross <=  eps, axis=1)
    inside = inside_ccw | inside_cw

    out[i0:i1+1, j0:j1+1] = inside.reshape((i1-i0+1, j1-j0+1))
    return out


def transform_map(source_map, dx, dy, dtheta, x_lim, y_lim, eps=0.0):

    src_prob = source_map[:, 0] 
    occ_mask = (src_prob >= 0.1)

    B, H, W = occ_mask.shape
    comps = get_neighbor_components(occ_mask)
    out = np.zeros((H, W), dtype=bool)

    for comp in comps:
        boundary = get_boundary_cells(comp, occ_mask)
        pts = np.array([[j + 0.5, i + 0.5] for (i, j) in boundary], dtype=np.float32)
        hull = convex_hull(pts)
        if hull.shape[0] < 3:
            continue

        hull_t1 = coord_transform(hull, dx, dy, dtheta)

        i0, i1, j0, j1 = calc_candidate_grid(hull_t1, H, W, x_lim, y_lim)
        out |= rasterize(hull_t1, H, W, x_lim, y_lim, i0, i1, j0, j1, eps=eps)
    
    return out


device = "cuda" if torch.cuda.is_available() else "cpu"
# 1) t frame 기준 로봇 [0,0,0]일 때 맵 
map1_t = torch.tensor([
    [1,0,0,0,0],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [0,0,1,0,1],
    [0,0,1,0,0],
], dtype=torch.float32, device=device).view(1,1,5,5)

# 2) t frame 기준인데 t+1 time step일 때 맵 = source_map 
source_map = torch.tensor([
    [1,0,0,0,0],
    [1,0,0,0,1],
    [1,0,1,0,1],
    [0,0,1,0,1],
    [0,0,0,0,0], 
], dtype=torch.float32, device=device).view(1,1,5,5)

# 3) t+1 frame 기준 t+1 time step GT 
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
dtheta = torch.tensor([pi_/2], dtype=torch.float32, device=device)

x_lim = [0.0, 5.0]
y_lim = [-2.5, 2.5]

fin_pred_map= transform_map(source_map, dx, dy, dtheta, x_lim, y_lim)

print("map1_t:\n", map1_t[0,0].double().cpu())
print("source_map:\n", source_map[0,0].double().cpu())
print("fin_pred_map:\n", print(fin_pred_map[0,:,:].double().cpu()))
# print("interp_pred_map:\n", interp_pred_map[0,0].double().cpu())
# print("merged map:\n", merged_mask[0,0].double().cpu())
# print("valid_mask:\n", valid_mask[0,0].double().cpu())
# print("mask_interp:\n", mask_interp[0,0].double().cpu())
# print("gt_map_t1:\n", gt_map_t1[0,0].double().cpu())
# print("L1 diff(fin_pred vs GT):", (fin_pred_map - gt_map_t1).abs().sum().item())