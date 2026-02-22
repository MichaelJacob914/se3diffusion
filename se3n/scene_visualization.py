import time
import numpy as np
import torch
import viser
import argparse
import os
import random
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

import torch.distributed as dist

from se3n_datasets import PoseCo3DDataset, DynamicBatchSampler  
from se3n_datasets import VisCo3DDataset, normalize_cameras, normalize_c2w_by_camera_centers                  
from feature_extractors import Dust3rFeatureExtractor, Pi3FeatureExtractor

from SE3NDiffusion import se3n_diffuser
from SE3NRegression import se3n_regressor
import numpy as np
import viser.transforms as vtf
import cv2
import torch.nn.functional as F
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2_metric.dpt import DepthAnythingV2 as MetricDepthAnythingV2

def K_to_fov_aspect(K: np.ndarray, W: int, H: int):
    fy = K[1, 1]
    fov_y = 2.0 * np.arctan2(H / 2.0, fy)
    aspect = float(W) / float(H)
    return float(fov_y), float(aspect)

def T_c2w_to_wxyz_pos(T_c2w: np.ndarray):
    R = T_c2w[:3, :3]
    t = T_c2w[:3, 3]
    wxyz = vtf.SO3.from_matrix(R).wxyz
    return tuple(wxyz), tuple(t)

def img_tensor_to_u8(img_3hw_t):
    img = img_3hw_t.detach().cpu().numpy().transpose(1,2,0)
    return (img * 255.0).clip(0,255).astype(np.uint8)


# ------------------------- dist/device -------------------------
def init_dist_and_device():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, device, local_rank
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, device, 0

def load_depth_anything_v2(device, encoder="vitl", metric=True, dataset="hypersim"):
    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    ckpt_root = "/vast/projects/kostas/geometric-learning/mgjacob/checkpoints"

    if metric:
        print("Using metric")
        max_depth = 20 if dataset == "hypersim" else 80
        ckpt_path = f"{ckpt_root}/depth_anything_v2_metric_{dataset}_{encoder}.pth"
        model = MetricDepthAnythingV2(**model_configs[encoder], max_depth=max_depth)
    else:
        ckpt_path = f"{ckpt_root}/depth_anything_v2_{encoder}.pth"
        model = DepthAnythingV2(**model_configs[encoder])

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()

@torch.no_grad()
def apply_sim3_to_c2w(T_c2w, s, R_align, t_align):
    R = T_c2w[..., :3, :3]
    t = T_c2w[..., :3, 3]

    R_new = R_align @ R
    t_new = s * (R_align @ t.unsqueeze(-1)).squeeze(-1) + t_align

    T_new = T_c2w.clone()
    T_new[..., :3, :3] = R_new
    T_new[..., :3, 3]  = t_new
    return T_new

def compute_optimal_alignment_left(A: torch.Tensor, B: torch.Tensor, eps=1e-9):
    """
    Solve A ≈ s * (R @ B) + t  
    A, B: (N,3)
    Returns: (s, R, t)
    """
    assert A.shape == B.shape and A.shape[1] == 3
    device = A.device

    mu_A = A.mean(0)
    mu_B = B.mean(0)

    A0 = A - mu_A
    B0 = B - mu_B

    # covariance for left-multiply form
    Sigma = (A0.T @ B0) / A.shape[0]  # (3,3)

    U, S, Vh = torch.linalg.svd(Sigma)

    # enforce det(R)=+1
    D = torch.eye(3, device=device, dtype=A.dtype)
    if torch.det(U @ Vh) < 0:
        D[2, 2] = -1.0

    R = U @ D @ Vh

    var_B = (B0 ** 2).sum() / A.shape[0]
    s = (S * torch.diag(D)).sum() / (var_B + eps)

    t = mu_A - s * (R @ mu_B)

    return s, R, t

@torch.no_grad()
def backproject_depth_rgb_to_camera(depth_hw, rgb_3hw, K, stride=4):
    device = depth_hw.device
    H, W = depth_hw.shape

    vv, uu = torch.meshgrid(
        torch.arange(0, H, stride, device=device),
        torch.arange(0, W, stride, device=device),
        indexing="ij",
    )

    z = depth_hw[vv, uu]
    valid = z > 1e-6
    if valid.sum() == 0:
        empty_pts = torch.empty((0, 3), device=device, dtype=torch.float32)
        empty_col = torch.empty((0, 3), device=device, dtype=torch.uint8)
        return empty_pts, empty_col

    vv = vv[valid]
    uu = uu[valid]
    z  = z[valid].float()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (uu.float() - cx) / fx * z
    y = (vv.float() - cy) / fy * z

    pts_c = torch.stack([x, y, z], dim=-1)  # (N,3)

    cols = rgb_3hw[:, vv, uu].permute(1, 0).contiguous()
    cols_u8 = (cols.clamp(0, 1) * 255.0).to(torch.uint8)

    return pts_c, cols_u8

@torch.no_grad()
def depth_anything_forward_224(model: DepthAnythingV2, imgs_3hw: torch.Tensor) -> torch.Tensor:
    """
    imgs_3hw: [3,H,W] float in [0,1] (H,W can be anything)
    returns: depth [H,W] (relative scale)
    """
    assert imgs_3hw.ndim == 3 and imgs_3hw.shape[0] == 3
    x = imgs_3hw.unsqueeze(0)  # [1,3,H,W]
    _, _, H, W = x.shape

    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1,3,1,1)
    x = (x - mean) / std

    pad_h = (14 - (H % 14)) % 14
    pad_w = (14 - (W % 14)) % 14
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")  # (L,R,T,B)

    d = model(x)  # [1,Hpad,Wpad] (or [1,1,Hpad,Wpad] depending on your wrapper)
    if d.ndim == 4:           # be robust
        d = d[:, 0]

    d = d[0, :H, :W]
    return d

@torch.no_grad()
def transform_points_c2w(T_c2w: torch.Tensor, pts_c: torch.Tensor) -> torch.Tensor:
    # T_c2w: (4,4), pts_c: (N,3)
    R = T_c2w[:3, :3]
    t = T_c2w[:3, 3]
    return (pts_c @ R.T) + t  # (N,3)

def pack_layers_c2w(R_layers, t_layers):
    L, B, V = R_layers.shape[:3]
    T = torch.eye(4, device=R_layers.device, dtype=R_layers.dtype).view(1,1,1,4,4).repeat(L,B,V,1,1)
    T[..., :3, :3] = R_layers
    T[..., :3, 3]  = t_layers
    return T

def anchor_to_first(T_c2w):
    if T_c2w.dim() == 3:
        return torch.linalg.inv(T_c2w[0]) @ T_c2w
    if T_c2w.dim() == 4:
        return torch.linalg.inv(T_c2w[:,0])[:,None] @ T_c2w
    raise ValueError(T_c2w.shape)

@torch.no_grad()
def build_world_cloud_for_layer(Ts_c2w: torch.Tensor, pts_c_list, cols_list, max_points=None):
    # Ts_c2w: (V,4,4)
    pts_w_all = []
    cols_all  = []
    for i in range(Ts_c2w.shape[0]):
        pts_c = pts_c_list[i]
        if pts_c.numel() == 0:
            continue
        pts_w = transform_points_c2w(Ts_c2w[i], pts_c)
        pts_w_all.append(pts_w)
        cols_all.append(cols_list[i])

    if len(pts_w_all) == 0:
        empty_pts = np.zeros((0,3), dtype=np.float32)
        empty_col = np.zeros((0,3), dtype=np.uint8)
        return empty_pts, empty_col

    pts_w = torch.cat(pts_w_all, dim=0)
    cols  = torch.cat(cols_all, dim=0)

    if max_points is not None and pts_w.shape[0] > max_points:
        idx = torch.randperm(pts_w.shape[0], device=pts_w.device)[:max_points]
        pts_w = pts_w[idx]
        cols  = cols[idx]

    return pts_w.detach().cpu().numpy().astype(np.float32), cols.detach().cpu().numpy()

def _collect_xy(depths_gt, depths_pred, eps=1e-6, qlo=0.01, qhi=0.99, min_pts=50):
    """
    Collect stacked x=gt depth, y=pred output over all views with trimming.
    Returns x, y as 1D tensors on the original device/dtype (float32).
    """
    X, Y = [], []
    V = depths_gt.shape[0]
    for i in range(V):
        gt_i = depths_gt[i]
        pr_i = depths_pred[i]

        mask = (gt_i > 0) & torch.isfinite(gt_i) & torch.isfinite(pr_i)
        if not mask.any():
            continue

        y = pr_i[mask].reshape(-1)
        x = gt_i[mask].reshape(-1)

        y_lo, y_hi = torch.quantile(y, torch.tensor([qlo, qhi], device=y.device))
        x_lo, x_hi = torch.quantile(x, torch.tensor([qlo, qhi], device=x.device))
        keep = (y >= y_lo) & (y <= y_hi) & (x >= x_lo) & (x <= x_hi)

        y = y[keep]
        x = x[keep]

        if y.numel() < min_pts:
            continue

        X.append(x)
        Y.append(y)

    if len(X) == 0:
        return None, None

    x_all = torch.cat(X, dim=0).float()
    y_all = torch.cat(Y, dim=0).float()

    good = torch.isfinite(x_all) & torch.isfinite(y_all) & (x_all > 0)
    x_all = x_all[good]
    y_all = y_all[good]

    if x_all.numel() < min_pts:
        return None, None

    return x_all, y_all


def _solve_lstsq(M, rhs):
    sol = torch.linalg.lstsq(
        M.detach().to("cpu", dtype=torch.float64),
        rhs.detach().to("cpu", dtype=torch.float64),
    ).solution
    return sol

def save_depth_vis(
    depth: torch.Tensor,
    path: str,
    vmin=None,
    vmax=None,
    percentile=(1, 99),
):
    """
    depth: [H,W] torch tensor
    Saves a uint8 PNG with percentile-based normalization.
    """
    d = depth.detach().cpu().numpy()

    mask = np.isfinite(d) & (d > 0)
    if mask.any():
        if vmin is None or vmax is None:
            lo, hi = np.percentile(d[mask], percentile)
        else:
            lo, hi = vmin, vmax
        d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    else:
        d[:] = 0

    d_u8 = (d * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, d_u8)


@torch.no_grad()
def fit_and_apply_global_calibration(
    depths_gt: torch.Tensor,      # [V,H,W]
    depths_pred: torch.Tensor,    # [V,H,W]
    is_metric: bool,
    eps: float = 1e-6,
    qlo: float = 0.01,
    qhi: float = 0.99,
    min_pts: int = 200,
    verbose: bool = True,
):
    """
    If is_metric:
        depth_scale: depth_gt ≈ s * pred
        apply: depth_cal = clamp(s * pred)

    Else:
        invdepth_affine: 1/depth_gt ≈ a + b * pred
        apply: depth_cal = 1 / clamp(a + b * pred)
    """

    device = depths_pred.device
    dtype = depths_pred.dtype

    x, y = _collect_xy(depths_gt, depths_pred, eps=eps, qlo=qlo, qhi=qhi, min_pts=min_pts)
    if x is None:
        if verbose:
            print("Calibration skipped: not enough valid samples; returning raw pred.")
        return depths_pred, {"model": "none", "reason": "no_samples"}

    if is_metric:
        # -------- SCALE-ONLY SOLUTION --------
        # Solve: min_s || s*y - x ||^2
        # Closed-form: s = (y^T x) / (y^T y)

        y64 = y.detach().to("cpu", dtype=torch.float64)
        x64 = x.detach().to("cpu", dtype=torch.float64)

        denom = torch.dot(y64, y64).clamp_min(1e-12)
        s = torch.dot(y64, x64) / denom
        s = s.to(device=device, dtype=torch.float32)

        if verbose:
            xhat = s * y
            medrel = torch.median((x - xhat).abs() / (x.abs() + eps)).item()
            print(f"[CALIB metric-scale] MedRel={medrel:.6g}  s={float(s.cpu()):.6g}")

        depths_cal = (s * depths_pred.float()).clamp_min(eps).to(device=device, dtype=dtype)
        info = {"model": "depth_scale", "params": {"s": s.to(device=device, dtype=dtype)}}
        return depths_cal, info

    else:
        # -------- ORIGINAL NON-METRIC BRANCH (unchanged) --------
        rhs = 1.0 / x.clamp_min(eps)

        ones = torch.ones_like(y)
        M = torch.stack([ones, y], dim=1)
        sol = _solve_lstsq(M, rhs)

        a = sol[0].to(device=device, dtype=torch.float32)
        b = sol[1].to(device=device, dtype=torch.float32)

        if verbose:
            inv = (a + b * y).clamp_min(eps)
            xhat = 1.0 / inv
            medrel = torch.median((x - xhat).abs() / (x.abs() + eps)).item()
            print(f"[CALIB non-metric] invdepth_affine: MedRel={medrel:.6g}  a={float(a.cpu()):.6g}  b={float(b.cpu()):.6g}")

        inv_full = (a + b * depths_pred.float()).clamp_min(eps)
        depths_cal = (1.0 / inv_full).to(device=device, dtype=dtype)
        info = {"model": "invdepth_affine", "params": {"a": a.to(device=device, dtype=dtype), "b": b.to(device=device, dtype=dtype)}}
        return depths_cal, info

def depth_error_stats(gt, pred, eps=1e-6):
    mask = (gt > 0) & torch.isfinite(gt) & torch.isfinite(pred)
    if mask.sum() < 100:
        return None

    gt_v = gt[mask]
    pr_v = pred[mask]

    abs_rel = torch.median((gt_v - pr_v).abs() / (gt_v.abs() + eps)).item()
    rmse = torch.sqrt(torch.mean((gt_v - pr_v) ** 2)).item()

    # scale drift diagnostic
    ratio = torch.median(gt_v / (pr_v + eps)).item()

    return {
        "med_abs_rel": abs_rel,
        "rmse": rmse,
        "median_scale_ratio": ratio,
    }

@torch.no_grad()
def fit_and_apply_per_view_metric_scale(
    depths_gt: torch.Tensor,      # [V,H,W]
    depths_pred: torch.Tensor,    # [V,H,W]
    eps: float = 1e-6,
    qlo: float = 0.01,
    qhi: float = 0.99,
    min_pts: int = 200,
    verbose: bool = True,
):
    """
    Per-view scale-only calibration for metric branch:
        For each view i: depth_gt_i ≈ s_i * pred_i
        Apply: depth_cal_i = clamp(s_i * pred_i, eps)

    Returns:
      depths_cal: [V,H,W]
      info: dict with s_per_view [V]
    """
    V = depths_gt.shape[0]
    device = depths_pred.device
    dtype = depths_pred.dtype

    s_list = []
    cal_list = []

    for i in range(V):
        gt_i = depths_gt[i]
        pr_i = depths_pred[i]

        # reuse your trimming logic by calling _collect_xy on single-view tensors
        x, y = _collect_xy(gt_i.unsqueeze(0), pr_i.unsqueeze(0), eps=eps, qlo=qlo, qhi=qhi, min_pts=min_pts)
        if x is None:
            # fallback: identity
            s_i = torch.tensor(1.0, device=device, dtype=torch.float32)
            cal_i = pr_i.float().clamp_min(eps).to(device=device, dtype=dtype)
            if verbose:
                print(f"[CALIB per-view] view {i}: skipped (no_samples); s=1.0")
        else:
            y64 = y.detach().to("cpu", dtype=torch.float64)
            x64 = x.detach().to("cpu", dtype=torch.float64)
            denom = torch.dot(y64, y64).clamp_min(1e-12)
            s_i = (torch.dot(y64, x64) / denom).to(device=device, dtype=torch.float32)

            cal_i = (s_i * pr_i.float()).clamp_min(eps).to(device=device, dtype=dtype)

            if verbose:
                xhat = s_i * y
                medrel = torch.median((x - xhat).abs() / (x.abs() + eps)).item()
                print(f"[CALIB per-view] view {i}: MedRel={medrel:.6g}  s={float(s_i.cpu()):.6g}")

        s_list.append(s_i.to(device=device, dtype=dtype))
        cal_list.append(cal_i)

    s_per_view = torch.stack(s_list, dim=0)        # [V]
    depths_cal = torch.stack(cal_list, dim=0)      # [V,H,W]
    info = {"model": "per_view_depth_scale", "params": {"s_per_view": s_per_view}}
    return depths_cal, info

def save_mask_u8(mask: torch.Tensor, path: str):
    m = mask.detach().cpu().numpy().astype(np.uint8) * 255
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, m)

@torch.no_grad()
def visualize_one_scene_viser(
    se3n,
    dataloader,
    device="cuda",
    stride=4,
    max_points=200_000,
    view_limit=None,
    use_train_split=True,
    full_res = False, 
    use_depth_anything: bool = False, 
    depth_anything_model=None, 
    is_metric = True, 
):
    """
    Uses se3n.sample(imgs, Ks, return_intermediate=True) to get layer-wise pose predictions.
    Assumes batch contains: imgs [B,V,3,H,W], depths [B,V,H,W], masks [B,V,H,W], K [B,V,3,3], camera_pose [B,V,4,4]
    """

    batch = next(iter(dataloader))
    b = 0

    # --- always needed downstream ---
    gt_pose   = batch["camera_pose"][b].to(device, non_blocking=True)  # [V,4,4] c2w
    imgs_small = batch["imgs"][b].to(device, non_blocking=True)        # [V,3,224,224]
    Ks_small   = batch["K"][b].to(device, non_blocking=True)           # [V,3,3]

    # --- used for pointcloud / visualization (full or small) ---
    if full_res:
        full = batch["full"]
        imgs_used  = full["imgs"][b].to(device, non_blocking=True)      # [V,3,H,W]
        depths_gt  = full["depths"][b].to(device, non_blocking=True)    # [V,H,W]
        Ks_used    = full["K"][b].to(device, non_blocking=True)         # [V,3,3]
    else:
        imgs_used  = imgs_small                                         # [V,3,224,224]
        depths_gt  = batch["depths"][b].to(device, non_blocking=True)   # [V,224,224]
        Ks_used    = Ks_small                                           # [V,3,3]

    V = depths_gt.shape[0]

    def depth_stats(name, d, mask=None):
        x = d[torch.isfinite(d)] if mask is None else d[mask & torch.isfinite(d)]
        if x.numel() == 0:
            print(f"{name}: no valid pixels")
            return None
        q = torch.quantile(x, torch.tensor([0.01, 0.5, 0.99], device=x.device))
        print(f"{name}: min={x.min().item():.4g}  p01={q[0].item():.4g}  med={q[1].item():.4g}  p99={q[2].item():.4g}  max={x.max().item():.4g}")
        return q[1].item()

    # --- stats on one view (GT mask) ---
    i0 = 0
    gt0 = depths_gt[i0]
    valid = (gt0 > 0) & torch.isfinite(gt0)

    gt_med = depth_stats("GT", gt0, valid)

    depths_da = None


    if use_depth_anything:
        assert depth_anything_model is not None
        eps = 1e-6

        # --- 1) run DepthAnything: raw output y (disparity / inverse-depth-like) ---
        pred_depths = []
        for i in range(V):
            d_i = depth_anything_forward_224(depth_anything_model, imgs_used[i])  # [H,W]
            pred_depths.append(d_i)
        depths_da = torch.stack(pred_depths, dim=0)  # [V,H,W]

    out_root = "/vast/home/m/mgjacob/PARCC/results/depth_debug"
    scene_id = batch["seq_id"]
    os.makedirs(out_root, exist_ok=True)

    for i in range(V):
        save_depth_vis(
            depths_gt[i],
            f"{out_root}/{scene_id}_view{i:02d}_gt.png",
        )
        save_depth_vis(
            depths_da[i],
            f"{out_root}/{scene_id}_view{i:02d}_da_raw.png",
        )

    depths_da_global, calib_info = fit_and_apply_global_calibration(
        depths_gt=depths_gt,
        depths_pred=depths_da,
        is_metric=is_metric,
        eps=1e-6,
        qlo=0.01,
        qhi=0.99,
        min_pts=200,
        verbose=True,
    )

    depths_da_pv, calib_pv = fit_and_apply_per_view_metric_scale(
        depths_gt=depths_gt,
        depths_pred=depths_da,
        eps=1e-6,
        qlo=0.01,
        qhi=0.99,
        min_pts=200,
        verbose=True,
    )

    # Choose which one you want to visualize downstream:
    depths_da = depths_da_pv   # <-- try per-view first

    print("\n--- DEPTH ALIGNMENT STATS (global scale) ---")
    for i in range(V):
        stats = depth_error_stats(depths_gt[i], depths_da_global[i])
        if stats is not None:
            print(f"view {i}: {stats}")

    print("\n--- DEPTH ALIGNMENT STATS (per-view scale) ---")
    for i in range(V):
        stats = depth_error_stats(depths_gt[i], depths_da_pv[i])
        if stats is not None:
            print(f"view {i}: {stats}")
    # --- choose which depths to USE downstream ---
    if(use_depth_anything and depths_da is not None): 
        print("USING DEPTHS ESTIMATED")
        depths_used = depths_da 
    else: 
        depths_used = depths_gt 

    eps = 1e-6
    for i in range(V):
        gt = depths_gt[i]
        pr = depths_da[i]

        valid = (gt > 0) & torch.isfinite(gt) & torch.isfinite(pr)

        # abs relative error map
        rel = torch.zeros_like(gt)
        rel[valid] = (gt[valid] - pr[valid]).abs() / (gt[valid].abs() + eps)

        # mark outliers (tune threshold)
        out = valid & (rel > 0.25)   # start with 25% rel error

        save_depth_vis(rel, f"{out_root}/{scene_id}_view{i:02d}_relerr.png", percentile=(1,99))
        save_mask_u8(out, f"{out_root}/{scene_id}_view{i:02d}_outliermask.png")

    # ---- add batch dimension for normalization
    c2w_norm    = gt_pose.unsqueeze(0)     # (1,V,4,4)
    K_norm      = Ks_used.unsqueeze(0)     # (1,V,3,3)
    depths_norm = depths_used.unsqueeze(0) # (1,V,H,W)

    # ---- downstream expects these to exist ----
    gt_pose_c2w = c2w_norm[0]   # [V,4,4]
    Ks_used     = K_norm[0]     # [V,3,3]
    depths_used = depths_norm[0]# [V,H,W]

    print("imgs_used.shape:", imgs_used.shape)
    print("depths_used.shape:", depths_used.shape)
    print("Ks_used.shape:", Ks_used.shape)
    print("gt_pose_c2w.shape:", gt_pose_c2w.shape)

    H, W = imgs_used.shape[-2:]
    

    C_gt   = gt_pose_c2w[:, :3, 3]        # (V,3)
    
    if(se3n is None):
        sample_out = None
    else:
        se3n.model.eval()
        sample_out = se3n.sample(imgs=imgs_small.unsqueeze(0), Ks=Ks_small.unsqueeze(0), return_intermediates=True)

        # layer-wise c2w components
        R_layers = sample_out["R_layers"].to(device)   # [L,B,V,3,3]
        t_layers = sample_out["t_layers"].to(device)   # [L,B,V,3]

        # pack to 4x4 c2w
        pred_layers_c2w = pack_layers_c2w(R_layers, t_layers)  # [L,B,V,4,4]
        pred_layers_c2w = pred_layers_c2w[:, 0]                # [L,V,4,4] (b=0)

        # final is last layer
        pred_final_c2w = pred_layers_c2w[-1]                   # [V,4,4]

        C_pred = pred_final_c2w[:, :3, 3]     # (V,3)


        s_align, R_align, t_align = compute_optimal_alignment_left(C_gt, C_pred)
        pred_layers_in_gt = apply_sim3_to_c2w(pred_layers_c2w, s_align, R_align, t_align)
        pred_final_in_gt  = pred_layers_in_gt[-1]                                       # [V,4,4]

    pts_c_list, cols_list = [], []
    for i in range(V):
        pts_c_i, cols_i = backproject_depth_rgb_to_camera(
            depth_hw=depths_used[i],
            rgb_3hw=imgs_used[i],
            K=Ks_used[i],
            stride=stride,
        )
        pts_c_list.append(pts_c_i)
        cols_list.append(cols_i)

    # ---- start viser once ----
    server = viser.ViserServer()

    pts_gt_np, cols_gt_np = build_world_cloud_for_layer(
        gt_pose_c2w, pts_c_list, cols_list, max_points=max_points
    )
    server.scene.add_point_cloud("pc_gt", points=pts_gt_np, colors=cols_gt_np, point_size=0.01)
    
    for i in range(V):
        T = gt_pose_c2w[i].detach().cpu().numpy()
        K = Ks_used[i].detach().cpu().numpy()
        fov, aspect = K_to_fov_aspect(K, W, H)
        wxyz, pos = T_c2w_to_wxyz_pos(T)
        server.scene.add_camera_frustum(
            name=f"/gt/cam_{i}",
            fov=fov,
            aspect=aspect,
            scale=1,
            color=(0, 255, 0),
            image=img_tensor_to_u8(imgs_used[i]),
            wxyz=wxyz,
            position=pos,
            variant="wireframe",
        )

    # ---- Pred: slider for layer point clouds + frustums ----
    if se3n is not None:
        layer_max = pred_layers_in_gt.shape[0] - 1

        # create handle once
        pts0_np, cols0_np = build_world_cloud_for_layer(
            pred_layers_in_gt[layer_max], pts_c_list, cols_list, max_points=max_points
        )
        pc_pred = server.scene.add_point_cloud(
            "pc_pred_layer",
            points=pts0_np,
            colors=cols0_np,
            point_size=0.01,
        )

        def set_layer(l: int):
            # frustums
            for i in range(V):
                try:
                    server.scene.remove(f"/pred/cam_{i}")
                except Exception:
                    pass

            T_use = pred_layers_in_gt[l].detach().cpu().numpy()  # [V,4,4]
            for i in range(V):
                T = T_use[i]
                K = Ks_used[i].detach().cpu().numpy()
                fov, aspect = K_to_fov_aspect(K, W, H)
                wxyz, pos = T_c2w_to_wxyz_pos(T)
                server.scene.add_camera_frustum(
                    name=f"/pred/cam_{i}",
                    fov=fov,
                    aspect=aspect,
                    scale=1,
                    color=(255, 0, 0),
                    image=img_tensor_to_u8(imgs_used[i]),
                    wxyz=wxyz,
                    position=pos,
                    variant="wireframe",
                )

            # point cloud
            pts_np, cols_np = build_world_cloud_for_layer(
                pred_layers_in_gt[l], pts_c_list, cols_list, max_points=max_points
            )
            pc_pred.points = pts_np
            pc_pred.colors = cols_np

        gui = server.gui
        s = gui.add_slider("layer", min=0, max=layer_max, step=1, initial_value=layer_max)
        set_layer(layer_max)

        @s.on_update
        def _(_e):
            set_layer(int(s.value))
    
    gui = server.gui
    print("Viser running. Open the URL printed in your terminal.")
    while True:
        time.sleep(1)

# ------------------------- main -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--load", action="store_true", help="Load model.pt from run directory")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--scene-idx", type=int, default=0, help="Index into dataset.sequence_list")
    parser.add_argument("--views", type=int, default=6, help="Number of views sampled per sequence")
    parser.add_argument("--batch-size", type=int, default=1, help="Use B=1 for visualization")
    parser.add_argument("--stride", type=int, default=4, help="Depth backprojection stride")
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-depth-anything", action="store_true")
    parser.add_argument("--da-encoder", type=str, default="vitl", choices=["vits","vitb","vitl","vitg"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_name        = cfg["name"]
    conditioning    = cfg["conditioning"]
    co3d_root       = cfg["dataset_dir"]
    ann_root        = cfg["ann_root"]
    model_type      = cfg["model_type"]
    attn_kwargs     = cfg["attn_kwargs"]
    attn_args       = cfg["attn_args"]
    scheme          = cfg["scheme"]
    prediction      = cfg["prediction"]
    so3_config      = cfg["so3_config"]
    r3_config       = cfg["r3_config"]
    feature_type    = cfg["feature_type"]
    update_type     = cfg["update_type"]
    num_sequences   = cfg.get("num_sequences", None)
    prediction_type = cfg.get("prediction_type", "diffuser")

    is_dist, device, local_rank = init_dist_and_device()
    world_size = dist.get_world_size() if is_dist else 1

    accelerator = Accelerator(mixed_precision="bf16")

    output_dir = Path(f"/vast/projects/kostas/geometric-learning/mgjacob/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if conditioning != "CO3D":
        raise ValueError("This visualizer expects conditioning == CO3D (needs cameras/depth).")

    if feature_type == "dust3r":
        extractor = Dust3rFeatureExtractor(
            ckpt_path=Path("/vast/projects/kostas/geometric-learning/mgjacob/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"),
            device=str(device),
        )
    elif feature_type == "pi3":
        extractor = Pi3FeatureExtractor(
            ckpt_path=Path("/vast/projects/kostas/geometric-learning/pi3_weights/model.safetensors"),
            device=str(device),
        )
    else:
        raise ValueError(f"unknown feature_type={feature_type}")

    depth_anything_model = None
    if args.use_depth_anything:
        depth_anything_model = load_depth_anything_v2(device=str(device), encoder=args.da_encoder, metric = True)

    vis_dataset = VisCo3DDataset(
        co3d_root=co3d_root,
        categories=("backpack",), 
        resize_hw=(224, 224),
        verbose=True,
        ann_root=ann_root,
        split=("train" if args.split == "train" else "test"),
        full_res=False,            
        return_full_data=True,
    )

    common = sorted(set(vis_dataset.sequence_list))
    if(num_sequences is not None): 
        keep = common[:num_sequences]
        if len(keep) < num_sequences:
            print(f"[warn] only {len(keep)} common sequences found; using: {keep}")

        def prune_to(ds, names):
            ds.sequence_list = names
            ds.rotations    = {s: ds.rotations[s] for s in names}
            ds.category_map = {s: ds.category_map[s] for s in names}

        prune_to(vis_dataset, keep)
        print("[INFO] NEW TRAINING  DATASET LENGTH=", len(vis_dataset))  # must be > 0

    sampler = DynamicBatchSampler(
        len(vis_dataset),
        dataset_len=args.batch_size,
        max_images=args.views,
        images_per_seq=[args.views, args.views+1],
    )

    vis_loader = DataLoader(
        vis_dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    vis_loader = accelerator.prepare(vis_loader)

    model_path = (output_dir / "model.pt") 
    
    se3_config = {
        "T": cfg.get("T", 100),
        "device": device,
        "is_dist": is_dist,
        "local_rank": local_rank,
        "world_size": world_size,
        "conditioning": conditioning,
        "dataloader": vis_loader,        
        "attn_args": attn_args,
        "attn_kwargs": attn_kwargs,
        "extractor": extractor,
        "feature_type": feature_type,
        "save_model": False,
        "model_type": model_type,
        "prediction": prediction,
        "so3_config": so3_config,
        "r3_config": r3_config,
        "save_path": output_dir / "model.pt",
        "model_path": model_path,
        "dataset_root": co3d_root,
        "dataset": vis_dataset,
        "num_workers": 0,
        "forward_process": "ve",
        "representation": "rot9d",
        "scheme": scheme,
        "accelerator": accelerator,
        "update_type": update_type,
    }
    if prediction_type == "regressor":
        print("[INFO] Using SE(3) Regressor")
        se3n = se3n_regressor(se3_config)
    else:
        print("[INFO] Using SE(3) Diffuser")
        se3n = se3n_diffuser(se3_config)

    if model_path is not None and model_path.exists():
        print("[INFO] Loading weights:", model_path)
        ckpt = torch.load(model_path, map_location="cpu")
        # adapt to your checkpoint format:
        if isinstance(ckpt, dict) and "model" in ckpt:
            se3n.model.load_state_dict(ckpt["model"], strict=False)
        else:
            se3n.model.load_state_dict(ckpt, strict=False)
    else:
        print("[INFO] No weights loaded (args.load is false or file missing).")

    print(args.use_depth_anything)
    visualize_one_scene_viser(
        se3n=se3n,
        dataloader=vis_loader,
        device=str(device),
        stride=args.stride,
        max_points=args.max_points,
        view_limit=args.views,
        use_depth_anything=args.use_depth_anything,
        depth_anything_model=depth_anything_model,
        full_res = True
    )


if __name__ == "__main__":
    main()