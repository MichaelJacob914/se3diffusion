import argparse
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
paths = [HERE / "gta", HERE / "gta" / "source"]
sys.path[:0] = [str(p) for p in paths if p.exists() and str(p) not in sys.path]
from SO3n import so3_diffuser, SO3Algebra
from R3n import r3_diffuser
import se3n_models
from torch.utils.data import Subset, DataLoader
from pathlib import Path
import os
import random
import contextlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from se3n_datasets import VisCo3DDataset, normalize_cameras, normalize_c2w_by_camera_centers        
from scipy.spatial.transform import Rotation as Rot
import math
import random
from matplotlib import pyplot as plt
import geoopt
from geoopt.optim import (RiemannianAdam)
Stiefel = geoopt.Stiefel()
from matplotlib import cm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import List, Dict, Tuple, Literal
from se3n_utils import rot_trans_from_se3, build_intrinsics
import os
import time
from mpl_toolkits.mplot3d import Axes3D  
from pi3.relpose.metric import (
     se3_to_relative_pose_error_batched_c2w, se3_to_relative_pose_error,
    calculate_auc, calculate_auc_np, rotation_angle, mat_to_quat, _sqrt_positive_part
)

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
import random


def axis_from_R(R: torch.Tensor):
    """
    Convert a rotation matrix (3×3 torch) to its axis (unit 3-vector).
    """
    rotvec = Rot.from_matrix(R.cpu().numpy()).as_rotvec()
    angle  = torch.linalg.norm(torch.from_numpy(rotvec))
    print(angle)
    axis   = rotvec / (angle + 1e-8)
    return axis


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """
    Invert SE(3) matrices.
    T: [...,4,4]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-2, -1)
    t_inv = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    out = torch.eye(4, device=T.device, dtype=T.dtype).expand(T.shape).clone()
    out[..., :3, :3] = Rt
    out[..., :3, 3] = t_inv
    return out
    
def compute_optimal_alignment(A, B):
    """
    Compute the optimal scale s, rotation R, and translation t that minimizes:
    || A - (s * B @ R + T) || ^ 2

    Reference: Umeyama (TPAMI 91)

    Args:
        A (torch.Tensor): (N, 3).
        B (torch.Tensor): (N, 3).

    Returns:
        s (float): scale.
        R (torch.Tensor): rotation matrix (3, 3).
        t (torch.Tensor): translation (3,).
    """
    A_bar = A.mean(0)
    B_bar = B.mean(0)
    # normally with R @ B, this would be A @ B.T
    H = (B - B_bar).T @ (A - A_bar)
    U, S, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det(U @ Vh)
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    variance = torch.sum((B - B_bar) ** 2)
    scale = 1 / variance * torch.trace(torch.diag(S) @ S_prime)
    R = U @ S_prime @ Vh
    t = A_bar - scale * B_bar @ R

    A_hat = scale * B @ R + t
    return A_hat, scale, R, t

def rot_geodesic_deg(Ra, Rb):
    R = Ra.transpose(-2, -1) @ Rb
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_th = 0.5 * (tr - 1.0)
    cos_th = torch.clamp(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_th) * (180.0 / np.pi)

def rel_from_two_c2w(R1, t1, R2, t2):
    Rt1 = R1.transpose(-2, -1)
    R_rel = Rt1 @ R2
    t_rel = (Rt1 @ (t2 - t1).unsqueeze(-1)).squeeze(-1)
    return R_rel, t_rel

def dir_angle_deg(a, b, eps=1e-12):
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na < eps or nb < eps:
        return float("inf")
    cos = torch.clamp((a @ b) / (na * nb), -1.0, 1.0)
    return float(torch.arccos(cos) * (180.0 / np.pi))

def pack_se3_c2w(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Batch-safe pack of SE(3) camera-to-world matrices.

    Args:
        R: [..., 3, 3]
        t: [..., 3]

    Returns:
        T: [..., 4, 4]
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must end with (3,3), got {R.shape}")
    if t.shape[-1] != 3:
        raise ValueError(f"t must end with (3,), got {t.shape}")
    if R.shape[:-2] != t.shape[:-1]:
        raise ValueError(f"Leading dims must match: R {R.shape[:-2]} vs t {t.shape[:-1]}")

    # Create identity with the right leading shape
    T = torch.eye(4, device=R.device, dtype=R.dtype).expand(R.shape[:-2] + (4, 4)).clone()

    T[..., :3, :3] = R
    T[..., :3,  3] = t
    return T

def relative_pose_from_se3(T_i, T_j, closed_form_inverse_se3):
    # matches their convention: T_rel = T_j * inv(T_i)
    return closed_form_inverse_se3(T_i) @ T_j

def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Invert SE(3) transforms for arrays/tensors with shape [...,4,4] or [...,3,4].
    Returns shape [...,4,4].
    """
    is_numpy = isinstance(se3, np.ndarray)

    if se3.shape[-2:] not in [(4, 4), (3, 4)]:
        raise ValueError(f"se3 must end with (4,4) or (3,4), got {se3.shape}.")

    if R is None:
        R = se3[..., :3, :3]     # [...,3,3]
    if T is None:
        T = se3[..., :3, 3:]     # [...,3,1]

    if is_numpy:
        R_T = np.swapaxes(R, -1, -2)           # [...,3,3]
        top_right = -np.matmul(R_T, T)         # [...,3,1]
        out_shape = se3.shape[:-2] + (4, 4)
        inv = np.broadcast_to(np.eye(4, dtype=se3.dtype), out_shape).copy()
        inv[..., :3, :3] = R_T
        inv[..., :3, 3:] = top_right
        return inv
    else:
        R_T = R.transpose(-1, -2)              # [...,3,3]
        top_right = -torch.matmul(R_T, T)      # [...,3,1]
        inv = torch.eye(4, device=se3.device, dtype=se3.dtype)
        inv = inv.expand(se3.shape[:-2] + (4, 4)).clone()
        inv[..., :3, :3] = R_T
        inv[..., :3, 3:] = top_right
        return inv

def trans_l2_with_opt_scale(t_gt: torch.Tensor, t_pr: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Per-pair optimal scale s minimizing || s t_pr - t_gt ||_2.
    t_gt, t_pr: [..., 3]
    Returns: [...], the minimized L2.
    """
    # s = (t_pr · t_gt) / (t_pr · t_pr)
    denom = (t_pr * t_pr).sum(dim=-1, keepdim=True).clamp_min(eps)
    s = (t_pr * t_gt).sum(dim=-1, keepdim=True) / denom
    return (s * t_pr - t_gt).norm(dim=-1)

def trans_l2_with_opt_scale_batched(t_gt, t_pr, eps=1e-12):
    # t_gt, t_pr: [B,3]
    num = (t_gt * t_pr).sum(dim=-1)                 # [B]
    den = (t_pr * t_pr).sum(dim=-1).clamp(min=eps)  # [B]
    s = num / den                                   # [B]
    diff = t_gt - s[:, None] * t_pr                 # [B,3]
    return diff.norm(dim=-1)          
# TODO: this code can be further cleaned up

@torch.no_grad()
def trans_l2_allpairs_opt_scale_c2w(pred_se3, gt_se3, eps=1e-8):
    """
    pred_se3, gt_se3: [B,V,4,4] camera-to-world
    Returns:
      trans_l2_b: [B]  mean || s*t_pr - t_gt || over all i<j pairs
      s: [B]      fitted global scale per item
    """
    device = pred_se3.device
    B, V = pred_se3.shape[:2]

    # indices for unique pairs
    i, j = torch.combinations(torch.arange(V, device=device), r=2).unbind(-1)  # [P], [P]
    P = i.numel()

    # gather poses: [B,P,4,4]
    gt_i = gt_se3[:, i]
    gt_j = gt_se3[:, j]
    pr_i = pred_se3[:, i]
    pr_j = pred_se3[:, j]

    # relative transforms: T_i^-1 T_j
    gt_rel = torch.linalg.inv(gt_i) @ gt_j     # [B,P,4,4]
    pr_rel = torch.linalg.inv(pr_i) @ pr_j     # [B,P,4,4]

    t_gt = gt_rel[:, :, :3, 3]                # [B,P,3]
    t_pr = pr_rel[:, :, :3, 3]                # [B,P,3]

    w = torch.ones((B, P), device=device, dtype=t_pr.dtype)
    num = (w.unsqueeze(-1) * (t_pr * t_gt)).sum(dim=(1, 2))              # [B]
    den = (w.unsqueeze(-1) * (t_pr * t_pr)).sum(dim=(1, 2)) + eps        # [B]
    s = num / den                                                        # [B]

    diff = (s[:, None, None] * t_pr) - t_gt                               # [B,P,3]
    trans_l2_b = diff.norm(dim=-1).mean(dim=1)                            # [B]

    return trans_l2_b, s

def rotation_angle(R_gt: torch.Tensor, R_pr: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Geodesic rotation error between rotation matrices.

    Args:
        R_gt: [..., 3, 3]
        R_pr: [..., 3, 3]

    Returns:
        angles: [...] in radians
    """
    # relative rotation
    R_rel = R_gt.transpose(-1, -2) @ R_pr

    # trace
    tr = R_rel.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

    # numerical safety
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)

    return torch.acos(cos_theta)

@torch.no_grad
def validation_statistics(
    se3,
    dataloader,
    k=4,
    N=1,
    device="cuda",
    thresholds=(5, 10, 15, 30),
    guidance = False, 
):
    se3.model.eval()

    thresholds = tuple(thresholds)

    # Accumulate pooled errors across ALL draws (Pi3-style global pool)
    all_r = []
    all_t = []

    # Also compute per-draw metrics so we can report mean/std across draws
    per_draw_metrics = []  # list of dicts: {"RRA@5":..., "AUC@30":..., ...}

    drawn_batches = 0
    for step, batch in enumerate(dataloader):
        if step >= 20:
          break

        imgs = batch["imgs"].to(device, non_blocking=True)  # [B,V,C,H,W]
        Ks   = batch["K"].to(device, non_blocking=True)

        R_gt = batch["R"].to(device, non_blocking=True)     # [B,V,3,3]
        T_gt = batch["T"].to(device, non_blocking=True)     # [B,V,3]
        gt_c2w = pack_se3_c2w(R_gt, T_gt)                   # [B,V,4,4]
        gt_w2c = se3_inverse(gt_c2w)                        # [B,V,4,4]

        num_views = batch.get("num_views", None)            # [B] or None
        B, V = imgs.shape[:2]
        objective = {"gt_poses": gt_c2w}

        for n in range(N):
            sample_out = se3.sample(imgs=imgs, Ks=Ks, guidance = False, optim_steps = 1000, objective = objective)
            pred_c2w = pack_se3_c2w(sample_out["R"], sample_out["t"])  # [B,V,4,4]
            pred_w2c = se3_inverse(pred_c2w)

            # Collect rr/tt for THIS draw (pool across items)
            draw_r_list = []
            draw_t_list = []

            for b in range(B):
                v = int(num_views[b]) if num_views is not None else V
                dev = pred_w2c.device
                pred = pred_w2c[b, :v].to(dev)
                gt   = gt_w2c[b, :v].to(dev)

                rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
                    pred_se3=pred,   # [v,4,4]
                    gt_se3=gt,       # [v,4,4]
                    num_frames=v,
                )
                draw_r_list.append(rel_rangle_deg.detach().cpu().numpy().reshape(-1))
                draw_t_list.append(rel_tangle_deg.detach().cpu().numpy().reshape(-1))

            draw_r = np.concatenate(draw_r_list, axis=0)
            draw_t = np.concatenate(draw_t_list, axis=0)

            all_r.append(draw_r)
            all_t.append(draw_t)

            # Compute per-draw metrics at thresholds
            m = {}
            for thr in thresholds:
                m[f"RRA@{thr}"] = float((draw_r < thr).mean() * 100.0)
                m[f"RTA@{thr}"] = float((draw_t < thr).mean() * 100.0)
                auc, _ = calculate_auc_np(draw_r, draw_t, max_threshold=thr)
                m[f"AUC@{thr}"] = float(auc * 100.0)
            per_draw_metrics.append(m)

        drawn_batches += 1

    # ---- pooled metrics (like Pi3 eval, over all draws/items/pairs) ----
    rError = np.concatenate(all_r, axis=0) if len(all_r) else np.array([])
    tError = np.concatenate(all_t, axis=0) if len(all_t) else np.array([])

    metrics = {}
    for thr in thresholds:
        metrics[f"RRA@{thr}"] = float((rError < thr).mean() * 100.0)
        metrics[f"RTA@{thr}"] = float((tError < thr).mean() * 100.0)
        auc, _ = calculate_auc_np(rError, tError, max_threshold=thr)
        metrics[f"AUC@{thr}"] = float(auc * 100.0)

    # ---- draw mean/std (uncertainty across stochastic samples) ----
    draw_stats = {}
    if len(per_draw_metrics):
        keys = list(per_draw_metrics[0].keys())
        for kkey in keys:
            vals = np.array([d[kkey] for d in per_draw_metrics], dtype=np.float64)
            draw_stats[f"{kkey}_mean_over_draws"] = float(vals.mean())
            draw_stats[f"{kkey}_std_over_draws"]  = float(vals.std(ddof=0))

    return {
        "pooled": metrics,
        "draw_stats": draw_stats,
        "num_draws": len(per_draw_metrics),
        "num_pairs_total": int(rError.size),
    }

@torch.no_grad()
def plot_so3_probabilities_full(
    se3,
    dataloader,
    plot_name,
    conditioning,                       # "depths" or "features" (same as visualize_pose_axes_full)
    k=4,                                # number of dataset entries
    N=50,                               # draws per entry
    device="cuda",
    display_threshold_probability=0.0,
    show_color_wheel=True,
    canonical_rotation=np.eye(3),
):
    """
    Wrapper around `visualize_so3_probabilities` that:
      • samples relative rotations from `se3` over k entries × N draws,
      • picks the permutation closer to GT (rotation-only) for plotting,
      • uses uniform probabilities,
      • saves the figure to `plot_name` (no metrics, no prints except the save path).

    Notes:
      - Expects dataloader batches with keys: "R1","t1","R2","t2" plus
        either ("depths","rgb") or ("feats","rgb") depending on `conditioning`.
      - Leaves `visualize_so3_probabilities` untouched.
    """

    # --- helpers (same math as your other viz) ---
    def rel_from_two(R1, t1, R2, t2):
        """Pose of (R2,t2) expressed in frame of (R1,t1)."""
        R_rel = R1.transpose(-2, -1) @ R2                    # R1^T R2
        t_rel = (R1.transpose(-2, -1) @ (t2 - t1).unsqueeze(-1)).squeeze(-1)  # R1^T (t2 - t1)
        return R_rel, t_rel

    def rot_geodesic_deg(Ra, Rb):
        R = Ra.transpose(-2, -1) @ Rb
        tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        cos_th = 0.5 * (tr - 1.0)
        cos_th = torch.clamp(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cos_th) * (180.0 / np.pi)

    sampled_rel_rots = []
    gt_rel_rots = []
    drawn = 0

    for batch in dataloader:
        if drawn >= k:
            break

        take = min(k - drawn, batch["R1"].shape[0])
        for b in range(take):
            # --- GT for this entry ---
            R1 = batch["R1"][b].to(device)  # [3,3]
            t1 = batch["t1"][b].to(device)  # [3]
            R2 = batch["R2"][b].to(device)
            t2 = batch["t2"][b].to(device)
            R_rel_gt, _ = rel_from_two(R1, t1, R2, t2)

            # --- conditioning payload (mirrors visualize_pose_axes_full) ---
            if conditioning == "depths":
                depths = batch["depths"][b].to(device)
                rgb    = batch["rgb"][b].to(device)
                if rgb.ndim == 3:  # [C,H,W] where C=3*N
                    C, H, W = rgb.shape
                    assert C % 3 == 0
                    Nviews = C // 3
                    rgb = rgb.view(Nviews, 3, H, W).unsqueeze(0)  # [1,N,3,H,W]
                elif rgb.ndim == 4:  # [N,3,H,W]
                    rgb = rgb.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected rgb shape: {rgb.shape}")
                sample_args = {"depths": depths, "rgb": rgb}

            elif conditioning == "features":
                rgb   = batch["rgb"][b]
                feats = batch["feats"][b]
                sample_args = {"feats": feats.to(device), "rgb": rgb.to(device)}
            else:
                raise ValueError(f"Unsupported conditioning: {conditioning}")

            # --- N draws; choose permutation closer to GT (rotation-only) ---
            for _ in range(N):
                R_pair, t_pair = se3.sample(**sample_args, B=1, N=2, guidance=False, optim_steps=1, cost=None)
                R0, R1p = R_pair[0], R_pair[1]
                t0, t1p = t_pair[0], t_pair[1]

                R_rel_01, _ = rel_from_two(R0,  t0,  R1p, t1p)
                R_rel_10, _ = rel_from_two(R1p, t1p, R0,  t0)

                ang01 = rot_geodesic_deg(R_rel_01, R_rel_gt).item()
                ang10 = rot_geodesic_deg(R_rel_10, R_rel_gt).item()
                R_rel_pred = R_rel_01 if ang01 <= ang10 else R_rel_10

                sampled_rel_rots.append(R_rel_pred.detach().cpu().numpy())

            # one GT marker per entry (nice for context)
            gt_rel_rots.append(R_rel_gt.detach().cpu().numpy())

            drawn += 1
            if drawn >= k:
                break

    if len(sampled_rel_rots) == 0:
        raise RuntimeError("No samples collected; check dataloader/conditioning.")

    rotations    = np.stack(sampled_rel_rots, axis=0)     # [M,3,3], M=k*N
    rotations_gt = np.stack(gt_rel_rots,    axis=0)       # [k,3,3]
    probabilities = np.full((rotations.shape[0],), 1.0 / rotations.shape[0], dtype=np.float64)

    # --- call the existing visualizer unchanged ---
    fig = visualize_so3_probabilities(
        rotations=rotations,
        probabilities=probabilities,
        rotations_gt=rotations_gt,
        to_image=False,  # get a matplotlib Figure back to save
        show_color_wheel=show_color_wheel,
        canonical_rotation=canonical_rotation
    )

    # --- save & close ---
    out_path = Path(plot_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"Saved SO(3) probability plot to {out_path.resolve()}")

def visualize_so3_probabilities(rotations,
                                probabilities,
                                rotations_gt=None,
                                ax=None,
                                fig=None,
                                display_threshold_probability=0,
                                to_image=True,
                                show_color_wheel=True,
                                canonical_rotation=np.eye(3)):
  """Plot a single distribution on SO(3) using the tilt-colored method.

  Args:
    rotations: [N, 3, 3] tensor of rotation matrices
    probabilities: [N] tensor of probabilities
    rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
    ax: The matplotlib.pyplot.axis object to paint
    fig: The matplotlib.pyplot.figure object to paint
    display_threshold_probability: The probability threshold below which to omit
      the marker
    to_image: If True, return a tensor containing the pixels of the finished
      figure; if False return the figure itself
    show_color_wheel: If True, display the explanatory color wheel which matches
      color on the plot with tilt angle
    canonical_rotation: A [3, 3] rotation matrix representing the 'display
      rotation', to change the view of the distribution.  It rotates the
      canonical axes so that the view of SO(3) on the plot is different, which
      can help obtain a more informative view.

  Returns:
    A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
  """
  def _show_single_marker(ax, rotation, marker, edgecolors=True,
                          facecolors=False):
    eulers = tfg.euler.from_rotation_matrix(rotation)
    xyz = rotation[:, 0]
    tilt_angle = eulers[0]
    longitude = np.arctan2(xyz[0], -xyz[1])
    latitude = np.arcsin(xyz[2])

    color = cmap(0.5 + tilt_angle / 2 / np.pi)
    ax.scatter(longitude, latitude, s=2500,
               edgecolors=color if edgecolors else 'none',
               facecolors=facecolors if facecolors else 'none',
               marker=marker,
               linewidth=4)

  if ax is None:
    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_subplot(111, projection='mollweide')
  if rotations_gt is not None and len(tf.shape(rotations_gt)) == 2:
    rotations_gt = rotations_gt[tf.newaxis]

  display_rotations = rotations @ canonical_rotation
  cmap = plt.cm.hsv
  scatterpoint_scaling = 4e3
  eulers_queries = tfg.euler.from_rotation_matrix(display_rotations)
  xyz = display_rotations[:, :, 0]
  tilt_angles = eulers_queries[:, 0]

  longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
  latitudes = np.arcsin(xyz[:, 2])

  which_to_display = (probabilities > display_threshold_probability)

  if rotations_gt is not None:
    # The visualization is more comprehensible if the GT
    # rotation markers are behind the output with white filling the interior.
    display_rotations_gt = rotations_gt @ canonical_rotation

    for rotation in display_rotations_gt:
      _show_single_marker(ax, rotation, 'o')
    # Cover up the centers with white markers
    for rotation in display_rotations_gt:
      _show_single_marker(ax, rotation, 'o', edgecolors=False,
                          facecolors='#ffffff')

  # Display the distribution
  ax.scatter(
      longitudes[which_to_display],
      latitudes[which_to_display],
      s=scatterpoint_scaling * probabilities[which_to_display],
      c=cmap(0.5 + tilt_angles[which_to_display] / 2. / np.pi))

  ax.grid()
  ax.set_xticklabels([])
  ax.set_yticklabels([])

  if show_color_wheel:
    # Add a color wheel showing the tilt angle to color conversion.
    ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
    theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
    radii = np.linspace(0.4, 0.5, 2)
    _, theta_grid = np.meshgrid(radii, theta)
    colormap_val = 0.5 + theta_grid / np.pi / 2.
    ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
    ax.set_yticklabels([])
    ax.set_xticklabels([r'90$\degree$', None,
                        r'180$\degree$', None,
                        r'270$\degree$', None,
                        r'0$\degree$'], fontsize=14)
    ax.spines['polar'].set_visible(False)
    plt.text(0.5, 0.5, 'Tilt', fontsize=14,
             horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)


    return fig