#!/usr/bin/env python
# coding: utf-8

# In[2]:
#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import sys
from SO3n import so3_diffuser, SO3Algebra
from R3n import r3_diffuser
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import math
import math
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# In[9]:


#This class defines various loss functions which are useful to have seperate from the diffuser class

def loss_pose(diffuser, R_clean, T_clean, X, T_0, t, step = 0): 
    """
    Compute rotation loss between predicted and ground-truth rotation matrices.

    Args:
        R_clean: [B, N, 3, 3] - ground truth rotation matrices
        X      : [B, N, 3]    - predicted 3D vectors to be interpreted as rotation
        t      : unused, but kept for interface consistency

    Returns:
        Scalar rotation loss (mean geodesic distance in radians)
    """
    B, N = X.shape[:2]


    # Convert X -> quaternions -> rotation matrices
    X_flat = X.reshape(B * N, 3)  # [B*N, 3]
    q = diffuser.so3.alg.make_unit_quaternion(X_flat)         # [B*N, 4]
    R_pred = diffuser.so3.alg.quaternion_to_rotmat(q)         # [B*N, 3, 3]
    R_pred = R_pred.reshape(B, N, 3, 3)            # [B, N, 3, 3]

    # Compute geodesic distance between R_pred and R_clean
    R_rel = torch.matmul(R_pred.transpose(-2, -1), R_clean)  # [B, N, 3, 3]
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]  # [B, N]

    # Clamp for numerical stability
    cos_theta = 0.5 * (trace - 1.0)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)

    loss_rot = .5 * torch.acos(cos_theta).mean()  # Mean over all B and N
    loss_trans = .5 * ((T_clean - T_0) ** 2).mean()
    return loss_rot + loss_trans

def loss_relative_pose(diffuser, R_clean, T_clean, X, T_0, R_t, ts):
    """
    Compute relative pose loss between predicted and true pose pairs (active convention).

    Args:
        R_clean: [B, 2, 3, 3] - GT rotation matrices for views 1 and 2
        T_clean: [B, 2, 3]    - GT translations
        X: [B, 2, 3]          - Predicted 3D vectors (to be converted to quats)
        T_0: [B, 2, 3]        - Predicted translations
        t: unused
    Returns:
        Scalar loss for relative pose: rotation + translation
    """
    B = X.shape[0]

    # Convert predicted 3D -> quaternion -> rotation matrix
    q = diffuser.so3.alg.make_unit_quaternion(X.reshape(B * 2, 3))             # [B*2, 4]
    R_pred = diffuser.so3.alg.quaternion_to_rotmat(q).reshape(B, 2, 3, 3)      # [B, 2, 3, 3]

    # Ground-truth relative pose (active)
    R1_gt, R2_gt = R_clean[:, 0], R_clean[:, 1]                    # [B, 3, 3]
    T1_gt, T2_gt = T_clean[:, 0], T_clean[:, 1]                    # [B, 3]

    R_gt_rel = torch.matmul(R2_gt, R1_gt.transpose(-2, -1))        # [B, 3, 3]
    T_gt_rel = -torch.matmul(R2_gt, torch.matmul(R1_gt.transpose(-2, -1), T1_gt.unsqueeze(-1))).squeeze(-1) + T2_gt  # [B, 3]

    # Predicted relative pose (active)
    R1_pred, R2_pred = R_pred[:, 0], R_pred[:, 1]
    T1_pred, T2_pred = T_0[:, 0], T_0[:, 1]

    R_pred_rel = torch.matmul(R2_pred, R1_pred.transpose(-2, -1))
    T_pred_rel = -torch.matmul(R2_pred, torch.matmul(R1_pred.transpose(-2, -1), T1_pred.unsqueeze(-1))).squeeze(-1) + T2_pred

    # Rotation geodesic loss
    R_rel_diff = torch.matmul(R_pred_rel.transpose(-2, -1), R_gt_rel)
    trace = R_rel_diff[..., 0, 0] + R_rel_diff[..., 1, 1] + R_rel_diff[..., 2, 2]
    cos_theta = 0.5 * (trace - 1.0)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    loss_rot = torch.acos(cos_theta).mean()

    # Translation loss
    loss_trans = ((T_pred_rel - T_gt_rel) ** 2).mean()

    return loss_rot + loss_trans

def loss_score_alt(
    diffuser,
    R_clean: torch.Tensor,         # [B,N,3,3]
    T_clean: torch.Tensor,         # [B,N,3]
    X: torch.Tensor,               # [B,N,3]  (map to SO(3))
    T_pred: torch.Tensor,          # [B,N,3]
    R_t: torch.Tensor,             # [B,N,3,3]
    T_t: torch.Tensor,             # [B,N,3]
    t: torch.Tensor,               # [B,N] discrete steps (expanded along N)
    rot_threshold: int = 5,
    trans_threshold: int = 0,
    w_score: float = 1.0,
):
    device, dtype = R_clean.device, R_clean.dtype
    B, N = X.shape[:2]
    assert t.shape == (B, N), f"t must be [B,N], got {t.shape}"

    # continuous taus (per-element)
    tau = (t.to(torch.float32) / 1000.0).clamp(0.0, 1.0)      # [B,N]
    tau_flat = tau.reshape(-1)                                 # [B*N]

    # --- map X -> R_pred ---
    q = diffuser.so3.alg.make_unit_quaternion(X.reshape(B*N, 3))
    R_pred = diffuser.so3.alg.quaternion_to_rotmat(q).reshape(B, N, 3, 3)

    # --- rotation pose loss map [B,N] ---
    R_rel  = torch.einsum('...ij,...jk->...ik', R_pred.transpose(-1, -2), R_clean)
    tr     = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_th = torch.clamp(0.5 * (tr - 1.0), -1.0 + 1e-6, 1.0 - 1e-6)
    theta  = torch.arccos(cos_th)
    rot_pose_map = 0.5 * theta                                  # [B,N]

    # --- rotation score loss map [B,N] (vectorized) ---
    Rt_flat, Rc_flat, Rp_flat = R_t.reshape(-1,3,3), R_clean.reshape(-1,3,3), R_pred.reshape(-1,3,3)
    s_true_rot = diffuser.so3.compute_score(Rt_flat, Rc_flat, tau_flat)   # [B*N,3]
    s_pred_rot = diffuser.so3.compute_score(Rt_flat, Rp_flat, tau_flat)   # [B*N,3]
    se_rot = (torch.as_tensor(s_true_rot, device=device, dtype=dtype)
            -torch.as_tensor(s_pred_rot, device=device, dtype=dtype)).pow(2).sum(dim=-1)  # [B*N]
    scale_rot = diffuser.so3.score_scaling(tau_flat.detach().cpu().numpy())                # [B*N]
    scale_rot = torch.as_tensor(scale_rot, device=device, dtype=dtype).clamp_min(1e-12)
    rot_score_map = (se_rot / (scale_rot**2)).view(B, N)                                    # [B,N]

    # --- translation pose loss map [B,N] ---
    trans_pose_map = 0.5 * (T_clean - T_pred).pow(2).sum(dim=-1)                            # [B,N]

    # --- translation score loss map [B,N] (vectorized) ---
    s_true_trans = diffuser.r3.score(T_t, T_clean, tau, use_torch=True, scale=False)        # [B,N,3]
    s_pred_trans = diffuser.r3.score(T_t, T_pred,  tau, use_torch=True, scale=False)        # [B,N,3]
    se_trans = (s_true_trans - s_pred_trans).pow(2).sum(dim=-1)                             # [B,N]
    scale_trans = diffuser.r3.score_scaling(tau_flat.detach().cpu().numpy())                # [B*N]
    scale_trans = torch.as_tensor(scale_trans, device=device, dtype=dtype).clamp_min(1e-12).view(B,N)
    trans_score_map = se_trans / (scale_trans**2)                                           # [B,N]

    # --- threshold gating (per element) ---
    rot_pose_mask    = (t <= int(rot_threshold))
    rot_score_mask   = ~rot_pose_mask
    trans_pose_mask  = (t <= int(trans_threshold))
    trans_score_mask = ~trans_pose_mask

    def mmean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.to(x.dtype)
        denom = m.sum().clamp_min(1.0)
        return (x * m).sum() / denom

    loss_rot   = mmean(rot_pose_map,   rot_pose_mask)   #+ w_score * mmean(rot_score_map,   rot_score_mask)
    loss_trans = mmean(trans_pose_map, trans_pose_mask) #+ w_score * mmean(trans_score_map, trans_score_mask)

    return loss_rot + loss_trans
    
def loss_score(
    diffuser,
    R_clean: torch.Tensor,         # [B,N,3,3] GT rotations
    T_clean: torch.Tensor,         # [B,N,3]   GT translations
    X: torch.Tensor,               # [B,N,3]   model rotation output (to be mapped to SO(3))
    T_pred: torch.Tensor,          # [B,N,3]   model translation output
    R_t: torch.Tensor,             # [B,N,3,3] noisy rotations at time t
    T_t: torch.Tensor,             # [B,N,3]   noisy translations at time t
    t: float | torch.Tensor,       # scalar (preferred) or any shape -> treated as ONE global t
    step: int = 0,
    w_score: float = 1.0,
    loss_mask: torch.Tensor | None = None,   # [B,N] mask (1 keeps, 0 drops)
) -> dict:
    """
    Batch loss with a single global timestep t:
      - rotation geodesic loss
      - translation MSE
      - rotation score loss (from predicted x0 vs true x0), using so3.compute_score
      - translation score loss (from predicted x0 vs true x0), using euclidean compute_score
    """
    device, dtype = R_clean.device, R_clean.dtype
    B, N = X.shape[:2]

    # ---- extract scalar t ----
    if torch.is_tensor(t):
        t_scalar_raw = float(t.reshape(-1)[0].item())
    else:
        t_scalar_raw = float(t)

    # tau in [0,1] for SO(3) calls, and discrete t_step for Euclidean compute_score
    if hasattr(diffuser, "T") and diffuser.T:
        Ttot = int(diffuser.T)
    else:
        Ttot = 1000  # sensible default for your euclidean compute_score

    # If user passed a discrete step (>=1), map to [0,1] for tau; otherwise assume it's already [0,1]
    tau = t_scalar_raw / float(Ttot) if t_scalar_raw >= 1.0 else t_scalar_raw
    tau = max(0.0, min(1.0, tau))

    # Make the integer step for Euclidean compute_score (expects integer grid 0..1000)
    t_step = int(round(tau * Ttot))
    t_b = torch.full((B, N), t_step, device=device, dtype=torch.long)  # [B,N]

    # ---- map X -> R_pred (SO(3)) ----
    q = diffuser.so3.alg.make_unit_quaternion(X.reshape(B * N, 3))             # [B*N,4]
    R_pred = diffuser.so3.alg.quaternion_to_rotmat(q).reshape(B, N, 3, 3)      # [B,N,3,3]

    # ---- rotation geodesic loss ----
    R_rel  = torch.einsum('...ij,...jk->...ik', R_pred.transpose(-1, -2), R_clean)
    trace  = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_th = torch.clamp(0.5 * (trace - 1.0), -1.0 + 1e-6, 1.0 - 1e-6)
    theta  = torch.arccos(cos_th)  # [B,N]
    if loss_mask is not None:
        theta = theta * loss_mask
    loss_rot = 0.5 * theta.mean()

    # ---- translation MSE ----
    diff_T = T_clean - T_pred  # [B,N,3]
    if loss_mask is not None:
        diff_T = diff_T * loss_mask[..., None]
    loss_trans = 0.5 * (diff_T ** 2).mean()

    # ---- rotation score loss (SO3) ----
    Rt_flat = R_t.reshape(B * N, 3, 3)
    Rc_flat = R_clean.reshape(B * N, 3, 3)
    Rp_flat = R_pred.reshape(B * N, 3, 3)

    with torch.no_grad():
        sR_true = diffuser.so3.compute_score(Rt_flat, Rc_flat, tau)   # [B*N,3]
    sR_pred = diffuser.so3.compute_score(Rt_flat, Rp_flat, tau)       # [B*N,3]

    sR_true = sR_true.to(device=device, dtype=dtype).reshape(B, N, 3)
    sR_pred = sR_pred.to(device=device, dtype=dtype).reshape(B, N, 3)

    if loss_mask is not None:
        sR_true = sR_true * loss_mask[..., None]
        sR_pred = sR_pred * loss_mask[..., None]

    # rotation scaling (EDM-ish tempering)
    scale_R_np = diffuser.so3.score_scaling(np.array([tau]))  # expects tau in [0,1]
    scale_R = torch.as_tensor(scale_R_np[0], device=device, dtype=dtype).clamp_min(1e-12)

    loss_score_rot = (((sR_true - sR_pred) ** 2) / (scale_R ** 2)).sum(dim=-1).mean()

    # ---- translation score loss (Euclidean) ----
    # Uses your batch-capable compute_score(x_t, t=[B,N], x0)
    with torch.no_grad():
        sT_true = diffuser.r3.compute_score(T_t, t_b, T_clean)   # [B,N,3]
    sT_pred = diffuser.r3.compute_score(T_t, t_b, T_pred)        # [B,N,3]

    sT_true = sT_true.to(device=device, dtype=dtype)
    sT_pred = sT_pred.to(device=device, dtype=dtype)

    if loss_mask is not None:
        sT_true = sT_true * loss_mask[..., None]
        sT_pred = sT_pred * loss_mask[..., None]

    # translation scaling: try diffuser.score_scaling(tau); fall back to rotation scale if not present
    try:
        scale_T_np = diffuser.score_scaling(np.array([tau]))
        scale_T = torch.as_tensor(scale_T_np[0], device=device, dtype=dtype).clamp_min(1e-12)
    except AttributeError:
        scale_T = scale_R  # reuse rotation scale if a separate one isn't defined
    #print(scale_T)
    loss_score_trans = (((sT_true - sT_pred) ** 2) / (scale_T ** 2)).sum(dim=-1).mean()

    # ---- combine ----
    loss_score_total = loss_score_rot #+ loss_score_trans
    total = loss_trans + w_score * loss_score_total

    return total

# %%


# In[ ]:




