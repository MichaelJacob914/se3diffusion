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
from se3n_utils import rot_trans_from_se3
from torchlie.functional import SE3

#This class defines various loss functions which are useful to have seperate from the diffuser class
def _to_torch(x, ref):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=ref.device, dtype=ref.dtype)

def diffusion_loss_relative_pose(diffuser, R_clean, T_clean, R_pred, T_pred, t, eps=1e-8, lambda_dir=1.0):
    """
    Time-weighted relative-pose loss with chordal (Frobenius) rotation term
    and full-vector relative translation differences (scene normalized).

    Inputs:
      R_* : camera->world rotations  [B,N,3,3]
      T_* : camera->world translations [B,N,3] 
      t   : timesteps [B,N] 


    Rot loss:   0.5 * E_pairs[ wR * || R_err - I ||_F^2 ],  R_err = R_pred_rel^T R_clean_rel
    Trans loss: 0.5 * E_pairs[ wT * || R_i^T (T_j - T_i) - R_i^T (T_j - T_i) ||_2^2 ] 
    """

    B, N = T_clean.shape[:2]
    device = R_clean.device

    # ---- relative rotations ----
    R_clean_iT = R_clean.transpose(-2, -1)
    R_pred_iT  = R_pred .transpose(-2, -1)

    R_clean_rel = R_clean_iT.unsqueeze(2) @ R_clean.unsqueeze(1)   # [B,N,N,3,3]
    R_pred_rel  = R_pred_iT .unsqueeze(2) @ R_pred .unsqueeze(1)   # [B,N,N,3,3]

    R_err   = R_pred_rel.transpose(-2, -1) @ R_clean_rel           # [B,N,N,3,3]
    R_err_I = R_err - torch.eye(3, device=device, dtype=R_err.dtype)
    rot_pair = (R_err_I ** 2).sum(dim=(-2, -1))                    # [B,N,N]

    trace   = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]

    cos_th  = 0.5 * (trace - 1.0)
    cos_th  = torch.clamp(cos_th, -1.0 + 1e-6, 1.0 - 1e-6)
    rot_pair_geodesic = torch.acos(cos_th)                          # [B,N,N]

    # ---- relative translations in local frames ----
    dT_clean = T_clean.unsqueeze(2) - T_clean.unsqueeze(1)          # [B,N,N,3]
    dT_pred  = T_pred.unsqueeze(2) - T_pred.unsqueeze(1)          # [B,N,N,3]

    t_clean_rel = (R_clean.transpose(-2,-1).unsqueeze(2) @ dT_clean.unsqueeze(-1)).squeeze(-1)  # [B,N,N,3]
    t_pred_rel  = (R_pred .transpose(-2,-1).unsqueeze(2) @ dT_pred .unsqueeze(-1)).squeeze(-1)  # [B,N,N,3]

    # ---- mask out diagonal pairs ----
    offdiag = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1)
    rot_flat   = rot_pair .masked_select(offdiag)
    rot_flat_geodesic = rot_pair_geodesic.masked_select(offdiag)

    # ---- timestep-dependent weights (per-B) ----
    t_b = t[:, 0].to(dtype=torch.float32)         # [B]
    t_b = t_b.clamp(0.01, 1.0)

    sR = _to_torch(diffuser.so3.score_scaling(t_b), R_clean)  # [B]
    sT = _to_torch(diffuser.r3.score_scaling(t_b),  R_clean)  # [B]
    
    wR_B = (sR).view(B, 1, 1).expand(B, N, N)
    wT_B = (sT).view(B, 1, 1).expand(B, N, N)

    wR = wR_B.masked_select(offdiag) + eps
    wT = wT_B.masked_select(offdiag) + eps

    # flatten pairs per batch element
    tp = t_pred_rel[offdiag].reshape(B, -1, 3)   # [B, M, 3]
    tg = t_clean_rel[offdiag].reshape(B, -1, 3)  # [B, M, 3]

    # timestep weights per pair 
    wT_pairs = wT.reshape(B, -1)                 # [B, M]

    # compute s* (per batch element)
    num = ((tp * tg)).sum(dim=(1,2))          # [B]
    den = ((tp * tp)).sum(dim=(1,2)) + eps    # [B]
    s_star = (num / den)                                                # [B]
    s_star = s_star.detach()#.clamp(-10.0, 10.0)

    tp_scaled = tp * s_star[:, None, None]

    tg_dir = tg / (tg.norm(dim=-1, keepdim=True) + eps)                # [B,M,3]
    tp_dir = tp / (tp.norm(dim=-1, keepdim=True) + eps)  # [B,M,3]

    dir_per_pair = 1.0 - (tp_dir * tg_dir).sum(dim=-1)                 # [B,M]

    loss_trans_dir = (wT_pairs * dir_per_pair).sum() / (wT_pairs.sum() + eps)
    loss_trans_dir = lambda_dir * loss_trans_dir

    delta = 0.1
    per_pair = torch.nn.functional.smooth_l1_loss(tp_scaled, tg, beta=delta, reduction="none").sum(dim=-1)  # [B,M]
    loss_trans = 0.5 * (wT_pairs * per_pair).sum() / (wT_pairs.sum() + eps)

    # ---- weighted means ----
    loss_rot   = 0.5 * (wR * rot_flat).sum()   / (wR.sum() + eps)
    loss_rot_geodesic = 0.5 * (wR * rot_flat_geodesic).sum() / (wR.sum() + eps)

    return loss_rot, loss_rot_geodesic, loss_trans, loss_trans_dir

def regression_loss_relative_pose(
    regresser,
    R_clean, T_clean, R_pred, T_pred,
    eps=1e-8,
    lambda_dir=0.0,
    delta=0.1,
):
    """
    Scale-invariant relative translation loss + relative rotation losses.
    No diffusion/timestep weighting.
    """

    B, N = T_clean.shape[:2]
    device = R_clean.device

    # ---- relative rotations ----
    R_clean_rel = R_clean.transpose(-2, -1).unsqueeze(2) @ R_clean.unsqueeze(1)  # [B,N,N,3,3]
    R_pred_rel  = R_pred .transpose(-2, -1).unsqueeze(2) @ R_pred .unsqueeze(1)  # [B,N,N,3,3]

    R_err = R_pred_rel.transpose(-2, -1) @ R_clean_rel  # [B,N,N,3,3]

    I = torch.eye(3, device=device, dtype=R_err.dtype)
    rot_pair_frob = ((R_err - I) ** 2).sum(dim=(-2, -1))  # [B,N,N]

    trace = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]
    cos_th = torch.clamp(0.5 * (trace - 1.0), -1.0 + 1e-6, 1.0 - 1e-6)
    rot_pair_geo = torch.acos(cos_th)  # [B,N,N]

    # ---- relative translations in local frames ----
    dT_clean = T_clean.unsqueeze(2) - T_clean.unsqueeze(1)  # [B,N,N,3]
    dT_pred  = T_pred .unsqueeze(2) - T_pred .unsqueeze(1)  # [B,N,N,3]

    t_clean_rel = (R_clean.transpose(-2, -1).unsqueeze(2) @ dT_clean.unsqueeze(-1)).squeeze(-1)  # [B,N,N,3]
    t_pred_rel  = (R_pred .transpose(-2, -1).unsqueeze(2) @ dT_pred .unsqueeze(-1)).squeeze(-1)  # [B,N,N,3]

    # ---- mask out diagonal ----
    offdiag = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)  # [1,N,N]
    offdiag = offdiag.expand(B, -1, -1)                                    # [B,N,N]

    # flatten pairs per batch element: M = N*(N-1)
    tp = t_pred_rel[offdiag].reshape(B, -1, 3)   # [B,M,3]
    tg = t_clean_rel[offdiag].reshape(B, -1, 3)  # [B,M,3]

    # ---- scale fit s* (uniform weights) ----
    # s = argmin_s || s*tp - tg ||^2  =>  s = <tp,tg>/<tp,tp>
    num = (tp * tg).sum(dim=(1, 2))                      # [B]
    den = (tp * tp).sum(dim=(1, 2)) + eps                # [B]
    s_star = (num / den).clamp(min=0.0)      # [B]  (detach recommended)

    tp_scaled = tp * s_star[:, None, None]

    # scale-invariant robust loss
    per_pair = F.smooth_l1_loss(tp_scaled, tg, beta=delta, reduction="none").sum(dim=-1)  # [B,M]
    loss_trans = 0.5 * per_pair.mean()

    # optional direction loss (scale-free)
    if lambda_dir > 0.0:
        tp_dir = tp / (tp.norm(dim=-1, keepdim=True) + eps)
        tg_dir = tg / (tg.norm(dim=-1, keepdim=True) + eps)
        # 1 - cosine similarity
        dir_pair = 1.0 - (tp_dir * tg_dir).sum(dim=-1)   # [B,M]
        loss_trans_dir = lambda_dir * dir_pair.mean()

    # rotation losses
    loss_rot = 0.5 * rot_pair_frob[offdiag].mean()
    loss_rot_geodesic = 0.5 * rot_pair_geo[offdiag].mean()

    return loss_rot, loss_rot_geodesic, loss_trans, loss_trans_dir


def loss_relative_pose(
    G_clean, G_pred, 
    eps=1e-8,
    lambda_dir=0.2,
    delta=0.0,
):
    """
    Scale-invariant relative translation loss + relative rotation losses.
    No diffusion/timestep weighting.
    """
    R_clean, T_clean = rot_trans_from_se3(G_clean)
    R_pred, T_pred =  rot_trans_from_se3(G_pred)
    B, N = T_clean.shape[:2]
    device = R_clean.device
    # ---- relative rotations ----
    R_clean_rel = R_clean.transpose(-2, -1).unsqueeze(2) @ R_clean.unsqueeze(1)  # [B,N,N,3,3]
    R_pred_rel  = R_pred .transpose(-2, -1).unsqueeze(2) @ R_pred .unsqueeze(1)  # [B,N,N,3,3]

    R_err = R_pred_rel.transpose(-2, -1) @ R_clean_rel  # [B,N,N,3,3]

    I = torch.eye(3, device=device, dtype=R_err.dtype)
    rot_pair_frob = ((R_err - I) ** 2).sum(dim=(-2, -1))  # [B,N,N]

    trace = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]
    cos_th = torch.clamp(0.5 * (trace - 1.0), -1.0 + 1e-6, 1.0 - 1e-6)
    rot_pair_geo = torch.acos(cos_th)  # [B,N,N]

    # ---- relative translations in local frames ----
    dT_clean = T_clean.unsqueeze(2) - T_clean.unsqueeze(1)  # [B,N,N,3]
    dT_pred  = T_pred .unsqueeze(2) - T_pred .unsqueeze(1)  # [B,N,N,3]

    t_clean_rel = (R_clean.transpose(-2, -1).unsqueeze(2) @ dT_clean.unsqueeze(-1)).squeeze(-1)  # [B,N,N,3]
    t_pred_rel  = (R_pred .transpose(-2, -1).unsqueeze(2) @ dT_pred .unsqueeze(-1)).squeeze(-1)  # [B,N,N,3]

    # ---- mask out diagonal ----
    offdiag = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)  # [1,N,N]
    offdiag = offdiag.expand(B, -1, -1)                                    # [B,N,N]

    # flatten pairs per batch element: M = N*(N-1)
    tp = t_pred_rel[offdiag].reshape(B, -1, 3)   # [B,M,3]
    tg = t_clean_rel[offdiag].reshape(B, -1, 3)  # [B,M,3]

    # ---- scale fit s* (uniform weights) ----
    # s = argmin_s || s*tp - tg ||^2  =>  s = <tp,tg>/<tp,tp>
    num = (tp * tg).sum(dim=(1, 2))                      # [B]
    den = (tp * tp).sum(dim=(1, 2)) + eps                # [B]
    s_star = (num / den).clamp(min=0.0)      # [B]  (detach recommended)
    #s_star = s_star.detach()

    tp_scaled = tp * s_star[:, None, None]

    # scale-invariant robust loss
    per_pair = F.smooth_l1_loss(tp_scaled, tg, beta=delta, reduction="none").sum(dim=-1)  # [B,M]
    loss_trans = 0.5 * per_pair.mean()

    # optional direction loss (scale-free)
    if lambda_dir > 0.0:
        tp_dir = tp / (tp.norm(dim=-1, keepdim=True) + eps)
        tg_dir = tg / (tg.norm(dim=-1, keepdim=True) + eps)
        # 1 - cosine similarity
        dir_pair = 1.0 - (tp_dir * tg_dir).sum(dim=-1)   # [B,M]
        loss_trans = loss_trans + lambda_dir * dir_pair.mean()

    # rotation losses
    loss_rot = 0.5 * rot_pair_frob[offdiag].mean()
    loss_rot_geodesic = 0.5 * rot_pair_geo[offdiag].mean()

    return loss_rot, loss_trans, loss_rot_geodesic

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
        Ttot = 1000  

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




