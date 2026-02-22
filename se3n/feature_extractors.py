import sys
from pathlib import Path
base = Path(__file__).resolve().parents[1]  # .../PARCC/scripts
sys.path[:0] = [str(base / "dust3r" / "croco"), str(base / "dust3r"), str(base / "pytorch3d_min")]
from dust3r.model        import load_model
from dust3r.utils.image  import load_images
from dust3r.image_pairs  import make_pairs

from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3

from torch.utils.data import BatchSampler

import os.path as osp
from dust3r.inference    import inference
import contextlib
from collections import defaultdict
from torch.amp import autocast
from itertools import combinations

import os, glob
# ...



from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
from typing import List, Dict, Tuple, Literal, Optional, Iterable
import os
import time
from mpl_toolkits.mplot3d import Axes3D  
from pytorch3d.loss import chamfer_distance
from typing import Union
import random

from torchvision.transforms import Resize, ToTensor, Compose
from tqdm import tqdm
import cv2 

from copy import deepcopy
from EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, reparameter
from EfficientLoFTR.src.utils.plotting import make_matching_figure
import matplotlib.cm as cm
import geoopt
from geoopt.optim import (RiemannianAdam)
import torch
import pytorch3d
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import numpy as np

import math
from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn
import geoopt
import cv2


sys.path.insert(0, "/vast/home/m/mgjacob/PARCC/scripts/theseus")
import theseus as th
from theseus.geometry.point_types import Point3 
from theseus.geometry.se3 import SE3


class Dust3rFeatureExtractor(torch.nn.Module):
    def __init__(self, ckpt_path: Path, device: str = "cuda"):
        super().__init__()
        self.model = load_model(str(ckpt_path), device=device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.device = torch.device(device)
        self.model.to(memory_format=torch.channels_last)
        
        # Cache amp dtype
        self.amp_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported()
            else torch.float16
        )

    @torch.inference_mode()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Expects img already on CUDA in channels_last format."""
        B, _, H, W = img.shape
        
        x_tok, x_pos = self.model.patch_embed(img)
        for blk in self.model.enc_blocks:
            x_tok = blk(x_tok, x_pos)
        x_tok = self.model.enc_norm(x_tok)
        
        H_, W_ = H // 16, W // 16
        feat = x_tok.transpose(1, 2).reshape(B, x_tok.shape[-1], H_, W_)
        return feat


@torch.no_grad()    
def compute_dust3r_feats(extractor, imgs):
    """imgs: [B, K, 3, H, W] -> [B, K, C, h, w]"""
    B, K, C, H, W = imgs.shape
    flat = imgs.reshape(B * K, C, H, W)  

    feats = extractor(flat) 

    return feats.reshape(B, K, *feats.shape[1:]).contiguous()

@torch.inference_mode()
def feat_from_tensor(img_tensor: torch.Tensor,
                     extractor: Dust3rFeatureExtractor,
                     device) -> torch.Tensor:
    return extractor(img_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
    
class Pi3FeatureExtractor(torch.nn.Module):
    def __init__(self, ckpt_path: Path, device: str = "cuda"):
        super().__init__()
        device = torch.device(device)

        model = Pi3().to(device).eval()

        ckpt_path = str(ckpt_path)
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            weight = load_file(ckpt_path)
        else:
            weight = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(weight)

        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.device = device
        self.model.to(memory_format=torch.channels_last)

        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        except Exception:
            pass

    @torch.inference_mode()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: [B, N, 3, H, W]
        returns: [B, N, C, H/14, W/14]
        """
        B, N, C, H, W = imgs.shape
        patch_h, patch_w = H // self.model.patch_size, W // self.model.patch_size

        imgs4 = imgs.reshape(B*N, C, H, W).to(self.device, memory_format=torch.channels_last)
        imgs4 = (imgs4 - self.model.image_mean) / self.model.image_std

        # Encoder
        tokens = self.model.encoder(imgs4.reshape(B * N, C, H, W), is_training=True)
        if isinstance(tokens, dict):
            tokens = tokens["x_norm_patchtokens"]  # [B*N, hw, enc_dim]

        # Cross-view decoder
        hidden, pos = self.model.decode(tokens, N, H, W)  # [B*N, T, dec_dim2]

        # Drop register tokens, keep only patch tokens
        hidden = hidden[:, self.model.patch_start_idx:, :]      # [B*N, patch_h*patch_w, C_dec]
        hidden = hidden.transpose(1, 2)                         # [B*N, C_dec, patch_h*patch_w]
        hidden = hidden.reshape(B, N, hidden.shape[1], patch_h, patch_w).contiguous()  # [B, N, C_dec, patch_h, patch_w]

        return hidden

@torch.no_grad()
def compute_pi3_feats(extractor: Pi3FeatureExtractor, imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs:  [B, N, 3, H, W]
    -> feats: [B, N, C, h, w]
    """
    # Make sure imgs are on same device / format as extractor expects
    imgs = imgs.to(extractor.device)     # no channels_last on 5D
    return extractor(imgs)

def se3_init_w2c(V: int, device: torch.device, rot_sigma_deg=10.0, trans_sigma=0.1) -> torch.Tensor:
    """
    Random init w2c: [V,4,4]
    """
    # axis-angle small
    axis = np.random.randn(V, 3)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
    angle = np.random.randn(V, 1) * np.deg2rad(rot_sigma_deg)
    rotvec = axis * angle

    # use torch Rodrigues via expmap on so(3)
    rv = torch.from_numpy(rotvec.astype(np.float32)).to(device)  # [V,3]
    theta = rv.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = rv / theta
    K = hat(k)
    I = torch.eye(3, device=device).expand(V, 3, 3)
    R = I + torch.sin(theta)[..., None] * K + (1 - torch.cos(theta))[..., None] * (K @ K)

    t = torch.randn(V, 3, device=device) * trans_sigma

    T = torch.eye(4, device=device).unsqueeze(0).repeat(V, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T

def hat(v: torch.Tensor) -> torch.Tensor:
    """
    v: [..., 3]
    returns: [..., 3, 3] skew-symmetric matrix (no inplace ops; vmap-safe)
    """
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([O, -z,  y], dim=-1),
            torch.stack([z,  O, -x], dim=-1),
            torch.stack([-y, x,  O], dim=-1),
        ],
        dim=-2,
    )

def get_fundamental_matrices(
    poses, 
    K, 
    K_inverse,
    height: int,
    width: int,
    index1: torch.LongTensor,
    index2: torch.LongTensor,
    l2_normalize_F=False,
):
    """Compute fundamental matrices for given camera parameters."""
    R = poses[..., :3, :3]              # [N,3,3]
    t = poses[..., :3, 3]               # [N,3]

    F, E = get_fundamental_matrix(K_inverse[index1], R[index1], t[index1], K_inverse[index2], R[index2], t[index2])

    if l2_normalize_F:
        F_scale = torch.norm(F, dim=(1, 2))
        F_scale = F_scale.clamp(min=0.0001)
        F = F / F_scale[:, None, None]

    return F

def get_fundamental_matrix(K1_inverse, R1, t1, K2_inverse, R2, t2):
    E = get_essential_matrix(R1, t1, R2, t2)
    F = K2_inverse.permute(0, 2, 1).matmul(E).matmul(K1_inverse)
    return F, E  # p2^T F p1 = 0


def get_essential_matrix(R1, t1, R2, t2):
    R12 = R2.matmul(R1.permute(0, 2, 1))
    t12 = t2 - R12.matmul(t1[..., None])[..., 0]
    E_R = R12
    E_t = -E_R.permute(0, 2, 1).matmul(t12[..., None])[..., 0]
    E = E_R.matmul(hat(E_t))
    return E

def _pair_list(N: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    ii, jj = torch.triu_indices(N, N, offset=1, device=device)
    return ii, jj

def ransac(
    mk0_px: np.ndarray,   # [M,2] pixels
    mk1_px: np.ndarray,   # [M,2] pixels
    K0: np.ndarray,       # [3,3]
    K1: np.ndarray,       # [3,3]
    thresh_px: float = 1.0,
    conf: float = 0.99999,
):
    """
    Returns:
      inliers: [M] bool
      E: [3,3] or None
    """
    M = mk0_px.shape[0]
    if M < 5:
        return None, None

    # normalize to camera coords
    k0 = (mk0_px - K0[[0,1],[2,2]][None]) / K0[[0,1],[0,1]][None]
    k1 = (mk1_px - K1[[0,1],[2,2]][None]) / K1[[0,1],[0,1]][None]

    # pixel threshold -> normalized threshold
    # (use average focal of the two views)
    f = 0.5 * (K0[0,0] + K1[1,1])
    thr_norm = float(thresh_px) / float(f)

    E, mask = cv2.findEssentialMat(
        k0, k1,
        np.eye(3),
        method=cv2.RANSAC,
        prob=conf,
        threshold=thr_norm,
    )
    if E is None or mask is None:
        return None, None

    inliers = (mask.ravel() > 0)
    return inliers, E


def _to_uint8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    """img_chw: [3,H,W] float in [0,1] -> uint8 HWC"""
    x = (img_chw.detach().cpu().clamp(0, 1) * 255.0).byte()
    return x.permute(1, 2, 0).numpy()

@torch.no_grad()
def dump_pair_matches(
    images_c: torch.Tensor,   # [N,3,Hc,Wc]
    vi: int,
    vj: int,
    pts0_k: torch.Tensor,     # [M,2] in image vi pixel coords
    pts1_k: torch.Tensor,     # [M,2] in image vj pixel coords
    w_k: torch.Tensor,        # [M]
    out_dir: str,
    tag: str,
    max_draw: int = 500,
):
    os.makedirs(out_dir, exist_ok=True)

    img0 = _to_uint8_hwc(images_c[vi])
    img1 = _to_uint8_hwc(images_c[vj])

    m = pts0_k.shape[0]
    if m == 0:
        return

    # subsample for visualization
    if m > max_draw:
        idx = torch.randperm(m, device=pts0_k.device)[:max_draw]
        p0 = pts0_k[idx].detach().cpu().numpy()
        p1 = pts1_k[idx].detach().cpu().numpy()
        ww = w_k[idx].detach().cpu().numpy()
    else:
        p0 = pts0_k.detach().cpu().numpy()
        p1 = pts1_k.detach().cpu().numpy()
        ww = w_k.detach().cpu().numpy()

    # color by weight using a colormap
    ww_n = (ww - ww.min()) / (ww.max() - ww.min() + 1e-9)
    cmap = plt.get_cmap("viridis")
    color = cmap(ww_n)  # [M,4]

    path = os.path.join(out_dir, f"{tag}_pair_{vi:02d}_{vj:02d}.png")
    make_matching_figure(
        img0, img1,
        p0, p1,
        color=color,
        text=[f"{tag}", f"pair {vi}-{vj}", f"shown {p0.shape[0]} / {m}"],
        dpi=100,
        path=path
    )

    # also save raw arrays to inspect numerically
    np.savez(
        os.path.join(out_dir, f"{tag}_pair_{vi:02d}_{vj:02d}.npz"),
        pts0=p0, pts1=p1, w=ww
    )

class GGSOptimizer(nn.Module):
    """
    Input:
      images: [N,3,H,W] float in [0,1]

    processed_matches = {
      "kp1_homo": [M,3],
      "kp2_homo": [M,3],
      "i1":       [P],
      "i2":       [P],
      "h":        scalar int,
      "w":        scalar int,
      "pair_idx": [M],   # in [0..P-1]
    }
    """

    def __init__(self, matcher: nn.Module, multiple: int = 32, use_amp_fp16: bool = True):
        super().__init__()
        self.matcher = matcher.eval()
        self.multiple = multiple
        self.use_amp_fp16 = use_amp_fp16
        self.processed_matches: Optional[Dict[str, torch.Tensor]] = None
        self.meta: Optional[Dict] = None
        self.Stiefel = geoopt.Stiefel()

    @staticmethod
    def rgb_to_gray(images_n3hw: torch.Tensor) -> torch.Tensor:
        r = images_n3hw[:, 0:1]
        g = images_n3hw[:, 1:2]
        b = images_n3hw[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b
    
    def clear_cached_matches(self):
        self.processed_matches = None
        self.meta = None
        self.inverse_intrinsics = None

    @torch.no_grad()
    def create_processed_matches(
        self,
        images: torch.Tensor,          # [N,3,H,W]
        Ks: torch.Tensor,              # [N,3,3]  (OpenCV intrinsics, pixel units)
        conf_thresh: Optional[float] = None,
        ransac_thresh_px: float = 1.0,
        ransac_conf: float = 0.99999,
        gt_w2c: Optional[torch.Tensor]=None, 
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        #assert images.ndim == 4 and images.shape[1] == 3, "images must be [N,3,H,W]"
        if images.ndim == 5 and images.shape[0] == 1:
            images = images.squeeze(0)
        if Ks.ndim == 4 and Ks.shape[0] == 1:
            Ks = Ks.squeeze(0)

        assert Ks.ndim == 3 and Ks.shape[-2:] == (3,3), "Ks must be [N,3,3]"
        device = images.device
        N, _, H, W = images.shape

        Hc = (H // self.multiple) * self.multiple
        Wc = (W // self.multiple) * self.multiple
        images_c = images[..., :Hc, :Wc]

        gray = self.rgb_to_gray(images_c)  # [N,1,Hc,Wc]

        i1, i2 = _pair_list(N, device=device)  # [P]
        P = i1.numel()

        kp1_chunks, kp2_chunks, pair_idx_chunks = [], [], []
        w_chunks = []

        # move Ks to cpu once for OpenCV (avoids repeated .cpu() in loop)
        Ks_cpu = Ks.detach().cpu().numpy()

        pairs_used = 0
        for p in range(P):
            vi = int(i1[p].item())
            vj = int(i2[p].item())

            batch = {"image0": gray[vi:vi+1], "image1": gray[vj:vj+1]}

            if self.use_amp_fp16 and images.is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    self.matcher(batch)
            else:
                self.matcher(batch)

            mk0 = batch["mkpts0_f"]          # [M,2] (torch, pixel coords in cropped frame)
            mk1 = batch["mkpts1_f"]
            mconf = batch.get("mconf", None) # [M] or None

            if conf_thresh is not None and mconf is not None:
                keep_conf = mconf >= conf_thresh
                mk0 = mk0[keep_conf]
                mk1 = mk1[keep_conf]
                mconf = mconf[keep_conf]

            if mk0.shape[0] < 5:
                continue

            K0 = Ks_cpu[vi]   # numpy [3,3]
            K1 = Ks_cpu[vj]

            mk0_np = mk0.detach().cpu().numpy()
            mk1_np = mk1.detach().cpu().numpy()

            k0 = (mk0_np - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
            k1 = (mk1_np - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

            f = 0.5 * (K0[0, 0] + K1[1, 1])
            thr_norm = float(ransac_thresh_px) / float(f)

            E_raw, mask = cv2.findEssentialMat(
                k0, k1, np.eye(3),
                method=cv2.RANSAC,
                prob=ransac_conf,
                threshold=thr_norm,
            )
            if E_raw is None or mask is None:
                continue

            inl_mask = (mask.ravel() > 0)
            Minl = int(inl_mask.sum())

            # HARD GATE: only use if > 40 inliers
            if Minl <= 40:
                continue
            else: 
                pairs_used +=1 

            # filter to inliers ONCE (torch)
            inl_t = torch.from_numpy(inl_mask).to(device=device)

            mk0 = mk0[inl_t]
            mk1 = mk1[inl_t]
            if mconf is not None:
                mconf = mconf[inl_t]

            # safety
            if mk0.shape[0] < 5:
                continue

            if mconf is None:
                w = torch.ones((mk0.shape[0],), device=device, dtype=torch.float32)
            else:
                w = mconf.float().clamp(min=0.0)

            w_chunks.append(w)

            ones = torch.ones((mk0.shape[0], 1), device=device, dtype=mk0.dtype)
            kp1_h = torch.cat([mk0, ones], dim=-1)  # [Minl,3]
            kp2_h = torch.cat([mk1, ones], dim=-1)

            pair_id = torch.full((mk0.shape[0],), p, device=device, dtype=torch.long)

            kp1_chunks.append(kp1_h)
            kp2_chunks.append(kp2_h)
            pair_idx_chunks.append(pair_id)


        print("GGSOptimizer: used", pairs_used, "out of", P, "pairs.")

        if len(kp1_chunks) == 0:
            kp1_homo = torch.empty((0, 3), device=device, dtype=torch.float32)
            kp2_homo = torch.empty((0, 3), device=device, dtype=torch.float32)
            pair_idx = torch.empty((0,), device=device, dtype=torch.long)
            weights = torch.empty((0,), device=device, dtype=torch.float32)
        else:
            kp1_homo = torch.cat(kp1_chunks, dim=0)
            kp2_homo = torch.cat(kp2_chunks, dim=0)
            pair_idx = torch.cat(pair_idx_chunks, dim=0)
            weights = torch.cat(w_chunks, dim=0)  # [M]

        processed_matches = {
            "kp1_homo": kp1_homo.float(),  # [M,3]
            "kp2_homo": kp2_homo.float(),  # [M,3]
            "weights":  weights.float(),   # [M]
            "i1": i1,              # [P]
            "i2": i2,              # [P]
            "h": torch.tensor(Hc, device=device, dtype=torch.long),
            "w": torch.tensor(Wc, device=device, dtype=torch.long),
            "pair_idx": pair_idx,  # [M] in [0..P-1]
        }

        meta = {
            "orig_hw": (H, W),
            "used_hw": (Hc, Wc),
            "cropped": (Hc != H) or (Wc != W),
            "multiple": self.multiple,
            "coord_frame": "cropped_image_pixels_(x,y)",
            "pair_indexing": "pair_idx is p in [0..P-1], p indexes (i<j)",
        }

        self.processed_matches = processed_matches
        self.meta = meta
        self.inverse_intrinsics = torch.inverse(Ks)
        return processed_matches, meta

    def compute_sampson_distance(
        self,
        poses: torch.Tensor,        # [N,3,4]
        Ks: torch.Tensor,           # [N,3,3]
        t: int,
        processed_matches: Dict[str, torch.Tensor],
        update_R: bool = True,
        update_T: bool = True,
        sampson_max: float = 10.0,
        use_weights = True,   
        eps: float = 1e-8,   
    ):
        if not update_R or not update_T:
            poses_use = poses.clone()
            if not update_R:
                poses_use[..., :3, :3] = poses_use[..., :3, :3].detach()
            if not update_T:
                poses_use[..., :3, 3] = poses_use[..., :3, 3].detach()
        else:
            poses_use = poses

        kp1_homo = processed_matches["kp1_homo"]  # [M,3]
        kp2_homo = processed_matches["kp2_homo"]  # [M,3]
        weights = processed_matches.get("weights", None)
        i1 = processed_matches["i1"]              # [P]
        i2 = processed_matches["i2"]              # [P]
        h = processed_matches["h"]
        w = processed_matches["w"]
        pair_idx = processed_matches["pair_idx"]  # [M]

        # One F per pair (i<j): [P,3,3]
        F_2_to_1 = get_fundamental_matrices(
            poses=poses_use,
            K=Ks,
            K_inverse = self.inverse_intrinsics,
            height=int(h.item()) if torch.is_tensor(h) else int(h),
            width=int(w.item()) if torch.is_tensor(w) else int(w),
            index1=i1,
            index2=i2,
            l2_normalize_F=False,
        )  # [P,3,3]

        F = F_2_to_1.transpose(-1, -2)  # [P,3,3]

        def _sampson_distance(F, kp1_homo, kp2_homo, pair_idx):
            left = torch.bmm(kp1_homo[:, None], F[pair_idx])
            right = torch.bmm(F[pair_idx], kp2_homo[..., None])

            bottom = (
                left[:, :, 0].square()
                + left[:, :, 1].square()
                + right[:, 0, :].square()
                + right[:, 1, :].square()
            )
            top = torch.bmm(left, kp2_homo[..., None]).square()

            sampson = (top[:, 0] / bottom).squeeze(-1)
            return sampson

        sampson = _sampson_distance(F, kp1_homo, kp2_homo, pair_idx)  # [M,1] -> used below

        keep = sampson < sampson_max
        sampson_kept = sampson[keep]

        if not use_weights:
            w_kept = None
            loss = sampson_kept.mean() if sampson_kept.numel() else sampson.sum() * 0.0
        else:
            w = weights
            w_kept = w[keep]
            w_kept = (w_kept / (w_kept.mean().clamp(min=1e-6))).clamp(max=10.0)
            denom = w_kept.sum().clamp(min=1e-6)
            loss = (w_kept * sampson_kept).sum() / denom

        sampson_to_print = sampson.detach().clamp(max=sampson_max).mean()
        return sampson_kept, sampson_to_print, loss
    
    def forward(
        self,
        images: Optional[torch.Tensor],
        Ks: torch.Tensor,                       # [N,3,3]
        poses = None,  
        optim_steps: int = 500,
        lr_R: float = 1e-3,
        lr_T: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        conf_thresh: Optional[float] = None,
        sampson_max: float = 10.0,
        verbose: bool = False, 
        gt_poses: Optional[torch.Tensor] = None,   # if you still want it for debugging
    ):
        # ----- matches -----
        if self.processed_matches is None and images is None:
            raise RuntimeError("No cached matches available; images input is required.")
        elif self.processed_matches is None:
            processed_matches, _ = self.create_processed_matches(
                images, Ks, conf_thresh=conf_thresh, gt_w2c=gt_poses
            )
        else:
            processed_matches = self.processed_matches

        device = Ks.device
        dtype = Ks.dtype

        # ----- init (poses ALWAYS c2w if provided) -----
        if poses is None:
            if images is None:
                raise RuntimeError("poses is None but images is also None; need images to infer N/device.")
            N = images.shape[0]
            poses_w2c = se3_init_w2c(N, images.device).to(dtype=dtype)
        else:
            if poses.ndim == 4 and poses.shape[0] == 1:
                poses = poses.squeeze(0)
            if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
                raise ValueError(f"poses must be [N,4,4] c2w if provided; got {poses.shape}")
            poses = poses.to(device=device, dtype=dtype)
            poses_w2c = se3_inverse(poses)   # c2w -> w2c

        R0 = poses_w2c[:, :3, :3].contiguous()
        t0 = poses_w2c[:, :3, 3].contiguous()
        N = poses_w2c.shape[0]

        # ----- optimize w2c -----
        R = geoopt.ManifoldParameter(R0.clone(), manifold=self.Stiefel, requires_grad=True)
        t = nn.Parameter(t0.clone(), requires_grad=True)

        optimizer = geoopt.optim.RiemannianAdam(
            [{"params": [R], "lr": lr_R}, {"params": [t], "lr": lr_T}],
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        for step in range(optim_steps):
            optimizer.zero_grad(set_to_none=True)

            # sampson expects [N,3,4]
            pose = torch.cat([R, t.unsqueeze(-1)], dim=-1)

            _, _, loss = self.compute_sampson_distance(
                poses=pose,
                Ks=Ks,
                t=step,
                processed_matches=processed_matches,
                update_R=True,
                update_T=True,
                sampson_max=sampson_max,
                eps=eps,
            )

            #if verbose and (step == optim_steps - 1):
            #    print(f"Step {step+1}/{optim_steps}: Sampson loss = {loss.item():.6f}")

            loss.backward()
            optimizer.step()

        # ----- refined w2c -> refined c2w -----
        poses_w2c_ref = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
        poses_w2c_ref[:, :3, :3] = R.detach()
        poses_w2c_ref[:, :3, 3]  = t.detach()

        poses_c2w_ref = se3_inverse(poses_w2c_ref)
        dictionary = {"camera_poses": poses_c2w_ref}
        return dictionary

def _pixel_grid(H: int, W: int, device, dtype=torch.float32) -> torch.Tensor:
    """[H,W,2] grid of pixel coords (x,y). Using integer coordinates (not +0.5)."""
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)  # [H,W,2]


def _to_normalized_grid(xy: torch.Tensor, H: int, W: int, align_corners: bool) -> torch.Tensor:
    """
    xy: [...,2] pixel coords (x,y)
    returns normalized coords in [-1,1] suitable for grid_sample.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    if align_corners:
        xn = 2.0 * x / (W - 1) - 1.0
        yn = 2.0 * y / (H - 1) - 1.0
    else:
        # align_corners=False mapping
        xn = (2.0 * (x + 0.5) / W) - 1.0
        yn = (2.0 * (y + 0.5) / H) - 1.0
    return torch.stack([xn, yn], dim=-1)


def _grid_sample_chw(x_chw: torch.Tensor, grid_hw2: torch.Tensor, *, mode="bilinear", align_corners=False) -> torch.Tensor:
    """
    x_chw: [C,H,W]
    grid_hw2: [H,W,2] in normalized coords
    returns: [C,H,W] sampled at grid locations
    """
    x_bchw = x_chw.unsqueeze(0)  # [1,C,H,W]
    g_bhw2 = grid_hw2.unsqueeze(0)  # [1,H,W,2]
    y = F.grid_sample(x_bchw, g_bhw2, mode=mode, padding_mode="zeros", align_corners=align_corners)
    return y[0]


def _homogenize(xy: torch.Tensor) -> torch.Tensor:
    """xy: [M,2] -> [M,3]"""
    ones = torch.ones((xy.shape[0], 1), device=xy.device, dtype=xy.dtype)
    return torch.cat([xy, ones], dim=-1)


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


class UFMGGSOptimizer(nn.Module):
    """
    Dense full-res correspondences from UFM, cached once, reused for sampson optimization.

    processed_matches = {
      "kp1_homo": [M,3]   # pixels in image i1 (cropped frame)
      "kp2_homo": [M,3]   # corresponding pixels in image i2 (cropped frame)
      "weights":  [M]
      "i1":       [P]
      "i2":       [P]
      "h":        scalar
      "w":        scalar
      "pair_idx": [M]
    }
    """

    def __init__(
        self,
        ufm_model: nn.Module,
        multiple: int = 32,
        use_amp_fp16: bool = True,
        conf_thresh: float = 0.5,
        min_kept: int = 500,
        use_cycle: bool = False,
        cycle_thresh_px: float = 2.0,
        cycle_align_corners: bool = False,
    ):
        super().__init__()
        self.model = ufm_model.eval()
        self.multiple = multiple
        self.use_amp_fp16 = use_amp_fp16

        self.conf_thresh = conf_thresh
        self.min_kept = min_kept

        self.use_cycle = use_cycle
        self.cycle_thresh_px = cycle_thresh_px
        self.cycle_align_corners = cycle_align_corners

        self.processed_matches: Optional[Dict[str, torch.Tensor]] = None
        self.meta: Optional[Dict] = None
        self.Stiefel = geoopt.Stiefel()

    def compute_sampson_distance(
        self,
        poses: torch.Tensor,        # [N,3,4]
        Ks: torch.Tensor,           # [N,3,3]
        t: int,
        processed_matches: Dict[str, torch.Tensor],
        update_R: bool = True,
        update_T: bool = True,
        sampson_max: float = 10.0,
        use_weights = True,   
        eps: float = 1e-8,   
    ):
        if not update_R or not update_T:
            poses_use = poses.clone()
            if not update_R:
                poses_use[..., :3, :3] = poses_use[..., :3, :3].detach()
            if not update_T:
                poses_use[..., :3, 3] = poses_use[..., :3, 3].detach()
        else:
            poses_use = poses

        kp1_homo = processed_matches["kp1_homo"]  # [M,3]
        kp2_homo = processed_matches["kp2_homo"]  # [M,3]
        weights = processed_matches.get("weights", None)
        i1 = processed_matches["i1"]              # [P]
        i2 = processed_matches["i2"]              # [P]
        h = processed_matches["h"]
        w = processed_matches["w"]
        pair_idx = processed_matches["pair_idx"]  # [M]

        # One F per pair (i<j): [P,3,3]
        F_2_to_1 = get_fundamental_matrices(
            poses=poses_use,
            K=Ks,
            K_inverse=self.inverse_intrinsics,
            height=int(h.item()) if torch.is_tensor(h) else int(h),
            width=int(w.item()) if torch.is_tensor(w) else int(w),
            index1=i1,
            index2=i2,
            l2_normalize_F=False,
        )  # [P,3,3]

        F = F_2_to_1.transpose(-1, -2)  # [P,3,3]

        def _sampson_distance(F, kp1_homo, kp2_homo, pair_idx):
            left = torch.bmm(kp1_homo[:, None], F[pair_idx])
            right = torch.bmm(F[pair_idx], kp2_homo[..., None])

            bottom = (
                left[:, :, 0].square()
                + left[:, :, 1].square()
                + right[:, 0, :].square()
                + right[:, 1, :].square()
            )
            top = torch.bmm(left, kp2_homo[..., None]).square()

            sampson = (top[:, 0] / bottom).squeeze(-1)
            return sampson

        sampson = _sampson_distance(F, kp1_homo.float(), kp2_homo.float(), pair_idx)  # [M,1] -> used below

        keep = sampson < sampson_max
        sampson_kept = sampson[keep]
        
        # DEBUG: no gating
        """
        sampson_kept = sampson
        loss = sampson.mean()
        keep = torch.ones_like(sampson, dtype=torch.bool)
        """
        
        if not use_weights:
            w_kept = None
            loss = sampson_kept.mean() if sampson_kept.numel() else sampson.sum() * 0.0
        else:
            w = weights.float()
            w_kept = w[keep]
            w_kept = (w_kept / (w_kept.mean().clamp(min=1e-6))).clamp(max=10.0)
            denom = w_kept.sum().clamp(min=1e-6)
            loss = (w_kept * sampson_kept).sum() / denom

        sampson_to_print = sampson.detach().clamp(max=sampson_max).mean()
        return sampson_kept, sampson_to_print, loss
    
    def clear_cached_matches(self):
        self.processed_matches = None
        self.meta = None
        self.inverse_intrinsics = None

    @torch.no_grad()
    def _infer_pair(self, img_i: torch.Tensor, img_j: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          flow_ij: [2,H,W] (pixel units)
          cov_ij:  [1,H,W] in [0,1]
        """
        # img_* are [3,H,W] float in [0,1]
        src_hwc = (img_i.permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8)
        tgt_hwc = (img_j.permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8)

        if self.use_amp_fp16 and img_i.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model.predict_correspondences_batched(source_image=src_hwc, target_image=tgt_hwc)
        else:
            out = self.model.predict_correspondences_batched(source_image=src_hwc, target_image=tgt_hwc)

        flow = out.flow.flow_output[0]                      # [2,H,W]
        cov  = out.covisibility.mask[0].unsqueeze(0)        # [1,H,W]
        return flow, cov

    @torch.no_grad()
    def create_processed_matches(self, images: torch.Tensor, Ks: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict]:
        device = images.device
        if(images.ndim == 5):
            images = images.squeeze(0)
        if(Ks.ndim == 4): 
            Ks = Ks.squeeze(0)

        N, _, H, W = images.shape

        Hc = (H // self.multiple) * self.multiple
        Wc = (W // self.multiple) * self.multiple
        images_c = images[:, :, :Hc, :Wc].contiguous()

        i1, i2 = _pair_list(N, device=device)
        P = i1.numel()

        grid = _pixel_grid(Hc, Wc, device=device, dtype=torch.float32) 

        kp1_chunks: List[torch.Tensor] = []
        kp2_chunks: List[torch.Tensor] = []
        w_chunks:   List[torch.Tensor] = []
        pidx_chunks: List[torch.Tensor] = []

        pairs_used = 0

        for p in range(P):
            vi = int(i1[p].item())
            vj = int(i2[p].item())

            flow_ij, cov_ij = self._infer_pair(images_c[vi], images_c[vj])  # [2,H,W], [1,H,W]
            flow_ij = flow_ij.permute(1, 2, 0).float().contiguous()         # [H,W,2]
            cov_ij  = cov_ij[0].float().contiguous()                        # [H,W]

            pts0 = grid                                                    # [H,W,2]
            pts1 = pts0 + flow_ij                                           # [H,W,2]

            inb = (
                (pts1[..., 0] >= 0) & (pts1[..., 0] <= (Wc - 1)) &
                (pts1[..., 1] >= 0) & (pts1[..., 1] <= (Hc - 1))
            )

            keep = inb & (cov_ij >= self.conf_thresh)

            if self.use_cycle:
                flow_ji, cov_ji = self._infer_pair(images_c[vj], images_c[vi])  # j->i
                pts1_norm = _to_normalized_grid(pts1, Hc, Wc, align_corners=self.cycle_align_corners)
                flow_ji_s = _grid_sample_chw(flow_ji.float(), pts1_norm, align_corners=self.cycle_align_corners)  # [2,H,W]
                flow_ji_s = flow_ji_s.permute(1, 2, 0).contiguous()  # [H,W,2]

                pts0_rec = pts1 + flow_ji_s
                cycle_err = torch.linalg.norm(pts0_rec - pts0, dim=-1)  # [H,W]

                cov_ji_s = _grid_sample_chw(cov_ji.float(), pts1_norm, align_corners=self.cycle_align_corners)[0]  # [H,W]
                keep = keep & (cov_ji_s >= self.conf_thresh) & (cycle_err <= self.cycle_thresh_px)

                wmap = (cov_ij * cov_ji_s).clamp(min=0.0)
                wmap = wmap * torch.exp(-0.5 * (cycle_err / (self.cycle_thresh_px + 1e-6)) ** 2)
            else:
                wmap = cov_ij

            Minl = int(keep.sum().item())
            if Minl < self.min_kept:
                continue
            pairs_used += 1

            pts0_k = pts0[keep].reshape(-1, 2)  # [Minl,2]
            pts1_k = pts1[keep].reshape(-1, 2)  # [Minl,2]
            w_k    = wmap[keep].reshape(-1)     # [Minl]

            kp1_chunks.append(_homogenize(pts0_k))
            kp2_chunks.append(_homogenize(pts1_k))
            w_chunks.append(w_k)

            if pairs_used <= 3: 
                dump_pair_matches(
                    images_c=images_c,
                    vi=vi,
                    vj=vj,
                    pts0_k=pts0_k,
                    pts1_k=pts1_k,
                    w_k=w_k,
                    out_dir="/vast/home/m/mgjacob/PARCC/results/UFM_Matches", 
                    tag=f"ufm_conf{self.conf_thresh:.2f}_p{p:03d}",
                    max_draw=500,
                )

            pidx_chunks.append(torch.full((Minl,), p, device=device, dtype=torch.long))

        if len(kp1_chunks) == 0:
            kp1_homo = torch.empty((0, 3), device=device, dtype=torch.float32)
            kp2_homo = torch.empty((0, 3), device=device, dtype=torch.float32)
            weights  = torch.empty((0,), device=device, dtype=torch.float32)
            pair_idx = torch.empty((0,), device=device, dtype=torch.long)
        else:
            kp1_homo = torch.cat(kp1_chunks, dim=0)
            kp2_homo = torch.cat(kp2_chunks, dim=0)
            weights  = torch.cat(w_chunks, dim=0)
            pair_idx = torch.cat(pidx_chunks, dim=0)

        processed_matches = {
            "kp1_homo": kp1_homo,
            "kp2_homo": kp2_homo,
            "weights":  weights,
            "i1": i1,
            "i2": i2,
            "h": torch.tensor(Hc, device=device, dtype=torch.long),
            "w": torch.tensor(Wc, device=device, dtype=torch.long),
            "pair_idx": pair_idx,
        }

        meta = {
            "used_hw": (Hc, Wc),
            "conf_thresh": self.conf_thresh,
            "use_cycle": self.use_cycle,
            "cycle_thresh_px": self.cycle_thresh_px,
            "cycle_align_corners": self.cycle_align_corners,
        }

        self.processed_matches = processed_matches
        self.meta = meta
        self.inverse_intrinsics = torch.inverse(Ks)
        return processed_matches, meta

    def forward(
        self,
        images: Optional[torch.Tensor],
        Ks: torch.Tensor,                       # [N,3,3]
        poses = None, 
        optim_steps: int = 5,
        lr_R: float = 5e-4,
        lr_T: float = 5e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        conf_thresh: Optional[float] = None,
        sampson_max: float = 10.0,
        verbose: bool = True,
        gt_poses=None,
    ):
        # ----- matches -----
        if self.processed_matches is None and images is None:
            raise RuntimeError("No cached matches available; images input is required.")
        elif self.processed_matches is None:
            processed_matches, _ = self.create_processed_matches(images, Ks)
        else:
            processed_matches = self.processed_matches

        device = Ks.device
        dtype = Ks.dtype

        # ----- init (poses are ALWAYS c2w if provided) -----
        if poses is None:
            if images is None:
                raise RuntimeError("poses is None but images is also None; need images to infer N/device.")
            N = images.shape[0]
            poses_w2c = se3_init_w2c(N, images.device).to(dtype=dtype)
        else:
            if poses.ndim == 4: 
                poses = poses.squeeze(0)
            if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
                raise ValueError(f"poses must be [N,4,4] c2w if provided; got {poses.shape}")
            poses = poses.to(device=device, dtype=dtype)
            poses_w2c = se3_inverse(poses)  # c2w -> w2c

        R0 = poses_w2c[:, :3, :3].contiguous()
        t0 = poses_w2c[:, :3, 3].contiguous()
        N = poses_w2c.shape[0]

        # ----- optimize w2c -----
        R = geoopt.ManifoldParameter(R0.clone(), manifold=self.Stiefel, requires_grad=True)
        t = nn.Parameter(t0.clone(), requires_grad=True)

        optimizer = geoopt.optim.RiemannianAdam(
            [{"params": [R], "lr": lr_R}, {"params": [t], "lr": lr_T}],
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        for step in range(optim_steps):
            optimizer.zero_grad(set_to_none=True)

            # poses for Sampson: [N,3,4]
            poses_cur = torch.cat([R, t.unsqueeze(-1)], dim=-1)

            sampson, _, loss = self.compute_sampson_distance(
                poses=poses_cur,
                Ks=Ks,
                t=step,
                processed_matches=processed_matches,
                update_R=True,
                update_T=True,
                sampson_max=sampson_max,
                eps=eps,
            )

            loss.backward()
            optimizer.step()

        # ----- refined w2c -> refined c2w -----
        poses_w2c_ref = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
        poses_w2c_ref[:, :3, :3] = R.detach()
        poses_w2c_ref[:, :3, 3] = t.detach()

        poses_c2w_ref = se3_inverse(poses_w2c_ref) 
        dictionary = {"camera_poses": poses_c2w_ref}
        return dictionary

def _unproject_depth_to_points(
    depth: torch.Tensor,  # [N,H,W]
    Ks: torch.Tensor,     # [N,3,3]
) -> torch.Tensor:
    """
    Unproject pixel grid using depth into camera coordinates.

    Returns:
      points_cam: [N, H*W, 3]
    """
    if depth.ndim != 3:
        raise ValueError(f"depth must be [N,H,W]; got {depth.shape}")
    if Ks.ndim == 4:
        Ks = Ks.squeeze(0)
    if Ks.ndim != 3 or Ks.shape[-2:] != (3, 3):
        raise ValueError(f"Ks must be [N,3,3]; got {Ks.shape}")

    N, H, W = depth.shape
    device, dtype = depth.device, depth.dtype

    grid = _pixel_grid(H, W, device=device, dtype=dtype)  # [H,W,2], (x,y)
    x = grid[..., 0]  # [H,W]
    y = grid[..., 1]  # [H,W]

    # Broadcast to [N,H,W]
    x = x.unsqueeze(0).expand(N, -1, -1)
    y = y.unsqueeze(0).expand(N, -1, -1)

    fx = Ks[:, 0, 0].view(N, 1, 1)
    fy = Ks[:, 1, 1].view(N, 1, 1)
    cx = Ks[:, 0, 2].view(N, 1, 1)
    cy = Ks[:, 1, 2].view(N, 1, 1)

    z = depth
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy

    pts = torch.stack([X, Y, z], dim=-1)   # [N,H,W,3]
    pts = pts.reshape(N, H * W, 3)         # [N,HW,3]
    return pts


def _transform_points(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """
    T:   [B,4,4]
    pts: [B,P,3]
    """
    B, P, _ = pts.shape
    ones = torch.ones((B, P, 1), device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=-1)            # [B,P,4]
    pts_t = (T @ pts_h.transpose(1, 2)).transpose(1, 2)  # [B,P,4]
    return pts_t[..., :3]

class DepthAnythingMetricOptimizer(nn.Module):
    """
    Metric-depth variant (scale-only).

    Raw model output y -> calibrated depth via a single global SCALE:
        depth = clamp(S * y, eps)

    Optimize:
      - global S (positive)
      - poses for views 1..N-1 (view 0 fixed to its initial value)

    Cache:
      - raw outputs y = model(images)  [N,H,W] (no-grad)
      - S_u_state for warm-start (cleared together with cached raw)
    """

    def __init__(
        self,
        depth_anything: nn.Module,
        use_amp_fp16: bool = True,
        cache_raw: bool = True,
    ):
        super().__init__()
        self.model = depth_anything.eval()
        self.use_amp_fp16 = use_amp_fp16
        self.cache_raw = cache_raw

        self.cached_raw: Optional[torch.Tensor] = None  # [N,H,W], no-grad
        self.meta: Optional[Dict] = None

        self.Stiefel = geoopt.Stiefel()

        # Warm-start state (buffers, not saved by default)
        self.register_buffer("S_u_state", None, persistent=False)   # shape [1] (unconstrained)

    def clear_cached(self):
        self.cached_raw = None
        self.meta = None
        self.S_u_state = None

    @torch.no_grad()
    def _get_raw(self, images: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cached_raw is not None:
            return self.cached_raw

        if images is None:
            raise RuntimeError("No cached raw outputs yet; pass `images` on the first forward().")

        if images.ndim == 5:
            images = images.squeeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be [N,3,H,W]; got {images.shape}")

        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype).view(1,3,1,1)
        images = (images - mean) / std

        if self.use_amp_fp16 and images.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                raw = self.model(images)  # [N,H,W]
        else:
            raw = self.model(images)

        raw = raw.float().detach()
        

        if self.cache_raw:
            self.cached_raw = raw
            self.meta = {"H": images.shape[-2], "W": images.shape[-1]}

        return raw

    def _init_scale(self, device: torch.device) -> nn.Parameter:
        """
        Global S_u (scalar), warm-started. We map to positive scale via softplus.
        """
        if self.S_u_state is None or self.S_u_state.numel() != 1:
            self.S_u_state = torch.zeros(1, device=device)

        S_u = nn.Parameter(self.S_u_state.clone(), requires_grad=True)  # [1]
        return S_u

    def _commit_scale(self, S_u: torch.Tensor):
        self.S_u_state = S_u.detach()

    @staticmethod
    def _raw_to_metric_depth_scale(raw: torch.Tensor, S: torch.Tensor, eps: float) -> torch.Tensor:
        """
        raw: [N,H,W]
        S: [1] scalar
        depth = clamp(S * raw, eps)
        """
        depth = (S.view(1, 1, 1) * raw).clamp_min(eps)
        return depth

    def compute_chamfer_loss_all_pairs(
        self,
        points_cam: torch.Tensor,  # [N,HW,3]
        poses_w2c: torch.Tensor,   # [N,4,4]
    ) -> torch.Tensor:
        N = points_cam.shape[0]
        device = points_cam.device

        i1, i2 = _pair_list(N, device=device)
        P = i1.numel()
        if P == 0:
            return points_cam.sum() * 0.0

        c2w = se3_inverse(poses_w2c)               # [N,4,4]
        T_i_to_j = poses_w2c[i2] @ c2w[i1]         # [P,4,4]
        T_j_to_i = poses_w2c[i1] @ c2w[i2]         # [P,4,4]

        Pi = points_cam[i1]                        # [P,HW,3]
        Pj = points_cam[i2]                        # [P,HW,3]

        Pi_in_j = _transform_points(T_i_to_j, Pi)  # [P,HW,3]
        Pj_in_i = _transform_points(T_j_to_i, Pj)  # [P,HW,3]

        loss_ij, _ = chamfer_distance(Pi_in_j, Pj)
        loss_ji, _ = chamfer_distance(Pj_in_i, Pi)
        return 0.5 * (loss_ij + loss_ji)

    def forward(
        self,
        images: Optional[torch.Tensor],
        Ks: torch.Tensor,                      # [N,3,3]
        poses: Optional[torch.Tensor] = None,  # [N,4,4] c2w if provided
        optim_steps: int = 5,
        lr_R: float = 5e-4,
        lr_T: float = 5e-4,
        lr_A: float = 5e-4,   # kept for signature compatibility (unused)
        lr_B: float = 5e-4,   # used as lr_S
        betas=(0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:

        if Ks.ndim == 4:
            Ks = Ks.squeeze(0)
        if Ks.ndim != 3 or Ks.shape[-2:] != (3, 3):
            raise ValueError(f"Ks must be [N,3,3]; got {Ks.shape}")

        device = Ks.device
        dtype = Ks.dtype

        # ---- cached raw model outputs ----
        raw = self._get_raw(images).to(device=device, dtype=torch.float32)  # [N,H,W]
        N = raw.shape[0]

        # ---- init poses_w2c ----
        if poses is None:
            poses_w2c_init = se3_init_w2c(N, device=device).to(dtype=dtype)  # [N,4,4]
        else:
            if poses.ndim == 4:
                poses = poses.squeeze(0)
            if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
                raise ValueError(f"poses must be [N,4,4] c2w if provided; got {poses.shape}")
            poses = poses.to(device=device, dtype=dtype)
            poses_w2c_init = se3_inverse(poses)

        # ---- FIX FIRST POSE TO ITS INITIAL VALUE ----
        w2c0_fixed = poses_w2c_init[0].detach()  # [4,4]

        # optimize views 1..N-1
        R0 = poses_w2c_init[1:, :3, :3].contiguous()
        t0 = poses_w2c_init[1:, :3, 3].contiguous()

        R = geoopt.ManifoldParameter(R0.clone(), manifold=self.Stiefel, requires_grad=True)  # [N-1,3,3]
        t = nn.Parameter(t0.clone(), requires_grad=True)                                     # [N-1,3]

        # ---- global SCALE ----
        S_u = self._init_scale(device)

        def S_pos():
            return F.softplus(S_u) + 1e-6

        optimizer = geoopt.optim.RiemannianAdam(
            [
                {"params": [R], "lr": lr_R},
                {"params": [t], "lr": lr_T},
                {"params": [S_u], "lr": lr_B},  # re-use lr_B as scale LR
            ],
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        loss_val = None
        for step in range(optim_steps):
            optimizer.zero_grad(set_to_none=True)

            poses_cur_w2c = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
            poses_cur_w2c[0] = w2c0_fixed.to(device=device, dtype=dtype)
            poses_cur_w2c[1:, :3, :3] = R
            poses_cur_w2c[1:, :3, 3]  = t

            depth = self._raw_to_metric_depth_scale(raw, S=S_pos(), eps=eps).to(dtype=dtype)  # [N,H,W]
            points = _unproject_depth_to_points(depth, Ks).to(device=device, dtype=dtype)     # [N,HW,3]

            loss = self.compute_chamfer_loss_all_pairs(points_cam=points, poses_w2c=poses_cur_w2c)
            loss.backward()
            optimizer.step()

            loss_val = loss.detach()
            if step % 25 == 0:
                print("Loss is ", loss_val)
            if verbose:
                print(
                    f"[DepthAnythingMetricChamfer-Scale] step {step+1}/{optim_steps} loss={float(loss_val):.6f}  "
                    f"S={float(S_pos().detach()):.6g}"
                )

        # ---- build refined poses ----
        poses_w2c_ref = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
        poses_w2c_ref[0] = w2c0_fixed.to(device=device, dtype=dtype)
        poses_w2c_ref[1:, :3, :3] = R.detach()
        poses_w2c_ref[1:, :3, 3]  = t.detach()
        poses_c2w_ref = se3_inverse(poses_w2c_ref)

        # ---- commit warm-start state ----
        self._commit_scale(S_u)

        S_out = S_pos().detach().expand(N)  # [N]

        return {
            "camera_poses": poses_c2w_ref,
            "loss": loss_val if loss_val is not None else raw.sum() * 0.0,
            "A": S_out,   # kept key name for downstream compatibility; now stores scale
            "B": S_out,   # kept key name for downstream compatibility; now stores scale
        }