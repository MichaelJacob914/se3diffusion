#!/usr/bin/env python
# coding: utf-8
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
paths = [HERE / "gta", HERE / "gta" / "source"]
sys.path[:0] = [str(p) for p in paths if p.exists() and str(p) not in sys.path]
import layers
from layers import Transformer as GTATransformer
from source.utils.gta import make_SO2mats, make_T2mats
from source.utils.common import positionalencoding2d, downsample, rigid_transform
from source.utils.wigner_d import rotmat_to_wigner_d_matrices
from SO3n import so3_diffuser, SO3Algebra
from R3n import r3_diffuser
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
import numpy as np
import copy
from scipy.spatial.transform import Rotation as Rot
import math
import math
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import geoopt
from geoopt.optim import (RiemannianAdam)
Stiefel = geoopt.Stiefel()
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# In[9]:


#TAKEN FROM PAPER
#NOT MY CODE


class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None, downsample=True):
        super().__init__()
        if hdim is None:
            hdim = idim

        if odim is None:
            odim = 2 * hdim

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2 if downsample else 1, **conv_kwargs),
            nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class GTASE3(nn.Module):
    def __init__(self, 
                 N: int,
                 d_model: int = 768,
                 num_att_blocks: int = 5,
                 num_conv_blocks: int = 5,
                 num_timesteps: int = 100,
                 heads: int = 12,
                 dim_out: int = 6,
                 device: str = "cuda",
                 dropout: float = 0.01,
                 attn_args: Optional[dict] = None,
                 attn_kwargs: Optional[dict] = None, 
                 num_images: int = None, 
                 batch_size: int = 64, 
                 num_gpus = 2):
        super().__init__()
        torch.inverse(torch.ones((1, 1), device="cuda:0"))

        self.N = N
        self.d_model = d_model
        self.h, self.w = 2, 2
        self.device = device
        self.coord = self.make_2dcoord(self.h, self.w)  # [H, W, 2] numpy array
        self.downsample = 0
        self.downsample_input_coord = 0
        self.num_images = num_images
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        #Image Processing
        self.lin_camera = nn.Linear(12, d_model)  # For camera
        self.lin_planar = nn.Linear(180, d_model) # For 2d positions
        self.emb_dim = 0
        conv_blocks = []
        cur_hdim = d_model // 8

        # First block: input from image (3 channels)
        conv_blocks.append(SRTConvBlock(idim=3, hdim=cur_hdim, odim=cur_hdim * 2))
        cur_hdim *= 2

        # Remaining blocks: keep track of increasing channels
        for i in range(1, num_conv_blocks):
            next_hdim = cur_hdim * 2
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, hdim=cur_hdim, odim=next_hdim))
            cur_hdim = next_hdim

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.feature_proj = nn.Conv2d(1024, d_model, kernel_size=1)
        

        #Pose Processing
        self.pose_proj = nn.Sequential(
            nn.Linear(12, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.pose_output_proj = nn.Linear(d_model, dim_out)

        #Time processing 
        self.time_embed = nn.Embedding(num_timesteps, d_model)

        # This should project to attdim after conv_blocks
        self.per_patch_linear = nn.Conv2d(cur_hdim, d_model, kernel_size=1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, attdim]

        self.transformer = GTATransformer(
            dim=d_model,
            depth=num_att_blocks,
            heads=heads,
            dim_head=d_model//heads,
            mlp_dim=d_model*2,
            selfatt=True,
            dropout=dropout,
            attn_args=attn_args
        )
        self.attn_kwargs = attn_kwargs

        self.extras = {}
        if(self.num_images is not None): 
            input_coord = self.generate_positional_coords(self.num_images)
            self.input_coord = input_coord.expand(int(self.batch_size/self.num_gpus), num_images, -1, -1)
            self.extras['input_coord'] = self.input_coord
            self.extras['target_coord'] = self.input_coord
            self.extras = self.pre_compute_reps(self.extras)
    
    def generate_feature_tokens(self, feats):
        """
        Args:
            feats: [B, N, 1024, 14, 14]
        Returns:
            tokens: [B, N * 4, d_model]
        """
        B, N, C, H, W = feats.shape
        feats = feats.view(B * N, C, H, W)                          # [B*N, 1024, 14, 14]
        pooled = F.adaptive_avg_pool2d(feats, output_size=(2, 2))  # [B*N, 1024, 3, 3]
        projected = self.feature_proj(pooled)                      # [B*N, d_model, 3, 3]
        tokens = projected.flatten(2).transpose(1, 2)              # [B*N, 9, d_model]
        return tokens.reshape(B, N * 4, -1)                           

    def generate_image_tokens(self, images, pose, batch_size, num_images, downsample_tokens = True):
        """
        Generate tokens per image with optional 2D positional and camera pose embeddings.

        Args:
            images: [B, N, 3, H, W]      - RGB images
            pose: [B, N, 4, 4] or [B, N, 12] - Camera poses (flattened or full)
            batch_size: int              - Batch size B
            num_images: int              - Number of views per sample (N)

        Returns:
            tokens: [B, N*T, attdim]     - Flattened image tokens per view
        """
        images = images.flatten(0, 1)  # [B*N, 3, H, W]

        # Apply CNN feature extractor and project to attention dimension
        images = self.conv_blocks(images)             # [B*N, C_out, H', W']
        images = self.per_patch_linear(images)        # [B*N, attdim, H', W']
        if downsample_tokens:
            images = F.adaptive_avg_pool2d(images, output_size=(2, 2))  # [B*N, attdim, 2, 2]
        H_attn, W_attn = images.shape[-2:]

        # 2D positional embedding
        emb_2dpos = positionalencoding2d(180, H_attn, W_attn).to(images.device)  # [180, H', W']
        emb_2dpos = emb_2dpos.permute(1, 2, 0)                # [H', W', 180]
        emb_2dpos = self.lin_planar(emb_2dpos)                # [H', W', attdim]
        emb_2dpos = emb_2dpos.permute(2, 0, 1)                # [attdim, H', W']
        emb_2dpos = emb_2dpos[None].repeat(batch_size * num_images, 1, 1, 1)  # [B*N, attdim, H', W']

        # Camera embedding
        pose = pose.reshape(-1, 4, 4)                          # [B*N, 4, 4]
        emb_camera = self.lin_camera(pose[:, :3, :].reshape(-1, 12))  # [B*N, attdim]
        emb_camera = emb_camera[:, :, None, None].expand(-1, -1, H_attn, W_attn)  # [B*N, attdim, H', W']

        # Combine embeddings with image features
        images = images + emb_2dpos + emb_camera              # [B*N, attdim, H', W']

        # Flatten spatial dimensions into tokens
        images = images.flatten(2, 3).permute(0, 2, 1)            # [B*N, H'*W', attdim]
        T, C = images.shape[1:]                                   # T = #patches per image

        # Reshape to [B, N*T, attdim]
        image_tokens = images.reshape(batch_size, num_images * T, C)
        return image_tokens

    def generate_time_tokens(self, t, batch_size):
        """
        Embed scalar timestep t into a single time token per sample.

        Args:
            t: [B] or scalar               - diffusion timestep(s)
            batch_size: int               - B

        Returns:
            time_tokens: [B, 1, attdim]
        """
        if isinstance(t, (int, float)):
            t = torch.tensor([t] * batch_size, device=self.device).float()
        elif t.ndim == 1:
            t = t.float().to(self.device)
        else:
            t = t.squeeze(-1).float().to(self.device)  # in case [B,1]

        t = t.view(-1).long()
        t_embed = self.time_embed(t)          # [B, attdim]
        time_tokens = t_embed[:, None, :]     # [B, 1, attdim]
        return time_tokens
    
    def generate_pose_tokens(self, pose, batch_size, num_images):
        """
        Embed each camera pose into a token.

        Args:
            pose: [B, N, 4, 4] or [B, N, 12] - pose per view
            batch_size: int                 - B
            num_images: int                 - N

        Returns:
            pose_tokens: [B, N, attdim]
        """
        if pose.shape[-2:] == (4, 4):
            pose = pose[:, :, :3, :].reshape(batch_size, num_images, 12)  # [B, N, 12]

        pose_tokens = self.pose_proj(pose)  # [B, N, attdim]
        return pose_tokens

    def compute_se3_transforms(self, R, T):
        """
        Construct full SE(3) transformation matrices from rotation and translation.

        Inputs:
            R: [B, N, 3, 3] - rotation matrices
            T: [B, N, 3]    - translation vectors

        Returns:
            input_transforms: [B, N, 4, 4] - SE(3) transform
            target_transforms: [B, N, 4, 4] - identical for now (can be used for symmetry)
            pose: [B, N, 12] - flattened [R | T] 3x4 matrix for each camera
        """
        B, N = R.shape[:2]

        # Create identity row: [0, 0, 0, 1]
        bottom_row = torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device).view(1, 1, 1, 4)
        bottom_row = bottom_row.expand(B, N, 1, 4)  # [B, N, 1, 4]

        # Construct [R | T]
        RT = torch.cat([R, T.unsqueeze(-1)], dim=-1)  # [B, N, 3, 4]

        # Full SE(3): concatenate bottom row
        se3 = torch.cat([RT, bottom_row], dim=-2)     # [B, N, 4, 4]

        return se3, RT.reshape(B, N, 12)

    def downsample(self, x, num_steps=1):
        if num_steps is None or num_steps < 1:
            return x
        stride = 2**num_steps
        return x[stride//2::stride, stride//2::stride]

        
    def make_2dcoord(self, H, W):
        """
        Return 2d coord values of shape [H, W, 2] 
        """
        x = np.arange(H, dtype=np.float32)/H   # [-0.5, 0.5)
        y = np.arange(W, dtype=np.float32)/W   # [-0.5, 0.5)
        x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
        return np.stack([x_grid.flatten(), y_grid.flatten()], -1).reshape(H, W, 2)


    def generate_positional_coords(self, num_images):
        """
        Generate [N, H'*W', 2] 2D coordinates for positional embedding.

        Args:
            num_images: int - number of views N

        Returns:
            input_coord: [N, H'*W', 2] numpy array
        """
        total_downsample = self.downsample + self.downsample_input_coord if self.downsample is not None else self.downsample_input_coord
        coord_ds = downsample(self.coord, total_downsample)  # [H', W', 2]
        input_coord = coord_ds.reshape(-1, 2)  # [H'*W', 2]
        input_coord = torch.tensor(input_coord, dtype=torch.float32, device=self.device)
        input_coord = input_coord.unsqueeze(0).repeat(num_images, 1, 1)
        return input_coord

    def forward(self, R, T, feats, imgs, t):
        """
        Inputs:
            R: [B, N, 3, 3]         - rotation matrices
            T: [B, N, 3]            - translation vectors
            feats: [B, N, 1024, 14, 14] - CNN feature maps
            imgs: [B, N, 3, 224, 224]   - RGB images
            t: scalar or [B]        - diffusion timestep
        Returns:
            x: [B, N*T, attdim] or [B, N*T, dim_out]
            extras: dict containing additional outputs
        
        print("R", R.shape)
        print("T", T.shape)
        print("feats", feats.shape)
        print("imgs", imgs.shape)
        """
        batch_size, num_images = R.shape[:2]
        device = R.device
        # --- Compute SE(3) Transforms ---
        input_transforms, pose = self.compute_se3_transforms(R, T)
        target_transforms = input_transforms

        # --- Positional Coordinates ---
        if(self.num_images is None): 
            # --- Extras Dictionary ---
            extras = {
                'input_transforms': input_transforms,
                'target_transforms': target_transforms,
            }
        else: 
            extras = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.extras.items()}
            extras['input_transforms'] = input_transforms
            extras['target_transforms'] = input_transforms

        # --- Precompute Representations for Geometric Attention ---
        extras = self.compute_reps(extras)

        # --- Time Tokens: expand to per-view ---
        time_tokens = self.generate_time_tokens(t, batch_size)  # [B, 1, d]
        time_tokens = time_tokens.expand(-1, num_images, -1)    # [B, N, d]

        # --- Pose Tokens ---
        pose_tokens = self.generate_pose_tokens(pose, batch_size, num_images)  # [B, N, d]

        # --- Image Tokens ---
        image_tokens = self.generate_image_tokens(imgs, input_transforms, batch_size, num_images)  # [B, N*T_i, d]
        image_tokens = image_tokens.reshape(batch_size, num_images, -1, image_tokens.shape[-1])       # [B, N, T_i, d]

        # --- Feature Tokens ---
        feature_tokens = self.generate_feature_tokens(feats)  # [B, N*T_f, d]
        feature_tokens = feature_tokens.reshape(batch_size, num_images, -1, feature_tokens.shape[-1]) # [B, N, T_f, d]

        # --- Stack per-view tokens: [pose, image*4, feature*4, time] => [B, N, 10, d]
        tokens = torch.cat([
            pose_tokens.unsqueeze(2),           # [B, N, 1, d]
            image_tokens,                       # [B, N, T_i=4, d]
            feature_tokens,                     # [B, N, T_f=4, d]
            time_tokens.unsqueeze(2),           # [B, N, 1, d]
        ], dim=2)  # [B, N, 10, d]

        # Flatten to [B, N*10, d]
        tokens = tokens.view(batch_size, num_images * 10, -1)  # [B, N*10, d]

        # --- Transformer ---
        x = self.transformer(tokens, None, extras)  # [B, N*10, d]
        
        pose_outputs = x[:, ::10, :]  # [B, N, d]
        pose_preds = self.pose_output_proj(pose_outputs)
        return pose_preds

    def pre_compute_reps(self, extras):
        """
        Precomputes so2 and t2 reps if possible
        """
        reps = extras
        f_dims = self.attn_kwargs['f_dims']
        flattened_reps = []
        flattened_invreps = []
        if 'so2' in f_dims and f_dims['so2'] > 0:
            coord = extras['input_coord']
            coord = coord.reshape(coord.shape[0], -1, 2)  # [B, Nq*Tq, 2]
            so2rep = make_SO2mats(coord,
                                    nfreqs=self.attn_kwargs['so2'],
                                    max_freqs=[self.attn_kwargs['max_freq_h'],
                                                self.attn_kwargs['max_freq_w']],
                                    shared_freqs=self.attn_kwargs['shared_freqs'] if 'shared_freqs' in self.attn_kwargs else False)  # [B, Nq*Tq, deg, 2, 2, 2]
            so2rep = so2rep.flatten(-4, -3)
            NqTq = so2rep.shape[1]
            reps['so2rep_q'] = reps['so2rep_k'] = so2rep  # [B, T, C, 2, 2]
            reps['so2fn'] = lambda A, x: torch.einsum(
                'btcij,bhtcj->bhtci', A, x)
            flattened = so2rep.reshape(
                so2rep.shape[0], so2rep.shape[1], -1)  # [B, T, C*2*2]
            flattened_reps.append(flattened)
            # [B, T, C*2*2]
            flattened_inv = so2rep.transpose(-2, -1).reshape(
                so2rep.shape[0], so2rep.shape[1], -1)
            flattened_invreps.append(flattened_inv)

        if 't2' in f_dims and f_dims['t2'] > 0:
            coord = extras['input_coord']
            coord = coord.reshape(coord.shape[0], -1, 2)  # [B, Nq*Tq, 2]
            t2rep = make_T2mats(coord)  # [B, Nq*Tq, 2] -> [B, Nq*Tq, 3, 3]
            reps['t2rep_q'] = reps['t2rep_k'] = t2rep
            reps['inv_t2rep_q'] = torch.linalg.inv(t2rep.clone())
            reps['t2fn'] = lambda A, x: torch.einsum(
                'btij,bhtcj->bhtci', A, x)

        reps['flattened_rep_q'] = reps['flattened_rep_k'] = torch.cat(flattened_reps, -1)  # 16 + 2*freqs*2*2
        reps['flattened_invrep_q'] = torch.cat(flattened_invreps, -1)
        return reps

    def compute_reps(self, extras):
        reps = extras
        f_dims = self.attn_kwargs['f_dims']
        flattened_reps = []
        flattened_invreps = []
        NqTq = 20

        if 'se3' in f_dims and f_dims['se3'] > 0:
            extrinsic = extras['input_transforms']
            se3rep = torch.linalg.inv(extrinsic.clone())  # [B, Nq, 4, 4]
            reps['se3fn'] = lambda A, x: torch.einsum(
                'bnij,bhntcj->bhntci', A, x)
            reps['se3rep_q'] = reps['se3rep_k'] = se3rep
            reps['inv_se3rep_q'] = extrinsic

            flattened = extrinsic.repeat_interleave(
                NqTq//extrinsic.shape[1], 1).transpose(-2, -1).reshape(se3rep.shape[0], -1, 16)  # [B, T, 4*4]
            flattened_reps.append(flattened)
            flattened_inv = extrinsic.repeat_interleave(
                NqTq//extrinsic.shape[1], 1).reshape(se3rep.shape[0], -1, 16)  # [B, T, 4*4]
            flattened_invreps.append(flattened_inv)

        if 'so3' in f_dims and f_dims['so3'] > 0:
            n_degs = self.attn_kwargs['so3']
            R_q = torch.linalg.inv(extras['input_transforms'].clone())[..., :3, :3]
            B, Nq = R_q.shape[0], R_q.shape[1]
            D_q = rotmat_to_wigner_d_matrices(n_degs, R_q.flatten(0, 1))[1:]
            for i, D in enumerate(D_q):
                if 'zeroout_so3' in self.attn_kwargs and self.attn_kwargs['zeroout_so3']:
                    D_q[i] = torch.zeros_like(
                        D.reshape(B, Nq, D.shape[-2], D.shape[-1]))
                elif 'id_so3' in self.attn_kwargs and self.attn_kwargs['id_so3']:
                    D_q[i] = torch.stack([torch.eye(
                        D.shape[-1])]*B*Nq, 0).reshape(B, Nq, D.shape[-2], D.shape[-1]).to(D.device)
                else:
                    D_q[i] = D.reshape(B, Nq, D.shape[-2], D.shape[-1])
            reps['so3rep_q'] = reps['so3rep_k'] = D_q
            reps['so3fn'] = lambda A, x: torch.einsum(
                'bnij,bhnkj->bhnki', A, x)

        reps['flattened_rep_q'] = reps['flattened_rep_k'] = torch.cat(flattened_reps, -1)  # 16 + 2*freqs*2*2
        reps['flattened_invrep_q'] = torch.cat(flattened_invreps, -1)
        return reps

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- helpers: 2D sinusoidal pos enc ----------------------
def _build_2d_sincos_pos_embed(h: int, w: int, d: int, device=None):
    """
    Returns [h*w, d] 2D sinusoidal embeddings.
    Requires d % 4 == 0.
    """
    assert d % 4 == 0, "d_model must be divisible by 4 for 2D sincos."
    m = d // 4
    # frequencies like ViT: 1/10000^{2i/d}
    inv_freq = 1.0 / (10000 ** (torch.arange(0, m, device=device, dtype=torch.float32) / float(m)))
    # grid
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [h,w]
    xx = xx.reshape(-1, 1)  # [h*w,1]
    yy = yy.reshape(-1, 1)  # [h*w,1]
    # pos enc per axis → [h*w, 2m]
    pos_x = torch.cat([torch.sin(xx * inv_freq), torch.cos(xx * inv_freq)], dim=-1)
    pos_y = torch.cat([torch.sin(yy * inv_freq), torch.cos(yy * inv_freq)], dim=-1)
    # concat → [h*w, d]
    pe = torch.cat([pos_x, pos_y], dim=-1)
    return pe  # [h*w, d]


# ---------------------- SwiGLU MLP (ratio=4) ----------------------
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.w1 = nn.Linear(d_model, hidden * 2)
        self.w2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w1(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        x = self.w2(x)
        return self.drop(x)


# ---------------------- Transformer block (pre-norm) ----------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        # x: [B, L, d]
        y = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.dropout1(y)
        x = x + self.mlp(self.norm2(x))
        return x
# ---------------------- Transformer (grid size configurable; default 3x3) ----------------------
class Transformer(nn.Module):
    def __init__(self,
                 N: int,
                 d_model: int = 768,
                 num_att_blocks: int = 6,          # (intra ↔ inter) alternations → 12 total blocks
                 num_heads: int = 12,
                 dim_out: int = 6,
                 num_timesteps: int = 100,
                 grid_hw: int = 5,                 # <<< 3x3 grid → 9 tokens/view
                 dropout: float = 0.0):
        super().__init__()

        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sincos PE."
        self.N = N
        self.d_model = d_model
        self.grid_hw = grid_hw
        self.n_img_tokens  = grid_hw * grid_hw     # 9 tokens/view
        self.n_feat_tokens = grid_hw * grid_hw     # 9 tokens/view

        # ----------- Projections / Encoders -----------
        self.pose_proj  = nn.Linear(12, d_model)
        self.time_embed = nn.Embedding(num_timesteps, d_model)

        # Stronger image tokenizer → grid_hw x grid_hw tokens/view
        self.img_backbone = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3), nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((self.grid_hw, self.grid_hw)),
            nn.Conv2d(256, d_model, kernel_size=1), nn.GELU()
        )

        # Feature tokenizer → grid_hw x grid_hw tokens/view
        self.feat_proj = nn.Conv2d(1024, d_model, kernel_size=1)

        # ----------- 2D positional encodings for img/feat grids -----------
        pos_img  = _build_2d_sincos_pos_embed(self.grid_hw, self.grid_hw, d_model)
        pos_feat = _build_2d_sincos_pos_embed(self.grid_hw, self.grid_hw, d_model)
        self.register_buffer("pos_img",  pos_img,  persistent=False)  # [grid^2, d]
        self.register_buffer("pos_feat", pos_feat, persistent=False)  # [grid^2, d]

        # ----------- Token-type embeddings (shared across views) -----------
        # 0=view_summary, 1=pose, 2=time, 3=feat, 4=img
        self.token_type = nn.Embedding(5, d_model)

        # Per-view token layout: [CLS, pose, time, feat×F, img×I]
        type_ids_view = [0, 1, 2] + [3]*self.n_feat_tokens + [4]*self.n_img_tokens
        self.K = len(type_ids_view)  # tokens per view (1+1+1+grid^2+grid^2 = 3 + 2*grid^2)
        self.register_buffer("type_ids_view", torch.tensor(type_ids_view, dtype=torch.long), persistent=False)

        # ----------- Learnable view summary token -----------
        self.view_cls = nn.Parameter(torch.zeros(1, 1, 1, d_model))  # [1,1,1,d]

        # ----------- Alternating intra/inter stacks -----------
        self.intra_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio=4.0, dropout=dropout) for _ in range(num_att_blocks)
        ])
        self.inter_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio=4.0, dropout=dropout) for _ in range(num_att_blocks)
        ])

        # ----------- Pose head reads from the view summary token -----------
        self.pose_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2*d_model), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(2*d_model, d_model), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_model, dim_out)
        )

    # ----- Tokenizers -----
    def _pose_tokens(self, R, T):
        B, N = R.shape[:2]
        pose_vec = torch.cat([R.reshape(B, N, 9), T], dim=-1)      # [B,N,12]
        return self.pose_proj(pose_vec)                            # [B,N,d]

    def _time_tokens(self, t, B, N, device):
        if isinstance(t, (int, float)):
            t = torch.full((B, N), int(t), device=device)
        elif t.ndim == 1:
            t = t[:, None].expand(-1, N)
        t = t.long()
        return self.time_embed(t)                                  # [B,N,d]

    def _image_tokens(self, imgs):
        # imgs: [B,N,3,H,W] → [B,N,grid^2,d]
        B, N = imgs.shape[:2]
        x = imgs.view(B*N, *imgs.shape[2:])                        # [B*N,3,H,W]
        x = self.img_backbone(x)                                   # [B*N,d,grid,grid]
        x = x.flatten(2).transpose(1, 2)                           # [B*N,grid^2,d]
        x = x + self.pos_img.unsqueeze(0)                          # add 2D PE
        return x.view(B, N, self.n_img_tokens, self.d_model)       # [B,N,grid^2,d]

    def _feat_tokens(self, feats):
        # feats: [B,N,1024,14,14] → [B,N,grid^2,d]
        B, N = feats.shape[:2]
        f = self.feat_proj(feats.view(B*N, *feats.shape[2:]))      # [B*N,d,14,14]
        f = F.adaptive_avg_pool2d(f, (self.grid_hw, self.grid_hw)) # [B*N,d,grid,grid]
        f = f.flatten(2).transpose(1, 2)                           # [B*N,grid^2,d]
        f = f + self.pos_feat.unsqueeze(0)                         # add 2D PE
        return f.view(B, N, self.n_feat_tokens, self.d_model)      # [B,N,grid^2,d]

    # ----- Forward -----
    def forward(self, R, T, feats, imgs, t):
        B, N = R.shape[:2]
        d = self.d_model
        K = self.K

        pose_tok = self._pose_tokens(R, T)                          # [B,N,d]
        time_tok = self._time_tokens(t if torch.is_tensor(t) else torch.tensor(t, device=R.device),
                                     B, N, device=R.device)         # [B,N,d]
        feat_tok = self._feat_tokens(feats)                         # [B,N,grid^2,d]
        img_tok  = self._image_tokens(imgs)                         # [B,N,grid^2,d]

        view_cls = self.view_cls.expand(B, N, 1, d)                 # [B,N,1,d]

        tokens = torch.cat([
            view_cls,                                               # [B,N,1,d]
            pose_tok.unsqueeze(2),                                  # [B,N,1,d]
            time_tok.unsqueeze(2),                                  # [B,N,1,d]
            feat_tok,                                               # [B,N,grid^2,d]
            img_tok                                                 # [B,N,grid^2,d]
        ], dim=2)                                                   # [B,N,K,d]  (K = 3 + 2*grid^2; with grid=3 → K=21)

        type_emb = self.token_type(self.type_ids_view.to(tokens.device))  # [K,d]
        tokens = tokens + type_emb[None, None, :, :]                      # [B,N,K,d]

        x = tokens
        for intra, inter in zip(self.intra_blocks, self.inter_blocks):
            x = x.reshape(B * N, K, d)            # intra (per view)
            x = intra(x)
            x = x.view(B, N, K, d)

            x = x.view(B, N * K, d)               # inter (across views)
            x = inter(x)
            x = x.view(B, N, K, d)

        view_summ = x[:, :, 0, :]                 # [B,N,d]
        pose_preds = self.pose_head(view_summ)    # [B,N,dim_out]
        return pose_preds