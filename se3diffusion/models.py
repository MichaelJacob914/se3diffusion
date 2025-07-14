#!/usr/bin/env python
# coding: utf-8

# In[8]:

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
from typing import List, Dict, Tuple, Literal
import os
import time
from mpl_toolkits.mplot3d import Axes3D  



# In[9]:


#TAKEN FROM PAPER
#NOT MY CODE


class ResidualSE3Head(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, n_blocks: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(12 + d_model, hidden_dim)

        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(dropout)
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.out_rot   = nn.Linear(hidden_dim, 3)
        self.out_trans = nn.Linear(hidden_dim, 3)

    def forward(self, x_flat: torch.Tensor, t_emb: torch.Tensor):
        x = self.input_proj(torch.cat([x_flat, t_emb], dim=1))
        for blk in self.blocks:
            x = x + blk(x)                    # residual connection
        rot   = self.out_rot(x)
        trans = self.out_trans(x)
        return torch.cat([rot, trans], dim=1)

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, p: float = .1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Dropout(p)
        )
    def forward(self, x):                     # (B,d)
        return x + self.block(x)

class DepthRGBSE3Model(nn.Module):
    """
    Predict the ε-noise on (R,t) from   depth + RGB + time-step.
    Depth   : 2×H×W   (z-maps for the two views)
    RGB     : 3×H×W   (concatenated colour images, already cropped to 256×256)
    """
    def __init__(self,
                 d_model: int = 256,
                 n_residual: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.depth_enc = nn.Sequential(
            nn.Conv2d(2,  32, 5, stride=2, padding=2), nn.ReLU(),   # 128×128
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),   # 64×64
            nn.AdaptiveAvgPool2d(1),  # → (B,64,1,1)
            nn.Flatten()              # → (B,64)
        )

        self.rgb_enc = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32,64, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()  # (B, 64)
        )

        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        in_dim = 64 + 64 + 9 + 3 + d_model   # depth + rgb + R + t + τ-emb
        self.input_proj = nn.Linear(in_dim, d_model)

        self.bottleneck = nn.Sequential(
            nn.LayerNorm(d_model), nn.SiLU(),
            nn.Linear(d_model, d_model*2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model)
        )

        self.res_blocks = nn.Sequential(
            *[ResidualMLPBlock(d_model, dropout) for _ in range(n_residual)]
        )

        self.out_rot   = nn.Linear(d_model, 3)   # so(3)-vector
        self.out_trans = nn.Linear(d_model, 3)   # R³  noise

    def forward(self,
                R_t: torch.Tensor,     # (B,3,3)
                T_t: torch.Tensor,     # (B,3)
                t:   torch.Tensor,     # (B,1)  or scalar
                D:   torch.Tensor,     # (B,2,H,W)  depth stack
                I:   torch.Tensor):    # (B,6,H,W)  RGB stack
        B = R_t.size(0)

        t = t.to(dtype=torch.float32, device=R_t.device)
        if t.ndim == 0:      # scalar
            t = t.view(1, 1).expand(B, 1)
        elif t.ndim == 1:    # (B,)
            t = t.view(B, 1)
        elif t.ndim == 2:    # already (B,1)
            pass
        else:
            raise ValueError("Unexpected time tensor shape")

        t_emb   = self.time_embed(t.float())               # (B,d)
        d_feat  = self.depth_enc(D)                # (B,64)
        rgb_feat= self.rgb_enc(I)                  # (B,64)
        R_flat  = R_t.reshape(B, 9)                # (B,9)

        x = torch.cat([R_flat, T_t, t_emb, d_feat, rgb_feat], dim=1)
        x = self.input_proj(x)
        x = self.bottleneck(x)
        x = self.res_blocks(x)

        return torch.cat([self.out_rot(x), self.out_trans(x)], dim=1)  # (B,6)

class TransformerDepthRGBSE3Model(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_layers=6, dropout=0.1):
        super().__init__()

        # Pose embedding: 9 for R, 3 for t
        self.pose_proj = nn.Sequential(
            nn.Linear(12, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        # RGB encoder (6 channels = stacked pair of RGBs)
        self.rgb_enc = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),    # (B,64)
            nn.Linear(64, d_model)
        )

        # Depth encoder (2 channels = stacked depth)
        self.depth_enc = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, d_model)
        )

        # Transformer-style fusion blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 6)  # 3 for rotation, 3 for translation noise
        )

    def forward(self, R_t, T_t, t, D, I):
        B = R_t.size(0)

        # Time embedding
        if isinstance(t, (int, float)):
            t = torch.full((B, 1), float(t), device=R_t.device)
        elif t.ndim == 1:
            t = t.view(B, 1).float()
        t_emb = self.time_embed(t.float())  # (B, d)

        # Pose embedding
        pose_vec = torch.cat([R_t.reshape(B, 9), T_t], dim=1)
        pose_feat = self.pose_proj(pose_vec)  # (B, d)

        # Image features
        rgb_feat = self.rgb_enc(I)    # (B, d)
        depth_feat = self.depth_enc(D)  # (B, d)

        # Stack for fusion: (B, 4, d)
        fusion_input = torch.stack([pose_feat, t_emb, rgb_feat, depth_feat], dim=1)

        # Transformer fusion
        fused = self.fusion_blocks(fusion_input)  # (B, 4, d)
        out = fused[:, 0]  # Take pose token (index 0) as final rep

        return self.output_proj(out)  # (B, 6)
        
class TransformerFeatureRGBSE3Model(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_layers=6, dropout=0.1):
        super().__init__()

        # Pose embedding: 9 for R, 3 for t
        self.pose_proj = nn.Sequential(
            nn.Linear(12, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        # RGB encoder (stacked RGB pair: 6 channels)
        self.rgb_enc = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, d_model)
        )

        # Feature encoder (stacked pair: 2×1024 channels)
        self.feat_enc = nn.Sequential(
            nn.Conv2d(2048, 512, 3, stride=2, padding=1), nn.ReLU(),  # [B,512,16,16]
            nn.Conv2d(512, 128, 3, stride=2, padding=1),  nn.ReLU(),  # [B,128,8,8]
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),                   # [B,128]
            nn.Linear(128, d_model)
        )

        # Transformer-style fusion blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 6)  # (rotation, translation) noise
        )

    def forward(self, R_t, T_t, t, F, I):
        """
        Inputs:
            R_t, T_t : (B, 3, 3), (B, 3)
            t        : (B,) or scalar
            F        : (B, 2, 1024, 32, 32) — stacked features
            I        : (B, 6, 256, 256)     — stacked RGB pair
        """
        B = R_t.size(0)
        if I.ndim == 3:
            I = I.unsqueeze(0)  
        # Time embedding
        if isinstance(t, (int, float)):
            t = torch.full((B, 1), float(t), device=R_t.device)
        elif t.ndim == 1:
            t = t.view(B, 1).float()
        t_emb = self.time_embed(t.float())  # (B, d)

        # Pose embedding
        pose_vec = torch.cat([R_t.reshape(B, 9), T_t], dim=1)
        pose_feat = self.pose_proj(pose_vec)  # (B, d)

        # Image encoding
        rgb_feat = self.rgb_enc(I)  # (B, d)

        # Feature encoding (stack 2 feats → 2048 channel input)
        feat_input = torch.cat([F[:, 0], F[:, 1]], dim=1)  # (B, 2048, 32, 32)
        feat_feat = self.feat_enc(feat_input)              # (B, d)

        # Fusion: [pose, time, rgb, features]
        fusion_input = torch.stack([pose_feat, t_emb, rgb_feat, feat_feat], dim=1)
        fused = self.fusion_blocks(fusion_input)  # (B, 4, d)

        return self.output_proj(fused[:, 0])  # (B, 6)

