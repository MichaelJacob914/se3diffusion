#!/usr/bin/env python
# coding: utf-8

# In[11]:

from SO3 import so3_diffuser, SO3Algebra
from R3 import r3_diffuser
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


def log_time(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# In[12]:


class PairDataset(Dataset):
    """
    Loads the samples we saved with `np.savez_compressed`, computes the
    relative pose (R 12, t 12) on-the-fly and returns the two-channel depth
    map that the network will be trained on.
    Each item:
        img1, img2 : (3,H,W) - float in [0,1]
        depths     : (2,H,W) - float32  (depth1, depth2)
        R          : (3,3)
        t          : (3,)
    """
    def __init__(self, root: str):
        self.root   = Path(root)
        self.files  = sorted(self.root.glob("*.npz"))
        self.to_t   = transforms.ToTensor()            

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = np.load(self.files[idx])

        img1 = self.to_t(sample["img1"])                 # (3,H,W) in [0,1]
        img2 = self.to_t(sample["img2"])

        d1   = torch.from_numpy(sample["depth1"])        # (H,W)
        d2   = torch.from_numpy(sample["depth2"])
        depths = torch.stack((d1, d2), dim=0)            # (2,H,W)

        R1 = torch.from_numpy(sample["R1"])              # (3,3)
        t1 = torch.from_numpy(sample["t1"])
        R2 = torch.from_numpy(sample["R2"])
        t2 = torch.from_numpy(sample["t2"])

        R12 = R1.T @ R2                                # (3,3)
        t12 = t2 - t1 @ R1.T @ R2                         # (3,)

        img1 = torch.from_numpy(sample["img1"])   # (3,H,W)
        img2 = torch.from_numpy(sample["img2"])   # (3,H,W)
        rgb  = torch.cat([img1, img2], dim=0)  # (6,H,W)

        return {
            "img1":  img1,
            "img2":  img2,
            "rgb": rgb,   
            "depths": depths,   
            "R":     R12,
            "t":     t12
        }



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

class DepthRGBGuidedSE3Model(nn.Module):
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

class CrossModalSE3Model(nn.Module):
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
        
class se3_diffuser:
    def __init__(self, T, batch_size=64, device="cuda", betas=None, model_path=None, save_model=False, save_path=None):
        self.device      = torch.device(device)
        self.batch_size  = batch_size
        self.save_model  = save_model
        self.save_path   = save_path

        self.so3 = so3_diffuser(T, batch_size, betas, device=device)
        self.r3  = r3_diffuser(T, batch_size, betas, device=device)

        self.model = CrossModalSE3Model(d_model=256).to(device)

        if model_path is not None:
            print(f"Loading model weights from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.opt = torch.optim.AdamW(self.model.parameters(),   
                                     lr=1e-5,
                                     weight_decay=0.0, betas=(0.9, 0.999))
        self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=50_000, eta_min=1e-5 * 0.1
        )
        self.clip_grad_norm = 1.0
        self.T   = T

    def iteration(self, R_clean, T_clean, depths, rgb, gamma = 1, step=None):
        """
        R_clean : (B, 3, 3)
        T_clean : (B, 3)
        depths  : (B, 2, H, W) 
        """
        self.model.train()
        B = R_clean.size(0)

        t = torch.randint(1, self.T, (1,), device=self.device).item()

        R_noise, v_noise = self.so3.generate_noise(t, batch_size = B)
        T_noise          = self.r3.generate_noise((3,), t = t, batch_size = B)

        R_t = self.so3.add_noise(R_clean, R_noise, t)
        T_t = self.r3.add_noise(T_clean, T_noise, t)
        sigma_t   = torch.sqrt(1 - self.r3.alpha_bars[int(t)])
        sigma_so3   = torch.sqrt(1 - self.so3.alpha_bars[int(t)])
        t_target     = T_noise / sigma_t
        v_target  = v_noise / sigma_so3 
        
    
        depths = depths.to(self.device, non_blocking=True)
        rgb = rgb.to(self.device, non_blocking= True)
        t_tensor = torch.full((B, 1), t, device=self.device).float()

        pred = self.model(R_t, T_t, torch.full((B, 1), float(t), device=self.device, dtype=torch.float32), depths, rgb)
        pred_rot, pred_trans = pred[:, :3], pred[:, 3:]

        loss = 0.5 * ((pred_rot - v_target)**2).mean() + gamma * 0.5 * ((pred_trans - t_target)**2).mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.opt.step()
        self.lr_sched.step()
        return loss.item()
    
    def train(self, dataloader, num_epochs=1, log_every=1000):
        """
        dataloader yields dicts with keys:
            'img1', 'img2'  : (B,3,H,W) float32 in [0,1]
            'R'             : (B,3,3)
            't'             : (B,3)
        Grayscale depth-proxy is created on-the-fly:
            depth = mean(img, dim=1, keepdim=True)   # (B,1,H,W)
        """
        losses = []
        for epoch in range(num_epochs):
            running = 0.0
            n_seen  = 0

            for step, batch in enumerate(dataloader):
                R_clean = batch["R"].to(self.device)   # (B,3,3)
                
                T_clean = batch["t"].to(self.device)   # (B,3)

                img1 = batch["img1"].to(self.device)   # (B,3,H,W)
                img2 = batch["img2"].to(self.device)

                depths = batch["depths"].to(self.device)
                rgb = batch["rgb"].to(self.device)

                loss_val = self.iteration(R_clean, T_clean, depths, rgb, step)
                
                running += loss_val * R_clean.size(0)
                n_seen  += R_clean.size(0)

                losses.append(loss_val)

                if step % log_every == 0:
                    print(f"[epoch {epoch}  step {step:04d}]  loss {loss_val:.6f}")

            epoch_loss = running / n_seen
            
            print(f"===> epoch {epoch} done, mean loss {epoch_loss:.6f}")
        
        if self.save_model and self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "opt":   self.opt.state_dict(),
                    "epoch": epoch,
                },
                self.save_path            # ← just use the full path, no extra suffix
            )


        return losses

    def train_alt(self, dataloader, num_epochs=1, log_every=1000):
        """
        One-time dataloader pass to store all (R, T, depth, rgb) samples in memory.
        Then train by sampling batches from this in-memory dataset.

        dataloader yields:
            - 'R':    (B,3,3)
            - 't':    (B,3)
            - 'depths': (B,2,H,W)
            - 'rgb':  (B,6,H,W)
        """
        log_time("Start loading dataset")
        memory = []
        for batch in dataloader:
            R_batch = batch["R"].to(self.device)        # (B,3,3)
            T_batch = batch["t"].to(self.device)        # (B,3)
            D_batch = batch["depths"].to(self.device)   # (B,2,H,W)
            I_batch = batch["rgb"].to(self.device)      # (B,6,H,W)

            for R, T, D, I in zip(R_batch, T_batch, D_batch, I_batch):
                memory.append((R, T, D, I))
        log_time("Loaded dataset")
        print(f"Loaded {len(memory)} total samples into memory.")

        losses = []
        B = self.batch_size
        print("B", B)
        print("length", len(memory))
        steps_per_epoch = 1

        for epoch in range(num_epochs):
            if(epoch % log_every == 0):
                log_time("1000 epochs")
            running = 0.0
            n_seen  = 0

            for step in range(steps_per_epoch):
                # Sample B random indices from memory
                indices = torch.randint(low=0, high=len(memory), size=(B,), device=self.device)
                R_list, T_list, D_list, I_list = zip(*(memory[i] for i in indices))

                R_clean = torch.stack(R_list).to(self.device)  # (B,3,3)
                T_clean = torch.stack(T_list).to(self.device)  # (B,3)
                depths  = torch.stack(D_list).to(self.device)  # (B,2,H,W)
                rgb     = torch.stack(I_list).to(self.device)  # (B,6,H,W)

                loss_val = self.iteration(R_clean, T_clean, depths, rgb)

                running += loss_val * B
                n_seen  += B

            epoch_loss = running / n_seen
            losses.append(epoch_loss)
            print(f"===> epoch {epoch} done, mean loss {epoch_loss:.6f}")

        if self.save_model and self.save_path is not None:
            torch.save(self.model.state_dict(), self.save_path)

        return losses

    @torch.no_grad()
    def sample(self,depths, rgb, N=1, guidance=False, optim_steps=1, cost=None):
        R_t = torch.stack([ torch.from_numpy(Rot.random().as_matrix()).float() for _ in range(N) ]).to(self.device)
        T_t = torch.randn((N, 3), device=self.device)

        depths = depths.to(self.device, non_blocking=True)

        for t in reversed(range(1, self.T)):
            # predict noise
    
            eps = self.model(R_t, T_t,
                             torch.full((N,1), t, device=self.device), depths, rgb)
            eps_rot, eps_trans = eps[:, :3], eps[:, 3:]

            # rotation reverse step
            R_t = self.so3._se_sample_batch(R_t, t, eps_rot,
                                            guidance, optim_steps, cost)[1]

            # translation reverse step
            T_t = self.r3._eu_sample_batch(T_t, t, eps_trans,
                                           guidance, optim_steps, cost)
        print("Rotation", R_t)
        print("Translation", T_t)
        return R_t, T_t
    


def axis_from_R(R: torch.Tensor):
    """
    Convert a rotation matrix (3×3 torch) to its axis (unit 3-vector).
    """
    rotvec = Rot.from_matrix(R.cpu().numpy()).as_rotvec()
    angle  = torch.linalg.norm(torch.from_numpy(rotvec))
    print(angle)
    axis   = rotvec / (angle + 1e-8)
    return axis

@torch.no_grad()
def visualize_pose_axes(se3, dataloader, plot_name, k=4, N=10, device="cuda"):
    """
    For k samples from the dataloader, generate N predicted poses from the same input
    and compare each to the ground-truth pose using axis visualization and angular error.
    
    Args:
        se3        : diffusion model with `.sample()` method
        dataloader : torch DataLoader object (dataset must support indexing)
        k          : number of poses to sample from the dataloader
        N          : number of diffusion samples per pose
        device     : computation device
    """
    se3.model.eval()
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(projection='3d')
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])

    colors = plt.cm.tab10.colors
    k = min(k, len(dataloader))
    chosen_steps = random.sample(range(len(dataloader)), k)
    chosen_steps = [0,1,2,3]

    for j, step_idx in enumerate(chosen_steps):
        batch = dataloader.dataset[step_idx]
        depths = batch["depths"].unsqueeze(0).to(device)  # (1,2,H,W)
        rgb    = batch["rgb"].unsqueeze(0).to(device)     # (1,6,H,W)
        R_gt   = batch["R"].unsqueeze(0).to(device)       # (1,3,3)
        t_gt   = batch["t"].unsqueeze(0).to(device)       # (1,3)

        axis_gt = axis_from_R(R_gt[0])
        ax.scatter(*axis_gt, color=colors[j % 10], marker='o', s=80,
                   label=f"GT {j}")

        for n in range(N):
            R_pred, t_pred = se3.sample(depths, rgb, N=1, guidance=False,
                                        optim_steps=1, cost=None)
            R_pred = R_pred[0]
            t_pred = t_pred[0]

            axis_pred = axis_from_R(R_pred)
            ax.scatter(*axis_pred, color=colors[j % 10], marker='x', s=40)

            cos_theta = 0.5 * (torch.trace(R_gt[0].T @ R_pred) - 1.0)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            ang_err   = torch.acos(cos_theta).item() * 180.0 / np.pi

            trans_err = torch.norm(t_gt[0] - t_pred).item()

            print(f"[Pose {j:02d} | Sample {n:02d}]  Δθ = {ang_err:6.2f}°   ‖Δt‖ = {trans_err:7.4f}")

    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi,   100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)

    ax.legend(loc="upper right")
    out_path = Path(f"{plot_name}/axis_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)  

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved axis-comparison plot to {out_path.resolve()}")
