#!/usr/bin/env python
# coding: utf-8

# In[89]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from SO3 import so3_diffuser, SO3Algebra
from R3 import r3_diffuser
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import math
import math
import random
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from mpl_toolkits.mplot3d import Axes3D
import geoopt
from geoopt.optim import (RiemannianAdam)
Stiefel = geoopt.Stiefel()
from matplotlib import cm


# THIS CODE IS BASED ON THE FOLLOWING TWO PAPERS: 
# DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL ALIGNMENT by LEACH et al
# SE(3) diffusion model with application to protein backbone generation by Yim et al 
# 
# Operations on SO(3) have been taken from: https://github.com/qazwsxal/diffusion-extensions

# In[59]:


import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot

so3 = SO3Algebra()

def test_batch(batch_size: int = 8, dtype=torch.float32, device='cpu'):
    """
    Check that log_map, get_v, and exp_map agree with SciPy on a batch
    of random rotation matrices.
    """
    R_np = SciRot.random(batch_size).as_matrix()          # (B,3,3)
    R_t  = torch.tensor(R_np, dtype=dtype, device=device) # (B,3,3)

    S_skew   = so3.log_map(R_t)                 # (B,3,3)
    v_tensor = so3.get_v(S_skew)                # (B,3)
    R_recon  = so3.exp_map(v_tensor)            # (B,3,3)

    v_gt = SciRot.from_matrix(R_np).as_rotvec() # (B,3)

    v_err  = torch.linalg.norm(
        torch.tensor(v_gt, dtype=dtype, device=device) - v_tensor, dim=-1)
    R_err  = torch.linalg.norm(
        torch.tensor(R_np, dtype=dtype, device=device) - R_recon, dim=(-2, -1))

    print("Axis-angle error ‖v_gt − v_pred‖ per sample:")
    for i, e in enumerate(v_err.cpu().numpy()):
        print(f"  sample {i:02d}: {e:.3e}")

    print("\nFrobenius error ‖R − exp_map(v_pred)‖_F per sample:")
    for i, e in enumerate(R_err.cpu().numpy()):
        print(f"  sample {i:02d}: {e:.3e}")

test_batch(10)


# In[ ]:


import torch, torch.nn as nn
from typing import Literal

class SE3MLP(nn.Module):
    """
    Joint SE(3) diffusion network.

    Args
    ----
    d_model : width of the time-embedding (and hidden width for *joint* / *split*)
    head    : 'joint' | 'split' | 'residual'
    hidden_dim : width inside the residual head (ignored for other heads)
    n_blocks    : number of residual blocks      (ignored for other heads)
    dropout     : dropout prob inside residual blocks
    """
    HeadType = Literal['joint', 'split', 'residual']

    def __init__(
        self,
        d_model: int = 256,
        head: HeadType = 'joint',
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.head = head

        # ---------- time embedding ----------
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU()
        )

        # ---------- JOINT head (old) ----------
        self.mlp_joint = nn.Sequential(
            nn.Linear(12 + d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 6)
        )

        # ---------- SPLIT heads (old 'split') ----------
        self.mlp_rot = nn.Sequential(
            nn.Linear(9 + d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 3)
        )
        self.mlp_trans = nn.Sequential(
            nn.Linear(3 + d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 3)
        )

        # ---------- RESIDUAL head ----------
        if head == 'residual':
            self.residual_head = ResidualSE3Head(
                d_model=d_model,
                hidden_dim=hidden_dim,
                n_blocks=n_blocks,
                dropout=dropout
            )

    # ---- forward helpers --------------------------------------------------

    def _forward_joint(self, R_t, T_t, t):
        B = R_t.size(0)
        flat_rot = R_t.reshape(B, 9)
        feats    = torch.cat([flat_rot, T_t], dim=1)              # (B,12)
        t_emb    = self.time_embed(t.view(B, 1).float())          # (B,d_model)
        return self.mlp_joint(torch.cat([feats, t_emb], dim=1))   # (B,6)

    def _forward_split(self, R_t, T_t, t):
        B = R_t.size(0)
        flat_rot = R_t.reshape(B, 9)
        t_emb    = self.time_embed(t.view(B, 1).float())

        v_rot   = self.mlp_rot(torch.cat([flat_rot, t_emb], dim=1))
        v_trans = self.mlp_trans(torch.cat([T_t,     t_emb], dim=1))
        return torch.cat([v_rot, v_trans], dim=1)                 # (B,6)

    # -----------------------------------------------------------------------

    def forward(self, R_t: torch.Tensor, T_t: torch.Tensor, t: torch.Tensor):
        if self.head == 'joint':
            return self._forward_joint(R_t, T_t, t)
        elif self.head == 'split':
            return self._forward_split(R_t, T_t, t)
        elif self.head == 'residual':
            B = R_t.size(0)
            flat_rot = R_t.reshape(B, 9)
            feats    = torch.cat([flat_rot, T_t], dim=1)          # (B,12)
            t_emb    = self.time_embed(t.view(B, 1).float())      # (B,d_model)
            return self.residual_head(feats, t_emb)               # (B,6)
        else:
            raise ValueError(f"Unknown head style '{self.head}'")


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


# In[ ]:


class se3_diffuser:
    def __init__(self, T, batch_size=64, device="cpu", betas=None, split = False):
        self.device      = torch.device(device)
        self.batch_size  = batch_size

        self.so3 = so3_diffuser(T, batch_size, betas, device=device)
        self.r3  = r3_diffuser(T, batch_size, betas, device=device)

        self.model = SE3MLP(
            head='residual',      
            d_model=256,          
            hidden_dim=256,      
            n_blocks=3,         
            dropout=0.1         
        ).to(self.device)
        self.opt   = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.T = T    
        self.split = split

    def batch_se3(self, R_clean, T_clean, epoch = None):
        """
        R_clean : (B,3,3) 
        T_clean : (B,3)    
        """
        self.model.train()
        B = R_clean.size(0)

        t = torch.randint(1, self.T, (1,), device=self.device).item()

        R_noise, v_noise = self.so3.generate_noise(t)         
           
        T_noise          = self.r3.generate_noise((3,), t)      

        noise = T_noise / torch.sqrt(1 - self.r3.alpha_bars[int(t)]) 
        R_t = self.so3.add_noise(R_clean, R_noise, t)
        T_t = self.r3.add_noise(T_clean, T_noise, t)
        pred = self.model(R_t, T_t, torch.full((B,1), t, device=self.device))
        pred_rot, pred_trans = pred[:, :3], pred[:, 3:]

        loss = 0.5 * ((pred_rot - v_noise)**2).mean() + 0.5 * ((pred_trans - T_noise)**2).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


    @torch.no_grad()
    def sample(self, N=1, guidance=False, optim_steps=1, cost=None):
        R_t = torch.stack([ torch.from_numpy(Rot.random().as_matrix()).float() for _ in range(N) ]).to(self.device)
        T_t = torch.randn((N, 3), device=self.device)

        for t in reversed(range(1, self.T)):
            # predict noise
            eps = self.model(R_t, T_t,
                             torch.full((N,1), t, device=self.device))
            eps_rot, eps_trans = eps[:, :3], eps[:, 3:]

            # rotation reverse step
            R_t = self.so3._se_sample_batch(R_t, t, eps_rot,
                                            guidance, optim_steps, cost)[1]

            # translation reverse step
            T_t = self.r3._eu_sample_batch(T_t, t, eps_trans,
                                           guidance, optim_steps, cost)
        return R_t, T_t
    


# In[83]:


import torch 

def build_pose_dataset(device):
    # Pose 1  (rotvec = axis * angle)
    rotvec1 = np.array([0.0, 1.0, 0.0]) * (np.pi / 4)        
    R1 = torch.tensor(Rot.from_rotvec(rotvec1).as_matrix(),
                      dtype=torch.float32, device=device)
    print(R1)
    T1 = torch.tensor([1., 0., 0.], dtype=torch.float32, device=device)

    # Pose 2
    rotvec2 = np.array([0.0, -1.0, 0.0]) * (np.pi / 4)          
    R2 = torch.tensor(Rot.from_rotvec(rotvec2).as_matrix(),
                      dtype=torch.float32, device=device)
    print(R2)
    T2 = torch.tensor([0., 0., -1.], dtype=torch.float32, device=device)

    return [(R1, T1), (R2, T2)] 

def train_synthetic_pairs(diffuser, epochs=100, log_every=10000):
    B      = diffuser.batch_size
    device = diffuser.device
    dataset = build_pose_dataset(device)  
    K = len(dataset)                     

    losses = []
    for ep in range(1, epochs + 1):
        idx = torch.randint(high=K, size=(B,), device=device)  
        Rs, Ts = zip(*[dataset[i] for i in idx.cpu().tolist()])
        Rs = torch.stack(Rs)    
        Ts = torch.stack(Ts)    

        loss = diffuser.batch_se3(Rs, Ts, ep)
        if(loss <= 10): 
            losses.append(loss)

        if ep % log_every == 0:
            print(f"[pair SE(3)] epoch {ep:4d} | loss {loss:.6f}")

    return losses

diffuser = se3_diffuser(T=100, batch_size=16, device="cpu", split = False)

losses = train_synthetic_pairs(diffuser, epochs = 200000)

plt.plot(losses)


# In[84]:


R, T = diffuser.sample()
R = R.squeeze(0)
print("R", R)
print("T", T)

print(R.T@R)


# In[94]:


def plot_samples(num_samples=100, guidance=False, optim_steps=1, cost=None):
    rot_axes   = []
    trans_axes = []

    for _ in range(num_samples):
        R_sample, t_sample = diffuser.sample()     
        R_sample = R_sample.squeeze(0)             
        t_sample = t_sample.squeeze(0)             

        omega  = Rot.from_matrix(R_sample).as_rotvec()
        theta  = np.linalg.norm(omega)
        if theta > 1e-5:
            axis = omega / theta
        else:                                      
            axis = np.array([1.0, 0.0, 0.0])
        print(axis)
        rot_axes.append(axis)

        trans_axis = t_sample / (np.linalg.norm(t_sample) + 1e-8)
        trans_axes.append(trans_axis)

    rot_axes   = np.stack(rot_axes)    # (N,3)
    trans_axes = np.stack(trans_axes)  # (N,3)

    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(rot_axes[:,0],   rot_axes[:,1],   rot_axes[:,2], c='blue',  s=20, label='rotation axis')
    ax.scatter(trans_axes[:,0], trans_axes[:,1], trans_axes[:,2], c='red',   s=20, label='translation axis')

    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi,   100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper right")
    plt.show()

    return theta, rot_axes

def plot_samples_alt(num_samples=100, guidance=False, optim_steps=1, cost=None):
    rot_axes, trans_axes = [], []

    for _ in range(num_samples):
        R_sample, t_sample = diffuser.sample()
        R_sample, t_sample = R_sample.squeeze(0), t_sample.squeeze(0)

        omega = Rot.from_matrix(R_sample).as_rotvec()
        theta = np.linalg.norm(omega)
        axis  = omega / theta if theta > 1e-5 else np.array([1., 0., 0.])
        rot_axes.append(axis)

        t_sample = t_sample / (np.linalg.norm(t_sample) + 1e-8)

        trans_axes.append(t_sample) 

    rot_axes, trans_axes = np.stack(rot_axes), np.stack(trans_axes)

    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection='3d')

    cmap   = cm.get_cmap("tab20", num_samples)
    colours = cmap(np.arange(num_samples))

    for i, colour in enumerate(colours):
        ax.scatter(*rot_axes[i],  c=[colour], marker='^', s=50, label=None)
        ax.scatter(*trans_axes[i], c=[colour], marker='o', s=50, label=None)
        ax.plot([rot_axes[i,0], trans_axes[i,0]],
                [rot_axes[i,1], trans_axes[i,1]],
                [rot_axes[i,2], trans_axes[i,2]],
                c=colour, alpha=0.5, linewidth=1)

    u, v = np.linspace(0, 2*np.pi, 60), np.linspace(0, np.pi, 30)
    x, y = np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v))
    z    = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.15, linewidth=0.3)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

    return rot_axes, trans_axes


# In[95]:


""" GOAL DISTRIBUTION
    # Pose 1  (rotvec = axis * angle)
    rotvec1 = np.array([0.0, 1.0, 0.0]) * (np.pi / 4)          # 45° about x
    R1 = torch.tensor(Rot.from_rotvec(rotvec1).as_matrix(),
                      dtype=torch.float32, device=device)
    print(R1)
    T1 = torch.tensor([1., 0., 0.], dtype=torch.float32, device=device)

    # Pose 2
    rotvec2 = np.array([0.0, -1.0, 0.0]) * (np.pi / 4)          # 90° about y
    R2 = torch.tensor(Rot.from_rotvec(rotvec2).as_matrix(),
                      dtype=torch.float32, device=device)
    print(R2)
    T2 = torch.tensor([0., 0., -1.], dtype=torch.float32, device=device)
"""
plot_samples(num_samples = 100)

plot_samples_alt(num_samples = 100)


# In[ ]:




