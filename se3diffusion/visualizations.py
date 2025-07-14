import argparse

from SO3 import so3_diffuser, SO3Algebra
from R3 import r3_diffuser
import models
import datasets
from datasets import RGBDepthDataset, RGBFeatureDataset
from torch.utils.data import Subset, DataLoader
from pathlib import Path
import os
import random
import matplotlib.pyplot as plt

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
def visualize_pose_axes(se3, dataloader, plot_name, conditioning, k=4, N=10, device="cuda"):
    """
    For k samples from the dataloader, generate N predicted poses from the same input
    and compare each to the ground-truth pose using axis visualization and angular error.
    """
    se3.model.eval()
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(projection='3d')
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])

    colors = plt.cm.tab10.colors
    k = min(k, len(dataloader))
    chosen_steps = [0, 1, 2, 3] if k >= 4 else random.sample(range(len(dataloader)), k)

    for j, step_idx in enumerate(chosen_steps):
        batch = dataloader.dataset[step_idx]
        R_gt   = batch["R"].unsqueeze(0).to(device)
        t_gt   = batch["t"].unsqueeze(0).to(device)

        if conditioning == "depths":
            depths = batch["depths"].unsqueeze(0).to(device)   # (1,2,H,W)
            rgb    = batch["rgb"].unsqueeze(0).to(device)      # (1,6,H,W)
            sample_args = {"depths": depths, "rgb": rgb}

        elif conditioning == "features":
            rgb     = batch["rgb"]     # (B,6,H,W)
            feats   = batch["feats"]   # (B,2,1024,32,32)
            if rgb.ndim == 3:
                rgb = rgb.unsqueeze(0)  
            if feats.ndim == 4: 
                feats = feats.unsqueeze(0)
            sample_args = {"feats": feats, "rgb": rgb}

        else:
            raise ValueError(f"Unsupported conditioning: {conditioning}")

        axis_gt = axis_from_R(R_gt[0])
        ax.scatter(*axis_gt, color=colors[j % 10], marker='o', s=80, label=f"GT {j}")

        for n in range(N):
            R_pred, t_pred = se3.sample(**sample_args, N=1, guidance=False, optim_steps=1, cost=None)
            R_pred, t_pred = R_pred[0], t_pred[0]

            axis_pred = axis_from_R(R_pred)
            ax.scatter(*axis_pred, color=colors[j % 10], marker='x', s=40)

            cos_theta = 0.5 * (torch.trace(R_gt[0].T @ R_pred) - 1.0)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            ang_err   = torch.acos(cos_theta).item() * 180.0 / np.pi
            trans_err = torch.norm(t_gt[0] - t_pred).item()

            print(f"[Pose {j:02d} | Sample {n:02d}]  Δθ = {ang_err:6.2f}°   ‖Δt‖ = {trans_err:7.4f}")

    # Draw unit sphere
    u, v = np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 100)
    x, y = np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)

    ax.legend(loc="upper right")
    out_path = Path(f"{plot_name}/axis_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved axis-comparison plot to {out_path.resolve()}")
