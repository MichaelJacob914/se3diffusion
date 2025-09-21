import argparse
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
paths = [HERE / "gta", HERE / "gta" / "source"]
sys.path[:0] = [str(p) for p in paths if p.exists() and str(p) not in sys.path]
from SO3n import so3_diffuser, SO3Algebra
from R3n import r3_diffuser
import se3n_models
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
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
import random

@torch.no_grad()
def visualize_pose_axes_full(
    se3, dataloader, plot_name, conditioning, k=4, N=10, device="cuda",
    rot_thresh_deg=15.0, trans_thresh=0.1, dir_thresh_deg=15.0
):
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import torch
    from scipy.spatial.transform import Rotation as Rot

    se3.model.eval()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    colors = plt.cm.tab10.colors

    def rot_geodesic_deg(Ra, Rb):
        R = Ra.transpose(-2, -1) @ Rb
        tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        cos_th = 0.5 * (tr - 1.0)
        cos_th = torch.clamp(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cos_th) * (180.0 / np.pi)

    def rel_from_two(R1, t1, R2, t2):
        R_rel = R2 @ R1.transpose(-2, -1)
        t_rel = t2 - (R2 @ R1.transpose(-2, -1) @ t1.unsqueeze(-1)).squeeze(-1)
        return R_rel, t_rel

    def dir_angle_deg(a, b, eps=1e-12):
        # angle between directions in degrees
        na = torch.linalg.norm(a)
        nb = torch.linalg.norm(b)
        if na < eps or nb < eps:
            return float("inf")  # undefined direction → count as failure
        cos = torch.clamp((a @ b) / (na * nb), -1.0, 1.0)
        return float(torch.arccos(cos) * (180.0 / np.pi))

    # ---- per-entry metrics (best draw) ----
    per_entry = []
    tot_cnt = rot_ok_cnt = trans_ok_cnt = joint_ok_cnt = 0

    # ---- across-all-draws metrics ----
    tot_draws = rot_ok_draws = trans_ok_draws = joint_ok_draws = dir_ok_draws = 0

    drawn = 0
    for batch in dataloader:
        if drawn >= k:
            break

        take = min(k - drawn, batch["R"].shape[0])
        for b in range(take):
            j = drawn

            # ---- GT (one sample) ----
            if(conditioning == "features"):
                R1 = batch["R1"][b].to(device)  # [3,3]
                t1 = batch["t1"][b].to(device)  # [3]
                R2 = batch["R2"][b].to(device)
                t2 = batch["t2"][b].to(device)
            elif(conditioning == "CO3D"):
                R = batch["R"][b].to(device)
                T = batch["t"][b].to(device)
                R1, R2 = R[0].contiguous(), R[1].contiguous()   
                t1, t2 = T[0].contiguous(), T[1].contiguous()
                
            R_rel_gt, t_rel_gt = rel_from_two(R1, t1, R2, t2)

            # Plot GT axis + translation direction
            rot_gt = Rot.from_matrix(R_rel_gt.detach().cpu().numpy()).as_rotvec()
            theta_gt = np.linalg.norm(rot_gt)
            axis_gt = rot_gt / (theta_gt + 1e-12) if theta_gt > 1e-8 else np.array([1.0, 0.0, 0.0])
            ax.scatter(*axis_gt, color=colors[j % 10], marker='o', s=80, label=f"GT rot {j}")

            tdir_gt_np = t_rel_gt.detach().cpu().numpy()
            tdir_gt_np = tdir_gt_np / (np.linalg.norm(tdir_gt_np) + 1e-12) if np.linalg.norm(tdir_gt_np) > 1e-8 else np.array([1.0, 0.0, 0.0])
            ax.scatter(*tdir_gt_np, color=colors[j % 10], marker='^', s=80, label=f"GT trans {j}")

            # ---- conditioning for this sample ----
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
                if depths.ndim >= 3 and depths.shape[0] in (2,):
                    depths = depths.unsqueeze(0)  # [1,N,...]
                sample_args = {"depths": depths, "rgb": rgb}

            elif conditioning == "features":
                rgb   = batch["rgb"][b]
                feats = batch["feats"][b]
                sample_args = {"feats": feats.to(device), "rgb": rgb.to(device)}
            elif conditioning == "CO3D":
                import contextlib
                p       = next(se3.model.parameters())
                dev     = p.device
                mdtype  = p.dtype

                # first two views → device (keep fp32 for extractor)
                imgs_pair = batch["imgs"][b][:2].to(dev, dtype=torch.float32, non_blocking=True).contiguous()

                # DUSt3R on same device (AMP if CUDA), then cast feats/rgb to model dtype
                amp = torch.autocast(device_type="cuda", dtype=torch.float16) if dev.type == "cuda" else contextlib.nullcontext()
                with amp:
                    feats_pair = se3.extractor(imgs_pair)

                feats_pair = feats_pair.clone().to(dev, dtype=mdtype, non_blocking=True).contiguous()
                rgb_pair   = imgs_pair.to(dev, dtype=mdtype, non_blocking=True).contiguous()

                sample_args = {"feats": feats_pair, "rgb": rgb_pair}

            else:
                raise ValueError(f"Unsupported conditioning: {conditioning}")

            # ---- search best over N draws & both permutations; also count per-draw metrics ----
            best_ang_deg = float("inf")
            best_trans_l2 = float("inf")
            best_dir_deg = float("inf")

            for n in range(N):
                R_pair, t_pair = se3.sample(**sample_args, B=1, N=2, guidance=False, optim_steps=1, cost=None)
                R0, R1p = R_pair[0], R_pair[1]
                t0, t1p = t_pair[0], t_pair[1]

                # two directions
                R_rel_01, t_rel_01 = rel_from_two(R0,  t0,  R1p, t1p)
                R_rel_10, t_rel_10 = rel_from_two(R1p, t1p, R0,  t0)

                # errors
                ang01 = rot_geodesic_deg(R_rel_01, R_rel_gt).item()
                ang10 = rot_geodesic_deg(R_rel_10, R_rel_gt).item()
                trans01 = torch.norm(t_rel_01 - t_rel_gt).item()
                trans10 = torch.norm(t_rel_10 - t_rel_gt).item()

                # translation direction angle (deg) for both permutations
                dir01 = dir_angle_deg(t_rel_01, t_rel_gt)
                dir10 = dir_angle_deg(t_rel_10, t_rel_gt)

                # choose permutation for plotting and "best" selection (min ang+trans)
                if (ang01 + trans01) <= (ang10 + trans10):
                    ang_deg, trans_l2, dir_deg = ang01, trans01, dir01
                    R_rel_pred, t_rel_pred = R_rel_01, t_rel_01
                else:
                    ang_deg, trans_l2, dir_deg = ang10, trans10, dir10
                    R_rel_pred, t_rel_pred = R_rel_10, t_rel_10

                # ---- per-draw counters (across all draws) ----
                tot_draws += 1
                r_ok = ang_deg < rot_thresh_deg
                t_ok = trans_l2 < trans_thresh
                d_ok = dir_deg < dir_thresh_deg
                rot_ok_draws   += int(r_ok)
                trans_ok_draws += int(t_ok)
                joint_ok_draws += int(r_ok and t_ok)
                dir_ok_draws   += int(d_ok)

                # plot the chosen one for this draw
                rot_pred = Rot.from_matrix(R_rel_pred.detach().cpu().numpy()).as_rotvec()
                theta = np.linalg.norm(rot_pred)
                axis_pred = rot_pred / (theta + 1e-12) if theta > 1e-8 else np.array([1.0, 0.0, 0.0])
                ax.scatter(*axis_pred, color=colors[j % 10], marker='x', s=40)

                tdir_pred_np = t_rel_pred.detach().cpu().numpy()
                tdir_pred_np = tdir_pred_np / (np.linalg.norm(tdir_pred_np) + 1e-12) if np.linalg.norm(tdir_pred_np) > 1e-8 else np.array([1.0, 0.0, 0.0])
                ax.scatter(*tdir_pred_np, color=colors[j % 10], marker='s', s=40)

                # keep the best for per-entry report
                if (ang_deg + trans_l2) < (best_ang_deg + best_trans_l2):
                    best_ang_deg = ang_deg
                    best_trans_l2 = trans_l2
                    best_dir_deg = dir_deg

                print(f"[k={j:02d} | draw {n:02d}]  Δθ_rel={ang_deg:6.2f}°  "
                      f"‖Δt_rel‖={trans_l2:7.4f}  dirΔ={dir_deg:6.2f}°")

            # ---- per-entry (best-draw) accuracy ----
            rot_ok   = best_ang_deg   < rot_thresh_deg
            trans_ok = best_trans_l2  < trans_thresh
            dir_ok   = best_dir_deg   < dir_thresh_deg
            joint_ok = rot_ok and trans_ok

            per_entry.append({
                "k": j,
                "rot_deg": best_ang_deg,
                "trans_l2": best_trans_l2,
                "dir_deg": best_dir_deg,
                "rot_ok": rot_ok,
                "trans_ok": trans_ok,
                "dir_ok_best": dir_ok,
                "joint_ok": joint_ok,
            })

            tot_cnt += 1
            rot_ok_cnt   += int(rot_ok)
            trans_ok_cnt += int(trans_ok)
            joint_ok_cnt += int(joint_ok)

            drawn += 1
            if drawn >= k:
                break

    # unit sphere
    u, v = np.linspace(0, 2*np.pi, 100), np.linspace(0, np.pi, 100)
    x, y = np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)

    ax.legend(loc="upper right", fontsize=8)
    out_path = Path(plot_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RELATIVE rotation+translation axis plot to {out_path.resolve()}")

    # ---- per-entry + overall summaries ----
    print("\nPer-entry (best draw) accuracy @ "
          f"rot<{rot_thresh_deg}°, trans<{trans_thresh}, dir<{dir_thresh_deg}°:")
    for e in per_entry:
        print(f"  k={e['k']:02d} | rot={e['rot_deg']:6.2f}° | trans={e['trans_l2']:7.4f} "
              f"| dirΔ={e['dir_deg']:6.2f}° | rot_ok={int(e['rot_ok'])} "
              f"trans_ok={int(e['trans_ok'])} dir_ok={int(e['dir_ok_best'])} "
              f"joint_ok={int(e['joint_ok'])}")

    if tot_cnt > 0:
        rot_acc   = rot_ok_cnt   / tot_cnt
        trans_acc = trans_ok_cnt / tot_cnt
        joint_acc = joint_ok_cnt / tot_cnt
        print(f"\nOverall (best-draw-per-entry) over {tot_cnt} entries:"
              f"\n  rot_acc   (<{rot_thresh_deg}°): {rot_acc:.3f}"
              f"\n  trans_acc (<{trans_thresh}):   {trans_acc:.3f}"
              f"\n  joint_acc (both):              {joint_acc:.3f}")

    if tot_draws > 0:
        rot_acc_d   = rot_ok_draws   / tot_draws
        trans_acc_d = trans_ok_draws / tot_draws
        joint_acc_d = joint_ok_draws / tot_draws
        dir_acc_d   = dir_ok_draws   / tot_draws
        print(f"\nAcross ALL draws ({tot_draws} draws total):"
              f"\n  rot_acc_draws   (<{rot_thresh_deg}°): {rot_acc_d:.3f}"
              f"\n  trans_acc_draws (<{trans_thresh}):   {trans_acc_d:.3f}"
              f"\n  dir_acc_draws   (<{dir_thresh_deg}°): {dir_acc_d:.3f}"
              f"\n  joint_acc_draws (rot & trans):       {joint_acc_d:.3f}")
              
@torch.no_grad()
def visualize_pose_predictions_with_inversion(se3, dataloader, plot_name, conditioning, k=4, N=10, device="cuda"):
    """
    For k samples from the dataloader, visualize:
    - Ground-truth relative pose
    - Ground-truth inverse pose
    - N model predictions from forward and inverse inputs
    """
    se3.model.eval()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])

    color_cycle = plt.cm.tab20.colors  # 20 distinct colors
    k = min(k, len(dataloader))
    chosen_steps = random.sample(range(len(dataloader)), k)

    for j, step_idx in enumerate(chosen_steps):
        batch = dataloader.dataset[step_idx]

        # Ground truth and relative poses
        R_rel = batch["R"].to(device)
        t_rel = batch["t"].to(device)
        R1 = batch["R1"].to(device)
        R2 = batch["R2"].to(device)
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)

        R_inv_gt = R_rel.T
        t_inv_gt = -t_rel @ R_rel.T

        # Assign unique colors for GT fwd and inv for this sample
        color_fwd = color_cycle[(2 * j) % len(color_cycle)]
        color_inv = color_cycle[(2 * j + 1) % len(color_cycle)]

        # Plot GT poses
        ax.scatter(*axis_from_R(R_rel), color=color_fwd, marker='o', s=80, label=f"GT Fwd {j}")
        ax.scatter(*axis_from_R(R_inv_gt), color=color_inv, marker='o', s=80, label=f"GT Inv {j}")

        # Sample inputs
        feats = batch["feats"].unsqueeze(0).to(device)
        rgb   = batch["rgb"].unsqueeze(0).to(device)

        # Inverted inputs
        feats_inv = batch["feats"].flip(0).unsqueeze(0).to(device)
        rgb_inv   = torch.cat([batch["rgb"][3:], batch["rgb"][:3]], dim=0).unsqueeze(0).to(device)

        for n in range(N):
            R_fwd, t_fwd = se3.sample(feats=feats, rgb=rgb, N=1, guidance=False, optim_steps=1, cost=None)
            R_inv, t_inv = se3.sample(feats=feats_inv, rgb=rgb_inv, N=1, guidance=False, optim_steps=1, cost=None)
            R_fwd, t_fwd = R_fwd[0], t_fwd[0]
            R_inv, t_inv = R_inv[0], t_inv[0]

            # Plot predictions with same color as their GT
            ax.scatter(*axis_from_R(R_fwd), color=color_fwd, marker='x', s=30)
            ax.scatter(*axis_from_R(R_inv), color=color_inv, marker='x', s=30)

            # Rotation equivariance check
            R21_expected = R_fwd.T
            t21_expected = -t_fwd @ R_fwd.T

            R_err = torch.norm(R_inv - R21_expected)
            t_err = torch.norm(t_inv - t21_expected)

            print(f"[Sample {j:02d} | Iter {n:02d}]  ‖ΔR‖ = {R_err:.4f}   ‖Δt‖ = {t_err:.4f}")

    # Draw unit sphere
    u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)

    ax.legend(loc="upper right", fontsize=8)
    out_path = Path(plot_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path.resolve()}")
    
@torch.no_grad()
def visualize_cameras(se3, dataloader, plot_name, conditioning, k=4, N=10, device="cuda"):
    from pytorch3d.renderer import PerspectiveCameras
    from pytorch3d.vis.plotly_vis import plot_scene
    import plotly.io as pio
    from pathlib import Path
    import numpy as np
    import random, torch

    se3.model.eval()
    k = min(k, len(dataloader))
    chosen_steps = random.sample(range(len(dataloader)), k)

    for j, step_idx in enumerate(chosen_steps):
        batch = dataloader.dataset[step_idx]

        # Load GT poses
        R_gt = torch.stack([batch["R1"], batch["R2"]], dim=0).to(device)
        t_gt = torch.stack([batch["t1"], batch["t2"]], dim=0).to(device)

        # Prepare model inputs
        if conditioning == "depths":
            sample_args = {
                "depths": batch["depths"].unsqueeze(0).to(device),
                "rgb": batch["rgb"].unsqueeze(0).to(device),
            }
        elif conditioning == "features":
            rgb = batch["rgb"]
            feats = batch["feats"]
            if rgb.ndim == 4: rgb = rgb.unsqueeze(0)
            if feats.ndim == 4: feats = feats.unsqueeze(0)
            sample_args = {
                "feats": feats.to(device),
                "rgb": rgb.to(device)
            }
        else:
            raise ValueError(f"Unsupported conditioning: {conditioning}")

        # Store predicted cameras
        all_R_pred, all_t_pred = [], []

        for n in range(N):
            R_pred, t_pred = se3.sample(**sample_args, stop=1, B=1, N=2, guidance=False, optim_steps=1, cost=None)

            # Find best permutation
            perms = [(0, 1), (1, 0)]
            best_perm, min_error = None, float('inf')
            for perm in perms:
                total_err = 0.0
                for i in range(2):
                    cos_theta = 0.5 * (torch.trace(R_gt[i].T @ R_pred[perm[i]]) - 1.0)
                    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                    ang_err = torch.acos(cos_theta) * 180.0 / np.pi
                    trans_err = torch.norm(t_gt[i] - t_pred[perm[i]])
                    total_err += ang_err + trans_err
                if total_err < min_error:
                    min_error = total_err
                    best_perm = perm

            matched_R = torch.stack([R_pred[best_perm[0]], R_pred[best_perm[1]]], dim=0)
            matched_t = torch.stack([t_pred[best_perm[0]], t_pred[best_perm[1]]], dim=0)

            all_R_pred.append(matched_R)
            all_t_pred.append(matched_t)

            # Print errors
            for i in range(2):
                ri = best_perm[i]
                cos_theta = 0.5 * (torch.trace(R_gt[i].T @ R_pred[ri]) - 1.0)
                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                ang_err = torch.acos(cos_theta).item() * 180.0 / np.pi
                trans_err = torch.norm(t_gt[i] - t_pred[ri]).item()
                print(f"[Sample {j:02d}, View {i}, Prediction {n:02d}] Δθ = {ang_err:6.2f}°, ‖Δt‖ = {trans_err:7.4f}")

        # Stack predicted cameras
        pred_R = torch.cat(all_R_pred, dim=0)
        pred_t = torch.cat(all_t_pred, dim=0)

        # Combine GT + Predicted into a single PerspectiveCameras object
        all_R = torch.cat([R_gt, pred_R], dim=0)
        all_T = torch.cat([t_gt, pred_t], dim=0)
        all_cams = PerspectiveCameras(R=all_R, T=all_T, device=device)

        # Assign colors: red for GT, blue for predictions
        red = torch.tensor([[1.0, 0.0, 0.0]], device=device)  # GT
        blue = torch.tensor([[0.0, 0.0, 1.0]], device=device) # Pred
        cam_colors = torch.cat([
            red.repeat(R_gt.shape[0], 1),
            blue.repeat(pred_R.shape[0], 1)
        ], dim=0)

        # One single frame: all cameras + color
        scene = {
            "scene": {
                "cameras": all_cams,
                "cameras_color": cam_colors
            }
        }

        # Plot and save
        fig = plot_scene(scene, layout=dict(width=800, height=800))
        out_path = Path(plot_name) / f"scene_{j:02d}.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, file=str(out_path), auto_open=False)
        print(f"[Saved] Overlayed GT (red) + Predictions (blue) for sample {j} → {out_path.resolve()}")

@torch.no_grad()
def visualize_cameras_sfm(se3, dataloader, plot_name, conditioning, k=4, N=10, device="cuda"):
    import trimesh
    import numpy as np
    import random, torch
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    import pyrender

    def geotrf(matrix, pts):
        pts_hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=-1)
        return (matrix @ pts_hom.T).T[:, :3]

    def add_scene_cam(scene, c2w, edge_color, image=None, focal=None, imsize=None, screen_width=0.03):
        OPENGL = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        if image is not None:
            H, W, THREE = image.shape
        elif imsize is not None:
            W, H = imsize
        elif focal is not None:
            H = W = focal / 1.1
        else:
            H = W = 1

        if focal is None:
            focal = min(H, W) * 1.1
        elif isinstance(focal, np.ndarray):
            focal = focal[0]

        height = focal * screen_width / H
        width = screen_width * np.sqrt(0.5)
        rot45 = np.eye(4)
        rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
        rot45[2, 3] = -height
        aspect_ratio = np.eye(4)
        aspect_ratio[0, 0] = W / H
        transform = c2w @ OPENGL @ aspect_ratio @ rot45
        cam = trimesh.creation.cone(width, height, sections=4)

        rot2 = np.eye(4)
        rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(4)).as_matrix()
        vertices = cam.vertices
        vertices_offset = 0.9 * cam.vertices
        vertices = np.r_[vertices, vertices_offset, geotrf(rot2, cam.vertices)]
        vertices = geotrf(transform, vertices)
        faces = []
        for face in cam.faces:
            if 0 in face: continue
            a, b, c = face
            a2, b2, c2 = face + len(cam.vertices)
            faces += [
                (a, b, b2), (a, a2, c), (c2, b, c),
                (a, b2, a2), (a2, c, c2), (c2, b2, b)
            ]
        faces += [(c, b, a) for a, b, c in faces]
        cam = trimesh.Trimesh(vertices=vertices, faces=faces)
        cam.visual.face_colors[:, :3] = edge_color
        scene.add_geometry(cam)

    se3.model.eval()
    k = min(k, len(dataloader))
    chosen_steps = random.sample(range(len(dataloader)), k)

    for j, step_idx in enumerate(chosen_steps):
        batch = dataloader.dataset[step_idx]
        R_gt = torch.stack([batch["R1"], batch["R2"]], dim=0).to(device)
        t_gt = torch.stack([batch["t1"], batch["t2"]], dim=0).to(device)

        # Get conditioning
        if conditioning == "depths":
            sample_args = {
                "depths": batch["depths"].unsqueeze(0).to(device),
                "rgb": batch["rgb"].unsqueeze(0).to(device),
            }
        elif conditioning == "features":
            rgb = batch["rgb"]
            feats = batch["feats"]
            if rgb.ndim == 4: rgb = rgb.unsqueeze(0)
            if feats.ndim == 4: feats = feats.unsqueeze(0)
            sample_args = {
                "feats": feats.to(device),
                "rgb": rgb.to(device)
            }
        else:
            raise ValueError(f"Unsupported conditioning: {conditioning}")

        all_R_pred, all_t_pred = [], []
        for n in range(N):
            R_pred, t_pred = se3.sample(**sample_args, stop=1, B=1, N=2, guidance=False, optim_steps=1, cost=None)

            # Match permutations
            perms = [(0, 1), (1, 0)]
            best_perm, min_error = None, float('inf')
            for perm in perms:
                total_err = 0.0
                for i in range(2):
                    cos_theta = 0.5 * (torch.trace(R_gt[i].T @ R_pred[perm[i]]) - 1.0)
                    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                    ang_err = torch.acos(cos_theta) * 180.0 / np.pi
                    trans_err = torch.norm(t_gt[i] - t_pred[perm[i]])
                    total_err += ang_err + trans_err
                    print(f"[Sample {j:02d}, View {i}, Prediction {n:02d}] Δθ = {ang_err:6.2f}°, ‖Δt‖ = {trans_err:7.4f}")
                if total_err < min_error:
                    min_error = total_err
                    best_perm = perm

            matched_R = torch.stack([R_pred[best_perm[0]], R_pred[best_perm[1]]], dim=0)
            matched_t = torch.stack([t_pred[best_perm[0]], t_pred[best_perm[1]]], dim=0)
            all_R_pred.append(matched_R)
            all_t_pred.append(matched_t)

        # Stack all predictions
        pred_R = torch.cat(all_R_pred, dim=0)
        pred_t = torch.cat(all_t_pred, dim=0)

        # Create scene
        scene = trimesh.Scene()

        # Add GT cams in red
        # Define 4 distinguishable colors (GT0, GT1, Pred0, Pred1)
        color_map = {
            "GT_0": [255, 0, 0],       # Red
            "GT_1": [17, 191, 17],       # Green
            "Pred_0": [26, 26, 237],     # Blue
            "Pred_1": [242, 164, 19],   # Orange
        }

        # Add GT cameras with distinct colors
        for i in range(2):
            c2w = torch.eye(4)
            c2w[:3, :3] = R_gt[i].cpu()
            c2w[:3, 3] = t_gt[i].cpu()
            cam_type = f"GT_{i}"
            add_scene_cam(scene, c2w.numpy(), edge_color=color_map[cam_type])

        # Add predicted cameras with matching index color (Pred_0 or Pred_1)
        for i in range(pred_R.shape[0]):
            c2w = torch.eye(4)
            c2w[:3, :3] = pred_R[i].cpu()
            c2w[:3, 3] = pred_t[i].cpu()
            cam_type = f"Pred_{i % 2}"  # alternate 0,1,0,1...
            add_scene_cam(scene, c2w.numpy(), edge_color=color_map[cam_type])


        # Save output
        out_path = Path(plot_name) / f"scene_{j:02d}.glb"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scene.export(out_path)
        print(f"[Saved] GT (red) + Predictions (blue) camera scene to → {out_path.resolve()}")

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
        R_rel = R2 @ R1.transpose(-2, -1)
        t_rel = t2 - (R2 @ R1.transpose(-2, -1) @ t1.unsqueeze(-1)).squeeze(-1)
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