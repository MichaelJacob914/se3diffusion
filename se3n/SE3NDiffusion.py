
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
paths = [HERE / "gta", HERE / "gta" / "source"]
sys.path[:0] = [str(p) for p in paths if p.exists() and str(p) not in sys.path]
import layers
from layers import Transformer as GTATransformer
from datasets import RGBDepthDataset, RGBFeatureDataset, Dust3rFeatureExtractor, Co3dDataset, compute_dust3r_feats
from source.utils.gta import make_SO2mats, make_T2mats
from source.utils.common import downsample
from source.utils.wigner_d import rotmat_to_wigner_d_matrices
from SO3n import so3_diffuser, SO3Algebra
from R3n import r3_diffuser
from se3n_models import GTASE3, Transformer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import math
import random
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from mpl_toolkits.mplot3d import Axes3D
import geoopt
from geoopt.optim import (RiemannianAdam)
Stiefel = geoopt.Stiefel()
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from torch.amp import autocast, GradScaler
import time
from losses import loss_relative_pose

class se3n_diffuser:
    def __init__(self, T, 
                 conditioning = "features", 
                 score = False, 
                 batch_size=64, 
                 device="cuda", 
                 is_dist=False, 
                 local_rank=0, 
                 world_size=1,
                 accum_steps=1,
                 dataloader = None, 
                 betas=None, 
                 save_model = True, 
                 save_path = None,
                 dataset_root = None, 
                 prediction = {"r": "noise", "t": "noise"}, 
                 so3_config = None, 
                 r3_config = None, 
                 model_type = "GTA", 
                 generate_noise = "batch", 
                 dataset = None, 
                 model_path = None, 
                 num_images = None, 
                 num_workers = 2, 
                 pose = True, 
                 attn_args = { "method": {"name": "gta", "args": { "so2": False,"max_freq_h": 3,"max_freq_w": 3, 
                                                                  "flash_attn": False,"f_dims": {"se3": 48,"so3": 48}}}}, 
                attn_kwargs = {'f_dims': {'se3': 48,  'so3': 48},
                    'max_freq_h': 3, 'max_freq_w': 3, 'so2': 2,  'so3': 2, 'shared_freqs': True, 'ray_to_se3': False, }  
                ):
        self.device      = torch.device(device)
        self.is_dist     = is_dist
        self.local_rank  = local_rank
        self.world_size  = world_size
        self.accum_steps = max(1, int(accum_steps))
        self.batch_size  = batch_size
        self.conditioning = conditioning
        self.prediction = prediction 
        self.so3_config = so3_config
        self.r3_config = r3_config
        self.so3 = so3_diffuser(prediction = prediction["r"], T = T, batch_size = batch_size, betas = betas, device=device, cfg = so3_config)
        self.r3  = r3_diffuser(prediction = prediction["t"], T = T, batch_size = batch_size, betas = betas, device=device, cfg = r3_config)
        self.score = score
        self.attn_args = attn_args
        self.attn_kwargs = attn_kwargs
        self.save_model   = save_model
        self.dataloader = dataloader
        self.save_path    = save_path
        self.dataset_root = dataset_root
        self.dataset = dataset
        self.num_workers = num_workers
        self.num_images = num_images
        self.pose = pose
        self.model_type = model_type
        self.generate_noise = generate_noise

        if(self.model_type == "GTA"):
            print("Using GTA Model")
            self.model = GTASE3(N = 2, attn_args = attn_args, attn_kwargs = attn_kwargs, device="cuda", num_images = None, batch_size = batch_size).to(self.device)
        else: 
            print("Using Standard Transformer Model")
            self.model = Transformer(N = 2).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(),
                            lr=3e-4,
                            weight_decay=0.0,
                            betas=(0.9, 0.999))
        if model_path is not None:
            print(f"[INFO] Loading model weights from {model_path}")
            checkpoint = torch.load(self.save_path, map_location=device)

            # Fix model state dict if it was saved with DataParallel
            state_dict = checkpoint["model"]
            if any(k.startswith("module.") for k in state_dict.keys()):
                print("[INFO] Detected DataParallel weights. Stripping 'module.' prefix.")
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)
            self.opt.load_state_dict(checkpoint["opt"])
            start_epoch = checkpoint["epoch"]

        if self.is_dist:
            # DDP only
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        elif torch.cuda.device_count() > 1:
            # DP only
            self.model = torch.nn.DataParallel(self.model)

        self.model = self.model.to(self.device)

        self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=50_000, eta_min=1e-6
        )
        self.clip_grad_norm = 1.0

        self.T = T    

    def iteration(self, R_clean, T_clean, depths=None, feats=None, point_clouds=None,rgb=None,gamma=1.0, step=None):
        """
        R_clean : (B, N, 3, 3)
        T_clean : (B, N, 3)
        Feats: (B, N, 1024, 14, 14)
        RGB: (B,N, 3, 224, 224)
        """
        self.model.train()
        B, N = R_clean.shape[:2]

        if self.generate_noise == "pose":      # per-view random 
            ts = torch.randint(1, self.T, (B, N), device=self.device)
        elif self.generate_noise == "batch":   # matches v1
            ts = torch.randint(1, self.T, (B, 1), device=self.device).expand(B, N)
        else:                                  # global
            t  = torch.randint(1, self.T, (1,), device=self.device)
            ts = t.expand(B, N)

        # --- generate noise in [B,N,...] ---
        R_noise, v_noise = self.so3.generate_noise(ts, B, N)   # [B,N,3,3], [B,N,3]
        T_noise          = self.r3.generate_noise(ts, B, N)    # [B,N,3]

        # --- apply noise ---
        R_t = self.so3.add_noise(R_clean, R_noise, ts)         # [B,N,3,3]
        T_t = self.r3.add_noise(T_clean, T_noise, ts)          # [B,N,3]

        # --- targets (split conditions) ---
        if self.prediction["r"] != "score":
            sigma_so3 = torch.sqrt(1 - self.so3.alpha_bars[ts])    # [B,N]
            v_target  = v_noise / sigma_so3[..., None]
        if self.prediction["t"] != "score":
            sigma_r3  = torch.sqrt(1 - self.r3.alpha_bars[ts])     # [B,N]
            t_target  = T_noise / sigma_r3[..., None]

        # --- model call (define t_tensor!) ---
        t_tensor = ts
        if self.conditioning == "features" or self.conditioning == "CO3D":
            pred = self.model(R_t, T_t, feats, rgb, t_tensor)  # ensure feats/rgb are [B,N,...]
        elif self.conditioning == "depths":
            pred = self.model(R_t, T_t, t_tensor, depths, rgb)
        elif self.conditioning == "point_cloud":
            pred = self.model(R_t, T_t, t_tensor, point_clouds, rgb)
        else:
            raise ValueError(...)

        pred_rot, pred_trans = pred[..., :3], pred[..., 3:]

        # --- losses ---
        if self.prediction["r"] == "noise":
            loss = 0.5 * ((pred_rot - v_target) ** 2).mean() + 0.5 * ((pred_trans - t_target) ** 2).mean()
        elif self.prediction["r"] == "score":
            loss = loss_score(self, R_clean, T_clean, pred_rot, pred_trans, R_t, T_t, ts)
        else:
            loss = loss_relative_pose(self, R_clean, T_clean, pred_rot, pred_trans, R_t, ts)  # pass R_t

        return loss
    
    
    def train(self, num_epochs=1, log_every=1000):
        """
        Generic training wrapper.
        Chooses dataloader and training loop based on self.conditioning:
            - "depths"   → RGBDepthDataset → train_depths(...)
            - "features" → RGBFeatureDataset → train_feats(...)
        """
        print(f"[INFO] Starting training with conditioning: {self.conditioning}")

        if self.conditioning == "depths":
            if(self.dataset is not None):
                dataset = self.dataset
            else: 
                dataset = RGBDepthDataset(root=self.dataset_root)
            dataloader = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.num_workers,
                                    pin_memory=True)
            return self.train_depths(dataloader, num_epochs=num_epochs, log_every=log_every)

        elif self.conditioning == "features":
            if(self.dataset is not None):
                dataset = self.dataset
            else: 
                dataset = RGBFeatureDataset(root=self.dataset_root, device=self.device)
            return self.train_feats(self.dataloader, num_epochs=num_epochs, log_every=log_every)
        elif self.conditioning == "CO3D": 
            if(self.dataset is not None):
                    dataset = self.dataset
            else: 
                dataset = Co3dDataset(root=self.dataset_root, device=self.device)
            self.extractor = dataset.extractor
            return self.train_co3d(self.dataloader, num_epochs=num_epochs, log_every=log_every)


        else:
            raise ValueError(f"Unknown conditioning type: {self.conditioning}")


    def train_feats(self, dataloader, num_epochs: int = 1, log_every: int = 1000):
        """
        Dataloader must yield dictionaries with keys
            'img1', 'img2' : (B, 3, H, W)   - float32 in [0,1]
            'rgb'          : (B, 6, H, W)   - stacked pair (optional convenience)
            'feats'        : (B, 2, 1024, 32, 32)
            'R'            : (B, 3, 3)
            't'            : (B, 3)

        The loop:
            • moves everything to self.device (CUDA)
            • calls self.iteration(R, T, feats, rgb, global_step)
            • keeps running loss and prints logs
            • saves the model at the end (same logic as train_depths)
        """
        losses   = []
        global_step = 0
        scaler = GradScaler('cuda')

        for epoch in range(num_epochs):
            start_time = time.time()
            running = 0.0
            n_seen  = 0

            for step, batch in enumerate(dataloader):
                R_1 = batch["R1"].to(self.device)       # (B,3,3)
                T_1 = batch["t1"].to(self.device)       # (B, 3)
                R_2 = batch["R2"].to(self.device)       # (B,3,3)
                T_2 = batch["t2"].to(self.device)       # (B, 3)
                R_clean = torch.stack([R_1, R_2], dim=1)  # (B, 2, 3, 3)
                T_clean = torch.stack([T_1, T_2], dim=1)  # (B, 2, 3)

                rgb = batch["rgb"].to(self.device)        # [B, 6, H, W]
                feats = batch["feats"].to(self.device, non_blocking=True)

                loss = self.iteration(R_clean=R_clean, T_clean=T_clean, feats=feats, rgb=rgb)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.opt.step()
                self.lr_sched.step()
                # -----------------------------------------------------------
                loss_val = loss.item()
                running += loss_val * R_clean.size(0)
                n_seen  += R_clean.size(0)
                global_step += 1

                if step % log_every == 0:
                    print(f"[epoch {epoch}  step {step:04d}]  loss {loss_val:.6f}")

            epoch_loss = running / n_seen
            losses.append(epoch_loss)
            epoch_time = time.time() - start_time
            print(f"===> epoch {epoch} done, mean loss {epoch_loss:.6f}, time {epoch_time:.2f} sec")

        # -------- optional checkpoint ------------------------------------
        if self.save_model and self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "opt":   self.opt.state_dict(),
                    "epoch": epoch,
                },
                self.save_path
            )

        return losses

    
    def train_co3d(self, dataloader, num_epochs: int = 1, log_every: int = 1000):
        losses = []
        scaler = GradScaler('cuda')

        for epoch in range(num_epochs):
            print("New epoch")
            epoch_start = time.time()  # <-- start timer
            # let distributed samplers reshuffle each epoch
            if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

            self.model.train()
            running, n_seen = 0.0, 0
            self.opt.zero_grad(set_to_none=True)

            for step, batch in enumerate(dataloader):
                # ---- DUSt3R feats on device (no grad) ----
                with torch.no_grad():
                    imgs_cpu = batch["imgs"]            # [B,K,3,H,W] on CPU
                feats = compute_dust3r_feats(self.extractor, imgs_cpu, device=self.device, use_amp=True) # returns [B,K,1024,h,w] on self.device

                # move the rest
                R_clean = batch["R"].to(self.device, non_blocking=True)
                T_clean = batch["t"].to(self.device, non_blocking=True)
                rgb     = imgs_cpu[:, :2].to(self.device, non_blocking=True)  # if you need rgb

                # ---- forward (AMP) + grad accumulation ----
                with autocast('cuda'):
                    loss = self.iteration(R_clean=R_clean, T_clean=T_clean, feats=feats, rgb=rgb)
                    loss = loss / self.accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % self.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    scaler.step(self.opt)
                    scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    self.lr_sched.step()

                # logging stats (undo the /accum_steps for reporting)
                running += float(loss) * self.accum_steps * R_clean.size(0)
                n_seen  += R_clean.size(0)

                if (step % log_every == 0) and (not self.is_dist or self.local_rank == 0):
                    print(f"[epoch {epoch} step {step:04d}] loss {float(loss)*self.accum_steps:.6f}")

            epoch_loss = running / max(1, n_seen)
            losses.append(epoch_loss)
            epoch_secs = time.time() - epoch_start
            if not self.is_dist or self.local_rank == 0:
                print(f"===> epoch {epoch} done, mean loss {epoch_loss:.6f} | time {epoch_secs:.2f}s")

        # save checkpoint on rank 0 only (handle DDP .module)
        if getattr(self, "save_model", False) and self.save_path is not None:
            if not self.is_dist or self.local_rank == 0:
                to_save = self.model.module if hasattr(self.model, "module") else self.model
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": to_save.state_dict(),
                            "opt":   self.opt.state_dict(),
                            "epoch": num_epochs-1},
                        self.save_path)

        return losses
    

    @torch.no_grad()
    def sample(self, feats, rgb, B=1, N=1, equivariance = False, solver = "SDE", threshold = 0, guidance=False, optim_steps=1, cost=None):
        """
        Sample B SE(3)^N structures.

        Args:
            conditioning: (B,) or scalar flag
            stop: starting timestep (usually 1)
            B: batch size
            N: number of SE(3) elements per structure
        Returns:
            R_t: [B, N, 3, 3]
            T_t: [B, N, 3]
        """
        # Initialize rotations and translations
        R_t = torch.stack([
            torch.stack([torch.from_numpy(Rot.random().as_matrix()).float()
                        for _ in range(N)])
            for _ in range(B)
        ]).to(self.device)  # [B, N, 3, 3]

        T_t = torch.randn((B, N, 3), device=self.device)  # [B, N, 3]
        print(self.T)
        for t in reversed(range(1, self.T)):
            t_tensor = torch.full((B,N), float(t), device=self.device)

            # Predict noise
            eps = self.model(R_t, T_t, feats.unsqueeze(0), rgb.unsqueeze(0), t_tensor)  # [B, N, 6]
            eps_rot, eps_trans = eps[..., :3], eps[..., 3:]     # [B, N, 3], [B, N, 3]

            # Reverse steps
            if(self.prediction["r"] == "score" and t > threshold): 
                R_t = self.so3._se_sample_score(R_t, t, eps_rot)[0]  # [B, N, 3, 3]
            elif(self.prediction["r"] == "noise"): 
                R_t = self.so3._se_sample_noise(R_t, t, eps_rot, equivariance = equivariance, guidance = guidance, optim_steps = optim_steps, cost = cost)[0]  # [B, N, 3, 3]
            else:
                R_t = self.so3._se_sample_pose(R_t, t, eps_rot, guidance = guidance, optim_steps = optim_steps, cost = cost)[0]  # [B, N, 3, 3]

            if(self.prediction["t"] == "score" and t > threshold): 
                T_t = self.r3._eu_sample_score(T_t, t, eps_trans, solver = solver, guidance = guidance, optim_steps = optim_steps, cost = cost)[0]    # [B, N, 3]
            elif(self.prediction["t"] == "noise"): 
                T_t = self.r3._eu_sample_noise(T_t, t, eps_trans, guidance = guidance, optim_steps = optim_steps, cost = cost)[0]    # [B, N, 3]
            else:
                T_t = self.r3._eu_sample_pose(T_t, t, eps_trans, guidance = guidance, optim_steps = optim_steps, cost = cost)[0]    # [B, N, 3]
            
            
        return R_t.squeeze(0), T_t.squeeze(0)  # [N, 3, 3], [N, 3]





