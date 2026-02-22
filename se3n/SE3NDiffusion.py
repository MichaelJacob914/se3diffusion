
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
paths = [HERE / "gta", HERE / "gta" / "source"]
sys.path[:0] = [str(p) for p in paths if p.exists() and str(p) not in sys.path]
import layers
from copy import deepcopy


from layers import Transformer as GTATransformer
from se3n_datasets import PoseCo3DDataset, normalize_cameras, VisCo3DDataset, normalize_c2w_by_camera_centers
from feature_extractors import compute_dust3r_feats, compute_pi3_feats, Dust3rFeatureExtractor, Pi3FeatureExtractor
from SO3n import so3_diffuser, SO3Algebra
from R3n import r3_diffuser
from se3n_models import StandardTransformer, PRoPEUpdatePoseTransformer, OldPRoPEUpdatePoseTransformer
from torch.nn.parallel import DistributedDataParallel as DDP
from se3n_utils import rot_trans_from_se3, build_intrinsics, se3_from_rot_trans, se3_compress, se3_expand
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import math
import random
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from torch.amp import autocast, GradScaler
import time
from losses import diffusion_loss_relative_pose, loss_relative_pose
import torchlie as lie
from accelerate import Accelerator

import geoopt
from geoopt.optim import (RiemannianAdam)
Stiefel = geoopt.Stiefel()

from itertools import combinations

# UFM
from uniflowmatch.models.ufm import (
    UniFlowMatchConfidence,
    UniFlowMatchClassificationRefinement,
)

from feature_extractors import UFMGGSOptimizer, GGSOptimizer

from EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, reparameter
sys.path.insert(0, "/vast/home/m/mgjacob/PARCC/scripts/theseus")
import theseus as th
from theseus.geometry.point_types import Point3 
from theseus.geometry.se3 import SE3
import time 

class se3n_diffuser:
    def __init__(self, config: dict):
        get  = config.get
        getn = lambda k, d: d if get(k) is None else get(k)

        self.T            = get("T") #Number of timesteps model is trained. not needed for cts diffusion but kept for consistency
        self.conditioning = getn("conditioning", "Co3d")
        self.feature_type = getn("feature_type", "dust3r")
        self.score        = getn("score", False)
        self.device       = torch.device(getn("device", "cuda"))
        self.dataloader   = get("dataloader")
        self.save_model   = getn("save_model", True)
        self.save_path    = get("save_path")
        self.num_images   = get("num_images")
        self.prediction   = getn("prediction", {"r": "noise", "t": "noise"})
        self.so3_config   = get("so3_config")
        self.r3_config    = get("r3_config")
        self.model_type   = getn("model_type", "GTA")
        self.generate_noise = getn("generate_noise", "batch")
        self.pose         = getn("pose", True)
        self.forward_process = getn("forward_process", "ve")
        self.representation = getn("representation", "rot6d")
        self.extractor = getn("extractor", None)
        self.scheme = getn("scheme", "GTA")
        self.accelerator = config.get("accelerator", None)
        self.update_type = config.get("update_type", "mult")
        self.normalization_type = config.get("normalization_type", "center_cameras")
        self.guidance_type = config.get("guidance_type")

        if(self.guidance_type == "ufm"):
            matcher = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")
            matcher = matcher.to(device).eval()
            self.guidance_module = UFMGGSOptimizer(matcher)
            #self.guidance_module = TheseusUFMGGSOptimizer(matcher)
        elif(self.guidance_type == "tufm"):
            matcher = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")
            matcher = matcher.to(device).eval()
            self.guidance_module = TheseusUFMGGSOptimizer(matcher)
        elif(self.guidance_type == "ggs"):
            # Initialize the matcher with default settings
            _default_cfg = deepcopy(full_default_cfg)
            matcher = LoFTR(config=_default_cfg)

            # Load pretrained weights
            ckpt = torch.load(
                "/vast/projects/kostas/geometric-learning/mgjacob/EfficientLoFTR/ELoFTR_checkpoint.ckpt",
                map_location="cpu",
                weights_only=False,  
            )
            state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            matcher.load_state_dict(state, strict=False)
            matcher = reparameter(matcher)  
            matcher = matcher.eval().cuda()
            self.guidance_module = GGSOptimizer(matcher)

        self.attn_args    = getn("attn_args", { "method": 
                            {"name": "gta", "args": 
                             {"so3": 16, "max_freq_h": 1, "max_freq_w": 1, "so2": 0,
                            "flash_attn": True,"f_dims": {"se3": 32,"so3": 32 , 'triv': 64, "so2": 0}, 
                            "q_emb_dim": 128}}})
        self.attn_kwargs  = getn("attn_kwargs", {'f_dims': {'se3': 32,  'so3': 32, 'triv': 64, 'so2': 0},
                            'max_freq_h': 1, 'max_freq_w': 1, 'so3': 2, 'so2': 0,
                            'shared_freqs': True, 'ray_to_se3': False})

        self.so3 = so3_diffuser(
            prediction=self.prediction["r"], forward_process = self.forward_process, 
            T=self.T, device=self.device, representation = self.representation, cfg=self.so3_config
        )
        self.r3  = r3_diffuser(
            prediction=self.prediction["t"],
            T=self.T, device=self.device, cfg=self.r3_config, recenter = True
        )

        if self.model_type == "pose":
            print("Using Pose Model")
            self.model = OldPRoPEUpdatePoseTransformer(attn_args=self.attn_args, attn_kwargs=self.attn_kwargs,
                device=self.device, representation = self.representation, scheme = self.scheme, feature_type = self.feature_type, update_type = self.update_type
            ).to(self.device)

        elif self.model_type == "patch": 
            print("Using Patch Model")
            self.model = GtaPatchTransformer(attn_args=self.attn_args, attn_kwargs=self.attn_kwargs,
                device=self.device, representation = self.representation, scheme = self.scheme,
            ).to(self.device)
        elif self.model_type == "GTA": 
            print("Using Standard GTA Model")
            self.model = GtaTransformer(attn_args=self.attn_args, attn_kwargs=self.attn_kwargs,
                device=self.device, representation = self.representation, scheme = self.scheme,
            ).to(self.device)
        else:
            print("Using Standard Transformer Model")
            self.model = StandardTransformer(representation = self.representation).to(self.device)

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        model_path = get("model_path")
        if model_path is not None:
            print(f"[INFO] Loading model weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=device) 

            state_dict = checkpoint["model"]
            self.model.load_state_dict(state_dict)
            if "opt" in checkpoint:
                self.opt.load_state_dict(checkpoint["opt"])

        self.model = self.model.to(self.device)

        
        self.clip_grad_norm = 1.0
        if(self.accelerator is not None):
            self.model, self.opt = self.accelerator.prepare(
                self.model, self.opt, 
            )


    def iteration(self, R_clean, T_clean, feats=None, rgb=None, Ks = None):
        """
        R_clean : (B, N, 3, 3)
        T_clean : (B, N, 3)
        Feats: (B, N, 1024, 14, 14)
        RGB: (B,N, 3, 224, 224)
        """
        self.model.train()
        B, N = R_clean.shape[:2]

        #GENERATE RANDOM NOISE FROM .01 to .99 
        t_min = 1.0 / self.T
        t_max = (self.T - 1.0) / self.T

        if self.generate_noise == "pose":
            # independent t per pose
            time_tensor = t_min + (t_max - t_min) * torch.rand((B, N), device=self.device)
        elif self.generate_noise == "batch":
            # one t per batch item
            t_batch = t_min + (t_max - t_min) * torch.rand((B, 1), device=self.device)
            time_tensor = t_batch.expand(B, N)
        else:
            # one global t shared across batch
            t_global = t_min + (t_max - t_min) * torch.rand((1,), device=self.device)
            time_tensor = t_global.expand(B, N)
        # --- generate noise in [B,N,...] ---
        R_noise, v_noise = self.so3.generate_noise(time_tensor, B, N)   # [B,N,3,3], [B,N,3]
        T_noise          = self.r3.generate_noise(time_tensor, B, N)    # [B,N,3]

        # --- apply noise ---
        R_t = self.so3.add_noise(R_clean, R_noise, time_tensor)         # [B,N,3,3]
        T_t = self.r3.add_noise(T_clean, T_noise, time_tensor)          # [B,N,3]

        pred = self.model(R_t, T_t, feats, rgb, Ks, time_tensor) 
        
        #APPEND THE TIMESTEP FOR LOSSES
        pred["timestep"] = time_tensor

        return pred 

    def train_co3d(self, dataloader, num_epochs: int = 1, log_every: int = 5, wandb_run=None):
        T_max = num_epochs * len(dataloader)
        self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=T_max,    
            eta_min=5e-8
        )
        device = self.device
        self.model.train()
        losses = []

        if self.extractor is None:
            print("We are not using features!")

        global_step = 0  # for wandb step indexing
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

            running, n_seen = 0.0, 0
            running_rot, running_trans = 0.0, 0.0
            running_trans_dir = 0.0
            running_rot_geodesic = 0.0
            
            for step, batch in enumerate(dataloader):
                # ----- inputs -----
                global_step += 1
                imgs = batch["imgs"].to(device, non_blocking=True)    
                gt_pose = batch["camera_pose"].to(device, non_blocking=True) # [V,4,4] c2w
                K_clean = batch["K"].to(device, non_blocking=True)           # [B,2,3,3]
                           
                B, V = imgs.shape[:2]

                if(self.normalization_type == "center_cameras"):
                    c2w, stats = normalize_c2w_by_camera_centers(c2w = gt_pose, scale_mode = "mean")
                else: 
                    depths = batch["depths"].to(device, non_blocking=True)   # [B,V,H,W]
                    c2w, K_clean, depths, _, stats = normalize_cameras(
                        c2w=gt_pose,        # (1,V,4,4)
                        K=K_clean,          # (1,V,3,3)
                        depths=depths, # (1,V,H,W)
                        stride=4,                 
                        return_world_points=False,
                    )
                

                R_clean = c2w[..., :3, :3]    # (B,V,3,3)
                T_clean = c2w[..., :3, 3]     # (B,V,3)   
   
                with torch.autocast(device_type="cuda", enabled=False):
                    if(self.feature_type == "pi3"):
                        feats = compute_pi3_feats(self.extractor, imgs)
                    elif(self.feature_type =="dust3r"):   
                        feats = compute_dust3r_feats(self.extractor, imgs)   

                    prediction = self.iteration(R_clean=R_clean, T_clean=T_clean,
                                        feats=feats, rgb=imgs, Ks=K_clean)


                R_pred, T_pred = prediction["R"], prediction["t"]  # (B,N,3,3), (B,N,3)
                time_tensor = prediction["timestep"]
                loss_rot, loss_rot_geodesic, loss_trans, loss_trans_dir = diffusion_loss_relative_pose(self, R_clean, T_clean, R_pred, T_pred, time_tensor)

                if(self.scheme == "GTA"):
                    fx_gt = K_clean[..., 0, 0]    # (B, N)
                    fy_gt = K_clean[..., 1, 1]    # (B, N)
                    # predicted focals from dict
                    fx_pred = prediction["fx"]    # (B, N)
                    fy_pred = prediction["fy"]    # (B, N)
                    loss_focal = 0.5 * (
                        torch.nn.functional.l1_loss(fx_pred, fx_gt) +
                        torch.nn.functional.l1_loss(fy_pred, fy_gt)
                    )
                    loss = loss_rot + loss_trans + loss_rot_geodesic + loss_trans_dir + loss_focal
                else: 
                    loss = loss_rot + loss_trans + loss_rot_geodesic + loss_trans_dir
                
               
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                    self.opt.step()
                    self.lr_sched.step()
                    self.opt.zero_grad(set_to_none=True)

                # ----- logging -----
                
                bs = R_clean.size(0)
                loss_det      = loss.detach()
                loss_rot_det  = loss_rot.detach()
                loss_trans_det= loss_trans.detach()
                loss_trans_dir = loss_trans_dir.detach()
                loss_rot_geodesic_det = loss_rot_geodesic.detach()
                
                running += loss_det * bs
                running_rot   += loss_rot_det * bs
                running_trans += loss_trans_det * bs
                running_trans_dir += loss_trans_dir * bs

                running_rot_geodesic += loss_rot_geodesic_det * bs
                n_seen  += R_clean.size(0)

            epoch_loss = (running / max(1, n_seen))
            epoch_rot  = (running_rot / max(1, n_seen))
            epoch_trans= (running_trans / max(1, n_seen))
            epoch_trans_dir = (running_trans_dir) / max(1, n_seen)
            epoch_rot_geodesic= (running_rot_geodesic / max(1, n_seen))

            if (epoch % log_every == 0) and wandb_run is not None and self.accelerator.is_main_process:
                wandb_run.log(
                    {
                        "epoch/loss": epoch_loss.item(),
                        "epoch/loss_rot": epoch_rot.item(),
                        "epoch/loss_trans": epoch_trans.item(),
                        "epoch/loss_trans_dir": epoch_trans_dir.item(),
                        "epoch/loss_rot_geodesic": epoch_rot_geodesic.item(),
                        "epoch/time": time.time() - epoch_start,
                        "epoch/index": epoch,
                    },
                    step=global_step,
                )
            
            if(epoch % log_every == 0): 
                losses.append(epoch_loss.cpu())
                print(
                    f"===> epoch {epoch:>3} | mean {epoch_loss:.6f} "
                    f"| rot {epoch_rot:.6f} "
                    f"| trans {epoch_trans:.6f} "
                    f"| trans_dir {epoch_trans_dir:.6f} "
                    f"| rot_geodesic {epoch_rot_geodesic:.6f} "
                    f"| time {time.time()-epoch_start:.2f}s",
                    flush=True
                )

        if getattr(self, "save_model", False) and getattr(self, "save_path", None) is not None:
            to_save = self.model.module if hasattr(self.model, "module") else self.model
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": to_save.state_dict(),
                "opt":   self.opt.state_dict(),
            }, self.save_path)

        return losses

    def val_co3d(self, dataloader, num_epochs: int = 1, log_every: int = 500):
        device = self.device
        self.model.eval()
        losses = []
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

            running, n_seen = 0.0, 0
            running_rot, running_trans = 0.0, 0.0
            running_trans_dir = 0.0
            running_rot_geodesic = 0.0
            global_step = 0
            
            with torch.inference_mode():
                for step, batch in enumerate(dataloader):
                    global_step += 1
                    imgs = batch["imgs"].to(device, non_blocking=True)    
                    gt_pose = batch["camera_pose"].to(device, non_blocking=True) # [V,4,4] c2w
                    K_clean = batch["K"].to(device, non_blocking=True)           # [B,2,3,3]
                            
                    B, V = imgs.shape[:2]
                

                    if(self.normalization_type == "center_cameras"):
                        c2w, stats = normalize_c2w_by_camera_centers(gt_pose)
                    else: 
                        depths = batch["depths"].to(device, non_blocking=True)   # [B,V,H,W]
                        c2w, K_clean, depths, _, stats = normalize_cameras(
                            c2w=gt_pose,        # (1,V,4,4)
                            K=K_clean,          # (1,V,3,3)
                            depths=depths, # (1,V,H,W)
                            stride=4,                 
                            return_world_points=False,
                        )

                    R_clean = c2w[..., :3, :3]    # (B,V,3,3)
                    T_clean = c2w[..., :3, 3]     # (B,V,3)T                

                    if(self.feature_type == "pi3"):
                        feats = compute_pi3_feats(self.extractor, imgs)
                    else:   
                        feats = compute_dust3r_feats(self.extractor, imgs)   
                    
                                        
                    prediction = self.iteration(R_clean=R_clean, T_clean=T_clean,
                                        feats=feats, rgb=imgs, Ks=K_clean)

                    fx_gt = K_clean[..., 0, 0]    # (B, N)
                    fy_gt = K_clean[..., 1, 1]    # (B, N)

                    # predicted focals from dict
                    #fx_pred = prediction["fx"]    # (B, N)
                    #fy_pred = prediction["fy"]    # (B, N)

                    # --- losses ---
                    if self.prediction["r"] == "noise":
                        eps_se3 = prediction["eps_se3"]              # (B, N, 6)
                        pred_rot, pred_trans = eps_se3[..., :3], eps_se3[..., 3:]
                        loss_pose = 0.5 * ((pred_rot - v_target) ** 2).mean() + \
                                    0.5 * ((pred_trans - t_target) ** 2).mean()
                    else:
                        R_pred, T_pred = prediction["R"], prediction["t"]  # (B,N,3,3), (B,N,3)
                        time_tensor = prediction["timestep"]
                        loss_rot, loss_rot_geodesic, loss_trans, loss_trans_dir = diffusion_loss_relative_pose(self, R_clean, T_clean, R_pred, T_pred, time_tensor)


                    if(self.scheme == "GTA"):
                        # predicted focals from dict
                        fx_pred = prediction["fx"]    # (B, N)
                        fy_pred = prediction["fy"]    # (B, N)
                        loss_focal = 0.5 * (
                            torch.nn.functional.l1_loss(fx_pred, fx_gt) +
                            torch.nn.functional.l1_loss(fy_pred, fy_gt)
                        )
                        loss = loss_rot + loss_trans + loss_focal
                    else: 
                        loss = loss_rot + loss_trans

                    # ----- logging -----
                    bs = R_clean.size(0)
                loss_det      = loss.detach()
                loss_rot_det  = loss_rot.detach()
                loss_trans_det= loss_trans.detach()
                loss_trans_dir = loss_trans_dir.detach()
                loss_rot_geodesic_det = loss_rot_geodesic.detach()
                
                running += loss_det * bs
                running_rot   += loss_rot_det * bs
                running_trans += loss_trans_det * bs
                running_trans_dir += loss_trans_dir * bs

                running_rot_geodesic += loss_rot_geodesic_det * bs
                n_seen  += R_clean.size(0)

            epoch_loss = (running / max(1, n_seen))
            epoch_rot  = (running_rot / max(1, n_seen))
            epoch_trans= (running_trans / max(1, n_seen))
            epoch_trans_dir = (running_trans_dir) / max(1, n_seen)
            epoch_rot_geodesic= (running_rot_geodesic / max(1, n_seen))

            losses.append(epoch_loss.cpu())
            print(
                f"===> epoch {epoch:>3} | mean {epoch_loss:.6f} "
                f"| rot {epoch_rot:.6f} "
                f"| trans {epoch_trans:.6f} "
                f"| trans_dir {epoch_trans_dir:.6f} "
                f"| rot_geodesic {epoch_rot_geodesic:.6f} "
                f"| time {time.time()-epoch_start:.2f}s",
                flush=True
            )

        return losses

    @torch.no_grad()
    def sample(self,imgs = None, Ks = None, solver = "SDE", threshold = 0, noise_scale = 0.0, guidance=False, optim_steps=150, cost=None, return_intermediates = False, objective = None):
        """
        Sample B SE(3)^N structures.

        Returns (dict):
            {
            "R":  [B, N, 3, 3]  (final rotations),
            "t":  [B, N, 3],    (final translations),
            "fx": Optional[Any],   # last-step aux if model returns
            "fy": Optional[Any],
            "timesteps": List[int],
            "traj_R": Optional[List[Tensor]],  # each [B,N,3,3] if return_traj
            "traj_t": Optional[List[Tensor]],  # each [B,N,3]
            }
        """
        self.model.eval()
        device = self.device

        B,N = imgs.shape[:2]
        t_init = torch.full((B, N), float(self.T - 1), device=device)
        R_t, v_t = self.so3.generate_noise(t_init, B, N)   # [B,N,3,3], [B,N,3]
        T_t          = self.r3.generate_noise(t_init, B, N)    # [B,N,3]

        if(self.feature_type == "pi3"):
            feats = compute_pi3_feats(self.extractor, imgs)
        elif(self.feature_type =="dust3r"):   
            feats = compute_dust3r_feats(self.extractor, imgs)   
        traj_R, traj_t = ([], []) if return_intermediates else (None, None)
        fx = fy = None
        
        """
        if(objective is not None): 
            gt_poses = objective["gt_poses"].to(device)  # [1,N,3,4]
            gt_poses, stats = normalize_c2w_by_camera_centers(c2w = gt_poses, scale_mode = "median")
            R_gt, T_gt = rot_trans_from_se3(gt_poses)
            R_t, T_t = R_gt, T_gt
        """
        
    
        for t in reversed(range(0, self.T)):
            t = (t-1)/(self.T)
            t_tensor = torch.full((B, N), float(t), device=device)
            
            # model prediction
            prediction = self.model(R_t, T_t, feats, imgs, Ks, t_tensor)
            
            if self.forward_process == "ve":
                R_pred = prediction["R"]        #[B,N,3,3]
                T_pred = prediction["t"]        # [B,N,3]

                
                camera_poses = se3_from_rot_trans(R_pred, T_pred)   # [B,N,4,4]

                if(t < 20): 
                    with torch.enable_grad():
                        if(guidance): 
                            camera_poses = self.guidance_module(imgs, Ks, poses = camera_poses, optim_steps = 250)["camera_poses"].unsqueeze(0)

                    R_pred, T_pred = rot_trans_from_se3(camera_poses)  # [B,N,3,3], [B,N,3]
                
                if(t > 0): 
                    R_t, R_0 = self.so3._se_sample_ve(R_t, t, R_pred, threshold=threshold, noise_scale = 0.0)
                    T_t, T_0 = self.r3._eu_sample_ve(
                        T_t, t, T_pred, solver=solver, threshold=threshold, noise_scale = noise_scale)
                
            else:
                # VP case expects epsilons; 
                eps = prediction["eps_se3"]     # [B,N,6]
                eps_rot, eps_trans = eps[..., :3], eps[..., 3:]
                R_t, _ = self.so3._se_sample_vp(R_t, t, eps_rot)
                T_t, _ = self.r3._eu_sample_vp(
                    T_t, t, eps_trans, threshold=threshold,
                    guidance=guidance, optim_steps=optim_steps, cost=cost
                )
            
            if return_intermediates:
                traj_R.append(R_t.clone())
                traj_t.append(T_t.clone())

            """
            if(objective is not None): 
                R_t = R_gt
                T_t = T_gt
            """
                #R_t[:, :3] = R_gt[:, :3]
                #T_t[:, :3] = T_gt[:, :3]
            
        camera_poses = se3_from_rot_trans(R_pred, T_pred)   # [B,N,4,4]
        with torch.enable_grad():
            if(guidance): 
                camera_poses = self.guidance_module(imgs, Ks, poses = camera_poses, optim_steps = 5000)["camera_poses"].unsqueeze(0)
                self.guidance_module.clear_cached_matches()
        
        out = {
            "camera_poses": camera_poses,
            "R": R_pred,            
            "t": T_pred,
            "fx": fx,
            "fy": fy
        }

        if return_intermediates:
            traj_R = torch.stack(traj_R, dim=0)   # [T, B, N, 3, 3]
            traj_t = torch.stack(traj_t, dim=0)   # [T, B, N, 3]
            out["R_layers"] = traj_R
            out["t_layers"] = traj_t
        

        return out

    @torch.no_grad()
    def eval_loss_by_timestep(self, dataloader, t_values, max_batches=50):
        device = self.device
        self.model.eval()
        out = {int(t): {"rot": 0.0, "rot_geo": 0.0, "trans": 0.0, "dir": 0.0, "n": 0} for t in t_values}

        for bi, batch in enumerate(dataloader):
            if bi >= max_batches: break

            imgs = batch["imgs"].to(device, non_blocking=True)
            gt_pose = batch["camera_pose"].to(device, non_blocking=True)
            K_clean = batch["K"].to(device, non_blocking=True)

            if self.normalization_type == "center_cameras":
                c2w, _ = normalize_c2w_by_camera_centers(gt_pose)
            else:
                depths = batch["depths"].to(device, non_blocking=True)
                c2w, K_clean, depths, _, _ = normalize_cameras(c2w=gt_pose, K=K_clean, depths=depths, stride=4, return_world_points=False)

            R_clean = c2w[..., :3, :3]
            T_clean = c2w[..., :3, 3]

            # features
            feats = compute_dust3r_feats(self.extractor, imgs)  # or pi3

            B, N = R_clean.shape[:2]

            for t0 in t_values:
                t_tensor = torch.full((B, N), int(t0), device=device)

                # generate noise exactly like training
                R_noise, v_noise = self.so3.generate_noise(t_tensor, B, N)
                T_noise          = self.r3.generate_noise(t_tensor, B, N)

                R_t = self.so3.add_noise(R_clean, R_noise, t_tensor)
                T_t = self.r3.add_noise(T_clean, T_noise, t_tensor)

                # forward model on (R_t, T_t, t)
                pred = self.model(R_t, T_t, feats, imgs, K_clean, t_tensor)
                R_pred, T_pred = pred["R"], pred["t"]

                # compute weighted loss at that t
                lR, lRgeo, lT, lDir = diffusion_loss_relative_pose(self, R_clean, T_clean, R_pred, T_pred, t_tensor)
                out[int(t0)]["rot"]     += float(lR) * B
                out[int(t0)]["rot_geo"] += float(lRgeo) * B
                out[int(t0)]["trans"]   += float(lT) * B
                out[int(t0)]["dir"]     += float(lDir) * B
                out[int(t0)]["n"]       += B

        # normalize
        for t0 in t_values:
            n = max(1, out[t0]["n"])
            for k in ["rot", "rot_geo", "trans", "dir"]:
                out[t0][k] /= n
        return out


def so3_geodesic_deg(R1: torch.Tensor, R2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Geodesic angle (degrees) between R1 and R2 on SO(3).
    R1, R2: [..., 3, 3]
    returns: [...] in degrees
    """
    R = R1.transpose(-1, -2) @ R2  # relative rotation
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = (tr - 1.0) * 0.5
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos) * (180.0 / torch.pi)