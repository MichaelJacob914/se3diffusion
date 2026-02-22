
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
paths = [HERE / "gta", HERE / "gta" / "source"]
sys.path[:0] = [str(p) for p in paths if p.exists() and str(p) not in sys.path]
import layers
from layers import Transformer as GTATransformer
from se3n_datasets import PoseCo3DDataset
from feature_extractors import compute_dust3r_feats, compute_pi3_feats, Dust3rFeatureExtractor, Pi3FeatureExtractor
from source.utils.gta import make_SO2mats, make_T2mats
from source.utils.common import downsample
from source.utils.wigner_d import rotmat_to_wigner_d_matrices
from SO3n import so3_diffuser, SO3Algebra
from R3n import r3_diffuser
from se3n_models import StandardTransformer, PRoPEUpdatePoseTransformer, RegressionNetwork
from torch.nn.parallel import DistributedDataParallel as DDP
from se3n_utils import rot_trans_from_se3, build_intrinsics
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
from losses import regression_loss_relative_pose
from accelerate import Accelerator

class se3n_regressor:
    def __init__(self, config: dict):
        get  = config.get
        getn = lambda k, d: d if get(k) is None else get(k)

        self.T            = get("T")
        self.conditioning = getn("conditioning", "Co3d")
        self.feature_type = getn("feature_type", "dust3r")
        self.score        = getn("score", False)
        device            = getn("device", "cuda")
        self.device       = torch.device(device)
        self.is_dist      = getn("is_dist", False)
        self.local_rank   = getn("local_rank", 0)
        self.world_size   = getn("world_size", 1)
        self.dataloader   = get("dataloader")
        betas             = get("betas")
        self.save_model   = getn("save_model", True)
        self.save_path    = get("save_path")
        self.dataset_root = get("dataset_root")
        self.dataset      = get("dataset")
        self.num_workers  = getn("num_workers", 2)
        self.num_images   = get("num_images")
        self.prediction   = getn("prediction", {"r": "noise", "t": "noise"})
        self.so3_config   = get("so3_config")
        self.r3_config    = get("r3_config")
        self.model_type   = getn("model_type", "GTA")
        self.generate_noise = getn("generate_noise", "batch")
        self.pose         = getn("pose", True)
        self.forward_process = getn("forward_process", "ve")
        self.representation = getn("representation", "exp")
        self.extractor = getn("extractor", None)
        self.scheme = getn("scheme", "GTA")
        self.accelerator = config.get("accelerator", None)
        self.update_type = config.get("update_type", "abs")

        self.attn_args    = getn("attn_args", { "method": 
                            {"name": "gta", "args": 
                             {"so3": 16, "max_freq_h": 1, "max_freq_w": 1, "so2": 0,
                            "flash_attn": True,"f_dims": {"se3": 32,"so3": 32 , 'triv': 64, "so2": 0}, 
                            "q_emb_dim": 128}}})
        self.attn_kwargs  = getn("attn_kwargs", {'f_dims': {'se3': 32,  'so3': 32, 'triv': 64, 'so2': 0},
                            'max_freq_h': 1, 'max_freq_w': 1, 'so3': 2, 'so2': 0,
                            'shared_freqs': True, 'ray_to_se3': False})

        if self.model_type == "pose":
            print("Using Pose Model")
            self.model = RegressionNetwork(attn_args=self.attn_args, attn_kwargs=self.attn_kwargs,
                device=device, representation = self.representation, scheme = self.scheme, feature_type = self.feature_type, update_type = self.update_type
            ).to(self.device)
        
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
        self.model, self.opt, = self.accelerator.prepare(
                self.model, self.opt, 
            )


    def iteration(self, R_clean, T_clean, depths=None, feats=None, point_clouds=None,rgb=None,gamma=1.0, step=None, Ks = None):
        """
        R_clean : (B, N, 3, 3)
        T_clean : (B, N, 3)
        Feats: (B, N, 1024, 14, 14)
        RGB: (B,N, 3, 224, 224)
        """
        self.model.train()
        B, N = R_clean.shape[:2]

        if self.conditioning == "features" or self.conditioning == "CO3D":
            pred = self.model(feats, rgb, Ks) 
        return pred 
    
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
            return self.train_co3d_ablate(self.dataloader, steps=total_steps, log_every=log_every)

        else:
            raise ValueError(f"Unknown conditioning type: {self.conditioning}")

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
            running_rot_geodesic = 0.0
            
            for step, batch in enumerate(dataloader):
                # ----- inputs -----
                global_step += 1
                imgs = batch["imgs"].to(device, non_blocking=True)                               
                B, K = imgs.shape[:2]

    
                R_clean = batch["R"].to(device, non_blocking=True)           
                T_clean = batch["T"].to(device, non_blocking=True)           # [B,2,3]
                f_clean = batch["fl"].to(device, non_blocking = True)
                pp_clean = batch["pp"].to(device, non_blocking = True)
                K_clean = build_intrinsics(f_clean, pp_clean)
                rgb     = imgs
                
                with self.accelerator.autocast():   
                    if(self.feature_type == "pi3"):
                        feats = compute_pi3_feats(self.extractor, imgs)
                    elif(self.feature_type =="dust3r"):   
                        feats = compute_dust3r_feats(self.extractor, imgs)   
                        
                    prediction = self.iteration(R_clean=R_clean, T_clean=T_clean,
                                        feats=feats, rgb=imgs, Ks=K_clean)


                R_pred, T_pred = prediction["R"], prediction["t"]  # (B,N,3,3), (B,N,3)
                loss_rot, loss_rot_geodesic, loss_trans, loss_trans_dir = regression_loss_relative_pose(self, R_clean, T_clean, R_pred, T_pred)
                loss = loss_rot + loss_rot_geodesic + loss_trans  + loss_trans_dir
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self.lr_sched.step()
                
                # ----- logging -----
                
                bs = R_clean.size(0)
                loss_det      = loss.detach()
                loss_rot_det  = loss_rot.detach()
                loss_trans_det= loss_trans.detach()
                loss_rot_geodesic_det = loss_rot_geodesic.detach()
                
                running += loss_det * bs
                running_rot   += loss_rot_det * bs
                running_trans += loss_trans_det * bs
                running_rot_geodesic += loss_rot_geodesic_det * bs
                n_seen  += R_clean.size(0)

            epoch_loss = (running / max(1, n_seen))
            epoch_rot  = (running_rot / max(1, n_seen))
            epoch_trans= (running_trans / max(1, n_seen))
            epoch_rot_geodesic= (running_rot_geodesic / max(1, n_seen))

            if (epoch % log_every == 0) and wandb_run is not None and self.accelerator.is_main_process:
                wandb_run.log(
                    {
                        "epoch/loss": epoch_loss.item(),
                        "epoch/loss_rot": epoch_rot.item(),
                        "epoch/loss_trans": epoch_trans.item(),
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
                    f"| rot_geodesic {epoch_rot_geodesic:.6f} "
                    f"| time {time.time()-epoch_start:.2f}s",
                    flush=True
                )


        if getattr(self, "save_model", False) and getattr(self, "save_path", None) is not None:
            if not getattr(self, "is_dist", False) or getattr(self, "local_rank", 0) == 0:
                to_save = self.model.module if hasattr(self.model, "module") else self.model
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": to_save.state_dict(),
                    "opt":   self.opt.state_dict(),
                }, self.save_path)

            if wandb_run is not None and self.accelerator.is_main_process:
                wandb_run.log_model(
                    str(self.save_path),
                    name="co3d_model",
                    aliases=[f"final_epoch-{num_epochs}"],
                )

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
            running_rot_geodesic = 0.0
            
            with torch.inference_mode():
                for step, batch in enumerate(dataloader):
                    
                    imgs = batch["imgs"].to(device, non_blocking=True)                               
                    B, K = imgs.shape[:2]

                    if(self.feature_type == "pi3"):
                        feats = compute_pi3_feats(self.extractor, imgs)
                    else:   
                        feats = compute_dust3r_feats(self.extractor, imgs)   
                    R_clean = batch["R"].to(device, non_blocking=True)           # [B,N=2,3,3] or [B,2,3,3]
                    T_clean = batch["T"].to(device, non_blocking=True)           # [B,2,3]
                    f_clean = batch["fl"].to(device, non_blocking = True)
                    pp_clean = batch["pp"].to(device, non_blocking = True)
                    K_clean = build_intrinsics(f_clean, pp_clean)
                    rgb     = imgs
                    
                                        
                    prediction = self.iteration(R_clean=R_clean, T_clean=T_clean, feats=feats, rgb=imgs, Ks=K_clean)

                    fx_gt = K_clean[..., 0, 0]    # (B, N)
                    fy_gt = K_clean[..., 1, 1]    # (B, N)

                    R_pred, T_pred = prediction["R"], prediction["t"]  # (B,N,3,3), (B,N,3)
                    loss_rot, loss_trans, loss_rot_geodesic = regression_loss_relative_pose(self, R_clean, T_clean, R_pred, T_pred)

                    loss = loss_rot + loss_trans + loss_rot_geodesic

                    # ----- logging -----
                    bs = R_clean.size(0)
                    loss_det      = loss.detach()
                    loss_rot_det  = loss_rot.detach()
                    loss_trans_det= loss_trans.detach()
                    loss_rot_geodesic_det = loss_rot_geodesic.detach()

                    running += loss_det * bs
                    running_rot   += loss_rot_det * bs
                    running_trans += loss_trans_det * bs
                    running_rot_geodesic += loss_rot_geodesic_det * bs
                    n_seen  += R_clean.size(0)

            epoch_loss = (running / max(1, n_seen))
            epoch_rot  = (running_rot / max(1, n_seen))
            epoch_trans= (running_trans / max(1, n_seen))
            epoch_rot_geodesic= (running_rot_geodesic / max(1, n_seen))

            losses.append(epoch_loss.cpu())
            print(
                f"===> epoch {epoch:>3} | mean {epoch_loss:.6f} "
                f"| rot {epoch_rot:.6f} "
                f"| trans {epoch_trans:.6f} "
                f"| rot_geodesic {epoch_rot_geodesic:.6f} "
                f"| time {time.time()-epoch_start:.2f}s",
                flush=True
            )

        return losses

    @torch.no_grad()
    def sample(self,imgs = None, Ks = None, return_intermediates = False):
        device = self.device
        self.model.eval()

        if(self.feature_type == "pi3"):
            feats = compute_pi3_feats(self.extractor, imgs)
        elif(self.feature_type =="dust3r"):   
            feats = compute_dust3r_feats(self.extractor, imgs)   
        feats_b = feats.unsqueeze(0) if feats.dim() == 4 else feats  
        rgb_b   = imgs.unsqueeze(0)   if imgs.dim()   == 4 else imgs
        Ks_b    = Ks.unsqueeze(0)    if (Ks is not None and Ks.dim() == 3) else Ks

        fx = fy = None

        prediction = self.model(feats_b, rgb_b, Ks_b, return_intermediates = return_intermediates)
        R_pred = prediction["R"]        #[B,N,3,3]
        T_pred = prediction["t"]        # [B,N,3]
        
        if(return_intermediates):
            R_layers = prediction["R_layers"]  # (L,B,N,3,3)
            t_layers = prediction["t_layers"]  # (L,B,N,3)
            out = {
                "R": R_pred,            
                "t": T_pred,
                "fx": fx,
                "fy": fy,
                "R_layers": R_layers,
                "t_layers": t_layers
            }
        else:
            out = {
                "R": R_pred,            
                "t": T_pred,
                "fx": fx,
                "fy": fy
            }
        return out
