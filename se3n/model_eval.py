import os
import os.path as osp
import json
import random
import argparse
import logging
from typing import List, Optional
import math

import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm

from pi3.models.pi3 import Pi3
from pi3.relpose.metric import se3_to_relative_pose_error, calculate_auc_np

from pi3.datasets.co3dv2_dataset import CO3DV2Dataset 
from se3n_datasets import PoseCo3DDataset, DynamicBatchSampler
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, DataLoader
from pi3.datasets.co3d_v2 import Co3dDataset
import torchvision.transforms.functional as TVF
import torchvision.transforms as tvf
import torch.nn.functional as F
from PIL import Image

import argparse, time, math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from accelerate import Accelerator
from copy import deepcopy


# UFM
from uniflowmatch.models.ufm import (
    UniFlowMatchConfidence,
    UniFlowMatchClassificationRefinement,
)

from feature_extractors import UFMGGSOptimizer, GGSOptimizer, Dust3rFeatureExtractor, DepthAnythingMetricOptimizer 

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2_metric.dpt import DepthAnythingV2 as MetricDepthAnythingV2

from EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, reparameter

from SE3NDiffusion import se3n_diffuser
from SE3NRegression import se3n_regressor
import yaml

PI3_CKPT = "yyfz233/Pi3"
LOAD_IMG_SIZE = 512
VERBOSE = False
SEED = 42

# ----------------------------
# Image loading helper
# ----------------------------
def load_and_resize14(filelist: List[str], new_width: int, device: str, verbose: bool):
    imgs = load_images(filelist, new_width=new_width, verbose=verbose).to(device)

    ori_h, ori_w = imgs.shape[-2:]
    patch_h, patch_w = ori_h // 14, ori_w // 14
    # (N, 3, h, w) -> (1, N, 3, h_14, w_14)
    imgs = F.interpolate(imgs, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=False, antialias=True).unsqueeze(0)
    return imgs

def load_images(filelist: List[str], PIXEL_LIMIT: int = 255000, new_width: Optional[int] = None, verbose: bool = False):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    
    # --- 1. Load image paths or video frames ---
    for img_path in filelist:
        try:
            sources.append(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    if verbose:
        print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    if new_width is None:
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    else:
        TARGET_W, TARGET_H = new_width, round(H_orig * (new_width / W_orig) / 14) * 14
    if verbose:
        print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = tvf.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)

def save_image_comparison(
    imgs_loader: torch.Tensor,   # [N,3,224,224]
    imgs_loaded: torch.Tensor,   # [N,3,224,224]
    img_paths: List[str],
    out_dir: str,
    prefix: str = "compare",
    max_imgs: int = 5,
):
    imgs_loaded = imgs_loaded.squeeze(0)
    os.makedirs(out_dir, exist_ok=True)

    N = min(imgs_loader.shape[0], max_imgs)
    for i in range(N):
        a = imgs_loader[i].detach().cpu().clamp(0, 1)
        b = imgs_loaded[i].detach().cpu().clamp(0, 1)

        # [3,H,W] → [H,W,3]
        a = (a.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        b = (b.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        side = np.concatenate([a, b], axis=1)
        name = os.path.basename(img_paths[i]).replace(".jpg", "").replace(".png", "")
        Image.fromarray(side).save(
            os.path.join(out_dir, f"{prefix}_{i:02d}_{name}.png")
        )

def pack_se3_c2w(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Batch-safe pack of SE(3) camera-to-world matrices.

    Args:
        R: [..., 3, 3]
        t: [..., 3]

    Returns:
        T: [..., 4, 4]
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must end with (3,3), got {R.shape}")
    if t.shape[-1] != 3:
        raise ValueError(f"t must end with (3,), got {t.shape}")
    if R.shape[:-2] != t.shape[:-1]:
        raise ValueError(f"Leading dims must match: R {R.shape[:-2]} vs t {t.shape[:-1]}")

    # Create identity with the right leading shape
    T = torch.eye(4, device=R.device, dtype=R.dtype).expand(R.shape[:-2] + (4, 4)).clone()

    T[..., :3, :3] = R
    T[..., :3,  3] = t
    return T

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

def infer_cameras_w2c(imgs, model, model_type, device: str, intrinsics=None, img_paths: Optional[List[str]] = None, gt_poses = None):
    if model_type in ("ufm", "ggs", "tufm", "tggs"):
        if intrinsics is not None:
            intrinsics = intrinsics.to(dtype=torch.float32)
        imgs_opt = imgs.to(dtype=torch.float32) 

        with torch.autocast(device_type="cuda", enabled=False):
            pred = model(imgs_opt, intrinsics, optim_steps = 1000)
            model.clear_cached_matches()
        
        poses_c2w = pred["camera_poses"].detach().cpu()
        pred_w2c = se3_inverse(poses_c2w)
    elif model_type == "depth_anything":
        intrinsics = intrinsics.to(dtype=torch.float32)
        imgs_opt = imgs.to(dtype=torch.float32)

        with torch.autocast(device_type="cuda", enabled=False):
            pred = model(images=imgs_opt, Ks=intrinsics, poses=gt_poses, optim_steps=50, verbose=False)
            model.clear_cached()

        poses_c2w = pred["camera_poses"].detach().cpu()   # [N,4,4]
        pred_w2c = se3_inverse(poses_c2w)
    elif model_type == "pi3":
        dtype = (
            torch.bfloat16
            if (device == "cuda" and torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 8)
            else torch.float16
        )
        with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
            with torch.no_grad():
                pred = model(imgs.unsqueeze(0))
        poses_c2w = pred["camera_poses"].detach().cpu()
        pred_w2c = se3_inverse(poses_c2w[0])
    else: 
        # diffusion / regression
        imgs_opt = imgs.to(dtype=torch.float32) 

        with torch.autocast(device_type="cuda", enabled=False):
            pred = model.sample(imgs.unsqueeze(0), intrinsics.unsqueeze(0), guidance = True)
        
        poses_c2w = pred["camera_poses"].detach().cpu()
        pred_w2c = se3_inverse(poses_c2w[0])

    return pred_w2c, None


def eval_dataset_co3dv2_fixed_seq(
    *,
    model: Any, 
    model_type: "ufm" ,
    dataset: Co3dDataset,
    seq_id_map_path: str,
    device: str,
    thresholds=(5, 10, 15, 30),
    limit_seqs: int = 0,
):
    logger = logging.getLogger("pi3-eval")

    with open(seq_id_map_path, "r") as f:
        raw = json.load(f)

    # allow matching by full key OR by basename (strip "category/" etc)
    seq_id_map = {}
    for k, v in raw.items():
        seq_id_map[k] = v
        seq_id_map[k.split("/")[-1]] = v

    seqs = list(dataset.sequence_list)
    if limit_seqs and limit_seqs > 0:
        seqs = seqs[:limit_seqs]
    rError = []
    tError = []

    tbar = tqdm(seqs, desc="[CO3Dv2 eval]")
    for seq_name in tbar:
        if seq_name not in seq_id_map:
            # Pi3 eval map should cover the relpose subset; skip anything missing.
            continue

        n = dataset.get_seq_framenum(sequence_name=seq_name)
        ids = [int(i) for i in seq_id_map[seq_name]]

        mx = int(np.max(ids)) if len(ids) else -1
        if mx >= n:
            print(f"[BAD IDS] {seq_name}: len(metadata)={n}, max_id={mx}, ids={ids}")


        if any(i < 0 or i >= n for i in ids):
            tbar.set_postfix_str(f"{seq_name} [skip bad ids: n={n}, max={max(ids)}]")
            continue

        batch = dataset.get_data(sequence_name=seq_name, ids=ids)
        gt_extrs = batch["extrs"]  # [V,4,4] (w2c)
        

        imgs = batch["imgs"].to(device)#, non_blocking=True)
        K    = batch["K"].to(device)#, non_blocking=True)
        img_paths = batch["image_paths"]
        
        gt_extrs_c2w = se3_inverse(gt_extrs)
        pred_extrs, _ = infer_cameras_w2c(imgs, model, model_type = model_type, device=device, intrinsics = K, img_paths = img_paths, gt_poses = gt_extrs_c2w)

        # metric expects w2c by default in Pi3's relpose-angular scripts
        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
            pred_se3=pred_extrs,
            gt_se3=gt_extrs,
            num_frames=len(ids),
        )

        rr = rel_rangle_deg.detach().cpu().numpy().reshape(-1)
        tt = rel_tangle_deg.detach().cpu().numpy().reshape(-1)
        rError.append(rr)
        tError.append(tt)

        tbar.set_postfix_str(f"{seq_name} r={rr.mean():5.2f}° t={tt.mean():5.2f}°")

    if not rError:
        raise RuntimeError("No sequences evaluated (rError empty). Check dataset.sequence_list / seq_id_map coverage.")

    rError = np.concatenate(rError, axis=0)
    tError = np.concatenate(tError, axis=0)

    metrics = {}
    for thr in thresholds:
        metrics[f"RRA@{thr}"] = float((rError < thr).mean() * 100.0)
        metrics[f"RTA@{thr}"] = float((tError < thr).mean() * 100.0)
        auc, _ = calculate_auc_np(rError, tError, max_threshold=thr)
        metrics[f"AUC@{thr}"] = float(auc * 100.0)

    logger.info(f"CO3Dv2 metrics: {metrics}")
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--co3d_root", default="/vast/projects/kostas/geometric-learning/Co3D",
        help="CO3Dv2 dataset root")

    ap.add_argument(
        "--ann_root", default="/vast/projects/kostas/geometric-learning/Co3D/CO3DANNOTATIONS",
        help="CO3Dv2 annotation root used by your dataset class")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit_seqs", type=int, default=0, help="0 = no limit; else evaluate only first N sequences")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--model_type", choices=["tggs", "tufm", "ufm", "ggs", "pi3", "diffusion", "vggt", "regression", "depth_anything"], default="ufm")
    ap.add_argument("--config", type=str, default="", help="model config if needed")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("pi3-eval")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    seq_id_map = "/vast/home/m/mgjacob/PARCC/scripts/pi3/datasets/CO3Dv2_relpose_seq-id-map_seed42.json"

    # 1) load model
    if args.model_type == "pi3":
        model = Pi3.from_pretrained(PI3_CKPT).to(device).eval()
    elif args.model_type == "tufm": 
        matcher = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")
        matcher = matcher.to(device).eval()
        model = TheseusUFMGGSOptimizer(matcher)
    elif args.model_type == "ufm":
        matcher = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")
        matcher = matcher.to(device).eval()
        model = UFMGGSOptimizer(matcher)
    elif args.model_type == "ggs":
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
        model = GGSOptimizer(matcher)
        print("Using GGS Optimizer")
    elif args.model_type == "tggs": 
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
        model = TheseusELofterOptimizer(matcher)
        print("Using Theseus")
    elif args.model_type == "diffusion" or args.model_type == "regression":
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            
        run_name     = cfg["name"]
        conditioning = cfg["conditioning"]
        co3d_root     = cfg["dataset_dir"]
        ann_root     = cfg["ann_root"]
        model_type   = cfg["model_type"]
        attn_kwargs  = cfg["attn_kwargs"]
        attn_args    = cfg["attn_args"]
        scheme       = cfg["scheme"]
        prediction  = cfg["prediction"]
        so3_config  = cfg["so3_config"]
        r3_config   = cfg["r3_config"]
        num_sequences = cfg["num_sequences"]
        plot_name = cfg["plot_name"]
        BATCH_GLOBAL = cfg["batch_size"]
        feature_type = cfg["feature_type"]
        prediction_type = cfg["prediction_type"]
        update_type = cfg["update_type"]
        output_dir  = Path(f"/vast/projects/kostas/geometric-learning/mgjacob/{run_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "model.pt"
        
        extractor = Dust3rFeatureExtractor(
            ckpt_path=Path("/vast/projects/kostas/geometric-learning/mgjacob/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"),
            device=str(device)  
        )

        se3_config = {
            "T": 100,
            "device": device,
            "conditioning": conditioning,
            "dataloader": None,
            "attn_args": attn_args, 
            "attn_kwargs": attn_kwargs, 
            "extractor": extractor, 
            "feature_type": feature_type,
            "save_model": True,
            "model_type": model_type,
            "prediction": prediction,
            "so3_config": so3_config,
            "r3_config": r3_config,
            "save_path": output_dir / "model.pt",
            "model_path": model_path,
            "dataset_root": args.co3d_root,
            "dataset": None,
            "forward_process": "ve", 
            "representation": "rot9d", 
            "scheme": scheme,
            "accelerator": None, 
            "update_type": update_type,
            "guidance_type": "ufm",
        }
        if(prediction_type == "regressor"):
            model = se3n_regressor(se3_config)
        else:
            model = se3n_diffuser(se3_config)
    elif(args.model_type == "depth_anything"): 
        model_configs = {
            "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }

        ckpt_root = "/vast/projects/kostas/geometric-learning/mgjacob/checkpoints"

        max_depth = 20
        encoder = "vitl"
        ckpt_path = f"{ckpt_root}/depth_anything_v2_metric_hypersim_{encoder}.pth"
        depth_model = MetricDepthAnythingV2(**model_configs[encoder], max_depth=max_depth)

        state = torch.load(ckpt_path, map_location="cpu")
        depth_model.load_state_dict(state)
        depth_model = depth_model.to(device).eval()
        model = DepthAnythingMetricOptimizer(depth_model)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print("MODEL:", type(model))
    if(args.model_type in ["diffusion", "regression"]):
        print("DEVICE:", next(model.model.parameters()).device)
    else:
        print("DEVICE:", next(model.parameters()).device, "EVAL:", (not model.training))

    ann_root = "/vast/projects/kostas/geometric-learning/Co3D/Co3DANNOTATIONS_FEWVIEW_DEV"
    dataset = Co3dDataset(
        CO3D_DIR=args.co3d_root,
        CO3D_ANNOTATION_DIR=ann_root,
        split_name="test",
        min_num_images=50,
    )

    metrics = eval_dataset_co3dv2_fixed_seq(
        model=model,
        model_type=args.model_type,
        dataset=dataset,
        seq_id_map_path=seq_id_map,
        device=device,
        thresholds=(5, 10, 15, 30),
        limit_seqs=args.limit_seqs,
    )

    print("\n=== RESULTS ===")
    for k, v in metrics.items():
        print(f"{k:8s}: {v:6.2f}")

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
