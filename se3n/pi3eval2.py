#!/usr/bin/env python3

import os
import os.path as osp
import json
import random
import argparse
import logging
from typing import List, Optional
import math

import numpy as np
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

def infer_cameras_w2c(filelist, model: Pi3, device: str, images = None):
    """
    """
    #imgs = load_and_resize14(filelist, new_width=LOAD_IMG_SIZE, device=device, verbose=VERBOSE)

    dtype = (
        torch.bfloat16
        if (device == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
        else torch.float16
    )

    if(images is not None): 
        imgs = images.to(device).unsqueeze(0)

        
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
            pred = model(imgs)

    poses_c2w_all = pred["camera_poses"].detach().cpu()   # [1,V,4,4]
    print("poses_c2w:", poses_c2w_all.mean().item(), poses_c2w_all.std().item())
    extrinsics_w2c = se3_inverse(poses_c2w_all[0])        # [V,4,4]
    return extrinsics_w2c, None


def eval_dataset_co3dv2_fixed_seq(
    *,
    model: Pi3,
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

        if 0 == 0:
            print("imgs:", imgs.shape, imgs.dtype, imgs.min().item(), imgs.max().item(), imgs.mean().item(), imgs.std().item())
            print("seq_id sample:", batch["seq_id"][:3] if isinstance(batch["seq_id"], list) else batch["seq_id"])


        pred_extrs, _ = infer_cameras_w2c(batch["image_paths"], model, device=device, images = batch["imgs"])

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


def eval_dataset_co3dv2(
    *,
    model: Pi3,
    dataset, 
    device: str,
    thresholds=(5, 10, 15, 30),
    num_imgs = [10,11],
):
    logger = logging.getLogger("pi3-eval")

    batch_size = 20
    num_workers = 4
    dtype = (
        torch.bfloat16
        if (device == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
        else torch.float16
    )
    sampler = DynamicBatchSampler(len(dataset), dataset_len = batch_size, max_images = 300, images_per_seq = num_imgs)

    loader = DataLoader(
        dataset,
        batch_sampler=sampler, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    rError, tError = [], []

    pbar = tqdm(loader, desc="[CO3Dv2 eval]")
    for step, batch in enumerate(pbar):

        imgs = batch["imgs"].to(device, non_blocking=True)   # [B,V,3,H,W]
        R_gt = batch["R"].to(device, non_blocking=True)      # [B,V,3,3]
        T_gt = batch["T"].to(device, non_blocking=True)      # [B,V,3]
        gt_c2w = pack_se3_c2w(R_gt, T_gt)                    # [B,V,4,4]
        gt_w2c = se3_inverse(gt_c2w)                         # [B,V,4,4]

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=(device == "cuda")):
                pred = model(imgs)

        # Expect [B,V,4,4] in c2w
        pred_c2w = pred["camera_poses"]                      # [B,V,4,4]
        pred_w2c = se3_inverse(pred_c2w)                     # [B,V,4,4]

        B, V = pred_w2c.shape[:2]

        num_views = batch.get("num_views", None)  # [B] or None

        # --- per-item evaluation ---
        rr_means, tt_means = [], []
        for b in range(B):
            v = int(num_views[b]) if num_views is not None else V

            pred_extrs_b = pred_w2c[b, :v]  # [v,4,4]
            gt_extrs_b   = gt_w2c[b, :v]    # [v,4,4]

            rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
                pred_se3=pred_extrs_b,
                gt_se3=gt_extrs_b,
                num_frames=v,
            )

            rr = rel_rangle_deg.detach().cpu().numpy().reshape(-1)
            tt = rel_tangle_deg.detach().cpu().numpy().reshape(-1)
            rError.append(rr)
            tError.append(tt)

            rr_means.append(rr.mean())
            tt_means.append(tt.mean())

        pbar.set_postfix_str(f"B={B} V~{V} r={np.mean(rr_means):5.2f}° t={np.mean(tt_means):5.2f}°")

    if not rError:
        raise RuntimeError("No sequences evaluated (rError empty). Check your dataset/dataloader.")

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
    ap.add_argument("--co3d_root", required=True, help="CO3Dv2 dataset root")
    ap.add_argument("--ann_root", required=True, help="CO3Dv2 annotation root used by your dataset class")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit_seqs", type=int, default=0, help="0 = no limit; else evaluate only first N sequences")
    ap.add_argument("--seed", type=int, default=SEED)
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
    model = Pi3.from_pretrained(PI3_CKPT).to(device).eval()
    logger.info(f"Loaded Pi3 checkpoint: {PI3_CKPT} on {device}")

    ann_root = "/vast/projects/kostas/geometric-learning/Co3D/Co3DANNOTATIONS_FEWVIEW_DEV"
    dataset = Co3dDataset(
        CO3D_DIR=args.co3d_root,
        CO3D_ANNOTATION_DIR=ann_root,   
        split_name="test",
        min_num_images=50,
    )

    # 3) evaluate
    metrics = eval_dataset_co3dv2_fixed_seq(
        model=model,
        dataset=dataset,
        seq_id_map_path=seq_id_map,
        device=device,
        thresholds=(5, 10, 15, 30),
        limit_seqs=args.limit_seqs)

    print("\n=== Fixed Sequence ===")
    for k, v in metrics.items():
        print(f"{k:8s}: {v:6.2f}")

    """
    dataset = PoseCo3DDataset(
        co3d_root=args.co3d_root,
        categories=("all",), 
        resize_hw=(224, 224),
        verbose=True, 
        ann_root=args.ann_root,           
        split="test",               
        )


    metrics = eval_dataset_co3dv2(
        model=model,
        dataset=dataset,
        device=device,
        thresholds=(5, 10, 15, 30),
        num_imgs = [10,11],
    )

    print("\n=== CO3Dv2 224 Res, 10 images, Dataloader ===")
    for k, v in metrics.items():
        print(f"{k:8s}: {v:6.2f}")

    torch.cuda.empty_cache()
    """

if __name__ == "__main__":
    main()
