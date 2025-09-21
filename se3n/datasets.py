#!/usr/bin/env python
# coding: utf-8

# In[8]:

import sys
from pathlib import Path
base = Path(__file__).resolve().parents[1]  # .../PARCC/scripts
sys.path[:0] = [str(base / "dust3r" / "croco"), str(base / "dust3r")]
from dust3r.model        import load_model
from dust3r.utils.image  import load_images
from dust3r.image_pairs  import make_pairs
from dust3r.inference    import inference
import contextlib
from torch.amp import autocast
# ...


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
from typing import List, Dict, Tuple, Literal, Optional
import os
import time
from mpl_toolkits.mplot3d import Axes3D  
from typing import Union
import random


import gzip, json, re, argparse, os, sys
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Compose
from tqdm import tqdm
import cv2 


class RGBDepthDataset(Dataset):
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
        t2 = torch.from_numpy(sample["t1"])

        t2 = torch.from_numpy(np.array([1,0,0]))
        t1 = t2 
        print("t2", t2)
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


class RGBFeatureDataset(Dataset):
    """
    Loads paired RGB + feature .pt files.
    
    If invert=True, returns the inverse sample:
        img1 ↔ img2, feats[0] ↔ feats[1], relative pose becomes (R21, t21)
    
    Returns:
        img1, img2 : (3, H, W)  RGB images
        rgb        : (6, H, W)  stacked RGB channels
        feats      : (2, 1024, 32, 32)
        R          : (3, 3)  relative rotation
        t          : (3,)    relative translation
    """
    def __init__(self,
                 root: Union[str, Path],
                 device: str = "cuda",
                 invert: bool = False):
        self.root   = Path(root)
        self.files  = sorted(self.root.glob("*.pt"))
        self.device = torch.device(device)
        self.invert = invert

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = torch.load(self.files[idx], map_location="cpu", weights_only= True)

        img1, img2 = d["img1"], d["img2"]
        f1, f2     = d["feat1"].to(torch.float32), d["feat2"].to(torch.float32)
        R1, t1     = d["R1"], d["t1"]
        R2, t2     = d["R2"], d["t2"]

        if f1.shape[-2] != 32:
            f1 = f1.transpose(-2, -1)
            f2 = f2.transpose(-2, -1)

        if self.invert and random.random() < 0.5:
            img1, img2 = img2, img1
            f1, f2     = f2, f1
            R1, R2     = R2, R1
            t1, t2     = t2, t1

        rgb = torch.stack([img1, img2], dim=0)
        feats = torch.stack((f1, f2), dim=0)

        R_rel = R1.T @ R2
        t_rel = t2 - t1 @ R1.T @ R2

        return {
            "img1":  img1,        # (3, H, W)
            "img2":  img2,
            "rgb":   rgb,         # (6, H, W)
            "feats": feats,       # (2, 1024, 32, 32)
            "R":     R_rel,       # (3, 3)
            "t":     t_rel,       # (3,)
            "R1": R1, 
            "R2": R2, 
            "t1": t1, 
            "t2": t2
        }

class Dust3rFeatureExtractor(torch.nn.Module):
    """
    Wrapper that returns per-image encoder features from Dust3r.

    Input : img  [B, 3, H, W] (H,W multiples of 16)
    Output: feat [B, 1024, H/16, W/16]  (CUDA by default)
    """
    def __init__(self, ckpt_path: Path, device: str = "cuda"):
        super().__init__()
        self.model = load_model(str(ckpt_path), device=device).eval()
        # freeze weights
        for p in self.model.parameters():
            p.requires_grad = False

        # perf knobs
        self.device = torch.device(device)
        self.model.to(memory_format=torch.channels_last)  # better mem layout
        try:
            # enable FlashAttention for SDPA-backed attention (PyTorch 2.x)
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        except Exception:
            pass

    @torch.inference_mode()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img should already be on CUDA for speed.
        Returns CUDA tensor by default (no .cpu()).
        """
        assert img.dim() == 4 and img.shape[1] == 3, "img must be [B,3,H,W]"
        # keep on GPU, channels-last
        if img.device.type != "cuda":
            img = img.to(self.device, non_blocking=True)
        img = img.to(memory_format=torch.channels_last, non_blocking=True)

        B, _, H, W = img.shape

        use_amp = (img.device.type == "cuda")
        amp_dtype = (
            torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
        with ctx:
            x_tok, x_pos = self.model.patch_embed(img)   # [B,N,C], [B,N,C] or similar
            for blk in self.model.enc_blocks:
                x_tok = blk(x_tok, x_pos)
            x_tok = self.model.enc_norm(x_tok)           # [B,N,C]

            H_, W_ = H // 16, W // 16
            # Optional shape check (can omit for speed)
            # if x_tok.shape[1] != H_ * W_:
            #     raise ValueError(f"Expected {H_}*{W_} tokens, got {x_tok.shape[1]}")

            feat = x_tok.transpose(1, 2).reshape(B, x_tok.shape[-1], H_, W_)  # [B,1024,H/16,W/16]

        return feat  # stays on CUDA

def is_black(p: Path, thr: int = 5) -> bool:
    try:
        return np.asarray(Image.open(p).convert("L")).mean() < thr
    except Exception:
        return True

def load_category_annotations(co3d_root: Path, categories):
    """Return dict: seq_name -> list of frames (each a dict)."""
    structured = {}
    for cat in categories:
        ann_bbox = co3d_root / cat / "frame_annotations.bbox.jgz"
        ann_orig = co3d_root / cat / "frame_annotations.jgz"
        ann_file = ann_bbox if ann_bbox.exists() else ann_orig
        if not ann_file.exists():
            print(f"[skip] {ann_file} not found")
            continue
        with gzip.open(ann_file, "rt") as f:
            frames = json.load(f)
        for fr in frames:
            fr["category"] = cat  # keep category explicit
            structured.setdefault(fr["sequence_name"], []).append(fr)
    return structured


def image_path(co3d_root: Path, fr: dict) -> Path:
    # CO3D json stores relative "image.path" already like "<cat>/<seq>/images/frameXXXXXX.jpg"
    return co3d_root / fr["image"]["path"]

def mask_path_from_image(img_path: Path) -> Path:
    # swap images -> masks and extension
    return img_path.parent.parent / "masks" / img_path.name.replace(".jpg", ".png")

def extract_frame_no(frame_json):
    m = re.search(r"frame(\d+)\.", Path(frame_json["image"]["path"]).name)
    return int(m.group(1)) if m else -1


def collect_pairs(frames: List[dict],
                  img_root: Path,
                  gap: int,
                  black_thr: int = 5) -> List[Tuple[int, int]]:
    """Return list of (i,j) indices such that j = i+gap and both frames are OK"""
    valid = [i for i, fr in enumerate(frames)
             if (img_root / fr["image"]["path"]).exists()
             and not is_black(img_root / fr["image"]["path"], thr=black_thr)]

    pairs = []
    for k in range(len(valid) - gap):
        i, j = valid[k], valid[k + gap]
        pairs.append((i, j))
    return pairs


def _get_bbox_from_mask(
    mask: np.ndarray,
    box_crop_context: float = 0.1,
    thr: float = 0.5,
    decrease_quant: float = 0.05,
) -> Tuple[int, int, int, int]:
    # bbox in xywh
    masks_for_box = np.zeros_like(mask)
    while masks_for_box.sum() <= 1.0:
        masks_for_box = (mask > thr).astype(np.float32)
        thr -= decrease_quant
    
    if masks_for_box.sum() <= 1.0:
        raise ValueError("Mask is empty or invalid; cannot extract bounding box.")
    assert thr > 0.0
    x0, x1 = _get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = _get_1d_bounds(masks_for_box.sum(axis=-1))
    h, w = y1 - y0 + 1, x1 - x0 + 1
    if box_crop_context > 0.0:
        c = box_crop_context
        x0 -= w * c / 2
        y0 -= h * c / 2
        h += h * c
        w += w * c
        x1 = x0 + w
        y1 = y0 + h
    x0, x1 = [np.clip(x_, 0, mask.shape[1]) for x_ in [x0, x1]]
    y0, y1 = [np.clip(y_, 0, mask.shape[0]) for y_ in [y0, y1]]
    return np.round(np.array([x0, x1, y0, y1])).astype(int).tolist()


def _get_1d_bounds(arr: np.ndarray) -> Tuple[int, int]:
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1]

def restructure_co3d(jgz_path: str, category: str) -> dict:
    """
    Loads CO3D annotations from a .jgz file and returns a dictionary:
    {
        sequence_name: [
            {
                filepath: str,  # now includes category prefix
                bbox: [x1, y1, x2, y2],
                R: [3x3],
                T: [3],
                focal_length: [2],
                principal_point: [2],
            },
            ...
        ]
    }
    """
    import gzip, json

    with gzip.open(jgz_path, "rt") as f:
        raw_annotations = json.load(f)

    structured = {}
    for data in raw_annotations:

        seq = data["sequence_name"]
        vp = data["viewpoint"]
        if sum(vp["T"]) > 1e5:
            continue

        frame_num = data["frame_number"]
        rel_path = f"{category}/{seq}/images/frame{frame_num:06d}.jpg"

        if seq not in structured:
            structured[seq] = []

        structured[seq].append({
            "filepath": rel_path,
            "bbox": data.get("bbox", [0, 0, 1, 1]),
            "R": vp["R"],
            "T": vp["T"],
            "focal_length": vp["focal_length"],
            "principal_point": vp["principal_point"]
        })

    return structured

def bbox_xyxy_to_xywh(xyxy):
    wh = xyxy[2:] - xyxy[:2]
    xywh = np.concatenate([xyxy[:2], wh])
    return xywh

def preprocess_co3d(
    img_path: str,
    structured_annotations: dict,
    resize_hw=(224, 224),
    center_box: bool = True,
    use_mask_bbox: bool = True,
    box_crop_context: float = 0.1,
    mask_root: str = None,  # optional override
):
    # Load image
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    # Infer relative path
    category = Path(img_path).parts[-4]
    sequence = Path(img_path).parts[-3]
    filename = Path(img_path).name
    rel_path = f"{category}/{sequence}/images/{filename}"

    frames = structured_annotations.get(sequence, [])
    metadata = next((f for f in frames if f["filepath"] == rel_path), None)
    if metadata is None:
        raise ValueError(f"No metadata found for image {rel_path}")

    # Determine crop box
    if use_mask_bbox:
        rel_path_no_category = "/".join(rel_path.split("/")[1:])
        mask_rel_path = rel_path_no_category.replace("images", "masks").replace(".jpg", ".png")
        mask_full_path = Path(mask_root or Path(img_path).parents[2]) / mask_rel_path
        if not mask_full_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_full_path}")
        mask = np.array(Image.open(mask_full_path).convert("L")) / 255.0
        bbox = _get_bbox_from_mask(mask, box_crop_context=box_crop_context)
        x0, x1, y0, y1 = bbox
        bbox = [x0, y0, x1, y1]
    elif center_box:
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        bbox = [left, top, left + min_dim, top + min_dim]
    else:
        bbox = metadata.get("bbox", [0, 0, w, h])

    # Crop + resize
    image = transforms.functional.crop(
        image,
        top=int(bbox[1]),
        left=int(bbox[0]),
        height=int(bbox[3] - bbox[1]),
        width=int(bbox[2] - bbox[0]),
    )
    img_tensor = transforms.Compose([
        transforms.Resize(resize_hw, antialias=True),
        transforms.ToTensor()
    ])(image)

    return img_tensor, metadata
    
@torch.inference_mode()
def feat_from_tensor(img_tensor: torch.Tensor,
                     extractor: Dust3rFeatureExtractor,
                     device) -> torch.Tensor:
    return extractor(img_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
    
@torch.no_grad()    
def compute_dust3r_feats(extractor, imgs, device, use_amp: bool = True):
    """
    imgs: [B, K, 3, H, W] on (usually) CPU
    returns: [B, K, C, h, w] on `device`
    """
    dev = torch.device(device)
    B, K, C, H, W = imgs.shape
    flat = imgs.view(B * K, C, H, W).to(dev, non_blocking=True)

    amp_dtype = (
        torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    ctx = autocast("cuda", dtype=amp_dtype) if (use_amp and dev.type == "cuda") else contextlib.nullcontext()
    with ctx:
        f = extractor(flat)

    return f.detach().clone().view(B, K, *f.shape[1:]).contiguous()

class OldCo3dDataset(Dataset):
    def __init__(self,
                 co3d_root: Path,
                 categories,
                 extractor,                   # kept, but not used in __getitem__
                 k: int = 2,
                 resize_hw=(224, 224),
                 box_crop_context: float = 0.1,
                 black_thr: int = 5,
                 device: str = "cuda",
                 prefer_bbox: bool = True,      
                 mask_fallback: bool = True,
                 verbose: bool = True):
        super().__init__()
        self.root = Path(co3d_root)
        self.categories = set(categories)
        self.prefer_bbox = prefer_bbox
        self.mask_fallback = mask_fallback
        self.k = k
        self.resize_hw = resize_hw
        self.box_crop_context = box_crop_context
        self.black_thr = black_thr
        self.device = torch.device(device)   # not used here, but kept for API parity
        self.extractor = extractor           # kept, but NOT called in __getitem__
        self.verbose = verbose

        if self.verbose:
            print(f"[INIT] loading annotations from {self.root} for {sorted(self.categories)}")
        self.structured = load_category_annotations(self.root, self.categories)

        def _n_valid(frames):
            return sum(1 for fr in frames if not is_black(image_path(self.root, fr), thr=self.black_thr))

        self.sequences = []
        self._valid_counts = {}
        for seq, frames in self.structured.items():
            n = _n_valid(frames)
            self._valid_counts[seq] = n
            if n >= k:
                self.sequences.append(seq)

        if self.verbose:
            print(f"[INIT] sequences usable (>= {self.k} valid frames): {len(self.sequences)} / {len(self.structured)}")
            for s in list(self.sequences)[:5]:
                print(f"        - {s}: {self._valid_counts[s]} valid frames")

        self.to_tensor = transforms.Compose([
            transforms.Resize(resize_hw, antialias=True),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        frames = self._valid_frames(self.structured[seq])
        if self.verbose:
            print(f"[GET] idx={idx} seq={seq} valid_frames={len(frames)} (k={self.k})")
        assert len(frames) >= self.k, f"[{seq}] only {len(frames)} valid frames (need {self.k})"

        sampled = random.sample(frames, self.k)
        if self.verbose:
            fnames = [Path(fr['image']['path']).name for fr in sampled]
            print(f"[GET] sampled frames: {fnames}")

        imgs, Rs, Ts = [], [], []
        try:
            for fr in sampled:
                img_p = image_path(self.root, fr)
                msk_p = mask_path_from_image(img_p)

                if self.verbose:
                    print(f"[IMG] {img_p}  |  [MSK] {msk_p}")

                if not img_p.exists():
                    raise FileNotFoundError(f"Image missing: {img_p}")
                if not msk_p.exists():
                    raise FileNotFoundError(f"Mask missing: {msk_p}")

                used_precomp = False
                if self.prefer_bbox and isinstance(fr.get("bbox_precomp"), dict):
                    b = fr["bbox_precomp"]
                    x0, x1, y0, y1 = int(b["x0"]), int(b["x1"]), int(b["y0"]), int(b["y1"])
                    used_precomp = True
                else:
                    msk_p = mask_path_from_image(img_p)
                    if not msk_p.exists():
                        if self.mask_fallback:
                            # last-resort center square crop if mask missing
                            W, H = Image.open(img_p).size
                            s = min(W, H); left = (W - s) // 2; top = (H - s) // 2
                            x0, x1, y0, y1 = left, left + s, top, top + s
                        else:
                            raise FileNotFoundError(f"Mask missing: {msk_p}")
                    else:
                        mask = np.array(Image.open(msk_p).convert("L")) / 255.0
                        x0, x1, y0, y1 = _get_bbox_from_mask(mask, box_crop_context=self.box_crop_context)

                if self.verbose:
                    src = "precomp" if used_precomp else "mask/center"
                    print(f"[BBOX-{src}] (x0,x1,y0,y1)=({x0},{x1},{y0},{y1})")

                image = Image.open(img_p).convert("RGB")
                mask  = np.array(Image.open(msk_p).convert("L")) / 255.0

                if self.verbose:
                    H, W = mask.shape
                    print(f"[BBOX] (x0,x1,y0,y1)=({x0},{x1},{y0},{y1}) in WxH=({W},{H})")

                image = transforms.functional.crop(image, top=int(y0), left=int(x0),
                                                   height=int(y1-y0), width=int(x1-x0))
                img_t = self.to_tensor(image)           # (3,H,W), CPU
                imgs.append(img_t)

                vp = fr["viewpoint"]
                Rs.append(torch.tensor(vp["R"], dtype=torch.float32).reshape(3,3))
                Ts.append(torch.tensor(vp["T"], dtype=torch.float32))
        except Exception as e:
            raise RuntimeError(f"[{seq}] failure while preparing sample.\n"
                               f"  root={self.root}\n"
                               f"  sampled={[Path(fr['image']['path']) for fr in sampled]}\n"
                               f"  error={e}") from e

        batch = {
            "imgs":  torch.stack(imgs),  # (k,3,H,W)  CPU
            "R":     torch.stack(Rs),    # (k,3,3)    CPU
            "t":     torch.stack(Ts),    # (k,3)      CPU
            "sequence": seq,
        }
        if self.verbose:
            print(f"[RET] imgs {tuple(batch['imgs'].shape)}  R {tuple(batch['R'].shape)}  t {tuple(batch['t'].shape)}")
        return batch

    def _valid_frames(self, frames):
        return [fr for fr in frames if not is_black(image_path(self.root, fr), thr=self.black_thr)]
        
def _discover_categories_from_ann(ann_root: Path, split: str) -> List[str]:
    cats = []
    for p in ann_root.glob("*.jgz"):
        name = p.name
        if name.endswith(f"_{split}.jgz"):
            cats.append(name[:-(len(f"_{split}.jgz"))])
    return sorted(set(cats))

class Co3dDataset(Dataset):
    """
    Multi-category CO3D dataset that mirrors OldCo3dDataset's API
    but reads from precomputed compact annotations.

    Differences vs OldCo3dDataset:
      - No mask I/O; uses precomputed bbox (xyxy pixels) from ann files
      - No black-frame scanning; assumed handled offline
      - Supports multiple categories natively by merging ann files
    """
    def __init__(self,
                 co3d_root: Path,
                 categories,                     # list[str] or "all"
                 extractor,                      # kept for API parity; not used
                 k: int = 2,
                 resize_hw=(224, 224),
                 box_crop_context: float = 0.1,  # kept for API parity; not used
                 black_thr: int = 5,             # kept for API parity; not used
                 device: str = "cuda",           # kept for API parity; not used
                 prefer_bbox: bool = True,       # kept for API parity; not used
                 mask_fallback: bool = True,     # kept for API parity; not used
                 verbose: bool = True,
                 *,
                 ann_root: Path = None,          # NEW: directory with <cat>_<split>.jgz
                 split: str = "train",           # NEW: which ann split to load
                 ):
        super().__init__()
        assert ann_root is not None, "Please pass ann_root=... (dir with <category>_<split>.jgz)"
        self.root = Path(co3d_root)
        self.k = int(k)
        self.resize_hw = tuple(resize_hw)
        self.verbose = verbose
        self.ann_root = Path(ann_root)
        self.split = split

        # Resolve categories
        if categories == "all" or categories == ["all"]:
            cats = _discover_categories_from_ann(self.ann_root, split=self.split)
            if verbose:
                print(f"[INIT] discovered categories from {self.ann_root}: {cats}")
        else:
            cats = list(categories)

        # Load all requested categories and merge into one by-sequence index
        # Each ann file is: {sequence_name: [ {filepath, R, T, focal_length, principal_point, bbox}, ... ] }
        merged_by_seq: Dict[str, List[dict]] = {}
        total_seq_in_files = 0

        for cat in cats:
            ann_path = self.ann_root / f"{cat}_{self.split}.jgz"
            if not ann_path.exists():
                if verbose:
                    print(f"[WARN] missing annotations for category '{cat}' at {ann_path}; skipping")
                continue

            with gzip.open(ann_path, "rt") as f:
                per_seq = json.load(f)
            total_seq_in_files += len(per_seq)

            # Use composite sequence key to avoid collisions across categories
            for seq, records in per_seq.items():
                comp_seq = f"{cat}/{seq}"
                # keep only sequences with >= k frames
                if len(records) >= self.k:
                    merged_by_seq[comp_seq] = records

        self.by_seq = merged_by_seq
        self.sequences = sorted(self.by_seq.keys())

        if self.verbose:
            print(f"[INIT] loaded {len(self.sequences)} usable sequences (>= {self.k} frames) "
                  f"from {len(cats)} categories (raw seq in files: {total_seq_in_files})")
            print(f"       split='{self.split}'  ann_root='{self.ann_root}'")
            if self.sequences[:3]:
                print("       examples:", self.sequences[:3])

        self.to_tensor = transforms.Compose([
            transforms.Resize(self.resize_hw, antialias=True),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]                  # composite key: "<cat>/<seq>"
        frames = self.by_seq[seq]
        if len(frames) < self.k:
            raise RuntimeError(f"[{seq}] has only {len(frames)} frames (k={self.k})")
        sampled = random.sample(frames, self.k)

        imgs, Rs, Ts = [], [], []
        try:
            for rec in sampled:
                img_p = self.root / rec["filepath"]  # "<cat>/<seq>/images/frameXXXXXX.jpg"
                if not img_p.exists():
                    raise FileNotFoundError(f"Image missing: {img_p}")

                # Crop using precomputed bbox (xyxy in pixels)
                x0, y0, x1, y1 = map(int, rec["bbox"])
                image = Image.open(img_p).convert("RGB")
                image = transforms.functional.crop(image, top=y0, left=x0,
                                                   height=y1 - y0, width=x1 - x0)
                img_t = self.to_tensor(image)           # (3,H,W)
                imgs.append(img_t)

                Rs.append(torch.tensor(rec["R"], dtype=torch.float32).reshape(3, 3))
                Ts.append(torch.tensor(rec["T"], dtype=torch.float32))
        except Exception as e:
            raise RuntimeError(f"[{seq}] failure while preparing sample: {e}") from e

        batch = {
            "imgs": torch.stack(imgs),   # (k,3,H,W)
            "R":    torch.stack(Rs),     # (k,3,3)
            "t":    torch.stack(Ts),     # (k,3)
            "sequence": seq,             # composite "<category>/<sequence_name>" to stay unique
        }
        return batch