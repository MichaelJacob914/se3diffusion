#!/usr/bin/env python
# coding: utf-8

# In[8]:

import sys
from pathlib import Path
import PIL
base = Path(__file__).resolve().parents[1]  # .../PARCC/scripts
sys.path[:0] = [str(base / "dust3r" / "croco"), str(base / "dust3r"), str(base / "pytorch3d_min")]
#from pytorch3d_min.pytorch3d_min.cameras import PerspectiveCameras, normalize_cameras

from torch.utils.data import BatchSampler

import os.path as osp
import contextlib
from collections import defaultdict
from torch.amp import autocast
from itertools import combinations

import os, glob
# ...



from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

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
from typing import List, Dict, Tuple, Literal, Optional, Iterable
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


TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = ["ball", "book", "couch", "frisbee", "hotdog", "kite", "remote", "sandwich", "skateboard", "suitcase"]


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

def bbox_xyxy_to_xywh(xyxy):
    wh = xyxy[2:] - xyxy[:2]
    xywh = np.concatenate([xyxy[:2], wh])
    return xywh

class DynamicBatchSampler(BatchSampler):
    def __init__(self, num_sequences, dataset_len=1024, max_images=128, images_per_seq=(2, 5)):
        # Batch sampler with a dynamic number of sequences
        # max_images >= number_of_sequences * images_per_sequence

        self.max_images = max_images
        self.images_per_seq = list(range(images_per_seq[0], images_per_seq[1]))
        self.num_sequences = num_sequences
        self.dataset_len = dataset_len

    def __iter__(self):
        for _ in range(self.dataset_len):
            # number per sequence
            n_per_seq = np.random.choice(self.images_per_seq)
            # number of sequences
            n_seqs = self.max_images // n_per_seq

            # randomly select sequences
            chosen_seq = self._capped_random_choice(self.num_sequences, n_seqs)

            # get item
            batches = [(bidx, n_per_seq) for bidx in chosen_seq]
            yield batches

    def _capped_random_choice(self, x, size, replace: bool = True):
        len_x = x if isinstance(x, int) else len(x)
        if replace:
            choice = np.random.choice(x, size=size, replace=len_x < size)
            return choice
        else:
            return np.random.choice(x, size=min(size, len_x), replace=False)

    def __len__(self):
        return self.dataset_len


def square_bbox(bbox, padding=0.0, astype=None):
    """
    Computes a square bounding box, with optional padding parameters.

    Args:
        bbox: Bounding box in xyxy format (4,).

    Returns:
        square_bbox in xyxy format (4,).
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2
    s = max(extents) * (1 + padding)
    square_bbox = np.array([center[0] - s, center[1] - s, center[0] + s, center[1] + s], dtype=astype)
    return square_bbox

def adjust_camera_to_bbox_crop_(fl, pp, image_size_wh: torch.Tensor, clamp_bbox_xywh: torch.Tensor):
    focal_length_px, principal_point_px = _convert_ndc_to_pixels(fl, pp, image_size_wh)
    principal_point_px_cropped = principal_point_px - clamp_bbox_xywh[:2]

    focal_length, principal_point_cropped = _convert_pixels_to_ndc(
        focal_length_px, principal_point_px_cropped, clamp_bbox_xywh[2:]
    )

    return focal_length, principal_point_cropped


def adjust_camera_to_image_scale_(fl, pp, original_size_wh: torch.Tensor, new_size_wh: torch.LongTensor):
    focal_length_px, principal_point_px = _convert_ndc_to_pixels(fl, pp, original_size_wh)

    # now scale and convert from pixels to NDC
    image_size_wh_output = new_size_wh.float()
    scale = (image_size_wh_output / original_size_wh).min(dim=-1, keepdim=True).values
    focal_length_px_scaled = focal_length_px * scale
    principal_point_px_scaled = principal_point_px * scale

    focal_length_scaled, principal_point_scaled = _convert_pixels_to_ndc(
        focal_length_px_scaled, principal_point_px_scaled, image_size_wh_output
    )
    return focal_length_scaled, principal_point_scaled


def _convert_ndc_to_pixels(focal_length: torch.Tensor, principal_point: torch.Tensor, image_size_wh: torch.Tensor):
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    return focal_length_px, principal_point_px


def _convert_pixels_to_ndc(
    focal_length_px: torch.Tensor, principal_point_px: torch.Tensor, image_size_wh: torch.Tensor
):
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point = (half_image_size - principal_point_px) / rescale
    focal_length = focal_length_px / rescale
    return focal_length, principal_point
        
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K

class ImageList:
    """ Convenience class to aply the same operation to a whole set of images.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch('crop', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def convert_ndc_to_pinhole(focal_length, principal_point, image_size):
    focal_length = np.array(focal_length)
    principal_point = np.array(principal_point)
    image_size_wh = np.array([image_size[1], image_size[0]])
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    fx, fy = focal_length_px[0], focal_length_px[1]
    cx, cy = principal_point_px[0], principal_point_px[1]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K

def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    R = R[None, :, :]
    T = T[None, :]
    focal = focal[None, :]
    p0 = p0[None, :]
    image_size = image_size[None, :]

    R_pytorch3d = R.copy()
    T_pytorch3d = T.copy()
    focal_pytorch3d = focal
    p0_pytorch3d = p0
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.transpose(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size[:, ::-1]

    # NDC to screen conversion.
    scale = np.min(image_size_wh, axis=1, keepdims=True) / 2.0
    scale = np.repeat(scale, 2, axis=1)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = np.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R[0], tvec[0], camera_matrix[0]


def rescale_image_depthmap(image, depthmap, camera_intrinsics, output_resolution, force=True, normal=None, far_mask=None):
    """ Jointly rescale a (image, depthmap) 
        so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic)
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)
        
    if normal is not None:
        normal = cv2.resize(normal, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)
    if far_mask is not None:
        far_mask = cv2.resize(far_mask, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), depthmap, camera_intrinsics, normal, far_mask
    
def center_crop_image_depthmap(image, depthmap, camera_intrinsics, crop_scale, normal=None, far_mask=None):
    """
    Jointly center-crop an image and its depthmap, and adjust the camera intrinsics accordingly.

    Parameters:
    - image: PIL.Image or similar, the input image.
    - depthmap: np.ndarray, the corresponding depth map.
    - camera_intrinsics: np.ndarray, the 3x3 camera intrinsics matrix.
    - crop_scale: float between 0 and 1, the fraction of the image to keep.

    Returns:
    - cropped_image: PIL.Image, the center-cropped image.
    - cropped_depthmap: np.ndarray, the center-cropped depth map.
    - adjusted_intrinsics: np.ndarray, the adjusted camera intrinsics matrix.
    """
    # Ensure crop_scale is valid
    assert 0 < crop_scale <= 1, "crop_scale must be between 0 and 1"

    # Convert image to ImageList for consistent processing
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (width, height)
    if depthmap is not None:
        # Ensure depthmap matches the image size
        assert depthmap.shape[:2] == tuple(image.size[::-1]), "Depthmap size must match image size"

    # Compute output resolution after cropping
    output_resolution = np.floor(input_resolution * crop_scale).astype(int)
    # get the correct crop_scale
    crop_scale = output_resolution / input_resolution

    # Compute margins (amount to crop from each side)
    margins = input_resolution - output_resolution
    offset = margins / 2  # Since we are center cropping

    # Calculate the crop bounding box
    l, t = offset.astype(int)
    r = l + output_resolution[0]
    b = t + output_resolution[1]
    crop_bbox = (l, t, r, b)

    # Crop the image and depthmap
    image = image.crop(crop_bbox)
    if depthmap is not None:
        depthmap = depthmap[t:b, l:r]
    if normal is not None:
        normal = normal[t:b, l:r]
    if far_mask is not None:
        far_mask = far_mask[t:b, l:r]

    # Adjust the camera intrinsics
    adjusted_intrinsics = camera_intrinsics.copy()

    # Adjust focal lengths (fx, fy)                         # no need to adjust focal lengths for cropping
    # adjusted_intrinsics[0, 0] /= crop_scale[0]  # fx
    # adjusted_intrinsics[1, 1] /= crop_scale[1]  # fy

    # Adjust principal point (cx, cy)
    adjusted_intrinsics[0, 2] -= l  # cx
    adjusted_intrinsics[1, 2] -= t  # cy

    return image.to_pil(), depthmap, adjusted_intrinsics, normal, far_mask


def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix


def crop_image_depthmap(image, depthmap, camera_intrinsics, crop_bbox, normal=None, far_mask=None):
    """
    Return a crop of the input view.
    """
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    depthmap = depthmap[t:b, l:r]
    if normal is not None:
        normal = normal[t:b, l:r]
    if far_mask is not None:
        far_mask = far_mask[t:b, l:r]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), depthmap, camera_intrinsics, normal, far_mask


def bbox_from_intrinsics_in_out(input_camera_matrix, output_camera_matrix, output_resolution):
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    crop_bbox = (l, t, l + out_width, t + out_height)
    return crop_bbox

def load_co3d_geometric_png(depth_path: str) -> np.ndarray:
    # packed float16 in 16-bit PNG
    with Image.open(depth_path) as depth_pil:
        u16 = np.array(depth_pil, dtype=np.uint16)          # shape (H,W), uint16
    d = u16.view(np.float16).astype(np.float32)             # reinterpret bits -> float16 -> float32
    d[~np.isfinite(d)] = 0.0
    d[d < 0] = 0.0
    return d

@torch.no_grad()
def normalize_cameras(
    c2w: torch.Tensor,         # (B,S,4,4)
    K: torch.Tensor,           # (B,S,3,3) or (S,3,3) or (3,3)
    depths: torch.Tensor,      # (B,S,H,W)
    depth_valid_thresh: float = 1e-6,
    stride: int = 4,           # subsample for speed/memory
    eps: float = 1e-8,
    return_world_points: bool = False,
    alter_depths: bool = True, 
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """
    VGGT-style: compute point map P in world coords, subtract mean (centroid) to remove translation gauge,
    then compute scale = mean ||P|| and normalize camera translations, point map, and depths.

    Returns:
      c2w_norm:   (B,S,4,4)
      K:          unchanged
      depths_norm:(B,S,H,W)
      Pw_norm:    (B,S,H,W,3) if return_world_points else None
      stats: dict with 'centroid' (B,3) and 'scale' (B,)
    """
    assert c2w.ndim == 4 and c2w.shape[-2:] == (4, 4)
    assert depths.ndim == 4
    B, S, H, W = depths.shape
    device = depths.device
    dtype = depths.dtype

    # ---- normalize/expand K to (B,S,3,3) ----
    if K.ndim == 2:
        K = K.view(1, 1, 3, 3).expand(B, S, 3, 3).to(device=device, dtype=dtype)
    elif K.ndim == 3:
        K = K.view(1, S, 3, 3).expand(B, S, 3, 3).to(device=device, dtype=dtype)
    else:
        assert K.shape[:2] == (B, S) and K.shape[-2:] == (3, 3)
        K = K.to(device=device, dtype=dtype)

    c2w = c2w.to(device=device, dtype=dtype)

    # ---- pixel grid (subsampled) ----
    vv = torch.arange(0, H, stride, device=device)
    uu = torch.arange(0, W, stride, device=device)
    v, u = torch.meshgrid(vv, uu, indexing="ij")  # (Hs,Ws)
    Hs, Ws = v.shape
    u = u.reshape(1, 1, Hs, Ws).expand(B, S, Hs, Ws)
    v = v.reshape(1, 1, Hs, Ws).expand(B, S, Hs, Ws)

    z = depths[:, :, ::stride, ::stride]  # (B,S,Hs,Ws)
    valid = z > depth_valid_thresh

    fx = K[..., 0, 0].unsqueeze(-1).unsqueeze(-1)  # (B,S,1,1)
    fy = K[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = K[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = K[..., 1, 2].unsqueeze(-1).unsqueeze(-1)

    # ---- backproject to camera coords ----
    x = (u.to(dtype) - cx) / (fx + eps) * z
    y = (v.to(dtype) - cy) / (fy + eps) * z

    # (B,S,Hs,Ws,3)
    Pc = torch.stack([x, y, z], dim=-1)

    # ---- transform to world coords ----
    R = c2w[:, :, :3, :3]              # (B,S,3,3)
    t = c2w[:, :, :3, 3]               # (B,S,3)

    # Pw = R @ Pc + t  (batched, with Pc last-dim)
    Pw = torch.einsum("bsij,bshwj->bshwi", R, Pc) + t.unsqueeze(2).unsqueeze(2)

    # ---- compute centroid over *all valid points across all views* per sequence ----
    m = valid.to(dtype).unsqueeze(-1)  # (B,S,Hs,Ws,1)
    denom = m.sum(dim=(1, 2, 3), keepdim=False).clamp_min(1.0)  # (B,1) effectively
    centroid = (Pw * m).sum(dim=(1, 2, 3)) / denom              # (B,3)

    # ---- center: subtract centroid from cameras + point map ----
    Pw0 = Pw - centroid.view(B, 1, 1, 1, 3)
    c2w_centered = c2w.clone()
    c2w_centered[:, :, :3, 3] = c2w_centered[:, :, :3, 3] - centroid.view(B, 1, 3)

    # ---- scale: mean Euclidean distance of centered points to origin ----
    r = Pw0.norm(dim=-1)  # (B,S,Hs,Ws)
    denom_r = valid.to(dtype).sum(dim=(1, 2, 3)).clamp_min(1.0)  # (B,)
    scale = (r * valid.to(dtype)).sum(dim=(1, 2, 3)) / denom_r   # (B,)
    scale = scale.clamp_min(1e-6)

    # ---- apply scale to cameras, point map, depths ----
    c2w_norm = c2w_centered.clone()
    c2w_norm[:, :, :3, 3] = c2w_norm[:, :, :3, 3] / scale.view(B, 1, 1)

    if(alter_depths):
        depths_norm = depths / scale.view(B, 1, 1, 1)
    else:
        depths_norm = depths
        
    Pw_norm = None
    if return_world_points:
        # Return at full resolution? This returns the subsampled point map Pw0/scale.
        Pw_norm = Pw0 / scale.view(B, 1, 1, 1, 1)

    stats = {
        "centroid": centroid,  # (B,3)
        "scale": scale,        # (B,)
    }
    return c2w_norm, K, depths_norm, Pw_norm, stats

@torch.no_grad()
def normalize_c2w_by_camera_centers(
    c2w: torch.Tensor,
    scale_mode: str = "median",   # "median", "mean", or "maxabs"
    eps: float = 1e-6,
    return_sim3: bool = False,
):
    """
    c2w: [B,V,4,4] camera-to-world

    Normalizes translations by:
      - centering camera centers by their per-sequence centroid, then
      - scaling by either:
          * "median": median radius ||C - centroid|| over views
          * "mean":   mean radius   ||C - centroid|| over views
          * "maxabs": max absolute coordinate (L_infty) over all views+coords,
                      so that after scaling, all translation coords lie in [-1, 1]

    Returns:
      c2w_norm: [B,V,4,4]
      stats: {"centroid": [B,3], "scale": [B]}
      (optional) sim3: (scale, centroid) where t' = (t - centroid) / scale
    """
    assert c2w.dim() == 4 and c2w.size(-1) == 4 and c2w.size(-2) == 4

    C = c2w[..., :3, 3]                 # [B,V,3] camera centers in world coords
    centroid = C.mean(dim=1)            # [B,3]
    C0 = C - centroid[:, None, :]       # [B,V,3]

    if scale_mode in ("median", "mean"):
        radii = C0.norm(dim=-1)         # [B,V]
        if scale_mode == "median":
            scale = radii.median(dim=1).values
        else:  # "mean"
            scale = radii.mean(dim=1)

    elif scale_mode == "maxabs":
        # L_infty scaling: ensure each coordinate is in [-1,1] after scaling.
        # scale[b] = max_{v,coord} |C0[b,v,coord]|
        scale = C0.abs().amax(dim=(1, 2))  # [B]

    else:
        raise ValueError(f"scale_mode={scale_mode} (expected 'median', 'mean', or 'maxabs')")

    scale = scale.clamp_min(eps)        # [B]

    c2w_out = c2w.clone()
    c2w_out[..., :3, 3] = C0 / scale[:, None, None]

    stats = {"centroid": centroid, "scale": scale, "scale_mode": scale_mode}
    if return_sim3:
        return c2w_out, stats, (scale, centroid)
    return c2w_out, stats

class VisCo3DDataset(Dataset):
    def __init__(
        self,
        categories=("all",),
        split="train",
        transform=None,
        verbose=False,
        min_num_images=50,
        resize_hw=(224, 224),
        eval_time=True,
        normalize_cameras=False,
        first_camera_transform=False,
        mask_images=False,
        co3d_root=None,
        ann_root=None,
        foreground_crop=True,
        center_box=False,
        sort_by_filename=False,
        compute_optical=True,
        full_res=False, 
        return_full_data=False, 
    ):
        category = categories
        if "seen" in category:
            category = TRAINING_CATEGORIES
        if "unseen" in category:
            category = TEST_CATEGORIES
        if "all" in category:
            category = TRAINING_CATEGORIES + TEST_CATEGORIES
        category = sorted(category)
        self.category = category

        if split == "train":
            split_name = "train"
        elif split == "test":
            split_name = "test"
        else:
            raise ValueError(f"Unknown split: {split}")

        self.low_quality_translations = []
        self.rotations = {}
        self.category_map = {}
        self.full_res = full_res
        self.return_full_data = return_full_data

        if co3d_root is None:
            raise ValueError("CO3D_DIR is not specified")
        print(f"CO3D_DIR is {co3d_root}")

        self.CO3D_DIR = co3d_root
        self.CO3D_ANNOTATION_DIR = ann_root
        self.center_box = center_box
        self.split_name = split_name
        self.min_num_images = min_num_images
        self.foreground_crop = foreground_crop

        # --------- load annotations ---------
        for c in category:
            annotation_file = osp.join(self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz")
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())

            for seq_name, seq_data in annotation.items():
                if len(seq_data) < min_num_images:
                    continue

                filtered_data = []
                self.category_map[seq_name] = c
                bad_seq = False
                for data in seq_data:
                    if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
                        bad_seq = True
                        self.low_quality_translations.append(seq_name)
                        break

                    filtered_data.append(
                        {
                            "filepath": data["filepath"],
                            "bbox": data["bbox"],
                            "R": data["R"],
                            "T": data["T"],
                            "focal_length": data["focal_length"],
                            "principal_point": data["principal_point"],
                        }
                    )

                if not bad_seq:
                    self.rotations[seq_name] = filtered_data

        self.sequence_list = list(self.rotations.keys())
        img_size, _ = resize_hw

        self.split = split
        self.debug = verbose
        self.sort_by_filename = sort_by_filename

        if transform is None:
            self.transform = transforms.ToTensor()

        self.img_size = img_size
        self.eval_time = eval_time
        self.normalize_cameras = normalize_cameras
        self.first_camera_transform = first_camera_transform
        self.mask_images = mask_images
        self.compute_optical = compute_optical
        self.aug_crop = False
        self.aug_focal = False

        print(f"Low quality translation sequences, not used: {self.low_quality_translations}")
        print(f"Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def _crop_image(self, image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new("RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255))
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image, top=bbox[1], left=bbox[0], height=bbox[3] - bbox[1], width=bbox[2] - bbox[0]
            )
        return image_crop

    def __getitem__(self, idx_N):
        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        ids = torch.randperm(len(metadata))[:n_per_seq].tolist()
        return self.get_data(index=index, ids=ids)

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, normal=None, far_mask=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics, normal, far_mask = crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        # NOTE: Here we don't care about portrait image.
        # assert resolution[0] >= resolution[1]
        # if H > 1.1*W:
        #     # image is portrait mode
        #     resolution = resolution[::-1]
        # elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
        #     # image is square, so we chose (portrait, landscape) randomly
        #     if rng.integers(2):
        #         resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
            image, depthmap, intrinsics, normal, far_mask = center_crop_image_depthmap(image, depthmap, intrinsics, crop_scale, normal=normal, far_mask=far_mask)

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, intrinsics, normal, far_mask = rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, normal=normal, far_mask=far_mask) # slightly scale the image a bit larger than the target resolution

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2, normal, far_mask = crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal, far_mask=far_mask)

        other = [x for x in [normal, far_mask] if x is not None]
        return image, depthmap, intrinsics2, *other

    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False, return_path=False):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        category = self.category_map[sequence_name]

        annos = [metadata[i] for i in ids]
        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["filepath"])

        images_transformed = []
        depths_transformed = []
        Ks = []
        camera_poses = []
        image_paths = []

        if(self.return_full_data): 
            images_full = []
            depths_full = []
            Ks_full = []

        for anno in annos:
            filepath = anno["filepath"]
            image_path = osp.join(self.CO3D_DIR, filepath)
            image_paths.append(image_path)

            # --- load RGB ---
            image = Image.open(image_path).convert("RGB")
            w_full, h_full = image.width, image.height
            image_size_hw = np.array([h_full, w_full], dtype=np.float32)

            # --- load depth ---
            depth_path = osp.join(self.CO3D_DIR, filepath.replace("images", "depths") + ".geometric.png")
            if osp.exists(depth_path):
                depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_u16 is None:
                    depth_np = np.zeros((h_full, w_full), dtype=np.float32)
                else:
                    if depth_u16.ndim == 3:
                        depth_u16 = depth_u16[..., 0]
                    depth_np = load_co3d_geometric_png(depth_path) if depth_u16.dtype == np.uint16 else depth_u16.astype(np.float32)
            else:
                depth_np = np.zeros((h_full, w_full), dtype=np.float32)

            # --- optional mask application (only if you actually want masked depth) ---
            if self.mask_images:  # replace True with self.mask_depths if you add that flag
                mask_name = osp.basename(filepath.replace(".jpg", ".png"))
                mask_path = osp.join(self.CO3D_DIR, category, sequence_name, "masks", mask_name)
                if osp.exists(mask_path):
                    mask = Image.open(mask_path).convert("L")
                    mask_np = (np.array(mask, dtype=np.float32) / 255.0) > 0.1
                    depth_np = depth_np * mask_np
                    if self.mask_images:
                        white_image = Image.new("RGB", image.size, (255, 255, 255))
                        mask_bin_u8 = (mask_np.astype("uint8") * 255)
                        image = Image.composite(image, white_image, Image.fromarray(mask_bin_u8))
                else:
                    # no mask file: leave depth as-is; optionally whiten nothing
                    pass

            # --- build full K and c2w pose ---
            R_np      = np.array(anno["R"], dtype=np.float32)
            T_np      = np.array(anno["T"], dtype=np.float32)
            focal_ndc = np.array(anno["focal_length"], dtype=np.float32)
            pp_ndc    = np.array(anno["principal_point"], dtype=np.float32)

            R_cv, tvec_cv, K_full = opencv_from_cameras_projection(R_np, T_np, focal_ndc, pp_ndc, image_size_hw)

            cam_to_world = np.eye(4, dtype=np.float32)
            cam_to_world[:3, :3] = R_cv
            cam_to_world[:3, 3]  = tvec_cv
            cam_to_world = np.linalg.inv(cam_to_world)
            camera_poses.append(torch.from_numpy(cam_to_world))

            if self.return_full_data:
                images_full.append(self.transform(image))
                depths_full.append(torch.from_numpy(depth_np.astype(np.float32)))
                Ks_full.append(torch.from_numpy(K_full.astype(np.float32)))

            # --- crop/resize + updated K ---
            rgb_pil, depth_resized_np, K_resized = self._crop_resize_if_necessary(
                image,
                depth_np,
                K_full.copy(),
                resolution=(self.img_size, self.img_size),
                rng=np.random,
                info=image_path,
            )[:3]

            images_transformed.append(self.transform(rgb_pil))
            depths_transformed.append(torch.from_numpy(depth_resized_np.astype(np.float32)))
            Ks.append(torch.from_numpy(K_resized.astype(np.float32)))

        batch = {
            "seq_id": sequence_name,
            "category": category,
            "n": len(metadata),
            "ind": torch.tensor(ids, dtype=torch.long),
            "imgs": torch.stack(images_transformed),
            "depths": torch.stack(depths_transformed),
            "K": torch.stack(Ks),
            "camera_pose": torch.stack(camera_poses),
        }

        if self.return_full_data:
            batch["full"] = {
                "imgs": torch.stack(images_full),
                "depths": torch.stack(depths_full),
                "K": torch.stack(Ks_full),
            }

        if return_path:
            return batch, image_paths
        return batch

def center_crop_image(image, camera_intrinsics, crop_scale, normal=None, far_mask=None):
    """
    Jointly center-crop an image, and adjust the camera intrinsics accordingly.

    Parameters:
    - image: PIL.Image or similar, the input image.
    - depthmap: np.ndarray, the corresponding depth map.
    - camera_intrinsics: np.ndarray, the 3x3 camera intrinsics matrix.
    - crop_scale: float between 0 and 1, the fraction of the image to keep.

    Returns:
    - cropped_image: PIL.Image, the center-cropped image.
    - cropped_depthmap: np.ndarray, the center-cropped depth map.
    - adjusted_intrinsics: np.ndarray, the adjusted camera intrinsics matrix.
    """
    # Ensure crop_scale is valid
    assert 0 < crop_scale <= 1, "crop_scale must be between 0 and 1"

    # Convert image to ImageList for consistent processing
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (width, height)

    # Compute output resolution after cropping
    output_resolution = np.floor(input_resolution * crop_scale).astype(int)
    # get the correct crop_scale
    crop_scale = output_resolution / input_resolution

    # Compute margins (amount to crop from each side)
    margins = input_resolution - output_resolution
    offset = margins / 2  # Since we are center cropping

    # Calculate the crop bounding box
    l, t = offset.astype(int)
    r = l + output_resolution[0]
    b = t + output_resolution[1]
    crop_bbox = (l, t, r, b)

    # Crop the image and depthmap
    image = image.crop(crop_bbox)
    if normal is not None:
        normal = normal[t:b, l:r]
    if far_mask is not None:
        far_mask = far_mask[t:b, l:r]

    # Adjust the camera intrinsics
    adjusted_intrinsics = camera_intrinsics.copy()

    # Adjust focal lengths (fx, fy)                         
    # adjusted_intrinsics[0, 0] /= crop_scale[0]  # fx
    # adjusted_intrinsics[1, 1] /= crop_scale[1]  # fy

    # Adjust principal point (cx, cy)
    adjusted_intrinsics[0, 2] -= l  # cx
    adjusted_intrinsics[1, 2] -= t  # cy

    return image.to_pil(), adjusted_intrinsics, normal, far_mask

def rescale_image(image, camera_intrinsics, output_resolution, force=True, normal=None, far_mask=None):
    """ rescale image
        so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        return (image.to_pil(), camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic)

    if normal is not None:
        normal = cv2.resize(normal, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)
    if far_mask is not None:
        far_mask = cv2.resize(far_mask, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), camera_intrinsics, normal, far_mask


def crop_image(image, camera_intrinsics, crop_bbox, normal=None, far_mask=None):
    """
    Return a crop of the input view.
    """
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    if normal is not None:
        normal = normal[t:b, l:r]
    if far_mask is not None:
        far_mask = far_mask[t:b, l:r]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), camera_intrinsics, normal, far_mask

class PoseCo3DDataset(Dataset):
    def __init__(
        self,
        categories=("all",),
        split="train",
        transform=None,
        verbose=False,
        random_aug=False,
        jitter_scale=[0.8, 1.2],
        jitter_trans=[-0.07, 0.07],
        min_num_images=50,
        resize_hw=(224, 224),
        eval_time=True,
        normalize_cameras=False,
        first_camera_transform=False,
        mask_images=False,
        co3d_root=None,
        ann_root=None,
        foreground_crop=True,
        center_box=False,
        sort_by_filename=False,
        compute_optical=True,
    ):
        category = categories
        if "seen" in category:
            category = TRAINING_CATEGORIES
        if "unseen" in category:
            category = TEST_CATEGORIES
        if "all" in category:
            category = TRAINING_CATEGORIES + TEST_CATEGORIES
        category = sorted(category)
        self.category = category

        if split == "train":
            split_name = "train"
        elif split == "test":
            split_name = "test"
        else:
            raise ValueError(f"Unknown split: {split}")

        self.low_quality_translations = []
        self.rotations = {}
        self.category_map = {}

        if co3d_root is None:
            raise ValueError("CO3D_DIR is not specified")
        print(f"CO3D_DIR is {co3d_root}")

        self.CO3D_DIR = co3d_root
        self.CO3D_ANNOTATION_DIR = ann_root
        self.center_box = center_box
        self.split_name = split_name
        self.min_num_images = min_num_images
        self.foreground_crop = foreground_crop

        # --------- load annotations ---------
        for c in category:
            annotation_file = osp.join(self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz")
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())

            for seq_name, seq_data in annotation.items():
                if len(seq_data) < min_num_images:
                    continue

                filtered_data = []
                self.category_map[seq_name] = c
                bad_seq = False
                for data in seq_data:
                    if np.linalg.norm(data["T"]) > 1e5:
                        bad_seq = True
                        self.low_quality_translations.append(seq_name)
                        break

                    filtered_data.append(
                        {
                            "filepath": data["filepath"],
                            "bbox": data["bbox"],
                            "R": data["R"],
                            "T": data["T"],
                            "focal_length": data["focal_length"],
                            "principal_point": data["principal_point"],
                        }
                    )

                if not bad_seq:
                    self.rotations[seq_name] = filtered_data

        self.sequence_list = list(self.rotations.keys())
        img_size, _ = resize_hw

        self.split = split
        self.debug = verbose
        self.sort_by_filename = sort_by_filename

        self.transform = transforms.ToTensor()

        if random_aug and not eval_time:
            self.jitter_scale = jitter_scale
            self.jitter_trans = jitter_trans
        else:
            self.jitter_scale = [1, 1]
            self.jitter_trans = [0, 0]

        self.img_size = img_size
        self.eval_time = eval_time
        self.normalize_cameras = normalize_cameras
        self.first_camera_transform = first_camera_transform
        self.mask_images = mask_images
        self.compute_optical = compute_optical
        self.aug_crop = False
        self.aug_focal = False


        print(f"Low quality translation sequences, not used: {self.low_quality_translations}")
        print(f"Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def _jitter_bbox(self, bbox):
        bbox = square_bbox(bbox.astype(np.float32))
        s = np.random.uniform(self.jitter_scale[0], self.jitter_scale[1])
        tx, ty = np.random.uniform(self.jitter_trans[0], self.jitter_trans[1], size=2)

        side_length = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
        extent = side_length / 2 * s

        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

    def _crop_image(self, image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new("RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255))
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image, top=bbox[1], left=bbox[0], height=bbox[3] - bbox[1], width=bbox[2] - bbox[0]
            )
        return image_crop

    def __getitem__(self, idx_N):
        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        ids = torch.randperm(len(metadata))[:n_per_seq].tolist()
        return self.get_data(index=index, ids=ids)

    def _crop_resize_if_necessary(self, image, intrinsics, resolution, rng=None, info=None, normal=None, far_mask=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, intrinsics, normal, far_mask = crop_image(image, intrinsics, crop_bbox, normal=normal)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        # NOTE: Here we don't care about portrait image.
        # assert resolution[0] >= resolution[1]
        # if H > 1.1*W:
        #     # image is portrait mode
        #     resolution = resolution[::-1]
        # elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
        #     # image is square, so we chose (portrait, landscape) randomly
        #     if rng.integers(2):
        #         resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
            image, intrinsics, normal, far_mask = center_crop_image(image, intrinsics, crop_scale, normal=normal, far_mask=far_mask)

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, intrinsics, normal, far_mask = rescale_image(image, intrinsics, target_resolution, normal=normal, far_mask=far_mask) # slightly scale the image a bit larger than the target resolution

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, intrinsics2, normal, far_mask = crop_image(image, intrinsics, crop_bbox, normal=normal, far_mask=far_mask)

        other = [x for x in [normal, far_mask] if x is not None]
        return image, intrinsics2, *other

    def get_data(self, index=None, sequence_name=None, ids=(0, 1), no_images=False, return_path=False):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        category = self.category_map[sequence_name]

        annos = [metadata[i] for i in ids]
        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["filepath"])

        images_transformed = []
        Ks = []
        camera_poses = []
        fl_px_list = []
        pp_px_list = []
        image_paths = []

        R_original_list = []
        T_original_list = []

        for anno in annos:
            filepath = anno["filepath"]
            image_path = osp.join(self.CO3D_DIR, filepath)
            image_paths.append(image_path)

            # ---- load RGB + mask (same as original first loop) ----
            image = Image.open(image_path).convert("RGB")

            w_full, h_full = image.width, image.height
            image_size_hw = np.array([h_full, w_full], dtype=np.float32)

            R_np      = np.array(anno["R"], dtype=np.float32)
            T_np      = np.array(anno["T"], dtype=np.float32)
            focal_ndc = np.array(anno["focal_length"], dtype=np.float32)
            pp_ndc    = np.array(anno["principal_point"], dtype=np.float32)

            R_cv, tvec_cv, K_full = opencv_from_cameras_projection(
                R_np, T_np, focal_ndc, pp_ndc, image_size_hw
            )

            cam_to_world = np.eye(4, dtype=np.float32)
            cam_to_world[:3, :3] = R_cv
            cam_to_world[:3, 3]  = tvec_cv
            cam_to_world = np.linalg.inv(cam_to_world)
            camera_poses.append(torch.from_numpy(cam_to_world))

            out = self._crop_resize_if_necessary(
                image,
                K_full.copy(),
                resolution=(self.img_size, self.img_size),
                rng=np.random,
                info=image_path,
            )
            rgb_pil, K_resized = out[:2]

            images_transformed.append(self.transform(rgb_pil))
            Ks.append(torch.from_numpy(K_resized.astype(np.float32)))

            fx = K_resized[0, 0]
            fy = K_resized[1, 1]
            cx = K_resized[0, 2]
            cy = K_resized[1, 2]
            fl_px_list.append(torch.tensor([fx, fy], dtype=torch.float32))
            pp_px_list.append(torch.tensor([cx, cy], dtype=torch.float32))

        batch = {
            "seq_id": sequence_name,
            "category": category,
            "n": len(metadata),
            "ind": torch.tensor(ids, dtype=torch.long),
        }

        images_tensor = torch.stack(images_transformed) if self.transform is not None else images_transformed

        Ks_tensor     = torch.stack(Ks)            # (N, 3, 3)
        poses_tensor  = torch.stack(camera_poses)  # (N, 4, 4)
        fl_px         = torch.stack(fl_px_list)    # (N, 2)
        pp_px         = torch.stack(pp_px_list)    # (N, 2)

        batch["imgs"] = images_tensor

        # CO3DV2-like camera info
        batch["R"] = poses_tensor[:, :3, :3]
        batch["T"] = poses_tensor[:, :3, 3]
        batch["camera_pose"] = poses_tensor
        batch["K"] = Ks_tensor
        batch["fl"] = fl_px
        batch["pp"] = pp_px

        if return_path:
            return batch, image_paths
        return batch
