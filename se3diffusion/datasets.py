#!/usr/bin/env python
# coding: utf-8

# In[8]:

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
from typing import Union


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
    Returns:
        img1, img2 : (3, H, W)  RGB images
        rgb        : (6, H, W)  stacked RGB channels
        feats      : (2, 1024, 32, 32)
        R          : (3, 3)  relative rotation
        t          : (3,)    relative translation
    """
    def __init__(self,
                 root: Union[str, Path],
                 device: str = "cuda"):
        self.root   = Path(root)
        self.files  = sorted(self.root.glob("*.pt"))
        self.device = torch.device(device)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = torch.load(self.files[idx], map_location="cpu")

        # ---- RGB ----
        img1 = d["img1"]  # (3, H, W)
        img2 = d["img2"]
        rgb  = torch.cat([img1, img2], dim=0)

        f1 = d["feat1"].to(torch.float32)
        f2 = d["feat2"].to(torch.float32)
        if f1.shape[-2] != 32:
            f1 = f1.transpose(-2, -1)
            f2 = f2.transpose(-2, -1)
        feats = torch.stack((f1, f2), dim=0)  # (2, 1024, 32, 32)
        

        R1 = d["R1"]
        t1 = d["t1"]
        R2 = d["R2"]
        t2 = d["t2"]
        
        R12 = R1.T @ R2                           
        t12 = t2 - t1 @ R1.T @ R2      

        return {
            "img1":  img1,        # (3, H, W)
            "img2":  img2,
            "rgb":   rgb,         # (6, H, W)
            "feats": feats,       # (2, 1024, 32, 32)
            "R":     R12,
            "t":     t12,
        }
