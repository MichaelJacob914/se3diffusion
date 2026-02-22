import argparse, os, sys, json, time
import yaml
import sys
from pathlib import Path
from visualizations import plot_so3_probabilities_full, validation_statistics
from SE3NDiffusion import se3n_diffuser
from SE3NRegression import se3n_regressor
from se3n_datasets import PoseCo3DDataset, DynamicBatchSampler, VisCo3DDataset
from feature_extractors import Dust3rFeatureExtractor, compute_dust3r_feats, compute_pi3_feats, Pi3FeatureExtractor
import os, socket, pathlib, stat
import json, argparse
import logging
from accelerate import Accelerator
from torch.utils.data import RandomSampler
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, DataLoader
import wandb

CATEGORIES = [
    "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
    "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave", "motorcycle",
    "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
    "skateboard", "stopsign", "suitcase", "teddybear", "toaster", "toilet", "toybus",
    "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
]


logger = logging.getLogger(__name__)
base = Path(__file__).resolve().parents[1]  # .../PARCC/scripts
sys.path[:0] = [str(base / "dust3r" / "croco"), str(base / "dust3r")]
from dust3r.model        import load_model
from dust3r.utils.image  import load_images
from dust3r.image_pairs  import make_pairs
from dust3r.inference    import inference
import matplotlib.pyplot as plt
import torch
import random
from torch.utils.data import Subset

import os, torch
import torch.distributed as dist
import numpy as np 

def init_dist_and_device():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, device, local_rank
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, device, 0


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--load", required=False)
parser.add_argument("--samples", required=False, default = 5)
parser.add_argument("--epochs", type = int, required=False, default = -1)
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)
    
run_name     = cfg["name"]
conditioning = cfg["conditioning"]
co3d_root     = cfg["dataset_dir"]
ann_root     = cfg["ann_root"]
model_type   = cfg["model_type"]
max_samples  = cfg["max_samples"] #MAX NUMBER OF SAMPLES USED IN THE DATASET, DEBUGGING ONLY
attn_kwargs  = cfg["attn_kwargs"]
attn_args    = cfg["attn_args"]
if(args.epochs >= 0): 
    epochs = args.epochs
else: 
    epochs       = cfg["epochs"]
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
MICRO_BATCH  = BATCH_GLOBAL

load = args.load
num_samples = args.samples
output_dir  = Path(f"/vast/projects/kostas/geometric-learning/mgjacob/{run_name}")
output_dir.mkdir(parents=True, exist_ok=True)

wandb_root = output_dir / "wandb"
wandb_root.mkdir(parents=True, exist_ok=True)

os.environ["WANDB_DIR"] = str(wandb_root / "runs")            # run logs/metadata
os.environ["WANDB_CACHE_DIR"] = str(wandb_root / "cache")     # artifact cache (the big one)
os.environ["WANDB_ARTIFACT_DIR"] = str(wandb_root / "artifacts")  # downloaded artifacts
os.environ["WANDB_START_METHOD"] = "thread"   # avoids some multiprocessing edge cases
num_workers = 4


is_dist, device, local_rank = init_dist_and_device()
world_size = dist.get_world_size() if is_dist else 1

accelerator = Accelerator(mixed_precision="bf16")

print("[INFO] Torch version:", torch.__version__)
print("[INFO] CUDA version:", torch.version.cuda)
print("[INFO] Script is running with:")
print(f"[INFO] Conditioning: {conditioning}")
print(f"[INFO] Features: {feature_type}")
print(f"[INFO]  Dataset:      {co3d_root}")


if conditioning == "depths":
    dataset = RGBDepthDataset(data_dir)
    val_dataset = RGBDepthDataset(val_data_dir) if val_data_dir else None
elif conditioning == "features":
    dataset = RGBFeatureDataset(data_dir, device="cuda", invert = False)
    val_dataset = RGBFeatureDataset(val_data_dir, device="cuda", invert = False) if val_data_dir else None
    train_dataloader = DataLoader(dataset,
                                    batch_size=BATCH_GLOBAL,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last = True)
elif conditioning == "CO3D":
    if(feature_type == "dust3r"):
        extractor = Dust3rFeatureExtractor(
            ckpt_path=Path("/vast/projects/kostas/geometric-learning/mgjacob/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"),
            device=str(device)  
        )
    elif(feature_type == "pi3"):
        extractor = Pi3FeatureExtractor(
            ckpt_path=Path("/vast/projects/kostas/geometric-learning/pi3_weights/model.safetensors"),
            device=str(device)  
        )

    train_dataset = PoseCo3DDataset(
        co3d_root=co3d_root,
        categories=("backpack",), 
        resize_hw=(224, 224),
        verbose=True, 
        ann_root=ann_root,           
        split="train",               
    )
    print("[INFO] TRAINING DATASET LENGTH=", len(train_dataset))  # must be > 0
    val_dataset = PoseCo3DDataset(
        co3d_root=co3d_root,
        categories=("backpack",),
        resize_hw=(224, 224),
        verbose=True, 
        ann_root=ann_root,           
        split="test",               
    )
    print("[INFO] VAL DATASET LENGTH =", len(val_dataset))  # must be > 0

    common = sorted(set(train_dataset.sequence_list) & set(val_dataset.sequence_list))
    if(num_sequences is not None): 
        keep = common[:num_sequences]
        if len(keep) < num_sequences:
            print(f"[warn] only {len(keep)} common sequences found; using: {keep}")

        def prune_to(ds, names):
            ds.sequence_list = names
            ds.rotations    = {s: ds.rotations[s] for s in names}
            ds.category_map = {s: ds.category_map[s] for s in names}

        prune_to(train_dataset, keep)
        prune_to(val_dataset,   keep)
        print("[INFO] NEW TRAINING  DATASET LENGTH=", len(train_dataset))  # must be > 0
        print("[INFO] NEW VALIDATION DATASET LENGTH =", len(val_dataset))  # must be > 0
    
    if max_samples is not None:
        if max_samples > len(dataset):
            raise ValueError(f"--max-samples ({args.max_samples}) exceeds dataset size ({len(dataset)})")
        subset_idx = random.sample(range(len(dataset)), args.max_samples)
        dataset = Subset(dataset, subset_idx)
        print(f"⚠️ Using only {len(dataset)} samples out of the full set")

    sampler = DynamicBatchSampler(len(train_dataset), dataset_len = BATCH_GLOBAL, max_images = 6, images_per_seq = [5,6])
    val_sampler = DynamicBatchSampler(len(val_dataset), dataset_len = BATCH_GLOBAL, max_images = 6, images_per_seq = [5, 6])
    train_dataloader = DataLoader(
        train_dataset,   
        batch_sampler=sampler,              
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,    
        batch_sampler=val_sampler,                
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

print("[INFO] Loading older model: ", load)
if not load: 
    model_path = None
else: 
    model_path = output_dir / "model.pt"

train_dataloader, val_dataloader = accelerator.prepare(train_dataloader, val_dataloader)
print("[INFO] model path: ", model_path)
se3_config = {
    "T": 100,
    "device": device,
    "is_dist": is_dist,
    "local_rank": local_rank,
    "world_size": world_size,
    "conditioning": conditioning,
    "dataloader": train_dataloader,
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
    "dataset_root": co3d_root,
    "dataset": train_dataset,
    "num_workers": num_workers,
    "forward_process": "ve", 
    "representation": "rot9d", 
    "scheme": scheme,
    "accelerator": accelerator, 
    "update_type": update_type,
    "guidance_type": "ggs"
}

if(prediction_type == "regressor"):
    print("[INFO] Using SE(3) Regressor")
    se3n = se3n_regressor(se3_config)
else:
    se3n = se3n_diffuser(se3_config)


#PRINT FINAL LOSSES 
losses = se3n.val_co3d(train_dataloader, num_epochs=0)
losses = se3n.val_co3d(val_dataloader, num_epochs=0)
plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"SE(3) Diffusion Validation Loss ({run_name})")
plt.grid(True)
plt.savefig(output_dir / f"validation_loss_curve_{plot_name}.png")
plt.close()

#out = se3n.eval_loss_by_timestep(train_dataloader, t_values, max_batches=50)
#print(out)
#out = se3n.eval_loss_by_timestep(val_dataloader, t_values, max_batches=50)
#print(out)
dictionary = validation_statistics(se3n, train_dataloader, k = 1)
metrics = dictionary["pooled"]

print("\n=== TRAINING SET===")
for k, v in metrics.items():
        print(f"{k:8s}: {v:6.2f}")

"""
print("\n=== VALIDATION SET===")
dictionary = validation_statistics(se3n, val_dataloader, k = 1)
metrics = dictionary["pooled"]
for k, v in metrics.items():
        print(f"{k:8s}: {v:6.2f}")
"""