import argparse
import sys
from pathlib import Path

sys.path.append("/home/mgjacob/scripts/se3n")

from visualizations import visualize_cameras, visualize_cameras_sfm
from SE3NDiffusion import se3n_diffuser
from datasets import RGBDepthDataset, RGBFeatureDataset

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch
import random
from torch.utils.data import Subset

# Dataset imports


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True,
                    help="Run name (used to name output directory)")
parser.add_argument("--conditioning", type=str, required=True,
                    choices=["depths", "features"],
                    help="Conditioning type: depths | features")
parser.add_argument("--dataset-dir", type=Path, required=True,
                    help="Path to directory containing .pt or .npz files")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--val-dataset-dir", type=Path, required=False,
                    help="Optional path to validation dataset")
parser.add_argument("--samples", type=int, default=5)
parser.add_argument("--load", type = bool, default = False)
parser.add_argument("--plot-name", type=str, default="default",
                    help="Suffix used to name saved plots to avoid overwriting.")
parser.add_argument(
    "--max-samples", type=int, default=None,
    help="If set, randomly draw only this many examples from the dataset"
)
args = parser.parse_args()

run_name    = args.name
conditioning = args.conditioning
data_dir    = args.dataset_dir
val_data_dir = args.val_dataset_dir
load = args.load
num_samples = args.samples
output_dir  = Path(f"/home/mgjacob/results/{run_name}")
output_dir.mkdir(parents=True, exist_ok=True)
num_workers = 8 

print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Script is running with:")
print(f"  Conditioning: {conditioning}")
print(f"  Dataset:      {data_dir}")


if conditioning == "depths":
    dataset = RGBDepthDataset(data_dir)
    val_dataset = RGBDepthDataset(val_data_dir) if val_data_dir else None
elif conditioning == "features":
    dataset = RGBFeatureDataset(data_dir, device="cuda", invert = False)
    val_dataset = RGBFeatureDataset(val_data_dir, device="cuda", invert = False) if val_data_dir else None
    

if args.max_samples is not None:
    if args.max_samples > len(dataset):
        raise ValueError(f"--max-samples ({args.max_samples}) "
                         f"exceeds dataset size ({len(dataset)})")
    subset_idx = random.sample(range(len(dataset)), args.max_samples)
    dataset = Subset(dataset, subset_idx)
    print(f"⚠️  Using only {len(dataset)} samples out of the full set")

train_dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,       # Total batch size (split across GPUs)
                        shuffle=True,
                        num_workers=num_workers,     # e.g., 4–8 usually works well
                        pin_memory=True, 
                        drop_last=True)

val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False) if val_dataset else None


print("Loading older model: ", load)
if not load: 
    model_path = None
else: 
    model_path = output_dir / "model.pt"

print("model path: ", model_path)

se3n = se3n_diffuser(
    T=100,
    batch_size=args.batch_size,
    device="cuda",
    conditioning=conditioning,
    save_model=True,
    save_path=output_dir / "model.pt",
    model_path=model_path, 
    dataset_root=data_dir,  
    dataset=dataset, 
    num_workers = num_workers
)


visualize_cameras_sfm(se3n, train_dataloader, k = num_samples, conditioning = conditioning, plot_name=output_dir / f"poses_plot_{args.plot_name}")
