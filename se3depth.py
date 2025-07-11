import argparse
from SE3COLMAP import PairDataset, se3_diffuser, visualize_pose_axes
from torch.utils.data import Subset, DataLoader
from pathlib import Path
import os
import random
import matplotlib.pyplot as plt

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)  
parser.add_argument("--toy_size", type=int, default=8)
args = parser.parse_args()

toy_size = args.toy_size
run_name = args.name
output_dir = Path(f"/home/mgjacob/results/{run_name}")
output_dir.mkdir(parents=True, exist_ok=True)


print(torch.__version__)
print(torch.version.cuda)
print("Script is running")
out_dir = Path("/home/mgjacob/data/co3d_npz_depths")
dataset = PairDataset(out_dir)
batch_size = 64

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

toy_size = args.toy_size
toy_batch  = 16                    # load all 8 every step
toy_ds     = Subset(dataset, list(range(toy_size)))

print("Toy-dataset poses (R, t):")
for local_idx in range(toy_size):
    sample = toy_ds[local_idx]                 # same as dataset[local_idx]
    R_i, t_i = sample["R"], sample["t"]
    print(f"\nsample {local_idx}:")
    print("R =")
    print(R_i)                                # (3×3)
    print("t =", t_i)                         # (3,)
print("-" * 40)

toy_loader = DataLoader(toy_ds,
                        batch_size=toy_batch,
                        shuffle=True,   # fine—still jitters grads
                        num_workers=0)


se3 = se3_diffuser(T=100, batch_size=batch_size, device = 'cuda', save_model=True, save_path=output_dir / "model.pt", model_path = output_dir / "model.pt")

losses = se3.train(dataloader= dataloader, num_epochs=150)

plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"SE(3) Diffusion Training Loss ({run_name})")
plt.grid(True)
plt.savefig(output_dir / "loss_curve.png")
plt.close()

visualize_pose_axes(se3, dataloader, k=4, plot_name=output_dir)
