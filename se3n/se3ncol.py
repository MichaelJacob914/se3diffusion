import argparse
import sys
from pathlib import Path
from visualizations import visualize_pose_axes_full, plot_so3_probabilities_full
from SE3NDiffusion import se3n_diffuser
from datasets import RGBDepthDataset, RGBFeatureDataset, Dust3rFeatureExtractor, Co3dDataset, compute_dust3r_feats

from torch.utils.data import DataLoader

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

# Dataset imports


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True,
                    help="Run name (used to name output directory)")
parser.add_argument("--conditioning", type=str, required=True,
                    choices=["depths", "features" ,"CO3D"],
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
output_dir  = Path(f"/vast/home/m/mgjacob/PARCC/results/{run_name}")
output_dir.mkdir(parents=True, exist_ok=True)
num_workers = 8 

print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Script is running with:")
print(f"  Conditioning: {conditioning}")
print(f"  Dataset:      {data_dir}")

is_dist, device, local_rank = init_dist_and_device()
world_size = dist.get_world_size() if is_dist else 1

from torch.utils.data import Sampler, RandomSampler

class DistributedReplacementSampler(Sampler[int]):
    def __init__(self, dataset, num_replicas, rank, num_samples, seed=0):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0
        assert len(dataset) > 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        total = self.num_samples * self.num_replicas
        idx = torch.randint(high=len(self.dataset), size=(total,), generator=g).tolist()
        idx = idx[self.rank:total:self.num_replicas]  # shard
        return iter(idx)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch

BATCH_GLOBAL = 4
MICRO_BATCH  = min(args.batch_size, 2)          # per-GPU; tune to fit VRAM
ACCUM_STEPS  = max(1, BATCH_GLOBAL // (world_size * MICRO_BATCH))
assert world_size * MICRO_BATCH * ACCUM_STEPS == BATCH_GLOBAL, "Adjust MICRO/ACCUM to hit 64"
STEPS_PER_EPOCH  = 10                         # how long an epoch is; tune
SAMPLES_PER_RANK = STEPS_PER_EPOCH * 20


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
    # 1) Build extractor + dataset first
    co3d_root = Path("/vast/projects/kostas/geometric-learning/Co3D/Co3Dmini")
    extractor = Dust3rFeatureExtractor(
        ckpt_path=Path("/vast/home/m/mgjacob/PARCC/scripts/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"),
        device=str(device)  # use the rank's device if DDP
    )
    dataset = Co3dDataset(
        co3d_root=co3d_root,
        categories=["chair", "couch", "car"],
        extractor=extractor,
        k=2,
        resize_hw=(224, 224),
        box_crop_context=0.1,
        device=str(device),
        verbose = False
    )
    val_dataset = None   # or make a second Co3dDataset if you want a real val set

    # 2) Replacement sampling to hit a fixed global batch
    STEPS_PER_EPOCH  = 10
    SAMPLES_PER_RANK = STEPS_PER_EPOCH * 2

    if is_dist:
        class DistributedReplacementSampler(Sampler[int]):
            def __init__(self, dataset, num_replicas, rank, num_samples, seed=0):
                self.dataset = dataset
                self.num_replicas = num_replicas
                self.rank = rank
                self.num_samples = num_samples
                self.seed = seed
                self.epoch = 0
                assert len(dataset) > 0
            def __iter__(self):
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                total = self.num_samples * self.num_replicas
                idx = torch.randint(high=len(self.dataset), size=(total,), generator=g).tolist()
                idx = idx[self.rank:total:self.num_replicas]
                return iter(idx)
            def __len__(self): return self.num_samples
            def set_epoch(self, epoch: int): self.epoch = epoch

        sampler = DistributedReplacementSampler(dataset, world_size, local_rank,
                                                SAMPLES_PER_RANK, seed=0)
    else:
        from torch.utils.data import RandomSampler
        sampler = RandomSampler(dataset, replacement=True, num_samples=SAMPLES_PER_RANK)

    train_dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH,       # per-rank micro-batch
        sampler=sampler,              # replacement sampler
        shuffle=False,                # sampler controls order
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) if val_dataset else None


if args.max_samples is not None:
    if args.max_samples > len(dataset):
        raise ValueError(f"--max-samples ({args.max_samples}) "
                         f"exceeds dataset size ({len(dataset)})")
    subset_idx = random.sample(range(len(dataset)), args.max_samples)
    dataset = Subset(dataset, subset_idx)
    print(f"⚠️  Using only {len(dataset)} samples out of the full set")


print("Loading older model: ", load)
if not load: 
    model_path = None
else: 
    model_path = output_dir / "model.pt"

print("model path: ", model_path)

prediction = {"r": "pose", "t": "pose"}
so3_config = {
    "schedule": "logarithmic",
    "min_sigma": 0.1,
    "max_sigma": 1.5,
    "num_sigma": 1000,
    "use_cached_score": False,
    "num_omega": 2000,
    "cache_dir": "./igso3_cache"
}
r3_config = {
    "min_b": 0.1,
    "max_b": 20.0,
    "coordinate_scaling": 1
}


se3n = se3n_diffuser(
    T=100,
    batch_size=MICRO_BATCH,   
    device=device,       
    is_dist=is_dist,
    local_rank=local_rank,
    world_size=world_size,
    accum_steps=ACCUM_STEPS,    
    conditioning=conditioning,
    dataloader=train_dataloader,
    save_model=True,
    model_type = "Transformer", 
    prediction = prediction, 
    so3_config = so3_config, 
    r3_config = r3_config, 
    save_path=output_dir / "model.pt",
    model_path=model_path, 
    dataset_root=data_dir,  
    dataset=dataset, 
    num_workers = num_workers
)

losses = se3n.train(num_epochs=args.epochs)

plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"SE(3) Diffusion Training Loss ({run_name})")
plt.grid(True)
plt.savefig(output_dir / f"loss_curve_{args.plot_name}.png")
plt.close()


visualize_pose_axes_full(se3n, train_dataloader, k = num_samples, conditioning = conditioning, plot_name=output_dir / f"alt_poses_plot_{args.plot_name}.png")
#plot_so3_probabilities_full(se3n, train_dataloader, k = num_samples, conditioning = conditioning, plot_name=output_dir / f"so3_probabilities_plot_{args.plot_name}.png")
