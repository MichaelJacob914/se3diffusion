import os, json, random
import numpy as np
import torch
from tqdm import tqdm

from datasets import PoseCo3DDataset

# ---- Dust3R ----
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# ---- VGGT/Pi3 metrics ----
from pi3.relpose.metric import se3_to_relative_pose_error, calculate_auc


CATEGORIES = [
    "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
    "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave", "motorcycle",
    "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
    "skateboard", "stopsign", "suitcase", "teddybear", "toaster", "toilet", "toybus",
    "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
]


def to_hwc_uint8(img_chw):
    # img_chw: torch (3,H,W) float in [0,1] or [0,255]
    x = img_chw.detach().cpu()
    if x.dtype != torch.uint8:
        x = (x.clamp(0, 1) * 255.0).to(torch.uint8)
    x = x.permute(1, 2, 0).contiguous().numpy()  # HWC
    return x


@torch.no_grad()
def dust3r_predict_poses_from_imgs(
    imgs_chw: torch.Tensor,   # [V,3,H,W] float in [0,1]
    model: AsymmetricCroCo3DStereo,
    device: str,
    batch_size: int = 1,
    niter: int = 300,
    lr: float = 0.01,
    schedule: str = "cosine",
    scene_graph: str = "complete",
    symmetrize: bool = True,
    poses_are_c2w: bool = True,   # <-- flip to False if get_im_poses() is w2c in your env
):
    """
    Returns w2c poses as torch.Tensor [V,4,4]
    """
    V = imgs_chw.shape[0]

    # Dust3R expects a list of dict-like images produced by dust3r.utils.image.load_images,
    # but load_images reads from paths. We can mimic the structure it expects.
    # The inference() function ultimately needs 'img' tensors and 'true_shape'.
    #
    # The simplest low-friction approach: create "images" list compatible with make_pairs.
    images = []
    for i in range(V):
        hwc = to_hwc_uint8(imgs_chw[i])
        H, W = hwc.shape[:2]
        images.append({
            "idx": i,
            "img": hwc,               # numpy HWC uint8
            "true_shape": (H, W),
            "instance": f"seq_img_{i}",
        })

    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=symmetrize)

    output = inference(pairs, model, device, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    _ = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    poses = scene.get_im_poses()  # torch [V,4,4] in SOME convention

    if poses_are_c2w:
        w2c = torch.linalg.inv(poses)
    else:
        w2c = poses

    return w2c


@torch.no_grad()
def eval_category_dust3r(
    category: str,
    co3d_root: str,
    ann_root: str,
    device: str = "cuda",
    split: str = "test",
    num_seqs: int = 100,
    views_minmax=(2, 6),
    auc_thresholds=(5, 10, 15, 30),
    resize_hw=(224, 224),

    # dust3r params
    dust3r_ckpt: str = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
    batch_size: int = 1,
    niter: int = 300,
    lr: float = 0.01,
    schedule: str = "cosine",
    poses_are_c2w: bool = True,   # <-- you may need to flip after a quick sanity check
):
    ds = PoseCo3DDataset(
        co3d_root=co3d_root,
        ann_root=ann_root,
        categories=(category,),
        split=split,
        resize_hw=resize_hw,
        verbose=False,
    )
    if len(ds.sequence_list) == 0:
        return None

    seqs = ds.sequence_list
    if num_seqs is not None and num_seqs < len(seqs):
        seqs = random.sample(seqs, num_seqs)

    # load Dust3R once per category
    model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_ckpt).to(device).eval()

    all_rerr, all_terr = [], []

    for seq in tqdm(seqs, desc=f"[Dust3R {category}]"):
        V = random.randint(views_minmax[0], views_minmax[1])

        n_frames = len(ds.rotations[seq])
        if n_frames < V:
            continue
        ids = random.sample(range(n_frames), V)

        batch = ds.get_data(sequence_name=seq, ids=ids)

        images = batch["imgs"].to(device)  # [V,3,H,W]

        images = batch["imgs"].to(device)  # [V,3,H,W]

        # Dust3R wants a list of images
        images_list = [img.permute(1,2,0).cpu().numpy() for img in images]

        images_d3r = load_images(images_list, size=512)
        pairs = make_pairs(images_d3r, scene_graph="complete", symmetrize=True)

        output = inference(pairs, model, device=device, batch_size=1)

        scene = global_aligner(
            output,
            device=device,
            mode=GlobalAlignerMode.PointCloudOptimizer
        )
        scene.compute_global_alignment(init="mst", niter=300, lr=0.01)

        pred_c2w = scene.get_im_poses().to(device)    # [V,4,4]
        pred_w2c = torch.linalg.inv(pred_c2w)

        # ---- GT ----
        gt_c2w = batch["camera_pose"].to(device)
        gt_w2c = torch.linalg.inv(gt_c2w)

        # ---- Metric (identical to Pi3) ----
        rerr_deg, terr_deg = se3_to_relative_pose_error(
            pred_se3=pred_w2c,
            gt_se3=gt_w2c,
            num_frames=pred_w2c.shape[0],
        )

        all_rerr.append(rerr_deg.detach().cpu())
        all_terr.append(terr_deg.detach().cpu())

    if not all_rerr:
        return None

    r = torch.cat(all_rerr, dim=0)
    t = torch.cat(all_terr, dim=0)

    out = {
        "category": category,
        "num_sequences": len(seqs),
        "num_pairs": int(r.numel()),
        "r_mean": float(r.mean().item()),
        "t_mean": float(t.mean().item()),
    }

    for thr in auc_thresholds:
        out[f"Racc_{thr}"] = float((r < thr).float().mean().item() * 100.0)
        out[f"Tacc_{thr}"] = float((t < thr).float().mean().item() * 100.0)
        out[f"AUC_{thr}"]  = float(calculate_auc(r, t, max_threshold=thr).item() * 100.0)

    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--co3d_root", required=True)
    parser.add_argument("--ann_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_seqs", type=int, default=50)  # Dust3R is slower; start smaller
    parser.add_argument("--out_json", default="dust3r_by_category.json")

    # dust3r knobs
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--schedule", default="cosine")
    parser.add_argument("--poses_are_c2w", action="store_true")  # set this if get_im_poses() is c2w
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0)

    results = []
    for cat in CATEGORIES:
        res = eval_category_dust3r(
            category=cat,
            co3d_root=args.co3d_root,
            ann_root=args.ann_root,
            split=args.split,
            num_seqs=args.num_seqs,
            device=device,
            niter=args.niter,
            lr=args.lr,
            schedule=args.schedule,
            poses_are_c2w=args.poses_are_c2w,
        )
        if res is not None:
            results.append(res)
            print(res)

    results.sort(key=lambda d: d.get("AUC_30", -1), reverse=True)

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
