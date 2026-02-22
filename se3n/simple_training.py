# diffusion_forward_corruption_diffusionloss_unweighted.py
import os, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator

HERE = Path(__file__).resolve().parent
paths = [HERE / "gta", HERE / "gta" / "source"]
sys.path[:0] = [str(p) for p in paths if p.exists() and str(p) not in sys.path]

from se3n_models import PRoPEUpdatePoseTransformer
from SO3n import so3_diffuser
from R3n import r3_diffuser
from losses import diffusion_loss_relative_pose  # adjust if different path


torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def set_sdp(mode: str):
    if mode == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif mode == "all":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    else:
        raise ValueError(mode)


SO3_CONFIG = {
    "schedule": "logarithmic",
    "min_sigma": 0.1,
    "max_sigma": 1.5,
    "num_sigma": 1000,
    "use_cached_score": False,
    "num_omega": 2000,
    "cache_dir": "./igso3_cache",
    "forward_process": "ve",
}

R3_CONFIG = {
    "min_b": 0.1,
    "max_b": 20.0,
    "coordinate_scaling": 1,
    "schedule": "cosine",
    "forward_process": "ve",
    "recenter": False,
}


def sample_t(B: int, N: int, T: int, device: str, scheme: str = "batch"):
    t_min = 1.0 / T
    t_max = (T - 1.0) / T
    if scheme == "pose":
        time_tensor = t_min + (t_max - t_min) * torch.rand((B, N), device=device)
    elif scheme == "batch":
        t_batch = t_min + (t_max - t_min) * torch.rand((B, 1), device=device)
        time_tensor = t_batch.expand(B, N)
    elif scheme == "global":
        t_global = t_min + (t_max - t_min) * torch.rand((1,), device=device)
        time_tensor = t_global.expand(B, N)
    else:
        raise ValueError(scheme)
    return time_tensor


# --- force weights=1 while keeping EXACT same diffusion loss code path ---
def diffusion_loss_relative_pose_unweighted(diffuser_like, R_clean, T_clean, R_pred, T_pred, t, eps=1e-8, lambda_dir=1.0):
    class _One:
        def score_scaling(self, t_b):
            # t_b: [B]
            return torch.ones_like(t_b)

    so3_saved, r3_saved = diffuser_like.so3, diffuser_like.r3
    diffuser_like.so3, diffuser_like.r3 = _One(), _One()
    try:
        return diffusion_loss_relative_pose(diffuser_like, R_clean, T_clean, R_pred, T_pred, t, eps=eps, lambda_dir=lambda_dir)
    finally:
        diffuser_like.so3, diffuser_like.r3 = so3_saved, r3_saved


class DiffuserLike:
    """Just so diffusion_loss_relative_pose(self, ...) can call self.so3/self.r3.score_scaling"""
    def __init__(self, so3, r3):
        self.so3 = so3
        self.r3 = r3


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # knobs
    T_train = 100
    t_scheme = "batch"
    sdp_mode = "math"
    use_autocast = False
    amp_dtype = "bf16"
    steps = 200
    lambda_dir = 1.0

    B, N = 2, 4
    d_model = 1024
    Hf = Wf = 14
    img_size = 224

    set_sdp(sdp_mode)

    accelerator = Accelerator(
        mixed_precision=(
            "bf16" if (use_autocast and amp_dtype == "bf16") else
            "fp16" if (use_autocast and amp_dtype == "fp16") else
            "no"
        )
    )

    model = PRoPEUpdatePoseTransformer(
        device=device,
        representation="rot6d",
        scheme="PRoPE",
        feature_type="dust3r",
        update_type="absolute",
    ).to(device).train()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    model, opt = accelerator.prepare(model, opt)

    # diffusion forward corruption (same as Mode 2)
    so3 = so3_diffuser(
        prediction="noise",
        forward_process="ve",
        T=T_train,
        device=torch.device(device),
        representation="rot6d",
        cfg=SO3_CONFIG,
    )
    r3 = r3_diffuser(
        prediction="noise",
        T=T_train,
        device=torch.device(device),
        cfg=R3_CONFIG,
        recenter=True,
    )
    diffuser_like = DiffuserLike(so3, r3)

    # synthetic clean inputs
    R_clean = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, N, 3, 3).contiguous()
    T_clean = 0.1 * torch.randn(B, N, 3, device=device)

    feats = torch.randn(B, N, d_model, Hf, Wf, device=device) * 3.0
    imgs  = torch.rand(B, N, 3, img_size, img_size, device=device)
    Ks    = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, N, 3, 3).contiguous()

    torch.cuda.synchronize()
    t0 = time.time()

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        t = sample_t(B, N, T_train, device=device, scheme=t_scheme)

        # corrupt
        R_noise, _ = so3.generate_noise(t, B, N)
        T_noise    = r3.generate_noise(t, B, N)
        R_t = so3.add_noise(R_clean, R_noise, t)
        T_t = r3.add_noise(T_clean, T_noise, t)

        with accelerator.autocast():
            out = model(R_t, T_t, feats, imgs, Ks, t)
            R_pred, T_pred = out["R"], out["t"]

            # Mode 3a: diffusion loss but UNWEIGHTED
            loss_rot, loss_rot_geo, loss_trans, loss_trans_dir = diffusion_loss_relative_pose_unweighted(
                diffuser_like, R_clean, T_clean, R_pred, T_pred, t, lambda_dir=lambda_dir
            )
            loss = loss_rot + loss_rot_geo + loss_trans + loss_trans_dir

        accelerator.backward(loss)
        opt.step()

        if step in (0, 1, 2, 5, 10, 25, 50, 100, steps - 1):
            accelerator.print(
                f"step={step:04d} loss={loss.detach().float().item():.6g} "
                f"(rot={float(loss_rot):.3g} geo={float(loss_rot_geo):.3g} "
                f"trans={float(loss_trans):.3g} dir={float(loss_trans_dir):.3g}) "
                f"t=[{float(t.min()):.3f},{float(t.max()):.3f}]"
            )

        if not torch.isfinite(loss.detach()):
            accelerator.print(f"NON-FINITE loss at step={step}")
            break

    torch.cuda.synchronize()
    accelerator.print(f"done. elapsed={time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()