#!/usr/bin/env python
# coding: utf-8

# In[2]:


from SO3n import so3_diffuser, SO3Algebra
from R3n import r3_diffuser
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as SciRot
import math
import math
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import geoopt
from geoopt.optim import (RiemannianAdam)
Stiefel = geoopt.Stiefel()
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.cm as cm

"""
This class is designed to test individual pieces of the SO3 and R3 diffusers to ensure they work consistently. 
"""

def test_so3_alg(batch_size: int = 8, dtype=torch.float32, device='cpu'):
    """
    Check that log_map, get_v, and exp_map agree with SciPy on a batch
    of random rotation matrices.
    """
    so3 = SO3Algebra()
    R_np = SciRot.random(batch_size).as_matrix()          # (B,3,3)
    R_t  = torch.tensor(R_np, dtype=dtype, device=device) # (B,3,3)

    S_skew   = so3.log_map(R_t)                 # (B,3,3)
    v_tensor = so3.get_v(S_skew)                # (B,3)
    R_recon  = so3.exp_map(v_tensor)            # (B,3,3)

    v_gt = SciRot.from_matrix(R_np).as_rotvec() # (B,3)

    v_err  = torch.linalg.norm(
        torch.tensor(v_gt, dtype=dtype, device=device) - v_tensor, dim=-1)
    R_err  = torch.linalg.norm(
        torch.tensor(R_np, dtype=dtype, device=device) - R_recon, dim=(-2, -1))

    print("Axis-angle error ‖v_gt − v_pred‖ per sample:")
    for i, e in enumerate(v_err.cpu().numpy()):
        print(f"  sample {i:02d}: {e:.3e}")

    print("\nFrobenius error ‖R − exp_map(v_pred)‖_F per sample:")
    for i, e in enumerate(R_err.cpu().numpy()):
        print(f"  sample {i:02d}: {e:.3e}")

def test_geodesic_and_tilde_nu_batching(N=10, dtype=torch.float32, device='cpu'):
    so3 = SO3Algebra()
    R_np = SciRot.random(N).as_matrix()                   # [N, 3, 3]
    R_t = torch.tensor(R_np, dtype=dtype, device=device)  # [N, 3, 3]

    # Clone for tilde_nu test
    x_0 = R_t.clone()
    x_t = SciRot.random(N).as_matrix()
    x_t = torch.tensor(x_t, dtype=dtype, device=device)

    T = 1000
    alphas = torch.linspace(0.99, 0.9, T, dtype=dtype)
    alpha_bars = torch.cumprod(alphas, dim=0)
    betas = 1 - alphas
    t = 500  

    gamma = 0.5  

    R_interp_batch = so3.geodesic_interpolation(gamma, R_t)

    # Per-sample call and stack
    R_interp_list = [
        so3.geodesic_interpolation(gamma, R_t[i].unsqueeze(0))[0]
        for i in range(N)
    ]
    R_interp_loop = torch.stack(R_interp_list)

    interp_error = torch.linalg.norm(R_interp_batch - R_interp_loop, dim=(-2, -1))
    print("Geodesic interpolation batch-vs-loop Frobenius error per sample:")
    for i, e in enumerate(interp_error.cpu().numpy()):
        print(f"  sample {i:02d}: {e:.3e}")

    # === 2. Tilde Nu === #
    # Batched call
    tilde_batch = so3.tilde_nu(x_t, x_0, t, alphas, alpha_bars, betas)

    # Per-sample call and stack
    tilde_loop = torch.stack([
        so3.tilde_nu(x_t[i].unsqueeze(0), x_0[i].unsqueeze(0), t, alphas, alpha_bars, betas)[0]
        for i in range(N)
    ])

    tilde_error = torch.linalg.norm(tilde_batch - tilde_loop, dim=(-2, -1))
    print("\nTilde nu batch-vs-loop Frobenius error per sample:")
    for i, e in enumerate(tilde_error.cpu().numpy()):
        print(f"  sample {i:02d}: {e:.3e}")

    
def test_generate_noise_correctness(B=4, N=6, t=500, tol=1e-4, dtype=torch.float32, device='cpu'):
    T = 1000
    alphas = torch.linspace(0.99, 0.9, T, dtype=dtype)
    alpha_bars = torch.cumprod(alphas, dim=0)
    betas = 1 - alphas
    diffuser = so3_diffuser(1000, B, betas, device=device)

    print(f"\nTesting generate_noise with B={B}, N={N}, t={t}")
    R, v = diffuser.generate_noise(t, B, N)

    # 1. Shape checks
    assert R.shape == (B, N, 3, 3), f"R shape mismatch: {R.shape}"
    assert v.shape == (B, N, 3), f"v shape mismatch: {v.shape}"

    # 2. Check orthonormality: RᵀR = I and det(R) = 1
    R_flat = R.view(-1, 3, 3)
    I = torch.eye(3, device=R.device)
    orthogonality_error = torch.linalg.norm(
        torch.matmul(R_flat.transpose(-1, -2), R_flat) - I, dim=(1, 2)
    )
    dets = torch.linalg.det(R_flat)

    print(f"Max orthogonality error ‖RᵀR - I‖: {orthogonality_error.max().item():.2e}")
    print(f"Min/Max determinant: {dets.min().item():.6f}, {dets.max().item():.6f}")

    assert (torch.abs(dets - 1) < tol).all(), "Some rotation matrices have det ≠ 1"
    assert (orthogonality_error < tol).all(), "Some RᵀR deviates from identity"

    # 3. Check that exp_map(v) ≈ R
    R_reconstructed = diffuser.alg.exp_map(v.view(-1, 3))  # [B*N, 3, 3]
    R_err = torch.linalg.norm(R_flat - R_reconstructed, dim=(1, 2))  # Frobenius norm

    print(f"Max exp_map error ‖exp(v) - R‖: {R_err.max().item():.2e}")
    print(f"Mean exp_map error: {R_err.mean().item():.2e}")

    assert (R_err < tol).all(), "exp_map(v) did not match R"

    # 4. Check that get_v(log_map(R)) ≈ v
    S_skew = diffuser.alg.log_map(R_flat)              # [B*N, 3, 3]
    v_from_log = diffuser.alg.get_v(S_skew)            # [B*N, 3]
    v_flat = v.view(-1, 3)

    R_from_logv = diffuser.alg.exp_map(v_from_log)   # [B*N, 3, 3]
    R_from_v    = diffuser.alg.exp_map(v_flat)

    v_angle_error = torch.linalg.norm(R_from_logv - R_from_v, dim=(1, 2))
    print(f"Max angle error ‖exp(log(R)) - exp(v)‖: {v_angle_error.max().item():.2e}")
    print(f"Mean angle error: {v_angle_error.mean().item():.2e}")
    assert (v_angle_error < tol).all(), "exp(log(R)) did not match exp(v)"

    print("generate_noise passed all checks.")

def test_add_noise_correctness(B=4, N=6, t=500, tol=1e-4, dtype=torch.float32, device='cpu'):
    T = 1000
    alphas = torch.linspace(0.99, 0.9, T, dtype=dtype)
    alpha_bars = torch.cumprod(alphas, dim=0)
    betas = 1 - alphas
    diffuser = so3_diffuser(1000, B, betas, device=device)

    print(f"\nTesting add_noise with B={B}, N={N}, t={t}")
    
    # Generate clean rotations
    x = torch.tensor(SciRot.random(B * N).as_matrix(), dtype=dtype, device=device).view(B, N, 3, 3)
    
    # Generate noise
    R_noises, v = diffuser.generate_noise(t, B, N)
    
    # Apply noise
    x_noisy = diffuser.add_noise(x, R_noises, t)

    # 1. Shape check
    assert x_noisy.shape == (B, N, 3, 3), f"x_noisy shape mismatch: {x_noisy.shape}"

    # 2. Check orthonormality and det = 1
    R_flat = x_noisy.view(-1, 3, 3)
    I = torch.eye(3, device=device)
    orthogonality_error = torch.linalg.norm(
        torch.matmul(R_flat.transpose(-1, -2), R_flat) - I, dim=(1, 2)
    )
    dets = torch.linalg.det(R_flat)

    print(f"Max orthogonality error ‖RᵀR - I‖: {orthogonality_error.max().item():.2e}")
    print(f"Min/Max determinant: {dets.min().item():.6f}, {dets.max().item():.6f}")

    assert (torch.abs(dets - 1) < tol).all(), "Some noisy matrices have det ≠ 1"
    assert (orthogonality_error < tol).all(), "Some noisy RᵀR deviates from identity"

    # 3. Verify that the noise applied corresponds to v
    # Compute scaled x: exp(sqrt(ᾱ_t) * log(x))
    S_scaled = diffuser.alg.log_map(x.view(-1, 3, 3))
    v_scaled = diffuser.alg.get_v(S_scaled) * torch.sqrt(alpha_bars[t])
    x_scaled = diffuser.alg.exp_map(v_scaled)

    # Estimate R_est = x_noisy @ x_scaled⁻¹ ≈ R_noises
    R_est = torch.bmm(R_flat, x_scaled.transpose(-1, -2))  # [B*N, 3, 3]

    # Compare estimated noise with original noise
    R_noise_flat = R_noises.view(-1, 3, 3)
    diff_err = torch.linalg.norm(R_est - R_noise_flat, dim=(1, 2))
    print(f"Max noise recovery error ‖R_est - R_noises‖: {diff_err.max().item():.2e}")
    print(f"Mean noise recovery error: {diff_err.mean().item():.2e}")
    assert (diff_err < tol).all(), "Reconstructed noise deviates from sampled noise"

    print("add_noise passed all checks.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--B", type=int, default=4, help="Batch size")
    parser.add_argument("--N", type=int, default=6, help="Number of elements per SE(3)^N")
    parser.add_argument("--t", type=int, default=500, help="Diffusion timestep to test")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for error assertions")
    args = parser.parse_args()

    device = args.device
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    B = args.B
    N = args.N
    t = args.t
    tol = args.tol

    print(f"Running tests with device={device}, dtype={dtype}, B={B}, N={N}, t={t}, tol={tol}\n")

    test_so3_alg(B, dtype=dtype, device=device)
    test_geodesic_and_tilde_nu_batching(N=N, dtype=dtype, device=device)
    test_generate_noise_correctness(B=B, N=N, t=t, tol=tol, dtype=dtype, device=device)
    test_add_noise_correctness(B=B, N=N, t=t, tol=tol, dtype=dtype, device=device)

    print("\nAll tests completed successfully.")

