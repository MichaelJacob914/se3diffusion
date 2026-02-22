#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
import math
import torch.nn as nn
import math
import random
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from mpl_toolkits.mplot3d import Axes3D
import geoopt
from geoopt.optim import (RiemannianAdam)

"""This class is designed to handle diffusion on R3^{n}"""
"""It has been stripped of the noise prediction and VP preserving paths mentioned below"""
"""VE and VP correspond to the forward process on SO(3), not on R3. A VE process on SO(3) corresponds to logarithm schedule on R3"""
class r3_diffuser: 
    def __init__(self, prediction, T, batch_size=64, device="cpu", forward_process = "ve", cfg = None, recenter = True, verbose = False, schedule = "cosine"):
        self.device = torch.device(device)
        self.prediction = prediction
        self.T = T
        self.verbose = verbose
        self.recenter = recenter
        self.schedule = schedule
        self.forward_process = forward_process
        if(self.forward_process == "ve"): 
            self._r3_conf = cfg
            self.min_b = cfg["min_b"]
            self.max_b = cfg["max_b"]

        self.batch_size = batch_size
        
    def make_schedules_from_marginal(self, T: int):
        # grid t_k = k/T, k=0..T
        t = torch.linspace(0., 1., T+1, device=self.device)
        m = self.marginal_b_t(t)                 # m(t) = ∫ b
        alpha_bar_grid = torch.exp(-m)           # [T+1], includes ᾱ_0
        alpha_bar_grid[0] = 1.0

        # α_t on 1..T from consecutive ratios
        alphas = alpha_bar_grid[1:] / torch.clamp_min(alpha_bar_grid[:-1], 1e-12)  # [T]
        betas  = (1. - alphas).clamp(1e-8, 0.999)
        alphas = 1. - betas  # re-sync after clamp

        # Return ᾱ_{1..T} (length T) to match betas/alphas
        alpha_bars = torch.cumprod(alphas, dim=0)  # [T]

        return betas, alphas, alpha_bars

    def generate_noise(self, t, B, N, scale = 1.0): 
        if(self.forward_process == "ve"): 
            return self.ve_generate(t)

    def ve_generate(self, ts: torch.LongTensor, scale: float = 1.0):
        """
        Generate Gaussian noise of shape [B, N, 3] with std based on timestep.

        Args:
            ts (torch.LongTensor): [B, N] timesteps
            scale (float): optional multiplier on the per-timestep std

        Returns:
            torch.Tensor: noise of shape [B, N, 3]
        """
        B, N = ts.shape
        device, dtype = ts.device, torch.float32

        # Compute m and sigma per entry
        ts_flat = ts.reshape(-1).to(torch.float32)
        tau = ts_flat.clamp(0.01, 1.0)
        m = self.marginal_b_t(tau).to(device=device, dtype=dtype)  # [B*N]
        sigma = scale * torch.sqrt(torch.clamp(1.0 - torch.exp(-m), min=0.0))  # [B*N]

        # Sample Gaussian noise with per-entry std
        eps = torch.randn(B * N, 3, device=device, dtype=dtype) * sigma[:, None]  # [B*N, 3]

        return eps.view(B, N, 3)
                

    def add_noise(self, x, noise, t): 
        if(self.forward_process == "ve"): 
            return self.ve_forward(x, noise, t)
        else: 
            return self.vp_forward(x, noise, t)
    
    def ve_forward(self, x0: torch.Tensor, noise: torch.Tensor, ts: torch.Tensor):
        """
        x0:   [B, N, 3]
        ts:   [B, N]      (timesteps)
        noise:[B, N, 3]   (already ~ N(0, sigma_t^2 I_3) with sigma_t = sqrt(1-exp(-m)))
        """
        tau = ts.to(x0.device, torch.float32).clamp(0.01, 1.0)
        m = self.marginal_b_t(tau).to(device=x0.device, dtype=x0.dtype)  # [B, N]
        a = torch.exp(-0.5 * m)[..., None]                        # [B, N, 1]
        x_t = a * x0 + noise     

        if(self.recenter): 
            return self.center(x_t)
        else: 
            return x_t
        
    def _eu_sample_ve(
        self,
        x_t: torch.Tensor,                  # [N, 3]
        t: torch.Tensor,                             # discrete step, 0..T
        x_0: torch.Tensor,                  # [N, 3] predicted clean sample
        solver: "SDE",
        threshold = 0, 
        guidance: bool = False,
        optim_steps: int = 1,
        cost=None,
        noise_scale: float = 0.0,
        mask: torch.Tensor | None = None,   # optional boolean/binary mask over [N]
    ):
        """
        One reverse step in R^3 using:
        - 'SDE'  -> reverse-time SDE (stochastic)
        - 'ODE'  -> probability-flow ODE (deterministic)
        - 'DDIM' -> DDIM-style step using x0_hat (deterministic if eta=0)

        Returns:
            x_prev: [B,N, 3]
            x_0:    [B,N, 3]
        """
        if t == 0:
            return x_0, x_0

        device, dtype = x_t.device, x_t.dtype
        B, N = x_t.shape[:2]

        t_cont = torch.as_tensor(t, device=device, dtype=torch.float32).clamp(0.01, 1.0)
        dt = 1.0 / float(self.T)   # keep small Euler step size

        if self.prediction == "score":
            # ---- Only compute score when needed ----
            # compute_score expects [B, N, 3] and [B, N] for t
            x_t_s = self._scale(x_t)
            x_0_s = self._scale(x_0)
            t_b = t_cont.expand(B, N)
            score_t = self.compute_score(x_t_s, t_b, x_0_s)

            x_prev = self.reverse_sde_torch(
                x_t=x_t, score_t=score_t, t_cont=t_cont, dt=dt,
                mask=mask, center=self.recenter, noise_scale=noise_scale
            )
            return x_prev, x_0
    
    def _scale(self, x):
        return x * self._r3_conf["coordinate_scaling"]

    def _unscale(self, x):
        return x / self._r3_conf["coordinate_scaling"]

    def b_t(self, t):
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.min_b + t*(self.max_b - self.min_b)

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return np.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t) * x

    def sample_ref(self, n_samples: float=1):
        return np.random.normal(size=(n_samples, 3))

    def marginal_b_t(self, t):
        return t*self.min_b + (1/2)*(t**2)*(self.max_b-self.min_b)

    def b_t_torch(self, t: torch.Tensor) -> torch.Tensor:
        return self.min_b + t * (self.max_b - self.min_b)

    def diffusion_coef_torch(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.b_t_torch(t))

    def drift_coef_torch(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.b_t_torch(t)[..., None, None] * x

    def calc_trans_0(self, score_t, x_t, t, use_torch=True):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1/2*beta_t)

    def forward(self, x_t_1: np.ndarray, t: float, num_t: int):
        """Samples marginal p(x(t) | x(t-1)).

        Args:
            x_t-1: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t).to(x_t_1.device)
        z_t_1 = torch.tensor(np.random.normal(size=x_t_1.shape)).to(x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * np.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std

    def forward_marginal(self, x_0: np.ndarray, t: float):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_0 = self._scale(x_0)
        x_t = np.random.normal(
            loc=np.exp(-1/2*self.marginal_b_t(t)) * x_0,
            scale=np.sqrt(1 - np.exp(-self.marginal_b_t(t)))
        )
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        return x_t, score_t

    def score_scaling(self, t: float):
        return 1 / torch.sqrt(self.conditional_var(t))

    def center(self, x_t):
        """
        Recenters translations so that the mean over points is 0.
        Expects x_t shape: (batch, N, 3) or (batch, ..., N, 3).
        """
        # mean over the points dimension (here assumed dim=-2)
        com = x_t.mean(dim=-2, keepdim=True)  # shape (batch, 1, 3)
        return x_t - com

    def reverse_sde(
            self,
            *,
            x_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            mask: np.ndarray=None,
            center: bool=True,
            noise_scale: float=0.0,
        ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * np.random.normal(size=score_t.shape)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(dt) * z

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = np.ones(x_t.shape[:-1])
        x_t_1 = x_t - perturb
        if center:
            com = np.sum(x_t_1, axis=-2) / np.sum(mask, axis=-1)[..., None]
            x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def reverse_sde_torch(
        self,
        x_t: torch.Tensor,        # [B,N,3] UNscaled
        score_t: torch.Tensor,    # [B,N,3] score wrt SCALED variable 
        t_cont: float,
        dt: float,
        mask: torch.Tensor | None = None,   # [B,N] or [N]
        center: bool = True,
        noise_scale: float = 0.0,
    ):
        device, dtype = x_t.device, x_t.dtype

        # scale x
        x = self._scale(x_t)  # [B,N,3]

        # t tensor as [B,1] for broadcast
        B, N = x.shape[0], x.shape[1]
        t = torch.full((B, 1), t_cont, device=device, dtype=torch.float32)

        g = self.diffusion_coef_torch(t)  # [B,1]
        b = self.b_t_torch(t)             # [B,1]
        f = -0.5 * b[..., None] * x       # [B,N,3]

        if noise_scale > 0.0:
            z = torch.randn_like(score_t) * noise_scale
        else:
            z = torch.zeros_like(score_t)

        perturb = (f - (g[..., None] ** 2) * score_t) * dt + g[..., None] * math.sqrt(dt) * z

        if mask is not None:
            if mask.dim() == 1:
                mask = mask[None, :].expand(B, -1)  # [B,N]
            perturb = perturb * mask[..., None]

        x_1 = x - perturb  # [B,N,3]

        if center:
            if mask is not None:
                wsum = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)     # [B,1]
                com = (x_1 * mask[..., None]).sum(dim=-2) / wsum         # [B,3]
                x_1 = x_1 - com[:, None, :]
            else:
                x_1 = self.center(x_1)

        return self._unscale(x_1)

    def conditional_var(self, t, use_torch=True):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        if use_torch:
            return 1 - torch.exp(-self.marginal_b_t(t))
        return 1 - np.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, use_torch=False, scale=False):
        if use_torch:
            exp_fn = torch.exp
        else:
            exp_fn = np.exp
        if scale:
            x_t = self._scale(x_t); x_0 = self._scale(x_0)
        if use_torch:
            t_t = torch.tensor([t], dtype=x_t.dtype, device=x_t.device)
            m = self.marginal_b_t(t_t)
            return -(x_t - torch.exp(-0.5 * m) * x_0) / (1 - torch.exp(-m))
        else:
            t_f = float(t)
            m = self.marginal_b_t(t_f)
            return -(x_t - np.exp(-0.5 * m) * x_0) / (1 - np.exp(-m))

    def compute_score(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor, eps: float = 1e-12):
        """
        Compute the true score ∇_{x_t} log p(x_t | x_0) using the continuous VP/SDE.
        Args:
            x_t: [B, N, 3]
            t:   [B, N] integer timesteps (expanded along N)
            x0:  [B, N, 3] predicted clean positions
            eps: small constant for numerical stability

        Returns:
            score_t: [B, N, 3]
        """
        device, dtype = x_t.device, x_t.dtype

        # τ = t / 100, keep one column since t is constant along N
        tau = t.to(device=device, dtype=torch.float32)[..., :1]

        # m(τ) = ∫ b(s) ds; a = exp(-m/2); 1 - a^2 = 1 - exp(-m)
        m = self.marginal_b_t(tau)                         # [B, 1] (torch ops)
        m = m.to(device=device, dtype=torch.float32)
        a = torch.exp(-0.5 * m).to(dtype=dtype)            # [B, 1]
        one_minus_a2 = (1.0 - torch.exp(-m)).to(dtype)     # [B, 1]
        one_minus_a2 = torch.clamp(one_minus_a2, min=eps)

        # Broadcast over N and the last dim=3
        a = a[..., None]                 # [B, 1, 1] -> broadcasts over N and 3
        one_minus_a2 = one_minus_a2[..., None]  # [B, 1, 1]

        # Score: (a * x0 - x_t) / (1 - a^2)
        return (a * x0 - x_t) / one_minus_a2



