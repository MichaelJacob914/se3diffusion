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

class r3_diffuser: 
    def __init__(self, prediction, T, batch_size=64, betas=None, device="cuda", cfg = None, verbose = False):
        self.device = torch.device(device)
        self.prediction = prediction
        self.T = T
        self.verbose = verbose
        prediction = "score" 
        if(prediction == "score"): 
            self._r3_conf = cfg
            self.min_b = cfg["min_b"]
            self.max_b = cfg["max_b"]
            self.betas, self.alphas, self.alpha_bars = self.make_schedules_from_marginal(self.T)
            self.beta_hats = self.compute_beta_hat(self.betas, self.alpha_bars)
        elif(prediction == "noise" or prediction == "pose"):
            if betas is None:
                self.betas = self.make_cosine_beta_schedule(T).to(self.device)
            else:
                self.betas = torch.as_tensor(betas, dtype=torch.float32,
                                            device=self.device)

            self.alphas, self.alpha_bars = self.compute_alpha_bars(self.betas)
            self.beta_hats = self.compute_beta_hat(self.betas, self.alpha_bars)
            self.betas, self.alphas, self.alpha_bars = self.make_schedules_from_marginal(self.T)
            self.beta_hats = self.compute_beta_hat(self.betas, self.alpha_bars)

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
   
    def make_beta_schedule(self, T, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

    def make_cosine_beta_schedule(self, T, s=0.008):
        steps = torch.arange(T + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s)
                                   * math.pi * 0.5) ** 2
        alphas_cumprod /= alphas_cumprod[0].clone()
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999)

    def compute_alpha_bars(self, betas):
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return alphas, alpha_bars

    def compute_beta_hat(self, betas, alpha_bars):
        beta_hats = torch.zeros_like(betas)
        beta_hats[1:] = ((1 - alpha_bars[:-1]) / (1 - alpha_bars[1:])) \
                        * betas[1:]
        return beta_hats

    def generate_noise(self, t, B, N, scale = 1.0): 
        if(self.prediction == "score"): 
            return self.score_generate(t)
        else: 
            return self.noise_generate(t, B, N)

    def score_generate(self, ts: torch.LongTensor, scale: float = 1.0):
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
        Ttot = int(self.T)
        tau = ts_flat / float(Ttot)
        m = self.marginal_b_t(tau).to(device=device, dtype=dtype)  # [B*N]
        sigma = scale * torch.sqrt(torch.clamp(1.0 - torch.exp(-m), min=0.0))  # [B*N]

        # Sample Gaussian noise with per-entry std
        eps = torch.randn(B * N, 3, device=device, dtype=dtype) * sigma[:, None]  # [B*N, 3]

        return eps.view(B, N, 3)
                
    def noise_generate(self, ts=None, B=None, N=None, scale=1.0):
        """
        Generate Gaussian noise of shape [B, N, 3] with std based on timestep.

        Args:
            t (int): Scalar diffusion timestep
            ts (Tensor): Tensor of timesteps with shape [B, N]
            B (int): Batch size
            N (int): Number of elements per sample
            scale (float): Optional multiplier for noise scale

        Returns:
            Tensor of shape [B, N, 3]
        """
        ts_flat = ts.reshape(-1).long()  # (B*N,)
        sigma_flat = scale * torch.sqrt(1 - self.alpha_bars[ts_flat])  # (B*N,)
        noise_flat = torch.randn((B * N, 3), device=self.device, dtype=torch.float32)  # (B*N, 3)
        noise_scaled = sigma_flat[:, None] * noise_flat  # (B*N, 3)
        return noise_scaled.view(B, N, 3)
        

    def add_noise(self, x, noise, t): 
        if(self.prediction == "score"): 
            return self.score_forward(x, noise, t)
        else: 
            return self.noise_forward(x, noise, t)
    
    def score_forward(self, x0: torch.Tensor, noise: torch.Tensor, ts: torch.Tensor):
        """
        x0:   [B, N, 3]
        ts:   [B, N]      (timesteps)
        noise:[B, N, 3]   (already ~ N(0, sigma_t^2 I_3) with sigma_t = sqrt(1-exp(-m)))
        """
        Ttot = int(self.T)
        tau = (ts.to(x0.device, torch.float32) / float(Ttot))          # [B, N]
        m = self.marginal_b_t(tau).to(device=x0.device, dtype=x0.dtype)  # [B, N]
        a = torch.exp(-0.5 * m)[..., None]                        # [B, N, 1]
        return a * x0 + noise     
        
    def noise_forward(self, x, noise, t):
        """
        Combine Gaussian noise with clean translations.

        Args:
            x     : [B, N, 3] — clean translation vectors
            noise : [B, N, 3] — noise vectors (already scaled appropriately)
            t     : scalar or [B, N] — diffusion timestep(s)

        Returns:
            x_noisy: [B, N, 3] — noisy translation vectors
        """
        if isinstance(t, (int, float)):
            scale = torch.sqrt(self.alpha_bars[int(t)])
            x_scaled = scale * x
        else:
            t_flat = t.reshape(-1).long()                           # [B*N]
            scale = torch.sqrt(self.alpha_bars[t_flat])          # [B*N]
            scale = scale.view(*x.shape[:2], 1)                  # [B, N, 1]
            x_scaled = x * scale                                 # [B, N, 3]

        return x_scaled + noise
        
    def descent(self, x_t, x_0, t, cost, num_updates = 1, lr = 1e-3): 
        x_0_optim = x_0.clone().detach().requires_grad_(True)
        x_t_optim = x_t.clone().detach().requires_grad_(True)

        for _ in range(num_updates):
            loss = cost(x_t_optim, x_0_optim, t)
            grad = torch.autograd.grad(loss, x_t_optim, create_graph=True)[0]
            x_t_optim = x_t_optim - lr * grad 

        return x_t_optim
    
    def _eu_sample_noise(
        self, x_t, t, eps_pred,
        guidance=False, optim_steps=1, cost=None
    ):
        """This function is used to sample a single vector in R^{3}^{N}. 
        Therefore, both x_t and eps_pred are assumed to be of size [N,3]
        It is important to note that this function assumes noise is passed in and estimates x_0 and x_t-1
        """
        if t > 1:
            v_noise = torch.sqrt(self.beta_hats[t]) * torch.randn_like(x_t)
        else:
            v_noise = torch.zeros_like(x_t)

        t_idx = t - 1
        beta_t = self.betas[t_idx]
        alpha_t = self.alphas[t_idx]
        alpha_bar_t = self.alpha_bars[t_idx]
        alpha_bar_tm1 = self.alpha_bars[t_idx - 1] if t > 1 else alpha_bar_t

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)

        x_mean = coef1 * (x_t - coef2 * eps_pred)

        sigma_t = torch.sqrt(beta_t) * torch.sqrt(1 - alpha_bar_tm1) / torch.sqrt(1 - alpha_bar_t)
        x_prev = x_mean + sigma_t * torch.randn_like(x_t) if t > 1 else x_mean

        if guidance and t % optim_steps == 0:
            with torch.no_grad():
                x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
                x_prev = self.descent(x_prev, x_0_pred, t, cost)

        if self.verbose and t in [1, 10, 50, 90, 100, self.T - 1]:
            print(f"\nStep t={t}")
            print("v_noise   = ", v_noise)
            print("x_prev", x_prev)
            print(f"  ε̂ norm        = {eps_pred.norm(dim=1).mean().item():.4f}")
            print(f"  x_t norm       = {x_t.norm(dim=1).mean().item():.4f}")

        return x_prev, x_mean
    
    def _eu_sample_pose(
        self, x_t, t, x_0,
        guidance=False, optim_steps=1, cost=None
    ):
        """
        Sample x_{t-1} from x_t by first computing x_0.
        Args:
            x_t: [N, 3]
            t: int timestep
            eps_pred: [N, 3] - predicted noise
        Returns:
            x_{t-1}, x_0
        """
        # Compute x_0 from x_t and predicted noise
        alpha_bar_t = self.alpha_bars[t]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

        # Optional guidance via gradient descent on x_0
        if guidance and t % optim_steps == 0:
            x_0 = self.descent(x_t, x_0, t, cost)

        # Compute x_{t-1}
        if t > 1:
            alpha_bar_tm1 = self.alpha_bars[t - 1]
            z = torch.randn_like(x_t)
        else:
            alpha_bar_tm1 = alpha_bar_t
            z = torch.zeros_like(x_t)

        x_prev = torch.sqrt(alpha_bar_tm1) * x_0 + torch.sqrt(1 - alpha_bar_tm1) * z

        if self.verbose and t in [1, 2,3,4,5,6,10, 50, 90, 100, self.T - 1]:
            print(f"\nStep t={t}")
            print("x_0 =", x_0)
            print("x_prev =", x_prev)
            print(f"  x_t norm       = {x_t.norm(dim=1).mean().item():.4f}")

        return x_prev, x_0
        
        
    def _eu_sample_score(
        self,
        x_t: torch.Tensor,                  # [N, 3]
        t: int,                             # discrete step, 0..T
        x_0: torch.Tensor,                  # [N, 3] predicted clean sample
        solver: "SDE",
        guidance: bool = False,
        optim_steps: int = 1,
        cost=None,
        center: bool = False,
        noise_scale: float = 0.0,
        mask: torch.Tensor | None = None,   # optional boolean/binary mask over [N]
        eta: float = 0.0,                   # DDIM stochasticity (0 = deterministic)
        noise: torch.Tensor | None = None,  # DDIM external noise if eta>0
    ):
        """
        One reverse step in R^3 using:
        - 'SDE'  -> reverse-time SDE (stochastic)
        - 'ODE'  -> probability-flow ODE (deterministic)
        - 'DDIM' -> DDIM-style step using x0_hat (deterministic if eta=0)

        Returns:
            x_prev: [N, 3]
            x_0:    [N, 3] (possibly guidance-updated)
        """
        assert isinstance(t, int) and 0 <= t <= self.T, "t must be an integer in [0, T]"
        if t == 0:
            return x_0, x_0

        device, dtype = x_t.device, x_t.dtype
        dt = 1.0 / float(self.T)
        t_cont = float(t) / float(self.T)

        # Optional guidance on x_0 (e.g., DPS/reguidance)
        if guidance and (t % optim_steps == 0):
            x_0 = self.descent(x_t, x_0, t, cost)

        if solver in ("SDE", "ODE"):
            # ---- Only compute score when needed ----
            # compute_score expects [B, N, 3] and [B, N] for t
            x_t_b = x_t.unsqueeze(0)                                  # [1, N, 3]
            x0_b  = x_0.unsqueeze(0)                                  # [1, N, 3]
            t_b   = torch.full((1, x_t.shape[0]), t, device=device, dtype=torch.long)  # [1, N]
            score_t = self.compute_score(x_t_b, t_b, x0_b).squeeze(0) # [N, 3]

            # undo coordinate scaling inside score (matches your earlier code)
            s = torch.as_tensor(self._r3_conf["coordinate_scaling"], device=device, dtype=dtype)
            score_t = score_t / s

            # Optional mask broadcasting to [..., 1]
            np_mask = None
            if mask is not None:
                np_mask = mask.detach().bool().cpu().numpy()

            # Convert inputs to numpy for the SDE/ODE steppers
            x_t_np     = x_t.detach().cpu().numpy()
            score_np   = score_t.detach().cpu().numpy()

            if solver == "SDE":
                x_prev_np = self.reverse(
                    x_t=x_t_np,
                    score_t=score_np,
                    t=t_cont,
                    dt=dt,
                    mask=np_mask,
                    center=center,
                    noise_scale=noise_scale,
                )
            else:  # "ODE"
                x_prev_np = self.reverse_ode(
                    x_t=x_t_np,
                    score_t=score_np,
                    t=t_cont,
                    dt=dt,
                    mask=np_mask,
                    center=center,
                    noise_scale=None,
                )

            x_prev = torch.as_tensor(x_prev_np, device=device, dtype=dtype)
            return x_prev, x_0

        elif solver == "DDIM":
            # DDIM uses x0_hat directly; score not required here
            x_prev = self.reverse_ddim(
                x_t=x_t,                # keep on-device
                x0_hat=x_0,
                t=t_cont,
                dt=dt,
                mask=mask,
                eta=eta,
                noise=noise,
                center=center,
                noise_scale=noise_scale,
            )
            return x_prev, x_0

        else:
            raise ValueError(f"Unknown process '{process}'. Use 'SDE', 'ODE', or 'DDIM'.")
            
    
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
        return 1 / np.sqrt(self.conditional_var(t))

    def reverse(
            self,
            *,
            x_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            mask: np.ndarray=None,
            center: bool=False,
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

    def reverse_ode(self, x_t, score_t, t, dt, mask=None, center = False, noise_scale = None):
        x_t = self._scale(x_t)
        g   = self.diffusion_coef(t)
        f   = self.drift_coef(x_t, t)
        # prob-flow ODE: f - 0.5 g^2 score
        perturb = (f - 0.5 * (g**2) * score_t) * dt
        if mask is not None: perturb *= mask[..., None]
        x_t_1 = x_t - perturb
        return self._unscale(x_t_1)

    def reverse_ddim(self,
                 x_t: torch.Tensor,     # [..., 3]
                 x0_hat: torch.Tensor,  # [..., 3] model’s predicted x0 (same shape)
                 t: float,              # current continuous time in [0,1]
                 dt: float,             # positive step size
                 mask: torch.Tensor | None = None,  # optional boolean/binary mask on the "..." dims
                 eta: float = 0.0,      # 0.0 = deterministic DDIM
                 noise: torch.Tensor | None = None,  # optional external noise if eta>0
                 center = False, 
                 noise_scale = None, 
                 score_t = None
                 ) -> torch.Tensor:
        """
        Continuous-time DDIM (VP) update: go from time t to s=t-dt using only b(t).
        Uses m(s) = integral_0^s b(u) du = marginal_b_t(s).
        """
        if not (isinstance(t, float) or isinstance(t, int)):
            raise ValueError("t must be a scalar float in [0,1].")
        s = max(0.0, float(t) - float(dt))

        # Compute sqrt(alpha_bar_s) and its complement via m(s)
        # alpha_bar(s) = exp(-m(s))
        # sqrt_alpha_bar_s = exp(-0.5 * m(s))
        # std_s = sqrt(1 - alpha_bar(s))
        m_s = self.marginal_b_t(torch.tensor([s], dtype=torch.float32, device=x_t.device))
        sqrt_alpha_bar_s = torch.exp(-0.5 * m_s)[0]                  # scalar tensor
        std_s = torch.sqrt(torch.clamp(1.0 - torch.exp(-m_s)[0], 0.0, 1.0))  # scalar tensor

        # Deterministic DDIM (eta=0) → z = 0
        if eta == 0.0:
            x_s = sqrt_alpha_bar_s * x0_hat
        else:
            # Stochastic DDIM (rarely needed here, but included for completeness)
            if noise is None:
                noise = torch.randn_like(x_t)
            x_s = sqrt_alpha_bar_s * x0_hat + (eta * std_s) * noise

        # Optional masking: update only masked entries
        if mask is not None:
            # mask shape should broadcast over the leading "..." dims
            x_s = torch.where(mask[..., None].bool(), x_s, x_t)

        return x_s

    def conditional_var(self, t, use_torch=False):
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
        Assumes t has shape [B, N] (constant along N) and we map τ = t/1000.

        Args:
            x_t: [B, N, 3]
            t:   [B, N] integer timesteps (expanded along N)
            x0:  [B, N, 3] predicted clean positions
            eps: small constant for numerical stability

        Returns:
            score_t: [B, N, 3]
        """
        device, dtype = x_t.device, x_t.dtype

        # τ = t / 1000, keep one column since t is constant along N
        tau = (t.to(device=device, dtype=torch.float32) / self.T)[..., :1]   # [B, 1]

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




