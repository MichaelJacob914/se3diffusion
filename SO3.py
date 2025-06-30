#!/usr/bin/env python
# coding: utf-8

# In[8]:


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
Stiefel = geoopt.Stiefel()


# In[9]:


#TAKEN FROM PAPER
#NOT MY CODE


class IGSO3Sampler:
    def __init__(self, num_sigma=100, num_omega=500, min_sigma=0.01, max_sigma=3.0, L=2000):
        self.L = L
        self.igso3_vals = self._calculate_igso3(num_sigma, num_omega, min_sigma, max_sigma)
        self.discrete_sigma = self.igso3_vals['discrete_sigma']

    def _igso3_expansion(self, omega, sigma):
        p = 0
        for l in range(self.L):
            term = (2 * l + 1) * np.exp(-l * (l + 1) * sigma**2 / 2)
            term *= np.sin(omega * (l + 0.5)) / np.sin(omega / 2)
            p += term
        return p

    def _density(self, expansion, omega):
        return expansion * (1 - np.cos(omega)) / np.pi

    def _calculate_igso3(self, num_sigma, num_omega, min_sigma, max_sigma):
        omega_vals = np.linspace(0, np.pi, num_omega + 1)[1:]  # skip omega = 0
        sigma_vals = 10 ** np.linspace(np.log10(min_sigma), np.log10(max_sigma), num_sigma + 1)[1:]

        expansions = np.asarray([
            self._igso3_expansion(omega_vals, sigma) for sigma in sigma_vals
        ])
        pdf_vals = np.asarray([
            self._density(exp, omega_vals) for exp in expansions
        ])
        cdf_vals = np.asarray([
            pdf.cumsum() / num_omega * np.pi for pdf in pdf_vals
        ])

        return {
            'cdf': cdf_vals,
            'pdf': pdf_vals,
            'discrete_omega': omega_vals,
            'discrete_sigma': sigma_vals,
        }

    def _sigma_to_idx(self, sigma):
        idx = np.searchsorted(self.discrete_sigma, sigma)
        if idx >= len(self.discrete_sigma):
            idx = len(self.discrete_sigma) - 1
        return idx

    def sample_vector(self, sigma, n_samples=1):
        """Returns [n_samples, 3] rotation vectors sampled from IGSO(3) with given std sigma."""
        sigma_idx = self._sigma_to_idx(sigma)
        cdf = self.igso3_vals['cdf'][sigma_idx]
        omega_vals = self.igso3_vals['discrete_omega']
        
        u = np.random.rand(n_samples)
        sampled_omega = np.interp(u, cdf, omega_vals)  # shape: [n_samples]

        axis = np.random.randn(n_samples, 3)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)

        return axis * sampled_omega[:, None]


# In[19]:


class SO3Algebra:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.exp_map = torch.linalg.matrix_exp


    def log_map_si(self, R_mat):
        #3x3 ROTATION MATRIX TO 3x3 SKEW ROTATION MATRIX 
        cos_theta = ((R_mat.diagonal().sum() - 1.0) / 2.0).clamp(-1.0, 1.0)
        theta     = torch.acos(cos_theta)

        if torch.isclose(theta, torch.tensor(0.0, dtype=R_mat.dtype, device=R_mat.device)):
            return torch.zeros_like(R_mat)

        return (theta / (2.0 * torch.sin(theta))) * (R_mat.transpose(-1, -2) - R_mat)
    
    def skew_si(self, v):
        #3x1 VECTOR TO 3x3 SKEW ROTATION MATRIX
        zero = torch.tensor(0.0, dtype=v.dtype, device=v.device)
        return torch.stack((
            torch.stack(( zero,  v[2], -v[1])),
            torch.stack((-v[2],  zero,  v[0])),
            torch.stack(( v[1], -v[0],  zero))
        ))
    
    def get_v_si(self, Sv):
        #3x3 SKEW ROTATION MATRIX TO 3x1 VECTOR 
        return torch.stack((
            -Sv[2, 1],
            -Sv[0, 2],
           Sv[1, 0]
        ))
        
    def exp_map_si(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Rodrigues’ formula: 3-vector (axis-angle) → 3×3 rotation matrix.
        """
        theta = torch.linalg.norm(omega)                     # ‖ω‖
        if theta < 1e-4:                                     # near-identity
            return torch.eye(3, dtype=omega.dtype, device=omega.device)

        K = self.skew(omega / theta)                         # 3×3 skew-mat
        I = torch.eye(3, dtype=omega.dtype, device=omega.device)

        return I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)

    # ------------------------------------------------------------------
    def geodesic_interpolation_si(self,
                                gamma: torch.Tensor,
                                R: torch.Tensor) -> torch.Tensor:
        """
        λ(gamma, R) = exp(gamma · log R)  — maps SO(3)×ℝ → SO(3)
        """
        return self.exp_map(gamma * self.get_v(self.log_map(R)))

    # ------------------------------------------------------------------
    def tilde_nu_si(self,
                    x_t: torch.Tensor,
                    x_0: torch.Tensor,
                    t: int,
                    alphas: torch.Tensor,
                    alpha_bars: torch.Tensor,
                    betas: torch.Tensor) -> torch.Tensor:
        """
        SE(3) analogue of DDPM’s “posterior mean” for rotations.
        """
        c1 = (torch.sqrt(alpha_bars[t - 1]) * betas[t]) / (1.0 - alpha_bars[t])
        c2 = (torch.sqrt(alphas[t - 1]) * (1.0 - alpha_bars[t - 1])) / (1.0 - alpha_bars[t])

        term1 = self.geodesic_interpolation(c1, x_0)  # (...,3,3)
        term2 = self.geodesic_interpolation(c2, x_t)  # (...,3,3)

        return term1 @ term2
    
    # ----- all helpers now torch --------------------------------------------
    """
    def skew(self, v):                                   # v (...,3)
        vx, vy, vz = v.unbind(-1)           # each has shape (...)

        O = torch.zeros_like(vx)            # same batch shape, scalar entries

        row0 = torch.stack((  O, vz,  -vy), dim=-1)
        row1 = torch.stack(( -vz,   O, vx), dim=-1)
        row2 = torch.stack((vy,  -vx,   O), dim=-1)

        return torch.stack((row0, row1, row2), dim=-2)

    def get_v(self, S):                                  # vee
        return torch.stack([S[..., 2, 1],
                        S[..., 0, 2],
                        S[..., 1, 0]], dim=-1)          # (...,3)

    
    def log_map(self, R):                                # (...,3,3)→(...,3,3)
        tr   = torch.einsum("...ii->...", R)
        cos  = ((tr - 1) / 2).clamp(-1.0, 1.0)
        theta = torch.acos(cos)
        S    = 0.5 * (R.transpose(-2, -1) - R)
        coef = theta / torch.sin(theta + 1e-8)
        coef = torch.where(theta < 1e-4, torch.ones_like(coef), coef)
        return coef[..., None, None] * S

    def exp_map(self, omegas: torch.Tensor) -> torch.Tensor:
        """ 
        """
        Vectorized SO(3) exponential map.
        Args
            omegas : (..., 3)  axis-angle tensor (1-D or N-D)
        Returns
            (..., 3, 3) rotation matrices
        """
        """
        # Ensure a batch dimension for iteration when ndim == 1
        if omegas.ndim == 1:
            omegas = omegas.unsqueeze(0)   # (1,3)

        # Loop over leading dimension(s) – cheap for small B
        R_list = [self.exp_map_single(omega) for omega in omegas]  # len == B
        R = torch.stack(R_list, dim=0)      # (B,3,3)

        # Squeeze back if caller passed a single vector (optional)
        return R
    
    """
        
    def get_v(self, skew: torch.Tensor) -> torch.Tensor:
        vec = torch.zeros_like(skew[..., 0])
        vec[..., 0] = skew[..., 2, 1]
        vec[..., 1] = -skew[..., 2, 0]
        vec[..., 2] = skew[..., 1, 0]
        return vec


    def skew(self, vec: torch.Tensor) -> torch.Tensor:
        skew = torch.repeat_interleave(torch.zeros_like(vec).unsqueeze(-1), 3, dim=-1)
        skew[..., 2, 1] = vec[..., 0]
        skew[..., 2, 0] = -vec[..., 1]
        skew[..., 1, 0] = vec[..., 2]
        return skew - skew.transpose(-1, -2)

    # ─────────────────────────────────────────────
    #  Batched exponential map  (public API)
    # ─────────────────────────────────────────────

    def log_map(self, r_mat: torch.Tensor) -> torch.Tensor:
        skew_mat = (r_mat - r_mat.transpose(-1, -2))
        sk_vec = self.get_v(skew_mat)
        s_angle = (sk_vec).norm(p=2, dim=-1) / 2
        c_angle = (torch.einsum('...ii', r_mat) - 1) / 2
        angle = torch.atan2(s_angle, c_angle)
        scale = (angle / (2 * s_angle))
        # if s_angle = 0, i.e. rotation by 0 or pi (180), we get NaNs
        # by definition, scale values are 0 if rotating by 0.
        # This also breaks down if rotating by pi, fix further down
        scale[angle == 0.0] = 0.0
        log_r_mat = scale[..., None, None] * skew_mat

        # Check for NaNs caused by 180deg rotations.
        nanlocs = log_r_mat[...,0,0].isnan()
        nanmats = r_mat[nanlocs]
        # We need to use an alternative way of finding the logarithm for nanmats,
        # Use eigendecomposition to discover axis of rotation.
        # By definition, these are symmetric, so use eigh.
        # NOTE: linalg.eig() isn't in torch 1.8,
        #       and torch.eig() doesn't do batched matrices
        eigval, eigvec = torch.linalg.eigh(nanmats)
        # Final eigenvalue == 1, might be slightly off because floats, but other two are -ve.
        # this *should* just be the last column if the docs for eigh are true.
        nan_axes = eigvec[...,-1,:]
        nan_angle = angle[nanlocs]
        nan_skew = self.skew(nan_angle[...,None] * nan_axes)
        log_r_mat[nanlocs] = nan_skew
        return log_r_mat


    # -----------------------------------------------------------------------
    def sample_ig_so3(self, sigma):
        v = torch.randn(3, device=self.device) * sigma
        R = torch.as_tensor(Rot.from_rotvec(v.cpu().numpy()).as_matrix(),
                            device=self.device, dtype=torch.float32)
        return v, R

    # ─────────────────────────────────────────────────────────────
    def geodesic_interpolation(self, gamma, R):
        """
        Parameters
        ----------
        gamma : float or torch.Tensor shape (...)      – interpolation weight
        R     : torch.Tensor shape (..., 3, 3)         – rotation matrix

        Returns
        -------
        torch.Tensor shape (..., 3, 3)
        """
        # Ensure gamma is a tensor on the same device / dtype
        gamma = torch.as_tensor(gamma, device=R.device, dtype=R.dtype)

        # Bring gamma to shape (..., 1) so broadcast with v works
        while gamma.dim() < R.dim() - 2:
            gamma = gamma.unsqueeze(-1)

        v = self.get_v(self.log_map(R))          # (..., 3)
        return self.exp_map(gamma * v)           # (..., 3, 3)


    # ─────────────────────────────────────────────────────────────
    #  ν̃ operator used in the diffusion reverse step
    # ─────────────────────────────────────────────────────────────
    def tilde_nu(self, x_t, x_0, t, alphas, alpha_bars, betas):
        """
        Implements the closed-form mean of the posterior q(x_{t-1}|x_t,x_0).

        Parameters
        ----------
        x_t, x_0 : (..., 3, 3)    – noisy rot at t   and clean rot
        t        : int            – current timestep  (1 … T-1)
        alphas,
        alpha_bars,
        betas    : 1-D tensors    – diffusion schedules (on same device)

        Returns
        -------
        torch.Tensor shape (..., 3, 3)   – rotation at timestep t-1
        """
        c1 = (torch.sqrt(alpha_bars[t-1]) * betas[t]) / (1.0 - alpha_bars[t])
        c2 = (torch.sqrt(alphas[t-1]) *
            (1.0 - alpha_bars[t-1])) / (1.0 - alpha_bars[t])

        term1 = self.geodesic_interpolation(c1, x_0)   # (...,3,3)
        term2 = self.geodesic_interpolation(c2, x_t)   # (...,3,3)

        return torch.bmm(term1, term2)                 # (...,3,3)


# In[21]:

class SO3DiffusionMLP(nn.Module):
    def __init__(self, d_model=128, d_out=3):
        super().__init__()
        self.d_out = d_out

        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU()
            
        )
       
        self.mlp = nn.Sequential(
            nn.Linear(9 + d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_out)
        )

        #self.mlp = nn.Linear(9 + d_model, d_out)

    def forward(self, x_t, t):
        B = x_t.shape[0]
        x_flat = x_t.reshape(B, 9) #FLATTEN ROTATION VECTOR
        t = t.view(B, 1).float()
        t_embed = self.time_embed(t)
        inp = torch.cat([x_flat, t_embed], dim=1)
        return self.mlp(inp)


class so3_diffuser:
    # ──────────────────────────────────────────────────────────────────────
    def __init__(self, T, batch_size=64, betas=None, device="cpu"):
        self.device = torch.device(device)
        self.T = T

        # β schedule -------------------------------------------------------
        if betas is None:
            self.betas = self.make_cosine_beta_schedule(T).to(self.device)
        else:
            self.betas = torch.as_tensor(betas, dtype=torch.float32,
                                         device=self.device)

        # α, ᾱ, β̂ --------------------------------------------------------
        self.alphas, self.alpha_bars = self.compute_alpha_bars(self.betas)
        self.beta_hats = self.compute_beta_hat(self.betas, self.alpha_bars)

        self.batch_size = batch_size
        self.alg = SO3Algebra(device=self.device)        # torch-based algebra

        self.model = SO3DiffusionMLP().to(self.device)

    # ──────────────────────────────────────────────────────────────────────


    # 1. schedules now torch ----------------------------------------------
    def make_beta_schedule(self, T, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

    def make_cosine_beta_schedule(self, T, s=0.008):
        steps = torch.arange(T + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s)
                                   * math.pi * 0.5) ** 2
        alphas_cumprod /= alphas_cumprod[0].clone()
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999)


    # 2. ᾱ & β̂ stay as helpers but torch-only ----------------------------
    def compute_alpha_bars(self, betas):
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return alphas, alpha_bars

    def compute_beta_hat(self, betas, alpha_bars):
        beta_hats = torch.zeros_like(betas)
        beta_hats[1:] = ((1 - alpha_bars[:-1]) / (1 - alpha_bars[1:])) \
                        * betas[1:]
        return beta_hats


    # 3. noise generation - torch -----------------------------------------
    def generate_noise(self, t, scale=1.0):
        sigma = scale * torch.sqrt(1 - self.alpha_bars[t-1])

        v_list, R_list = zip(*[self.alg.sample_ig_so3(sigma)
                               for _ in range(self.batch_size)])
        R_noises = torch.stack(R_list)                      # (B,3,3)
        #v_tensor = torch.stack(v_list).float()              # (B,3)
        Sv = torch.stack([self.alg.log_map(R) for R in R_list])  
        v = torch.stack([self.alg.get_v(S) for S in Sv])  
        v_tensor = v.float() 
        return R_noises, v_tensor


    # 4. forward noising ---------------------------------------------------
    def add_noise(self, x, R_noises, t):
        S   = self.alg.log_map(x)                           # (B, 3, 3)
        v   = self.alg.get_v(S) * torch.sqrt(self.alpha_bars[t-1])
        #x_scaled = self.alg.exp_map(v)
        x_scaled = torch.stack([self.alg.exp_map(v_i) for v_i in v]).squeeze(1)     # (B, 3, 3)
        return torch.bmm(R_noises, x_scaled)   


    # 5. reverse step (single) --------------------------------------------
    def _se_sample(self, x_t, t, noise,
                   guidance=False, optim_steps=1, cost=None):
        R = (self.alg.sample_ig_so3(self.beta_hats[t])[1]
             if t > 1 else torch.eye(3, device=self.device))
        print(x_t.shape)
        x_t = x_t.squeeze(0)
        

        v = noise * torch.sqrt(1 - self.alpha_bars[t-1])    # (3,)
        v = v.squeeze(0)
        a1 = self.alg.exp_map_si(
                self.alg.get_v_si(self.alg.log_map_si(x_t))
                / torch.sqrt(self.alpha_bars[t]))
        
        a2 = self.alg.exp_map_si(v / torch.sqrt(1 - self.alpha_bars[t]))
        x_0 = a1 @ a2.transpose(-1, -2)

        x_t = self.alg.tilde_nu_si(x_t, x_0, t,
                                self.alphas, self.alpha_bars, self.betas) @ R
        if guidance and t % optim_steps == 0:
            x_t = self.descent(x_t, x_0, t, cost).detach()
        return x_0, x_t


    # 6. reverse step (batch) ---------------------------------------------
    def _se_sample_batch(self, x_t, t, noise,
                         guidance=False, optim_steps=1, cost=None):
        B, device = x_t.size(0), x_t.device
        
        # Rₜ noise
        if t > 1:
            beta_hat = self.beta_hats[t]
            R_noise = torch.stack(
                [self.alg.sample_ig_so3(beta_hat)[1] for _ in range(B)])
        else:
            R_noise = torch.eye(3, device=device).repeat(B,1,1)

        v_scaled = noise * torch.sqrt(1 - self.alpha_bars[t-1])

        S   = self.alg.log_map(x_t)
        omega = self.alg.get_v(S)
        a1  = self.alg.exp_map(omega / torch.sqrt(self.alpha_bars[t]))
        a2  = self.alg.exp_map(v_scaled / torch.sqrt(1 - self.alpha_bars[t]))
        x0  = torch.bmm(a1, a2.transpose(1,2))

        xtm1 = self.alg.tilde_nu(x_t, x0, t,
                                 self.alphas, self.alpha_bars, self.betas)

        xtm1 = torch.bmm(xtm1, R_noise)

        if guidance and t % optim_steps == 0:
            xtm1 = self.descent(xtm1, x0, t, cost).detach()

        return x0, xtm1
    
    def batch_so3(self, R_clean, epoch=None, scale=1.0):
        self.model.train()
        B = R_clean.size(0)

        # ---- 1. sample a single diffusion step t ----------------------------
        t = torch.randint(1, self.T, (1,), device=self.device).item()

        R_noise, v_noise = self.so3.generate_noise(t, scale)

        # broadcast the one noise sample across the whole batch
        R_noise = R_noise.expand(B, 3, 3)       # (B,3,3)
        v_noise = v_noise.expand(B, 3)          # (B,3)

        R_t = self.so3.add_noise(R_clean, R_noise, t)          # (B,3,3)

        t_tensor = torch.full((B, 1), t, device=self.device)   # (B,1)
        v_pred   = self.model(R_t, t_tensor)                   # (B,3)

        if epoch is not None and epoch % 20_000 == 0:
            print(f"[epoch {epoch}]  true_noise = {v_noise[0].cpu().numpy()}")
            print(f"[epoch {epoch}]  pred_noise = {v_pred[0].detach().cpu().numpy()}")

        loss = 0.5 * ((v_pred - v_noise) ** 2).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        return loss.item()
            
    def train(self, distribution, num_iters, lr = 0.001 , scale = 1, batch_size=64, plot_losses = True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
        losses = []
    
        for i in range(num_iters):
            if(i % 10000 == 0):
                print("Iteration: ", i)
            if(self.continuous): 
                loss, v_tensor, v_pred = self.iteration(distribution = None, distribution_func = distribution)
            else: 
                loss, v_tensor, v_pred = self.iteration(distribution)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if loss.item() > 100:
                continue
            losses.append(loss.item())
    
        if(plot_losses): 
            plt.plot(losses)
            plt.title("Training Loss over Iterations")
            plt.xlabel("Iteration")
            plt.ylabel("MSE Loss")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
        return self.model, losses
        
    def descent(self, x_t, x_0, t, cost, num_updates = 1, lr = 1e-3): 
        x_0_optim = geoopt.ManifoldParameter(x_0, manifold=Stiefel)
        x_t_optim = geoopt.ManifoldParameter(x_t, manifold=Stiefel)
        for i in range(num_updates): 
            loss = cost(x_t_optim, x_0_optim, t)
            loss.backward()
            rgrad_x_t = Stiefel.egrad2rgrad(x_t_optim, x_t_optim.grad)
    
            with(torch.no_grad()): 
                x_t_optim.set_(Stiefel.retr(x_t_optim, -lr * rgrad_x_t))
    
            x_t_optim.grad.zero_()
        return x_t_optim
        

    def sample(self, guidance = False, optim_steps = 1, cost = None):
        x_t = Rot.random().as_matrix()
    
        for t in range(self.T-1, 0, -1):
            if t > 1:
                beta_hat = self.beta_hats[t]
                v_noise, R = self.alg.sample_ig_so3(beta_hat)
                R = R.squeeze(0)
            else:
                R = np.eye(3)
    
            x_t_tensor = torch.tensor(x_t, dtype=torch.float32).unsqueeze(0)
            t_tensor = torch.tensor([t], dtype=torch.float32)
            v_tensor = self.model(x_t_tensor, t_tensor) * np.sqrt(1 - self.alpha_bars[t - 1]) #I dont train the model to predict the normalized noise. Training seemed to stabilize with this change, so I correct that with this multiplication
    
            v = v_tensor[0].detach().numpy().flatten()
    
            a1 = self.alg.exp_map(self.alg.get_v(self.alg.log_map(x_t)) / np.sqrt(self.alpha_bars[t]))
            a2 = self.alg.exp_map(v / np.sqrt(1 - self.alpha_bars[t]))
    
    
            x_0 = a1 @ a2.T
            
            x_t = self.alg.tilde_nu(x_t, x_0, t, self.alphas, self.alpha_bars, self.betas) @ R
            if(guidance and t % optim_steps == 0):
                x_t = self.descent(x_t, x_0, t, cost).data.numpy()
    
        return x_0, x_t


# In[ ]:




