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
#from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
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
        self.sampler = IGSO3Sampler()
        #self.exp_map = torch.linalg.matrix_exp

    def exp_map(self, omega: torch.Tensor) -> torch.Tensor: 
        return torch.linalg.matrix_exp(self.skew(omega))
        
    #Code from leach et al
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

    def sample_ig_so3(self, sigma: torch.Tensor | float, n_samples: int = 1):
        """
        Draw n_samples rotations from IGSO(3, σ) and return
        (rotvecs, R) with shapes (n,3) and (n,3,3)
        """
        sigma_val = float(torch.as_tensor(sigma))
        rotvec_np = self.sampler.sample_vector(sigma_val, n_samples)

        # torch-ify the axis–angle batch
        rotvec = torch.as_tensor(rotvec_np, dtype=torch.float32, device=self.device)  # (n,3)

        # use your algebra’s batched exponential map
        R = self.exp_map(rotvec)   # shape (n,3,3)
        return rotvec, R

    
    def sample_ig_so3_approx(self, sigma):
        v = torch.randn(3, device=self.device) * sigma
        R = torch.as_tensor(Rot.from_rotvec(v.cpu().numpy()).as_matrix(),
                            device=self.device, dtype=torch.float32)
        return v, R

    def geodesic_interpolation(self, gamma, R):
        """
        Gamma is a scalar,
        R is of shape [N,3,3], broadcasts gamma to all rotation matrices. 
        """
        gamma = torch.as_tensor(gamma, device=R.device, dtype=R.dtype)

        while gamma.dim() < R.dim() - 2:
            gamma = gamma.unsqueeze(-1)

        v = self.get_v(self.log_map(R))          # (..., 3)
        return self.exp_map(gamma * v)           # (..., 3, 3)


    def tilde_nu(self, x_t, x_0, t, alphas, alpha_bars, betas):
        """
        Args:
        x_t (Tensor): Noisy rotations at timestep t, shape [N, 3, 3]
        x_0 (Tensor): Clean rotations, shape [N, 3, 3]
        t (int): Timestep
        alphas, alpha_bars, betas: Arrays/lists of scalars, indexable by timestep

        Returns:
            Tensor of shape [N, 3, 3]: result of batched matrix multiplication of interpolated geodesics
        """
        c1 = (torch.sqrt(alpha_bars[t-1]) * betas[t]) / (1.0 - alpha_bars[t])
        c2 = (torch.sqrt(alphas[t-1]) *
            (1.0 - alpha_bars[t-1])) / (1.0 - alpha_bars[t])

        term1 = self.geodesic_interpolation(c1, x_0)   # (...,3,3)
        term2 = self.geodesic_interpolation(c2, x_t)   # (...,3,3)

        return torch.bmm(term1, term2)                 # (...,3,3)

    def move_to_np(t: torch.Tensor):
        return t.detach().cpu().numpy()

    def igso3_expansion(omega, eps, L=1000, use_torch=False):
        """Truncated sum of IGSO(3) distribution.

        This function approximates the power series in equation 5 of
        "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
        ALIGNMENT"
        Leach et al. 2022

        This expression diverges from the expression in Leach in that here, eps =
        sqrt(2) * eps_leach, if eps_leach were the scale parameter of the IGSO(3).

        With this reparameterization, IGSO(3) agrees with the Brownian motion on
        SO(3) with t=eps^2.

        Args:
            omega: rotation of Euler vector (i.e. the angle of rotation)
            eps: std of IGSO(3).
            L: Truncation level
            use_torch: set true to use torch tensors, otherwise use numpy arrays.
        """

        lib = torch if use_torch else np
        ls = lib.arange(L)
        if use_torch:
            ls = ls.to(omega.device)
        if len(omega.shape) == 2:
            # Used during predicted score calculation.
            ls = ls[None, None]  # [1, 1, L]
            omega = omega[..., None]  # [num_batch, num_res, 1]
            eps = eps[..., None]
        elif len(omega.shape) == 1:
            # Used during cache computation.
            ls = ls[None]  # [1, L]
            omega = omega[..., None]  # [num_batch, 1]
        else:
            raise ValueError("Omega must be 1D or 2D.")
        p = (2*ls + 1) * lib.exp(-ls*(ls+1)*eps**2/2) * lib.sin(omega*(ls+1/2)) / lib.sin(omega/2)
        if use_torch:
            return p.sum(dim=-1)
        else:
            return p.sum(axis=-1)


    def density(expansion, omega, marginal=True):
        """IGSO(3) density.

        Args:
            expansion: truncated approximation of the power series in the IGSO(3)
            density.
            omega: length of an Euler vector (i.e. angle of rotation)
            marginal: set true to give marginal density over the angle of rotation,
                otherwise include normalization to give density on SO(3) or a
                rotation with angle omega.
        """
        if marginal:
            # if marginal, density over [0, pi], else over SO(3)
            return expansion * (1-np.cos(omega))/np.pi
        else:
            # the constant factor doesn't affect any actual calculations though
            return expansion / 8 / np.pi**2


    def score(exp, omega, eps, L=1000, use_torch=False):  # score of density over SO(3)
        """score uses the quotient rule to compute the scaling factor for the score
        of the IGSO(3) density.

        This function is used within the Diffuser class to when computing the score
        as an element of the tangent space of SO(3).

        This uses the quotient rule of calculus, and take the derivative of the
        log:
            d hi(x)/lo(x) = (lo(x) d hi(x)/dx - hi(x) d lo(x)/dx) / lo(x)^2
        and
            d log expansion(x) / dx = (d expansion(x)/ dx) / expansion(x)

        Args:
            exp: truncated expansion of the power series in the IGSO(3) density
            omega: length of an Euler vector (i.e. angle of rotation)
            eps: scale parameter for IGSO(3) -- as in expansion() this scaling
                differ from that in Leach by a factor of sqrt(2).
            L: truncation level
            use_torch: set true to use torch tensors, otherwise use numpy arrays.

        Returns:
            The d/d omega log IGSO3(omega; eps)/(1-cos(omega))

        """

        lib = torch if use_torch else np
        ls = lib.arange(L)
        if use_torch:
            ls = ls.to(omega.device)
        ls = ls[None]
        if len(omega.shape) == 2:
            ls = ls[None]
        elif len(omega.shape) > 2:
            raise ValueError("Omega must be 1D or 2D.")
        omega = omega[..., None]
        eps = eps[..., None]
        hi = lib.sin(omega * (ls + 1 / 2))
        dhi = (ls + 1 / 2) * lib.cos(omega * (ls + 1 / 2))
        lo = lib.sin(omega / 2)
        dlo = 1 / 2 * lib.cos(omega / 2)
        dSigma = (2 * ls + 1) * lib.exp(-ls * (ls + 1) * eps**2/2) * (lo * dhi - hi * dlo) / lo ** 2
        if use_torch:
            dSigma = dSigma.sum(dim=-1)
        else:
            dSigma = dSigma.sum(axis=-1)
        return dSigma / (exp + 1e-4)
    
    def score(
            self,
            vec: np.ndarray,
            t: float,
            eps: float=1e-6
        ):
        """Computes the score of IGSO(3) density as a rotation vector.

        Args:
            vec: [..., 3] array of axis-angle rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            [..., 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        torch_score = self.torch_score(torch.tensor(vec), torch.tensor(t)[None])
        return torch_score.numpy()

    def torch_score(
            self,
            vec: torch.tensor,
            t: torch.tensor,
            eps: float=1e-6,
        ):
        """Computes the score of IGSO(3) density as a rotation vector.

        Same as score function but uses pytorch and performs a look-up.

        Args:
            vec: [..., 3] array of axis-angle rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            [..., 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        omega = torch.linalg.norm(vec, dim=-1) + eps
        if self.use_cached_score:
            score_norms_t = self._score_norms[self.t_to_idx(du.move_to_np(t))]
            score_norms_t = torch.tensor(score_norms_t).to(vec.device)
            omega_idx = torch.bucketize(
                omega, torch.tensor(self.discrete_omega[:-1]).to(vec.device))
            omega_scores_t = torch.gather(
                score_norms_t, 1, omega_idx)
        else:
            sigma = self.discrete_sigma[self.t_to_idx(du.move_to_np(t))]
            sigma = torch.tensor(sigma).to(vec.device)
            omega_vals = igso3_expansion(omega, sigma[:, None], use_torch=True)
            omega_scores_t = score(omega_vals, omega, sigma[:, None], use_torch=True)
        return omega_scores_t[..., None] * vec / (omega[..., None] + eps)

    def score_scaling(self, t: np.ndarray):
        """Calculates scaling used for scores during trianing."""
        return self._score_scaling[self.t_to_idx(t)]

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
    def __init__(self, T, batch_size=64, betas=None, device="cpu"):
        self.device = torch.device(device)
        self.T = T

        if betas is None:
            self.betas = self.make_cosine_beta_schedule(T).to(self.device)
        else:
            self.betas = torch.as_tensor(betas, dtype=torch.float32,
                                         device=self.device)

        self.alphas, self.alpha_bars = self.compute_alpha_bars(self.betas)
        self.beta_hats = self.compute_beta_hat(self.betas, self.alpha_bars)

        self.batch_size = batch_size
        self.alg = SO3Algebra(device=self.device)       

        self.model = SO3DiffusionMLP().to(self.device)

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

    def generate_noise(self, t, B, N, scale=1.0):
        """
        Generate SO(3)^N noise for B samples.

        Returns:
            R_noises: [B, N, 3, 3]  — rotation matrices
            v_tensor: [B, N, 3]     — corresponding axis-angle vectors
        """
        sigma = scale * torch.sqrt(1 - self.alpha_bars[t])

        # Fully batched sampling
        v_tensor_flat, R_tensor_flat = self.alg.sample_ig_so3(sigma, n_samples=B * N)
        v_tensor = v_tensor_flat.view(B, N, 3)
        R_tensor = R_tensor_flat.view(B, N, 3, 3)

        return R_tensor, v_tensor

    def add_noise(self, x, R_noises, t):
        """
        Applies multiplicative SO(3) noise to a batch of rotations x using noise R_noises.

        Args:
            x         : [B, N, 3, 3] — clean rotations
            R_noises  : [B, N, 3, 3] — sampled noise rotations
            t         : scalar timestep

        Returns:
            x_noisy   : [B, N, 3, 3] — noisy rotations
        """
        B, N = x.shape[:2]
        x_flat = x.view(-1, 3, 3)             # [B*N, 3, 3]
        R_flat = R_noises.view(-1, 3, 3)      # [B*N, 3, 3]

        # Apply scaled log map and reproject
        S = self.alg.log_map(x_flat)                     # [B*N, 3, 3]
        v = self.alg.get_v(S) * torch.sqrt(self.alpha_bars[t])  # [B*N, 3]
        x_scaled = self.alg.exp_map(v)                   # [B*N, 3, 3]

        # Apply noise via matrix multiplication: R * x_scaled
        x_noisy_flat = torch.bmm(R_flat, x_scaled)       # [B*N, 3, 3]
        return x_noisy_flat.view(B, N, 3, 3) 

    def _se_sample_batch(self, x_t, t, noise, guidance=False, optim_steps=1, cost=None):
        """
        Perform one SE(3) sampling step for N rotation matrices.

        Args:
            x_t     : [N, 3, 3] — current noisy rotations
            t       : int       — timestep
            noise   : [N, 3]    — predicted axis-angle noise
            guidance: bool      — whether to apply descent
            optim_steps: int    — descent frequency
            cost    : any       — optional guidance cost

        Returns:
            x0      : [N, 3, 3] — predicted clean rotations
            xtm1    : [N, 3, 3] — sampled R_{t-1}
        """
        N, device = x_t.shape[1], x_t.device

        # Rₜ noise
        if t > 1:
            beta_hat = self.beta_hats[t]
            _, R_noise = self.alg.sample_ig_so3(beta_hat, n_samples=N)  # [N, 3, 3]
        else:
            R_noise = torch.eye(3, device=device).expand(N, 3, 3)        # [N, 3, 3]

        # Reconstruct x₀ = exp(ω / sqrt(ᾱₜ)) · exp(√(1−ᾱₜ) · v)
        v_scaled = noise * torch.sqrt(1 - self.alpha_bars[t])           # [N, 3]
        x_t = x_t.squeeze(0)
        v_scaled = v_scaled.squeeze(0)
        omega = self.alg.get_v(self.alg.log_map(x_t))                   # [N, 3]
        a1 = self.alg.exp_map(omega / torch.sqrt(self.alpha_bars[t]))  # [N, 3, 3]
        a2 = self.alg.exp_map(v_scaled)                                 # [N, 3, 3]
        x0 = torch.bmm(a1, a2.transpose(1, 2))  # [B*N, 3, 3]

        xtm1 = self.alg.tilde_nu(x_t, x0, t,
                                self.alphas, self.alpha_bars, self.betas)  # [N, 3, 3]
        xtm1 = torch.bmm(xtm1, R_noise)                                    # [N, 3, 3]
        
        # Optional guidance
        if guidance and t % optim_steps == 0:
            xtm1 = self.descent(xtm1, x0, t, cost).detach()                # [N, 3, 3]

        return xtm1.unsqueeze(0), x0.unsqueeze(0)


    def reverse(
                self,
                rot_t: np.ndarray,
                score_t: np.ndarray,
                t: float,
                dt: float,
                mask: np.ndarray=None,
                noise_scale: float=1.0,
                ):
            """Simulates the reverse SDE for 1 step using the Geodesic random walk.

            Args:
                rot_t: [..., 3] current rotations at time t.
                score_t: [..., 3] rotation score at time t.
                t: continuous time in [0, 1].
                dt: continuous step size in [0, 1].
                add_noise: set False to set diffusion coefficent to 0.
                mask: True indicates which residues to diffuse.

            Returns:
                [..., 3] rotation vector at next step.
            """
            if not np.isscalar(t): raise ValueError(f'{t} must be a scalar.')

            g_t = self.diffusion_coef(t)
            z = noise_scale * np.random.normal(size=score_t.shape)
            perturb = (g_t ** 2) * score_t * dt + g_t * np.sqrt(dt) * z

            if mask is not None: perturb *= mask[..., None]
            n_samples = np.cumprod(rot_t.shape[:-1])[-1]

            # Right multiply.
            rot_t_1 = du.compose_rotvec(
                rot_t.reshape(n_samples, 3),
                perturb.reshape(n_samples, 3)
            ).reshape(rot_t.shape)
            return rot_t_1
        
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

    def score_from_pair(
        self,
        rot0: torch.Tensor,    # (..., 3) clean rotation (axis–angle)
        rot_t: torch.Tensor,   # (..., 3) noisy rotation (axis–angle)
        t: torch.Tensor,       # scalar in [0, 1] or shape (1,)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Computes IGSO(3) score from a clean/noisy pair:
            s(Δ),  where Δ = rot0 ∘ rot_t⁻¹

        Args:
            rot0 : (..., 3) clean rotation in axis–angle
            rot_t: (..., 3) noisy rotation in axis–angle
            t    : scalar float in [0, 1]
            eps  : small number for stability

        Returns:
            (..., 3) score vector
        """
        # Ensure inputs are torch tensors
        assert rot0.shape == rot_t.shape
        assert rot0.shape[-1] == 3
        assert torch.is_tensor(t), "t should be a torch scalar or (1,) tensor"

        # Convert t to shape (1,) if needed
        if t.ndim == 0:
            t = t[None]

        # Relative rotation vector: Δ = rot0 ∘ rot_t⁻¹
        # Axis–angle inverse is just negation
        delta = du.compose_rotvec(rot0, -rot_t)  # (..., 3)

        # Now apply your existing score computation
        return self.torch_score(delta, t, eps=eps)  # (..., 3)

    def compose_rotvec(a: torch.Tensor, b: torch.Tensor, fast: bool = False) -> torch.Tensor:
        """
        Compose two axis-angle rotations:
            R(a) · R(b)  →  axis-angle
        Args:
            a, b : (..., 3) axis–angle vectors
            fast : whether to use fast approximate mode for axis_angle_to_matrix
        Returns:
            (..., 3) axis–angle vector representing composed rotation
        """
        Ra = axis_angle_to_matrix(a, fast=fast)   # (..., 3, 3)
        Rb = axis_angle_to_matrix(b, fast=fast)   # (..., 3, 3)
        Rab = Ra @ Rb                             # composed rotation
        return matrix_to_axis_angle(Rab)
            


# In[ ]:




