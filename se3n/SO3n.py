#!/usr/bin/env python
# coding: utf-8

# In[8]:
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
import math
import torch.nn as nn
import math
import os, logging
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

    def igso3_expansion(self, omega, eps, L=1000, use_torch=False):
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

    def density(self, expansion, omega, marginal=True):
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

    def sample_ig_so3(self, sigma: torch.Tensor | float = None, n_samples: int = 1, sigmas: torch.Tensor = None):
        """
        Draw n_samples rotations from IGSO(3, σ)

        If `sigmas` is provided (shape [n]), sample one rotation per sigma.
        Otherwise, sample `n_samples` from IGSO(3, sigma).

        Returns:
            rotvecs: Tensor of shape (n, 3)
            R:       Tensor of shape (n, 3, 3)
        """
        if sigmas is not None:
            # Per-sample σ
            rotvecs = []
            Rs = []
            for sigma_val in sigmas:
                sigma_float = float(sigma_val)
                vec_np = self.sampler.sample_vector(sigma_float, 1)  # shape (1, 3)
                vec = torch.as_tensor(vec_np, dtype=torch.float32, device=self.device)
                R = self.exp_map(vec)  # shape (1, 3, 3)
                rotvecs.append(vec)
                Rs.append(R)
            rotvecs = torch.cat(rotvecs, dim=0)  # (n, 3)
            Rs = torch.cat(Rs, dim=0)            # (n, 3, 3)
        else:
            # Single σ for all
            sigma_val = float(torch.as_tensor(sigma))
            rotvec_np = self.sampler.sample_vector(sigma_val, n_samples)  # (n_samples, 3)
            rotvecs = torch.as_tensor(rotvec_np, dtype=torch.float32, device=self.device)
            Rs = self.exp_map(rotvecs)  # (n_samples, 3, 3)

        return rotvecs, Rs

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


    def tilde_nu(self, x_t, x_0, t, alphas, alpha_bars, betas, equivariance = False):
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
        if(equivariance): 
            term1 = self.geodesic_interpolation(c1, x_0)   # (...,3,3)
            term2 = self.geodesic_interpolation(c2 / 2, x_t)   # (...,3,3)
            xtm1 = torch.bmm(term2,term1)
            xtm1 = torch.bmm(xtm1, term2)
            return xtm1
        else:
            term1 = self.geodesic_interpolation(c1, x_0)   # (...,3,3)
            term2 = self.geodesic_interpolation(c2, x_t)   # (...,3,3)

        return torch.bmm(term1, term2)                 # (...,3,3)
    

    def make_unit_quaternion(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes a [B, 3] tensor, appends a 1 to form a [B, 4] quaternion, and normalizes it.

        Args:
            x: Tensor of shape [B, 3]

        Returns:
            Tensor of shape [B, 4] representing unit quaternions
        """
        assert x.ndim == 2 and x.shape[1] == 3, "Input must be of shape [B, 3]"
        
        ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        quat = torch.cat([x, ones], dim=1)  # [B, 4]
        quat = quat / quat.norm(dim=1, keepdim=True)  # Normalize to unit quaternion
        return quat
    
    def quaternion_to_rotmat(self, q: torch.Tensor) -> torch.Tensor:
        """
        Converts unit quaternions [B, 4] -> rotation matrices [B, 3, 3]
        Assumes quaternion format is (x, y, z, w), where w is the scalar.

        Args:
            q: Tensor of shape [B, 4], each row is a unit quaternion

        Returns:
            Tensor of shape [B, 3, 3] rotation matrices
        """
        assert q.ndim == 2 and q.shape[1] == 4, "Input must be of shape [B, 4]"

        x, y, z, w = q.unbind(dim=-1)

        B = q.shape[0]
        R = torch.empty(B, 3, 3, device=q.device, dtype=q.dtype)

        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x*y - z*w)
        R[:, 0, 2] = 2 * (x*z + y*w)

        R[:, 1, 0] = 2 * (x*y + z*w)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y*z - x*w)

        R[:, 2, 0] = 2 * (x*z - y*w)
        R[:, 2, 1] = 2 * (y*z + x*w)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        return R
    
    def quaternion_to_matrix(quaternions):
        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))



    def move_to_np(self, t: torch.Tensor):
        return t.detach().cpu().numpy()

    def score(self, exp, omega, eps, L=1000, use_torch=False):  # score of density over SO(3)
        lib = torch if use_torch else np
        ls = lib.arange(L)
        if use_torch:
            ls = ls.to(omega.device)

        # Shape handling to avoid extra broadcast dims
        if omega.ndim == 2:
            # omega: [B,N] -> add last dim; ls: [1,1,L]; eps: [B,N,1]
            ls    = ls[None, None, :]         # [1,1,L]
            omega = omega[..., None]          # [B,N,1]
            if eps.ndim == 1:
                eps = eps[:, None]            # [B*N,1] unlikely here, but safe
            eps   = eps[..., None]            # [B,N,1]
        elif omega.ndim == 1:
            # omega: [M] -> add last dim; ls: [1,L]; eps: [M,1]
            ls    = ls[None, :]               # [1,L]
            omega = omega[..., None]          # [M,1]
            if eps.ndim == 1:
                eps = eps[:, None]            # [M,1]
            # IMPORTANT: do NOT add another None-dim here
        else:
            raise ValueError("Omega must be 1D or 2D.")

        hi  = lib.sin(omega * (ls + 1/2))                    # [..., L]
        dhi = (ls + 1/2) * lib.cos(omega * (ls + 1/2))       # [..., L]
        lo  = lib.sin(omega / 2)                             # [..., 1]
        dlo = 0.5 * lib.cos(omega / 2)                       # [..., 1]

        expo   = lib.exp(-ls * (ls + 1) * (eps**2) / 2)      # [..., L]
        dSigma = (2*ls + 1) * expo * (lo * dhi - hi * dlo) / (lo ** 2)  # [..., L]

        if use_torch:
            dSigma = dSigma.sum(dim=-1)                      # [...,]
        else:
            dSigma = dSigma.sum(axis=-1)                     # [...,]

        return dSigma / (exp + 1e-4)                         # same shape as exp
  

# In[21]:

class so3_diffuser:
    def __init__(self, prediction, T, representation = 'quat', forward_process = "ve", batch_size=64, betas=None, device="cpu", cfg = None):
        self.device = torch.device(device)
        self.T = T
        self.representation = representation
        self.prediction = prediction
        self.forward_process = forward_process
        self.alg = SO3Algebra(device=self.device)      
        if(forward_process == "ve"): 
            self.schedule         = cfg.get("schedule", "logarithmic")
            self.min_sigma        = float(cfg.get("min_sigma", 0.1))
            self.max_sigma        = float(cfg.get("max_sigma",  1.5))
            self.num_sigma        = int(cfg.get("num_sigma", 1000))
            self.use_cached_score = bool(cfg.get("use_cached_score", False))
            self.num_omega        = int(cfg.get("num_omega", 2048))
            self.cache_dir =  cfg.get("cache_dir")
            self._log             = logging.getLogger(__name__)
            self._log = logging.getLogger(__name__)

            # Discretize omegas for calculating CDFs. Skip omega=0.
            self.discrete_omega = np.linspace(0, np.pi, self.num_omega+1)[1:]

            # Precompute IGSO3 values.
            replace_period = lambda x: str(x).replace('.', '_')
            cache_dir = os.path.join(
                self.cache_dir,
                f'eps_{self.num_sigma}_omega_{self.num_omega}_min_sigma_{replace_period(self.min_sigma)}_max_sigma_{replace_period(self.max_sigma)}_schedule_{self.schedule}'
            )

            # If cache directory doesn't exist, create it
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)
            pdf_cache = os.path.join(cache_dir, 'pdf_vals.npy')
            cdf_cache = os.path.join(cache_dir, 'cdf_vals.npy')
            score_norms_cache = os.path.join(cache_dir, 'score_norms.npy')

            if os.path.exists(pdf_cache) and os.path.exists(cdf_cache) and os.path.exists(score_norms_cache):
                self._log.info(f'Using cached IGSO3 in {cache_dir}')
                self._pdf = np.load(pdf_cache)
                self._cdf = np.load(cdf_cache)
                self._score_norms = np.load(score_norms_cache)
            else:
                self._log.info(f'Computing IGSO3. Saving in {cache_dir}')
                # compute the expansion of the power series
                exp_vals = np.asarray(
                    [self.alg.sampler.igso3_expansion(self.discrete_omega, sigma) for sigma in self.discrete_sigma])
                # Compute the pdf and cdf values for the marginal distribution of the angle
                # of rotation (which is needed for sampling)
                self._pdf  = np.asarray(
                    [self.alg.sampler.density(x, self.discrete_omega, marginal=True) for x in exp_vals])
                self._cdf = np.asarray(
                    [pdf.cumsum() / self.num_omega * np.pi for pdf in self._pdf])

                # Compute the norms of the scores.  This are used to scale the rotation axis when
                # computing the score as a vector.
                self._score_norms = np.asarray(
                    [self.alg.score(exp_vals[i], self.discrete_omega, x) for i, x in enumerate(self.discrete_sigma)])

                # Cache the precomputed values
                np.save(pdf_cache, self._pdf)
                np.save(cdf_cache, self._cdf)
                np.save(score_norms_cache, self._score_norms)
            self.discrete_omega = torch.from_numpy(self.discrete_omega).to(self.device).float()   # [O]
            self._pdf          = torch.from_numpy(self._pdf).to(self.device).float()              # [S, O]
            self._cdf          = torch.from_numpy(self._cdf).to(self.device).float()              # [S, O]
            self._score_norms  = torch.from_numpy(self._score_norms).to(self.device).float()      # [S, O or O-1]

            num = (self._score_norms ** 2 * self._pdf).sum(dim=-1)                     # [S]
            den = self._pdf.sum(dim=-1).clamp_min(1e-12)                               # [S]
            self._score_scaling = torch.sqrt(torch.abs(num / den)) / math.sqrt(3.0)    # [S]

        self.batch_size = batch_size

    def make_betas_logsigma(
        self,
        T: int,
        clamp_min: float = 1e-8,
        clamp_max: float = 0.999,
    ) -> torch.Tensor:
        """
        Discretize the SO(3) log-σ schedule into betas[1..T].

            σ(t) = log( (1 - t) * e^{σ_min} + t * e^{σ_max} ),  t in [0,1].
        For VP-style corruption:
            m(t) = σ(t)^2 - σ_min^2   (so ᾱ(t) = exp(-m(t)) and m(0)=0).

        Discretizations:
        - 'right':     β_k = 1 - exp( -[m(t_k) - m(t_{k-1})] ), exact on the grid.
        - 'midpoint':  β_k ≈ m'(t_{k-1/2}) / T,
                        with m'(t) = 2 σ(t) σ'(t), σ'(t) = B / (A + B t).

        Returns:
        betas: torch.float32 tensor of shape [T] on self.device.
        """
        device = self.device
        dtype = torch.float32

        # Constants for σ(t) = log(A + B t)
        A = math.exp(self.min_sigma)
        B = math.exp(self.max_sigma) - A

        def sigma_t(t: torch.Tensor) -> torch.Tensor:
            return torch.log(A + B * t)

        # Grid t_k = k/T, k=0..T; exact ᾱ(t_k) matching via m(t_k)
        t = torch.linspace(0.0, 1.0, steps=T + 1, device=device, dtype=dtype)
        sig = sigma_t(t)                        # [T+1]
        m = sig * sig - (self.min_sigma ** 2)   # [T+1]
        dm = m[1:] - m[:-1]                     # [T]
        betas = 1.0 - torch.exp(-dm)            # [T]
        # Numerical safety
        betas = betas.clamp(clamp_min, clamp_max)
        return betas.to(device=device, dtype=dtype)
    
    @property
    def discrete_sigma(self):
        return self.sigma(
            np.linspace(0.0, 1.0, self.num_sigma)
        )

    def sigma_idx(self, sigma: np.ndarray):
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        return np.digitize(sigma, self.discrete_sigma) - 1

    def sigma(self, t: np.ndarray):
        """Extract \sigma(t) corresponding to chosen sigma schedule."""
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        if self.schedule == 'logarithmic':
            return np.log(t * np.exp(self.max_sigma) + (1 - t) * np.exp(self.min_sigma))
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')

    def diffusion_coef(self, t):
        """Torch-only diffusion coefficient g(t) for the logarithmic σ schedule.

        Args:
            t: torch.Tensor (any shape) with values in [0, 1]

        Returns:
            torch.Tensor g(t) with the same shape as t
        """
        t = torch.as_tensor(t, device=self.device, dtype=torch.float32).clamp(0.0, 1.0)

        # σ(t) = log((1−t) e^{σ_min} + t e^{σ_max}) = log(A + B t)
        A = math.exp(self.min_sigma)
        B = math.exp(self.max_sigma) - A

        sigma_t = torch.log(A + B * t)                      # σ(t)
        g2 = 2.0 * B * sigma_t / torch.exp(sigma_t)         # g^2(t)
        g  = torch.sqrt(torch.clamp(g2, min=1e-20))
        return g

    def t_to_idx(self, t):
        """Map t in [0,1] (scalar, [B], or [B,N]) to indices into self.discrete_sigma (torch)."""
        A = math.exp(self.min_sigma)
        B = math.exp(self.max_sigma) - A

        t = torch.as_tensor(t, device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        sigma_t = torch.log(A + B * t)  # same shape as t

        ds = torch.as_tensor(self.discrete_sigma, device=self.device, dtype=torch.float32)  # [S]
        idx = torch.bucketize(sigma_t.reshape(-1), ds, right=True) - 1                      # [...]
        idx = idx.clamp(0, ds.numel() - 1).long()
        return idx.view_as(sigma_t)  # same shape as t

    def sample_igso3(self, t, n_samples: int = 1):
        """
        Invert the cached IGSO(3) CDF at time(s) t.
        - t scalar  -> returns [n_samples]
        - t shape [B] -> returns [B, n_samples]
        """
        device = self.device
        cdf_all = torch.as_tensor(self._cdf, device=device, dtype=torch.float32)          # [S, O]
        omega_g = torch.as_tensor(self.discrete_omega, device=device, dtype=torch.float32) # [O]

        idx = self.t_to_idx(t)  # scalar long or [B]
        if idx.ndim == 0:
            cdf_row = cdf_all[idx]                              # [O]
            u = torch.rand(n_samples, device=device)            # [n]
            bins = torch.bucketize(u, cdf_row).clamp(1, omega_g.numel() - 1)
            c0, c1 = cdf_row[bins - 1], cdf_row[bins]
            w = (u - c0) / torch.clamp(c1 - c0, min=1e-12)
            omega = omega_g[bins - 1] + w * (omega_g[bins] - omega_g[bins - 1])  # [n]
            return omega
        elif idx.ndim == 1:
            B = idx.shape[0]
            u = torch.rand(B, n_samples, device=device)          # [B,n]
            out = torch.empty(B, n_samples, device=device)
            for b in range(B):
                cdf_row = cdf_all[idx[b]]                        # [O]
                bins = torch.bucketize(u[b], cdf_row).clamp(1, omega_g.numel() - 1)  # [n]
                c0, c1 = cdf_row[bins - 1], cdf_row[bins]
                w = (u[b] - c0) / torch.clamp(c1 - c0, min=1e-12)
                out[b] = omega_g[bins - 1] + w * (omega_g[bins] - omega_g[bins - 1])
            return out
        else:
            raise ValueError("t must be scalar or 1-D [B].")

    def sample(
            self,
            t: float,
            n_samples: float=1):
        """Generates rotation vector(s) from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_sample: number of samples to generate.

        Returns:
            [n_samples, 3] axis-angle rotation vectors sampled from IGSO(3).
        """
        x = np.random.randn(n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x * self.sample_igso3(t, n_samples=n_samples)[:, None]

    def sample_ref(self, n_samples: float=1):
        return self.sample(1, n_samples=n_samples)
    
    def generate_noise(self, t, B, N, scale=1.0):
        if self.forward_process == "ve":
            t = torch.as_tensor(t, device=self.device, dtype=torch.float32).clamp(0.01, 1.0)
            return self.ve_noise_generate(t, B, N)

    #variance exploding method
    def ve_noise_generate(
        self,
        t,                 # torch.FloatTensor [B,N] in [0,1]
        B: int,
        N: int,
        scale: float = 1.0,
        device: str | torch.device = "cpu",
        dtype=torch.float32,
    ):
        t = torch.as_tensor(t, device=self.device, dtype=torch.float32)
        assert t.shape == (B, N), f"t must be [B,N]; got {tuple(t.shape)}"

        idx = self.t_to_idx(t)  # [B,N] long
        ds  = torch.as_tensor(self.discrete_sigma, device=self.device, dtype=torch.float32)
        sigmas = ds[idx].reshape(-1)  # [B*N]

        # sample from
        v_flat, R_flat = self.alg.sample_ig_so3(sigmas=sigmas)  # [B*N,3], [B*N,3,3] (torch)

        v = v_flat.view(B, N, 3).to(device=self.device, dtype=dtype)
        R = R_flat.view(B, N, 3, 3).to(device=self.device, dtype=dtype)
        return R, v

    def add_noise(self, x, R_noises, t): 
        if(self.forward_process == "ve"): 
            return self.ve_forward(x, R_noises, t)
    
    #variance exploding
    def ve_forward(self, x, R_noises, t): 
        x  = x.to(dtype=R_noises.dtype, device=R_noises.device)
        R_t = torch.einsum('...ij,...jk->...ik', x, R_noises)
        return R_t
    
    #variance exploding schema
    @torch.no_grad()
    def _se_sample_ve(self, x_t, t, x_0, noise_scale=0.0, threshold = 0):
        """
            One reverse step on SO(3)^N using score model, batched over B.
            Expects x_t, x_0: [B, N, 3, 3]
        """
        device, dtype = x_t.device, x_t.dtype
        B, N = x_t.shape[0], x_t.shape[1]

        t_cont = torch.as_tensor(t, device=device, dtype=torch.float32)
        dt_cont = 1.0 / float(self.T)

        # rot_t_vec should match x_t batch shape: [B, N, 3]
        rot_t_vec = self.alg.get_v(self.alg.log_map(x_t))   # [B, N, 3]

        # score should match: [B, N, 3]
        s_pred = self.compute_score(x_t, x_0, t_cont)       # [B, N, 3]

        # reverse should return [B, N, 3]
        rot_tm1 = self.reverse(
            rot_t=rot_t_vec,
            score_t=s_pred,
            t=t_cont,            
            dt=dt_cont,
            mask=None,
            noise_scale=noise_scale,
        )                           # [B, N, 3]

        xtm1 = self.alg.exp_map(rot_tm1).to(device=device, dtype=dtype)  # [B, N, 3, 3]
        return xtm1, x_0

    def compose_rotvec(self, a, b):
        Ra = self.alg.exp_map(torch.as_tensor(a))
        Rb = self.alg.exp_map(torch.as_tensor(b))
        Rab = torch.einsum('...ij,...jk->...ik', Ra, Rb)
        return self.alg.get_v(self.alg.log_map(Rab)).cpu().numpy()

            
    def compute_score(self, Rt: torch.Tensor, R0: torch.Tensor, t) -> torch.Tensor:
        """
        Rt: [...,3,3]  current rotations
        R0: [...,3,3] or [...,3] (rotvecs)
        t : scalar or tensor broadcastable to Rt batch (e.g. [B*N] or [B,N])
        """
        Rrel = R0.transpose(-1, -2) @ Rt                 # [B, N, 3, 3]
        r_rel = self.alg.get_v(self.alg.log_map(Rrel))   # [B, N, 3]

        # Make t a torch tensor on the right device; allow scalar / [B] / [B,N]
        t_tensor = torch.as_tensor(t, device=Rt.device, dtype=torch.float32)

        # torch_score expects t broadcastable to omega shape ([B,N])
        if t_tensor.ndim == 1:
            t_tensor = t_tensor[:, None]                # [B,1]

        return self.torch_score(r_rel, t_tensor)         # [B, N, 3]
        
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
        """Computes the score of IGSO(3) density as a rotation vector (torch path)."""
        device = vec.device
        dtype  = vec.dtype
        omega = torch.linalg.norm(vec, dim=-1) + eps             # shape [...]
        t     = torch.as_tensor(t, device=device, dtype=torch.float32)  # scalar or batched

        # σ(t) via log-schedule (torch)
        import math
        A = math.exp(self.min_sigma)
        B = math.exp(self.max_sigma) - A
        sigma_t = torch.log(A + B * t)                           # shape like t

        if self.use_cached_score:
            # time -> row indices in precomputed tables
            ds = torch.as_tensor(self.discrete_sigma, device=device, dtype=torch.float32)   # [S]
            idx = torch.bucketize(sigma_t, ds, right=False) - 1
            idx = idx.clamp(0, ds.numel() - 1)                   # shape like t

            # ω -> column bins
            omega_grid = torch.as_tensor(self.discrete_omega[:-1], device=device, dtype=torch.float32)  # [O-1]
            bins = torch.bucketize(omega.reshape(-1), omega_grid).clamp(0, omega_grid.numel() - 1)      # [M]

            # gather per-(t,ω) score norms
            score_norms = torch.as_tensor(self._score_norms, device=device, dtype=torch.float32)        # [S, O-1 or O]
            idx_exp = idx.expand(omega.shape).reshape(-1).long()  # [M]
            rows    = score_norms.index_select(0, idx_exp)        # [M, O-1 or O]
            vals    = rows.gather(1, bins.unsqueeze(-1)).squeeze(-1)     # [M]
            omega_scores = vals.view_as(omega)                    # [...]
        else:
            # analytic torch path
            sigma_exp  = sigma_t.expand(omega.shape).reshape(-1).unsqueeze(-1)  # [M,1]

            omega_flat = omega.reshape(-1)                                      # [M]

            omega_vals = self.alg.sampler.igso3_expansion(
                omega_flat, sigma_exp, use_torch=True
            )

            omega_scores = self.alg.score(
                omega_vals, omega_flat, sigma_exp, use_torch=True
            )
            omega_scores = omega_scores.view_as(omega)
        
        return omega_scores[..., None] * vec / (omega[..., None])

    def score_scaling(self, t: np.ndarray):
        """Calculates scaling used for scores during trianing."""
        return self._score_scaling[self.t_to_idx(t)]

    def forward_marginal(self, rot_0: np.ndarray, t: float):
        """Samples from the forward diffusion process at time index t.

        Args:
            rot_0: [..., 3] initial rotations.
            t: continuous time in [0, 1].

        Returns:
            rot_t: [..., 3] noised rotation vectors.
            rot_score: [..., 3] score of rot_t as a rotation vector.
        """
        n_samples = np.cumprod(rot_0.shape[:-1])[-1]
        sampled_rots = self.sample(t, n_samples=n_samples)
        rot_score = self.score(sampled_rots, t).reshape(rot_0.shape)

        # Right multiply.
        rot_t = self.compose_rotvec(rot_0, sampled_rots).reshape(rot_0.shape)
        return rot_t, rot_score

    def reverse(
            self,
            rot_t,
            score_t,
            t,
            dt,
            mask=None,
            noise_scale: float = 0.0,
            ):
        """Simulates the reverse SDE for 1 step using a right-multiplicative update (torch)."""
        # ensure torch tensors on the right device/dtype
        device = rot_t.device if torch.is_tensor(rot_t) else (score_t.device if torch.is_tensor(score_t) else self.device)
        rot_t   = torch.as_tensor(rot_t,   device=device, dtype=torch.float32)
        score_t = torch.as_tensor(score_t, device=device, dtype=torch.float32)
        mask_t  = None if mask is None else torch.as_tensor(mask, device=device, dtype=torch.float32)

        # g(t) from the torch-only diffusion_coef (supports broadcasting)
        t_torch = torch.as_tensor(t, device=device, dtype=torch.float32)
        g = self.diffusion_coef(t_torch)                         # scalar or broadcastable tensor

        # perturbation
        if noise_scale > 0.0:
            z = torch.randn_like(score_t) * noise_scale
        else:
            z = torch.zeros_like(score_t)

        perturb = (g[..., None]**2) * score_t * dt + g[..., None] * math.sqrt(dt) * z  # [...,3]
        if mask_t is not None:
            perturb = perturb * mask_t[..., None]

        # right-multiply: exp(rot_t) * exp(perturb)
        Ra = self.alg.exp_map(rot_t)
        Rb = self.alg.exp_map(perturb)
        Rab = torch.matmul(Ra, Rb)
        rot_t_1 = self.alg.get_v(self.alg.log_map(Rab))          # [...,3]
        return rot_t_1

    def score_scaling(self, t: np.ndarray):
        """Calculates scaling used for scores during trianing."""
        return self._score_scaling[self.t_to_idx(t)]


