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
    def __init__(self, T, batch_size=64, betas=None, device="cpu", verbose = False):
        self.device = torch.device(device)
        self.T = T
        self.verbose = verbose

        if betas is None:
            self.betas = self.make_cosine_beta_schedule(T).to(self.device)
        else:
            self.betas = torch.as_tensor(betas, dtype=torch.float32,
                                         device=self.device)

        self.alphas, self.alpha_bars = self.compute_alpha_bars(self.betas)
        self.beta_hats = self.compute_beta_hat(self.betas, self.alpha_bars)

        self.batch_size = batch_size
   
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
        Generate Gaussian noise of shape [B, N, 3] with std based on timestep t.

        Args:
            t (int): Current diffusion timestep
            B (int): Batch size
            N (int): Number of elements in SE(3)^N
            scale (float): Optional multiplier for the noise

        Returns:
            Tensor of shape [B, N, 3]
        """
        sigma = torch.sqrt(1 - self.alpha_bars[int(t)]) * scale
        return sigma * torch.randn((B, N, 3), device=self.device, dtype=torch.float32)
        
    def add_noise(self, x, noise, t):
        """
        Combine Gaussian Noise of shape [B, N, 3] with x also of shape [B,N,3]. 
        noise is assumed to be scaled appropriately on input. 
        """
        return torch.sqrt(self.alpha_bars[t]) * x + noise

    def descent(self, x_t, x_0, t, cost, num_updates = 1, lr = 1e-3): 
        x_0_optim = x_0.clone().detach().requires_grad_(True)
        x_t_optim = x_t.clone().detach().requires_grad_(True)

        for _ in range(num_updates):
            loss = cost(x_t_optim, x_0_optim, t)
            grad = torch.autograd.grad(loss, x_t_optim, create_graph=True)[0]
            x_t_optim = x_t_optim - lr * grad 

        return x_t_optim
    
    def _eu_sample_n(
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



# In[ ]:




