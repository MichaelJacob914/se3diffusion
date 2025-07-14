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
Stiefel = geoopt.Stiefel()

class r3_diffuser: 
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
            
    def generate_noise(self, shape, t, batch_size, scale=1.0):
        sigma = torch.sqrt(1 - self.alpha_bars[int(t)]) * scale
        noise = sigma * torch.randn((batch_size, *shape),
                                    device=self.device, dtype=torch.float32)
        return noise

    def add_noise(self, x, noise, t):
        return torch.sqrt(self.alpha_bars[t]) * x + noise

    def descent(self, x_t, x_0, t, cost, num_updates = 1, lr = 1e-3): 
        x_0_optim = x_0.clone().detach().requires_grad_(True)
        x_t_optim = x_t.clone().detach().requires_grad_(True)

        for _ in range(num_updates):
            loss = cost(x_t_optim, x_0_optim, t)
            grad = torch.autograd.grad(loss, x_t_optim, create_graph=True)[0]
            x_t_optim = x_t_optim - lr * grad 

        return x_t_optim
        
    def _eu_sample(self, x_t, t, noise,
                   guidance=False, optim_steps=1, cost=None):
        #THIS IS NOT TESTED, USE EU_SAMPLE_BATCH

        beta_t       = self.betas[t]              
        alpha_t      = self.alphas[t]               
        alpha_bar_t  = self.alpha_bars[t]           
        alpha_bar_tm1 = self.alpha_bars[t-1] if t > 1 else self.alpha_bars[0]

        x0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * noise) \
                / torch.sqrt(alpha_bar_t)

        coef1 = torch.sqrt(alpha_bar_tm1) * beta_t / (1 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t)       * (1 - alpha_bar_tm1) / (1 - alpha_bar_t)
        mu = coef1 * x0_hat + coef2 * x_t

        if t > 1:
            sigma_t = torch.sqrt(beta_t)            
            x_tm1 = mu + sigma_t * torch.randn_like(x_t)
        else:
            x_tm1 = mu

        if guidance and t % optim_steps == 0:
            x_tm1 = self.descent(x_tm1, x0_hat, t, cost).detach()

        return x0_hat, x_tm1

    
    def _eu_sample_batch(
        self, x_t, t, eps_pred,
        guidance=False, optim_steps=1, cost=None
    ):
        if t > 1:
            v_noise = torch.sqrt(self.beta_hats[t]) * torch.randn_like(x_t)
        else:
            v_noise = torch.zeros_like(x_t)


        """

        v_tensor = (1 - self.alpha_bars[t])/(torch.sqrt(1 - self.alpha_bars[t])) *  eps_pred
        x_prev = (1 / torch.sqrt(torch.tensor(self.alphas[t]))) * (x_t - v_tensor) + v_noise
    
        """
        t_idx = t - 1

        beta_t  = self.betas[t_idx]
        alpha_t = self.alphas[t_idx]
        alpha_bar_t  = self.alpha_bars[t_idx]
        alpha_bar_tm1 = self.alpha_bars[t_idx-1] if t > 1 else alpha_bar_t

        coef1 = 1.0 / torch.sqrt(alpha_t)                          # 1/√α_t
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)    # (1-α_t)/√(1-ᾱ_t)

        x_mean = coef1 * (x_t - coef2 * eps_pred)                  # deterministic part

        # variance term (set add_noise=False if you want DDIM-style deterministic sampling)
        sigma_t = torch.sqrt(beta_t) * torch.sqrt(1 - alpha_bar_tm1) / torch.sqrt(1 - alpha_bar_t)
        x_prev  = x_mean + sigma_t * torch.randn_like(x_t) if t > 1 else x_mean

        if guidance and t % optim_steps == 0:
            with torch.no_grad():                       
                x_prev = self.descent(x_prev,
                                    (x_t - sqrt_one_m_abar * eps_pred) / torch.sqrt(abar_t),
                                    t, cost)
        
        #OPTIONAL PRINT STATEMENTS TO MONITOR DENOISING 
        """
        if t in [1, 10, 50, 96, 95, 94, 93, 92, 91 , 90, 97, 98, 99, 100, 200, 500, 996, 997, 998, self.T - 1]:
            alpha_t = self.alphas[t].item()
            alpha_bar_t = self.alpha_bars[t].item()
            beta_hat_t = self.beta_hats[t].item()
            coeff = (1 - alpha_t) / max((1 - alpha_bar_t)**0.5, 1e-5)

            print(f"\nStep t={t}")
            print("v_noise   = ", v_noise)
            print("x_prev", x_prev)
            print(f"  ε̂ norm        = {eps_pred.norm(dim=1).mean().item():.4f}")
            print(f"  x_t norm       = {x_t.norm(dim=1).mean().item():.4f}")
        """
        return x_prev
