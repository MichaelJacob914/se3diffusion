#!/usr/bin/env python
# coding: utf-8
import sys
from pathlib import Path
import os
cwd = os.getcwd()
print("Working directory", cwd)
HERE = Path(__file__).resolve().parent
paths = [HERE / "gta", HERE / "gta" / "source"]
sys.path[:0] = [str(p) for p in paths if p.exists() and str(p) not in sys.path]
from prope.torch import PropeDotProductAttention 
import layers
from layers import Transformer as GTATransformer
from source.utils.gta import make_SO2mats, make_T2mats
from torch.backends.cuda import sdp_kernel
from torch.nn.attention import sdpa_kernel, SDPBackend

from layers import Attention, FeedForward
from source.utils.common import positionalencoding2d, downsample, rigid_transform
from source.utils.wigner_d import rotmat_to_wigner_d_matrices
from SO3n import so3_diffuser, SO3Algebra
from R3n import r3_diffuser
from se3n_utils import vec_to_pose, se3_from_rot_trans, Rigid, Rotation, rot9d_to_rotmat
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
import numpy as np
import copy
from scipy.spatial.transform import Rotation as Rot
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
from rotary_embedding_torch import RotaryEmbedding
from ray_rope.pos_enc.rayrope import RayRoPE_DotProductAttention
from ray_rope.pos_enc.utils.rayrope_mha import MultiheadAttention


#TAKEN FROM PAPER
#NOT MY CODE

import torch
import torch.nn as nn
import torch.nn.functional as F



def _build_2d_sincos_pos_embed(h: int, w: int, d: int, device=None):
    """
    Returns [h*w, d] 2D sinusoidal embeddings.
    Requires d % 4 == 0.
    """
    assert d % 4 == 0, "d_model must be divisible by 4 for 2D sincos."
    m = d // 4
    # frequencies like ViT: 1/10000^{2i/d}
    inv_freq = 1.0 / (10000 ** (torch.arange(0, m, device=device, dtype=torch.float32) / float(m)))
    # grid
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [h,w]
    xx = xx.reshape(-1, 1) 
    yy = yy.reshape(-1, 1) 

    pos_x = torch.cat([torch.sin(xx * inv_freq), torch.cos(xx * inv_freq)], dim=-1)
    pos_y = torch.cat([torch.sin(yy * inv_freq), torch.cos(yy * inv_freq)], dim=-1)
    pe = torch.cat([pos_x, pos_y], dim=-1)
    return pe  # [h*w, d]


def pre_compute_reps(attn_kwargs, extras):
    coord = extras.get('input_coord', None)
    NqTq = None
    if coord is not None:
        coord = coord.reshape(coord.shape[0], -1, coord.shape[-1])  # [B, Nq*Tq, 2]
        NqTq = coord.shape[1]
    f_dims = attn_kwargs['f_dims']
    flattened_reps = []
    flattened_invreps = []
    if 'so2' in f_dims and f_dims['so2'] > 0:
        coord = extras['input_coord']
        coord = coord.reshape(coord.shape[0], -1, 2)  # [B, Nq*Tq, 2]
        so2rep = make_SO2mats(coord,
                                nfreqs=attn_kwargs['so2'],
                                max_freqs=[attn_kwargs['max_freq_h'],
                                            attn_kwargs['max_freq_w']],
                                shared_freqs=attn_kwargs['shared_freqs'] if 'shared_freqs' in attn_kwargs else False)  # [B, Nq*Tq, deg, 2, 2, 2]
        so2rep = so2rep.flatten(-4, -3)
        NqTq = so2rep.shape[1]
        extras['so2rep_q'] = extras['so2rep_k'] = so2rep  # [B, T, C, 2, 2]
        extras['so2fn'] = lambda A, x: torch.einsum(
            'btcij,bhtcj->bhtci', A, x)
        flattened = so2rep.reshape(
            so2rep.shape[0], so2rep.shape[1], -1)  # [B, T, C*2*2]
        flattened_reps.append(flattened)
        # [B, T, C*2*2]
        flattened_inv = so2rep.transpose(-2, -1).reshape(
            so2rep.shape[0], so2rep.shape[1], -1)
        flattened_invreps.append(flattened_inv)
    if 't2' in f_dims and f_dims['t2'] > 0:
        coord = extras['input_coord']
        coord = coord.reshape(coord.shape[0], -1, 2)  # [B, Nq*Tq, 2]
        t2rep = make_T2mats(coord)  # [B, Nq*Tq, 2] -> [B, Nq*Tq, 3, 3]
        extras['t2rep_q'] = extras['t2rep_k'] = t2rep
        extras['inv_t2rep_q'] = torch.linalg.inv(t2rep)
        extras['t2fn'] = lambda A, x: torch.einsum(
            'btij,bhtcj->bhtci', A, x)

    if 'se3' in f_dims and f_dims['se3'] > 0:
        extrinsic = extras['input_transforms']
        se3rep = torch.linalg.inv(extrinsic)  # [B, Nq, 4, 4]
        if 'ray_to_se3' in attn_kwargs and attn_kwargs['ray_to_se3']:
            B, Nq = se3rep.shape[0], se3rep.shape[1]
            input_rays = downsample(
                extras['input_rays'], 3).reshape(B, Nq, -1, 3)
            # [B, Nq, T, 4, 4]
            R = ray2rotation(input_rays, return_4x4=True)
            se3rep = torch.einsum(
                'bnij,bntjk->bntik', se3rep, R)  # mul from right
            extrinsic = torch.einsum(
                'bntij,bnjk->bntik',  R.transpose(-2, -1), extrinsic)  # mul from left
            extras['se3fn'] = lambda A, x: torch.einsum(
                'bntij,bhntcj->bhntci', A, x)
        else:
            extras['se3fn'] = lambda A, x: torch.einsum(
                'bnij,bhntcj->bhntci', A, x)
        extras['se3rep_q'] = extras['se3rep_k'] = se3rep
        extras['inv_se3rep_q'] = extrinsic

        flattened = extrinsic.repeat_interleave(
            NqTq//extrinsic.shape[1], 1).transpose(-2, -1).reshape(se3rep.shape[0], -1, 16)  # [B, T, 4*4]
        flattened_reps.append(flattened)
        flattened_inv = extrinsic.repeat_interleave(
            NqTq//extrinsic.shape[1], 1).reshape(se3rep.shape[0], -1, 16)  # [B, T, 4*4]
        flattened_invreps.append(flattened_inv)

    if 'so3' in f_dims and f_dims['so3'] > 0:
        n_degs = attn_kwargs['so3']
        R_q = torch.linalg.inv(extras['input_transforms'])[..., :3, :3]
        B, Nq = R_q.shape[0], R_q.shape[1]
        D_q = rotmat_to_wigner_d_matrices(n_degs, R_q.flatten(0, 1))[1:]
        for i, D in enumerate(D_q):
            if 'zeroout_so3' in attn_kwargs and attn_kwargs['zeroout_so3']:
                D_q[i] = torch.zeros_like(
                    D.reshape(B, Nq, D.shape[-2], D.shape[-1]))
            elif 'id_so3' in attn_kwargs and attn_kwargs['id_so3']:
                D_q[i] = torch.stack([torch.eye(
                    D.shape[-1])]*B*Nq, 0).reshape(B, Nq, D.shape[-2], D.shape[-1]).to(D.device)
            else:
                D_q[i] = D.reshape(B, Nq, D.shape[-2], D.shape[-1])
        extras['so3rep_q'] = extras['so3rep_k'] = D_q
        extras['so3fn'] = lambda A, x: torch.einsum(
            'bnij,bhnkj->bhnki', A, x)

    extras['flattened_rep_q'] = extras['flattened_rep_k'] = torch.cat(
        flattened_reps, -1)  # 16 + 2*freqs*2*2
    extras['flattened_invrep_q'] = torch.cat(flattened_invreps, -1)



class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.w1 = nn.Linear(d_model, hidden * 2)
        self.w2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w1(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        x = self.w2(x)
        return self.drop(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        # x: [B, L, d]
        y = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.dropout1(y)
        x = x + self.mlp(self.norm2(x))
        return x
        
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//4, 1), nn.GELU(),
            nn.Conv2d(c//4, c, 1), nn.Sigmoid()
        )
    def forward(self, x):
        y = self.net(x)
        y = y * self.se(x)
        return x + y

    
class StandardTransformer(nn.Module):
    """
    Alternating intra/inter attention
    """
    def __init__(self,
                 d_model: int = 384,
                 num_layers: int = 8,
                 representation: str = "quat",
                 prediction: str = "pose",     # "pose" | "noise"
                 num_heads: int = 8,
                 dropout: float = 0.00,
                 num_timesteps: int = 100,
                 grid_hw_img: int = 0, 
                 grid_hw_feat: int = 6):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sincos PE."

        self.d_model = d_model
        self.num_layers = num_layers
        self.representation = representation
        self.prediction = prediction
        self.grid_hw_img = grid_hw_img
        self.grid_hw_feat = grid_hw_feat
        self.n_img_tokens  = grid_hw_img * grid_hw_img
        self.n_feat_tokens = grid_hw_feat * grid_hw_feat

        # ----------- Projections / Encoders -----------
        self.pose_proj  = nn.Sequential(nn.Linear(12, d_model), 
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        self.img_backbone = nn.Sequential(
            nn.Conv2d(3, 128, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(),
            ResBlock(256), ResBlock(256),
            nn.AdaptiveAvgPool2d((grid_hw_img, grid_hw_img)),
            nn.Conv2d(256, d_model, 1), nn.GELU(),
            ResBlock(d_model)
        )

        self.feat_proj = nn.Sequential(
            nn.Conv2d(1024, d_model, 1), nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1), nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1), nn.GELU(),
        )

        # 2D positional encodings for img/feat grids (first-model style)
        pos_img  = _build_2d_sincos_pos_embed(grid_hw_img, grid_hw_img, d_model)   # [G², d]
        pos_feat = _build_2d_sincos_pos_embed(grid_hw_feat, grid_hw_feat, d_model)   # [G², d]
        self.register_buffer("pos_img",  pos_img,  persistent=False)
        self.register_buffer("pos_feat", pos_feat, persistent=False)

        # Token-type embeddings: 0=CLS, 1=pose, 2=time, 3=feat, 4=img
        self.token_type = nn.Embedding(5, d_model)
        type_ids_view = [0, 1, 2] + [3]*self.n_feat_tokens + [4]*self.n_img_tokens
        self.K = len(type_ids_view)  # tokens per view
        self.register_buffer("type_ids_view", torch.tensor(type_ids_view, dtype=torch.long),
                             persistent=False)

        # Learnable per-view CLS token (shared across views)
        self.view_cls = nn.Parameter(torch.zeros(1, 1, 1, d_model))  # [1,1,1,d]

        def make_layer():
            return nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True, norm_first=True
            )
        n_intra = (num_layers + 1) // 2  # start with intra
        n_inter = num_layers // 2
        self.intra_layers = nn.ModuleList([make_layer() for _ in range(n_intra)])
        self.inter_layers = nn.ModuleList([make_layer() for _ in range(n_inter)])

        if (representation in {"quat", "exp"}) or (prediction == "noise"):
            dim_out = 6
        else:
            dim_out = 9
        

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_model, dim_out)
        )


    def _pose_tokens(self, R, T):
        B, N = R.shape[:2]
        pose_vec = torch.cat([R.reshape(B, N, 9), T], dim=-1)      # [B,N,12]
        return self.pose_proj(pose_vec)                            # [B,N,d]

    def _time_tokens(self, t, B, N, device, dtype):
        if isinstance(t, (int, float)):
            t = torch.full((B, N, 1), int(t), device=device, dtype=dtype)
        else:
            if t.ndim == 2:
                t = t.unsqueeze(-1)
            t = t.to(device=device, dtype=dtype)
        return self.time_embed(t)                                  # [B,N,d]

    def _image_tokens(self, imgs):
        # imgs: [B,N,3,H,W] → [B,N,G²,d] + 2D PE
        B, N = imgs.shape[:2]
        x = imgs.view(B*N, *imgs.shape[2:])                        # [B*N,3,H,W]
        x = self.img_backbone(x)                                   # [B*N,d,G,G]
        x = x.flatten(2).transpose(1, 2)                          
        x = x + self.pos_img.unsqueeze(0)                          # add PE
        return x.view(B, N, self.n_img_tokens, self.d_model)       # [B,N,G²,d]

    def _feat_tokens(self, feats):
        # feats: [B,N,1024,Hf,Wf] → [B,N,G²,d] + 2D PE
        B, N = feats.shape[:2]
        f = self.feat_proj(feats.view(B*N, *feats.shape[2:]))      # [B*N,d,Hf,Wf]
        f = F.adaptive_avg_pool2d(f, (self.grid_hw_feat, self.grid_hw_feat)) 
        f = f.flatten(2).transpose(1, 2)                         
        f = f + self.pos_feat.unsqueeze(0)                         # add PE
        return f.view(B, N, self.n_feat_tokens, self.d_model)      # [B,N,G²,d]

    # ----- Forward -----
    def forward(self, R, T, feats, imgs, t):
        """
        Inputs:
          R:     [B,N,3,3]
          T:     [B,N,3]
          feats: [B,N,1024,Hf,Wf]
          imgs:  [B,N,3,H,W]
          t:     int or [B,N] (ints)
        Output: [B,N,4,4] pose
        """
        B, N = R.shape[:2]
        d = self.d_model
        K = self.K

        pose_tok = self._pose_tokens(R, T)                         # [B,N,d]
        time_tok = self._time_tokens(t, B, N, device=R.device,
                                     dtype=pose_tok.dtype)         # [B,N,d]
        feat_tok = self._feat_tokens(feats)                      
        img_tok  = self._image_tokens(imgs)               

        view_cls = self.view_cls.expand(B, N, 1, d)                # [B,N,1,d]

        tokens = torch.cat([
            view_cls,                                              # [B,N,1,d]
            pose_tok.unsqueeze(2),                                 # [B,N,1,d]
            time_tok.unsqueeze(2),                                 # [B,N,1,d]
            feat_tok,                                          
            img_tok                                       
        ], dim=2)                                                  # [B,N,K,d]

        type_emb = self.token_type(self.type_ids_view.to(tokens.device))                # [K,d]
        x = tokens + type_emb.view(1, 1, K, d)                       # [B,N,K,d]

        intra_i, inter_i = 0, 0
        for k in range(self.num_layers):
            if k % 2 == 0:
                y = x.reshape(B * N, K, d)                         # intra per view
                y = self.intra_layers[intra_i](y)
                x = y.view(B, N, K, d)
                intra_i += 1
            else:
                y = x.view(B, N * K, d)                            # inter across views
                y = self.inter_layers[inter_i](y)
                x = y.view(B, N, K, d)
                inter_i += 1

        pose_slot = x[:, :, 1, :]                                                  # [B,N,d]
        out_vec   = self.output_proj(pose_slot)    

        if self.representation == "rot6d":
            return vec_to_pose(out_vec.reshape(B*N, 9), self.representation).reshape(B, N, 4, 4)
        elif self.prediction == "noise":
            return out_vec.reshape(B*N, 6).reshape(B, N, 4, 4)
        else:
            return vec_to_pose(out_vec.reshape(B*N, 6), self.representation).reshape(B, N, 4, 4)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(
            dim) if dim is not None else lambda x: torch.nn.functional.normalize(x, dim=-1)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
        

class PRoPEAttention(nn.Module):
    """
    Multi-head attention that wraps PRoPE around SDPA.
    - Projects tokens to Q/K/V
    - Applies PRoPE transforms (q/k/v)
    - Calls SDPA 
    - Applies inverse PRoPE on outputs
    """
    def __init__(self, dim, heads, head_dim, patches_x, patches_y, image_width, image_height, dropout=0.):
        super().__init__()
        assert dim == heads * head_dim, "dim must equal heads * head_dim"
        assert head_dim % 4 == 0, "PRoPE requires head_dim % 4 == 0"

        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height

        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

        self.prope = PropeDotProductAttention(
            head_dim   = head_dim,
            patches_x  = patches_x,
            patches_y  = patches_y,
            image_width  = image_width,
            image_height = image_height,
        )

    def forward(self, x, viewmats, Ks=None, attn_mask=None, is_causal=False):
        """
        x:        (B, L, D) where L = cameras * (patches_x * patches_y)
        viewmats: (B, cameras, 4, 4)
        Ks:       (B, cameras, 3, 3) or None
        """
        B, L, D = x.shape
        H, Dh   = self.heads, self.head_dim
        cams    = viewmats.shape[1]
        per_cam = self.patches_x * self.patches_y
        assert L == cams * per_cam, f"seq len {L} != cameras({cams})*patches({per_cam})"

        self.prope._precompute_and_cache_apply_fns(viewmats=viewmats, Ks=Ks)

        qkv = self.qkv(x).view(B, L, 3, H, Dh).permute(0, 3, 1, 2, 4)
        q, k, v = qkv.unbind(dim=3)

        q = self.prope._apply_to_q(q)
        k = self.prope._apply_to_kv(k)
        v = self.prope._apply_to_kv(v)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        y = self.prope._apply_to_o(out)                           # (B,H,L,Dh)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, L, D)    # (B,L,D)
        return self.drop(self.proj(y))

class RayRoPEAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, patches_x, patches_y, image_width, image_height, dropout=0.):
        super().__init__()
        assert dim == heads * head_dim

        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.dropout = dropout

        self.rayrope_attn = RayRoPE_DotProductAttention(
            head_dim=head_dim,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_width,
            image_height=image_height,
        )

        # IMPORTANT: RayRoPE needs depth+sigma (2 dims) 
        self.mha_layer = MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            predict_d='predict_dsig',
            sdpa_fn=self.rayrope_attn.forward,
        )

    def forward(self, x, w2cs, Ks=None, attn_mask=None, is_causal=False, context_depths=None):
        # Cache ray geometry for this forward
        self.rayrope_attn._precompute_and_cache_apply_fns(w2cs, Ks, context_depths=context_depths)

        # Call token-in MHA; it will:
        #  - project to q/k/v
        #  - (optionally) predict depth+sigma per token and pass it to RayRoPE
        #  - call RayRoPE SDPA
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            y = self.mha_layer(x, x, x)

        return y

class RoPEAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, rotary_dim=32, dropout=0.):
        super().__init__()
        assert dim == heads * head_dim, "dim must equal heads * head_dim"
        assert head_dim % 2 == 0, "head_dim must be multiple of 2 for RoPE"

        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.rotary_dim = head_dim
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rotary_emb = RotaryEmbedding(dim=rotary_dim)

    def forward(self, x, attn_mask=None, is_causal=False):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        
        H, Dh = self.heads, self.head_dim

        qkv = self.qkv(x).view(B, L, 3, H, Dh).permute(0, 3, 1, 2, 4)
        q, k, v = qkv.unbind(dim=3)  # (B, H, L, Dh)

        # Apply rotary embedding to Q and K only
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)


        with sdpa_kernel([SDPBackend.MATH]):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=is_causal
            )

        y = y.view(B, H, L, Dh).permute(0, 2, 1, 3).reshape(B, L, D)
        return self.proj_drop(self.proj(y))

class PoseToPoseAttn(nn.Module):
    """
    Attention among special tokens across views.

    Modes:
      - concatenate_pairs=True : N tokens of size 2D  (per-view concat of [view_tok, cls_tok])
      - concatenate_pairs=False: 2N tokens of size D  (self-attn over all specials)
    Returns: [B, N, 2, D]
    """
    def __init__(self, dim, heads, dropout=0.0, time_dim=None, concatenate_pairs=False, head_dim=None):
        super().__init__()
        self.concatenate_pairs = concatenate_pairs
        self.dim = dim
        self.heads = heads
        self.drop = nn.Dropout(dropout)

        attn_dim = 2 * dim if concatenate_pairs else dim
        if head_dim is None:
            assert attn_dim % heads == 0, "attn_dim must be divisible by heads"
            head_dim = attn_dim // heads
        self.head_dim = head_dim
        assert attn_dim == heads * head_dim, "attn_dim must equal heads * head_dim"

        self.ln = nn.LayerNorm(attn_dim)
        self.qkv = nn.Linear(attn_dim, 3 * attn_dim, bias=False)
        self.proj = nn.Linear(attn_dim, attn_dim, bias=False)

        self.use_time = time_dim is not None
        if self.use_time:
            self.time_to_bias = nn.Sequential(
                nn.Linear(time_dim, dim),
                nn.Tanh()
            )

    def forward(self, specials, time_emb=None, attn_mask=None, is_causal=False):
        """
        specials: [B, N, 2, D]  (two specials per view)
        time_emb: [B, N, T]     optional
        attn_mask: broadcastable to (B*H, L, L)
        """
        B, N, S, D = specials.shape
        assert S == 2 and D == self.dim

        if self.use_time and time_emb is not None:
            bias = self.time_to_bias(time_emb).unsqueeze(2)  # [B,N,1,D]
            specials = specials + bias

        H, Dh = self.heads, self.head_dim

        if self.concatenate_pairs:
            x = specials.reshape(B, N, 2 * D)          # [B,N,2D]
            x_ln = self.ln(x)
            qkv = self.qkv(x_ln).view(B, N, 3, H, Dh).permute(0, 3, 1, 2, 4)
            q, k, v = qkv.unbind(dim=3)               # [B,H,N,Dh]

            with sdpa_kernel([SDPBackend.MATH]):
                y = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=attn_mask,
                                                dropout_p=0.0,
                                                is_causal=is_causal)   # [B*H,N,Dh]

            y = y.view(B, H, N, Dh).permute(0, 2, 1, 3).reshape(B, N, H*Dh)  # [B,N,2D]
            y = self.drop(self.proj(y))
            out = x + y
            return out.reshape(B, N, 2, D)

        else:
            # ----- Mode B: 2N tokens of size D -----
            L = 2 * N
            x = specials.reshape(B, L, D)             # [B,2N,D]
            x_ln = self.ln(x)
            qkv = self.qkv(x_ln).view(B, L, 3, H, Dh).permute(0, 3, 1, 2, 4)
            q, k, v = qkv.unbind(dim=3)               # [B,H,2N,Dh]
            q = q.reshape(B*H, L, Dh)
            k = k.reshape(B*H, L, Dh)
            v = v.reshape(B*H, L, Dh)

            with sdpa_kernel([SDPBackend.MATH]):
                y = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=attn_mask,
                                                dropout_p=0.0,
                                                is_causal=is_causal)   # [B*H,2N,Dh]

            y = y.view(B, H, L, Dh).permute(0, 2, 1, 3).reshape(B, L, H*Dh)  # [B,2N,D]
            y = self.drop(self.proj(y))
            out = x + y
            return out.reshape(B, N, 2, D)

class DiffusionAttentionBlock(nn.Module):
    def __init__(
        self,
        dim, heads, dim_head, mlp_dim,
        patches_x, patches_y, image_width, image_height,
        special_tokens_per_view=2, 
        dropout=0.0,
        use_local=True,
        use_s2s=True,
        use_p2s=False,
        use_rayrope = False, 
        use_global=True,
        concatenate_pairs=False,
        time_dim=None 
    ):
        super().__init__()
        self.S = int(special_tokens_per_view)
        self.P = int(patches_x * patches_y)

        self.use_local        = bool(use_local)
        self.use_s2s          = bool(use_s2s)
        self.use_global = bool(use_global)
        self.use_rayrope = bool(use_rayrope)

        # Local attention using RoPE
        if(self.use_local):
            self.local = (
                RoPEAttention(
                    dim, heads, dim_head, rotary_dim=dim_head, dropout=dropout
                )
            )
            self.local_attn = PreNorm(dim, self.local)
            self.local_ff   = PreNorm(dim, FeedForward(dim, 2*dim, dropout=dropout))


        # Pose-to-pose attention with time conditioning
        self.s2s = (
            PoseToPoseAttn(dim, heads, dropout=dropout, time_dim=time_dim, concatenate_pairs=concatenate_pairs)
            if self.use_s2s else None
        )

        # Global PRoPE attention (patches only)
        if self.use_global:
            if(use_rayrope): 
                if use_rayrope:
                    self.global_attn = PreNorm(dim, RayRoPEAttention(
                        dim=dim, heads=heads, head_dim=dim_head,
                        patches_x=patches_x, patches_y=patches_y,
                        image_width=image_width, image_height=image_height,
                        dropout=dropout
                    ))
                    self.global_ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            else: 
                self.global_attn = PreNorm(dim, PRoPEAttention(
                    dim=dim, heads=heads, head_dim=dim_head,
                    patches_x=patches_x, patches_y=patches_y,
                    image_width=image_width, image_height=image_height,
                    dropout=dropout
                ))
                self.global_ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        else:
            self.global_attn = None
            self.global_ff   = None

    def forward(self, x, B, N, T, extras_global=None, Ks=None, block_number = None):
        D = x.shape[-1]
        assert T == self.S + self.P, f"Expected T={self.S + self.P}, got {T}"

        # LOCAL attention (within-view: special tokens and patches)
        if self.use_local:
            y = x.reshape(B * N, T, D)
            y = self.local_attn(y) + y
            y = self.local_ff(y) + y
            x = y.reshape(B, N, T, D)

        # Split into special tokens and patches
        specials = x[:, :, :self.S, :]  # [B, N, S=2, D] (view_tok + cls_tok)
        patches  = x[:, :,  self.S:, :]  # [B, N, P, D]

        # GLOBAL PRoPE attention (patches only, time-agnostic)
        if self.global_attn is not None:
            if extras_global is None:
                raise ValueError("Global PRoPE requires extras_global with keys 'R' and 'T'.")
            R  = extras_global["R"]   # [B,N,3,3]
            Tt = extras_global["T"]   # [B,N,3]

            R_c2w = R.reshape(B*N, 3, 3)
            t_c2w = Tt.reshape(B*N, 3, 1)

            with torch.amp.autocast("cuda", enabled=False):
                R_w2c = R_c2w.transpose(1, 2).float()
                t_w2c = (-R_w2c @ t_c2w.float()).squeeze(-1)

            viewmats = torch.eye(4, device=R.device, dtype=torch.float32).repeat(B*N, 1, 1)
            viewmats[:, :3, :3] = R_w2c
            viewmats[:, :3, 3]  = t_w2c
            viewmats = viewmats.view(B, N, 4, 4)
            
            y = patches.reshape(B, N * self.P, D)

            if self.use_rayrope:
                # RayRoPEAttention expects w2cs (same as viewmats here) and optional context_depths
                context_depths = extras_global.get("context_depths", None)  # (B,N,H,W,1) if you ever have it
                y = self.global_attn(y, w2cs=viewmats, Ks=Ks, context_depths=context_depths) + y
            else:
                y = self.global_attn(y, viewmats=viewmats, Ks=Ks) + y
            y = self.global_ff(y) + y
            patches = y.reshape(B, N, self.P, D)

        # S2S: Pose-to-pose attention WITH light time conditioning
        if self.s2s is not None:
            time_emb = extras_global.get("time_emb", None) if extras_global else None
            specials = self.s2s(specials, time_emb=time_emb)
        

        out = torch.cat([specials, patches], dim=2)

        return out


class OldPRoPEUpdatePoseTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 1024,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 device: str = "cuda",
                 representation: str = "rot9d",
                 update_type: str = "mult", 
                 feature_type = "dust3r", 
                 img_size: int = 224,
                 prediction = "pose",
                 grid_hw_img: int = 0, 
                 attn_args: Optional[dict] = None, 
                 scheme = "PRoPE",
                 attn_kwargs: Optional[dict] = None): 

        super().__init__()
        self.device = device
        self.representation = representation
        self.update_type = update_type
        self.feature_type = feature_type
        if(feature_type == "dust3r"): 
            self.grid_hw_feat = 14
            self.num_heads = num_heads
            self.d_model = d_model
            self.num_layers = num_layers
        else: 
            self.grid_hw_feat = 16
            self.d_model = 2048
            self.num_layers = 6
            self.num_heads = 16
        
        self.scheme = scheme
        if(self.scheme == "rayrope"):
            use_rayrope = True
            self.d_model = 1008
            self.num_heads = 14
            print("USING RAYROPE")

        else:
            use_rayrope = False
            print("USING ", scheme)
        self.n_feat_tokens = self.grid_hw_feat * self.grid_hw_feat

        # DUSt3r features: [1024,14,14] -> [d,14,14]
        self.feat_channel_proj = nn.Identity() if self.d_model == 1024 else nn.Conv2d(1024, self.d_model, 1, bias=False)

        # 2D sincos PE
        self.register_buffer("pos_feat",
            _build_2d_sincos_pos_embed(self.grid_hw_feat, self.grid_hw_feat, self.d_model),
            persistent=False
        )

        
        # View token encoder (R,T) -> d 
        self.view_token_init = nn.Sequential(
            nn.Linear(12, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model), nn.SiLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        # CLS token 
        self.cls_token_init = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        # Time embedding
        self.max_timesteps = 100
        self.time_embed = nn.Embedding(self.max_timesteps, self.d_model // 2)
        self.alpha_logits = nn.Parameter(torch.full((self.num_layers,), 1.3863))  # init 0.8

        # PRoPE blocks 
        self.blocks = nn.ModuleList([
            DiffusionAttentionBlock(
                dim=self.d_model, heads=self.num_heads, dim_head=self.d_model // self.num_heads, mlp_dim=2 * self.d_model,
                patches_x=self.grid_hw_feat, patches_y=self.grid_hw_feat,
                image_width=img_size, image_height=img_size,
                special_tokens_per_view=2, 
                dropout=dropout,
                use_local=True,
                use_s2s=True, 
                use_global=True,
                use_rayrope = use_rayrope, 
                concatenate_pairs=True,
                time_dim=self.d_model // 2  
            )
            for _ in range(self.num_layers)
        ])

        # Decoder: uses token + time
        if representation == "rot9d":
            dim_out = 12
        elif representation == "rot6d":
            dim_out = 9
        if update_type == "mult":
            dim_out = 6

        self.pose_decode = nn.Sequential(
            nn.Linear(self.d_model + self.d_model // 2, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, dim_out),
        )

        self.output_norm = nn.LayerNorm(self.d_model)
        for blk in self.blocks:
            blk.debug_grad = True

        print("Initialized PRoPE Update Pose Transformer using", feature_type, " features and", update_type, " update type.")

    def _feat_tokens(self, feats: torch.Tensor) -> torch.Tensor:
        B, N, C, Hf, Wf = feats.shape
        assert C == 1024 and Hf == self.grid_hw_feat and Wf == self.grid_hw_feat
        x = feats.view(B * N, C, Hf, Wf)
        x = self.feat_channel_proj(x)                         # [B*N, d, Hf, Wf]
        x = x.flatten(2).transpose(1, 2)       # [B*N, Tf, d]
        
        if self.scheme not in ["PRoPE", "rayrope"]:
            print("USING POSITIONAL EMBEDDINGS")
            x = x + self.pos_feat.to(x.device).unsqueeze(0)
        return x.view(B, N, Hf * Wf, self.d_model)

    def _time_vec(self, t, B, N, device):
        t = torch.as_tensor(t, device=device, dtype=torch.float32)

        # allow scalar / [B] / [B,N]
        if t.ndim == 0:
            t = t.expand(B, N)
        elif t.ndim == 1:
            t = t.view(B, 1).expand(B, N)

        # continuous time in (0,1], clamp away from 0
        t = t.clamp(0.01, 1.0)

        # bin to embedding indices
        tids = ((t - 0.01) / (1.0 - 0.01) * (self.max_timesteps - 1)).round().long()
        return self.time_embed(tids)  # [B,N,d_t]


    def forward(self, R, T, feats, imgs, Ks, t, return_intermediate: bool = False):
        """
        R:[B,N,3,3], T:[B,N,3]
        feats:[B,N,1024,14,14]
        t: scalar or [B] or [B,N]
        """
        B, N = R.shape[:2]
        d = self.d_model

        # Initialize tokens
        feat_tok = self._feat_tokens(feats)                              # [B,N,P,d]
        
        # View token
        view_tok = self.view_token_init(
            torch.cat([R.reshape(B, N, 9), T], dim=-1)
        )  # [B,N,d]
        
        # CLS token
        cls_tok = self.cls_token_init.expand(B, N, -1)                   # [B,N,d]
        
        # Time embedding 
        time_vec = self._time_vec(t, B, N, device=R.device)              # [B,N,d_t]

        # Build per-view sequence: [view_tok, cls_tok, feature_tokens]
        tokens = torch.cat([
            view_tok.unsqueeze(2),  # [B,N,1,d]
            cls_tok.unsqueeze(2),   # [B,N,1,d]
            feat_tok                # [B,N,P,d]
        ], dim=2)  # [B,N,2+P,d]
        
        T_all = tokens.shape[2]

        R_cur, t_cur = R, T
        inter_R, inter_t = [], []

        extras = {
                "R": R_cur, 
                "T": t_cur,
                "time_emb": time_vec  
            }

        for L in range(self.num_layers):
            alpha = torch.sigmoid(self.alpha_logits[L]).to(tokens.dtype)  # scalar for this layer
            # PRoPE attention uses current geometry + time info
            extras["R"] = R_cur
            extras["T"] = t_cur

            x = self.blocks[L](tokens, B=B, N=N, T=T_all, extras_global=extras, Ks=Ks)

            x_view = x[:, :, 0, :]  # View token (geometric context)

            x_view_norm = self.output_norm(x_view)

            # Decode pose from pose token + time
            decode_in = torch.cat([x_view_norm, time_vec], dim=-1)  # [B,N,d+d_t]
            if(self.update_type == "mult"): 
                with torch.autocast(device_type="cuda", enabled=False):
                    upd = self.pose_decode(decode_in).view(B, N, 6)
                    curr = Rigid(
                        Rotation(rot_mats=R_cur.reshape(B*N, 3, 3), quats=None),
                        t_cur.reshape(B*N, 3),
                    )
                    curr = curr.compose_q_update_vec(upd.reshape(B*N, 6))
                    R_cur = curr.get_rots().get_rot_mats().reshape(B, N, 3, 3)
                    R_cur = rot9d_to_rotmat(R_cur.reshape(B, N, 9)).reshape(B, N, 3, 3)  # project to SO(3)
                    t_cur = curr.get_trans().reshape(B, N, 3)
                    pose_4x4_next = curr.to_tensor_4x4().reshape(B, N, 4, 4)

                    pose_4x4_next = pose_4x4_next.clone()
                    pose_4x4_next[..., :3, :3] = R_cur
                    pose_4x4_next[..., :3, 3]  = t_cur
            else: 
                pose_vec_next = self.pose_decode(decode_in)
                pose_4x4_next = vec_to_pose(
                    pose_vec_next.reshape(B * N, -1),
                    self.representation
                ).reshape(B, N, 4, 4)
                R_cur = pose_4x4_next[..., :3, :3]
                t_cur = pose_4x4_next[..., :3, 3]


            Rt = R_cur
            tt = t_cur
            view_tok_fresh = self.view_token_init(torch.cat([Rt.reshape(B, N, 9), tt], dim=-1))
            tokens = x.clone()

            tok0 = tokens[:, :, 0, :]                               # [B,N,D] view
            tok0_new = alpha * tok0 + (1 - alpha) * view_tok_fresh  # [B,N,D]

            tokens = torch.cat([tok0_new.unsqueeze(2), tokens[:, :, 1:, :]], dim=2)

            if return_intermediate:
                inter_R.append(R_cur.clone()) 
                inter_t.append(t_cur.clone())

        out = {"R": R_cur, "t": t_cur, "pose": pose_4x4_next}
        if return_intermediate:
            out["R_layers"] = torch.stack(inter_R, dim=0)
            out["t_layers"] = torch.stack(inter_t, dim=0)
        return out

class PRoPEUpdatePoseTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 1024,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 device: str = "cuda",
                 representation: str = "rot9d",
                 update_type: str = "mult", 
                 feature_type = "dust3r", 
                 img_size: int = 224,
                 prediction = "pose",
                 grid_hw_img: int = 0, 
                 attn_args: Optional[dict] = None, 
                 scheme = "PRoPE",
                 attn_kwargs: Optional[dict] = None): 

        super().__init__()
        self.device = device
        self.representation = representation
        self.update_type = update_type
        self.feature_type = feature_type
        if(feature_type == "dust3r"): 
            self.grid_hw_feat = 14
            self.num_heads = num_heads
            self.d_model = d_model
            self.num_layers = num_layers
        else: 
            self.grid_hw_feat = 16
            self.d_model = 2048
            self.num_layers = 6
            self.num_heads = 16
        
        self.scheme = scheme
        if(self.scheme == "rayrope"):
            use_rayrope = True
            self.d_model = 1008
            self.num_heads = 14

        else:
            use_rayrope = False
        self.n_feat_tokens = self.grid_hw_feat * self.grid_hw_feat

        # DUSt3r features: [1024,14,14] -> [d,14,14]
        self.feat_channel_proj = nn.Identity() if self.d_model == 1024 else nn.Conv2d(1024, self.d_model, 1, bias=False)

        # 2D sincos PE
        self.register_buffer("pos_feat",
            _build_2d_sincos_pos_embed(self.grid_hw_feat, self.grid_hw_feat, self.d_model),
            persistent=False
        )

        
        # View token encoder (R,T) -> d 
        self.view_token_init = nn.Sequential(
            nn.Linear(12, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model), nn.SiLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        # CLS token 
        self.cls_token_init = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        # Time embedding
        self.max_timesteps = 100
        self.time_embed = nn.Embedding(self.max_timesteps, self.d_model // 2)

        # PRoPE blocks 
        self.blocks = nn.ModuleList([
            DiffusionAttentionBlock(
                dim=self.d_model, heads=self.num_heads, dim_head=self.d_model // self.num_heads, mlp_dim=2 * self.d_model,
                patches_x=self.grid_hw_feat, patches_y=self.grid_hw_feat,
                image_width=img_size, image_height=img_size,
                special_tokens_per_view=2, 
                dropout=dropout,
                use_local=True,
                use_s2s=True, 
                use_global=True,
                use_rayrope = use_rayrope, 
                concatenate_pairs=True,
                time_dim=self.d_model // 2  
            )
            for _ in range(self.num_layers)
        ])

        # Decoder: uses token + time
        if representation == "rot9d":
            dim_out = 12
        elif representation == "rot6d":
            dim_out = 9
        if update_type == "mult":
            dim_out = 6

        self.pose_decode = nn.Sequential(
            nn.Linear(self.d_model + self.d_model // 2, self.d_model),
            nn.SiLU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, dim_out),
        )

        nn.init.normal_(self.pose_decode[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.pose_decode[-1].bias)

        self.output_norm = nn.LayerNorm(self.d_model)

        print("Initialized PRoPE Update Pose Transformer using", feature_type, " features ", update_type, " update type on", self.representation, "and ", scheme, "embeddings.")

    def _feat_tokens(self, feats: torch.Tensor) -> torch.Tensor:
        B, N, C, Hf, Wf = feats.shape
        assert C == 1024 and Hf == self.grid_hw_feat and Wf == self.grid_hw_feat
        x = feats.view(B * N, C, Hf, Wf)
        x = self.feat_channel_proj(x)                         # [B*N, d, Hf, Wf]
        x = x.flatten(2).transpose(1, 2)       # [B*N, Tf, d]
        
        if self.scheme not in ["PRoPE", "rayrope"]:
            print("USING POSITIONAL EMBEDDINGS")
            x = x + self.pos_feat.to(x.device).unsqueeze(0)
        return x.view(B, N, Hf * Wf, self.d_model)

    def _time_vec(self, t, B, N, device):
        t = torch.as_tensor(t, device=device, dtype=torch.float32)

        # allow scalar / [B] / [B,N]
        if t.ndim == 0:
            t = t.expand(B, N)
        elif t.ndim == 1:
            t = t.view(B, 1).expand(B, N)

        # continuous time in (0,1], clamp away from 0
        t = t.clamp(0.01, 1.0)

        # bin to embedding indices
        tids = ((t - 0.01) / (1.0 - 0.01) * (self.max_timesteps - 1)).round().long()
        return self.time_embed(tids)  # [B,N,d_t]

    def forward(self, R, T, feats, imgs, Ks, t, return_intermediate: bool = False):
        """
        R:[B,N,3,3], T:[B,N,3]
        feats:[B,N,1024,14,14]
        t: scalar or [B] or [B,N]
        """
        B, N = R.shape[:2]
        d = self.d_model


        feat_tok = self._feat_tokens(feats)
        view_in = torch.cat([R.reshape(B, N, 9), T], dim=-1)

        view_tok = self.view_token_init(view_in)

        cls_tok = self.cls_token_init.expand(B, N, -1)

        tokens = torch.cat([view_tok.unsqueeze(2), cls_tok.unsqueeze(2), feat_tok], dim=2)
        
        # Time embedding 
        time_vec = self._time_vec(t, B, N, device=R.device)              # [B,N,d_t]
        
        T_all = tokens.shape[2]

        R_cur, t_cur = R, T
        inter_R, inter_t = [], []

        extras = {
                "R": R_cur, 
                "T": t_cur,
                "time_emb": time_vec  
            }

        for L in range(self.num_layers):
            # PRoPE attention uses current geometry + time info
            extras["R"] = R_cur
            extras["T"] = t_cur

            x = self.blocks[L](tokens, B=B, N=N, T=T_all, extras_global=extras, Ks=Ks, block_number = L)

            x_view = x[:, :, 0, :]  # View token (geometric context)

            x_view_norm = self.output_norm(x_view)

            # Decode pose from pose token + time
            decode_in = torch.cat([x_view_norm, time_vec], dim=-1)  # [B,N,d+d_t]
            if(self.update_type == "mult"): 
                with torch.autocast(device_type="cuda", enabled=False):
                    upd = self.pose_decode(decode_in).view(B, N, 6)
                    curr = Rigid(
                        Rotation(rot_mats=R_cur.reshape(B*N, 3, 3), quats=None),
                        t_cur.reshape(B*N, 3),
                    )
                    curr = curr.compose_q_update_vec(upd.reshape(B*N, 6))
                    R_cur = curr.get_rots().get_rot_mats().reshape(B, N, 3, 3)
                    R_cur = rot9d_to_rotmat(R_cur.reshape(B, N, 9)).reshape(B, N, 3, 3)  # project to SO(3)
                    t_cur = curr.get_trans().reshape(B, N, 3)
                    pose_4x4_next = curr.to_tensor_4x4().reshape(B, N, 4, 4)

                    pose_4x4_next = pose_4x4_next.clone()
                    pose_4x4_next[..., :3, :3] = R_cur
                    pose_4x4_next[..., :3, 3]  = t_cur
            else: 
                pose_vec_next = self.pose_decode(decode_in)
                pose_4x4_next = vec_to_pose(
                    pose_vec_next.reshape(B * N, -1),
                    self.representation
                ).reshape(B, N, 4, 4)
                R_cur = pose_4x4_next[..., :3, :3]
                t_cur = pose_4x4_next[..., :3, 3]

            tokens = x

            if return_intermediate:
                inter_R.append(R_cur.clone()) 
                inter_t.append(t_cur.clone())

        out = {"R": R_cur, "t": t_cur, "pose": pose_4x4_next}
        if return_intermediate:
            out["R_layers"] = torch.stack(inter_R, dim=0)
            out["t_layers"] = torch.stack(inter_t, dim=0)
        return out

class RegressionNetwork(nn.Module):
    def __init__(self,
                 d_model: int = 1024,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 device: str = "cuda",
                 representation: str = "rot9d",
                 update_type: str = "mult",
                 feature_type = "dust3r", 
                 img_size: int = 224,
                 prediction = "pose",
                 grid_hw_img: int = 0, 
                 attn_args: Optional[dict] = None, 
                 scheme = "PRoPE",
                 attn_kwargs: Optional[dict] = None): 

        super().__init__()
        self.device = device
        self.representation = representation
        self.feature_type = feature_type
        if(feature_type == "dust3r"): 
            self.grid_hw_feat = 14
            self.num_heads = num_heads
            self.d_model = d_model
            self.num_layers = num_layers
        else: 
            self.grid_hw_feat = 16
            self.d_model = 2048
            self.num_layers = 6
            self.num_heads = 16
        
        self.scheme = scheme
        self.n_feat_tokens = self.grid_hw_feat * self.grid_hw_feat
        self.update_type = update_type

        # 2D sincos PE
        self.register_buffer("pos_feat",
            _build_2d_sincos_pos_embed(self.grid_hw_feat, self.grid_hw_feat, d_model),
            persistent=False
        )

        # View token encoder (R,T) -> d 
        self.view_token_init = nn.Sequential(
            nn.Linear(12, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        nn.init.zeros_(self.view_token_init[-1].weight)
        nn.init.zeros_(self.view_token_init[-1].bias)
        
        self.alpha_logits = nn.Parameter(torch.full((self.num_layers,), 1.3863))  # init 0.8
        
        # CLS token 
        self.cls_token_init = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        # Time embedding
        self.max_timesteps = 100

        # PRoPE blocks 
        self.blocks = nn.ModuleList([
            DiffusionAttentionBlock(
                dim=self.d_model, heads=self.num_heads, dim_head=self.d_model // self.num_heads, mlp_dim=2 * self.d_model,
                patches_x=self.grid_hw_feat, patches_y=self.grid_hw_feat,
                image_width=img_size, image_height=img_size,
                special_tokens_per_view=2, 
                dropout=dropout,
                use_local=True,
                use_s2s=True, 
                use_global=False,
                use_rayrope = False, 
                concatenate_pairs=True,
                time_dim=None
            )
            for _ in range(self.num_layers)
        ])

        # Decoder: uses token + time
        if representation == "rot9d":
            dim_out = 12
        elif representation == "rot6d":
            dim_out = 9
            
        if self.update_type == "mult":
            dim_out = 6

        self.pose_decode = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, dim_out),
        )

        self.output_norm = nn.LayerNorm(self.d_model)
        print("Initialized PRoPE Update Pose Transformer")

    def _feat_tokens(self, feats: torch.Tensor) -> torch.Tensor:
        B, N, C, Hf, Wf = feats.shape
        assert C == self.d_model and Hf == self.grid_hw_feat and Wf == self.grid_hw_feat
        x = feats.view(B * N, C, Hf, Wf)
        #x = self.feat_channel_proj(x)                         # [B*N, d, Hf, Wf]
        x = x.flatten(2).transpose(1, 2)       # [B*N, Tf, d]
        
        if not self.scheme == "PRoPE":
            x = x + self.pos_feat.to(x.device).unsqueeze(0)
        return x.view(B, N, Hf * Wf, self.d_model)

    def forward(self, feats, imgs, Ks, return_intermediates: bool = False):
        """
        R:[B,N,3,3], T:[B,N,3]
        feats:[B,N,1024,14,14]
        t: scalar or [B] or [B,N]
        """
        B, N = feats.shape[:2]
        device = feats.device
        R = torch.eye(3, device=device).expand(B, N, 3, 3).clone()
        T = torch.zeros((B, N, 3), device=device)

        d = self.d_model

        # Initialize tokens
        feat_tok = self._feat_tokens(feats)                              # [B,N,P,d]
        

        view_tok = self.view_token_init(
            torch.cat([R.reshape(B, N, 9), T], dim=-1)
        )  # [B,N,d]
        
        # CLS token
        cls_tok = self.cls_token_init.expand(B, N, -1)                   # [B,N,d]

        # Build per-view sequence: [view_tok, cls_tok, feature_tokens]
        tokens = torch.cat([
            view_tok.unsqueeze(2),  # [B,N,1,d]
            cls_tok.unsqueeze(2),   # [B,N,1,d]
            feat_tok                # [B,N,P,d]
        ], dim=2)  # [B,N,2+P,d]
        
        T_all = tokens.shape[2]

        R_cur, t_cur = R, T
        inter_R, inter_t = [], []

        extras = {
                "R": R_cur, 
                "T": t_cur,
            }

        for L in range(self.num_layers):
            alpha = torch.sigmoid(self.alpha_logits[L]).to(tokens.dtype)  # scalar for this layer
            extras["R"] = R_cur
            extras["T"] = t_cur

            x = self.blocks[L](tokens, B=B, N=N, T=T_all, extras_global=extras, Ks=Ks)

            x_view = x[:, :, 0, :]  # View token (geometric context)

            x_view_norm = self.output_norm(x_view)

            decode_in = x_view_norm
            if(self.update_type == "mult"): 
                upd = self.pose_decode(decode_in).view(B, N, 6)
                curr = Rigid(
                    Rotation(rot_mats=R_cur.reshape(B*N, 3, 3), quats=None),
                    t_cur.reshape(B*N, 3),
                )
                curr = curr.compose_q_update_vec(upd.reshape(B*N, 6))
                R_cur = curr.get_rots().get_rot_mats().reshape(B, N, 3, 3)
                t_cur = curr.get_trans().reshape(B, N, 3)
                pose_4x4_next = curr.to_tensor_4x4().reshape(B, N, 4, 4)
            else: 
                pose_vec_next = self.pose_decode(decode_in)
                pose_4x4_next = vec_to_pose(
                    pose_vec_next.reshape(B * N, -1),
                    self.representation
                ).reshape(B, N, 4, 4)
                R_cur = pose_4x4_next[..., :3, :3]
                t_cur = pose_4x4_next[..., :3, 3]

            Rt = R_cur
            tt = t_cur
            view_tok_fresh = self.view_token_init(torch.cat([Rt.reshape(B, N, 9), tt], dim=-1))
            tokens = x.clone()

            tok0 = tokens[:, :, 0, :]                               # [B,N,D] view
            tok0_new = alpha * tok0 + (1 - alpha) * view_tok_fresh  # [B,N,D]

            tokens = torch.cat([tok0_new.unsqueeze(2), tokens[:, :, 1:, :]], dim=2)

            if return_intermediates:
                inter_R.append(R_cur.clone()) 
                inter_t.append(t_cur.clone())

        out = {"R": R_cur, "t": t_cur, "pose": pose_4x4_next}
        if return_intermediates:
            out["R_layers"] = torch.stack(inter_R, dim=0)
            out["t_layers"] = torch.stack(inter_t, dim=0)
        return out