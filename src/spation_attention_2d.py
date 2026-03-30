# =========================================================
# Spatial MHSA with 2D RoPE
# =========================================================
import torch
import torch.nn as nn
from rotation_embedding_2d import RotaryEmbedding2D, apply_rotary_2d  
from vision_config import VisionConfig  

class SpatialAttention2D(nn.Module):
    def __init__(self, config: VisionConfig, qkv_bias=True, isFilter=False):
        super().__init__()
        if isFilter:
            self.num_heads = config.number_heads_spatial_kalman_attn
            self.H_patch, self.W_patch = config.h_patch, config.w_patch
        else:
            self.num_heads = config.num_heads_spatial_attn
            self.H_patch, self.W_patch = config.h_patch_after_patch_embedding, config.w_patch_after_patch_embedding

        assert config.embed_dim % self.num_heads == 0
        self.dim = config.embed_dim
        self.head_dim = config.embed_dim // self.num_heads

        self.norm = nn.LayerNorm(self.dim)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)

        self.rope = RotaryEmbedding2D(self.head_dim)


    def forward(self, x):
        BxT, N, C = x.shape

        qkv = self.qkv(x).reshape(BxT, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ys = torch.arange(self.H_patch, device=x.device)
        xs = torch.arange(self.W_patch, device=x.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        h_idx, w_idx = yy.reshape(-1), xx.reshape(-1)

        cos, sin = self.rope.forward(h_idx, w_idx)
        q, k = apply_rotary_2d(q, k, cos, sin)


        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(BxT, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
    
