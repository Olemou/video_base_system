import torch
import torch.nn as nn
from rotation_embedding_2d import RotaryEmbedding2D, apply_rotary_2d  
from vision_config import VisionConfig
  
class CrossAttention2D(nn.Module):

  def __init__(self, config: VisionConfig, qkv_bias=True):
        super().__init__()
        assert config.embed_dim % config.number_heads_cross_attn == 0
        self.dim = config.embed_dim
        self.num_heads = config.number_heads_cross_attn
        self.head_dim = self.dim // self.num_heads

        self.query_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.key_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.value_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)
        
        self.rope = RotaryEmbedding2D(self.head_dim)

  def forward(self, query, key, value, H_patch, W_patch):
        B, N, D = query.shape

        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        ys = torch.arange(H_patch, device=query.device)
        xs = torch.arange(W_patch, device=query.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        h_idx, w_idx = yy.reshape(-1), xx.reshape(-1)

        cos, sin = self.rope.forward(h_idx, w_idx)
        q, k = apply_rotary_2d(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out