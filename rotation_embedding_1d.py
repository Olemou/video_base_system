import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import rotate_half

class RotaryEmbedding1D(nn.Module):
      def __init__(self, dim: int, num_heads: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.theta = theta

        # Standard RoPE frequency computation (consecutive pairing)
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
      def forward(self, seqlen: int):
          """Compute RoPE embeddings for a given sequence length."""
          seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
          freqs = torch.outer(seq, self.inv_freq)                 # [L, head_dim//2]
          cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)   # [L, 2*quarter]
          sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)   # [L, 2*quarter]
          cos = cos[:, None, :].expand(seqlen, self.num_heads, self.head_dim)
          sin = sin[:, None, :].expand(seqlen, self.num_heads, self.head_dim)
          return cos, sin
      
def apply_rotary_1d(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
