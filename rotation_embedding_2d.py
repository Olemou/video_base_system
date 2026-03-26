import torch
import torch.nn as nn
from helper import rotate_half

class RotaryEmbedding2D(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        """2D RoPE implementation for vision transformers."""
        super().__init__()
        assert head_dim % 4 == 0, "For 2D RoPE, head_dim must be divisible by 4"
        self.head_dim = head_dim
        self.quarter = head_dim // 4

        inv_freq = 1.0 / (theta ** (torch.arange(0, self.quarter).float() / self.quarter))
        self.register_buffer("inv_freq", inv_freq)

    def get_cos_sin_1d(self, positions):
        freqs = torch.outer(positions.float(), self.inv_freq)   # [L, quarter]
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)   # [L, 2*quarter]
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)   # [L, 2*quarter]
        return cos, sin

    def forward(self, h_idx, w_idx):
        """_summary_

        Args:
            h_idx (_type_): _height positions, shape [L]
            w_idx (_type_): _width positions, shape [L]

        Returns:
            _type_: cos and sin tensors for height and width dimensions, each of shape [L, head_dim]
        """
        cos_h, sin_h = self.get_cos_sin_1d(h_idx)
        cos_w, sin_w = self.get_cos_sin_1d(w_idx)
        cos = torch.cat([cos_h, cos_w], dim=-1)  # [L, head_dim]
        sin = torch.cat([sin_h, sin_w], dim=-1)  # [L, head_dim]
        return cos, sin
    
def apply_rotary_2d(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k