import torch
import torch.nn as nn
from app.rotation_embedding_1d import RotaryEmbedding1D, apply_rotary_1d
from app.kalman_shift_mask import build_kalman_shifted_mask
from src.src_utils.vision_config import VisionConfig

class temporalShiftedAttentionSignal(nn.Module):
    def __init__(self, config: VisionConfig, qkv_bias=True, use_rotary=True, theta=10000.0):
        """
        Args:
            x: [seq_len, dim] - Input tensor
            use_rotary: Whether to use rotary positional embeddings
            theta: Base period for rotary embeddings

        Returns:
            attn: Attention weights
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.num_heads = config.num_heads_temporal_attn
        self.head_dim = config.embed_dim // config.num_heads_temporal_attn
        self.scale = self.head_dim ** -0.5
        # Rotary embeddings
        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary_emb = RotaryEmbedding1D(config.embed_dim, config.num_heads_temporal_attn, theta)

    def forward(self, x, cu_seqlens, patch_len, number_of_tokens, device):
        """ args:
            x: [L, C] where L = B * T_patch * K
            cu_seqlens: Cumulative sequence lengths for each batch (for masking)
            patch_len: Length of each temporal patch
            number_of_tokens: Total number of tokens in the sequence
            device: Device to run the model on
        returns:
            attn: Attention weights
        """

        L, C = x.shape

        # Get rotary embeddings if enabled
        cos, sin = None, None
        if self.use_rotary:
            cos, sin = self.rotary_emb(L)


        # QKV projection
        qkv = self.qkv(x).reshape(L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)  # [3, L, num_heads, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings
        if self.use_rotary:
            q, k = apply_rotary_1d(q, k, cos, sin)

        # Attention
        q = q.transpose(0, 1)  # [num_heads, L, head_dim]
        k = k.transpose(0, 1)  # [num_heads, L, head_dim]
        v = v.transpose(0, 1)  # [num_heads, L, head_dim]

        attn = (q @ k.transpose(-2, -1))
        mask = build_kalman_shifted_mask(cu_seqlens, patch_len, number_of_tokens, device)

        attn = (attn + mask.unsqueeze(0)) * self.scale  # [num_heads, L, L]

        attn = attn.softmax(dim=-1)
        
        # Debug your mask

        #out = attn @ v  # [num_heads, L, head_dim]

        # Reshape and project
        #out = out.transpose(0, 1).reshape(L, C)  # [L, C]
        #out = self.proj(out)

        return attn