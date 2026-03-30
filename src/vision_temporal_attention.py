
import math
import torch
import torch.nn as nn
from .rotation_embedding_1d import RotaryEmbedding1D, apply_rotary_1d
from .vision_config import VisionConfig
class VisionTemporalAttention(nn.Module):


    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.hidden_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads_temporal_attn
        self.num_heads = max(1, round(config.num_heads_temporal_attn))
        self.neck_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(self.hidden_dim, self.neck_dim * 3, bias=True)
        self.proj = nn.Linear(self.neck_dim, self.hidden_dim)
        self.scaling = self.head_dim **-0.5
        self.attn_weight_output = None  # For visualization

        self.RotaryEmbedding1D = RotaryEmbedding1D(self.hidden_dim, self.num_heads)
        
    def _build_block_mask(self, cu_seqlens: torch.Tensor, seq_len: int, device: torch.device):
        """
        Block-diagonal mask: tokens attend freely within the same video (all frames),
        but cannot attend across different videos.
        """
        indices = torch.arange(seq_len, device=device)
        # seq_id[i] = which video token i belongs to
        seq_id = torch.bucketize(indices, cu_seqlens[1:-1])
        mask = seq_id.unsqueeze(0) == seq_id.unsqueeze(1)  # True if same video
        mask = torch.where(mask, 0.0, float("-inf"))       # 0 for allowed, -inf for blocked
        return mask

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
        return_attn: bool = True
    ) -> torch.Tensor:
        B, T, N, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])  # [B * T_patch * K, C]

        qkv = self.qkv(hidden_states)
        seq_length = hidden_states.shape[0]

        query_states, key_states, value_states = (
            qkv.reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        # -------------------
        # Rotary embeddings
        # -------------------
        cos, sin = self.RotaryEmbedding1D(seq_length)
        query_states, key_states = apply_rotary_1d(
            query_states, key_states, cos, sin
        )

        # -------------------
        # reshape → [1, H, L, D]
        # -------------------
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states   = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # -------------------
        # Attention scores
        # -------------------
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling
        # shape: [1, H, L, L]

        # -------------------
        # Block mask (replaces splitting)
        # -------------------
        attn_mask = self._build_block_mask(cu_seqlens, seq_length, hidden_states.device)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        attn_weights = attn_weights + attn_mask

        self.attn_weight_output = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        weighted_output = torch.matmul(self.attn_weight_output, value_states)
        weighted_output = weighted_output.transpose(1, 2)

        output = weighted_output.reshape(seq_length, -1)
        output = self.proj(output)
        final_output = output.reshape(B, T, N, output.shape[-1])
        if return_attn:
            return final_output, self.attn_weight_output
        return final_output