import torch
import torch.nn as nn
from spation_attention_2d import SpatialAttention2D
from cross_attention_2d import CrossAttention2D

class KalmanFormerNet(nn.Module):
    def __init__(self, dim, num_heads: int = 8, H_patch: int = 7, W_patch: int = 7):
        super().__init__()
        self.dim = dim
        self.H_patch = H_patch
        self.W_patch = W_patch

        # Encoder: self-attention on [innovation, state_evol_diff] (2D)
        self.encoder = SpatialAttention2D(dim=dim * 2, num_heads=num_heads)

        # Decoder: cross-attention
        self.decoder = CrossAttention2D(dim=dim, num_heads=num_heads)

        # Projections
        self.query_proj = nn.Linear(dim * 2, dim)
        self.gain_proj = nn.Linear(dim, dim)

    def forward(self, encoder_input, decoder_kv):
        """
        encoder_input: [B, 1, N, 2*D] - [innovation, state_evol_diff]
        decoder_kv: [B, 1, N, 2*D] - [evolution_diff, state_update_diff]
        """
        B, seq_len, N, _ = encoder_input.shape

        # Remove sequence dimension
        enc_in = encoder_input[:, 0]  # [B, N, 2D]
        dec_kv = decoder_kv[:, 0]     # [B, N, 2D]

        # Encoder: self-attention

        context = self.encoder(enc_in, self.H_patch, self.W_patch)  # [B, N, 2D]

        # Project encoder output to query
        query = self.query_proj(context)  # [B, N, D]

        # Split decoder key/value
        key = dec_kv[..., :self.dim]      # [B, N, D] - evolution_diff
        value = dec_kv[..., self.dim:]    # [B, N, D] - state_update_diff

        # Decoder: cross-attention
        gain = self.decoder(query, key, value, self.H_patch, self.W_patch)  # [B, N, D]

        # Final projection
        gain = self.gain_proj(gain)  # [B, N, D]

        return gain.unsqueeze(1)  # [B, 1, N, D]