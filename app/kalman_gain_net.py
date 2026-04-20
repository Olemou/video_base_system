import torch
import torch.nn as nn
from app.spation_attention_2d import SpatialAttention2D
from app.cross_attention_2d import CrossAttention2D
from  src.src_utils.vision_config import VisionConfig

class kalmanGainNet(nn.Module):
    """_summary_
        Args:
            config (VisionConfig): _description_    
        returns:
        K_gain [B, 1, N, D] Kalman gain for each token
        """
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.dim = config.embed_dim
        
        # Encoder: self-attention on [innovation, state_evol_diff] (2D)
        self.encoder = SpatialAttention2D(config=config, isFilter=True)

        # Decoder: cross-attention
        self.decoder = CrossAttention2D(config=config)

        # Projections
        self.query_proj = nn.Linear(self.dim * 2, self.dim)
        self.gain_proj = nn.Linear(self.dim, self.dim)
    def forward(self, encoder_input, decoder_kv):
        """
        encoder_input: [B, 1, N, 2*D] - [innovation, state_evol_diff]
        decoder_kv: [B, 1, N, 2*D] - [evolution_diff, state_update_diff]
        """
        B, seq_len, N, _ = encoder_input.shape
        # [B, 1, N, 2*D] --- IGNORE ---

        # Remove sequence dimension
        enc_in = encoder_input  # [B, T,N, 2D]
        dec_kv = decoder_kv[:, 0]     # [B, N, 2D]

        # Encoder: self-attention

        context = self.encoder(enc_in)  # [B, N, 2D]

        # Project encoder output to query
        query = self.query_proj(context)  # [B, N, D]

        # Split decoder key/value
        key = dec_kv[..., :self.dim]      # [B, N, D] - evolution_diff
        value = dec_kv[..., self.dim:]    # [B, N, D] - state_update_diff

        # Decoder: cross-attention
        gain = self.decoder(query, key, value)  # [B, N, D]

        # Final projection
        gain = self.gain_proj(gain)  # [B, N, D]

        return gain.unsqueeze(1)  # [B, 1, N, D]