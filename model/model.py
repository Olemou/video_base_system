import torch
import torch.nn as nn
from src import (
    VisionConfig, SpatialAttention2D, VisionTemporalAttention,
    TokenLearner, KalmanFormerNet, PatchMerging,
    TemporalSpatialStateGRU, PatchEmbedding3D
)

class AttentionBlock(nn.Module):
    def __init__(self,
                 config: VisionConfig, device: torch.device, qkv_bias=True, return_attn=False, dropout_prob=0.1):
        super().__init__()
        
        self.device = device
        self.return_attn = return_attn

        # Layer norms
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.norm3 = nn.LayerNorm(config.embed_dim)
        self.norm4 = nn.LayerNorm(config.embed_dim)
        self.norm_mlp = nn.LayerNorm(config.embed_dim)

        # Dropout (always nn.Dropout, auto-disabled in eval)
        self.dropout = nn.Dropout(dropout_prob)

        # Attention and state modules
        self.spatial_attn = SpatialAttention2D(config, qkv_bias=qkv_bias)
        self.temporal_attn = VisionTemporalAttention(config, qkv_bias=qkv_bias)
        self.kalmanformerNet = KalmanFormerNet(config, device)
        self.token_learner = TokenLearner(config)
        self.patchmerging = PatchMerging(config)
        self.gru = TemporalSpatialStateGRU(config)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.mlp_dim, config.embed_dim),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        attn_weights = None
        B = self.batch_size
       
        # === Spatial Attention with residual + dropout ===
        x = x + self.dropout(self.spatial_attn(self.norm1(x)))

        # === Patch Merging ===
        x = self.patchmerging(x)  # reduces H*W internally
        x = self.norm2(x)

        # === Kalman Filter ===
        x_kalman, cu_seqlens = self.kalmanformerNet(x)
        x_kalman = self.norm2(x_kalman)

        # === Temporal Attention (first pass) ===
        x_temp = self.temporal_attn(self.norm3(x), cu_seqlens, return_attn=False)
        x = x_kalman + self.dropout(x_temp)

        # === GRU fusion ===
        x_gru, _ = self.gru(x_kalman, x)
        x = x + self.dropout(x_gru)
        x = self.norm4(x)

        # === Token Learner ===
        x, cu_seqlens = self.token_learner(x)
        x = self.norm4(x)

        # === Temporal Attention AFTER GRU ===
        if not self.training and self.return_attn:
            x_attn, attn_weights = self.temporal_attn(self.norm4(x), cu_seqlens, return_attn=True)
            x = x + self.dropout(x_attn)
        else:
            x = x + self.dropout(self.temporal_attn(self.norm4(x), cu_seqlens, return_attn=False))

        # === Final MLP with residual ===
        x = self.norm_mlp(x)
        x = x + self.mlp(x)

        if not self.training and self.return_attn:
            return x, attn_weights
        return x


class visionVideoTransformer(nn.Module):
    def __init__(self, config: VisionConfig, device: torch.device,
                 qkv_bias=True, return_attn=False, dropout_prob=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding3D(config)
        self.attention_block = AttentionBlock(
                config = config, device = device,
            qkv_bias=qkv_bias, return_attn=return_attn, dropout_prob=dropout_prob
        )
        self.number_attn_layer = config.number_attn_layer

    def forward(self, video):
        # Patch embedding
        x = self.patch_embedding(video)
        # Attention block
        for _ in range(self.number_attn_layer):
            x = self.attention_block(x)
        return x