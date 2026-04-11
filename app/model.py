import torch
import torch.nn as nn
import os
import sys
# Add parent folder to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import (
    VisionConfig, SpatialAttention2D, VisionTemporalAttention,
    TokenLearner, KalmanFormerNet, PatchMerging,
    TemporalSpatialStateGRU, PatchEmbedding3D
)

class AttentionBlock(nn.Module):
    def __init__(self,
                 config: VisionConfig, device: torch.device, qkv_bias=True, return_attn=True, dropout_prob=0.1):
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
        self.temporal_attn = VisionTemporalAttention(config)
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
        x,cu_seqlens= self.token_learner(x)
        x = self.norm4(x)
        x = x + self.dropout(self.temporal_attn(self.norm4(x), cu_seqlens, return_attn=False))
        return x

class Mlp(nn.Module):
    def __init__(self, embed_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = embed_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
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
        self.blocks = nn.ModuleList(
            [AttentionBlock(
                config = config, device = device,
            qkv_bias=qkv_bias, return_attn=return_attn, dropout_prob=dropout_prob
        ) for _ in range(config.depth)]
        )
        self.mlp = Mlp(config.embed_dim)


    def forward(self, video):
        # Patch embedding
        x = self.patch_embedding(video)
        # Attention block
        for block in self.blocks:
            x = block(x)
        x = self.mlp(x)
        weights = torch.softmax(x.mean(dim=-1).mean(dim=-1), dim=1)  # [B, T]
        z_t = torch.einsum('btnd,bt->bnd', x, weights)          # [B, N, D]
        return z_t
    
# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
config = VisionConfig()
model = visionVideoTransformer(config, device).to(device)

# Optional: test with dummy input
if __name__ == "__main__":
    # Example video tensor: [B, C, T, H, W]
    dummy_video = torch.randn(2, 3, 4, 224, 224).to(device)
    
    # Forward pass
    output = model(dummy_video)
    print(output.shape)