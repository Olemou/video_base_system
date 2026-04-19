import math
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.kalman_former_net import KalmanFormerNet
from  app.spation_attention_2d import SpatialAttention2D
from  app.vision_temporal_attention import VisionTemporalAttention
from  app.tokenlearner import TokenLearner
from app.patch_embedding import PatchEmbedding3D, PatchMerging
from  app.gru_customized import TemporalSpatialStateGRU
from app.spatial_temporal_attention import RoPEAttention
from src.src_utils.vision_config import VisionConfig
from src.src_utils.utils import trunc_normal_

   
class AttentionBlock(nn.Module):
    def __init__(self, config, qkv_bias=True,return_attn=True):
        super().__init__()


        self.norm_spatial = nn.LayerNorm(config.embed_dim)
        self.norm_temporal = nn.LayerNorm(config.embed_dim)
        self.norm_kalman = nn.LayerNorm(config.embed_dim)
        self.norm_gru = nn.LayerNorm(config.embed_dim)
        self.norm_mlp = nn.LayerNorm(config.embed_dim)
        self.norm_spatial_temporal = nn.LayerNorm(config.embed_dim)

        self.dropout = nn.Dropout(config.dropout)
        self.return_attn = return_attn

        # modules
        self.spatial_attn = SpatialAttention2D(config, qkv_bias=qkv_bias)
        self.temporal_attn = VisionTemporalAttention(config)
        self.kalmanformerNet = KalmanFormerNet(config)
        self.patchmerging = PatchMerging(config)
        self.gru = TemporalSpatialStateGRU(config)
        self.spatial_temporal_attn = RoPEAttention(config, qkv_bias=qkv_bias)
        self.attn_weight = None 

        self.mlp = MLP(config.embed_dim, config.mlp_dim, config.embed_dim, drop=config.dropout)

    def forward(self, x, block_index=0):

        # =========================
        # 1. Spatial Attention (GPT style)
        # =========================
        x = x + self.dropout(self.spatial_attn(self.norm_spatial(x)))

        # =========================
        # 2. Patch Merging (outside norm logic)
        # =========================
        if block_index == 0:
            x = self.patchmerging(x)
        
        # =========================
        # 3. Kalman branch (residual style)
        # =========================
        x_kalman, cu_seqlens = self.kalmanformerNet(self.norm_kalman(x))
        x = x + x_kalman

        # =========================
        # 4. Temporal Attention (GPT style)
        # =========================
        x = x + self.dropout(
            self.temporal_attn(self.norm_temporal(x), cu_seqlens)
        )

        # =========================
        # 5. GRU fusion (residual)
        # =========================
        x_gru = self.gru(self.norm_gru(x), x)
        x = x + self.dropout(x_gru)
        
        # spatial_temporal_attention_heads = 8
        B, T, N, C = x.shape
        x = x.view(B, T * N, C)  # [B, T*N, C]
                  
        tuple_out = self.spatial_temporal_attn(x = self.norm_spatial_temporal(x),return_attn=self.return_attn, T= T, H_patches= int(math.sqrt(N)), W_patches= int(math.sqrt(N)))
        
        
        spatial_temporal_output, self.attn_weight = tuple_out
        x = x + self.dropout(spatial_temporal_output)
        
        # =========================
        # 6. MLP 
        # =========================
        x = x + self.dropout(self.mlp(self.norm_mlp(x)))
        x = x.reshape(B, T, N, C)
        return x
    
   
class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class ProjectionHead(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.projection  = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.GELU(),
                nn.Linear(config.embed_dim, config.projection_dim)
            )
                    
        nn.Linear(config.embed_dim, config.projection_dim)

    def forward(self, x):
        # x: # [B, N, D]
        B, N, D = x.shape
        x = x.view(B * N, D)  # [B*N, D]
        x = self.projection(x)     # [B*N, projection_dim]
        x = x.view(B, N, -1)   # [B, N, projection_dim]
        x = F.normalize(x, dim=-1)
        return x
    
class KalmanFormerNetVideoModel(nn.Module):
    def __init__(self, config: VisionConfig,
                 qkv_bias=True, return_attn=False, init_type="xavier_uniform", init_std=0.02):
        super().__init__()
        self.patch_embedding = PatchEmbedding3D(config)
        self.head = ProjectionHead(config)  
        self.attn_layers = nn.ModuleList(
            [AttentionBlock(
                config = config,
            qkv_bias=qkv_bias, return_attn=return_attn
        ) for _ in range(config.depth)]
        )
        self.init_type = init_type
        self.init_std = init_std
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            return
        if self.init_type == "default":
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=self.init_std)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif self.init_type == "xavier_uniform":
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv3d)
            ):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif self.init_type == "xavier_normal":
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv3d)
            ):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            raise ValueError(f"Unknown init type {self.init_type}")

    def forward(self, video):
        # Patch embedding
        x = self.patch_embedding(video)
        # Attention block
        for index, block in enumerate(self.attn_layers):
            x = block(x, index)
        weights = torch.softmax(x.mean(dim=-1).mean(dim=-1), dim=1)  # [B, T]
        z_t = torch.einsum('btnd,bt->bnd', x, weights)          # [B, N, D]
        out = self.head(z_t)  # [B, N, projection_dim]
        return out
    
# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
config = VisionConfig()
model = KalmanFormerNetVideoModel(config, device).to(device)

# Optional: test with dummy input
if __name__ == "__main__":
    # Example video tensor: [B, C, T, H, W]
    dummy_video = torch.randn(2, 3, 4, 224, 224).to(device)
    
    # Forward pass
    output = model(dummy_video)
    
    attn = model.attn_layers[0].attn_weight
    if attn is not None:
        print("Attention weights shape:", attn.shape)
    print("output.shape:", output.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")