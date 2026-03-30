import torch
import torch.nn as nn
import torch.nn.functional as F
from vision_config import VisionConfig

# =========================================================
# Patch Embedding
# =========================================================
class PatchEmbedding3D(nn.Module):
    def __init__(
        self,
        config: VisionConfig
    ):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim

        kernel_size = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(
            config.channel,
            config.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def forward(self, video):
        """
        video: [B, C, T, H, W]
        returns:
            x: [B*T_patch, N_frame, C]
            T_patch, H_patch, W_patch
        """
        x = self.proj(video)   # [B, C, T', H', W']
        x = x.flatten(3)                # [B, C, T', H'*W']
        x = x.permute(0, 2, 3, 1)       # [B, T', N_frame, C]
        B, T_patch, N_frame, C = x.shape
        x = x.reshape(B * T_patch, N_frame, C)
    
        return x
    
    

class PatchMerging(nn.Module):
    def __init__(self, config: VisionConfig, output_dim: int = None):
        """
        Args:
            dim: input channel dimension
            output_dim: output channel dimension (default: dim)
        """
        super().__init__()
        output_dim = output_dim or config.embed_dim
        self.linear = nn.Linear(4 * config.embed_dim, output_dim)

    def forward(self, x):
        """
        Input:  (B, T, N, C) where N = H * W
        Output: (B, T, N_new, output_dim) where N_new = (H//2) * (W//2)
        """
        B, T, N, C = x.shape


        H = int(N ** 0.5)
        W = H
        assert H * W == N, f"N={N} must be a perfect square"

        # Reshape to spatial grid
        x = x.view(B, T, H, W, C)

        # Group into 2x2 patches
        x = x.view(B, T, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()

        # Merge spatial patches
        x = x.view(B, T, H // 2, W // 2, 4 * C)

        x = self.linear(x)
        N_new = (H // 2) * (W // 2)
        x = x.view(B, T, N_new, -1)

        return x