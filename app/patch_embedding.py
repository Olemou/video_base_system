import torch
import torch.nn as nn
import torch.nn.functional as F
from src.src_utils.vision_config import VisionConfig
import math

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
        return x
    


class PatchMerging(nn.Module):
    def __init__(self, config, output_dim=None):
        super().__init__()
        output_dim = output_dim or config.embed_dim
        self.linear = nn.Linear(4 * config.embed_dim, output_dim)

    def forward(self, x):
        """
        Input:  (B, T, N, C)
        Output: (B, T, N_new, output_dim)
        """
        B, T, N, C = x.shape

        # --- Step 1: find closest grid ---
        H = int(math.sqrt(N))
        W = math.ceil(N / H)

        # adjust H again to fit
        H = math.ceil(N / W)

        total = H * W
        pad_tokens = total - N

        # --- Step 2: pad if needed ---
        if pad_tokens > 0:
            pad = torch.zeros(B, T, pad_tokens, C, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=2)

        # --- Step 3: reshape to grid ---
        x = x.view(B, T, H, W, C)

        # --- Step 4: ensure even dims for 2x2 ---
        if H % 2 != 0 or W % 2 != 0:
            new_H = H + (H % 2)
            new_W = W + (W % 2)

            pad_h = new_H - H
            pad_w = new_W - W

            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # pad H and W
            H, W = new_H, new_W

        # --- Step 5: 2x2 merging ---
        x = x.view(B, T, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, T, H // 2, W // 2, 4 * C)

        # --- Step 6: linear projection ---
        x = self.linear(x)

        # --- Step 7: flatten back ---
        N_new = (H // 2) * (W // 2)
        x = x.view(B, T, N_new, -1)

        return x