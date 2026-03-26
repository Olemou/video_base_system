
import torch
import torch.nn as nn
from src import VisionConfig, SpatialAttention2D, VisionTemporalAttention, PatchEmbedding3D, PatchMerging, temporalShiftedAttentionSignal ,select_tokens_from_attn, TokenLearner, KalmanFormerNet

class AttentionBlock(nn.Module):
    def __init__(self, config: VisionConfig, qkv_bias=True):
        super().__init__()
        self.spatial_attn = SpatialAttention2D(config, qkv_bias=qkv_bias)
        self.temporal_attn = VisionTemporalAttention(config, qkv_bias=qkv_bias)
        self.kalmanformerNet = KalmanFormerNet(config)
        
        