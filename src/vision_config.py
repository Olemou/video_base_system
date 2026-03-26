class VisionConfig:
  def __init__(self, embed_dim=768, num_heads_spatial_attn=12, num_heads_temporal_attn=4, number_heads_cross_attn=4, dropout=0.1, 
               num_layers=12, num_tokens=16, bottleneck_dim=256,
               number_heads_spatial_kalman_attn=4,
               patch_size=16, temporal_patch_size=4,
               channel=3,
               ):
        self.embed_dim = embed_dim
        self.num_heads_spatial_attn = num_heads_spatial_attn
        self.num_heads_temporal_attn = num_heads_temporal_attn
        self.number_heads_cross_attn = number_heads_cross_attn
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.bottleneck_dim = bottleneck_dim
        self.number_heads_spatial_kalman_attn = number_heads_spatial_kalman_attn
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.channel = channel
        self.mlp_dim = 4 * embed_dim
        