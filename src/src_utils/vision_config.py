
class VisionConfig:
  def __init__(self, embed_dim: int = 768, num_heads_spatial_attn: int = 12, num_heads_temporal_attn: int = 4, number_heads_cross_attn: int = 4, dropout: float = 0.1,
               num_layers: int = 12, num_tokens: int = 16, bottleneck_dim: int = 256,
               number_heads_spatial_kalman_attn: int = 4,
               patch_size: int = 16, temporal_patch_size: int = 4,
               channel: int = 3,
               gready_token_threshold: float = 0.01,
               h_patch_after_patch_embedding: int = 14,
               w_patch_after_patch_embedding: int = 14,
               h_patch: int = 7,
               w_patch: int = 7,
               depth: int = 12,
               proj_drop: float = 0.1,
               attn_drop: float = 0.1,
               use_sdpa=False,
               is_causal=False,
               grid_size=14,
               n_registers=0,
                has_cls_first=False,
                 interpolate_rope=False,
                 qkv_bias=False,
                 spatial_temporal_attention_heads=8,
                 grid_size_spatial_temporal=7,
                 projection_dim =256
               ):
        """_summary_

        Args:
            embed_dim (int): _description_. Defaults to 768.
            num_heads_spatial_attn (int): _description_. Defaults to 12.
            num_heads_temporal_attn (int): _description_. Defaults to 4.
            number_heads_cross_attn (int): _description_. Defaults to 4.
            dropout (float, optional): _description_. Defaults to 0.1.
            num_layers (int, optional): _description_. Defaults to 12.
            num_tokens (int, optional): _description_. Defaults to 16.
            bottleneck_dim (int, optional): _description_. Defaults to 256.
            number_heads_spatial_kalman_attn (int, optional): _description_. Defaults to 4.
            patch_size (int, optional): _description_. Defaults to 16.
            temporal_patch_size (int, optional): _description_. Defaults to 4.
            channel (int, optional): _description_. Defaults to 3.
            gready_token_threshold (float, optional): _description_. Defaults to 0.01.
            h_patch (int): _description_. Defaults to 7.
            w_patch (int,): _description_. Defaults to 7.
        """
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
        self.gready_token_threshold = gready_token_threshold
        self.h_patch = h_patch
        self.w_patch = w_patch
        self.h_patch_after_patch_embedding = h_patch_after_patch_embedding
        self.w_patch_after_patch_embedding = w_patch_after_patch_embedding
        self.embeding_gru = 2 * embed_dim
        self.depth = depth
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal
        self.grid_size = grid_size
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first
        self.interpolate_rope = interpolate_rope
        self.qkv_bias = qkv_bias
        self.spatial_temporal_attention_heads = spatial_temporal_attention_heads
        self.grid_size_spatial_temporal = grid_size_spatial_temporal
        self.projection_dim = projection_dim
        