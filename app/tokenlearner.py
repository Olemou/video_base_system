import torch
import torch.nn.functional as F
import torch.nn as nn
from src.src_utils.vision_config import VisionConfig

class MlpBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        if config.mlp_dim is None:
            config.mlp_dim = 4 * config.embed_dim
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.num_tokens)
        self.act = nn.GELU()
        if config.dropout and self.training > 0:
            self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
    
# =========================================================
# TokenLearner
# =========================================================
class TokenLearner(nn.Module):
    def __init__(self, config: VisionConfig):
        """_summary_

        Args:
            embed_dim (int): _description_
            num_tokens (int, optional): _description_. Defaults to 16.
            bottleneck_dim (int, optional): _description_. Defaults to 256.
        """
        super().__init__()
        self.num_tokens = config.num_tokens
        self.norm = nn.LayerNorm(config.embed_dim)

        self.mask = MlpBlock(config)  # [B, T, N, C] -> [B, T, N, K]
    def forward(self, x):
        """
        x: [B, T, N, C]
        returns:
            tokens: [B, T, K, C]
        """
        x = self.norm(x)                    # [B, T, N, C]
        scores = self.mask(x)               # [B, T, N, K]
        scores = scores.transpose(2, 3)     # [B, T, K, N]
        probs = F.softmax(scores, dim=-1)   # [B, T, K, N]

        tokens = torch.einsum("btkn,btnc->btkc", probs, x)  # [B, T, K, C]
        with torch.no_grad():
            cu_seqlen = torch.arange(0, (tokens.shape[0] + 1) * tokens.shape[1] * tokens.shape[2], tokens.shape[1] * tokens.shape[2],
                              device=x.device, dtype=torch.int32)

        return tokens, cu_seqlen