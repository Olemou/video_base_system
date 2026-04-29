import torch
import torch.nn as nn
from src.src_utils.vision_config import VisionConfig

class TemporalSpatialStateGRU(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.embed_dim

        self.W_z = nn.Linear(config.embeding_gru, config.embed_dim)
        self.W_r = nn.Linear(config.embeding_gru, config.embed_dim)
        self.W_h = nn.Linear(config.embeding_gru, config.embed_dim)
        
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.old_to_embed = nn.Linear(config.embed_dim, config.embed_dim)
        
        self.delta_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.update_bias = nn.Parameter(torch.zeros(config.embed_dim))

    def forward(self, candidate, old_state):
        combined = torch.cat([candidate, old_state], dim=-1)
        
        delta_signal = self.delta_proj(torch.abs(candidate - self.old_to_embed(old_state)))
        delta_signal = delta_signal / (torch.norm(delta_signal, dim=-1, keepdim=True) + 1e-6)
        
        update_gate = torch.sigmoid(self.W_z(combined) + delta_signal + self.update_bias)
        reset_gate = torch.sigmoid(self.W_r(combined))
        
        candidate_state = torch.tanh(self.W_h(torch.cat([candidate, reset_gate * old_state], dim=-1)))
        new_state = update_gate * candidate_state + (1 - update_gate) * old_state
        
        output = self.out_proj(new_state)
        return output