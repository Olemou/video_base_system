
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.src_utils.vision_config import VisionConfig
from app.kalman_gain_net import kalmanGainNet
from app.greedy import GreedyTokenSelector
from app.temporal_shift_attn_signal import temporalShiftedAttentionSignal

def kalman_step(prev_tokens: torch.Tensor, kalmanformer_net: nn.Module):
    """
    prev_tokens: [B, T, N, D]
    kalmanformer_net: KalmanFormerNet instance
    """
    T = prev_tokens.shape[1]
    device, dtype = prev_tokens.device, prev_tokens.dtype

    # Output buffers
    x_post_all = torch.zeros_like(prev_tokens)
    # First frame
    x_post_all[:, 0] = prev_tokens[:, 0]

    # Initialize
    prev_innovation = torch.zeros_like(prev_tokens[:, 0], device=device, dtype=dtype)
    prev_state_update_diff = torch.zeros_like(prev_tokens[:, 0], device=device, dtype=dtype)  # Add this
    curr_vel = torch.zeros_like(prev_tokens, device=device, dtype=dtype)

    for t in range(1, T):
        # Prediction
        prev_assoc = prev_tokens[:, t-1]
        vel_assoc = curr_vel[:, t-1]
        pred = prev_assoc + vel_assoc

        observation = prev_tokens[:, t]

        # Innovations
        innovation = observation - pred
        state_evol_diff = pred - prev_assoc
        evolution_diff = innovation - prev_innovation

        # Prepare for KalmanFormer
        encoder_input = torch.cat([innovation, state_evol_diff], dim=-1).unsqueeze(1)
        decoder_kv = torch.cat([evolution_diff, prev_state_update_diff], dim=-1).unsqueeze(1)

        # Compute gain
        K_gain = kalmanformer_net(encoder_input, decoder_kv).squeeze(1)

        # Update
        x_post = pred + K_gain * innovation
        state_update_diff = x_post - pred
        new_vel = x_post - prev_assoc

        # Store
        x_post_all[:, t] = x_post

        # Update for next step
        curr_vel[:, t] = new_vel
        prev_innovation = innovation
        prev_state_update_diff = state_update_diff  # Store for next iteration
    cu_seqlen = torch.arange(0, (x_post_all.shape[0] + 1) * x_post_all.shape[1] * x_post_all.shape[2], x_post_all.shape[1] * x_post_all.shape[2],
                          device=device, dtype=torch.int32)
    return x_post_all, cu_seqlen

class KalmanFormerNet(nn.Module):
    def __init__(self,config: VisionConfig):
        super().__init__()
    
        self. kalman_gain_net = kalmanGainNet(config)
        self.temporalShiftedAttentionSignal = temporalShiftedAttentionSignal(config)
        self.greedy_selector = GreedyTokenSelector(config.gready_token_threshold)

        
    def forward(self, x):
        batch_size = x.shape[0]
        number_of_tokens = x.shape[2]
        patch_len = x.shape[1]
        x = x.view(-1, x.shape[-1])  # [B * T_patch * K, C]

        cu_seqlens = torch.arange(0, (batch_size + 1) * patch_len * number_of_tokens, patch_len * number_of_tokens,
                          device=x.device, dtype=torch.int32)
        attn = self.temporalShiftedAttentionSignal(
            x=x,
            cu_seqlens=cu_seqlens,
            patch_len=patch_len,
            number_of_tokens=number_of_tokens,
            device = x.device
        )
        selected_tokens = self.greedy_selector(x, attn)
        selected_tokens = selected_tokens.reshape(batch_size, patch_len, number_of_tokens, x.shape[-1])  # [B, T_patch, K, C]
        
        x_kalman, cu_seqlens = kalman_step(selected_tokens, self.kalman_gain_net)
        return x_kalman, cu_seqlens
    