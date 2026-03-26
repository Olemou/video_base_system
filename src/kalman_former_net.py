

import torch
import torch.nn.functional as F
import torch.nn as nn
from kalman_gain_net import kalmanGainNet
from vision_config import VisionConfig
from greedy import select_tokens_from_attn
from temporal_shift_attn_signal import temporalShiftedAttentionSignal

def kalman_step(prev_tokens: torch.Tensor, kalmanformer_net: nn.Module):
    """
    prev_tokens: [B, T, N, D]
    kalmanformer_net: KalmanFormerNet instance
    """
    B, T, N, D = prev_tokens.shape
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

    return x_post_all

class KalmanFormerNet(nn.Module):
    def __init__(self,cu_seqlens:torch.Tensor, patch_len: int, k: int, batch_size: int, temporal_shift: int, device: torch.device, config: VisionConfig):
        super().__init__()
        self. kalman_gain_net = kalmanGainNet(config)
        self.temporalShiftedAttentionSignal = temporalShiftedAttentionSignal(config)
        self.cu_seqlens = cu_seqlens
        self.patch_len = patch_len
        self.batch_size = batch_size
        self.k = k
        self.temporal_shift = temporal_shift
        self.device = device
    def forward(self, x):
        attn = self.temporalShiftedAttentionSignal(
            x=x,
            cu_seqlens=self.cu_seqlens,
            patch_len=self.patch_len,
            k=self.k,
            device = self.device
        )
        selected_tokens = select_tokens_from_attn(x, attn)
        select_tokens_from_attn = selected_tokens.reshape(self.batch_size, self.temporal_shift, self.k, -1)
        return kalman_step(selected_tokens, self.kalman_gain_net)
         
        
    