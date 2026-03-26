
import torch
import torch.nn.functional as F 
import torch.nn as nn

def build_kalman_shifted_mask(cu_seqlens: torch.Tensor, patch_len: int, K: int, device: torch.device):
        """cu_seqlens: [B+1], patch_len: int,
        K number of tokens for all videos: int 
        Returns:
            mask: [seq_len, seq_len] with -inf for masked positions and 0 for valid positions
        """
    
        seq_len = cu_seqlens[-1].item()
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)

        frames_per_video = (cu_seqlens[1:] - cu_seqlens[:-1]) // K
        total_frames = frames_per_video.sum().item()
        if total_frames <= 1:
            return mask

        frame_cumsum = torch.cat([torch.tensor([0], device=device, dtype=torch.long),
                                  frames_per_video.cumsum(0)])

        global_frame_id = torch.arange(total_frames, device=device, dtype=torch.long)

        video_id_for_frame = torch.searchsorted(frame_cumsum[1:], global_frame_id, side='right')


        local_frame_id = global_frame_id - frame_cumsum[video_id_for_frame]

        patch_id = local_frame_id // patch_len

        same_video = video_id_for_frame[:-1] == video_id_for_frame[1:]
        same_patch = patch_id[:-1] == patch_id[1:]

        valid_pairs = same_video & same_patch

        query_global_frame = torch.nonzero(valid_pairs, as_tuple=True)[0]
        if len(query_global_frame) == 0:
            return mask

        key_global_frame = query_global_frame + 1
        q_video_id = video_id_for_frame[query_global_frame]
        q_local_frame_id = local_frame_id[query_global_frame]
        k_video_id = video_id_for_frame[key_global_frame]
        k_local_frame_id = local_frame_id[key_global_frame]

        q_start = cu_seqlens[q_video_id] + q_local_frame_id * K
        k_start = cu_seqlens[k_video_id] + k_local_frame_id * K

        offsets = torch.arange(K, device=device)
        q_grid = q_start[:, None, None] + offsets[None, :, None]
        k_grid = k_start[:, None, None] + offsets[None, None, :]
        q_all = q_grid.expand(-1, -1, K).reshape(-1)
        k_all = k_grid.expand(-1, K, -1).reshape(-1)
        mask[q_all, k_all] = 0.0

        # Add self-attention for all tokens
        idx = torch.arange(seq_len, device=device)
        mask[idx, idx] = 0.0

        return mask