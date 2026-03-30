
import torch
import torch.nn.functional as F
import torch.nn as nn

class GreedyTokenSelector(nn.Module):
    def __init__(self, threshold: float=0.01):
        super().__init__()
        self.threshold = threshold

    def forward(self, x, attn):
        """
        x: [seq_len, dim] - input tokens
        attn: [num_heads, seq_len, seq_len] - attention weights

        Returns:
            x_zeroed: [seq_len, dim] - matched tokens kept, unmatched = 0
            selected_indices: [K] - indices of matched tokens
        """
        attn_mean = attn.mean(dim=0)  # [N, N]
        top1_idx = vectorized_one_to_one(attn_mean.unsqueeze(0), self.threshold).squeeze(0)

        # Which KEYS were selected?
        valid_queries = top1_idx >= 0
        selected_keys = top1_idx[valid_queries]  # Unique key indices that were selected

        # Create mask for selected keys
        key_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        key_mask[selected_keys] = True

        # Zero-out unselected keys
        x_zeroed = x.clone()
        x_zeroed[~key_mask] = 0.0  # ← Unselected become ZERO

        return x_zeroed
    
def vectorized_one_to_one(attn_avg: torch.Tensor, threshold: float=0.01):
    head, N, _ = attn_avg.shape
    device = attn_avg.device

    scores = torch.where(
        attn_avg < threshold,
        torch.tensor(-1e9, device=device),
        attn_avg
    )
    top1_vals, top1_idx = scores.max(dim=-1)

    # Build one-hot only for valid indices (>=0)
    valid = (top1_idx >= 0)                       # [head, N]
    one_hot = torch.zeros(head, N, N, dtype=torch.long, device=device)
    head_idx = torch.arange(head, device=device).view(head, 1).expand(-1, N)
    query_idx = torch.arange(N, device=device).view(1, N).expand(head, -1)
    one_hot[head_idx[valid], query_idx[valid], top1_idx[valid]] = 1

    cumsum = one_hot.cumsum(dim=2)
    duplicate_mask = (cumsum > 1).any(dim=-1)

    top1_idx = top1_idx.masked_fill(duplicate_mask, -1)
    top1_idx = top1_idx.masked_fill(top1_vals < threshold, -1)

    return top1_idx

""" def select_tokens_from_attn(x, attn, threshold=0.01):
    
    x: [seq_len, dim] - input tokens
    attn: [num_heads, seq_len, seq_len] - attention weights

    Returns:
        x_zeroed: [seq_len, dim] - matched tokens kept, unmatched = 0
        selected_indices: [K] - indices of matched tokens
    
    attn_mean = attn.mean(dim=0)  # [N, N]
    top1_idx = vectorized_one_to_one(attn_mean.unsqueeze(0), threshold).squeeze(0)

    # Which KEYS were selected?
    valid_queries = top1_idx >= 0
    selected_keys = top1_idx[valid_queries]  # Unique key indices that were selected

    # Create mask for selected keys
    key_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    key_mask[selected_keys] = True

    # Zero-out unselected keys
    x_zeroed = x.clone()
    x_zeroed[~key_mask] = 0.0  # ← Unselected become ZERO

    return x_zeroed """