import torch


def apply_masks(x, masks, concat=True):
    """
    Apply masking on token dimension for video tensors.

    Args:
        x: Tensor of shape (B, T, N, C)
        masks: list of tensors, each of shape (B, K)
               containing indices of tokens to keep
        concat: if True, concatenate outputs along batch dim

    Returns:
        If concat=True:
            Tensor of shape (len(masks)*B, T, K, C)
        else:
            List of tensors, each (B, T, K, C)
    """
    B, T, N, C = x.shape
    all_x = []

    for m in masks:
        # m: (B, K)

        # --- Step 1: expand mask to time dimension ---
        # (B, 1, K) -> (B, T, K)
        m_exp = m.unsqueeze(1).expand(-1, T, -1)

        # --- Step 2: expand for channel dimension ---
        # (B, T, K) -> (B, T, K, C)
        m_exp = m_exp.unsqueeze(-1).expand(-1, -1, -1, C)

        # --- Step 3: gather along token dimension (N) ---
        # dim=2 corresponds to N
        x_masked = torch.gather(x, dim=2, index=m_exp)

        all_x.append(x_masked)

    if not concat:
        return all_x

    # --- Step 4: concatenate along batch dimension ---
    return torch.cat(all_x, dim=0)