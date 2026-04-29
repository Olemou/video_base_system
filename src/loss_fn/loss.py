import torch
import torch.nn.functional as F
from src.loss_fn.koopman import koopman 

def contrastive_with_masks(
    x, labels, g_sem, W_inner,
    temperature=0.1, eps=1e-8
):
    B, T, N, D = x.shape
    device = x.device

    # --------------------------------------------------
    # 1. STRUCTURE ONLY IN REPRESENTATION
    # --------------------------------------------------
    K_temp = koopman(x)

    x_temp = torch.einsum("btnd,bdk->btnk", x, K_temp)

    g_sem = torch.tanh(g_sem)
    x_sem = x * g_sem[:, None, None, :]

    W_inner = W_inner / (W_inner.sum(dim=-1, keepdim=True) + eps)
    x_inner = torch.einsum("btnm,btmd->btnd", W_inner, x)

    z = x + x_temp + x_inner + x_sem
    z = F.normalize(z, dim=-1)

    z = z.reshape(B * T * N, D)
    M = z.shape[0]

    # --------------------------------------------------
    # 2. SIMILARITY
    # --------------------------------------------------

    sim = torch.matmul(z, z.T) / temperature

    eye = torch.eye(M, device=device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    sim = sim.clamp(-20, 20)

    exp_sim = torch.exp(sim)

    # --------------------------------------------------
    # 3. POSITIVE MASKS
    # --------------------------------------------------

    base = torch.arange(M, device=device).view(B, T, N)

    # temporal positives
    anchor_t = base[:, :-1, :].reshape(-1)
    pos_t = base[:, 1:, :].reshape(-1)

    temporal_pos = torch.zeros((M, M), device=device)
    temporal_pos[anchor_t, pos_t] = 1.0

    # semantic positives
    flat_labels = labels.repeat_interleave(T * N)
    semantic_pos = (flat_labels[:, None] == flat_labels[None, :]).float()
    semantic_pos.fill_diagonal_(0.0)

    # --------------------------------------------------
    # 4. INNER MASK (THIS IS WHAT YOU WERE MISSING)
    # --------------------------------------------------

    idx = base.reshape(B * T, N)

    row = idx.unsqueeze(2)
    col = idx.unsqueeze(1)

    inner_mask = torch.zeros((M, M), device=device)
    inner_mask[row, col] = 1.0
    inner_mask.fill_diagonal_(0.0)

    # --------------------------------------------------
    # 5. COMBINE POSITIVES
    # --------------------------------------------------

    pos_mask = temporal_pos + semantic_pos + inner_mask
    pos_mask = pos_mask > 0
    neg_mask = ~pos_mask

    # --------------------------------------------------
    # 6. INFO NCE LOSS
    # --------------------------------------------------

    numerator = (exp_sim * pos_mask.float()).sum(dim=1, keepdim=True)

    exp_neg = exp_sim * neg_mask.float()
    denominator = exp_neg.sum(dim=1, keepdim=True) + numerator + eps

    loss = -torch.log((numerator + eps) / denominator)

    return loss.squeeze(1).mean()

