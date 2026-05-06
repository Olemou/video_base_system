import torch
import torch.nn.functional as F
from src.loss_fn.koopman import koopman 
from src.loss_fn.utils import CoCluster

class KolmanAdptiveContrastiveLoss:
    def __init__(self, prior_weight=0.5, TotalEpochs: int = 100, temperature: float = 0.1):
        super().__init__()
        self.co_cluster_loss = CoCluster(prior_weight)
        self.TotalEpochs = TotalEpochs
        self.temperature = temperature
        
    def weighted(u: torch.Tensor):
        weight = 1 + torch.exp(-u)
        return weight

    def contrastive_masks(
        self,
        x, labels,
        temperature=0.1, eps=1e-8
    ):
        B, T, N, D = x.shape
        device = x.device
        # --------------------------------------------------
        # 1. STRUCTURE ONLY IN REPRESENTATION
        # --------------------------------------------------

        # Koopman temporal transform
        K_temp = koopman(x)
        x_temp = torch.einsum("btnd,bdk->btnk", x, K_temp)

        # semantic gating
        g_sem_uncertainity = self.co_cluster_loss(x_mean = x.mean(dim=(1, 2)))
        g_sem = self.weighted(g_sem_uncertainity)
        x_sem = x * g_sem[:, None, None, :]

        # inner graph mixing
        W_inner_uncertainity = self.co_cluster_loss(x)
        W_inner = self.weighted(W_inner_uncertainity)
        x_inner = torch.einsum("btnm,btmd->btnd", W_inner, x)

        # final representation
        z = x + x_temp + x_inner + x_sem
        z = F.normalize(z, dim=-1)

        # flatten
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
        # 4. INNER MASK
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

        pos_mask = (temporal_pos + semantic_pos + inner_mask) > 0
        neg_mask = ~pos_mask

        # --------------------------------------------------
        # 6.  LOSS compute
        # --------------------------------------------------

        numerator = exp_sim 
        denominator = numerator + (exp_sim * neg_mask.float()).sum(dim=1, keepdim=True) + eps
        log_prob = torch.log((numerator + eps) / denominator)
        loss_matrix = -log_prob * pos_mask.float()

        num_pos = pos_mask.sum(dim=1)
        valid = num_pos > 0

        loss = (loss_matrix.sum(dim=1)[valid] / (num_pos[valid] + eps)).mean()

        return loss
     
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.contrastive_masks(z, labels)

