import torch
import torch.nn as nn
import torch.nn.functional as F


class CoCluster(nn.Module):
    def __init__(self, embed_dim: int = 256, prior_weight: float = 0.5):
        super().__init__()

        self.embed_dim = embed_dim
        self.prior_weight = prior_weight

    @staticmethod
    def similarity_to_evidence(sim: torch.Tensor, prior_weight: float):
        e_pos = torch.exp(F.softmax(sim, dim=-1))
        e_neg = torch.exp(F.softmax(1.0 - sim, dim=-1))
        total = e_pos + e_neg + prior_weight
        return prior_weight / total

    def forward(self, z: torch.Tensor):

        # =====================================================
        # CASE 1: (B, D)
        # =====================================================
        if z.dim() == 2:
            B, D = z.shape

            if D != self.embed_dim:
                raise ValueError(f"Expected {self.embed_dim}, got {D}")
            sim = torch.einsum('bd,be->bde', z, z)

            # remove diagonal per sample
            eye = torch.eye(D, device=z.device).unsqueeze(0)
            sim = sim.masked_fill(eye.bool(), 0.0)

            uncertainty = self.similarity_to_evidence(sim, self.prior_weight)

            # keep per-feature uncertainty
            uncertainty = uncertainty.mean(dim=-1)  # (B, D)

            return uncertainty

        # =====================================================
        # CASE 2: (B, T, N, D)
        # =====================================================
        elif z.dim() == 4:
            B, T, N, D = z.shape

            if D != self.embed_dim:
                raise ValueError(f"Expected {self.embed_dim}, got {D}")

            sim = z @ z.transpose(-1, -2)  # (B,T,N,N)

            eye = torch.eye(N, device=z.device).view(1, 1, N, N)
            sim = sim.masked_fill(eye.bool(), 0.0)

            uncertainty = self.similarity_to_evidence(sim, self.prior_weight)

            return uncertainty

        else:
            raise ValueError(f"Expected (B,D) or (B,T,N,D), got {tuple(z.shape)}")

def compute_lambda(uncertainty: torch.Tensor, epoch: int, TotalEpochs: int):
    """
    uncertainty: (..., M)
    Retourne Lambda de même shape que uncertainty.
    """

    *prefix, M = uncertainty.shape
    delta_t = epoch / TotalEpochs

    # argsort descending sur la dernière dimension
    u_desc_indices = torch.argsort(uncertainty, dim=-1, descending=True)

    # ranks a la même shape que uncertainty
    ranks = torch.zeros_like(uncertainty, dtype=torch.float32)

    # scatter les rangs
    arange = torch.arange(M, dtype=torch.float32, device=uncertainty.device)
    ranks.scatter_(-1, u_desc_indices, arange)

    # normalisation
    phi_rho = ranks / M

    # Lambda
    Lambda = 1 + torch.exp(-delta_t * phi_rho)

    return Lambda