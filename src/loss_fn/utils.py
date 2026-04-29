

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoCluster(nn.Module):
    def __init__(self, prior_weight=0.5):
        super().__init__()
        self.prior_weight = prior_weight

    @staticmethod
    def similarity_to_evidence(sim: torch.Tensor, prior_weight=0.5) -> torch.Tensor:
        # sim: (B, T, N, N)
        g_sim = sim
        g_dsim = 1 - sim

        e_pos = torch.exp(F.softmax(g_sim, dim=-1))
        e_neg = torch.exp(F.softmax(g_dsim, dim=-1))

        total_mass = e_pos + e_neg + prior_weight
        return prior_weight / total_mass

    def forward(self, z: torch.Tensor, labels=None):
        """
        Supports:
        Case 1: (B, T, N, D)  -> video batch
        Case 2: (N, D)        -> single frame / token set
        """

        # -----------------------------
        # Case 1: Video batch
        # -----------------------------
        if z.dim() == 4:
            B, T, N, D = z.shape

            # verify transpose compatibility
            if z.shape[-1] != D:
                raise ValueError("Invalid last dimension before transpose.")

            sim = torch.matmul(z, z.transpose(-1, -2))

            # remove diagonal per frame
            eye = torch.eye(N, device=z.device).unsqueeze(0).unsqueeze(0)
            sim = sim * (1 - eye)

        # -----------------------------
        # Case 2: Single frame
        # -----------------------------
        elif z.dim() == 2:
            N, D = z.shape

            if z.shape[-1] != D:
                raise ValueError("Invalid last dimension before transpose.")

            sim = torch.matmul(z, z.t())

            # remove diagonal
            eye = torch.eye(N, device=z.device)
            sim = sim * (1 - eye)

        else:
            raise ValueError(
                f"Expected z to be 2D or 4D (N,D) or (B,T,N,D), got {tuple(z.shape)}"
            )

        # Convert similarity to uncertainty
        uncertainty = self.similarity_to_evidence(sim, self.prior_weight)
        return uncertainty
    
    
    
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


