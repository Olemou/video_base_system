from cProfile import label

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoClusterOpinionLoss(nn.Module):
    def __init__(self, prior_weight=0.5):
        super().__init__()
        self.prior_weight = prior_weight
        
    @staticmethod
    def similarity_to_evidence(sim: torch.Tensor, prior_weight=0.5) -> torch.Tensor:
        g_sim = sim
        g_dsim = 1 - sim
        e_pos = torch.exp(F.softmax(g_sim, dim=1))
        e_neg = torch.exp(F.softmax(g_dsim, dim=1))
        total_mass = e_pos + e_neg + prior_weight
        return prior_weight / total_mass

    def forward(self, Z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        assert labels.shape[0] == Z.shape[0], f"{labels.shape[0]} != {Z.shape[0]}"

        # Similarity
        sim = torch.matmul(Z, Z.T)
        sim.fill_diagonal_(0.0)

        uncertainty = self.similarity_to_evidence(sim)

        # Same-class mask
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        uncertainty[~mask] = 0.0

        return uncertainty


def compute_lambda(
    uncertainty: torch.Tensor, epoch: int, TotalEpochs: int
) -> torch.Tensor:
    """
    Compute Lambda(t, u_ij) = tanh( (t/T) * (rank(u_ij)/n) ) + 1

    Args:
        u: Tensor of shape [N, M] — uncertainty matrix
        t: int — current epoch
        T: int — total number of epochs

    Returns:
        Lambda: Tensor of shape [N, M]
    """

    N, M = uncertainty.shape
    delta_t = epoch / TotalEpochs  # scalar in [0, 1]
    u_desc_indices = torch.argsort(uncertainty, dim=1, descending=True)
    ranks = torch.zeros_like(uncertainty, dtype=torch.float32)
    for i in range(N):
        ranks[i, u_desc_indices[i]] = torch.arange(
            M, dtype=torch.float32, device=uncertainty.device
        )
    phi_rho = ranks / M
    Lambda = 1 + torch.exp(-delta_t * phi_rho)

    return Lambda
