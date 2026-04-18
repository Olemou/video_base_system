
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_fn.utils import compute_lambda, CoClusterOpinionLoss

class UncertaintyAwareLoss(nn.Module):
    def __init__(self, prior_weight=0.5, TotalEpochs: int = 100, temperature: float = 0.1):
        super().__init__()
        self.co_cluster_loss = CoClusterOpinionLoss(prior_weight)
        self.TotalEpochs = TotalEpochs
        self.temperature = temperature

    def uncertainty_aware_loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
        eps: float = 1e-8,
    ):
        B,N, D = z.shape
        z = z.reshape(B*N, D)
        labels = labels.repeat_interleave(N)

        uncertainty = self.co_cluster_loss(z = z, labels=labels)  # [B, N, N]

        # Normalize embeddings
        sim = torch.matmul(z, z.T) / self.temperature  # [N, N]
        exp_sim = torch.exp(sim)

        eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        # Positive and negative masks
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        same_class[eye] = False

        pos_mask = same_class 
        neg_mask = ~same_class
        # Dynamic diff-image weight
        diff_img_weight = compute_lambda(uncertainty, epoch, self.TotalEpochs) 
        # Assign weights
        pos_weights = diff_img_weight * pos_mask.float()
        pos_weights[~pos_mask] = 0.0

        # Numerator: weighted positives
        numerator = exp_sim * pos_weights  # [N, N]
        exp_neg = exp_sim * neg_mask.float()
        neg_weights = exp_neg / (exp_neg.sum(dim=1, keepdim=True) + eps)
        neg_term = neg_weights * exp_sim * neg_mask.float()

        denominator = numerator + neg_term.sum(dim=1, keepdim=True) + eps
        log_prob = torch.log(numerator / denominator + eps)
        loss_matrix = -log_prob * pos_mask.float()
        num_positives = pos_mask.sum(dim=1)
        valid = num_positives > 0
        loss = loss_matrix.sum(dim=1)[valid] / (num_positives[valid] + eps)

        return loss.mean()

    def forward(self, z: torch.Tensor, labels: torch.Tensor, epoch: int) -> torch.Tensor:
        return self.uncertainty_aware_loss(z, labels, epoch)

