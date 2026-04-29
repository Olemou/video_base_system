import torch
import torch.nn.functional as F


# -----------------------------
# 1. Spatial pooling ()
# -----------------------------
def spatial_pool(X):
    # X: (B, T, N, D)
    return X.mean(dim=2)  # (B, T, D)


# -----------------------------
# 2. Temporal weighting
# -----------------------------
def compute_weights(X):
    # X: (B, T, D)
    x1, x2 = X[:, :-1], X[:, 1:]

    sim = F.cosine_similarity(x1, x2, dim=-1)  # (B, T-1)
    w = torch.exp(-sim)

    return w


# -----------------------------
# 3. Apply weights
# -----------------------------
def apply_weights(X1, w):
    # X1: (B, T-1, D)
    return X1 * w.unsqueeze(-1)


# -----------------------------
# 4. FAST batched Koopman (NO LOOP)
# -----------------------------
def koopman_batched(X1, X2, eps=1e-4):
    """
    X1, X2: (B, T-1, D)
    returns: (B, D, D)
    """

    B, Tm1, D = X1.shape

    # reshape to (B, D, T)
    X1t = X1.transpose(1, 2)  # (B, D, T-1)
    X2t = X2.transpose(1, 2)  # (B, D, T-1)

    # Compute covariance matrices
    # A = X2 X1^T
    A = torch.bmm(X2t, X1t.transpose(1, 2))  # (B, D, D)

    # B = X1 X1^T
    Bmat = torch.bmm(X1t, X1t.transpose(1, 2))  # (B, D, D)

    # Regularization for stability
    I = torch.eye(D, device=X1.device).unsqueeze(0)
    Bmat = Bmat + eps * I

    # Inverse
    B_inv = torch.linalg.inv(Bmat)  # (B, D, D)

    # Koopman operator
    K = torch.bmm(A, B_inv)  # (B, D, D)

    return K

# -----------------------------
# 5. Full pipeline
# -----------------------------
def koopman(X):
    """
    X: (B, T, N, D)
    """
    # 1. spatial pooling
    X = spatial_pool(X)  # (B, T, D)
    # 2. DMD split
    X1 = X[:, :-1, :]
    X2 = X[:, 1:, :]
    # 3. weights
    w = compute_weights(X)

    # 4. weighted X1
    X1 = apply_weights(X1, w)

    # 5. Koopman (FAST)
    K = koopman_batched(X1, X2)

    return K