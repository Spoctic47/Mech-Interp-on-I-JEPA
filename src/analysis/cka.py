"""
src/analysis/cka.py
--------------------
Centered Kernel Alignment (CKA) for comparing layer representations.

CKA measures the similarity between two sets of representations,
invariant to orthogonal transformations and isotropic scaling.

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited", ICML 2019.  https://arxiv.org/abs/1905.00414
"""

import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def center_gram(K: np.ndarray) -> np.ndarray:
    """Double-center a Gram matrix (in-place efficient)."""
    n = K.shape[0]
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    total_mean = K.mean()
    return K - row_mean - col_mean + total_mean


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two feature matrices.

    Args:
        X: (N, D1) feature matrix from layer A.
        Y: (N, D2) feature matrix from layer B.

    Returns:
        CKA similarity in [0, 1].
    """
    # Gram matrices
    K = X @ X.T   # (N, N)
    L = Y @ Y.T   # (N, N)

    # Center
    K_c = center_gram(K.copy())
    L_c = center_gram(L.copy())

    # HSIC estimators
    hsic_xy = np.sum(K_c * L_c)
    hsic_xx = np.sum(K_c * K_c)
    hsic_yy = np.sum(L_c * L_c)

    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0

    return float(hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy)))


def compute_cka_matrix(
    features: Dict[int, np.ndarray],
    layer_indices: List[int],
    max_samples: int = 500,
) -> np.ndarray:
    """
    Compute an L x L CKA similarity matrix over all pairs of layers.

    Args:
        features:     Dict mapping layer_idx -> (N, D) feature array.
        layer_indices: Ordered list of layer indices to compare.
        max_samples:  Subsample for speed (CKA scales as O(N^2)).

    Returns:
        cka_matrix: np.ndarray of shape (L, L) with values in [0, 1].
    """
    L = len(layer_indices)
    cka_matrix = np.zeros((L, L), dtype=np.float32)

    # Optionally subsample
    n = features[layer_indices[0]].shape[0]
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False)
        sub_features = {li: features[li][idx] for li in layer_indices}
    else:
        sub_features = features

    logger.info(f"Computing {L}x{L} CKA matrix ({min(n, max_samples)} samples) ...")

    for i, li in enumerate(layer_indices):
        for j, lj in enumerate(layer_indices):
            if i > j:
                cka_matrix[i, j] = cka_matrix[j, i]   # symmetric
            else:
                cka_matrix[i, j] = linear_cka(sub_features[li], sub_features[lj])

    return cka_matrix
