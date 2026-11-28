"""PatchCore scoring (task A2.4, PatchCore paper Eq. 6-7).

Patch score: distance to the nearest memory-bank patch (k = 1). Image score: the largest patch
score. This matches the official patchcore-inspection default and Deng's ISP-AD setup.
Eq. 7's reweighting by the b nearest neighbours of the matched bank patch is available as
`reweight_b`; it is off by default because the official code doesn't use it.
"""

from __future__ import annotations

import torch

from argus.memory.search import knn


@torch.inference_mode()
def score_patches(
    features: torch.Tensor, bank: torch.Tensor, reweight_b: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """features (B, P, D), bank (N, D) on the same device. Returns image scores (B,) and patch
    scores (B, P), both float32."""
    b, p, d = features.shape
    dist, idx = knn(features.reshape(b * p, d), bank, k=1)
    patch = dist[:, 0].reshape(b, p)
    image, where = patch.max(dim=1)
    if reweight_b:
        image = image * _eq7_weights(features, bank, where, idx[:, 0].reshape(b, p), reweight_b)
    return image.float(), patch.float()


def _eq7_weights(features, bank, where, nn_idx, b_nn: int) -> torch.Tensor:
    """w = 1 - exp(|m_test - m*|) / sum over N_b(m*) of exp(|m_test - m|), per image."""
    rows = torch.arange(len(features), device=features.device)
    m_test = features[rows, where]  # (B, D) the most anomalous test patch
    m_star = bank[nn_idx[rows, where]]  # (B, D) its nearest bank patch
    _, nb = knn(m_star, bank, k=min(b_nn, len(bank)))  # N_b(m*), includes m* itself
    d_all = torch.linalg.vector_norm(m_test[:, None, :] - bank[nb], dim=-1)  # (B, b)
    d_star = torch.linalg.vector_norm(m_test - m_star, dim=-1)
    # Softmax form, numerically safe: exp(d*) / sum exp(d) = exp(d* - logsumexp(d)).
    return 1 - torch.exp(d_star - torch.logsumexp(d_all, dim=1))
