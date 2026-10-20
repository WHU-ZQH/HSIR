"""H-GRPO equations 4, 5 and 9 (arXiv:2605.24998v1).

See README.md for explicit conventions where the paper is underspecified.
This module requires only PyTorch, independently of TRL and Open-R1.
"""

import torch


def attention_spectral_score(hidden, attention, mask, ridge=1e-3):
    """Score one solution: hidden [tokens, features], attention [heads, Q, K].

    Use all unmasked prompt and completion tokens. Max-pool over heads and
    query positions to obtain one importance weight per key token. SVD returns
    singular values in descending order, as in the retained scoring code.
    """
    if hidden.ndim != 2 or attention.ndim != 3 or mask.ndim != 1:
        raise ValueError("Expected hidden [L,D], attention [heads,L,L], mask [L]")
    length = hidden.shape[0]
    if mask.shape[0] != length or attention.shape[1:] != (length, length):
        raise ValueError("Hidden states, attention and mask must have matching lengths")
    if ridge <= 0:
        raise ValueError("ridge must be positive for rank-deficient covariance matrices")
    valid = mask.bool()
    if not valid.any():
        raise ValueError("Cannot score an empty solution")
    dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    h = hidden[valid].to(dtype=dtype)
    a = attention[:, valid][:, :, valid].to(dtype=dtype)
    if not torch.isfinite(h).all() or not torch.isfinite(a).all() or (a < 0).any():
        raise ValueError("Hidden states and nonnegative attention must be finite")
    # H in the paper is [features, tokens]. Center across features, not tokens.
    h = h - h.mean(dim=1, keepdim=True)
    covariance = h @ h.T
    covariance.diagonal().add_(ridge)
    singular_values = torch.linalg.svdvals(covariance)
    weights = a.amax(dim=(0, 1))
    if weights.sum() <= 0:
        raise ValueError("Attention has no mass on valid tokens; use eager attention")
    weights = weights / weights.sum()
    return (weights * singular_values.clamp_min(torch.finfo(h.dtype).tiny).log()).sum()


def normalize_indiv_rewards(scores, num_generations):
    """Apply Eq. 5 then Eq. 9 to contiguous, globally gathered prompt groups.

    Constant groups have zero diversity reward. Values are intentionally not
    clamped to [0,1]: below-mean scores receive negative rewards in Eq. 9.
    """
    if scores.ndim != 1 or num_generations < 2 or scores.numel() % num_generations:
        raise ValueError("Scores must contain complete groups with at least two generations")
    if not torch.isfinite(scores).all():
        raise ValueError("InDiv scores must be finite")
    grouped = scores.float().reshape(-1, num_generations)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, correction=0, keepdim=True)
    nonconstant = std > 0
    indiv = centered / torch.where(nonconstant, std, torch.ones_like(std))
    maximum = indiv.amax(dim=1, keepdim=True)
    reward = indiv / torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    return torch.where(nonconstant, reward, torch.zeros_like(reward)).flatten()
