"""Shared factorized-action helpers (padding and strict BC NLL)."""
from __future__ import annotations

import torch
from torch import Tensor

from .constants import INTENT_SLOTS, MAX_ACTORS

# Masked logits use finfo.min, not -inf — still "illegal" for BC.
_ILLEGAL_LOGPROB = -1e20


def pad_actions(actions: dict) -> dict[str, Tensor]:
    """Pad nested teacher action lists to [MAX_ACTORS, INTENT_SLOTS] long tensors."""
    out: dict[str, Tensor] = {}
    for key in ("types", "dirs", "targets", "amounts"):
        rows = actions.get(key) or []
        t = torch.zeros(MAX_ACTORS, INTENT_SLOTS, dtype=torch.long)
        for ai, row in enumerate(rows[:MAX_ACTORS]):
            for si, v in enumerate((row or [])[:INTENT_SLOTS]):
                t[ai, si] = int(v)
        out[key] = t
    return out


def safe_bc_nll(
    logprob: Tensor,
    eligible: Tensor | None = None,
    *,
    strict: bool = False,
) -> tuple[Tensor, float]:
    """Mean −logπ over eligible legal demonstration factors.

    Masked illegal actions produce ~finfo.min logπ (finite but huge magnitude),
    not −∞. Drop non-finite and extremely negative logprobs so they cannot
    dominate the joint loss. Returns (nll, fraction_legal).
    """
    if eligible is None:
        eligible = torch.ones_like(logprob, dtype=torch.bool)
    else:
        eligible = eligible.to(device=logprob.device, dtype=torch.bool)
    valid = eligible & torch.isfinite(logprob) & (logprob > _ILLEGAL_LOGPROB)
    denominator = int(eligible.sum().item())
    frac = float(valid.sum().item() / denominator) if denominator else 1.0
    invalid = eligible & ~valid
    if strict and bool(invalid.any()):
        raise ValueError(
            f"teacher contract violation: {int(invalid.sum().item())}/"
            f"{denominator} eligible factors are masked or non-finite"
        )
    if denominator == 0:
        return logprob.sum() * 0.0, frac
    if not bool(valid.any()):
        return logprob.sum() * 0.0, frac
    safe = torch.where(valid, logprob, torch.zeros_like(logprob))
    nll = -safe.sum() / valid.sum().clamp_min(1)
    return nll, frac
