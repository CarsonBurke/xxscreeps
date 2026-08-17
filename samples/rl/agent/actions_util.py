"""Shared factorized-action helpers (padding and strict BC NLL)."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .constants import INTENT_SLOTS, MAX_ACTORS, N_BODY_PART

# Masked logits use finfo.min, not -inf — still "illegal" for BC.
_ILLEGAL_LOGPROB = -1e20


def pad_actions(actions: dict) -> dict[str, Tensor]:
    """Pad nested teacher action lists to [MAX_ACTORS, INTENT_SLOTS] long tensors."""
    out: dict[str, Tensor] = {}
    for key in (
        "types", "dirs", "targets", "amounts",
        "construction_types", "construction_tiles",
    ):
        rows = actions.get(key)
        if rows is None:
            rows = []
        t = torch.zeros(MAX_ACTORS, INTENT_SLOTS, dtype=torch.long)
        if torch.is_tensor(rows) or isinstance(rows, np.ndarray):
            source = torch.as_tensor(rows, dtype=torch.long)
            if source.dim() == 3 and source.shape[0] == 1:
                source = source[0]
            if source.dim() != 2:
                raise ValueError(f"teacher actions.{key} must be rank 2, got {source.shape}")
            if source.shape == t.shape:
                out[key] = source.contiguous()
                continue
            actor_count = min(MAX_ACTORS, source.shape[0])
            slot_count = min(INTENT_SLOTS, source.shape[1])
            t[:actor_count, :slot_count].copy_(source[:actor_count, :slot_count])
        else:
            for ai, row in enumerate(rows[:MAX_ACTORS]):
                for si, v in enumerate((row or [])[:INTENT_SLOTS]):
                    t[ai, si] = int(v)
        out[key] = t
    for key in ("body_counts", "body_order"):
        rows = actions.get(key)
        if rows is None:
            rows = []
        default = torch.arange(N_BODY_PART) if key == "body_order" else torch.zeros(N_BODY_PART)
        body = default.long().view(1, 1, N_BODY_PART).expand(
            MAX_ACTORS, INTENT_SLOTS, -1,
        ).clone()
        if torch.is_tensor(rows) or isinstance(rows, np.ndarray):
            source = torch.as_tensor(rows, dtype=torch.long)
            if source.dim() == 4 and source.shape[0] == 1:
                source = source[0]
            if source.dim() != 3:
                raise ValueError(f"teacher actions.{key} must be rank 3, got {source.shape}")
            if source.shape == body.shape:
                out[key] = source.contiguous()
                continue
            actor_count = min(MAX_ACTORS, source.shape[0])
            slot_count = min(INTENT_SLOTS, source.shape[1])
            part_count = min(N_BODY_PART, source.shape[2])
            body[:actor_count, :slot_count, :part_count].copy_(
                source[:actor_count, :slot_count, :part_count]
            )
        else:
            for ai, actor_row in enumerate(rows[:MAX_ACTORS]):
                for si, slot_row in enumerate((actor_row or [])[:INTENT_SLOTS]):
                    values = list(slot_row or [])[:N_BODY_PART]
                    if values:
                        body[ai, si, : len(values)] = torch.as_tensor(values, dtype=torch.long)
        out[key] = body
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
