"""Discounted-return and CleanRL-style GAE helpers.

PPO uses one GAE recurrence for both outputs:
  · actor advantage = GAE(gamma, lambda)
  · critic target = actor advantage + behavior-policy value

Supervised critic pretraining instead uses finite-horizon discounted
reward-to-go at the same gamma.  It has no reliable behavior-value baseline
from which to construct GAE before the critic is trained.

Time-limit vs terminal (Gym TimeLimit):
  · true terminal → bootstrap 0
  · truncation (max episode) → bootstrap V(s′)
"""
from __future__ import annotations

import torch
from torch import Tensor


def compute_gae_tn(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float,
    lam: float | Tensor,
    next_value: Tensor | None = None,
    truncations: Tensor | None = None,
    next_values_tn: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    rewards/values/dones: [T, N]
    lam: scalar or [N] (per-env) or [T, N]
    next_value: [N] bootstrap after last step (used if next_values_tn is None)
    truncations: optional [T, N] — 1 = time-limit (bootstrap), 0 = true terminal if done
    next_values_tn: optional [T, N] explicit V(s_{t+1}) per step (for terminal-obs bootstrap
                    without overwriting values[t+1] used as V(s) for the next transition)
    returns advantages, returns  both [T, N]
    """
    T, N = rewards.shape
    device = rewards.device
    adv = torch.zeros_like(rewards)
    if next_value is None:
        next_value = torch.zeros(N, device=device, dtype=rewards.dtype)
    if truncations is None:
        trunc = torch.zeros_like(dones)
    else:
        trunc = truncations.to(device=device, dtype=rewards.dtype)

    if not torch.is_tensor(lam):
        lam_t = torch.full((N,), float(lam), device=device, dtype=rewards.dtype)
        per_step = False
    elif lam.dim() == 0:
        lam_t = lam.expand(N).to(device=device, dtype=rewards.dtype)
        per_step = False
    elif lam.shape == (N,):
        lam_t = lam.to(device=device, dtype=rewards.dtype)
        per_step = False
    else:
        lam_tn = lam.to(device=device, dtype=rewards.dtype)
        per_step = True

    if next_values_tn is not None:
        nv = next_values_tn.to(device=device, dtype=rewards.dtype)
    else:
        nv = None

    lastgaelam = torch.zeros(N, device=device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        # Bootstrap V(s') on both true terminals (V≈0 via next_values) and time-limits
        # (V(terminal_obs) spliced into next_values_tn / next_value).
        # Always CUT the λ-chain on any done — otherwise advantages leak into the
        # post-reset episode (Gym/SB3-style time-limit bootstrapping).
        done_t = dones[t]
        if nv is not None:
            nextvalues = nv[t]
        elif t == T - 1:
            nextvalues = next_value
        else:
            nextvalues = values[t + 1]
        # Truncation: bootstrap V; true terminal: nextvalues should already be 0
        # (or we zero via (1-done)+done*trunc path). Use V whenever not a pure terminal.
        trunc_t = trunc[t]
        terminated = done_t * (1.0 - trunc_t)
        # δ uses V on non-terminated steps; on pure terminal next_nonterminal=0 → no V
        bootstrap_ok = 1.0 - terminated
        delta = rewards[t] + gamma * nextvalues * bootstrap_ok - values[t]
        # λ-chain cuts on ANY trajectory cut: a true terminal, a time limit, or a
        # start-state segment boundary. A segment boundary reports done=0 with
        # truncation=1 because the environment episode continues elsewhere, and
        # leaving the chain intact there would leak one world's advantages into
        # the unrelated world restored in its place.
        chain = 1.0 - torch.clamp(done_t + trunc_t, max=1.0)
        if per_step:
            lastgaelam = delta + gamma * lam_tn[t] * chain * lastgaelam
        else:
            lastgaelam = delta + gamma * lam_t * chain * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def cleanrl_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    next_value: Tensor | None = None,
    truncations: Tensor | None = None,
    next_values_tn: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """
    Returns:
      advantages — GAE(gamma, lambda), used by the policy surrogate
      returns    — advantages + behavior values, used by the value loss
      info       — estimator parameters and geometric effective horizon
    """
    advantages, returns = compute_gae_tn(
        rewards,
        values,
        dones,
        gamma=gamma,
        lam=gae_lambda,
        next_value=next_value,
        truncations=truncations,
        next_values_tn=next_values_tn,
    )
    decay = float(gamma) * float(gae_lambda)
    info = {
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "gae_decay": decay,
        "effective_horizon": 1.0 / (1.0 - decay),
    }
    return advantages, returns, info


def discounted_returns_tn(
    rewards: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    next_value: Tensor,
    truncations: Tensor | None = None,
    next_values_tn: Tensor | None = None,
) -> Tensor:
    """Finite-horizon discounted reward-to-go for critic pretraining.

    `next_value` optionally bootstraps the last step; `next_values_tn` can splice
    V(terminal_obs) on mid-chunk time limits.  With a zero endpoint at the end
    of a declared finite lifecycle this is the exact discounted return target.
    Returns shape [T, N].
    """
    T, N = rewards.shape
    values_dummy = torch.zeros(T, N, device=rewards.device, dtype=rewards.dtype)
    trunc = dones if truncations is None else truncations
    _, returns = compute_gae_tn(
        rewards,
        values_dummy,
        dones,
        gamma,
        lam=1.0,
        next_value=next_value,
        truncations=trunc,
        next_values_tn=next_values_tn,
    )
    return returns
