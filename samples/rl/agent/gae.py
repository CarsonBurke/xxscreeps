"""GAE helpers for decoupled policy and critic return estimation.

VAPO (arXiv:2504.05118):
  · critic targets use full Monte Carlo (λ_critic = 1.0)
  · policy advantages use one explicit λ for every transition

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
        # post-reset episode (SB3/CleanRL TimeLimit handling).
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
        # λ-chain cuts on ANY episode end (done), including truncation
        chain = 1.0 - done_t
        if per_step:
            lastgaelam = delta + gamma * lam_tn[t] * chain * lastgaelam
        else:
            lastgaelam = delta + gamma * lam_t * chain * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def decoupled_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float = 0.99,
    policy_lambda: float | None = None,
    next_value: Tensor | None = None,
    truncations: Tensor | None = None,
    next_values_tn: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """
    Returns:
      advantages_policy  — explicit-λ GAE (for policy)
      returns_critic     — λ=1.0 MC-style returns (for value loss)
      info               — mean λ_policy etc.
    """
    T, N = rewards.shape
    device = rewards.device

    if policy_lambda is None:
        # Retain the helper for experiments, but use a single explicit lambda in
        # production. A rollout-end partial length is not the episode length and
        # previously assigned the wrong lambda to earlier reset segments.
        policy_lambda = 0.95
    lam_policy = torch.full(
        (N,), float(policy_lambda), device=device, dtype=rewards.dtype,
    )

    adv_policy, _ = compute_gae_tn(
        rewards, values, dones, gamma, lam_policy, next_value,
        truncations=truncations, next_values_tn=next_values_tn,
    )
    _, ret_critic = compute_gae_tn(
        rewards, values, dones, gamma, 1.0, next_value,
        truncations=truncations, next_values_tn=next_values_tn,
    )
    info = {
        "lambda_policy_mean": float(lam_policy.mean().item()),
        "lambda_policy_min": float(lam_policy.min().item()),
        "lambda_policy_max": float(lam_policy.max().item()),
        "lambda_critic": 1.0,
    }
    return adv_policy, ret_critic, info


def mc_returns_tn(
    rewards: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    next_value: Tensor,
    truncations: Tensor | None = None,
    next_values_tn: Tensor | None = None,
) -> Tensor:
    """λ=1 Monte Carlo returns with values≡0 (critic pretrain / joint pretrain).

    `next_value` bootstraps the last step; `next_values_tn` can splice V(terminal_obs)
    on mid-chunk time-limits. Returns shape [T, N].
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
