"""GAE helpers — including VAPO decoupled + length-adaptive λ.

VAPO (arXiv:2504.05118):
  · critic targets use full Monte Carlo (λ_critic = 1.0)
  · policy advantages use length-adaptive
        λ_policy = 1 - 1 / (α · L)
    with α ≈ 0.05 and L = trajectory length.
"""
from __future__ import annotations

import torch
from torch import Tensor


def length_adaptive_lambda(length: int, alpha: float = 0.05, lam_min: float = 0.5, lam_max: float = 0.99) -> float:
    """λ_policy = 1 - 1/(α L), clamped for short trajectories."""
    L = max(1, int(length))
    lam = 1.0 - 1.0 / (alpha * L)
    return float(min(lam_max, max(lam_min, lam)))


def compute_gae_tn(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float,
    lam: float | Tensor,
    next_value: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    rewards/values/dones: [T, N]
    lam: scalar or [N] (per-env) or [T, N]
    next_value: [N] bootstrap after last step (0 if terminal)
    returns advantages, returns  both [T, N]
    """
    T, N = rewards.shape
    device = rewards.device
    adv = torch.zeros_like(rewards)
    if next_value is None:
        next_value = torch.zeros(N, device=device, dtype=rewards.dtype)

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
        # [T, N]
        lam_tn = lam.to(device=device, dtype=rewards.dtype)
        per_step = True

    lastgaelam = torch.zeros(N, device=device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - dones[t]
            nextvalues = next_value
        else:
            next_nonterminal = 1.0 - dones[t]
            nextvalues = values[t + 1]
        # When done[t]=1, this step ended an episode; bootstrap 0
        delta = rewards[t] + gamma * nextvalues * next_nonterminal - values[t]
        if per_step:
            lastgaelam = delta + gamma * lam_tn[t] * next_nonterminal * lastgaelam
        else:
            lastgaelam = delta + gamma * lam_t * next_nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def decoupled_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float = 0.99,
    alpha: float = 0.05,
    next_value: Tensor | None = None,
    episode_lengths: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    """
    Returns:
      advantages_policy  — length-adaptive λ GAE (for policy)
      returns_critic     — λ=1.0 MC-style returns (for value loss)
      advantages_mc      — returns_critic - values (optional diagnostics)
      info               — mean λ_policy etc.
    """
    T, N = rewards.shape
    device = rewards.device

    # Per-env trajectory length for adaptive λ. Prefer provided episode lengths;
    # else use T for every env (fixed-horizon rollout chunk).
    if episode_lengths is None:
        lengths = torch.full((N,), T, device=device, dtype=torch.float32)
    else:
        lengths = episode_lengths.to(device=device, dtype=torch.float32).clamp_min(1)

    lam_policy = torch.tensor(
        [length_adaptive_lambda(int(L.item()), alpha=alpha) for L in lengths],
        device=device,
        dtype=rewards.dtype,
    )

    adv_policy, _ = compute_gae_tn(rewards, values, dones, gamma, lam_policy, next_value)
    # Critic: full MC via λ = 1.0
    _, ret_critic = compute_gae_tn(rewards, values, dones, gamma, 1.0, next_value)
    adv_mc = ret_critic - values

    info = {
        "lambda_policy_mean": float(lam_policy.mean().item()),
        "lambda_policy_min": float(lam_policy.min().item()),
        "lambda_policy_max": float(lam_policy.max().item()),
        "lambda_critic": 1.0,
    }
    return adv_policy, ret_critic, adv_mc, info
