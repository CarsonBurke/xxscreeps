"""Discounted-return and GAE helpers for the VAPO-style PPO objective.

PPO runs the same GAE recurrence twice per rollout with two different λ
(VAPO §4.1, Decoupled-GAE):
  · actor advantage = GAE(gamma, λ_policy), λ_policy length-adaptive per env
  · critic target   = GAE(gamma, λ_critic=1) + behavior value, i.e. the
    discounted Monte-Carlo return over the segment

A single λ has to serve both, and 0.95 served neither: the critic target then
decays the observed reward by 0.95^k and learns almost entirely from its own
bootstrap, while the policy's credit window collapses to a couple of dozen
ticks. λ_critic=1 removes the bias from the critic target; λ_policy is chosen
per environment from the segment length it actually collected (VAPO eq. 5) so
the TD-error weights stay spread over the whole segment.

Supervised critic pretraining instead uses finite-horizon discounted
reward-to-go at the same gamma.  It has no reliable behavior-value baseline
from which to construct GAE before the critic is trained.

Time-limit vs terminal (Gym TimeLimit):
  · true terminal → bootstrap 0
  · truncation (max episode) → bootstrap V(s′)
"""
from __future__ import annotations

import math

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


def _cut_tn(dones: Tensor, truncations: Tensor | None) -> Tensor:
    """The λ-chain cut indicator `compute_gae_tn` uses, as floats. [T, N]"""
    if dones.dim() != 2:
        raise ValueError(f"dones must be [T, N], got {tuple(dones.shape)}")
    cut = dones.float()
    if truncations is not None:
        if truncations.shape != dones.shape:
            raise ValueError("truncations must match dones")
        cut = torch.clamp(cut + truncations.to(device=cut.device).float(), max=1.0)
    return cut


def _segment_pieces(dones: Tensor, truncations: Tensor | None) -> tuple[int, Tensor]:
    """Uncut trajectory pieces each environment contributed to one rollout. [N]

    A cut at step ``t`` breaks the chain between ``t`` and ``t+1``, so only cuts
    before the final collected step split the rollout; a cut on the last step
    ends the piece that is already being counted.
    """
    T, N = dones.shape
    cut = _cut_tn(dones, truncations)
    if T > 1:
        return T, cut[:-1].sum(dim=0) + 1.0
    return T, torch.ones(N, device=dones.device)


def segment_lengths_tn(dones: Tensor, truncations: Tensor | None = None) -> Tensor:
    """Length, in transitions, of the segment each transition belongs to. [T, N]

    This is the ``l`` of VAPO eq. 5 read literally: there ``l`` is the length of
    the response the token sits in, and here it is the length of the uncut run of
    transitions the λ-chain actually accumulates over. Summarizing an
    environment by one mean length instead is strictly worse under heterogeneous
    segments, because the long segment holding almost all of an environment's
    transitions is dragged toward the short ones: for segments of 1, 1 and 4094
    transitions the mean is 1365 against the 4094 that 99.9% of the transitions
    actually experienced, which halves their credit window.
    """
    T, N = dones.shape
    cut = _cut_tn(dones, truncations)
    # Segment index of each transition: the number of cuts strictly before it.
    index = torch.zeros_like(cut)
    if T > 1:
        index[1:] = cut[:-1].cumsum(dim=0)
    counts = torch.zeros_like(cut)
    counts.scatter_add_(0, index.long(), torch.ones_like(cut))
    # Gather each transition's own segment length back out of the histogram.
    return counts.gather(0, index.long())


def length_adaptive_lambda(lengths: Tensor, alpha: float) -> Tensor:
    """VAPO eq. 5: ``λ_policy = 1 - 1/(alpha * l)``, clamped to ``[0, 1)``.

    Eq. 4 asks the TD-error coefficients to sum to ``alpha * l`` so their mass
    spreads over the whole segment instead of the first few transitions. Short
    segments (``alpha * l <= 1``) have no window to spread over and clamp to a
    one-step TD target; the upper clamp stays strictly below 1 so the λ-chain
    remains a contraction in floating point.
    """
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"length-adaptive alpha must be finite and positive, got {alpha}")
    lengths = lengths.float()
    # NaN fails `< 0`, and clamp propagates it into every advantage downstream.
    if not bool(torch.isfinite(lengths).all()) or bool((lengths < 0).any()):
        raise ValueError("segment lengths must be finite and non-negative")
    scaled = (alpha * lengths).clamp_min(torch.finfo(lengths.dtype).tiny)
    lam = 1.0 - 1.0 / scaled
    return lam.clamp(0.0, 1.0 - torch.finfo(lam.dtype).eps)


def decoupled_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    gamma: float,
    lambda_policy: float | Tensor,
    lambda_critic: float = 1.0,
    next_value: Tensor | None = None,
    truncations: Tensor | None = None,
    next_values_tn: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """VAPO §4.1 Decoupled-GAE: one estimator per consumer, one call.

    Both passes share the truncation semantics of `compute_gae_tn`: a true
    terminal bootstraps 0, a time-limit truncation bootstraps V(terminal_obs),
    and the λ-chain cuts on any done. With that bootstrapping the λ_critic=1
    pass is the discounted Monte-Carlo return over the segment, which is what
    makes it an unbiased critic target rather than a bootstrapped one.

    Returns:
      advantages — GAE(gamma, λ_policy), used by the policy surrogate
      returns    — GAE(gamma, λ_critic) + behavior values, the critic target
      info       — estimator parameters and both geometric effective horizons
    """
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be finite and in (0, 1], got {gamma}")
    lambda_critic = float(lambda_critic)
    if not 0.0 <= lambda_critic <= 1.0:
        raise ValueError(f"lambda_critic must be in [0, 1], got {lambda_critic}")
    lam_p = torch.as_tensor(lambda_policy, dtype=torch.float32)
    # A λ outside [0, 1) turns the chain from a contraction into a divergence,
    # and NaN poisons every advantage in the rollout. Validate both λ, not one.
    if not bool(torch.isfinite(lam_p).all()) or bool(
        ((lam_p < 0.0) | (lam_p >= 1.0)).any()
    ):
        raise ValueError("lambda_policy must be finite and in [0, 1)")
    advantages, _ = compute_gae_tn(
        rewards,
        values,
        dones,
        gamma=gamma,
        lam=lambda_policy,
        next_value=next_value,
        truncations=truncations,
        next_values_tn=next_values_tn,
    )
    _, returns = compute_gae_tn(
        rewards,
        values,
        dones,
        gamma=gamma,
        lam=lambda_critic,
        next_value=next_value,
        truncations=truncations,
        next_values_tn=next_values_tn,
    )
    lam_mean = float(lam_p.mean())
    critic_decay = gamma * lambda_critic
    # Mean of the per-transition horizons, not the horizon of the mean λ: the
    # two differ by Jensen once λ varies, and the former is the quantity the
    # rollout's transitions actually got.
    horizons = 1.0 / (1.0 - gamma * lam_p).clamp_min(torch.finfo(lam_p.dtype).tiny)
    info = {
        "gamma": gamma,
        "gae_lambda_critic": lambda_critic,
        "gae_lambda_policy_mean": lam_mean,
        "gae_lambda_policy_min": float(lam_p.min()),
        "gae_lambda_policy_max": float(lam_p.max()),
        "gae_policy_decay": gamma * lam_mean,
        "policy_effective_horizon": float(horizons.mean()),
        "policy_effective_horizon_min": float(horizons.min()),
        "critic_effective_horizon": _geometric_horizon(critic_decay),
    }
    return advantages, returns, info


def _geometric_horizon(decay: float) -> float:
    """``1/(1 - gamma*lambda)``, the transitions a TD-error still reaches."""
    if decay >= 1.0:
        return math.inf
    return 1.0 / (1.0 - decay)


def segment_returns_per_env(
    rewards: Tensor, dones: Tensor, truncations: Tensor | None = None,
) -> Tensor:
    """Mean undiscounted return of the segments an environment collected. [N]

    Undiscounted, because a discounted sum would rank an environment by where in
    its segment the reward landed rather than by how much it earned. Per segment
    rather than per rollout, because the unit VAPO imitates is one trajectory:
    an environment that ran four segments earning 25 each behaves worse than one
    that ran a single segment earning 100, and the plain rollout sum calls them
    equal. Every environment collects exactly T transitions, so this equals the
    reward rate times the mean segment length; the cost of that is a second
    penalty on an environment whose creeps keep dying, since each death both
    ends a segment and forfeits its future reward.
    """
    if rewards.shape != dones.shape:
        raise ValueError("rewards and dones must have the same [T, N] shape")
    _, pieces = _segment_pieces(dones, truncations)
    return rewards.float().sum(dim=0) / pieces


def self_imitation_mask(
    rewards: Tensor,
    dones: Tensor,
    truncations: Tensor | None = None,
    *,
    quantile: float,
) -> Tensor:
    """[T, N] bool — transitions eligible for the positive-example NLL term.

    VAPO eq. 9 imitates the samples that got the answer right (§4.3). A verifier
    makes that set exact; our reward is dense, so "right" has to be relative:
    the top ``1 - quantile`` of this rollout's per-environment segment returns,
    recomputed per rollout so the bar tracks the policy instead of freezing an
    absolute threshold the policy either always or never clears.

    Selection is by rank, not by comparison against the interpolated quantile
    value. On distinct returns the two agree exactly, but `>=` a quantile is not
    tie-safe, and ties are the normal early-training state here: the reward is
    `0.1*harvest + 1.0*control`, so every colony that harvests nothing scores
    exactly 0.0. With 20 of 24 environments tied at zero, `>= quantile(0.8)`
    admits all 24 — eq. 9 would silently become unconditional cloning of the
    policy at the moment it is least worth cloning. A rank cut keeps the
    positive set at the intended size whatever the distribution does, and an
    entirely flat rollout yields an empty set rather than five arbitrary
    failures.
    """
    quantile = float(quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"self-imitation quantile must be in [0, 1], got {quantile}")
    T, N = rewards.shape
    segment_return = segment_returns_per_env(rewards, dones, truncations)
    selected = torch.zeros(N, dtype=torch.bool, device=rewards.device)
    # A rank cut fixes the size of the positive set but cannot tell whether its
    # members are worth imitating. When every environment scored identically —
    # the collapsed all-zero rollout — there is no upper slice, and imitating an
    # arbitrary five failures is worse than imitating nothing.
    if bool(torch.isfinite(segment_return).all()) and float(
        segment_return.max() - segment_return.min()
    ) > 0.0:
        # ceil((1-q)*N) is exactly how many samples sit at or above the
        # interpolated q-quantile of N distinct values; keep at least the best.
        keep = max(1, math.ceil((1.0 - quantile) * N))
        winners = torch.topk(segment_return, keep, largest=True, sorted=False).indices
        selected[winners] = True
    return selected.unsqueeze(0).expand(T, N)


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
