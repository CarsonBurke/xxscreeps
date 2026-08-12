"""Running statistics for CleanRL / Gymnasium-style reward normalization.

Matches the intent of gymnasium.wrappers.NormalizeReward and SB3 VecNormalize:
  · track discounted return stream R_t = γ R_{t-1} (1−done) + r_t
  · divide step rewards by √Var(R) so GAE / value targets stay O(1)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class RunningMeanStd:
    """Welford parallel algorithm (same as OpenAI baselines / CleanRL)."""

    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        # flatten batch dims, keep trailing shape
        if x.ndim == 0:
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1.0
        else:
            axes = tuple(range(x.ndim - len(self.mean.shape))) if self.mean.shape else tuple(range(x.ndim))
            if not axes:
                batch_mean = x
                batch_var = np.zeros_like(x)
                batch_count = 1.0
            else:
                batch_mean = np.mean(x, axis=axes)
                batch_var = np.var(x, axis=axes)
                batch_count = float(np.prod([x.shape[a] for a in axes]))
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot


class RewardNormalizer:
    """
    CleanRL / Gymnasium NormalizeReward for vectorized rollouts.

    Call `normalize(rewards, dones)` with tensors [T, N] (or [N] per step).
    Returns rewards scaled by 1/√(return_var + eps). Updates running stats
    from the discounted return stream (not from raw r alone).
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8, clip: float | None = 10.0):
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.clip = clip
        self.return_rms = RunningMeanStd(shape=())
        # Per-env discounted return (carried across rollout chunks)
        self._returns: np.ndarray | None = None

    def reset_env(self, env_index: int) -> None:
        if self._returns is not None and 0 <= env_index < len(self._returns):
            self._returns[env_index] = 0.0

    def normalize(self, rewards: Tensor, dones: Tensor) -> Tensor:
        """
        rewards, dones: [T, N] or [N]
        Returns same shape, float32, same device as rewards.
        """
        device = rewards.device
        dtype = rewards.dtype
        r = rewards.detach().float().cpu().numpy()
        d = dones.detach().float().cpu().numpy()
        single = r.ndim == 1
        if single:
            r = r[None, ...]
            d = d[None, ...]
        T, N = r.shape
        if self._returns is None or len(self._returns) != N:
            self._returns = np.zeros(N, dtype=np.float64)

        # Vectorized discounted return stream (still sequential in time, but
        # no Python per-env loops). RMS updated once per timestep batch.
        out = np.empty_like(r, dtype=np.float64)
        ret = self._returns  # (N,)
        g = self.gamma
        for t in range(T):
            ret = ret * g + r[t]
            self.return_rms.update(ret)
            scale = float(np.sqrt(self.return_rms.var + self.epsilon))
            out[t] = r[t] / scale
            ret = ret * (1.0 - d[t])
        self._returns = ret

        if self.clip is not None:
            out = np.clip(out, -self.clip, self.clip)

        ten = torch.as_tensor(out, dtype=dtype, device=device)
        return ten[0] if single else ten

    def stats(self) -> dict[str, float]:
        return {
            "reward_rms_mean": float(self.return_rms.mean),
            "reward_rms_std": float(np.sqrt(self.return_rms.var + self.epsilon)),
            "reward_rms_var": float(self.return_rms.var),
            "reward_rms_count": float(self.return_rms.count),
        }

    def state_dict(self) -> dict:
        return {
            "mean": float(self.return_rms.mean),
            "var": float(self.return_rms.var),
            "count": float(self.return_rms.count),
            "returns": None if self._returns is None else self._returns.tolist(),
            "gamma": self.gamma,
            "clip": self.clip,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: dict, *, restore_returns: bool = False) -> None:
        """Restore aggregate moments; live traces require matching simulator state."""
        self.gamma = float(state.get("gamma", self.gamma))
        self.clip = state.get("clip", self.clip)
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.return_rms.mean = np.array(state["mean"], dtype=np.float64)
        self.return_rms.var = np.array(state["var"], dtype=np.float64)
        self.return_rms.count = float(state["count"])
        rets = state.get("returns") if restore_returns else None
        self._returns = None if rets is None else np.asarray(rets, dtype=np.float64)
