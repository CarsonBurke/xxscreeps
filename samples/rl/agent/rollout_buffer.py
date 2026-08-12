"""Lazily growing host rollout buffers (no list+stack thrash).

PPO collect writes into preallocated [T_max, N, ...] tensors. This is the
reasonable middle ground between:
  · list-append + torch.stack every rollout (alloc churn, fragmented host RAM)
  · true multi-producer ring across updates (more complexity than we need)

Capacity starts at the base horizon and grows only when adaptive collection extends it.
Compile stability is separate: GPU graphs key on B (N or mb), not T.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .constants import (
    ACTOR_FEAT,
    GLOBAL_FEAT,
    INTENT_SLOTS,
    MAX_ACTORS,
    MAX_ROOMS,
    MAX_TARGETS,
    N_AMOUNT,
    N_DIR,
    N_INTENT,
    PATCH_FLAT,
    PATCHES_PER_ROOM,
    TARGET_FEAT,
)


# Store every supported room. Expansion may happen mid-episode; slicing to the
# reset-time room count silently trained on a different observation than acting.
_HOST_R = MAX_ROOMS

# Host dtypes: float feats / uint8 masks (promote on H2D only).
_OBS_SPEC: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "patches": ((_HOST_R, PATCHES_PER_ROOM, PATCH_FLAT), torch.uint8),
    "room_mask": ((_HOST_R,), torch.uint8),
    "room_coords": ((_HOST_R, 2), torch.float32),
    "actors": ((MAX_ACTORS, ACTOR_FEAT), torch.float32),
    "actor_mask": ((MAX_ACTORS,), torch.uint8),
    "targets": ((MAX_TARGETS, TARGET_FEAT), torch.float32),
    "target_mask": ((MAX_TARGETS,), torch.uint8),
    "intent_mask": ((MAX_ACTORS, INTENT_SLOTS, N_INTENT), torch.uint8),
    "dir_mask": ((MAX_ACTORS, INTENT_SLOTS, N_DIR), torch.uint8),
    "target_select_mask": ((N_INTENT, MAX_TARGETS), torch.uint8),
    "amount_mask": ((MAX_ACTORS, INTENT_SLOTS, N_INTENT, N_AMOUNT), torch.uint8),
    "globals": ((GLOBAL_FEAT,), torch.float32),
}


class HostRolloutBuffer:
    """Preallocated [T_max, N, ...] host storage for one PPO rollout."""

    def __init__(self, t_max: int, n_envs: int):
        if t_max < 1 or n_envs < 1:
            raise ValueError(f"t_max={t_max} n_envs={n_envs} must be ≥1")
        self.t_max = int(t_max)
        self.n = int(n_envs)
        self.t = 0  # next write index / steps filled

        self.obs: dict[str, Tensor] = {}
        for k, (shape, dtype) in _OBS_SPEC.items():
            self.obs[k] = torch.zeros((self.t_max, self.n, *shape), dtype=dtype)

        # Device-side action/value rows stay on GPU during collect; we only
        # preallocate host copies if needed. Here we keep device lists out —
        # caller stores device tensors in parallel fixed buffers optionally.
        self.types = torch.zeros(self.t_max, self.n, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long)
        self.dirs = torch.zeros_like(self.types)
        self.targets = torch.zeros_like(self.types)
        self.amounts = torch.zeros_like(self.types)
        self.logprob = torch.zeros(self.t_max, self.n, MAX_ACTORS, dtype=torch.float32)
        self.value = torch.zeros(self.t_max, self.n, dtype=torch.float32)
        self.reward = torch.zeros(self.t_max, self.n, dtype=torch.float32)
        self.done = torch.zeros(self.t_max, self.n, dtype=torch.float32)
        self.trunc = torch.zeros(self.t_max, self.n, dtype=torch.float32)
        # next-values for GAE: V(s_{t+1}) / terminal bootstrap; filled after collect
        self.next_value = torch.zeros(self.t_max, self.n, dtype=torch.float32)
        # Per-step terminal value override (0 if none)
        self.term_value = torch.zeros(self.t_max, self.n, dtype=torch.float32)
        self.has_term = torch.zeros(self.t_max, self.n, dtype=torch.bool)

    def reset(self) -> None:
        self.t = 0

    def ensure_capacity(self, required: int) -> None:
        """Grow lazily when adaptive rollouts extend beyond their base horizon."""
        required = int(required)
        if required <= self.t_max:
            return
        new_cap = max(required, self.t_max + max(64, self.t_max // 2))

        def grow(tensor: Tensor) -> Tensor:
            out = torch.zeros((new_cap, *tensor.shape[1:]), dtype=tensor.dtype)
            out[: self.t].copy_(tensor[: self.t])
            return out

        self.obs = {k: grow(v) for k, v in self.obs.items()}
        for name in (
            "types", "dirs", "targets", "amounts", "logprob", "value",
            "reward", "done", "trunc", "next_value", "term_value", "has_term",
        ):
            setattr(self, name, grow(getattr(self, name)))
        self.t_max = new_cap

    def __len__(self) -> int:
        return self.t

    def write_obs_host(self, t: int, host_obs: dict[str, Tensor]) -> None:
        """Copy stacked host obs [N,...] into slot t (uint8 masks OK).

        Retains every configured room so mid-episode expansion remains learnable.
        """
        for k, buf in self.obs.items():
            v = host_obs[k]
            if v.dim() == buf.dim() and v.shape[0] == 1:
                v = v.squeeze(0)
            if v.shape[0] != self.n:
                raise RuntimeError(f"obs[{k}] N={v.shape[0]} expected {self.n}")
            # The encoder currently pads to MAX_ROOMS; keep defensive slicing in
            # case a future wire schema advertises a larger room capacity.
            if k == "patches" and v.shape[1] > _HOST_R:
                v = v[:, :_HOST_R]
            if k == "room_mask" and v.shape[1] > _HOST_R:
                v = v[:, :_HOST_R]
            if v.dtype != buf.dtype:
                if buf.dtype == torch.uint8:
                    if k == "patches":
                        v = (v.clamp(0, 1) * 255).round().to(dtype=torch.uint8)
                    else:
                        v = (v > 0).to(dtype=torch.uint8)
                else:
                    v = v.to(dtype=buf.dtype)
            if v.shape[1:] != buf.shape[2:]:
                raise RuntimeError(f"obs[{k}] shape {tuple(v.shape)} vs buf {tuple(buf.shape)}")
            buf[t].copy_(v)

    def write_step(
        self,
        *,
        host_obs: dict[str, Tensor],
        types: Tensor,
        dirs: Tensor,
        targets: Tensor,
        amounts: Tensor,
        logprob: Tensor,
        value: Tensor,
        reward: Tensor,
        done: Tensor,
        trunc: Tensor,
    ) -> int:
        """Write one timestep; returns index written. Raises if capacity exceeded."""
        if self.t >= self.t_max:
            raise RuntimeError(f"rollout buffer full t_max={self.t_max}")
        t = self.t
        self.write_obs_host(t, host_obs)
        # actions may be [N,A,S] on device — copy to host buffer
        self.types[t].copy_(types.detach().to("cpu", dtype=torch.long))
        self.dirs[t].copy_(dirs.detach().to("cpu", dtype=torch.long))
        self.targets[t].copy_(targets.detach().to("cpu", dtype=torch.long))
        self.amounts[t].copy_(amounts.detach().to("cpu", dtype=torch.long))
        self.logprob[t].copy_(logprob.detach().float().cpu())
        self.value[t].copy_(value.detach().float().cpu())
        self.reward[t].copy_(reward.detach().float().cpu())
        self.done[t].copy_(done.detach().float().cpu())
        self.trunc[t].copy_(trunc.detach().float().cpu())
        self.has_term[t].zero_()
        self.term_value[t].zero_()
        self.t += 1
        return t

    def set_term_values(self, t: int, term_v: Tensor, mask: Tensor) -> None:
        """term_v, mask shape [N] on any device — host store for GAE splice."""
        self.term_value[t].copy_(term_v.detach().float().cpu())
        self.has_term[t].copy_(mask.detach().cpu().bool())

    def as_flat_obs(self) -> dict[str, Tensor]:
        """[T*N, ...] host views for PPO update (zero-copy reshape)."""
        T = self.t
        out: dict[str, Tensor] = {}
        for k, buf in self.obs.items():
            v = buf[:T]
            out[k] = v.reshape(T * self.n, *v.shape[2:]).contiguous()
        return out

    def tn(self, name: str) -> Tensor:
        """[T, N] or [T, N, ...] slice for filled steps."""
        T = self.t
        return getattr(self, name)[:T]
