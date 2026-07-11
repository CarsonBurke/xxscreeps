"""Vectorized Screeps env — N independent Node sims, stepped in parallel."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from .env_client import ScreepsEnv, stack_batches


_OBS_KEYS = (
    "patches",
    "room_mask",
    "actors",
    "actor_mask",
    "targets",
    "target_mask",
    "intent_mask",
    "dir_mask",
    "target_select_mask",
    "amount_mask",
    "globals",
)


def _to_device(obs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Single bulk H2D for model keys (metadata stays host-side if present)."""
    out: dict[str, torch.Tensor] = {}
    non_blocking = device.type == "cuda"
    for k, v in obs.items():
        if k.startswith("_"):
            out[k] = v
            continue
        if non_blocking and v.device.type == "cpu" and not v.is_pinned():
            v = v.pin_memory()
        out[k] = v.to(device, non_blocking=non_blocking)
    return out


class VecScreepsEnv:
    """N Node env processes; reset/step fan out on a fixed thread pool."""

    def __init__(
        self,
        n: int,
        *,
        node: str | None = None,
        room: str = "W7N3",
        max_episode: int = 2000,
        device: str | torch.device = "cpu",
        headful: bool = False,
        headful_password: str = "rlwatch",
        tick_ms: int | None = None,
        no_open: bool = False,
    ):
        self.n = n
        self.device = torch.device(device)
        # Host-side obs from workers (thread-safe); one bulk .to(device) after join.
        # Headful only on env 0 (single Screeps client on :21025).
        self.envs = [
            ScreepsEnv(
                node=node,
                room=room,
                max_episode=max_episode,
                device="cpu",
                headful=(headful and i == 0),
                headful_password=headful_password,
                tick_ms=tick_ms if (headful and i == 0) else None,
                no_open=no_open,
            )
            for i in range(n)
        ]
        self._pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="screeps-env")
        self.headful = bool(headful)

    def reset(self) -> dict[str, torch.Tensor]:
        futs = [self._pool.submit(env.reset) for env in self.envs]
        batches = [f.result() for f in futs]
        return _to_device(stack_batches(batches), self.device)

    def step(
        self, actions: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """
        actions tensors are [N, ...] (batched over envs).
        All N Node sims run concurrently; returns stacked obs on self.device.
        """
        # Slice on host once so worker threads never touch CUDA tensors.
        acts_cpu = {k: v.detach().to("cpu") for k, v in actions.items()}

        def _one(i: int) -> tuple[dict[str, torch.Tensor], float, bool, dict[str, Any]]:
            act_i = {k: v[i : i + 1] for k, v in acts_cpu.items()}
            o, r, d, info = self.envs[i].step(act_i)
            if d:
                o = self.envs[i].reset()
                info = {**info, "episode_done": True}
            else:
                info = {**info, "episode_done": False}
            return o, float(r), bool(d), info

        futs = [self._pool.submit(_one, i) for i in range(self.n)]
        results = [f.result() for f in futs]

        obs_list = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [float(r[2]) for r in results]
        infos = [r[3] for r in results]

        obs = _to_device(stack_batches(obs_list), self.device)
        return (
            obs,
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device),
            infos,
        )

    def close(self) -> None:
        try:
            futs = [self._pool.submit(env.close) for env in self.envs]
            for f in futs:
                try:
                    f.result(timeout=10)
                except Exception:
                    pass
        finally:
            self._pool.shutdown(wait=False, cancel_futures=True)
            for env in self.envs:
                try:
                    if env.proc.poll() is None:
                        env.proc.kill()
                except Exception:
                    pass
