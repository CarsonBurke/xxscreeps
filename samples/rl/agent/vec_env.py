"""Vectorized Screeps env — N independent Node sims, stepped in parallel."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from .env_client import ScreepsEnv, stack_batches


_OBS_KEYS = (
    "patches",
    "room_mask",
    "room_coords",
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


def promote_obs_device(
    obs: dict[str, torch.Tensor],
    device: torch.device | str,
    *,
    pin_hold: list[torch.Tensor] | None = None,
    non_blocking: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Host→device promotion. Spatial uint8 is dequantized; masks become float32."""
    device = torch.device(device)
    if non_blocking is None:
        non_blocking = device.type == "cuda"
    out: dict[str, torch.Tensor] = {}
    for k, v in obs.items():
        if k.startswith("_"):
            out[k] = v
            continue
        if not torch.is_tensor(v):
            continue
        src = v
        if non_blocking and src.device.type == "cpu" and not src.is_pinned():
            src = src.pin_memory()
            if pin_hold is not None:
                pin_hold.append(src)
        if src.dtype == torch.uint8:
            t = src.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            if k == "patches":
                t = t.mul(1.0 / 255.0)
        else:
            t = src.to(device, non_blocking=non_blocking)
        # Contiguous once at H2D — avoids per-slot .contiguous() storms in actor heads.
        if not t.is_contiguous():
            t = t.contiguous()
        out[k] = t
    return out


def _clone_host_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detach/clone host tensors for rollout storage (no D2H).

    Keep uint8 masks on host (4× less RAM than float32). Promote only on H2D
    via promote_obs_device — large target_select_mask savings over T×N.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in obs.items():
        if k.startswith("_"):
            continue
        out[k] = v.detach().clone()
    return out


class VecScreepsEnv:
    """N Node env processes; reset/step fan out on a fixed thread pool.

    Dual obs path:
      · `host_obs` — stacked CPU tensors (uint8 masks and quantized patches)
      · device return value — single bulk H2D for the policy
    """

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
        curriculum: str | list[str] | tuple[str, ...] | None = None,
        lean_meta: bool | None = None,
    ):
        self.n = n
        self.device = torch.device(device)
        # Host-side obs from workers (thread-safe); one bulk .to(device) after join.
        # Headful only on env 0 (single Screeps client on :21025).
        if isinstance(curriculum, str):
            curricula = [item.strip() for item in curriculum.split(",") if item.strip()]
        elif curriculum:
            curricula = [str(item) for item in curriculum]
        else:
            curricula = ["empty"]
        self.curricula = tuple(curricula)
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
                curriculum=self.curricula[i % len(self.curricula)],
                lean_meta=lean_meta,
            )
            for i in range(n)
        ]
        self._pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="screeps-env")
        self.headful = bool(headful)
        self.host_obs: dict[str, torch.Tensor] | None = None
        # Pinned host buffers for rewards/dones (avoid per-step tiny H2D allocs).
        self._reward_host = torch.zeros(n, dtype=torch.float32).pin_memory() if self.device.type == "cuda" else torch.zeros(n, dtype=torch.float32)
        self._done_host = torch.zeros(n, dtype=torch.float32).pin_memory() if self.device.type == "cuda" else torch.zeros(n, dtype=torch.float32)
        # Keep last non_blocking H2D sources alive until the next transfer.
        self._pin_hold: list[torch.Tensor] = []

    def reset(self) -> dict[str, torch.Tensor]:
        futs = [self._pool.submit(env.reset) for env in self.envs]
        batches = [f.result() for f in futs]
        self.host_obs = stack_batches(batches)
        self._pin_hold = []
        return promote_obs_device(self.host_obs, self.device, pin_hold=self._pin_hold)

    def _finish_step(
        self,
        results: list[tuple[dict[str, torch.Tensor], float, bool, dict[str, Any]]],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Stack worker results → host_obs + device tensors + infos."""
        obs_list = [r[0] for r in results]
        for i, r in enumerate(results):
            self._reward_host[i] = r[1]
            self._done_host[i] = float(r[2])
        infos = [r[3] for r in results]

        self.host_obs = stack_batches(obs_list)
        pin_hold: list[torch.Tensor] = []
        obs = promote_obs_device(self.host_obs, self.device, pin_hold=pin_hold)
        self._pin_hold = pin_hold
        rew = self._reward_host.clone()
        done = self._done_host.clone()
        return (
            obs,
            rew.to(self.device, non_blocking=False),
            done.to(self.device, non_blocking=False),
            infos,
        )

    @staticmethod
    def _handle_done(
        env: ScreepsEnv,
        o: dict[str, torch.Tensor],
        info: dict[str, Any],
        d: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Time-limit truncation: keep pre-reset obs for GAE bootstrap."""
        if d:
            term = {
                k: v.detach().clone()
                for k, v in o.items()
                if not str(k).startswith("_") and torch.is_tensor(v)
            }
            o = env.reset()
            info = {
                **info,
                "episode_done": True,
                "truncated": True,  # pure time-limit env today
                "terminal_observation": term,
            }
        else:
            info = {**info, "episode_done": False, "truncated": False}
        return o, info

    def step(
        self, actions: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """
        actions tensors are [N, ...] (batched over envs).
        All N Node sims run concurrently; returns stacked obs on self.device.
        Also updates `self.host_obs` for zero-D2H rollout buffering.
        """
        # Slice on host once so worker threads never touch CUDA tensors.
        acts_cpu = {k: v.detach().to("cpu", non_blocking=False) for k, v in actions.items()}

        def _one(i: int) -> tuple[dict[str, torch.Tensor], float, bool, dict[str, Any]]:
            act_i = {k: v[i : i + 1] for k, v in acts_cpu.items()}
            o, r, d, info = self.envs[i].step(act_i)
            o, info = self._handle_done(self.envs[i], o, info, bool(d))
            return o, float(r), bool(d), info

        futs = [self._pool.submit(_one, i) for i in range(self.n)]
        results = [f.result() for f in futs]
        return self._finish_step(results)

    def step_scripted(
        self,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
        dict[str, torch.Tensor],
    ]:
        """Vectorized scripted teacher step (same expert for BC + critic).

        Returns (obs_dev, rewards, dones, infos, actions_tn) where actions_tn is
        padded [N, MAX_ACTORS, INTENT_SLOTS] long tensors for types/dirs/targets/amounts.
        Pre-step decision state is the previous `host_obs` (caller must buffer it).
        """
        from .actions_util import pad_actions
        from .constants import INTENT_SLOTS, MAX_ACTORS

        def _one(i: int) -> tuple[
            dict[str, torch.Tensor], float, bool, dict[str, Any], dict[str, torch.Tensor]
        ]:
            try:
                o, r, d, info, actions = self.envs[i].step_scripted()
                o, info = self._handle_done(self.envs[i], o, info or {}, bool(d))
                return o, float(r), bool(d), info, pad_actions(actions)
            except Exception as err:  # noqa: BLE001
                # Keep fleet alive: hard-reset dead worker; mark done so MC chain cuts.
                # invalid_demo: joint collector must not BC on empty actions.
                print(f"[vec_env] step_scripted env={i} failed ({err!s:.120}); hard_reset", flush=True)
                try:
                    o = self.envs[i].hard_reset()
                except Exception as err2:  # noqa: BLE001
                    print(f"[vec_env] hard_reset env={i} failed ({err2!s:.120})", flush=True)
                    raise
                info = {
                    "episode_done": True,
                    "truncated": True,
                    "recovered": True,
                    "invalid_demo": True,
                    "harvestDelta": 0,
                    "controlDelta": 0,
                }
                empty = {
                    k: torch.zeros(MAX_ACTORS, INTENT_SLOTS, dtype=torch.long)
                    for k in ("types", "dirs", "targets", "amounts")
                }
                return o, 0.0, True, info, empty

        futs = [self._pool.submit(_one, i) for i in range(self.n)]
        results = [f.result() for f in futs]
        base = [(r[0], r[1], r[2], r[3]) for r in results]
        obs, rew, done, infos = self._finish_step(base)
        acts = {
            k: torch.stack([r[4][k] for r in results], dim=0)
            for k in ("types", "dirs", "targets", "amounts")
        }
        return obs, rew, done, infos, acts

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
