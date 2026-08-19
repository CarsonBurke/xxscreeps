"""Vectorized Screeps env — N independent Node sims, stepped in parallel."""
from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from .env_client import ScreepsEnv, stack_batches
from .constants import MAX_ACTORS, MAX_ROOMS, MAX_TARGETS, N_BODY_PART


def configure_host_threads() -> int:
    """Pin host intra-op parallelism to one thread and report the setting.

    Every environment worker thread calls into torch on the host to decode and
    stack observations. With torch's default intra-op pool each of those calls
    fans out across all cores, so N worker threads contend for N times the
    hardware and the vector environment stops scaling entirely: measured
    throughput was flat at ~370 aggregate ticks per second from four to
    twenty-four environments, and rose to ~1,170 with a single intra-op thread.
    Host-side work here is small tensors and byte copies, which never benefited
    from intra-op parallelism in the first place.
    """
    torch.set_num_threads(1)
    return torch.get_num_threads()


_OBS_KEYS = (
    "patches",
    "room_mask",
    "room_coords",
    "actors",
    "actor_mask",
    "actor_outcome",
    "targets",
    "target_mask",
    "intent_mask",
    "dir_mask",
    "target_select_mask",
    "amount_mask",
    "construction_mask",
    "globals",
)

_ACTION_HOST_DTYPES = {
    "types": torch.uint8,
    "dirs": torch.uint8,
    "targets": torch.uint8,
    "amounts": torch.uint8,
    "body_counts": torch.uint8,
    "body_order": torch.uint8,
    "construction_types": torch.uint8,
    "construction_tiles": torch.int16,
    "_behavior_logprob": torch.float32,
    "_behavior_state_latent": torch.float32,
}

_ACTOR_BUCKETS = (8, 16, 32, 64, MAX_ACTORS)
_TARGET_BUCKETS = (16, 32, 64, MAX_TARGETS)
# Live room count reaches MAX_ROOMS once scouting and expansion expose
# neighbors, but spends most of an episode at the seed room plus one. Two
# buckets keep the common case cheap; both are captured up front by
# `PPOTrainer.warmup`, so neither costs a mid-run recompile.
ROOM_BUCKETS = (2, MAX_ROOMS)

# Compaction trades a smaller attended sequence for a variable one. Rooms are
# always compacted: the count is stable at the live rooms and the model's frozen
# room pack is built from it. Actor and target capacity is what churns, climbing
# 8 -> 16 -> 32 -> 64 -> 100 as a colony grows, so a compiled graph must see one
# fixed capacity or `dynamic=False` mints a graph and a CUDA-graph pool per
# bucket, mid-episode.
_ENTITY_COMPACTION = True


def set_entity_compaction(enabled: bool) -> bool:
    """Enable or disable actor/target compaction; returns the new state."""
    global _ENTITY_COMPACTION
    _ENTITY_COMPACTION = bool(enabled)
    return _ENTITY_COMPACTION


def _capacity_bucket(live: int, buckets: tuple[int, ...]) -> int:
    for capacity in buckets:
        if live <= capacity:
            return capacity
    return buckets[-1]


def _compact_entity_prefixes(
    obs: dict[str, torch.Tensor],
    *,
    entities: bool = True,
    rooms: int | None = None,
) -> dict[str, torch.Tensor]:
    """Slice front-packed entity tensors to a small finite capacity bucket.

    Node packs all live actors and targets contiguously.  Host rollout storage
    retains the full ABI; only model-bound batches are compacted.  Finite
    buckets keep eager execution from attending over mostly-padding
    100-actor/128-target sequences.  A compiled graph wants the opposite: every
    dimension fixed, because `dynamic=False` specializes per shape and both the
    entity bucket and the live room count move during an episode.
    """
    actor_mask = obs.get("actor_mask")
    target_mask = obs.get("target_mask")
    if actor_mask is None or target_mask is None:
        return obs
    def required_prefix(mask: torch.Tensor) -> int:
        live = torch.nonzero(mask.detach().cpu() > 0, as_tuple=False)
        return 1 if live.numel() == 0 else int(live[:, 1].max().item()) + 1

    out = dict(obs)
    room_live = required_prefix(obs["room_mask"])
    room_cap = rooms if rooms is not None else _capacity_bucket(
        max(1, room_live), ROOM_BUCKETS,
    )
    for key in ("patches", "room_mask", "room_coords", "construction_mask"):
        if key not in out:
            continue
        stored = out[key]
        rooms = stored.shape[1]
        if rooms > room_cap:
            out[key] = stored[:, :room_cap]
        elif rooms < room_cap:
            # Host storage packs rooms to its own capacity, which can be below
            # the model bucket. Pad rather than hand a compiled graph a third
            # room shape; the extra rooms are masked out.
            pad = stored.new_zeros(
                (stored.shape[0], room_cap - rooms, *stored.shape[2:]),
            )
            out[key] = torch.cat((stored, pad), dim=1)
    if not entities:
        return out
    actor_live = required_prefix(actor_mask)
    target_live = required_prefix(target_mask)
    actor_cap = _capacity_bucket(max(1, actor_live), _ACTOR_BUCKETS)
    target_cap = _capacity_bucket(max(1, target_live), _TARGET_BUCKETS)
    for key in (
        "actors", "actor_mask", "actor_outcome", "intent_mask",
        "dir_mask", "amount_mask",
    ):
        if key in out:
            out[key] = out[key][:, :actor_cap]
    for key in ("targets", "target_mask"):
        if key in out:
            out[key] = out[key][:, :target_cap]
    if "target_select_mask" in out:
        out["target_select_mask"] = out["target_select_mask"][..., :target_cap]
    return out


def promote_obs_device(
    obs: dict[str, torch.Tensor],
    device: torch.device | str,
    *,
    pin_hold: list[torch.Tensor] | None = None,
    non_blocking: bool | None = None,
    rooms: int | None = None,
) -> dict[str, torch.Tensor]:
    """Host→device promotion. Spatial uint8 is dequantized; masks become float32.

    `rooms` pins the room pack instead of bucketing it. The compiled training
    path uses it: one update graph at `MAX_ROOMS` costs padded spatial tokens,
    while a graph per room pack costs a second minibatch-sized CUDA-graph
    capture pool, and two of those do not fit on one device.
    """
    device = torch.device(device)
    if non_blocking is None:
        non_blocking = device.type == "cuda"
    out: dict[str, torch.Tensor] = {}
    compact = _compact_entity_prefixes(
        obs, entities=_ENTITY_COMPACTION, rooms=rooms,
    )
    for k, v in compact.items():
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
            dtype = torch.uint8 if k == "construction_mask" else torch.float32
            t = src.to(device=device, dtype=dtype, non_blocking=non_blocking)
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
        expert: bool = False,
        bot_dir: str | None = None,
        capture_expert_intents: bool = False,
        seed: int = 0,
    ):
        self.n = n
        self.device = torch.device(device)
        self.expert = bool(expert)
        self.bot_dir = bot_dir
        self.capture_expert_intents = bool(capture_expert_intents)
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
                expert=self.expert,
                bot_dir=bot_dir,
                capture_expert_intents=self.capture_expert_intents,
                seed=int(seed) + i,
            )
            for i in range(n)
        ]
        self._pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="screeps-env")
        self.headful = bool(headful)
        self.host_obs: dict[str, torch.Tensor] | None = None
        # Two reusable host stacks keep s_t alive while workers produce s_{t+1}.
        # CUDA stacks are pinned once; this removes per-tick cat allocations and
        # fourteen repeated pin_memory allocations without racing rollout writes.
        self._obs_host_buffers: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] = (
            {}, {},
        )
        self._obs_host_index = 0
        # Pinned host buffers for rewards/dones (avoid per-step tiny H2D allocs).
        self._reward_host = torch.zeros(n, dtype=torch.float32).pin_memory() if self.device.type == "cuda" else torch.zeros(n, dtype=torch.float32)
        self._done_host = torch.zeros(n, dtype=torch.float32).pin_memory() if self.device.type == "cuda" else torch.zeros(n, dtype=torch.float32)
        # Keep last non_blocking H2D sources alive until the next transfer.
        self._pin_hold: list[torch.Tensor] = []
        # Reused pinned D2H action planes.  Copying every factor with `.cpu()`
        # serialized the CUDA stream once per factor; these buffers launch all
        # copies and require one synchronization before worker threads consume
        # them.
        self._action_host: dict[str, torch.Tensor] = {}
        self.last_host_actions: dict[str, torch.Tensor] | None = None
        self.last_host_reward: torch.Tensor | None = None
        self.last_host_done: torch.Tensor | None = None
        self._restart_state_init()

    def _stack_host_obs(
        self,
        batches: list[dict[str, torch.Tensor]],
        *,
        buffer_index: int,
    ) -> dict[str, torch.Tensor]:
        if len(batches) != self.n:
            raise RuntimeError(f"observation batch count {len(batches)} expected {self.n}")
        buffers = self._obs_host_buffers[buffer_index]
        out: dict[str, torch.Tensor] = {}
        for key in _OBS_KEYS:
            first = batches[0][key]
            shape = (self.n, *first.shape[1:])
            destination = buffers.get(key)
            if (
                destination is None
                or tuple(destination.shape) != shape
                or destination.dtype != first.dtype
            ):
                destination = torch.empty(
                    shape,
                    dtype=first.dtype,
                    device="cpu",
                    pin_memory=self.device.type == "cuda",
                )
                buffers[key] = destination
            torch.cat([batch[key] for batch in batches], dim=0, out=destination)
            out[key] = destination
        return out

    def _actions_to_host(
        self, actions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.device.type != "cuda":
            return {key: value.detach().cpu() for key, value in actions.items()}

        copied: dict[str, torch.Tensor] = {}
        for key, value in actions.items():
            dtype = _ACTION_HOST_DTYPES.get(key, value.dtype)
            buffer = self._action_host.get(key)
            if (
                buffer is None
                or buffer.shape != value.shape
                or buffer.dtype != dtype
            ):
                buffer = torch.empty(
                    value.shape, dtype=dtype, device="cpu", pin_memory=True,
                )
                self._action_host[key] = buffer
            buffer.copy_(value.detach(), non_blocking=True)
            copied[key] = buffer
        torch.cuda.current_stream(self.device).synchronize()
        return copied

    def reset(self) -> dict[str, torch.Tensor]:
        """Start every env. With a `start_provider`, lanes that own a start state
        begin there instead of at tick zero, so a resumed run does not have to
        replay a full fresh segment before its reservoir takes effect."""
        def _one(index: int) -> dict[str, torch.Tensor]:
            path = self.start_provider(index) if self.start_provider is not None else None
            if path is None:
                self._note_restart("reset")
                return self.envs[index].reset()
            self._note_restart("restore")
            return self.envs[index].restore(path)

        futs = [self._pool.submit(_one, index) for index in range(self.n)]
        batches = [f.result() for f in futs]
        self._obs_host_index = 0
        self.host_obs = self._stack_host_obs(batches, buffer_index=self._obs_host_index)
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

        self._obs_host_index = 1 - self._obs_host_index
        self.host_obs = self._stack_host_obs(
            obs_list, buffer_index=self._obs_host_index,
        )
        pin_hold: list[torch.Tensor] = []
        obs = promote_obs_device(self.host_obs, self.device, pin_hold=pin_hold)
        self._pin_hold = pin_hold
        rew = self._reward_host.clone()
        done = self._done_host.clone()
        # Expose the owning CPU copies for rollout storage.  Callers must consume
        # these before the next step overwrites the reusable pinned buffers.
        self.last_host_reward = rew
        self.last_host_done = done
        # Training targets and normalization are host-side; callers that truly
        # need device rewards can promote these tiny vectors explicitly.
        return obs, rew, done, infos

    def _restart_state_init(self) -> None:
        """Initialize start-state control (called from the constructor)."""
        self.start_provider: Callable[[int], str | None] | None = None
        self._restart_requests: set[int] = set()
        self.restart_counts = {"reset": 0, "restore": 0}
        # Restarts run on the worker pool; the counters must not race.
        self._restart_lock = threading.Lock()

    def _note_restart(self, kind: str) -> None:
        with self._restart_lock:
            self.restart_counts[kind] += 1

    def request_restart(self, index: int) -> None:
        """Truncate env `index` at the next step boundary and start a new segment."""
        if not 0 <= index < self.n:
            raise IndexError(f"env index {index} outside [0,{self.n})")
        self._restart_requests.add(int(index))

    def pending_restarts(self) -> frozenset[int]:
        return frozenset(self._restart_requests)

    def snapshot(
        self, requests: Sequence[tuple[int, str, Sequence[str]]],
    ) -> list[dict[str, Any]]:
        """Capture snapshots for several envs concurrently.

        Each request is `(env index, destination path, event tags)`. Capture must
        happen while the env still holds the observed post-tick state, so callers
        issue it immediately after `step` and before the next action.
        """
        if not requests:
            return []
        futures = [
            self._pool.submit(self.envs[index].snapshot, path, tags)
            for index, path, tags in requests
        ]
        out: list[dict[str, Any]] = []
        for (index, path, _tags), future in zip(requests, futures, strict=True):
            try:
                descriptor = future.result()
            except Exception as error:  # noqa: BLE001
                print(
                    f"[vec_env] snapshot env={index} path={path} failed ({error!s:.160})",
                    flush=True,
                )
                continue
            out.append({**descriptor, "env": int(index)})
        return out

    def _handle_done(
        self,
        index: int,
        env: ScreepsEnv,
        o: dict[str, torch.Tensor],
        info: dict[str, Any],
        d: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Episode end or requested segment end: keep pre-restart obs for GAE."""
        requested = index in self._restart_requests
        if not d and not requested:
            return o, {**info, "episode_done": False, "truncated": False}
        self._restart_requests.discard(index)
        term = {
            k: v.detach().clone()
            for k, v in o.items()
            if not str(k).startswith("_") and torch.is_tensor(v)
        }
        path = self.start_provider(index) if self.start_provider is not None else None
        if path is None:
            o = env.reset()
            kind = "reset"
        else:
            o = env.restore(path)
            kind = "restore"
        self._note_restart(kind)
        start_info = env.last_info or {}
        return o, {
            **info,
            # A completed 20k horizon is an episode boundary; a requested segment
            # end is not, but both cut the trajectory and bootstrap from V.
            "episode_done": bool(d),
            "truncated": True,
            "segment_boundary": bool(requested and not d),
            "restart_kind": kind,
            "restart_path": path,
            "start_tick": int(start_info.get("time") or 0),
            "start_step": int(start_info.get("step") or 0),
            "terminal_observation": term,
        }

    def step(
        self, actions: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """
        actions tensors are [N, ...] (batched over envs). Private underscore
        planes may carry behavior-policy statistics to the host staging copy;
        the wire encoder ignores them.
        All N Node sims run concurrently; returns stacked obs on self.device.
        Also updates `self.host_obs` for zero-D2H rollout buffering.
        """
        if self.expert:
            raise RuntimeError(
                "step(actions) drives the learner policy; an expert VecScreepsEnv "
                "must call step_expert() so the teacher owns the intents"
            )
        # Slice on host once so worker threads never touch CUDA tensors.
        acts_cpu = self._actions_to_host(actions)
        self.last_host_actions = acts_cpu

        def _one(i: int) -> tuple[dict[str, torch.Tensor], float, bool, dict[str, Any]]:
            act_i = {k: v[i : i + 1] for k, v in acts_cpu.items()}
            o, r, d, info = self.envs[i].step(act_i)
            o, info = self._handle_done(i, self.envs[i], o, info, bool(d))
            return o, float(r), bool(d), info

        futs = [self._pool.submit(_one, i) for i in range(self.n)]
        results = [f.result() for f in futs]
        return self._finish_step(results)

    def step_expert(
        self,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Vectorized teacher step: the bot in the engine chooses every intent.

        Every `ScreepsEnv` here was built with `expert=True`, so its `step()`
        dispatches `step_expert` and returns the engine's own decisions in `info`
        (`expertIntents`, `expertActorMeta`, `expertTargetMeta`,
        `expertRoomNames`). Those keys are passed back untouched for corpus
        labelling; obs go through the same reused host stacks as `step_scripted`,
        so callers must buffer the pre-step `host_obs` themselves.
        """
        if not self.expert:
            raise RuntimeError(
                "step_expert requires VecScreepsEnv(expert=True); this fleet runs "
                "learner sessions, use step()/step_scripted()"
            )

        def _one(i: int) -> tuple[dict[str, torch.Tensor], float, bool, dict[str, Any]]:
            o, r, d, info = self.envs[i].step()
            o, info = self._handle_done(i, self.envs[i], o, info or {}, bool(d))
            return o, float(r), bool(d), info

        futs = [self._pool.submit(_one, i) for i in range(self.n)]
        results = [f.result() for f in futs]
        return self._finish_step(results)

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
