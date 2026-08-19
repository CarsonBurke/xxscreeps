"""Lazily growing host rollout buffers with sparse spatial pages.

PPO collect writes into preallocated [T_max, N, ...] tensors. This is the
reasonable middle ground between:
  · list-append + torch.stack every rollout (alloc churn, fragmented host RAM)
  · true multi-producer ring across updates (more complexity than we need)

Non-spatial observations retain ordinary dense ``[T, N, ...]`` storage. Room
patches are different: the model ABI reserves four rooms, but most transitions
only contain one or two. Storing four roughly 68 KiB pages per transition wastes
most rollout RAM. Active room pages are therefore appended to reusable chunks;
the dense room-slot ABI is reconstructed only for the optimizer minibatch being
transferred to the device.

Capacity starts at the base horizon and grows only when adaptive collection
extends it. Compile stability is separate: GPU graphs key on B (N or mb), not T.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .constants import (
    ACTOR_FEAT,
    CONSTRUCTION_MASK_BYTES,
    GLOBAL_FEAT,
    INTENT_SLOTS,
    MAX_ACTORS,
    MAX_ROOMS,
    MAX_TARGETS,
    MODEL_CFG,
    N_AMOUNT,
    N_BODY_PART,
    N_CONSTRUCTION_TYPE,
    N_DIR,
    N_INTENT,
    PATCH_FLAT,
    PATCHES_PER_ROOM,
    TARGET_FEAT,
)


# Store every supported room. Expansion may happen mid-episode; slicing to the
# reset-time room count silently trained on a different observation than acting.
_HOST_R = MAX_ROOMS
_LATENT_DIM = int(MODEL_CFG["dModel"])
_ROOM_BUCKETS = (1, 2, MAX_ROOMS)

# Page-store granularity. Every minibatch gather runs one Python-level
# `index_copy_` per chunk its rows touch, so a fixed chunk size makes that loop
# grow with the rollout: 256-page chunks are ~100 iterations at 512×24 and ~800
# at 4096×24. Size chunks from the rollout instead, holding the chunk count
# roughly constant, with a floor for the tiny buffers the tests build and a
# ceiling so the partially filled last chunk cannot waste much.
_PAGE_CHUNK_TARGET = 64
_PAGE_CHUNK_MIN_PAGES = 256
_PAGE_CHUNK_MAX_PAGES = 2048
# Room pages are addressed by int32 ids in `patch_refs`; a wider store would
# wrap silently into another transition's rooms.
_MAX_PAGES = 2**31 - 1


def _pages_per_chunk(t_max: int, n_envs: int) -> int:
    """Chunk size that keeps the gather loop bounded for a `t_max`×`n_envs` rollout.

    Sized on two active rooms, the common case: one room is the early economy
    and four is a remote-mining fleet that only some runs reach.
    """
    expected_pages = max(1, int(t_max) * int(n_envs) * 2)
    chunk = -(-expected_pages // _PAGE_CHUNK_TARGET)
    return max(_PAGE_CHUNK_MIN_PAGES, min(_PAGE_CHUNK_MAX_PAGES, chunk))


# Host dtypes: float feats / uint8 masks (promote on H2D only).
_OBS_SPEC: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "patches": ((_HOST_R, PATCHES_PER_ROOM, PATCH_FLAT), torch.uint8),
    "room_mask": ((_HOST_R,), torch.uint8),
    "room_coords": ((_HOST_R, 2), torch.float32),
    "actors": ((MAX_ACTORS, ACTOR_FEAT), torch.float32),
    "actor_mask": ((MAX_ACTORS,), torch.uint8),
    "actor_outcome": ((MAX_ACTORS,), torch.uint8),
    "targets": ((MAX_TARGETS, TARGET_FEAT), torch.float32),
    "target_mask": ((MAX_TARGETS,), torch.uint8),
    "intent_mask": ((MAX_ACTORS, INTENT_SLOTS, N_INTENT), torch.uint8),
    "dir_mask": ((MAX_ACTORS, INTENT_SLOTS, N_DIR), torch.uint8),
    "target_select_mask": ((N_INTENT, MAX_TARGETS), torch.uint8),
    "amount_mask": ((MAX_ACTORS, INTENT_SLOTS, N_INTENT, N_AMOUNT), torch.uint8),
    "construction_mask": (
        (MAX_ROOMS, N_CONSTRUCTION_TYPE, CONSTRUCTION_MASK_BYTES), torch.uint8,
    ),
    "globals": ((GLOBAL_FEAT,), torch.float32),
}
_AUX_BUFFER_NAMES = (
    "types", "dirs", "targets", "amounts",
    "construction_types", "construction_tiles",
    "body_counts", "body_order", "logprob", "value",
    "actor_latent", "reward", "done", "trunc", "next_value", "term_value", "has_term",
)


def _tensor_nbytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class _PatchPageStore:
    """Append-only, reusable chunks of active uint8 room patches."""

    def __init__(self, pages_per_chunk: int = _PAGE_CHUNK_MIN_PAGES):
        if pages_per_chunk < 1:
            raise ValueError("pages_per_chunk must be positive")
        self.pages_per_chunk = int(pages_per_chunk)
        self.chunks: list[Tensor] = []
        self.count = 0

    def reset(self) -> None:
        """Reuse allocated pages on the next rollout without retaining old refs."""
        self.count = 0

    def _ensure_chunks(self, page_count: int) -> None:
        required = (page_count + self.pages_per_chunk - 1) // self.pages_per_chunk
        while len(self.chunks) < required:
            self.chunks.append(
                torch.empty(
                    self.pages_per_chunk,
                    PATCHES_PER_ROOM,
                    PATCH_FLAT,
                    dtype=torch.uint8,
                )
            )

    def append(self, pages: Tensor) -> Tensor:
        """Store ``[K, P, F]`` pages and return stable int64 page ids."""
        if pages.dim() != 3 or tuple(pages.shape[1:]) != (PATCHES_PER_ROOM, PATCH_FLAT):
            raise RuntimeError(f"patch pages have invalid shape {tuple(pages.shape)}")
        if pages.device.type != "cpu":
            pages = pages.detach().cpu()
        if pages.dtype != torch.uint8:
            pages = (pages.clamp(0, 1) * 255).round().to(torch.uint8)
        pages = pages.contiguous()
        n_pages = int(pages.shape[0])
        ids = torch.arange(self.count, self.count + n_pages, dtype=torch.int64)
        if n_pages == 0:
            return ids
        # Ids run count .. count+n_pages-1, so it is the largest of those that
        # has to stay representable.
        if self.count + n_pages - 1 > _MAX_PAGES:
            raise RuntimeError(
                f"patch page store would exceed the int32 reference range: "
                f"last id {self.count + n_pages - 1} > {_MAX_PAGES}"
            )
        self._ensure_chunks(self.count + n_pages)
        src = 0
        while src < n_pages:
            page_id = self.count + src
            chunk_id, offset = divmod(page_id, self.pages_per_chunk)
            take = min(n_pages - src, self.pages_per_chunk - offset)
            self.chunks[chunk_id][offset : offset + take].copy_(pages[src : src + take])
            src += take
        self.count += n_pages
        return ids

    def gather(self, refs: Tensor, *, room_capacity: int | None = None) -> Tensor:
        """Expand page ids into the requested leading room-capacity bucket."""
        if refs.dim() != 2 or refs.shape[1] != _HOST_R:
            raise RuntimeError(f"patch refs have invalid shape {tuple(refs.shape)}")
        refs = refs.detach().cpu().long()
        room_capacity = _HOST_R if room_capacity is None else int(room_capacity)
        if room_capacity < 1 or room_capacity > _HOST_R:
            raise ValueError(f"invalid room capacity {room_capacity}")
        refs = refs[:, :room_capacity]
        out = torch.zeros(
            refs.shape[0], room_capacity, PATCHES_PER_ROOM, PATCH_FLAT,
            dtype=torch.uint8,
        )
        flat_refs = refs.reshape(-1)
        valid_pos = torch.nonzero(flat_refs >= 0, as_tuple=False).squeeze(-1)
        if valid_pos.numel() == 0:
            return out
        valid_refs = flat_refs.index_select(0, valid_pos)
        if int(valid_refs.max()) >= self.count:
            raise RuntimeError("patch reference points beyond active page storage")
        chunk_ids = torch.div(valid_refs, self.pages_per_chunk, rounding_mode="floor")
        out_flat = out.view(-1, PATCHES_PER_ROOM, PATCH_FLAT)
        for chunk_id in torch.unique(chunk_ids).tolist():
            selected = chunk_ids == chunk_id
            dst = valid_pos[selected]
            local = valid_refs[selected] - chunk_id * self.pages_per_chunk
            out_flat.index_copy_(0, dst, self.chunks[chunk_id].index_select(0, local))
        return out

    @property
    def used_bytes(self) -> int:
        return self.count * PATCHES_PER_ROOM * PATCH_FLAT

    @property
    def allocated_bytes(self) -> int:
        return sum(_tensor_nbytes(chunk) for chunk in self.chunks)


class FlatRolloutObservations:
    """Lazy flat rollout view which materializes patches per minibatch only."""

    def __init__(
        self,
        dense: dict[str, Tensor],
        patch_refs: Tensor,
        patch_pages: _PatchPageStore,
    ):
        self.dense = dense
        self.patch_refs = patch_refs
        self.patch_pages = patch_pages
        self.batch_size = int(patch_refs.shape[0])

    def gather_minibatch(self, indices: Tensor) -> dict[str, Tensor]:
        """Gather observations and reconstruct only the required room bucket."""
        indices = indices.detach().cpu().long()
        out = {key: value.index_select(0, indices) for key, value in self.dense.items()}
        refs = self.patch_refs.index_select(0, indices)
        live_rooms = torch.nonzero(out["room_mask"] > 0, as_tuple=False)
        required = 1 if live_rooms.numel() == 0 else int(live_rooms[:, 1].max()) + 1
        room_capacity = next(cap for cap in _ROOM_BUCKETS if required <= cap)
        out["patches"] = self.patch_pages.gather(
            refs, room_capacity=room_capacity,
        )
        return out


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
            if k != "patches":
                self.obs[k] = torch.zeros((self.t_max, self.n, *shape), dtype=dtype)
        self.patch_refs = torch.full(
            (self.t_max, self.n, _HOST_R), -1, dtype=torch.int32,
        )
        self.patch_pages = _PatchPageStore(_pages_per_chunk(self.t_max, self.n))

        # Device-side action/value rows stay on GPU during collect; we only
        # preallocate host copies if needed. Here we keep device lists out —
        # caller stores device tensors in parallel fixed buffers optionally.
        self.types = torch.zeros(self.t_max, self.n, MAX_ACTORS, INTENT_SLOTS, dtype=torch.uint8)
        self.dirs = torch.zeros_like(self.types)
        self.targets = torch.zeros_like(self.types)
        self.amounts = torch.zeros_like(self.types)
        self.construction_types = torch.zeros_like(self.types)
        self.construction_tiles = torch.zeros(
            self.t_max, self.n, MAX_ACTORS, INTENT_SLOTS, dtype=torch.int16,
        )
        self.body_counts = torch.zeros(
            self.t_max,
            self.n,
            MAX_ACTORS,
            INTENT_SLOTS,
            N_BODY_PART,
            dtype=torch.uint8,
        )
        self.body_order = torch.arange(N_BODY_PART, dtype=torch.uint8).view(
            1, 1, 1, 1, N_BODY_PART,
        ).expand(self.t_max, self.n, MAX_ACTORS, INTENT_SLOTS, -1).clone()
        self.logprob = torch.zeros(self.t_max, self.n, MAX_ACTORS, dtype=torch.float32)
        self.actor_latent = torch.zeros(
            self.t_max, self.n, _LATENT_DIM, dtype=torch.float32,
        )
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
        self.patch_pages.reset()

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
        refs = torch.full((new_cap, self.n, _HOST_R), -1, dtype=torch.int32)
        refs[: self.t].copy_(self.patch_refs[: self.t])
        self.patch_refs = refs
        if self.patch_pages.count == 0:
            # Growth before a rollout starts can still right-size the chunking;
            # mid-rollout it cannot, because live page ids are addressed modulo
            # the current chunk size.
            self.patch_pages = _PatchPageStore(_pages_per_chunk(new_cap, self.n))
        for name in _AUX_BUFFER_NAMES:
            setattr(self, name, grow(getattr(self, name)))
        self.t_max = new_cap

    def __len__(self) -> int:
        return self.t

    def write_obs_host(self, t: int, host_obs: dict[str, Tensor]) -> None:
        """Copy stacked host obs [N,...] into slot t (uint8 masks OK).

        Retains every configured room so mid-episode expansion remains learnable.
        """
        patches = host_obs["patches"]
        if patches.dim() == 5 and patches.shape[0] == 1:
            patches = patches.squeeze(0)
        if patches.shape[0] != self.n:
            raise RuntimeError(f"obs[patches] N={patches.shape[0]} expected {self.n}")
        patches = patches[:, :_HOST_R]
        expected = (self.n, _HOST_R, PATCHES_PER_ROOM, PATCH_FLAT)
        if tuple(patches.shape) != expected:
            raise RuntimeError(f"obs[patches] shape {tuple(patches.shape)} vs expected {expected}")
        room_mask = host_obs["room_mask"]
        if room_mask.dim() == 3 and room_mask.shape[0] == 1:
            room_mask = room_mask.squeeze(0)
        active = room_mask[:, :_HOST_R].detach().cpu() > 0
        selected = patches.detach().cpu()[active]
        ids = self.patch_pages.append(selected)
        self.patch_refs[t].fill_(-1)
        self.patch_refs[t][active] = ids.to(torch.int32)

        for k, buf in self.obs.items():
            v = host_obs[k]
            if v.dim() == buf.dim() and v.shape[0] == 1:
                v = v.squeeze(0)
            if v.shape[0] != self.n:
                raise RuntimeError(f"obs[{k}] N={v.shape[0]} expected {self.n}")
            # The encoder currently pads to MAX_ROOMS; keep defensive slicing in
            # case a future wire schema advertises a larger room capacity.
            if k == "room_mask" and v.shape[1] > _HOST_R:
                v = v[:, :_HOST_R]
            if v.dtype != buf.dtype:
                if buf.dtype == torch.uint8:
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
        actor_latent: Tensor | None = None,
        value: Tensor,
        reward: Tensor,
        done: Tensor,
        trunc: Tensor,
        body_counts: Tensor | None = None,
        body_order: Tensor | None = None,
        construction_types: Tensor | None = None,
        construction_tiles: Tensor | None = None,
    ) -> int:
        """Write one timestep; returns index written. Raises if capacity exceeded.

        Action/reward/value inputs may already be CPU tensors.  The collector
        passes the exact host action planes sent to Node so rollout storage does
        not repeat GPU→CPU transfers.
        """
        if self.t >= self.t_max:
            raise RuntimeError(f"rollout buffer full t_max={self.t_max}")
        t = self.t
        self.write_obs_host(t, host_obs)
        def _host(value: Tensor, dtype: torch.dtype) -> Tensor:
            value = value.detach()
            if value.device.type == "cpu" and value.dtype == dtype:
                return value
            return value.to("cpu", dtype=dtype)

        def _copy_actor_prefix(destination: Tensor, value: Tensor, dtype: torch.dtype) -> None:
            source = _host(value, dtype)
            if source.shape[0] != self.n or source.shape[1] > MAX_ACTORS:
                raise RuntimeError(
                    f"action prefix shape {tuple(source.shape)} is incompatible with N={self.n}"
                )
            destination.zero_()
            destination[:, : source.shape[1]].copy_(source)

        _copy_actor_prefix(self.types[t], types, torch.uint8)
        _copy_actor_prefix(self.dirs[t], dirs, torch.uint8)
        _copy_actor_prefix(self.targets[t], targets, torch.uint8)
        _copy_actor_prefix(self.amounts[t], amounts, torch.uint8)
        if body_counts is not None:
            _copy_actor_prefix(self.body_counts[t], body_counts, torch.uint8)
        else:
            self.body_counts[t].zero_()
        if body_order is not None:
            _copy_actor_prefix(self.body_order[t], body_order, torch.uint8)
        else:
            self.body_order[t].copy_(
                torch.arange(N_BODY_PART, dtype=torch.uint8).view(1, 1, -1)
            )
        if construction_types is not None:
            _copy_actor_prefix(self.construction_types[t], construction_types, torch.uint8)
        else:
            self.construction_types[t].zero_()
        if construction_tiles is not None:
            _copy_actor_prefix(self.construction_tiles[t], construction_tiles, torch.int16)
        else:
            self.construction_tiles[t].zero_()
        _copy_actor_prefix(self.logprob[t], logprob, torch.float32)
        if actor_latent is None:
            self.actor_latent[t].zero_()
        else:
            latent = _host(actor_latent, torch.float32)
            if tuple(latent.shape) != (self.n, _LATENT_DIM):
                raise RuntimeError(
                    f"actor latent shape {tuple(latent.shape)} expected {(self.n, _LATENT_DIM)}"
                )
            self.actor_latent[t].copy_(latent)
        self.value[t].copy_(value.detach().float().cpu())
        self.reward[t].copy_(_host(reward, torch.float32))
        self.done[t].copy_(_host(done, torch.float32))
        self.trunc[t].copy_(trunc.detach().float().cpu())
        self.has_term[t].zero_()
        self.term_value[t].zero_()
        self.t += 1
        return t

    def set_term_values(self, t: int, term_v: Tensor, mask: Tensor) -> None:
        """term_v, mask shape [N] on any device — host store for GAE splice."""
        self.term_value[t].copy_(term_v.detach().float().cpu())
        self.has_term[t].copy_(mask.detach().cpu().bool())

    def as_flat_obs(self) -> FlatRolloutObservations:
        """Lazy ``[T*N, ...]`` view; patches remain sparse until mb gather."""
        T = self.t
        out: dict[str, Tensor] = {}
        for k, buf in self.obs.items():
            v = buf[:T]
            out[k] = v.reshape(T * self.n, *v.shape[2:]).contiguous()
        refs = self.patch_refs[:T].reshape(T * self.n, _HOST_R)
        return FlatRolloutObservations(out, refs, self.patch_pages)

    def storage_bytes(self, *, allocated: bool = True) -> int:
        """Current host allocation, useful for capacity and regression tests."""
        dense = sum(_tensor_nbytes(value) for value in self.obs.values())
        dense += _tensor_nbytes(self.patch_refs)
        for name in _AUX_BUFFER_NAMES:
            dense += _tensor_nbytes(getattr(self, name))
        spatial = (
            self.patch_pages.allocated_bytes if allocated else self.patch_pages.used_bytes
        )
        return dense + spatial

    @staticmethod
    def _per_transition_bytes() -> int:
        """Dense host bytes one transition occupies, excluding room pages."""
        per_transition = 0
        for key, (shape, dtype) in _OBS_SPEC.items():
            if key == "patches":
                continue
            elements = 1
            for dim in shape:
                elements *= dim
            per_transition += elements * torch.empty((), dtype=dtype).element_size()
        action_elements = (5 + 2 * N_BODY_PART) * MAX_ACTORS * INTENT_SLOTS
        per_transition += action_elements * torch.empty((), dtype=torch.uint8).element_size()
        per_transition += (
            MAX_ACTORS * INTENT_SLOTS
            * torch.empty((), dtype=torch.int16).element_size()
        )
        per_transition += MAX_ACTORS * torch.empty((), dtype=torch.float32).element_size()
        per_transition += _LATENT_DIM * torch.empty((), dtype=torch.float32).element_size()
        # value, reward, done, trunc, next_value, term_value + has_term
        per_transition += 6 * torch.empty((), dtype=torch.float32).element_size()
        per_transition += torch.empty((), dtype=torch.bool).element_size()
        return per_transition

    @staticmethod
    def dense_storage_bytes(t_max: int, n_envs: int) -> int:
        """Bytes the previous dense-four-room implementation allocated."""
        t_max, n_envs = int(t_max), int(n_envs)
        dense_patches = (
            t_max * n_envs * _HOST_R * PATCHES_PER_ROOM * PATCH_FLAT
        )
        return t_max * n_envs * HostRolloutBuffer._per_transition_bytes() + dense_patches

    @staticmethod
    def projected_bytes(t_max: int, n_envs: int, *, rooms: int = 1) -> int:
        """Host bytes a filled rollout needs with `rooms` live rooms per transition.

        Scaling the horizon scales this linearly, and a 4096-step rollout across
        24 environments is 98,304 transitions: the dense tables alone are close
        to 6 GiB and the pages add that again per live room. Worth printing
        before allocating rather than discovering as an OOM mid-rollout.
        """
        if not 1 <= int(rooms) <= _HOST_R:
            raise ValueError(f"rooms must be in [1, {_HOST_R}], got {rooms}")
        t_max, n_envs = int(t_max), int(n_envs)
        per_transition = HostRolloutBuffer._per_transition_bytes()
        per_transition += _HOST_R * torch.empty((), dtype=torch.int32).element_size()
        per_transition += int(rooms) * PATCHES_PER_ROOM * PATCH_FLAT
        return t_max * n_envs * per_transition

    def tn(self, name: str) -> Tensor:
        """[T, N] or [T, N, ...] slice for filled steps."""
        T = self.t
        return getattr(self, name)[:T]
