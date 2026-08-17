#!/usr/bin/env python3
"""Immutable, reusable pretraining-corpus collection and storage.

On disk a corpus is a content-addressed directory, not a pickle::

    <output-root>/<corpus-sha256>/
      manifest.json             canonical JSON, no executable Python objects
      shards/<sha256>.pt        tensor-only, grouped by dtype and exact shape

Shards are loaded with ``weights_only=True`` and memory mapping where supported.
The logical object returned by :func:`load_corpus` contains only dictionaries,
lists, scalar primitives, and CPU tensors.  It contains no model or optimizer
state and is safe to reuse across optimizer experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

try:
    from samples.rl.agent.artifacts import directory_signature
    from samples.rl.agent.constants import (
        ACTOR_FEAT, CONSTRUCTION_MASK_BYTES, GLOBAL_FEAT, INTENT_SLOTS,
        INTENT_TYPES, MAX_BODY_PARTS, N_AMOUNT, N_BODY_PART,
        N_CONSTRUCTION_TILE, N_CONSTRUCTION_TYPE, N_DIR, PATCHES_PER_ROOM,
        PATCH_FLAT, PPO_CFG, SCHEMA, SCHEMA_SHA256, TARGET_FEAT,
    )
    from samples.rl.agent.gae import discounted_returns_tn
    from samples.rl.agent.vec_env import _compact_entity_prefixes, configure_host_threads
except ImportError:
    from agent.artifacts import directory_signature
    from agent.constants import (
        ACTOR_FEAT, CONSTRUCTION_MASK_BYTES, GLOBAL_FEAT, INTENT_SLOTS,
        INTENT_TYPES, MAX_BODY_PARTS, N_AMOUNT, N_BODY_PART,
        N_CONSTRUCTION_TILE, N_CONSTRUCTION_TYPE, N_DIR, PATCHES_PER_ROOM,
        PATCH_FLAT, PPO_CFG, SCHEMA, SCHEMA_SHA256, TARGET_FEAT,
    )
    from agent.gae import discounted_returns_tn
    from agent.vec_env import _compact_entity_prefixes


CORPUS_KIND = "pretrain_corpus"
CORPUS_SCHEMA_VERSION = 3
LIFECYCLE_RESERVOIR_CAPACITY = 64
TEMPORAL_REPLAY_PER_STRATUM = 64
TI_CRITIC_RESERVOIR_CAPACITY = 64
TI_FACTOR_RESERVOIR_ALGORITHM = "algorithm_r"
SHARD_MAX_BYTES = 256 * 1024 * 1024
OBS_KEYS = {
    "patches", "room_mask", "room_coords", "actors", "actor_mask",
    "actor_outcome", "targets", "target_mask", "intent_mask", "dir_mask",
    "target_select_mask", "amount_mask", "construction_mask", "globals",
}
TEMPORAL_OBS_KEYS = {
    "patches", "room_mask", "room_coords", "actors", "actor_mask",
    "actor_outcome", "targets", "target_mask", "globals",
}
ACTION_KEYS = {
    "types", "dirs", "targets", "amounts", "body_counts", "body_order",
    "construction_types", "construction_tiles",
}
ACTOR_CAPACITIES = {8, 16, 32, 64, int(SCHEMA["maxActors"])}
TARGET_CAPACITIES = {16, 32, 64, int(SCHEMA["maxTargets"])}
ROOM_CAPACITIES = {1, 2, int(SCHEMA["maxRooms"])}
_FORMAT_SPEC = {
    "version": CORPUS_SCHEMA_VERSION,
    "directory": ["manifest.json", "shards/<sha256>.pt"],
    "splits": ["train", "holdout", "ti_train", "ti_holdout"],
    "scripted_split": [
        "lifecycle_replay", "temporal_replay", "temporal_sampling",
        "rewards_tn", "dones_tn", "returns_tn",
    ],
    "ti_split": ["critic_replay", "rewards_tn", "dones_tn", "returns_tn"],
    "lifecycle_row": ["stratum", "timestep", "env_index", "return_target", "obs", "action"],
    "temporal_row": [
        "stratum", "timestep", "env_index", "terminated", "truncated",
        "obs", "action", "counterfactual_action", "next_obs",
    ],
    "temporal_counterfactual": (
        "same_state_one_live_issued_command_replaced_with_canonical_none"
    ),
    "critic_row": ["stratum", "timestep", "env_index", "return_target", "obs"],
    "spawn_row": ["stratum", "actor_index", "obs", "action"],
    "ti_factor_row": ["timestep", "obs", "action", "eligible"],
    "shard": "tensor-only stacked exact-shape group",
}
CORPUS_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(_FORMAT_SPEC, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

_RL_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY = _RL_ROOT.parents[1]
DEFAULT_OUTPUT = _RL_ROOT / "runs" / "pretrain-corpora"
DEFAULT_TI_BOT_DIR = (_REPOSITORY.parent / "The-International-Open-Source" / "dist").resolve()
DEFAULT_CURRICULUM = "empty,seed_creep,seed_full,seed_claimer,seed_outpost"
OUTPOST_LATE_WINDOW_STEPS = 1_000


def _late_window_size(steps: int) -> int:
    return min(OUTPOST_LATE_WINDOW_STEPS, max(1, steps // 5))

@dataclass(frozen=True)
class CorpusConfig:
    """Immutable simulator and sampling parameters."""

    num_envs: int = 32
    steps: int = 20_000
    max_episode: int = 20_000
    curriculum: str = DEFAULT_CURRICULUM
    room: str = "W7N3"
    seed: int = 3
    holdout_seed_offset: int = 10_000
    gamma: float = float(PPO_CFG["gamma"])
    ti_actor_steps: int = 20_000
    ti_replay_capacity: int = 20_000
    ti_critic_replay_per_stratum: int = TI_CRITIC_RESERVOIR_CAPACITY
    temporal_replay_per_stratum: int = TEMPORAL_REPLAY_PER_STRATUM
    ti_bot_dir: str | None = None
    node: str | None = None

    @property
    def holdout_seed(self) -> int:
        return self.seed + self.holdout_seed_offset

    @property
    def resolved_ti_bot_dir(self) -> Path:
        return Path(self.ti_bot_dir or DEFAULT_TI_BOT_DIR).resolve()

    @property
    def curricula(self) -> tuple[str, ...]:
        return tuple(part.strip() for part in self.curriculum.split(",") if part.strip())

    def env_map(self, seed: int) -> list[dict[str, Any]]:
        return [
            {"env_index": index, "seed": seed + index,
             "curriculum": self.curricula[index % len(self.curricula)]}
            for index in range(self.num_envs)
        ]

    def validate(self) -> None:
        if self.num_envs <= 0 or self.steps <= 0 or self.max_episode <= 0:
            raise ValueError("num_envs, steps, and max_episode must be positive")
        if not self.curricula:
            raise ValueError("curriculum must contain a stage")
        if not math.isfinite(self.gamma) or not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in (0, 1]")
        if self.ti_actor_steps <= 1:
            raise ValueError("ti_actor_steps must be greater than one")
        if self.ti_replay_capacity <= 0:
            raise ValueError("ti_replay_capacity must be positive")
        if self.ti_critic_replay_per_stratum <= 0:
            raise ValueError("ti_critic_replay_per_stratum must be positive")
        if self.temporal_replay_per_stratum <= 0:
            raise ValueError("temporal_replay_per_stratum must be positive")
        train_seeds = {self.seed + index for index in range(self.num_envs)}
        holdout_seeds = {self.holdout_seed + index for index in range(self.num_envs)}
        overlap = train_seeds & holdout_seeds
        if overlap:
            raise ValueError(f"train and holdout environment seeds overlap: {sorted(overlap)}")
        if self.seed in {self.holdout_seed, self.holdout_seed + 1}:
            raise ValueError("TI train/holdout seeds overlap")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_path_set(paths: Iterator[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in paths if path.is_file()}):
        name = path.relative_to(root.resolve()).as_posix().encode()
        digest.update(len(name).to_bytes(4, "little"))
        digest.update(name)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _collection_source_signature() -> str:
    """Fingerprint only code that changes collected labels or return semantics."""
    # Written without a broad agent-directory hash: model/optimizer edits do not
    # invalidate an expensive data artifact.
    paths = [_RL_ROOT / "schema.json"] + [
        _RL_ROOT / "agent" / name for name in (
            "env_client.py", "gae.py", "ti_intents.py", "vec_env.py",
        )
    ] + [
        _RL_ROOT / "env" / name for name in (
            "actions.mjs", "encode.mjs", "scripted_baseline.mjs", "server.mjs",
        )
    ]
    engine = _REPOSITORY / "packages" / "xxscreeps"
    paths.append(engine / "package.json")
    paths.extend(
        path for path in (engine / "dist").rglob("*.js")
        if ".test." not in path.name and "test" not in path.parts
    )
    digest = hashlib.sha256(_hash_path_set(iter(paths), _REPOSITORY).encode())
    for function in (
        _record_teacher_step, _lifecycle_sampling, _scripted_split, _ti_stratum,
        _temporal_stratum, _same_state_counterfactual_action,
        _replace_command_with_none,
        _invalid_scripted_intent_details,
        _late_window_size,
        _append_temporal_replay, _temporal_sampling,
        _reservoir_add, _empty_ti_actions, _ti_split,
    ):
        source = inspect.getsource(function).encode("utf-8")
        digest.update(function.__name__.encode("ascii") + b"\0")
        digest.update(len(source).to_bytes(8, "little"))
        digest.update(source)
    try:
        from samples.rl.agent import pretrain_joint
    except ImportError:
        from agent import pretrain_joint
    for name in (
        "_append_scripted_lifecycle_replay", "_append_spawn_replay",
        "_record_spawn_labels", "_collect_spawn_contract_replay",
        "_parts_to_count_order", "_body_label_factor_mask",
    ):
        source = inspect.getsource(getattr(pretrain_joint, name)).encode("utf-8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(len(source).to_bytes(8, "little"))
        digest.update(source)
    digest.update(_canonical_json({
        "spawn_curricula": list(pretrain_joint.SPAWN_CURRICULA),
        "spawn_replay_per_stratum": pretrain_joint.SPAWN_REPLAY_PER_STRATUM,
        "scripted_replay_per_stratum": pretrain_joint.SCRIPTED_REPLAY_PER_STRATUM,
        "ti_critic_replay_default_per_stratum": TI_CRITIC_RESERVOIR_CAPACITY,
        "non_invalid_intent_codes": sorted(_NON_INVALID_INTENT_CODES),
        "outpost_late_window_steps": OUTPOST_LATE_WINDOW_STEPS,
    }))
    return digest.hexdigest()


def _ti_runtime_signature(directory: Path) -> str | None:
    """Match server-loaded TI runtime inputs: JavaScript modules and WASM."""
    if not directory.is_dir():
        return None
    digest = hashlib.sha256()
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".js", ".mjs", ".cjs", ".wasm"}
    )
    for path in paths:
        relative = path.relative_to(directory).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _runtime_provenance(config: CorpusConfig) -> dict[str, Any]:
    node = config.node or os.environ.get("RL_NODE", "node")
    try:
        node_path = shutil.which(node) or str(Path(node).resolve())
        node_version = subprocess.check_output(
            [node, "--version"], text=True, stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        node_path, node_version = node, "unresolved"
    lock = _RL_ROOT / "uv.lock"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "node_executable": node_path,
        "node_version": node_version,
        "observation_format": os.environ.get("RL_OBS_FMT", "bin"),
        "command_format": os.environ.get("RL_CMD_FMT", "bin"),
        "command_protocol_version": 5,
        "observation_protocol": "XRL1",
        "command_protocol": "XAC1",
        "lean_meta": False,
        "source_count": os.environ.get("RL_SOURCE_COUNT"),
        "expansion_room": os.environ.get("RL_EXPANSION_ROOM"),
        "dependency_lock_sha256": _sha256_file(lock) if lock.is_file() else None,
    }


def _new_totals() -> dict[str, Any]:
    return {
        "transitions": 0, "harvest": 0.0, "control": 0.0, "skill": 0.0,
        "delivery": 0.0, "build": 0.0, "claims": 0, "max_creeps": 0,
        "remote_harvest": 0.0, "remote_home_delivery": 0.0,
        "remote_staffed_peak": 0.0, "remote_productive_peak": 0.0,
        "remote_owned_peak": 0.0, "neutral_outposts": 0.0,
        "late_transitions": 0, "late_remote_harvest": 0.0,
        "late_remote_home_delivery": 0.0, "late_remote_staffed_ticks": 0,
        "late_remote_productive_ticks": 0, "late_spawn_success": 0,
        "spawn_success": 0, "issued": 0, "invalid": 0, "recovered": 0,
        "action_type_hist": {name: 0 for name in INTENT_TYPES},
    }


def _new_stage_metrics() -> dict[str, Any]:
    result = _new_totals()
    result["transitions"] = 0.0
    result["intent_by_type"] = {}
    return result


def _record_teacher_step(
    totals: dict[str, Any],
    by_curriculum: dict[str, dict[str, Any]],
    obs: dict[str, torch.Tensor],
    actions: dict[str, torch.Tensor],
    infos: list[dict],
    record_spawn_labels,
    *,
    late_window: bool = False,
) -> None:
    live = obs["actor_mask"] > 0
    chosen = actions["types"][:, :, 0][live]
    histogram = torch.bincount(chosen.long(), minlength=len(INTENT_TYPES))
    for index, name in enumerate(INTENT_TYPES):
        totals["action_type_hist"][name] += int(histogram[index])
    for env_index, raw in enumerate(infos):
        info = raw or {}
        metrics = by_curriculum.setdefault(
            str(info.get("curriculum") or "unknown"), _new_stage_metrics(),
        )
        values = {
            "harvest": float(info.get("harvestDelta") or 0),
            "control": float(info.get("controlDelta") or 0),
            "delivery": float(info.get("transferDelta") or 0),
            "build": float(info.get("buildDelta") or 0),
            "claims": int(info.get("claimDelta") or 0),
            "spawn_success": int(info.get("spawnSuccess") or 0),
            "issued": int(info.get("intentIssued") or 0),
            "invalid": int(info.get("intentInvalid") or 0),
            "recovered": int(bool(info.get("recovered") or info.get("invalid_demo"))),
            "remote_harvest": float(info.get("remoteHarvestDelta") or 0),
            "remote_home_delivery": float(info.get("remoteHomeDeliveryDelta") or 0),
        }
        maxima = {
            "remote_staffed_peak": float(info.get("remoteRoomsStaffedPeak") or 0),
            "remote_productive_peak": float(
                info.get("remoteProductiveCreepsPeak") or 0
            ),
            "remote_owned_peak": float(info.get("remoteOwnedRoomsPeak") or 0),
            "neutral_outposts": float(info.get("neutralOutpostRooms") or 0),
        }
        late_values = {
            "late_remote_harvest": values["remote_harvest"],
            "late_remote_home_delivery": values["remote_home_delivery"],
            "late_remote_staffed_ticks": int(
                float(info.get("remoteRoomsStaffed") or 0) > 0
            ),
            "late_remote_productive_ticks": int(
                float(info.get("remoteProductiveCreeps") or 0) > 0
            ),
            "late_spawn_success": values["spawn_success"],
        }
        for target in (totals, metrics):
            target["transitions"] += 1
            for key, value in values.items():
                target[key] = target.get(key, 0) + value
            # Skill is derived exactly once from its two components.
            target["skill"] += values["harvest"] + values["control"]
            target["max_creeps"] = max(target["max_creeps"], int(info.get("creeps") or 0))
            for key, value in maxima.items():
                target[key] = max(float(target.get(key, 0.0)), value)
            if late_window:
                target["late_transitions"] = target.get("late_transitions", 0) + 1
                for key, value in late_values.items():
                    target[key] = target.get(key, 0) + value
        for name, counts in (info.get("intentByType") or {}).items():
            if not isinstance(counts, dict):
                continue
            aggregate = metrics["intent_by_type"].setdefault(
                str(name), {"issued": 0, "invalid": 0},
            )
            aggregate["issued"] += int(counts.get("issued") or 0)
            aggregate["invalid"] += int(counts.get("invalid") or 0)
        record_spawn_labels(metrics, actions, obs, env_index)


def _lifecycle_sampling(seed: int, seen: Mapping[str, int], rows: list[dict]) -> dict[str, Any]:
    retained: dict[str, int] = {}
    for row in rows:
        retained[row["stratum"]] = retained.get(row["stratum"], 0) + 1
    return {
        "algorithm": "algorithm_r_per_stratum",
        "capacity_per_stratum": LIFECYCLE_RESERVOIR_CAPACITY,
        "rng": "numpy.default_rng/PCG64",
        "seed": seed,
        "seen_by_stratum": dict(sorted(seen.items())),
        "retained_by_stratum": dict(sorted(retained.items())),
    }


def _temporal_stratum(
    obs: dict[str, torch.Tensor],
    actions: dict[str, torch.Tensor],
    info: Mapping[str, Any],
    env_index: int,
) -> str:
    """Bucket a causal transition without depending on another replay lane.

    The action signature is computed directly from the complete joint action at
    this tick.  In particular, temporal pairs are never reconstructed by
    sorting or joining independently sampled lifecycle, TI, or DAgger rows.
    """
    selected = {
        INTENT_TYPES[int(value)]
        for value in actions["types"][env_index, :, 0].tolist()
        if 0 <= int(value) < len(INTENT_TYPES) and int(value) != 0
    }
    categories: list[str] = []
    category_members = {
        "spawn": {"spawnCreep"},
        "harvest": {"harvest"},
        "logistics": {"transfer", "withdraw", "pickup", "drop"},
        "construction": {"build", "repair", "createConstructionSite"},
        "control": {"upgradeController", "claimController", "reserveController"},
    }
    for name, members in category_members.items():
        if selected & members:
            categories.append(name)
    signature = "+".join(categories) if categories else ("other" if selected else "none")
    rcl = int(round(float(obs["globals"][env_index, 0]) * 8))
    population = int(round(float(obs["globals"][env_index, 3]) * 50))
    pop_bucket = "p0_3" if population <= 3 else "p4_11" if population <= 11 else "p12p"
    stage = str(info.get("curriculum") or "unknown")
    return f"{stage}:r{rcl}:{pop_bucket}:{signature}"


def _same_state_counterfactual_action(
    obs: dict[str, torch.Tensor],
    action: dict[str, torch.Tensor],
    env_index: int,
    actor_cap: int,
) -> dict[str, torch.Tensor] | None:
    """Replace one issued command with canonical legal ``none`` in the same state."""
    result = {
        key: value[env_index : env_index + 1, :actor_cap].clone()
        for key, value in action.items()
    }
    live = torch.nonzero(
        obs["actor_mask"][env_index, :actor_cap] > 0, as_tuple=False,
    ).flatten().tolist()
    for actor_index in live:
        chosen_type = int(result["types"][0, actor_index, 0])
        if chosen_type == 0:
            continue
        _replace_command_with_none(result, actor_index)
        return result
    return None


def _replace_command_with_none(
    action: dict[str, torch.Tensor], actor_index: int,
) -> None:
    """Mutate one compact joint-action command to its canonical ``none`` form."""
    for key in (
        "types", "dirs", "targets", "amounts",
        "construction_types", "construction_tiles",
    ):
        action[key][0, actor_index, 0] = 0
    action["body_counts"][0, actor_index, 0].zero_()
    action["body_order"][0, actor_index, 0] = torch.arange(N_BODY_PART)


def _append_temporal_replay(
    replay: list[dict[str, Any]],
    seen_by_stratum: dict[str, int],
    retained_by_stratum: dict[str, list[int]],
    excluded: dict[str, int],
    rng: np.random.Generator,
    obs: dict[str, torch.Tensor],
    actions: dict[str, torch.Tensor],
    next_obs: dict[str, torch.Tensor],
    dones: torch.Tensor,
    infos: list[dict],
    *,
    timestep: int,
    capacity_per_stratum: int,
) -> None:
    """Reservoir-sample genuine ``(s_t, a_t, s_{t+1})`` transitions.

    Episode-ending rows are deliberately excluded because VecScreepsEnv has
    already reset those workers in ``next_obs``.  This prevents a reset state
    from ever becoming a false dynamics target.  The exclusion counters retain
    the terminal-versus-time-limit provenance in the immutable artifact.
    """
    for env_index, raw_info in enumerate(infos):
        info = raw_info or {}
        truncated = bool(info.get("truncated", False))
        episode_done = bool(info.get("episode_done", bool(dones[env_index])))
        terminated = episode_done and not truncated
        if episode_done:
            key = "excluded_truncated" if truncated else "excluded_terminal"
            excluded[key] = excluded.get(key, 0) + 1
            continue
        if bool(dones[env_index]):
            # A done bit without episode metadata is conservatively terminal.
            excluded["excluded_terminal"] = excluded.get("excluded_terminal", 0) + 1
            continue
        stratum = _temporal_stratum(obs, actions, info, env_index)
        count = seen_by_stratum.get(stratum, 0) + 1
        seen_by_stratum[stratum] = count
        indices = retained_by_stratum.setdefault(stratum, [])
        replacement_index: int | None = None
        if len(indices) >= capacity_per_stratum:
            replacement = int(rng.integers(0, count))
            if replacement >= capacity_per_stratum:
                continue
            replacement_index = indices[replacement]

        compact_obs = _compact_entity_prefixes({
            key: value[env_index : env_index + 1] for key, value in obs.items()
        })
        compact_next_obs = _compact_entity_prefixes({
            key: value[env_index : env_index + 1] for key, value in next_obs.items()
        })
        actor_cap = compact_obs["actors"].shape[1]
        counterfactual = _same_state_counterfactual_action(
            obs, actions, env_index, actor_cap,
        )
        row = {
            "stratum": stratum,
            "timestep": int(timestep),
            "env_index": int(env_index),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "obs": {
                key: compact_obs[key].clone() for key in TEMPORAL_OBS_KEYS
            },
            "action": {
                key: value[env_index : env_index + 1, :actor_cap].clone()
                for key, value in actions.items()
            },
            "counterfactual_action": counterfactual,
            "next_obs": {
                key: compact_next_obs[key].clone() for key in TEMPORAL_OBS_KEYS
            },
        }
        if replacement_index is None:
            replay.append(row)
            indices.append(len(replay) - 1)
        else:
            replay[replacement_index] = row


def _temporal_sampling(
    seed: int,
    capacity_per_stratum: int,
    seen: Mapping[str, int],
    rows: list[dict],
    excluded: Mapping[str, int],
) -> dict[str, Any]:
    retained: dict[str, int] = {}
    for row in rows:
        retained[row["stratum"]] = retained.get(row["stratum"], 0) + 1
    return {
        "algorithm": "algorithm_r_per_stratum_causal_pairs",
        "capacity_per_stratum": int(capacity_per_stratum),
        "rng": "numpy.default_rng/PCG64",
        "seed": int(seed),
        "pairing": "same_env_same_tick_pre_action_to_post_action",
        "terminal_policy": "exclude_episode_end_before_vec_reset",
        "seen_by_stratum": dict(sorted(seen.items())),
        "retained_by_stratum": dict(sorted(retained.items())),
        "excluded_terminal": int(excluded.get("excluded_terminal", 0)),
        "excluded_truncated": int(excluded.get("excluded_truncated", 0)),
        "counterfactual_retained": sum(
            row["counterfactual_action"] is not None for row in rows
        ),
    }


_NON_INVALID_INTENT_CODES = frozenset((0, -2, -4, -11))


def _invalid_scripted_intent_details(
    infos: list[dict[str, Any]], env_map: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return stable, actionable provenance for every invalid teacher intent."""
    details: list[dict[str, Any]] = []
    for env_index, raw in enumerate(infos):
        info = raw or {}
        if not int(info.get("intentInvalid") or 0):
            continue
        invalid_results = [
            {
                key: result.get(key)
                for key in (
                    "actor", "type", "code", "err", "executed", "targetKind",
                    "targetId", "targetStructureType",
                )
                if key in result
            }
            for result in (info.get("intentResults") or ())
            if int(result.get("code", -999)) not in _NON_INVALID_INTENT_CODES
        ]
        provenance = env_map[env_index]
        details.append({
            "env_index": env_index,
            "seed": int(provenance["seed"]),
            "curriculum": str(info.get("curriculum") or provenance["curriculum"]),
            "intent_invalid": int(info.get("intentInvalid") or 0),
            "intent_by_type": info.get("intentByType") or {},
            "results": invalid_results,
        })
    return details


def _scripted_split(config: CorpusConfig, *, seed: int, sampling_seed: int) -> tuple[dict, list, dict, dict]:
    try:
        from samples.rl.agent.pretrain_joint import (
            _append_scripted_lifecycle_replay, _append_spawn_replay,
            _record_spawn_labels,
        )
        from samples.rl.agent.vec_env import VecScreepsEnv
    except ImportError:
        from agent.pretrain_joint import (
            _append_scripted_lifecycle_replay, _append_spawn_replay,
            _record_spawn_labels,
        )
        from agent.vec_env import VecScreepsEnv
    rows: list[Any] = []
    seen: dict[str, int] = {}
    retained: dict[str, list[int]] = {}
    rng = np.random.default_rng(sampling_seed)
    temporal_rows: list[dict[str, Any]] = []
    temporal_seen: dict[str, int] = {}
    temporal_retained: dict[str, list[int]] = {}
    temporal_excluded = {"excluded_terminal": 0, "excluded_truncated": 0}
    temporal_seed = sampling_seed ^ 0x4E455854
    temporal_rng = np.random.default_rng(temporal_seed)
    spawn: list[tuple] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    totals = _new_totals()
    by_stage: dict[str, dict[str, Any]] = {}
    envs = VecScreepsEnv(
        config.num_envs, node=config.node, room=config.room, seed=seed,
        max_episode=config.max_episode, device="cpu", curriculum=config.curriculum,
        lean_meta=False,
    )
    env_map = config.env_map(seed)
    try:
        envs.reset()
        for timestep in range(config.steps):
            assert envs.host_obs is not None
            pre = envs.host_obs
            _next, reward, done, infos, actions = envs.step_scripted()
            assert envs.host_obs is not None
            post = envs.host_obs
            if any((info or {}).get("recovered") or (info or {}).get("invalid_demo") for info in infos):
                raise RuntimeError(f"scripted teacher recovered at step={timestep + 1}")
            invalid_details = _invalid_scripted_intent_details(infos, env_map)
            if invalid_details:
                raise RuntimeError(
                    "scripted teacher emitted invalid intent "
                    f"at step={timestep + 1}: "
                    f"{json.dumps(invalid_details, sort_keys=True, separators=(',', ':'))}"
                )
            rewards.append(reward.detach().cpu().float())
            dones.append(done.detach().cpu().float())
            _append_spawn_replay(spawn, pre, actions)
            _append_scripted_lifecycle_replay(
                rows, seen, retained, rng, pre, actions, infos, timestep=timestep,
            )
            _append_temporal_replay(
                temporal_rows, temporal_seen, temporal_retained,
                temporal_excluded, temporal_rng, pre, actions, post, done, infos,
                timestep=timestep,
                capacity_per_stratum=config.temporal_replay_per_stratum,
            )
            _record_teacher_step(
                totals, by_stage, pre, actions, infos, _record_spawn_labels,
                late_window=(
                    timestep >= config.steps - _late_window_size(config.steps)
                ),
            )
            if (timestep + 1) % 250 == 0 or timestep + 1 == config.steps:
                print(f"[corpus] scripted seed={seed} {timestep + 1}/{config.steps}", flush=True)
    finally:
        envs.close()
    rewards_tn, dones_tn = torch.stack(rewards), torch.stack(dones)
    returns_tn = discounted_returns_tn(
        rewards_tn, dones_tn, gamma=config.gamma,
        next_value=torch.zeros(config.num_envs), truncations=torch.zeros_like(dones_tn),
    )
    logical_rows = [{
        "stratum": sample.stratum, "timestep": int(sample.timestep),
        "env_index": int(sample.env_index),
        "return_target": returns_tn[sample.timestep, sample.env_index].clone(),
        "obs": sample.obs, "action": sample.action,
    } for sample in rows]
    return ({
        "lifecycle_replay": logical_rows,
        "temporal_replay": temporal_rows,
        "temporal_sampling": _temporal_sampling(
            temporal_seed, config.temporal_replay_per_stratum,
            temporal_seen, temporal_rows, temporal_excluded,
        ),
        "rewards_tn": rewards_tn,
        "dones_tn": dones_tn, "returns_tn": returns_tn,
        "sampling": _lifecycle_sampling(sampling_seed, seen, logical_rows),
    }, spawn, totals, by_stage)


def _ti_stratum(obs: dict[str, torch.Tensor], timestep: int) -> str:
    values = obs["globals"][0]
    rcl = int(round(float(values[0]) * 8))
    population = int(round(float(values[3]) * 50))
    rooms = int(round(float(values[6]) * 16))
    pop = "p0_3" if population <= 3 else "p4_11" if population <= 11 else "p12p"
    time_bucket = timestep // 1000
    return f"r{rcl}:rooms{rooms}:{pop}:k{time_bucket}"


def _reservoir_add(
    rows: list[dict], retained: dict[str, list[int]], seen: dict[str, int],
    rng: np.random.Generator, row: dict, capacity: int,
) -> None:
    stratum = row["stratum"]
    count = seen.get(stratum, 0) + 1
    seen[stratum] = count
    indices = retained.setdefault(stratum, [])
    if len(indices) < capacity:
        rows.append(row)
        indices.append(len(rows) - 1)
        return
    replacement = int(rng.integers(0, count))
    if replacement < capacity:
        rows[indices[replacement]] = row


def _empty_ti_actions(actor_cap: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    actions = {
        "types": torch.zeros(1, actor_cap, 1, dtype=torch.long),
        "dirs": torch.zeros(1, actor_cap, 1, dtype=torch.long),
        "targets": torch.zeros(1, actor_cap, 1, dtype=torch.long),
        "amounts": torch.zeros(1, actor_cap, 1, dtype=torch.long),
        "construction_types": torch.zeros(1, actor_cap, 1, dtype=torch.long),
        "construction_tiles": torch.zeros(1, actor_cap, 1, dtype=torch.long),
        "body_counts": torch.zeros(1, actor_cap, 1, N_BODY_PART, dtype=torch.long),
        "body_order": torch.arange(N_BODY_PART).view(1, 1, 1, -1).expand(
            1, actor_cap, 1, -1,
        ).clone(),
    }
    return actions, torch.zeros(1, actor_cap, 6 + 2 * N_BODY_PART, dtype=torch.bool)


def _ti_split(
    config: CorpusConfig, *, seed: int, factor_labels: bool,
) -> tuple[dict, list[dict], dict[str, int]]:
    try:
        from samples.rl.agent.env_client import ScreepsEnv
        from samples.rl.agent.pretrain_joint import _body_label_factor_mask, _parts_to_count_order
        from samples.rl.agent.ti_intents import translate_ti_intents
        from samples.rl.agent.vec_env import _clone_host_obs, _compact_entity_prefixes
    except ImportError:
        from agent.env_client import ScreepsEnv
        from agent.pretrain_joint import _body_label_factor_mask, _parts_to_count_order
        from agent.ti_intents import translate_ti_intents
        from agent.vec_env import _clone_host_obs, _compact_entity_prefixes
    steps = config.ti_actor_steps
    critic_rows: list[dict] = []
    critic_seen: dict[str, int] = {}
    critic_retained: dict[str, list[int]] = {}
    critic_seed = seed ^ 0x43524954
    critic_rng = np.random.default_rng(critic_seed)
    factor_rows: list[dict] = []
    factor_counts: dict[str, int] = {}
    factor_seen = 0
    factor_seed = seed ^ 0x5449
    factor_rng = np.random.default_rng(factor_seed)
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    env = ScreepsEnv(
        node=config.node, room=config.room, max_episode=steps + 5, expert=True,
        bot_dir=str(config.resolved_ti_bot_dir), lean_meta=not factor_labels,
        capture_expert_intents=factor_labels, seed=seed,
    )
    try:
        obs = env.reset()
        for timestep in range(steps):
            pre = _clone_host_obs(obs)
            obs, reward, done, info = env.step()
            if done and timestep + 1 < steps:
                raise RuntimeError(f"TI seed={seed} ended early at {timestep + 1}")
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            dones.append(torch.tensor([float(done)], dtype=torch.float32))
            compact = _compact_entity_prefixes(pre)
            _reservoir_add(
                critic_rows, critic_retained, critic_seen, critic_rng,
                {"stratum": _ti_stratum(pre, timestep), "timestep": timestep,
                 "env_index": 0, "obs": {k: v.clone() for k, v in compact.items()}},
                config.ti_critic_replay_per_stratum,
            )
            if factor_labels:
                labels = translate_ti_intents(
                    info.get("expertIntents"), info.get("expertActorMeta") or [],
                    info.get("expertTargetMeta") or [], info.get("expertRoomNames") or [],
                )
                actor_cap = pre["actors"].shape[1]
                actions, eligible = _empty_ti_actions(actor_cap)
                for label in labels:
                    if label.rejection:
                        key = f"rejected:{label.rejection}"
                        factor_counts[key] = factor_counts.get(key, 0) + 1
                        continue
                    if label.actor_index is None or label.actor_index >= actor_cap or label.intent is None:
                        continue
                    if label.intent == "move":
                        key = "rejected:macro_incompatible_move"
                        factor_counts[key] = factor_counts.get(key, 0) + 1
                        continue
                    ai = label.actor_index
                    actions["types"][0, ai, 0] = INTENT_TYPES.index(label.intent)
                    if label.full_action:
                        eligible[0, ai, 0] = True
                    if label.direction is not None:
                        actions["dirs"][0, ai, 0] = label.direction
                        eligible[0, ai, 1] = True
                    if label.target_index is not None:
                        actions["targets"][0, ai, 0] = label.target_index
                        eligible[0, ai, 2] = True
                    if label.construction_type is not None:
                        actions["construction_types"][0, ai, 0] = label.construction_type
                        actions["construction_tiles"][0, ai, 0] = int(label.construction_tile or 0)
                        eligible[0, ai, 4:6] = True
                    if label.body_parts is not None:
                        counts, order, exact = _parts_to_count_order(label.body_parts)
                        actions["body_counts"][0, ai, 0] = counts
                        actions["body_order"][0, ai, 0] = order
                        eligible[0, ai] |= _body_label_factor_mask(counts, exact)
                        if not exact:
                            key = "partial:spawn_body_order_interleaved"
                            factor_counts[key] = factor_counts.get(key, 0) + 1
                    key = f"accepted:{label.intent}:{'full' if label.full_action else 'factor'}"
                    factor_counts[key] = factor_counts.get(key, 0) + 1
                if bool(eligible.any()):
                    compact_cap = compact["actors"].shape[1]
                    row = {
                        "timestep": timestep,
                        "obs": {k: v.clone() for k, v in compact.items()},
                        "action": {k: v[:, :compact_cap].clone() for k, v in actions.items()},
                        "eligible": eligible[:, :compact_cap].clone(),
                    }
                    factor_seen += 1
                    if len(factor_rows) < config.ti_replay_capacity:
                        factor_rows.append(row)
                    else:
                        replacement = int(factor_rng.integers(0, factor_seen))
                        if replacement < config.ti_replay_capacity:
                            factor_rows[replacement] = row
            if (timestep + 1) % 500 == 0 or timestep + 1 == steps:
                print(f"[corpus] TI seed={seed} {timestep + 1}/{steps}", flush=True)
    finally:
        env.close()
    rewards_tn, dones_tn = torch.stack(rewards), torch.stack(dones)
    returns_tn = discounted_returns_tn(
        rewards_tn, dones_tn, gamma=config.gamma, next_value=torch.zeros(1),
        truncations=torch.zeros_like(dones_tn),
    )
    for row in critic_rows:
        row["return_target"] = returns_tn[row["timestep"], 0].clone()
    factor_counts.update({
        "eligible_ticks_seen": factor_seen,
        "replay_rows": len(factor_rows),
    })
    split = {
        "critic_replay": critic_rows, "rewards_tn": rewards_tn,
        "dones_tn": dones_tn, "returns_tn": returns_tn,
        "sampling": _lifecycle_sampling(critic_seed, critic_seen, critic_rows) | {
            "capacity_per_stratum": config.ti_critic_replay_per_stratum,
            "stratum": "observable_rcl_room_count_population_time_bucket",
        },
    }
    if factor_labels:
        split["factor_sampling"] = {
            "algorithm": TI_FACTOR_RESERVOIR_ALGORITHM,
            "capacity": config.ti_replay_capacity,
            "rng": "numpy.default_rng/PCG64",
            "seed": factor_seed,
            "seen": factor_seen,
            "retained": len(factor_rows),
            "counts": dict(sorted(factor_counts.items())),
            "translator_policy": "exact_factors_only;macro_incompatible_move_rejected",
            "translator_abi": SCHEMA["artifact"]["teacherAbi"],
        }
    return split, factor_rows, factor_counts


def _spawn_row(row: tuple) -> dict[str, Any]:
    stratum, actor_index, obs, action = row
    return {"stratum": stratum, "actor_index": int(actor_index), "obs": obs, "action": action}


def _merge_stage_metrics(
    economy: Mapping[str, dict[str, Any]], contracts: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Preserve the historical trainer view while retaining raw sources."""
    result = {stage: dict(metrics) for stage, metrics in economy.items()}
    for stage, metrics in contracts.items():
        target = result.setdefault(stage, {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key == "max_creeps":
                    target[key] = max(float(target.get(key, 0)), float(value))
                else:
                    target[key] = float(target.get(key, 0)) + float(value)
            else:
                target[key] = value
    return result


def collect_corpus(config: CorpusConfig = CorpusConfig()) -> dict[str, Any]:
    """Collect scripted and TI train/holdout corpora entirely on CPU."""
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    train, spawn, train_totals, train_stages = _scripted_split(
        config, seed=config.seed, sampling_seed=config.seed + 7919,
    )
    holdout, _holdout_spawn, holdout_totals, holdout_stages = _scripted_split(
        config, seed=config.holdout_seed, sampling_seed=config.holdout_seed + 7919,
    )
    try:
        from samples.rl.agent.pretrain_joint import _collect_spawn_contract_replay
    except ImportError:
        from agent.pretrain_joint import _collect_spawn_contract_replay
    contracts: list[tuple] = []
    contract_metrics: dict[str, dict[str, Any]] = {}
    _collect_spawn_contract_replay(
        contracts, contract_metrics, node=config.node, room=config.room, seed=config.seed,
    )
    ti_train, ti_factors, ti_counts = _ti_split(config, seed=config.seed, factor_labels=True)
    ti_holdout, _unused_factors, _unused_counts = _ti_split(
        config, seed=config.holdout_seed, factor_labels=False,
    )
    meta = {
        **asdict(config),
        "holdout_seed": config.holdout_seed,
        "ti_bot_dir": str(config.resolved_ti_bot_dir),
        "ti_bot_source_sha256": directory_signature(config.resolved_ti_bot_dir),
        "ti_runtime_source_sha256": _ti_runtime_signature(config.resolved_ti_bot_dir),
        "environment_schema_version": SCHEMA["version"],
        "schema_sha256": SCHEMA_SHA256,
        "collection_source_sha256": _collection_source_signature(),
        "contracts": dict(SCHEMA["artifact"]),
        "runtime": _runtime_provenance(config),
        "expert": "scripted+ti",
        "critic_target": "finite_horizon_discounted_return",
        "critic_endpoint": "zero_at_declared_lifecycle_horizon",
        "teacher_late_window_steps": _late_window_size(config.steps),
        "train_env_map": config.env_map(config.seed),
        "holdout_env_map": config.env_map(config.holdout_seed),
        "ti_env_map": [
            {"split": "train", "env_index": 0, "seed": config.seed, "expert": "ti"},
            {"split": "holdout", "env_index": 0, "seed": config.holdout_seed, "expert": "ti"},
        ],
    }
    data = {
        "train": train, "holdout": holdout,
        "ti_train": ti_train, "ti_holdout": ti_holdout,
        "spawn_replay": [_spawn_row(row) for row in (*contracts, *spawn)],
        "ti_factor_replay": ti_factors,
        "teacher": {
            "train_totals": train_totals,
            "holdout_totals": holdout_totals,
            "train_by_curriculum": train_stages,
            "holdout_by_curriculum": holdout_stages,
            "spawn_contract_by_curriculum": contract_metrics,
            "train_by_curriculum_with_spawn_contracts": _merge_stage_metrics(
                train_stages, contract_metrics,
            ),
            "ti_factor_counts": ti_counts,
        },
    }
    return assemble_corpus(meta, data)


def _hash_value(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(_canonical_json(list(tensor.shape)))
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    elif isinstance(value, Mapping):
        digest.update(b"dict\0")
        for key in sorted(value):
            _hash_value(digest, str(key)); _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"list\0" + len(value).to_bytes(8, "little"))
        for item in value:
            _hash_value(digest, item)
    elif isinstance(value, np.generic):
        _hash_value(digest, value.item())
    elif isinstance(value, float):
        digest.update(b"float\0" + value.hex().encode())
    else:
        digest.update(_canonical_json(value))


def content_sha256(value: Any) -> str:
    digest = hashlib.sha256(); _hash_value(digest, value); return digest.hexdigest()


def assemble_corpus(meta: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    corpus = {
        "kind": CORPUS_KIND, "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_schema_sha256": CORPUS_SCHEMA_SHA256,
        "meta": dict(meta), "data": dict(data),
    }
    corpus["integrity"] = {
        "algorithm": "sha256-semantic-v2",
        "corpus_sha256": content_sha256(corpus),
    }
    validate_corpus(corpus, verify_hashes=True)
    return corpus


def _validate_tensor_dict(value: Any, keys: set[str], location: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{location} keys differ from schema")
    if any(not torch.is_tensor(tensor) or tensor.device.type != "cpu" for tensor in value.values()):
        raise ValueError(f"{location} must contain only CPU tensors")


def _validate_obs(
    obs: Any, location: str, *, keys: set[str] = OBS_KEYS,
) -> tuple[int, int, int]:
    _validate_tensor_dict(obs, keys, location)
    if any(tensor.shape[0] != 1 for tensor in obs.values()):
        raise ValueError(f"{location} must have singleton batch dimensions")
    room, actor, target = obs["patches"].shape[1], obs["actors"].shape[1], obs["targets"].shape[1]
    if room not in ROOM_CAPACITIES or actor not in ACTOR_CAPACITIES or target not in TARGET_CAPACITIES:
        raise ValueError(f"{location} has unsupported compact capacity {(room, actor, target)}")
    expected_shapes = {
        "patches": (1, room, PATCHES_PER_ROOM, PATCH_FLAT),
        "room_mask": (1, room),
        "room_coords": (1, room, 2),
        "actors": (1, actor, ACTOR_FEAT),
        "actor_mask": (1, actor),
        "actor_outcome": (1, actor),
        "targets": (1, target, TARGET_FEAT),
        "target_mask": (1, target),
        "intent_mask": (1, actor, INTENT_SLOTS, len(INTENT_TYPES)),
        "dir_mask": (1, actor, INTENT_SLOTS, 8),
        "target_select_mask": (1, len(INTENT_TYPES), target),
        "amount_mask": (1, actor, INTENT_SLOTS, len(INTENT_TYPES), N_AMOUNT),
        "construction_mask": (
            1, room, N_CONSTRUCTION_TYPE, CONSTRUCTION_MASK_BYTES,
        ),
        "globals": (1, GLOBAL_FEAT),
    }
    expected_dtypes = {
        "patches": torch.uint8, "room_mask": torch.uint8, "room_coords": torch.float32,
        "actors": torch.float32, "actor_mask": torch.uint8, "actor_outcome": torch.uint8,
        "targets": torch.float32, "target_mask": torch.uint8, "intent_mask": torch.uint8,
        "dir_mask": torch.uint8, "target_select_mask": torch.uint8,
        "amount_mask": torch.uint8, "construction_mask": torch.uint8,
        "globals": torch.float32,
    }
    for key in keys:
        dtype = expected_dtypes[key]
        if obs[key].dtype != dtype:
            raise ValueError(f"{location}.{key} dtype={obs[key].dtype}; expected={dtype}")
        if tuple(obs[key].shape) != expected_shapes[key]:
            raise ValueError(
                f"{location}.{key} shape={tuple(obs[key].shape)}; "
                f"expected={expected_shapes[key]}"
            )
        if obs[key].is_floating_point() and not torch.isfinite(obs[key]).all():
            raise ValueError(f"{location}.{key} contains non-finite data")
    return room, actor, target


def _validate_action(
    action: Any, actor_cap: int, location: str, *, target_cap: int | None = None,
) -> None:
    _validate_tensor_dict(action, ACTION_KEYS, location)
    for key, tensor in action.items():
        if tensor.dtype != torch.long:
            raise ValueError(f"{location}.{key} dtype={tensor.dtype}; expected=torch.int64")
        if tensor.shape[0] != 1 or tensor.shape[1] != actor_cap:
            raise ValueError(f"{location}.{key} actor capacity mismatch")
        expected = (
            (1, actor_cap, INTENT_SLOTS, N_BODY_PART)
            if key in {"body_counts", "body_order"}
            else (1, actor_cap, INTENT_SLOTS)
        )
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{location}.{key} shape={tuple(tensor.shape)}; expected={expected}")
    if bool((action["types"] < 0).any()) or bool((action["types"] >= len(INTENT_TYPES)).any()):
        raise ValueError(f"{location}.types out of bounds")
    bounds = {
        "dirs": N_DIR,
        "amounts": N_AMOUNT,
        "construction_types": N_CONSTRUCTION_TYPE,
        "construction_tiles": N_CONSTRUCTION_TILE,
    }
    if target_cap is not None:
        bounds["targets"] = target_cap
    for key, upper in bounds.items():
        if bool((action[key] < 0).any()) or bool((action[key] >= upper).any()):
            raise ValueError(f"{location}.{key} out of bounds")
    body_counts = action["body_counts"]
    if (
        body_counts.shape[-1] != N_BODY_PART
        or bool((body_counts < 0).any())
        or bool((body_counts.sum(dim=-1) > MAX_BODY_PARTS).any())
    ):
        raise ValueError(f"{location}.body_counts invalid")
    body_order = action["body_order"]
    expected_order = torch.arange(N_BODY_PART).expand_as(body_order)
    if not torch.equal(body_order.sort(dim=-1).values, expected_order):
        raise ValueError(f"{location}.body_order is not a permutation")


def _validate_temporal_replay(
    rows: Any,
    sampling: Any,
    *,
    steps: int,
    envs: int,
    dones_tn: torch.Tensor,
    location: str,
) -> None:
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(f"{location}.temporal_replay needs at least two rows")
    retained: dict[str, int] = {}
    for index, row in enumerate(rows):
        row_location = f"{location}.temporal_replay[{index}]"
        required = {
            "stratum", "timestep", "env_index", "terminated", "truncated",
            "obs", "action", "counterfactual_action", "next_obs",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"{row_location} keys differ from schema")
        timestep, env_index = int(row["timestep"]), int(row["env_index"])
        if not 0 <= timestep < steps or not 0 <= env_index < envs:
            raise ValueError(f"{row_location} trajectory reference out of bounds")
        if type(row["terminated"]) is not bool or type(row["truncated"]) is not bool:
            raise ValueError(f"{row_location} terminal flags must be booleans")
        if row["terminated"] or row["truncated"]:
            raise ValueError(f"{row_location} episode-ending transition was retained")
        if bool(dones_tn[timestep, env_index]):
            raise ValueError(f"{row_location} references an episode-ending trajectory cell")
        _, actor_cap, target_cap = _validate_obs(
            row["obs"], f"{row_location}.obs", keys=TEMPORAL_OBS_KEYS,
        )
        _validate_action(
            row["action"], actor_cap, f"{row_location}.action",
            target_cap=target_cap,
        )
        counterfactual = row["counterfactual_action"]
        if counterfactual is not None:
            _validate_action(
                counterfactual, actor_cap, f"{row_location}.counterfactual_action",
                target_cap=target_cap,
            )
            changed_actors: set[int] = set()
            for key in ACTION_KEYS:
                changed = (counterfactual[key] != row["action"][key]).reshape(
                    actor_cap, -1,
                ).any(dim=1)
                changed_actors.update(torch.nonzero(changed, as_tuple=False).flatten().tolist())
            if len(changed_actors) != 1:
                raise ValueError(
                    f"{row_location}.counterfactual_action must change exactly one actor"
                )
            actor_index = changed_actors.pop()
            if not bool(row["obs"]["actor_mask"][0, actor_index]):
                raise ValueError(
                    f"{row_location}.counterfactual_action changes an inactive actor"
                )
            if int(row["action"]["types"][0, actor_index, 0]) == 0:
                raise ValueError(
                    f"{row_location}.counterfactual_action removes no issued command"
                )
            expected = {key: value.clone() for key, value in row["action"].items()}
            _replace_command_with_none(expected, actor_index)
            if any(
                not torch.equal(counterfactual[key], expected[key])
                for key in ACTION_KEYS
            ):
                raise ValueError(
                    f"{row_location}.counterfactual_action is not canonical none"
                )
        _validate_obs(
            row["next_obs"], f"{row_location}.next_obs", keys=TEMPORAL_OBS_KEYS,
        )
        stratum = str(row["stratum"])
        retained[stratum] = retained.get(stratum, 0) + 1
    if not isinstance(sampling, dict):
        raise ValueError(f"{location}.temporal_sampling invalid")
    if sampling.get("algorithm") != "algorithm_r_per_stratum_causal_pairs":
        raise ValueError(f"{location}.temporal_sampling algorithm invalid")
    if sampling.get("pairing") != "same_env_same_tick_pre_action_to_post_action":
        raise ValueError(f"{location}.temporal_sampling pairing provenance invalid")
    if sampling.get("terminal_policy") != "exclude_episode_end_before_vec_reset":
        raise ValueError(f"{location}.temporal_sampling terminal policy invalid")
    if sampling.get("retained_by_stratum") != dict(sorted(retained.items())):
        raise ValueError(f"{location}.temporal_sampling retained counts mismatch")
    capacity = int(sampling.get("capacity_per_stratum", 0))
    if capacity <= 0 or any(count > capacity for count in retained.values()):
        raise ValueError(f"{location}.temporal_sampling capacity violated")
    seen = sampling.get("seen_by_stratum")
    if not isinstance(seen, dict) or any(
        int(seen.get(key, 0)) < count for key, count in retained.items()
    ):
        raise ValueError(f"{location}.temporal_sampling seen counts invalid")
    excluded_terminal = int(sampling.get("excluded_terminal", -1))
    excluded_truncated = int(sampling.get("excluded_truncated", -1))
    if excluded_terminal < 0 or excluded_truncated < 0:
        raise ValueError(f"{location}.temporal_sampling exclusion counts invalid")
    if sum(int(value) for value in seen.values()) + excluded_terminal + excluded_truncated != steps * envs:
        raise ValueError(f"{location}.temporal_sampling transition accounting differs")
    if excluded_terminal + excluded_truncated != int(dones_tn.sum()):
        raise ValueError(f"{location}.temporal_sampling episode-end accounting differs")
    if int(sampling.get("counterfactual_retained", -1)) != sum(
        row["counterfactual_action"] is not None for row in rows
    ):
        raise ValueError(f"{location}.temporal_sampling counterfactual count differs")


def _validate_return_split(
    split: Any, *, steps: int, envs: int, gamma: float,
    rows_key: str, location: str,
) -> None:
    required = {rows_key, "rewards_tn", "dones_tn", "returns_tn", "sampling"}
    if rows_key == "lifecycle_replay":
        required |= {"temporal_replay", "temporal_sampling"}
    optional = {"factor_sampling"}
    if not isinstance(split, dict) or set(split) - optional != required:
        raise ValueError(f"{location} keys differ from schema")
    shape = (steps, envs)
    for key in ("rewards_tn", "dones_tn", "returns_tn"):
        value = split[key]
        if not torch.is_tensor(value) or value.device.type != "cpu" or tuple(value.shape) != shape:
            raise ValueError(f"{location}.{key} shape/device invalid")
        if not torch.isfinite(value).all():
            raise ValueError(f"{location}.{key} non-finite")
    if not bool(((split["dones_tn"] == 0) | (split["dones_tn"] == 1)).all()):
        raise ValueError(f"{location}.dones_tn is not binary")
    expected = discounted_returns_tn(
        split["rewards_tn"], split["dones_tn"], gamma=gamma,
        next_value=torch.zeros(envs), truncations=torch.zeros_like(split["dones_tn"]),
    )
    if not torch.equal(expected, split["returns_tn"]):
        raise ValueError(f"{location}.returns_tn fails exact semantic replay")
    rows = split[rows_key]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{location}.{rows_key} is empty")
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        row_location = f"{location}.{rows_key}[{index}]"
        required_row = {"stratum", "timestep", "env_index", "return_target", "obs"}
        if rows_key == "lifecycle_replay":
            required_row.add("action")
        if not isinstance(row, dict) or set(row) != required_row:
            raise ValueError(f"{row_location} keys differ from schema")
        timestep, env_index = int(row["timestep"]), int(row["env_index"])
        if not 0 <= timestep < steps or not 0 <= env_index < envs:
            raise ValueError(f"{row_location} trajectory reference out of bounds")
        target = row["return_target"]
        if not torch.is_tensor(target) or target.numel() != 1:
            raise ValueError(f"{row_location}.return_target must be scalar tensor")
        if not torch.equal(target.reshape(()), split["returns_tn"][timestep, env_index]):
            raise ValueError(f"{row_location}.return_target differs from trajectory")
        _, actor_cap, target_cap = _validate_obs(row["obs"], f"{row_location}.obs")
        if rows_key == "lifecycle_replay":
            _validate_action(
                row["action"], actor_cap, f"{row_location}.action",
                target_cap=target_cap,
            )
        seen[row["stratum"]] = seen.get(row["stratum"], 0) + 1
    sampling = split["sampling"]
    if not isinstance(sampling, dict) or sampling.get("algorithm") != "algorithm_r_per_stratum":
        raise ValueError(f"{location}.sampling algorithm invalid")
    retained = sampling.get("retained_by_stratum")
    if retained != dict(sorted(seen.items())):
        raise ValueError(f"{location}.sampling retained counts mismatch")
    cap = int(sampling.get("capacity_per_stratum", 0))
    if cap <= 0 or any(count > cap for count in seen.values()):
        raise ValueError(f"{location}.sampling capacity violated")
    observed = sampling.get("seen_by_stratum")
    if not isinstance(observed, dict) or any(int(observed.get(key, 0)) < count for key, count in seen.items()):
        raise ValueError(f"{location}.sampling seen counts invalid")
    if rows_key == "lifecycle_replay":
        _validate_temporal_replay(
            split["temporal_replay"], split["temporal_sampling"],
            steps=steps, envs=envs, dones_tn=split["dones_tn"], location=location,
        )


def _validate_outpost_teacher_readiness(
    teacher: Mapping[str, Any], *, split: str,
) -> None:
    """Fail closed before persisting a corpus with a dead outpost teacher."""
    by_curriculum = teacher.get(f"{split}_by_curriculum")
    if not isinstance(by_curriculum, Mapping):
        raise ValueError(f"teacher {split} curriculum metrics missing")
    metrics = by_curriculum.get("seed_outpost")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"teacher {split} seed_outpost metrics missing")

    positive = (
        "late_transitions",
        "late_remote_harvest",
        "late_remote_home_delivery",
        "late_remote_staffed_ticks",
        "late_remote_productive_ticks",
    )
    missing = [key for key in positive if float(metrics.get(key, 0)) <= 0]
    if missing:
        raise ValueError(
            f"teacher {split} seed_outpost late-window readiness failed: "
            f"nonpositive {','.join(missing)}"
        )
    forbidden = {
        "claims": float(metrics.get("claims", 0)),
        "remote_owned_peak": float(metrics.get("remote_owned_peak", 0)),
        "invalid": float(metrics.get("invalid", 0)),
    }
    nonzero = [key for key, value in forbidden.items() if value != 0]
    if nonzero:
        raise ValueError(
            f"teacher {split} seed_outpost neutrality/validity failed: "
            f"nonzero {','.join(nonzero)}"
        )


def validate_corpus(
    corpus: Mapping[str, Any], *, expected: Mapping[str, Any] | CorpusConfig | None = None,
    verify_hashes: bool = True, verify_source: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(corpus, dict) or set(corpus) != {
        "kind", "corpus_schema_version", "corpus_schema_sha256", "meta", "data", "integrity",
    }:
        raise ValueError("corpus top-level schema invalid")
    if corpus["kind"] != CORPUS_KIND or corpus["corpus_schema_version"] != CORPUS_SCHEMA_VERSION:
        raise ValueError("corpus kind/version incompatible")
    if corpus["corpus_schema_sha256"] != CORPUS_SCHEMA_SHA256:
        raise ValueError("corpus format schema fingerprint differs")
    meta, data = corpus["meta"], corpus["data"]
    required_meta = {
        "num_envs", "steps", "max_episode", "curriculum", "room", "seed",
        "holdout_seed_offset", "holdout_seed", "gamma", "ti_actor_steps",
        "ti_replay_capacity", "ti_critic_replay_per_stratum",
        "temporal_replay_per_stratum",
        "ti_bot_dir", "ti_bot_source_sha256",
        "ti_runtime_source_sha256", "node", "environment_schema_version",
        "schema_sha256", "collection_source_sha256", "contracts", "runtime",
        "expert", "critic_target", "critic_endpoint", "train_env_map",
        "holdout_env_map", "ti_env_map",
    }
    if not isinstance(meta, dict) or required_meta - set(meta):
        raise ValueError("corpus metadata incomplete")
    if meta["schema_sha256"] != SCHEMA_SHA256 or meta["contracts"] != SCHEMA["artifact"]:
        raise ValueError("environment schema/ABI differs")
    if meta["critic_target"] != "finite_horizon_discounted_return" or meta["critic_endpoint"] != "zero_at_declared_lifecycle_horizon":
        raise ValueError("critic target semantics differ")
    config = CorpusConfig(**{
        key: meta[key] for key in asdict(CorpusConfig())
    })
    config.validate()
    if meta["train_env_map"] != config.env_map(config.seed) or meta["holdout_env_map"] != config.env_map(config.holdout_seed):
        raise ValueError("environment seed/curriculum maps differ")
    if verify_source and meta["collection_source_sha256"] != _collection_source_signature():
        raise ValueError("collection-semantic source fingerprint differs")
    required_data = {
        "train", "holdout", "ti_train", "ti_holdout", "spawn_replay",
        "ti_factor_replay", "teacher",
    }
    if not isinstance(data, dict) or set(data) != required_data:
        raise ValueError("corpus data schema invalid")
    gamma = float(meta["gamma"])
    _validate_return_split(
        data["train"], steps=int(meta["steps"]), envs=int(meta["num_envs"]),
        gamma=gamma, rows_key="lifecycle_replay", location="data.train",
    )
    _validate_return_split(
        data["holdout"], steps=int(meta["steps"]), envs=int(meta["num_envs"]),
        gamma=gamma, rows_key="lifecycle_replay", location="data.holdout",
    )
    for split_name in ("train", "holdout"):
        temporal_sampling = data[split_name]["temporal_sampling"]
        if int(temporal_sampling["capacity_per_stratum"]) != config.temporal_replay_per_stratum:
            raise ValueError(f"data.{split_name}.temporal_sampling capacity differs")
    if data["train"]["temporal_sampling"]["seed"] == data["holdout"]["temporal_sampling"]["seed"]:
        raise ValueError("temporal train/holdout sampling RNG seeds overlap")
    _validate_return_split(
        data["ti_train"], steps=int(meta["ti_actor_steps"]), envs=1,
        gamma=gamma, rows_key="critic_replay", location="data.ti_train",
    )
    _validate_return_split(
        data["ti_holdout"], steps=int(meta["ti_actor_steps"]), envs=1,
        gamma=gamma, rows_key="critic_replay", location="data.ti_holdout",
    )
    for index, row in enumerate(data["spawn_replay"]):
        if not isinstance(row, dict) or set(row) != {"stratum", "actor_index", "obs", "action"}:
            raise ValueError(f"spawn row {index} schema invalid")
        _, actor, target = _validate_obs(row["obs"], f"spawn[{index}].obs")
        _validate_action(
            row["action"], actor, f"spawn[{index}].action", target_cap=target,
        )
        if not 0 <= int(row["actor_index"]) < actor:
            raise ValueError(f"spawn row {index} actor reference invalid")
    if len(data["ti_factor_replay"]) > int(meta["ti_replay_capacity"]):
        raise ValueError("TI factor replay exceeds declared capacity")
    for index, row in enumerate(data["ti_factor_replay"]):
        if not isinstance(row, dict) or set(row) != {"timestep", "obs", "action", "eligible"}:
            raise ValueError(f"TI factor row {index} schema invalid")
        _, actor, target = _validate_obs(row["obs"], f"ti_factor[{index}].obs")
        _validate_action(
            row["action"], actor, f"ti_factor[{index}].action", target_cap=target,
        )
        eligible = row["eligible"]
        if not torch.is_tensor(eligible) or eligible.dtype != torch.bool or eligible.shape[:2] != (1, actor):
            raise ValueError(f"TI factor row {index} eligibility invalid")
    factor_sampling = data["ti_train"].get("factor_sampling")
    if not isinstance(factor_sampling, dict) or factor_sampling.get("algorithm") != TI_FACTOR_RESERVOIR_ALGORITHM:
        raise ValueError("TI factor sampling provenance invalid")
    if int(factor_sampling.get("retained", -1)) != len(data["ti_factor_replay"]):
        raise ValueError("TI factor retained count mismatch")
    if int(factor_sampling.get("seen", -1)) < len(data["ti_factor_replay"]):
        raise ValueError("TI factor seen count invalid")
    teacher = data["teacher"]
    for split in ("train", "holdout"):
        totals = teacher.get(f"{split}_totals", {}) if isinstance(teacher, dict) else {}
        if int(totals.get("transitions", -1)) != int(meta["steps"]) * int(meta["num_envs"]):
            raise ValueError(f"teacher {split} transition total differs")
        if int(totals.get("invalid", -1)) or int(totals.get("recovered", -1)):
            raise ValueError(f"teacher {split} contains invalid/recovered transitions")
        if float(totals.get("skill", float("nan"))) != float(totals.get("harvest", 0)) + float(totals.get("control", 0)):
            raise ValueError(f"teacher {split} skill total double-counted or inconsistent")
        if "seed_outpost" in config.curricula:
            _validate_outpost_teacher_readiness(teacher, split=split)
    expected_values = asdict(expected) if isinstance(expected, CorpusConfig) else expected
    if expected_values:
        mismatch = {key: (meta.get(key), value) for key, value in expected_values.items() if meta.get(key) != value}
        if mismatch:
            raise ValueError(f"corpus parameters differ: {mismatch}")
    if verify_hashes:
        unhashed = {key: value for key, value in corpus.items() if key != "integrity"}
        if corpus["integrity"].get("algorithm") != "sha256-semantic-v2" or corpus["integrity"].get("corpus_sha256") != content_sha256(unhashed):
            raise ValueError("corpus semantic SHA-256 mismatch")
    return corpus


def _encode_tensors(value: Any, pending: dict[tuple[str, tuple[int, ...]], list[tuple[str, torch.Tensor]]], path: str = "root") -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        signature = (str(tensor.dtype), tuple(tensor.shape))
        index = len(pending.setdefault(signature, []))
        pending[signature].append((path, tensor))
        return {"$tensor_group": [signature[0], list(signature[1])], "$index": index}
    if isinstance(value, Mapping):
        return {key: _encode_tensors(item, pending, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_tensors(item, pending, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_shards(directory: Path, groups: dict, encoded: Any) -> tuple[Any, list[dict]]:
    shard_dir = directory / "shards"; shard_dir.mkdir()
    group_manifest: dict[str, list[dict]] = {}
    shards: list[dict] = []
    for signature in sorted(groups):
        rows = groups[signature]
        item_bytes = max(1, rows[0][1].numel() * rows[0][1].element_size())
        per_shard = max(1, SHARD_MAX_BYTES // item_bytes)
        key = json.dumps([signature[0], list(signature[1])], separators=(",", ":"))
        locations: list[dict] = []
        for start in range(0, len(rows), per_shard):
            batch = torch.stack([tensor for _path, tensor in rows[start:start + per_shard]])
            temporary = shard_dir / f"group-{len(shards):05d}.pt"
            torch.save({"tensor": batch}, temporary)
            sha = _sha256_file(temporary)
            target = shard_dir / f"{sha}.pt"
            temporary.replace(target)
            record = {"file": f"shards/{target.name}", "sha256": sha, "count": batch.shape[0], "dtype": signature[0], "shape": list(signature[1])}
            shards.append(record)
            locations.append({"file": record["file"], "start": start, "count": batch.shape[0]})
        group_manifest[key] = locations
    def resolve_refs(item: Any) -> Any:
        if isinstance(item, dict) and "$tensor_group" in item:
            key = json.dumps(item["$tensor_group"], separators=(",", ":"))
            index = item["$index"]
            for location in group_manifest[key]:
                if location["start"] <= index < location["start"] + location["count"]:
                    return {"$tensor": location["file"], "$index": index - location["start"]}
            raise AssertionError("tensor group reference unresolved")
        if isinstance(item, dict):
            return {key: resolve_refs(value) for key, value in item.items()}
        if isinstance(item, list):
            return [resolve_refs(value) for value in item]
        return item
    return resolve_refs(encoded), shards


def save_corpus(corpus: Mapping[str, Any], output_root: str | Path) -> Path:
    """Write ``output_root/<content-id>`` once; identical saves are idempotent."""
    validate_corpus(corpus, verify_hashes=True)
    content_id = str(corpus["integrity"]["corpus_sha256"])
    root = Path(output_root)
    destination = root / content_id
    if destination.exists():
        loaded = load_corpus(destination, verify_source=False)
        if loaded["integrity"]["corpus_sha256"] != content_id:
            raise FileExistsError(f"non-identical corpus already exists at {destination}")
        return destination
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{content_id}.", dir=root))
    try:
        groups: dict = {}
        encoded = _encode_tensors(corpus, groups)
        encoded, shards = _write_shards(temporary, groups, encoded)
        manifest = {
            "storage_format": "tensor-shards-v1", "content_id": content_id,
            "corpus": encoded, "shards": shards,
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
        (temporary / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        os.rename(temporary, destination)
        for path in destination.rglob("*"):
            path.chmod(0o444 if path.is_file() else 0o555)
        destination.chmod(0o555)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_corpus(
    directory: str | Path, *, expected: Mapping[str, Any] | CorpusConfig | None = None,
    verify_hashes: bool = True, verify_source: bool = False,
) -> dict[str, Any]:
    """Load a canonical manifest and tensor-only shards on CPU."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_manifest_sha = manifest.pop("manifest_sha256", None)
    if recorded_manifest_sha != hashlib.sha256(_canonical_json(manifest)).hexdigest():
        raise ValueError("manifest SHA-256 mismatch")
    shard_meta = {record["file"]: record for record in manifest["shards"]}
    cache: dict[str, torch.Tensor] = {}
    def decode(item: Any) -> Any:
        if isinstance(item, dict) and set(item) == {"$tensor", "$index"}:
            relative = item["$tensor"]
            if relative not in cache:
                path = root / relative
                record = shard_meta.get(relative)
                if record is None or (verify_hashes and _sha256_file(path) != record["sha256"]):
                    raise ValueError(f"tensor shard integrity failure: {relative}")
                payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if not isinstance(payload, dict) or set(payload) != {"tensor"}:
                    raise ValueError(f"tensor shard schema failure: {relative}")
                cache[relative] = payload["tensor"]
            return cache[relative][int(item["$index"])]
        if isinstance(item, dict):
            return {key: decode(value) for key, value in item.items()}
        if isinstance(item, list):
            return [decode(value) for value in item]
        return item
    corpus = decode(manifest["corpus"])
    if manifest["content_id"] != corpus["integrity"]["corpus_sha256"] or root.name != manifest["content_id"]:
        raise ValueError("content-addressed directory name/manifest differs")
    validate_corpus(corpus, expected=expected, verify_hashes=verify_hashes, verify_source=verify_source)
    return corpus


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--max-episode", type=int, default=20_000)
    parser.add_argument("--curriculum", default=DEFAULT_CURRICULUM)
    parser.add_argument("--room", default="W7N3")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--holdout-seed-offset", type=int, default=10_000)
    parser.add_argument("--gamma", type=float, default=float(PPO_CFG["gamma"]))
    parser.add_argument("--ti-actor-steps", type=int, default=20_000)
    parser.add_argument("--ti-replay-capacity", type=int, default=20_000)
    parser.add_argument(
        "--ti-critic-replay-per-stratum", type=int,
        default=TI_CRITIC_RESERVOIR_CAPACITY,
    )
    parser.add_argument(
        "--temporal-replay-per-stratum", type=int,
        default=TEMPORAL_REPLAY_PER_STRATUM,
    )
    parser.add_argument("--ti-bot-dir", default=None)
    parser.add_argument("--node", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Intra-op parallelism makes environment workers contend; see
    # vec_env.configure_host_threads.
    configure_host_threads()
    args = parse_args(argv)
    config = CorpusConfig(
        num_envs=args.num_envs, steps=args.steps, max_episode=args.max_episode,
        curriculum=args.curriculum, room=args.room, seed=args.seed,
        holdout_seed_offset=args.holdout_seed_offset, gamma=args.gamma,
        ti_actor_steps=args.ti_actor_steps, ti_replay_capacity=args.ti_replay_capacity,
        ti_critic_replay_per_stratum=args.ti_critic_replay_per_stratum,
        temporal_replay_per_stratum=args.temporal_replay_per_stratum,
        ti_bot_dir=args.ti_bot_dir, node=args.node,
    )
    corpus = collect_corpus(config)
    destination = save_corpus(corpus, args.output)
    print(f"[corpus] saved immutable artifact {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
