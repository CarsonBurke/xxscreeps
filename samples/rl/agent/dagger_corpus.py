#!/usr/bin/env python3
"""Collect immutable scripted labels on states visited by a learned policy.

This is a standalone DAgger correction artifact.  It deliberately stores no
rewards or critic targets: every retained row is one exact scripted actor label
for the learner's pre-action observation.  Ordinary ``none`` actors are dropped;
the only retained waits are spawn actors for which spawning was a legal choice.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

if not __package__:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    __package__ = "samples.rl.agent"

from .artifacts import (
    load_full_state, source_signature, state_signature, validate_artifact,
)
from .constants import (
    ACTOR_FEAT, ACTOR_FEATURE_INDEX, AMOUNT_BINS, BODY_PART_COSTS,
    CONSTRUCTION_MASK_BYTES,
    CONSTRUCTION_TYPES, GLOBAL_FEAT, INTENT_SLOTS, INTENT_SPECS, INTENT_TYPES,
    MAX_ACTORS, MAX_BODY_PARTS, MAX_ROOM_ENERGY, MAX_ROOMS, MAX_TARGETS,
    N_AMOUNT, N_BODY_PART, N_CONSTRUCTION_TILE, N_CONSTRUCTION_TYPE, N_DIR,
    PATCHES_PER_ROOM, PATCH_FLAT, ROOM_SIZE, SCHEMA, SCHEMA_SHA256, TARGET_FEAT,
)
from .env_client import _COMMAND_MAGIC, _COMMAND_VERSION, _RESPONSE_VERSION
from .model import Actor, Critic, _LOCAL_TARGET_TYPES, _TARGET_RANGES
from .vec_env import VecScreepsEnv, _compact_entity_prefixes, configure_host_threads


DAGGER_CORPUS_KIND = "dagger_correction_corpus"
DAGGER_CORPUS_VERSION = 2
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "runs" / "dagger-corpora"
DEFAULT_PER_STRATUM = 64
SHARD_MAX_BYTES = 256 * 1024 * 1024
OUTPOST_CURRICULUM = "seed_outpost"
OUTPOST_LATE_WINDOW_STEPS = 1_000
OUTPOST_PHASES = ("early", "late")
OUTPOST_SCOPES = ("remote_harvest", "homebound_transfer", "local_other")
OUTPOST_COVERAGE_CONTRACT = "seed_outpost_teacher_scope_phase_v1"
OUTPOST_REQUIRED_SCOPES = ("remote_harvest", "homebound_transfer")
OUTPOST_MIN_RETAINED_OVERALL = 64
OUTPOST_MIN_RETAINED_LATE = 32
OUTPOST_MIN_DISTINCT_LATE_SEEDS = 3
TARGET_KIND_FEATURE_INDEX = 0
TARGET_ROOM_FEATURE_INDEX = 3
_POLICY_TEACHER_ABI_BRIDGE = {
    "schema_version": 4,
    "checkpoint_teacher_abi": 19,
    "current_teacher_abi": 20,
    "checkpoint_schema_sha256": (
        "a990f0295b95b3dea3eceef3a7504e39f5caf2860a17583d4e2de50c3d6aba34"
    ),
    "current_schema_sha256": (
        "63c26abcff5431b9caf47e6a50c32ec2be9eb44bc4dc25c579104687db6c8dbe"
    ),
}
OBS_KEYS = {
    "patches", "room_mask", "room_coords", "actors", "actor_mask",
    "actor_outcome", "targets", "target_mask", "intent_mask", "dir_mask",
    "target_select_mask", "amount_mask", "construction_mask", "globals",
}
ACTION_KEY_ORDER = (
    "types", "dirs", "targets", "amounts", "body_counts", "body_order",
    "construction_types", "construction_tiles",
)
ACTION_KEYS = set(ACTION_KEY_ORDER)
_ACTOR_CAPACITIES = {8, 16, 32, 64, MAX_ACTORS}
_TARGET_CAPACITIES = {16, 32, 64, MAX_TARGETS}
_ROOM_CAPACITIES = {1, 2, MAX_ROOMS}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_FORMAT_SPEC = {
    "version": DAGGER_CORPUS_VERSION,
    "directory": ["manifest.json", "shards/<sha256>.pt"],
    "row": [
        "kind", "stratum", "timestep", "env_index", "actor_index", "obs",
        "action",
    ],
    "row_kinds": ["exact_intent", "spawn_positive", "spawn_wait_legal"],
    "sampling": "algorithm_r_per_semantic_stratum",
    "outpost_coverage": {
        "contract": OUTPOST_COVERAGE_CONTRACT,
        "phases": list(OUTPOST_PHASES),
        "scopes": list(OUTPOST_SCOPES),
        "minimum_retained_overall": OUTPOST_MIN_RETAINED_OVERALL,
        "minimum_retained_late": OUTPOST_MIN_RETAINED_LATE,
        "minimum_distinct_late_seeds": OUTPOST_MIN_DISTINCT_LATE_SEEDS,
    },
    "teacher_alignment": "pre_action_observation",
    "policy_source_provenance": [
        "checkpoint_source_sha256", "source_sha256",
        "policy_source_mismatch", "policy_source_mismatch_allowed",
    ],
    "policy_teacher_provenance": [
        "checkpoint_schema_version", "checkpoint_schema_sha256",
        "checkpoint_contracts",
        "checkpoint_teacher_abi", "current_teacher_abi",
        "policy_teacher_abi_mismatch", "policy_teacher_abi_mismatch_allowed",
    ],
}
DAGGER_CORPUS_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(_FORMAT_SPEC, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
_COLLECTOR_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class DaggerConfig:
    """Declared simulator and bounded-sampling configuration."""

    num_envs: int
    steps: int
    seed: int
    curriculum: str = "empty"
    room: str = "W7N3"
    max_episode: int | None = None
    per_stratum: int = DEFAULT_PER_STRATUM
    reservoir_seed: int | None = None
    node: str | None = None
    device: str = "cpu"

    @property
    def curricula(self) -> tuple[str, ...]:
        return tuple(part.strip() for part in self.curriculum.split(",") if part.strip())

    @property
    def resolved_max_episode(self) -> int:
        return int(self.max_episode if self.max_episode is not None else self.steps + 5)

    @property
    def resolved_reservoir_seed(self) -> int:
        return int(
            self.reservoir_seed
            if self.reservoir_seed is not None
            else (int(self.seed) ^ 0x44414747)
        )

    def env_map(self) -> list[dict[str, Any]]:
        return [
            {
                "env_index": index,
                "seed": int(self.seed) + index,
                "curriculum": self.curricula[index % len(self.curricula)],
            }
            for index in range(self.num_envs)
        ]

    def validate(self) -> None:
        if self.num_envs <= 0 or self.steps <= 0:
            raise ValueError("num_envs and steps must be positive")
        if not self.curricula:
            raise ValueError("curriculum must contain at least one stage")
        if any(not _SAFE_LABEL_RE.fullmatch(stage) for stage in self.curricula):
            raise ValueError("curriculum labels must be portable identifier strings")
        if self.resolved_max_episode <= self.steps:
            raise ValueError("max_episode must exceed steps to prevent reset-alignment ambiguity")
        if self.per_stratum <= 0:
            raise ValueError("per_stratum must be positive")
        if not -(2**63) <= self.resolved_reservoir_seed < 2**63:
            raise ValueError("reservoir_seed must fit a signed 64-bit integer")


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


def _collector_source_sha256() -> str:
    """Fingerprint the collector source actually imported by this process."""
    return _COLLECTOR_SOURCE_SHA256


def _assert_collector_file_unchanged() -> None:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _COLLECTOR_SOURCE_SHA256:
        raise RuntimeError("DAgger collector source changed after module import")


def _hash_value(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(_canonical_json(list(tensor.shape)))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"dict\0")
        for key in sorted(value):
            _hash_value(digest, str(key))
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"list\0" + len(value).to_bytes(8, "little"))
        for item in value:
            _hash_value(digest, item)
    elif isinstance(value, np.generic):
        _hash_value(digest, value.item())
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("artifact metadata cannot contain non-finite floats")
        digest.update(b"float\0" + value.hex().encode())
    else:
        digest.update(_canonical_json(value))


def content_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def checkpoint_file_sha256(path: str | Path) -> str:
    return _sha256_file(Path(path))


def _weights_only_checkpoint_load(payload: bytes) -> Any:
    """Load legacy NumPy RNG state without permitting arbitrary pickle globals."""
    numpy_safe_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.UInt32DType,
    ]
    with torch.serialization.safe_globals(numpy_safe_globals):
        return torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=True,
        )


def _load_joint_actor(
    checkpoint_path: str | Path,
    *,
    base_corpus_id: str,
    device: torch.device,
    allow_policy_source_mismatch: bool = False,
    allow_policy_teacher_abi_mismatch: bool = False,
) -> tuple[Actor, dict[str, Any], str]:
    """Load one complete joint actor, qualified or not, with an audited escape hatch."""
    path = Path(checkpoint_path)
    payload = path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
    checkpoint = _weights_only_checkpoint_load(payload)
    if not isinstance(checkpoint, dict):
        raise ValueError("joint checkpoint root must be a dictionary")
    actor, critic = Actor(), Critic()
    meta = checkpoint.get("meta")
    checkpoint_contracts = meta.get("contracts") if isinstance(meta, dict) else None
    current_contracts = SCHEMA["artifact"]
    teacher_mismatch = (
        isinstance(checkpoint_contracts, dict)
        and checkpoint_contracts.get("teacherAbi") != current_contracts["teacherAbi"]
    )
    if not teacher_mismatch:
        meta = validate_artifact(
            checkpoint, actor, critic, kinds=("joint_pretrain",),
            allow_source_mismatch=allow_policy_source_mismatch,
        )
    else:
        if not allow_policy_teacher_abi_mismatch:
            raise ValueError(
                "checkpoint teacher ABI differs; explicit collector authorization required"
            )
        if meta.get("kind") != "joint_pretrain":
            raise ValueError("teacher-ABI compatibility is limited to joint_pretrain policies")
        if meta.get("schema_version") != SCHEMA["version"]:
            raise ValueError("checkpoint schema version differs from current schema")
        if (
            not isinstance(meta.get("schema_sha256"), str)
            or not _SHA256_RE.fullmatch(meta["schema_sha256"])
        ):
            raise ValueError("checkpoint semantic schema fingerprint is invalid")
        if meta["schema_sha256"] == SCHEMA_SHA256:
            raise ValueError(
                "checkpoint schema fingerprint is inconsistent with its older teacher ABI"
            )
        if set(checkpoint_contracts) != set(current_contracts):
            raise ValueError("checkpoint artifact contract keys differ from current contracts")
        non_teacher_mismatches = {
            key: (checkpoint_contracts[key], current_contracts[key])
            for key in current_contracts
            if key != "teacherAbi" and checkpoint_contracts[key] != current_contracts[key]
        }
        if non_teacher_mismatches:
            raise ValueError(
                "checkpoint non-teacher artifact ABIs differ: "
                f"{non_teacher_mismatches}"
            )
        old_teacher_abi = checkpoint_contracts["teacherAbi"]
        current_teacher_abi = current_contracts["teacherAbi"]
        bridge = _POLICY_TEACHER_ABI_BRIDGE
        if not (
            meta["schema_version"] == bridge["schema_version"]
            and SCHEMA["version"] == bridge["schema_version"]
            and type(old_teacher_abi) is int
            and old_teacher_abi == bridge["checkpoint_teacher_abi"]
            and type(current_teacher_abi) is int
            and current_teacher_abi == bridge["current_teacher_abi"]
            and meta["schema_sha256"] == bridge["checkpoint_schema_sha256"]
            and SCHEMA_SHA256 == bridge["current_schema_sha256"]
        ):
            raise ValueError(
                "checkpoint does not match the audited teacher ABI 19-to-20 bridge"
            )
        if meta.get("source_sha256") != source_signature() and not allow_policy_source_mismatch:
            raise ValueError("checkpoint executable RL source fingerprint differs from current source")
        if meta.get("actor_state_signature") != state_signature(actor):
            raise ValueError("checkpoint actor parameter ABI differs from current model")
        if meta.get("critic_state_signature") != state_signature(critic):
            raise ValueError("checkpoint critic parameter ABI differs from current model")
    if bool(meta.get("partial", True)):
        raise ValueError("DAgger requires a complete joint checkpoint, not a partial save")
    global_epoch = int(meta.get("global_epoch", -1))
    global_epochs = int(meta.get("global_epochs", -2))
    if global_epoch <= 0 or global_epoch != global_epochs:
        raise ValueError("joint checkpoint did not complete its declared global epochs")
    if str(meta.get("corpus_sha256")) != base_corpus_id:
        raise ValueError("joint checkpoint base corpus differs from the declared corpus ID")
    if not isinstance(checkpoint.get("actor"), dict) or not isinstance(
        checkpoint.get("critic"), dict
    ):
        raise ValueError("joint checkpoint is missing actor or critic state")
    load_full_state(actor, checkpoint["actor"], name="actor")
    load_full_state(critic, checkpoint["critic"], name="critic")
    actor.to(device).eval()
    return actor, meta, checkpoint_sha256


def _action_dict(output: Any) -> dict[str, torch.Tensor]:
    return {key: getattr(output, key) for key in ACTION_KEY_ORDER}


def _population_bucket(population: int) -> str:
    if population <= 3:
        return "p0_3"
    if population <= 7:
        return "p4_7"
    if population <= 15:
        return "p8_15"
    if population <= 31:
        return "p16_31"
    return "p32p"


def _budget_bucket(budget: int) -> str:
    if budget <= 300:
        return "e0_300"
    if budget < 550:
        return "e301_549"
    if budget < 650:
        return "e550_649"
    if budget < 800:
        return "e650_799"
    return "e800p"


def _actor_kind(obs: dict[str, torch.Tensor], env_index: int, actor_index: int) -> str:
    actor = obs["actors"][env_index, actor_index]
    if float(actor[ACTOR_FEATURE_INDEX["isRoom"]]) > 0.5:
        return "room"
    if float(actor[ACTOR_FEATURE_INDEX["isSpawn"]]) > 0.5:
        return "spawn"
    if float(actor[ACTOR_FEATURE_INDEX["isNonCreep"]]) > 0.5:
        return "structure"
    return "creep"


def _late_window_size(steps: int) -> int:
    """Match the training/qualification late-window contract exactly."""
    return min(OUTPOST_LATE_WINDOW_STEPS, max(1, int(steps) // 5))


def _decoded_room_index(value: torch.Tensor) -> int:
    return int(round(float(value) * max(1, MAX_ROOMS - 1)))


def _outpost_phase_scope(
    obs: dict[str, torch.Tensor],
    action: dict[str, torch.Tensor],
    *,
    curriculum: str,
    timestep: int,
    steps: int,
    env_index: int,
    actor_index: int,
) -> tuple[str, str] | None:
    """Classify one seed-outpost teacher label by deployment phase and scope."""
    if curriculum != OUTPOST_CURRICULUM:
        return None
    phase = (
        "late"
        if int(timestep) >= int(steps) - _late_window_size(steps)
        else "early"
    )
    intent_index = int(action["types"][env_index, actor_index, 0])
    if not 0 <= intent_index < len(INTENT_TYPES):
        raise ValueError("teacher intent index is out of bounds")
    intent = INTENT_TYPES[intent_index]
    scope = "local_other"
    if intent in {"harvest", "transfer"}:
        target_index = int(action["targets"][env_index, actor_index, 0])
        target_cap = int(obs["targets"].shape[1])
        if (
            not 0 <= target_index < target_cap
            or not bool(obs["target_mask"][env_index, target_index])
        ):
            raise ValueError("teacher target index does not name a live compact target")
        target_room = _decoded_room_index(
            obs["targets"][env_index, target_index, TARGET_ROOM_FEATURE_INDEX]
        )
        if intent == "harvest" and target_room > 0:
            scope = "remote_harvest"
        elif intent == "transfer":
            actor_room = _decoded_room_index(
                obs["actors"][
                    env_index, actor_index, ACTOR_FEATURE_INDEX["roomIndex"]
                ]
            )
            if actor_room > 0 and target_room == 0:
                scope = "homebound_transfer"
    return phase, scope


def _empty_outpost_counts() -> dict[str, dict[str, int]]:
    return {
        scope: {phase: 0 for phase in OUTPOST_PHASES}
        for scope in OUTPOST_SCOPES
    }


def _empty_outpost_seed_sets() -> dict[str, dict[str, set[int]]]:
    return {
        scope: {phase: set() for phase in OUTPOST_PHASES}
        for scope in OUTPOST_SCOPES
    }


def _serialized_seed_sets(
    values: Mapping[str, Mapping[str, set[int]]],
) -> dict[str, dict[str, list[int]]]:
    return {
        scope: {
            phase: sorted(int(seed) for seed in values[scope][phase])
            for phase in OUTPOST_PHASES
        }
        for scope in OUTPOST_SCOPES
    }


def _row_semantics(
    obs: dict[str, torch.Tensor],
    action: dict[str, torch.Tensor],
    *,
    curriculum: str,
    timestep: int,
    steps: int,
    env_index: int,
    actor_index: int,
) -> tuple[str, str] | None:
    """Return ``(row_kind, semantic_stratum)`` for an eligible actor."""
    if not bool(obs["actor_mask"][env_index, actor_index]):
        return None
    intent_index = int(action["types"][env_index, actor_index, 0])
    if not 0 <= intent_index < len(INTENT_TYPES):
        raise ValueError("teacher intent index is out of bounds")
    intent = INTENT_TYPES[intent_index]
    spawn_index = INTENT_TYPES.index("spawnCreep")
    actor_kind = _actor_kind(obs, env_index, actor_index)
    budget = int(round(
        float(obs["actors"][env_index, actor_index, ACTOR_FEATURE_INDEX["roomEnergyAvailable"]])
        * MAX_ROOM_ENERGY
    ))
    if intent == "none":
        spawn_legal = (
            actor_kind == "spawn"
            and bool(obs["intent_mask"][env_index, actor_index, 0, spawn_index])
            and budget + 1e-4 >= min(BODY_PART_COSTS)
        )
        if not spawn_legal:
            return None
        row_kind = "spawn_wait_legal"
        argument = f"budget={_budget_bucket(budget)}"
    elif intent == "spawnCreep":
        if actor_kind != "spawn":
            raise ValueError("teacher emitted spawnCreep for a non-spawn actor")
        counts = action["body_counts"][env_index, actor_index, 0].long()
        signature = "_".join(str(int(value)) for value in counts)
        row_kind = "spawn_positive"
        argument = (
            f"budget={_budget_bucket(budget)}|length={int(counts.sum())}|body={signature}"
        )
    else:
        row_kind = "exact_intent"
        factors = INTENT_SPECS[intent]["factors"]
        arguments: list[str] = []
        if "direction" in factors:
            arguments.append(f"direction={int(action['dirs'][env_index, actor_index, 0])}")
        if "target" in factors:
            target_index = int(action["targets"][env_index, actor_index, 0])
            target_cap = int(obs["targets"].shape[1])
            if (
                not 0 <= target_index < target_cap
                or not bool(obs["target_mask"][env_index, target_index])
            ):
                raise ValueError("teacher target index does not name a live compact target")
            target_kind = int(round(
                float(obs["targets"][
                    env_index, target_index, TARGET_KIND_FEATURE_INDEX
                ]) * 6
            ))
            arguments.append(f"targetkind={target_kind}")
        if "amount" in factors:
            arguments.append(f"amount={int(action['amounts'][env_index, actor_index, 0])}")
        if "constructionType" in factors:
            construction = int(
                action["construction_types"][env_index, actor_index, 0]
            )
            if not 0 <= construction < len(CONSTRUCTION_TYPES):
                raise ValueError("teacher construction type is out of bounds")
            arguments.append(f"structure={CONSTRUCTION_TYPES[construction]}")
        argument = "|".join(arguments) if arguments else "argument=none"
    globals_row = obs["globals"][env_index]
    rcl = int(round(float(globals_row[0]) * 8))
    population = int(round(float(globals_row[3]) * 50))
    rooms = int(round(float(globals_row[6]) * 16))
    stratum = (
        f"{row_kind}|intent={intent}|curriculum={curriculum}|rcl={rcl}|"
        f"pop={_population_bucket(population)}|rooms={rooms}|actor={actor_kind}|{argument}"
    )
    outpost_semantics = _outpost_phase_scope(
        obs, action, curriculum=curriculum, timestep=timestep, steps=steps,
        env_index=env_index, actor_index=actor_index,
    )
    if outpost_semantics is not None:
        phase, scope = outpost_semantics
        stratum += f"|phase={phase}|scope={scope}"
    return row_kind, stratum


def _clone_row(
    obs: dict[str, torch.Tensor],
    action: dict[str, torch.Tensor],
    *,
    row_kind: str,
    stratum: str,
    timestep: int,
    env_index: int,
    actor_index: int,
) -> dict[str, Any]:
    compact = _compact_entity_prefixes({
        key: value[env_index : env_index + 1] for key, value in obs.items()
    })
    actor_cap = int(compact["actors"].shape[1])
    return {
        "kind": row_kind,
        "stratum": stratum,
        "timestep": int(timestep),
        "env_index": int(env_index),
        "actor_index": int(actor_index),
        "obs": {key: value.detach().cpu().clone() for key, value in compact.items()},
        "action": {
            key: action[key][env_index : env_index + 1, :actor_cap]
            .detach().cpu().long().clone()
            for key in ACTION_KEY_ORDER
        },
    }


def _reservoir_offer(
    rows: list[dict[str, Any]],
    retained_indices: dict[str, list[int]],
    seen: dict[str, int],
    *,
    stratum: str,
    capacity: int,
    rng: np.random.Generator,
    make_row,
) -> bool:
    count = seen.get(stratum, 0) + 1
    seen[stratum] = count
    indices = retained_indices.setdefault(stratum, [])
    if len(indices) < capacity:
        rows.append(make_row())
        indices.append(len(rows) - 1)
        return True
    replacement = int(rng.integers(0, count))
    if replacement >= capacity:
        return False
    rows[indices[replacement]] = make_row()
    return True


@torch.inference_mode()
def collect_rows(
    actor: Any,
    envs: Any,
    config: DaggerConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Collect bounded exact labels using the ``step_labeled`` pre-state ABI."""
    config.validate()
    rng = np.random.default_rng(config.resolved_reservoir_seed)
    rows: list[dict[str, Any]] = []
    retained_indices: dict[str, list[int]] = {}
    seen: dict[str, int] = {}
    row_kind_seen: dict[str, int] = {
        "exact_intent": 0, "spawn_positive": 0, "spawn_wait_legal": 0,
    }
    outpost_seen = _empty_outpost_counts()
    outpost_seen_seeds = _empty_outpost_seed_sets()
    learner_invalid = 0
    learner_issued = 0
    obs_device = envs.reset()
    for timestep in range(config.steps):
        if envs.host_obs is None:
            raise RuntimeError("step_labeled collector requires VecScreepsEnv.host_obs")
        pre = envs.host_obs
        learner_output = actor(obs_device, deterministic=True)
        obs_device, _rewards, dones, infos, teacher = envs.step_labeled(
            _action_dict(learner_output)
        )
        if bool(dones.any()):
            raise RuntimeError(
                "DAgger episode ended during collection; increase max_episode to preserve alignment"
            )
        for env_index, info in enumerate(infos):
            info = info or {}
            if info.get("recovered") or info.get("invalid_demo"):
                raise RuntimeError("DAgger environment recovered; labels are not admissible")
            learner_invalid += int(info.get("intentInvalid") or 0)
            learner_issued += int(info.get("intentIssued") or 0)
            curriculum = config.curricula[env_index % len(config.curricula)]
            live = torch.nonzero(pre["actor_mask"][env_index] > 0, as_tuple=False).flatten()
            for actor_tensor in live:
                actor_index = int(actor_tensor)
                semantics = _row_semantics(
                    pre, teacher, curriculum=curriculum,
                    timestep=timestep, steps=config.steps,
                    env_index=env_index, actor_index=actor_index,
                )
                if semantics is None:
                    continue
                row_kind, stratum = semantics
                row_kind_seen[row_kind] += 1
                outpost_semantics = _outpost_phase_scope(
                    pre, teacher, curriculum=curriculum,
                    timestep=timestep, steps=config.steps,
                    env_index=env_index, actor_index=actor_index,
                )
                if outpost_semantics is not None:
                    phase, scope = outpost_semantics
                    outpost_seen[scope][phase] += 1
                    outpost_seen_seeds[scope][phase].add(
                        int(config.seed) + env_index
                    )
                _reservoir_offer(
                    rows, retained_indices, seen, stratum=stratum,
                    capacity=config.per_stratum, rng=rng,
                    make_row=lambda rk=row_kind, st=stratum, ti=timestep,
                    ei=env_index, ai=actor_index: _clone_row(
                        pre, teacher, row_kind=rk, stratum=st, timestep=ti,
                        env_index=ei, actor_index=ai,
                    ),
                )
        if (timestep + 1) % 500 == 0 or timestep + 1 == config.steps:
            print(
                f"[dagger] {timestep + 1}/{config.steps} rows={len(rows)} "
                f"strata={len(seen)}",
                flush=True,
            )
    if not rows:
        raise RuntimeError("DAgger collection retained no exact actor labels")
    rows.sort(key=lambda row: (
        row["stratum"], row["timestep"], row["env_index"], row["actor_index"],
    ))
    retained = {
        stratum: sum(row["stratum"] == stratum for row in rows)
        for stratum in sorted(seen)
    }
    outpost_retained = _empty_outpost_counts()
    outpost_retained_seeds = _empty_outpost_seed_sets()
    env_map = {entry["env_index"]: entry for entry in config.env_map()}
    for row in rows:
        env = env_map[int(row["env_index"])]
        outpost_semantics = _outpost_phase_scope(
            row["obs"], row["action"], curriculum=str(env["curriculum"]),
            timestep=int(row["timestep"]), steps=config.steps,
            env_index=0, actor_index=int(row["actor_index"]),
        )
        if outpost_semantics is None:
            continue
        phase, scope = outpost_semantics
        outpost_retained[scope][phase] += 1
        outpost_retained_seeds[scope][phase].add(int(env["seed"]))
    sampling = {
        "algorithm": "algorithm_r_per_semantic_stratum",
        "capacity_per_stratum": int(config.per_stratum),
        "rng": "numpy.default_rng/PCG64",
        "seed": int(config.resolved_reservoir_seed),
        "seen_by_stratum": dict(sorted(seen.items())),
        "retained_by_stratum": retained,
        "seen_by_row_kind": row_kind_seen,
        "outpost_coverage": {
            "contract": OUTPOST_COVERAGE_CONTRACT,
            "late_window_size": _late_window_size(config.steps),
            "readiness_targets": {
                "minimum_retained_overall": OUTPOST_MIN_RETAINED_OVERALL,
                "minimum_retained_late": OUTPOST_MIN_RETAINED_LATE,
                "minimum_distinct_late_seeds": OUTPOST_MIN_DISTINCT_LATE_SEEDS,
            },
            "seen_by_scope_phase": outpost_seen,
            "retained_by_scope_phase": outpost_retained,
            "seen_seeds_by_scope_phase": _serialized_seed_sets(
                outpost_seen_seeds
            ),
            "retained_seeds_by_scope_phase": _serialized_seed_sets(
                outpost_retained_seeds
            ),
        },
        "retained": len(rows),
    }
    collection = {
        "transitions": int(config.steps * config.num_envs),
        "learner_intent_issued": int(learner_issued),
        "learner_intent_invalid": int(learner_invalid),
    }
    return rows, sampling, collection


def _runtime_provenance(config: DaggerConfig) -> dict[str, Any]:
    node = config.node or os.environ.get("RL_NODE", "node")
    try:
        node_path = shutil.which(node) or str(Path(node).resolve())
        node_version = subprocess.check_output(
            [node, "--version"], text=True, stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        node_path, node_version = node, "unresolved"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "node_executable": node_path,
        "node_version": node_version,
        "observation_format": os.environ.get("RL_OBS_FMT", "bin"),
        "command_format": os.environ.get("RL_CMD_FMT", "bin"),
        "observation_protocol": "XRL1",
        "response_protocol_version": int(_RESPONSE_VERSION),
        "command_protocol": _COMMAND_MAGIC.decode("ascii"),
        "command_protocol_version": int(_COMMAND_VERSION),
        "step_api": "VecScreepsEnv.step_labeled/v1",
        "teacher_alignment": "teacher_actions_label_pre_action_host_obs",
        "learner_action": "deterministic_actor_argmax",
        "lean_meta": False,
    }


def assemble_dagger_corpus(
    meta: Mapping[str, Any],
    rows: list[dict[str, Any]],
    sampling: Mapping[str, Any],
    collection: Mapping[str, Any],
) -> dict[str, Any]:
    corpus = {
        "kind": DAGGER_CORPUS_KIND,
        "corpus_schema_version": DAGGER_CORPUS_VERSION,
        "corpus_schema_sha256": DAGGER_CORPUS_SCHEMA_SHA256,
        "meta": dict(meta),
        "data": {
            "rows": rows,
            "sampling": dict(sampling),
            "collection": dict(collection),
        },
    }
    corpus["integrity"] = {
        "algorithm": "sha256-semantic-v1",
        "corpus_sha256": content_sha256(corpus),
    }
    validate_dagger_corpus(corpus, verify_source=False)
    return corpus


def collect_dagger_corpus(
    config: DaggerConfig,
    *,
    checkpoint_path: str | Path,
    base_corpus_id: str,
    allow_policy_source_mismatch: bool = False,
    allow_policy_teacher_abi_mismatch: bool = False,
) -> dict[str, Any]:
    """Load the declared policy and collect one immutable correction corpus."""
    config.validate()
    if not _SHA256_RE.fullmatch(base_corpus_id):
        raise ValueError("base_corpus_id must be a lowercase SHA-256")
    _assert_collector_file_unchanged()
    collection_source_sha256 = source_signature()
    collector_source_sha256 = _collector_source_sha256()
    device = torch.device(config.device)
    actor, checkpoint_meta, checkpoint_sha256 = _load_joint_actor(
        checkpoint_path, base_corpus_id=base_corpus_id, device=device,
        allow_policy_source_mismatch=allow_policy_source_mismatch,
        allow_policy_teacher_abi_mismatch=allow_policy_teacher_abi_mismatch,
    )
    if source_signature() != collection_source_sha256:
        raise RuntimeError("executable RL source changed while loading the DAgger policy")
    envs = VecScreepsEnv(
        config.num_envs, node=config.node, room=config.room,
        max_episode=config.resolved_max_episode, device=device,
        curriculum=config.curriculum, lean_meta=False, seed=config.seed,
    )
    try:
        rows, sampling, collection = collect_rows(actor, envs, config)
    finally:
        envs.close()
    _assert_collector_file_unchanged()
    if source_signature() != collection_source_sha256:
        raise RuntimeError("executable RL source changed during DAgger collection")
    checkpoint_source_sha256 = str(checkpoint_meta["source_sha256"])
    policy_source_mismatch = checkpoint_source_sha256 != collection_source_sha256
    checkpoint_contracts = dict(checkpoint_meta["contracts"])
    checkpoint_teacher_abi = int(checkpoint_contracts["teacherAbi"])
    current_teacher_abi = int(SCHEMA["artifact"]["teacherAbi"])
    policy_teacher_abi_mismatch = checkpoint_teacher_abi != current_teacher_abi
    meta = {
        **asdict(config),
        "max_episode": config.resolved_max_episode,
        "reservoir_seed": config.resolved_reservoir_seed,
        "env_map": config.env_map(),
        "base_corpus_sha256": base_corpus_id,
        "checkpoint_file_sha256": checkpoint_sha256,
        "checkpoint_kind": checkpoint_meta["kind"],
        "checkpoint_qualified": bool(checkpoint_meta.get("qualified", False)),
        "checkpoint_partial": bool(checkpoint_meta["partial"]),
        "checkpoint_global_epoch": int(checkpoint_meta["global_epoch"]),
        "checkpoint_global_epochs": int(checkpoint_meta["global_epochs"]),
        "checkpoint_schema_version": int(checkpoint_meta["schema_version"]),
        "checkpoint_schema_sha256": str(checkpoint_meta["schema_sha256"]),
        "checkpoint_contracts": checkpoint_contracts,
        "checkpoint_source_sha256": checkpoint_source_sha256,
        "source_sha256": collection_source_sha256,
        "policy_source_mismatch": policy_source_mismatch,
        "policy_source_mismatch_allowed": bool(allow_policy_source_mismatch),
        "checkpoint_teacher_abi": checkpoint_teacher_abi,
        "current_teacher_abi": current_teacher_abi,
        "policy_teacher_abi_mismatch": policy_teacher_abi_mismatch,
        "policy_teacher_abi_mismatch_allowed": bool(
            allow_policy_teacher_abi_mismatch
        ),
        "collector_source_sha256": collector_source_sha256,
        "schema_sha256": SCHEMA_SHA256,
        "environment_schema_version": SCHEMA["version"],
        "contracts": dict(SCHEMA["artifact"]),
        "runtime": _runtime_provenance(config),
    }
    return assemble_dagger_corpus(meta, rows, sampling, collection)


def _validate_tensor_dict(value: Any, keys: set[str], location: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{location} keys differ from schema")
    if any(
        not torch.is_tensor(tensor) or tensor.device.type != "cpu"
        for tensor in value.values()
    ):
        raise ValueError(f"{location} must contain only CPU tensors")


def _validate_obs(obs: Any, location: str) -> tuple[int, int, int]:
    _validate_tensor_dict(obs, OBS_KEYS, location)
    if any(tensor.shape[0] != 1 for tensor in obs.values()):
        raise ValueError(f"{location} must have singleton batch dimensions")
    room = int(obs["patches"].shape[1])
    actor = int(obs["actors"].shape[1])
    target = int(obs["targets"].shape[1])
    if room not in _ROOM_CAPACITIES or actor not in _ACTOR_CAPACITIES or target not in _TARGET_CAPACITIES:
        raise ValueError(f"{location} has unsupported compact capacity {(room, actor, target)}")
    shapes = {
        "patches": (1, room, PATCHES_PER_ROOM, PATCH_FLAT),
        "room_mask": (1, room), "room_coords": (1, room, 2),
        "actors": (1, actor, ACTOR_FEAT), "actor_mask": (1, actor),
        "actor_outcome": (1, actor), "targets": (1, target, TARGET_FEAT),
        "target_mask": (1, target),
        "intent_mask": (1, actor, INTENT_SLOTS, len(INTENT_TYPES)),
        "dir_mask": (1, actor, INTENT_SLOTS, N_DIR),
        "target_select_mask": (1, len(INTENT_TYPES), target),
        "amount_mask": (1, actor, INTENT_SLOTS, len(INTENT_TYPES), N_AMOUNT),
        "construction_mask": (1, room, N_CONSTRUCTION_TYPE, CONSTRUCTION_MASK_BYTES),
        "globals": (1, GLOBAL_FEAT),
    }
    dtypes = {
        "patches": torch.uint8, "room_mask": torch.uint8,
        "room_coords": torch.float32, "actors": torch.float32,
        "actor_mask": torch.uint8, "actor_outcome": torch.uint8,
        "targets": torch.float32, "target_mask": torch.uint8,
        "intent_mask": torch.uint8, "dir_mask": torch.uint8,
        "target_select_mask": torch.uint8, "amount_mask": torch.uint8,
        "construction_mask": torch.uint8, "globals": torch.float32,
    }
    for key in OBS_KEYS:
        tensor = obs[key]
        if tensor.dtype != dtypes[key] or tuple(tensor.shape) != shapes[key]:
            raise ValueError(f"{location}.{key} dtype/shape differs from schema")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{location}.{key} contains non-finite data")
    binary_masks = (
        "room_mask", "actor_mask", "target_mask", "intent_mask", "dir_mask",
        "target_select_mask", "amount_mask",
    )
    if any(bool(obs[key].gt(1).any()) for key in binary_masks):
        raise ValueError(f"{location} contains a non-binary mask")
    for mask, name in (
        (obs["actor_mask"][0], "actor"),
        (obs["target_mask"][0], "target"),
        (obs["room_mask"][0], "room"),
    ):
        live = torch.nonzero(mask > 0, as_tuple=False).flatten()
        expected = torch.arange(live.numel())
        if not bool(live.eq(expected).all()):
            raise ValueError(f"{location} {name} mask is not a compact prefix")
    return room, actor, target


def _validate_action(action: Any, actor_cap: int, location: str) -> None:
    _validate_tensor_dict(action, ACTION_KEYS, location)
    for key, tensor in action.items():
        expected = (
            (1, actor_cap, INTENT_SLOTS, N_BODY_PART)
            if key in {"body_counts", "body_order"}
            else (1, actor_cap, INTENT_SLOTS)
        )
        if tensor.dtype != torch.long or tuple(tensor.shape) != expected:
            raise ValueError(f"{location}.{key} dtype/shape differs from schema")
    bounds = {
        "types": len(INTENT_TYPES), "dirs": N_DIR, "targets": MAX_TARGETS,
        "amounts": N_AMOUNT, "construction_types": N_CONSTRUCTION_TYPE,
        "construction_tiles": N_CONSTRUCTION_TILE,
    }
    for key, upper in bounds.items():
        if bool((action[key] < 0).any()) or bool((action[key] >= upper).any()):
            raise ValueError(f"{location}.{key} is out of bounds")
    counts, order = action["body_counts"], action["body_order"]
    if bool((counts < 0).any()) or bool((counts > MAX_BODY_PARTS).any()):
        raise ValueError(f"{location}.body_counts is out of bounds")
    if bool(counts.sum(dim=-1).gt(MAX_BODY_PARTS).any()):
        raise ValueError(f"{location}.body_counts exceeds maximum body length")
    expected_order = torch.arange(N_BODY_PART).view(1, 1, 1, -1)
    if not bool(torch.sort(order, dim=-1).values.eq(expected_order).all()):
        raise ValueError(f"{location}.body_order is not a permutation")
    nonzero = counts > 0
    ordered_nonzero = nonzero.gather(-1, order)
    active = torch.arange(N_BODY_PART).view(1, 1, 1, -1) < nonzero.sum(
        dim=-1, keepdim=True,
    )
    if not bool(ordered_nonzero.eq(active).all()):
        raise ValueError(f"{location}.body_order does not place nonzero types first")
    descending = order[..., 1:] < order[..., :-1]
    if bool((descending & ~active[..., 1:] & ~active[..., :-1]).any()):
        raise ValueError(f"{location}.body_order zero-count suffix is not canonical")


def _validate_selected_action_legality(
    obs: dict[str, torch.Tensor],
    action: dict[str, torch.Tensor],
    *,
    actor_index: int,
    location: str,
) -> None:
    intent = int(action["types"][0, actor_index, 0])
    if not bool(obs["intent_mask"][0, actor_index, 0, intent]):
        raise ValueError(f"{location} teacher intent is illegal in its pre-state")
    factors = INTENT_SPECS[INTENT_TYPES[intent]]["factors"]
    if "direction" in factors:
        direction = int(action["dirs"][0, actor_index, 0])
        if not bool(obs["dir_mask"][0, actor_index, 0, direction]):
            raise ValueError(f"{location} teacher direction is illegal in its pre-state")
    if "target" in factors:
        target = int(action["targets"][0, actor_index, 0])
        if (
            target >= obs["targets"].shape[1]
            or not bool(obs["target_mask"][0, target])
            or not bool(obs["target_select_mask"][0, intent, target])
        ):
            raise ValueError(f"{location} teacher target is illegal in its pre-state")
        actor_features = obs["actors"][0, actor_index]
        target_features = obs["targets"][0, target]
        room_scale = max(1, MAX_ROOMS - 1)
        room_cap = int(obs["room_mask"].shape[1])
        actor_room = min(max(int(round(float(actor_features[3]) * room_scale)), 0), room_cap - 1)
        target_room = min(max(int(round(float(target_features[3]) * room_scale)), 0), room_cap - 1)
        same_room = actor_room == target_room
        actor_is_creep = float(actor_features[ACTOR_FEATURE_INDEX["isNonCreep"]]) < 0.5
        target_kind = int(round(float(target_features[0]) * 6))
        same_position = bool(
            (actor_features[1:3] - target_features[1:3]).abs().amax() < 1e-6
        )
        compatible = actor_is_creep or same_room
        if actor_is_creep and float(actor_features[ACTOR_FEATURE_INDEX["activeMove"]]) <= 0:
            distance = float(
                (actor_features[1:3] - target_features[1:3]).abs().amax()
            ) * (ROOM_SIZE - 1)
            compatible = same_room and distance <= float(_TARGET_RANGES[intent]) + 1e-4
        if INTENT_TYPES[intent] == "transfer" and actor_is_creep and target_kind == 4:
            compatible = compatible and not (same_room and same_position)
        actor_is_tower = (
            not actor_is_creep
            and float(actor_features[ACTOR_FEATURE_INDEX["isTower"]]) > 0.5
        )
        if INTENT_TYPES[intent] == "attack" and actor_is_tower:
            compatible = compatible and target_kind == 4
        if bool(_LOCAL_TARGET_TYPES[intent]):
            compatible = compatible and same_room
        if not compatible:
            raise ValueError(
                f"{location} teacher target fails actor-local policy compatibility"
            )
    if "amount" in factors:
        amount = int(action["amounts"][0, actor_index, 0])
        if not bool(obs["amount_mask"][0, actor_index, 0, intent, amount]):
            raise ValueError(f"{location} teacher amount is illegal in its pre-state")
        intent_name = INTENT_TYPES[intent]
        if intent_name in {"transfer", "withdraw", "drop"}:
            actor_features = obs["actors"][0, actor_index]
            target = int(action["targets"][0, actor_index, 0])
            target_features = obs["targets"][0, target]
            target_kind = int(round(float(target_features[0]) * 6))
            energy_scale = 1_000_000.0 if target_kind == 2 else 2_000.0
            target_energy = float(target_features[13]) * energy_scale
            target_capacity = float(target_features[14]) * energy_scale
            actor_energy = (
                float(actor_features[ACTOR_FEATURE_INDEX["storedEnergy"]])
                * MAX_ROOM_ENERGY
            )
            actor_capacity = (
                float(actor_features[ACTOR_FEATURE_INDEX["storeCapacity"]])
                * MAX_ROOM_ENERGY
            )
            actor_free = (
                float(actor_features[ACTOR_FEATURE_INDEX["storeFree"]])
                * actor_capacity
            )
            if intent_name == "transfer":
                resource_limit = min(actor_energy, max(0.0, target_capacity - target_energy))
            elif intent_name == "withdraw":
                resource_limit = min(actor_free, target_energy)
            else:
                resource_limit = actor_energy
            value = float(AMOUNT_BINS[amount])
            if value != 0 and not value < resource_limit - 1e-4:
                raise ValueError(
                    f"{location} teacher amount exceeds policy resource limit"
                )
    if "constructionType" in factors or "constructionTile" in factors:
        construction_type = int(action["construction_types"][0, actor_index, 0])
        tile = int(action["construction_tiles"][0, actor_index, 0])
        room_scale = max(1, MAX_ROOMS - 1)
        room = int(round(
            float(obs["actors"][0, actor_index, ACTOR_FEATURE_INDEX["roomIndex"]])
            * room_scale
        ))
        room = min(max(room, 0), obs["room_mask"].shape[1] - 1)
        if not bool(obs["room_mask"][0, room]):
            raise ValueError(f"{location} teacher construction room is not live")
        packed = int(obs["construction_mask"][0, room, construction_type, tile // 8])
        if not bool((packed >> (tile % 8)) & 1):
            raise ValueError(f"{location} teacher construction tile is illegal in its pre-state")


def _validate_outpost_count_grid(
    value: Any, *, location: str,
) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict) or set(value) != set(OUTPOST_SCOPES):
        raise ValueError(f"{location} scope keys differ from schema")
    result = _empty_outpost_counts()
    for scope in OUTPOST_SCOPES:
        phases = value[scope]
        if not isinstance(phases, dict) or set(phases) != set(OUTPOST_PHASES):
            raise ValueError(f"{location}.{scope} phase keys differ from schema")
        for phase in OUTPOST_PHASES:
            count = phases[phase]
            if type(count) is not int or count < 0:
                raise ValueError(f"{location}.{scope}.{phase} is not a count")
            result[scope][phase] = count
    return result


def _validate_outpost_seed_grid(
    value: Any,
    *,
    location: str,
    allowed_seeds: set[int],
) -> dict[str, dict[str, set[int]]]:
    if not isinstance(value, dict) or set(value) != set(OUTPOST_SCOPES):
        raise ValueError(f"{location} scope keys differ from schema")
    result = _empty_outpost_seed_sets()
    for scope in OUTPOST_SCOPES:
        phases = value[scope]
        if not isinstance(phases, dict) or set(phases) != set(OUTPOST_PHASES):
            raise ValueError(f"{location}.{scope} phase keys differ from schema")
        for phase in OUTPOST_PHASES:
            seeds = phases[phase]
            if (
                not isinstance(seeds, list)
                or any(type(seed) is not int for seed in seeds)
                or seeds != sorted(set(seeds))
                or not set(seeds) <= allowed_seeds
            ):
                raise ValueError(
                    f"{location}.{scope}.{phase} seed list is invalid"
                )
            result[scope][phase] = set(seeds)
    return result


def _validate_outpost_coverage(
    value: Any,
    *,
    config: DaggerConfig,
    observed_counts: Mapping[str, Mapping[str, int]],
    observed_seeds: Mapping[str, Mapping[str, set[int]]],
) -> None:
    expected_keys = {
        "contract", "late_window_size", "readiness_targets",
        "seen_by_scope_phase", "retained_by_scope_phase",
        "seen_seeds_by_scope_phase", "retained_seeds_by_scope_phase",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("DAgger outpost coverage keys differ from schema")
    if value["contract"] != OUTPOST_COVERAGE_CONTRACT:
        raise ValueError("DAgger outpost coverage contract differs")
    if value["late_window_size"] != _late_window_size(config.steps):
        raise ValueError("DAgger outpost late-window contract differs")
    expected_targets = {
        "minimum_retained_overall": OUTPOST_MIN_RETAINED_OVERALL,
        "minimum_retained_late": OUTPOST_MIN_RETAINED_LATE,
        "minimum_distinct_late_seeds": OUTPOST_MIN_DISTINCT_LATE_SEEDS,
    }
    if value["readiness_targets"] != expected_targets:
        raise ValueError("DAgger outpost readiness targets differ")
    outpost_seeds = {
        int(entry["seed"])
        for entry in config.env_map()
        if entry["curriculum"] == OUTPOST_CURRICULUM
    }
    seen_counts = _validate_outpost_count_grid(
        value["seen_by_scope_phase"],
        location="data.sampling.outpost_coverage.seen_by_scope_phase",
    )
    retained_counts = _validate_outpost_count_grid(
        value["retained_by_scope_phase"],
        location="data.sampling.outpost_coverage.retained_by_scope_phase",
    )
    seen_seeds = _validate_outpost_seed_grid(
        value["seen_seeds_by_scope_phase"],
        location="data.sampling.outpost_coverage.seen_seeds_by_scope_phase",
        allowed_seeds=outpost_seeds,
    )
    retained_seeds = _validate_outpost_seed_grid(
        value["retained_seeds_by_scope_phase"],
        location="data.sampling.outpost_coverage.retained_seeds_by_scope_phase",
        allowed_seeds=outpost_seeds,
    )
    if retained_counts != observed_counts or retained_seeds != observed_seeds:
        raise ValueError("DAgger retained outpost coverage differs from rows")
    for scope in OUTPOST_SCOPES:
        for phase in OUTPOST_PHASES:
            if (
                seen_counts[scope][phase] < retained_counts[scope][phase]
                or not seen_seeds[scope][phase] >= retained_seeds[scope][phase]
                or seen_counts[scope][phase] < len(seen_seeds[scope][phase])
            ):
                raise ValueError("DAgger seen outpost coverage is inconsistent")
    if OUTPOST_CURRICULUM not in config.curricula:
        if any(
            seen_counts[scope][phase] or seen_seeds[scope][phase]
            for scope in OUTPOST_SCOPES for phase in OUTPOST_PHASES
        ):
            raise ValueError("DAgger non-outpost corpus records outpost coverage")
        return
    failures: list[str] = []
    for scope in OUTPOST_REQUIRED_SCOPES:
        retained_overall = sum(retained_counts[scope].values())
        seen_overall = sum(seen_counts[scope].values())
        retained_late = retained_counts[scope]["late"]
        seen_late = seen_counts[scope]["late"]
        retained_seed_count = len(retained_seeds[scope]["late"])
        seen_seed_count = len(seen_seeds[scope]["late"])
        if retained_overall < OUTPOST_MIN_RETAINED_OVERALL:
            failures.append(
                f"{scope}.overall retained={retained_overall} "
                f"seen={seen_overall} required={OUTPOST_MIN_RETAINED_OVERALL}"
            )
        if retained_late < OUTPOST_MIN_RETAINED_LATE:
            failures.append(
                f"{scope}.late retained={retained_late} "
                f"seen={seen_late} required={OUTPOST_MIN_RETAINED_LATE}"
            )
        if retained_seed_count < OUTPOST_MIN_DISTINCT_LATE_SEEDS:
            failures.append(
                f"{scope}.late_seeds retained={retained_seed_count} "
                f"seen={seen_seed_count} "
                f"required={OUTPOST_MIN_DISTINCT_LATE_SEEDS}"
            )
    if failures:
        raise ValueError(
            "DAgger outpost readiness coverage is insufficient: "
            + "; ".join(failures)
        )


def validate_dagger_corpus(
    corpus: Mapping[str, Any],
    *,
    verify_hashes: bool = True,
    verify_source: bool = True,
    expected_base_corpus_id: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> Mapping[str, Any]:
    if not isinstance(corpus, dict) or set(corpus) != {
        "kind", "corpus_schema_version", "corpus_schema_sha256", "meta", "data",
        "integrity",
    }:
        raise ValueError("DAgger corpus top-level schema invalid")
    if corpus["kind"] != DAGGER_CORPUS_KIND or corpus["corpus_schema_version"] != DAGGER_CORPUS_VERSION:
        raise ValueError("DAgger corpus kind/version incompatible")
    if corpus["corpus_schema_sha256"] != DAGGER_CORPUS_SCHEMA_SHA256:
        raise ValueError("DAgger corpus schema fingerprint differs")
    meta, data = corpus["meta"], corpus["data"]
    required_meta = {
        "num_envs", "steps", "seed", "curriculum", "room", "max_episode",
        "per_stratum", "reservoir_seed", "node", "device", "env_map",
        "base_corpus_sha256", "checkpoint_file_sha256",
        "checkpoint_kind", "checkpoint_qualified", "checkpoint_partial",
        "checkpoint_global_epoch", "checkpoint_global_epochs",
        "checkpoint_schema_version", "checkpoint_schema_sha256",
        "checkpoint_contracts",
        "checkpoint_source_sha256", "source_sha256", "collector_source_sha256",
        "policy_source_mismatch", "policy_source_mismatch_allowed",
        "checkpoint_teacher_abi", "current_teacher_abi",
        "policy_teacher_abi_mismatch", "policy_teacher_abi_mismatch_allowed",
        "schema_sha256", "environment_schema_version", "contracts", "runtime",
    }
    if not isinstance(meta, dict) or set(meta) != required_meta:
        raise ValueError("DAgger corpus metadata keys differ from schema")
    config = DaggerConfig(**{
        key: meta[key] for key in (
            "num_envs", "steps", "seed", "curriculum", "room", "max_episode",
            "per_stratum", "reservoir_seed", "node", "device",
        )
    })
    config.validate()
    if meta["env_map"] != config.env_map():
        raise ValueError("DAgger environment seed/curriculum map differs")
    for key in (
        "base_corpus_sha256", "checkpoint_file_sha256",
        "checkpoint_schema_sha256", "checkpoint_source_sha256",
        "source_sha256", "collector_source_sha256",
    ):
        if not isinstance(meta[key], str) or not _SHA256_RE.fullmatch(meta[key]):
            raise ValueError(f"DAgger metadata {key} is not a SHA-256")
    if expected_base_corpus_id is not None and meta["base_corpus_sha256"] != expected_base_corpus_id:
        raise ValueError("DAgger base corpus differs from expected")
    if expected_checkpoint_sha256 is not None and meta["checkpoint_file_sha256"] != expected_checkpoint_sha256:
        raise ValueError("DAgger checkpoint fingerprint differs from expected")
    if (
        meta["checkpoint_kind"] != "joint_pretrain"
        or type(meta["checkpoint_qualified"]) is not bool
        or meta["checkpoint_partial"] is not False
        or int(meta["checkpoint_global_epoch"]) <= 0
        or int(meta["checkpoint_global_epoch"]) != int(meta["checkpoint_global_epochs"])
    ):
        raise ValueError("DAgger checkpoint provenance is not a complete joint policy")
    if (
        type(meta["checkpoint_schema_version"]) is not int
        or meta["checkpoint_schema_version"] != meta["environment_schema_version"]
    ):
        raise ValueError("DAgger checkpoint and current schema versions differ")
    if (
        type(meta["policy_source_mismatch"]) is not bool
        or type(meta["policy_source_mismatch_allowed"]) is not bool
    ):
        raise ValueError("DAgger policy source mismatch audit flags are not booleans")
    actual_policy_mismatch = (
        meta["checkpoint_source_sha256"] != meta["source_sha256"]
    )
    if meta["policy_source_mismatch"] != actual_policy_mismatch:
        raise ValueError("DAgger policy source mismatch audit differs from recorded hashes")
    if actual_policy_mismatch and not meta["policy_source_mismatch_allowed"]:
        raise ValueError("DAgger policy source mismatch was not explicitly authorized")
    checkpoint_contracts = meta["checkpoint_contracts"]
    current_contracts = meta["contracts"]
    if (
        not isinstance(checkpoint_contracts, dict)
        or not isinstance(current_contracts, dict)
        or set(checkpoint_contracts) != set(current_contracts)
    ):
        raise ValueError("DAgger checkpoint artifact contracts differ in shape")
    non_teacher_mismatches = {
        key for key in current_contracts
        if key != "teacherAbi" and checkpoint_contracts[key] != current_contracts[key]
    }
    if non_teacher_mismatches:
        raise ValueError("DAgger checkpoint contains non-teacher ABI mismatches")
    if (
        type(meta["checkpoint_teacher_abi"]) is not int
        or type(meta["current_teacher_abi"]) is not int
        or type(meta["policy_teacher_abi_mismatch"]) is not bool
        or type(meta["policy_teacher_abi_mismatch_allowed"]) is not bool
    ):
        raise ValueError("DAgger policy teacher ABI audit fields have invalid types")
    if (
        meta["checkpoint_teacher_abi"] != checkpoint_contracts["teacherAbi"]
        or meta["current_teacher_abi"] != current_contracts["teacherAbi"]
    ):
        raise ValueError("DAgger policy teacher ABI values differ from contracts")
    actual_teacher_mismatch = (
        meta["checkpoint_teacher_abi"] != meta["current_teacher_abi"]
    )
    if meta["policy_teacher_abi_mismatch"] != actual_teacher_mismatch:
        raise ValueError("DAgger policy teacher ABI mismatch audit is inconsistent")
    if actual_teacher_mismatch:
        if not meta["policy_teacher_abi_mismatch_allowed"]:
            raise ValueError("DAgger policy teacher ABI mismatch was not authorized")
        bridge = _POLICY_TEACHER_ABI_BRIDGE
        if not (
            meta["checkpoint_schema_version"] == bridge["schema_version"]
            and meta["environment_schema_version"] == bridge["schema_version"]
            and meta["checkpoint_teacher_abi"] == bridge["checkpoint_teacher_abi"]
            and meta["current_teacher_abi"] == bridge["current_teacher_abi"]
            and meta["checkpoint_schema_sha256"]
            == bridge["checkpoint_schema_sha256"]
            and meta["schema_sha256"] == bridge["current_schema_sha256"]
        ):
            raise ValueError(
                "DAgger policy does not match the audited teacher ABI 19-to-20 bridge"
            )
    elif meta["checkpoint_schema_sha256"] != meta["schema_sha256"]:
        raise ValueError("DAgger checkpoint schema hash differs without a teacher ABI mismatch")
    if (
        meta["schema_sha256"] != SCHEMA_SHA256
        or meta["environment_schema_version"] != SCHEMA["version"]
        or meta["contracts"] != SCHEMA["artifact"]
    ):
        raise ValueError("DAgger observation/action ABI differs")
    runtime = meta["runtime"]
    if not isinstance(runtime, dict) or runtime.get("step_api") != "VecScreepsEnv.step_labeled/v1" or runtime.get("teacher_alignment") != "teacher_actions_label_pre_action_host_obs":
        raise ValueError("DAgger step-label alignment provenance differs")
    if verify_source:
        current_source = source_signature()
        if (
            meta["source_sha256"] != current_source
            or meta["collector_source_sha256"] != _collector_source_sha256()
        ):
            raise ValueError("DAgger executable or collector source fingerprint differs")
    if not isinstance(data, dict) or set(data) != {"rows", "sampling", "collection"}:
        raise ValueError("DAgger corpus data keys differ from schema")
    rows, sampling = data["rows"], data["sampling"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("DAgger corpus rows are empty")
    expected_sampling_keys = {
        "algorithm", "capacity_per_stratum", "rng", "seed",
        "seen_by_stratum", "retained_by_stratum", "seen_by_row_kind",
        "outpost_coverage", "retained",
    }
    if (
        not isinstance(sampling, dict)
        or set(sampling) != expected_sampling_keys
        or sampling.get("algorithm") != "algorithm_r_per_semantic_stratum"
    ):
        raise ValueError("DAgger sampling provenance invalid")
    if int(sampling.get("capacity_per_stratum", 0)) != config.per_stratum:
        raise ValueError("DAgger sampling capacity differs from metadata")
    seen_by_stratum = sampling.get("seen_by_stratum")
    retained_by_stratum = sampling.get("retained_by_stratum")
    if not isinstance(seen_by_stratum, dict) or not isinstance(retained_by_stratum, dict):
        raise ValueError("DAgger per-stratum sampling counts are missing")
    observed: dict[str, int] = {}
    row_kind_counts = {"exact_intent": 0, "spawn_positive": 0, "spawn_wait_legal": 0}
    observed_outpost = _empty_outpost_counts()
    observed_outpost_seeds = _empty_outpost_seed_sets()
    env_map = {int(row["env_index"]): row for row in meta["env_map"]}
    for index, row in enumerate(rows):
        location = f"data.rows[{index}]"
        if not isinstance(row, dict) or set(row) != set(_FORMAT_SPEC["row"]):
            raise ValueError(f"{location} keys differ from schema")
        kind = str(row["kind"])
        if kind not in row_kind_counts:
            raise ValueError(f"{location}.kind invalid")
        timestep, env_index, actor_index = (
            int(row["timestep"]), int(row["env_index"]), int(row["actor_index"]),
        )
        if not 0 <= timestep < config.steps or env_index not in env_map:
            raise ValueError(f"{location} trajectory reference invalid")
        _, actor_cap, _ = _validate_obs(row["obs"], f"{location}.obs")
        _validate_action(row["action"], actor_cap, f"{location}.action")
        if not 0 <= actor_index < actor_cap or not bool(row["obs"]["actor_mask"][0, actor_index]):
            raise ValueError(f"{location}.actor_index is not a live compact actor")
        semantics = _row_semantics(
            row["obs"], row["action"],
            curriculum=str(env_map[env_index]["curriculum"]),
            timestep=timestep, steps=config.steps,
            env_index=0, actor_index=actor_index,
        )
        if semantics != (kind, row["stratum"]):
            raise ValueError(f"{location} semantic stratum does not match tensors")
        _validate_selected_action_legality(
            row["obs"], row["action"], actor_index=actor_index,
            location=location,
        )
        if kind == "spawn_positive":
            counts = row["action"]["body_counts"][0, actor_index, 0]
            if int(counts.sum()) <= 0:
                raise ValueError(f"{location} positive spawn body is empty")
            budget = int(round(
                float(row["obs"]["actors"][0, actor_index, ACTOR_FEATURE_INDEX["roomEnergyAvailable"]])
                * MAX_ROOM_ENERGY
            ))
            cost = sum(int(counts[part]) * BODY_PART_COSTS[part] for part in range(N_BODY_PART))
            if cost > budget:
                raise ValueError(f"{location} positive spawn body is unaffordable")
        observed[row["stratum"]] = observed.get(row["stratum"], 0) + 1
        row_kind_counts[kind] += 1
        outpost_semantics = _outpost_phase_scope(
            row["obs"], row["action"],
            curriculum=str(env_map[env_index]["curriculum"]),
            timestep=timestep, steps=config.steps,
            env_index=0, actor_index=actor_index,
        )
        if outpost_semantics is not None:
            phase, scope = outpost_semantics
            observed_outpost[scope][phase] += 1
            observed_outpost_seeds[scope][phase].add(
                int(env_map[env_index]["seed"])
            )
    if observed != retained_by_stratum:
        raise ValueError("DAgger retained_by_stratum differs from rows")
    if set(seen_by_stratum) != set(observed) or any(
        int(seen_by_stratum[key]) < count or count > config.per_stratum
        for key, count in observed.items()
    ):
        raise ValueError("DAgger per-stratum reservoir bounds/counts invalid")
    if int(sampling.get("retained", -1)) != len(rows):
        raise ValueError("DAgger retained row count differs")
    if sampling.get("seen_by_row_kind") != row_kind_counts:
        # seen may exceed retained, but it may never be smaller than retained.
        recorded = sampling.get("seen_by_row_kind")
        if not isinstance(recorded, dict) or any(
            int(recorded.get(kind, -1)) < count for kind, count in row_kind_counts.items()
        ):
            raise ValueError("DAgger seen_by_row_kind is inconsistent")
    _validate_outpost_coverage(
        sampling["outpost_coverage"], config=config,
        observed_counts=observed_outpost,
        observed_seeds=observed_outpost_seeds,
    )
    collection = data["collection"]
    if not isinstance(collection, dict) or set(collection) != {
        "transitions", "learner_intent_issued", "learner_intent_invalid",
    }:
        raise ValueError("DAgger collection metrics differ from schema")
    if int(collection["transitions"]) != config.steps * config.num_envs:
        raise ValueError("DAgger transition count differs")
    if verify_hashes:
        unhashed = {key: value for key, value in corpus.items() if key != "integrity"}
        integrity = corpus["integrity"]
        if integrity.get("algorithm") != "sha256-semantic-v1" or integrity.get("corpus_sha256") != content_sha256(unhashed):
            raise ValueError("DAgger semantic SHA-256 mismatch")
    return corpus


def _encode_tensors(
    value: Any,
    pending: dict[tuple[str, tuple[int, ...]], list[torch.Tensor]],
) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        signature = (str(tensor.dtype), tuple(tensor.shape))
        index = len(pending.setdefault(signature, []))
        pending[signature].append(tensor)
        return {"$tensor_group": [signature[0], list(signature[1])], "$index": index}
    if isinstance(value, Mapping):
        return {key: _encode_tensors(item, pending) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_tensors(item, pending) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_shards(directory: Path, groups: dict, encoded: Any) -> tuple[Any, list[dict]]:
    shard_dir = directory / "shards"
    shard_dir.mkdir()
    locations_by_group: dict[str, list[dict[str, Any]]] = {}
    shards: list[dict[str, Any]] = []
    for signature in sorted(groups):
        tensors = groups[signature]
        item_bytes = max(1, tensors[0].numel() * tensors[0].element_size())
        per_shard = max(1, SHARD_MAX_BYTES // item_bytes)
        group_key = json.dumps([signature[0], list(signature[1])], separators=(",", ":"))
        locations: list[dict[str, Any]] = []
        for start in range(0, len(tensors), per_shard):
            batch = torch.stack(tensors[start : start + per_shard])
            temporary = shard_dir / f"group-{len(shards):05d}.pt"
            torch.save({"tensor": batch}, temporary)
            sha256 = _sha256_file(temporary)
            target = shard_dir / f"{sha256}.pt"
            temporary.replace(target)
            record = {
                "file": f"shards/{target.name}", "sha256": sha256,
                "count": int(batch.shape[0]), "dtype": signature[0],
                "shape": list(signature[1]),
            }
            shards.append(record)
            locations.append({
                "file": record["file"], "start": start, "count": int(batch.shape[0]),
            })
        locations_by_group[group_key] = locations

    def resolve(item: Any) -> Any:
        if isinstance(item, dict) and set(item) == {"$tensor_group", "$index"}:
            key = json.dumps(item["$tensor_group"], separators=(",", ":"))
            index = int(item["$index"])
            for location in locations_by_group[key]:
                if location["start"] <= index < location["start"] + location["count"]:
                    return {
                        "$tensor": location["file"],
                        "$index": index - location["start"],
                    }
            raise AssertionError("DAgger tensor group reference unresolved")
        if isinstance(item, dict):
            return {key: resolve(value) for key, value in item.items()}
        if isinstance(item, list):
            return [resolve(value) for value in item]
        return item

    return resolve(encoded), shards


def save_dagger_corpus(corpus: Mapping[str, Any], output_root: str | Path) -> Path:
    """Write ``output_root/<semantic-sha256>`` atomically and make it immutable."""
    validate_dagger_corpus(corpus, verify_source=False)
    content_id = str(corpus["integrity"]["corpus_sha256"])
    root = Path(output_root)
    destination = root / content_id
    if destination.exists():
        loaded = load_dagger_corpus(destination, verify_source=False)
        if loaded["integrity"]["corpus_sha256"] != content_id:
            raise FileExistsError(f"non-identical DAgger corpus exists at {destination}")
        return destination
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{content_id}.", dir=root))
    try:
        groups: dict = {}
        encoded = _encode_tensors(corpus, groups)
        encoded, shards = _write_shards(temporary, groups, encoded)
        manifest = {
            "storage_format": "tensor-shards-v1",
            "content_id": content_id,
            "corpus": encoded,
            "shards": shards,
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


def load_dagger_corpus(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
    verify_source: bool = True,
    expected_base_corpus_id: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Load only canonical JSON plus hash-checked ``weights_only`` tensor shards."""
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("DAgger corpus directory must not be a symlink")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("DAgger manifest must not be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {
        "storage_format", "content_id", "corpus", "shards", "manifest_sha256",
    }:
        raise ValueError("DAgger manifest schema invalid")
    recorded_manifest_sha = manifest.pop("manifest_sha256")
    if recorded_manifest_sha != hashlib.sha256(_canonical_json(manifest)).hexdigest():
        raise ValueError("DAgger manifest SHA-256 mismatch")
    if manifest["storage_format"] != "tensor-shards-v1":
        raise ValueError("DAgger storage format unsupported")
    records: dict[str, dict[str, Any]] = {}
    for record in manifest["shards"]:
        if not isinstance(record, dict) or set(record) != {
            "file", "sha256", "count", "dtype", "shape",
        }:
            raise ValueError("DAgger shard record schema invalid")
        relative = str(record["file"])
        parts = Path(relative).parts
        if len(parts) != 2 or parts[0] != "shards" or not _SHA256_RE.fullmatch(Path(relative).stem) or Path(relative).suffix != ".pt":
            raise ValueError("DAgger shard path is not canonical")
        if relative in records or record["sha256"] != Path(relative).stem:
            raise ValueError("DAgger shard identity is duplicated or inconsistent")
        records[relative] = record
    actual_shards = {
        path.relative_to(root).as_posix()
        for path in (root / "shards").glob("*.pt")
    }
    if actual_shards != set(records):
        raise ValueError("DAgger shard directory differs from manifest")
    actual_entries = {
        path.relative_to(root).as_posix() for path in root.rglob("*")
    }
    expected_entries = {"manifest.json", "shards", *records}
    if actual_entries != expected_entries:
        raise ValueError("DAgger corpus directory contains undeclared entries")
    cache: dict[str, torch.Tensor] = {}
    referenced: set[str] = set()

    def decode(item: Any) -> Any:
        if isinstance(item, dict) and set(item) == {"$tensor", "$index"}:
            relative = str(item["$tensor"])
            record = records.get(relative)
            if record is None:
                raise ValueError("DAgger tensor reference names an undeclared shard")
            referenced.add(relative)
            if relative not in cache:
                path = root / relative
                if path.is_symlink():
                    raise ValueError("DAgger shards must not be symlinks")
                if verify_hashes and _sha256_file(path) != record["sha256"]:
                    raise ValueError(f"DAgger tensor shard integrity failure: {relative}")
                payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if not isinstance(payload, dict) or set(payload) != {"tensor"} or not torch.is_tensor(payload["tensor"]):
                    raise ValueError(f"DAgger tensor shard schema failure: {relative}")
                tensor = payload["tensor"]
                if (
                    int(tensor.shape[0]) != int(record["count"])
                    or str(tensor.dtype) != record["dtype"]
                    or list(tensor.shape[1:]) != record["shape"]
                ):
                    raise ValueError(f"DAgger tensor shard metadata failure: {relative}")
                cache[relative] = tensor
            index = int(item["$index"])
            if not 0 <= index < cache[relative].shape[0]:
                raise ValueError("DAgger tensor reference index is out of range")
            return cache[relative][index]
        if isinstance(item, dict):
            return {key: decode(value) for key, value in item.items()}
        if isinstance(item, list):
            return [decode(value) for value in item]
        return item

    corpus = decode(manifest["corpus"])
    if referenced != set(records):
        raise ValueError("DAgger manifest contains unreferenced tensor shards")
    content_id = str(manifest["content_id"])
    if root.name != content_id or corpus.get("integrity", {}).get("corpus_sha256") != content_id:
        raise ValueError("DAgger content-addressed directory name/manifest differs")
    validate_dagger_corpus(
        corpus, verify_hashes=verify_hashes, verify_source=verify_source,
        expected_base_corpus_id=expected_base_corpus_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    return corpus


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-corpus-id", required=True)
    parser.add_argument(
        "--allow-policy-source-mismatch", action="store_true",
        help=(
            "explicitly authorize a joint policy whose executable-source hash differs; "
            "schema version, model state, artifact ABIs, completeness, and base corpus "
            "remain independently enforced"
        ),
    )
    parser.add_argument(
        "--allow-policy-teacher-abi-mismatch", action="store_true",
        help=(
            "explicitly authorize a strictly older teacher ABI on an otherwise "
            "ABI-identical complete joint policy"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--room", default="W7N3")
    parser.add_argument("--max-episode", type=int, default=None)
    parser.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM)
    parser.add_argument("--reservoir-seed", type=int, default=None)
    parser.add_argument("--node", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Intra-op parallelism makes environment workers contend; see
    # vec_env.configure_host_threads.
    configure_host_threads()
    args = parse_args(argv)
    config = DaggerConfig(
        num_envs=args.num_envs, steps=args.steps, seed=args.seed,
        curriculum=args.curriculum, room=args.room, max_episode=args.max_episode,
        per_stratum=args.per_stratum, reservoir_seed=args.reservoir_seed,
        node=args.node, device=args.device,
    )
    corpus = collect_dagger_corpus(
        config, checkpoint_path=args.checkpoint,
        base_corpus_id=args.base_corpus_id,
        allow_policy_source_mismatch=args.allow_policy_source_mismatch,
        allow_policy_teacher_abi_mismatch=(
            args.allow_policy_teacher_abi_mismatch
        ),
    )
    path = save_dagger_corpus(corpus, args.output)
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
