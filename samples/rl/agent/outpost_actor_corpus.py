#!/usr/bin/env python3
"""Collect an immutable scripted-teacher actor supplement for seed_outpost.

This artifact is deliberately not DAgger: the scripted teacher both chooses and
applies every action, so its rows come from teacher-forced trajectories.  Rows
use the DAgger actor-row ABI only so the joint trainer can consume the two
independently validated sources through the same supervised lanes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

from .artifacts import source_signature
from .constants import (
    ACTOR_FEATURE_INDEX, BODY_PART_COSTS, MAX_ROOM_ENERGY, N_BODY_PART,
    SCHEMA, SCHEMA_SHA256,
)
from .dagger_corpus import (
    ACTION_KEYS, DEFAULT_PER_STRATUM, OUTPOST_COVERAGE_CONTRACT,
    OUTPOST_CURRICULUM, OUTPOST_MIN_DISTINCT_LATE_SEEDS,
    OUTPOST_MIN_RETAINED_LATE, OUTPOST_MIN_RETAINED_OVERALL, OUTPOST_PHASES,
    OUTPOST_REQUIRED_SCOPES, OUTPOST_SCOPES, _canonical_json, _clone_row,
    _empty_outpost_counts, _empty_outpost_seed_sets, _encode_tensors,
    _late_window_size, _outpost_phase_scope, _reservoir_offer, _row_semantics,
    _serialized_seed_sets, _sha256_file, _validate_action, _validate_obs,
    _validate_outpost_count_grid, _validate_outpost_seed_grid,
    _validate_selected_action_legality, _write_shards, content_sha256,
)
from .env_client import _COMMAND_MAGIC, _COMMAND_VERSION, _RESPONSE_VERSION
from .pretrain_corpus import _invalid_scripted_intent_details
from .vec_env import VecScreepsEnv, configure_host_threads


OUTPOST_ACTOR_CORPUS_KIND = "scripted_teacher_supplement"
OUTPOST_ACTOR_CORPUS_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "runs" / "outpost-actor-corpora"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROW_KEYS = {
    "kind", "stratum", "timestep", "env_index", "actor_index", "obs",
    "action",
}
_RUNTIME_KEYS = {
    "python", "numpy", "torch", "platform", "node_executable", "node_version",
    "observation_protocol", "response_protocol_version", "command_protocol",
    "command_protocol_version", "step_api", "teacher_alignment",
    "trajectory_policy", "lean_meta",
}
_FORMAT_SPEC = {
    "kind": OUTPOST_ACTOR_CORPUS_KIND,
    "version": OUTPOST_ACTOR_CORPUS_VERSION,
    "storage": "tensor-shards-v1",
    "trajectory_authority": "scripted_teacher_applied_action",
    "row_abi": sorted(_ROW_KEYS),
    "row_kinds": ["exact_intent", "spawn_positive", "spawn_wait_legal"],
    "semantic_strata": OUTPOST_COVERAGE_CONTRACT,
    "readiness": {
        "minimum_retained_overall": OUTPOST_MIN_RETAINED_OVERALL,
        "minimum_retained_late": OUTPOST_MIN_RETAINED_LATE,
        "minimum_distinct_late_seeds": OUTPOST_MIN_DISTINCT_LATE_SEEDS,
    },
}
OUTPOST_ACTOR_CORPUS_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(_FORMAT_SPEC, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
_COLLECTOR_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class OutpostActorConfig:
    num_envs: int
    steps: int
    seed: int
    curriculum: str = OUTPOST_CURRICULUM
    per_stratum: int = DEFAULT_PER_STRATUM
    device: str = "cpu"
    room: str = "W7N3"
    node: str | None = None
    reservoir_seed: int | None = None

    @property
    def resolved_max_episode(self) -> int:
        return int(self.steps) + 5

    @property
    def resolved_reservoir_seed(self) -> int:
        return int(
            self.reservoir_seed
            if self.reservoir_seed is not None
            else (int(self.seed) ^ 0x4F555450)
        )

    def env_map(self) -> list[dict[str, Any]]:
        return [
            {
                "env_index": env_index,
                "seed": int(self.seed) + env_index,
                "curriculum": OUTPOST_CURRICULUM,
            }
            for env_index in range(self.num_envs)
        ]

    def validate(self) -> None:
        if self.num_envs <= 0 or self.steps <= 0:
            raise ValueError("num_envs and steps must be positive")
        if self.curriculum != OUTPOST_CURRICULUM:
            raise ValueError("actor supplement curriculum must be seed_outpost")
        if self.per_stratum <= 0:
            raise ValueError("per_stratum must be positive")
        if not -(2**63) <= self.resolved_reservoir_seed < 2**63:
            raise ValueError("reservoir_seed must fit a signed 64-bit integer")
        try:
            torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError("device is not a valid torch device") from error


def classify_outpost_actor_row(
    obs: dict[str, torch.Tensor],
    action: dict[str, torch.Tensor],
    *,
    timestep: int,
    steps: int,
    env_index: int,
    actor_index: int,
) -> tuple[str, str]:
    """Expose the exact DAgger-v2 seed-outpost phase/scope classifier."""
    classified = _outpost_phase_scope(
        obs, action, curriculum=OUTPOST_CURRICULUM, timestep=timestep,
        steps=steps, env_index=env_index, actor_index=actor_index,
    )
    assert classified is not None
    return classified


def _collector_source_sha256() -> str:
    return _COLLECTOR_SOURCE_SHA256


def _assert_collector_unchanged() -> None:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _COLLECTOR_SOURCE_SHA256:
        raise RuntimeError("outpost actor collector source changed after import")


def _coverage_from_rows(
    rows: list[dict[str, Any]], config: OutpostActorConfig,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, set[int]]]]:
    counts = _empty_outpost_counts()
    seeds = _empty_outpost_seed_sets()
    env_map = {entry["env_index"]: entry for entry in config.env_map()}
    for row in rows:
        phase, scope = classify_outpost_actor_row(
            row["obs"], row["action"], timestep=int(row["timestep"]),
            steps=config.steps, env_index=0, actor_index=int(row["actor_index"]),
        )
        counts[scope][phase] += 1
        seeds[scope][phase].add(int(env_map[int(row["env_index"])]["seed"]))
    return counts, seeds


@torch.inference_mode()
def collect_outpost_actor_rows(
    envs: Any, config: OutpostActorConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Collect exact labels while the scripted teacher controls the trajectory."""
    config.validate()
    rng = np.random.default_rng(config.resolved_reservoir_seed)
    rows: list[dict[str, Any]] = []
    retained_indices: dict[str, list[int]] = {}
    seen_by_stratum: dict[str, int] = {}
    seen_by_row_kind = {
        "exact_intent": 0, "spawn_positive": 0, "spawn_wait_legal": 0,
    }
    seen_counts = _empty_outpost_counts()
    seen_seeds = _empty_outpost_seed_sets()
    teacher_issued = 0
    teacher_invalid = 0
    envs.reset()
    for timestep in range(config.steps):
        if envs.host_obs is None:
            raise RuntimeError("step_scripted collector requires VecScreepsEnv.host_obs")
        pre = envs.host_obs
        _obs, _rewards, dones, infos, teacher = envs.step_scripted()
        if bool(dones.any()):
            raise RuntimeError(
                "scripted supplement episode ended during collection; alignment lost"
            )
        for info_value in infos:
            info = info_value or {}
            if info.get("recovered") or info.get("invalid_demo"):
                raise RuntimeError("scripted supplement environment recovery is inadmissible")
        invalid_details = _invalid_scripted_intent_details(infos, config.env_map())
        if invalid_details:
            raise RuntimeError(
                "scripted supplement teacher emitted invalid intent "
                f"at step={timestep + 1}: "
                f"{json.dumps(invalid_details, sort_keys=True, separators=(',', ':'))}"
            )
        for info_value in infos:
            info = info_value or {}
            teacher_issued += int(info.get("intentIssued") or 0)
            teacher_invalid += int(info.get("intentInvalid") or 0)
        for env_index, _info_value in enumerate(infos):
            live = torch.nonzero(
                pre["actor_mask"][env_index] > 0, as_tuple=False,
            ).flatten()
            for actor_tensor in live:
                actor_index = int(actor_tensor)
                semantics = _row_semantics(
                    pre, teacher, curriculum=OUTPOST_CURRICULUM,
                    timestep=timestep, steps=config.steps,
                    env_index=env_index, actor_index=actor_index,
                )
                if semantics is None:
                    continue
                row_kind, stratum = semantics
                seen_by_row_kind[row_kind] += 1
                phase, scope = classify_outpost_actor_row(
                    pre, teacher, timestep=timestep, steps=config.steps,
                    env_index=env_index, actor_index=actor_index,
                )
                seen_counts[scope][phase] += 1
                seen_seeds[scope][phase].add(int(config.seed) + env_index)
                _reservoir_offer(
                    rows, retained_indices, seen_by_stratum, stratum=stratum,
                    capacity=config.per_stratum, rng=rng,
                    make_row=lambda rk=row_kind, st=stratum, ti=timestep,
                    ei=env_index, ai=actor_index: _clone_row(
                        pre, teacher, row_kind=rk, stratum=st, timestep=ti,
                        env_index=ei, actor_index=ai,
                    ),
                )
        if (timestep + 1) % 500 == 0 or timestep + 1 == config.steps:
            print(
                f"[outpost-actor] {timestep + 1}/{config.steps} "
                f"rows={len(rows)} strata={len(seen_by_stratum)}",
                flush=True,
            )
    if not rows:
        raise RuntimeError("scripted supplement retained no exact actor labels")
    rows.sort(key=lambda row: (
        row["stratum"], row["timestep"], row["env_index"], row["actor_index"],
    ))
    retained_by_stratum = {
        stratum: sum(row["stratum"] == stratum for row in rows)
        for stratum in sorted(seen_by_stratum)
    }
    retained_counts, retained_seeds = _coverage_from_rows(rows, config)
    sampling = {
        "algorithm": "algorithm_r_per_semantic_stratum",
        "capacity_per_stratum": int(config.per_stratum),
        "rng": "numpy.default_rng/PCG64",
        "seed": int(config.resolved_reservoir_seed),
        "seen_by_stratum": dict(sorted(seen_by_stratum.items())),
        "retained_by_stratum": retained_by_stratum,
        "seen_by_row_kind": seen_by_row_kind,
        "outpost_coverage": {
            "contract": OUTPOST_COVERAGE_CONTRACT,
            "late_window_size": _late_window_size(config.steps),
            "readiness_targets": {
                "minimum_retained_overall": OUTPOST_MIN_RETAINED_OVERALL,
                "minimum_retained_late": OUTPOST_MIN_RETAINED_LATE,
                "minimum_distinct_late_seeds": OUTPOST_MIN_DISTINCT_LATE_SEEDS,
            },
            "seen_by_scope_phase": seen_counts,
            "retained_by_scope_phase": retained_counts,
            "seen_seeds_by_scope_phase": _serialized_seed_sets(seen_seeds),
            "retained_seeds_by_scope_phase": _serialized_seed_sets(retained_seeds),
        },
        "retained": len(rows),
    }
    collection = {
        "transitions": int(config.steps * config.num_envs),
        "teacher_intent_issued": int(teacher_issued),
        "teacher_intent_invalid": int(teacher_invalid),
    }
    return rows, sampling, collection


def _runtime_provenance(config: OutpostActorConfig) -> dict[str, Any]:
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
        "observation_protocol": "XRL1",
        "response_protocol_version": int(_RESPONSE_VERSION),
        "command_protocol": _COMMAND_MAGIC.decode("ascii"),
        "command_protocol_version": int(_COMMAND_VERSION),
        "step_api": "VecScreepsEnv.step_scripted/v1",
        "teacher_alignment": "scripted_actions_apply_to_pre_action_host_obs",
        "trajectory_policy": OUTPOST_ACTOR_CORPUS_KIND,
        "lean_meta": False,
    }


def assemble_outpost_actor_corpus(
    meta: Mapping[str, Any], rows: list[dict[str, Any]],
    sampling: Mapping[str, Any], collection: Mapping[str, Any],
) -> dict[str, Any]:
    corpus = {
        "kind": OUTPOST_ACTOR_CORPUS_KIND,
        "corpus_schema_version": OUTPOST_ACTOR_CORPUS_VERSION,
        "corpus_schema_sha256": OUTPOST_ACTOR_CORPUS_SCHEMA_SHA256,
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
    validate_outpost_actor_corpus(corpus, verify_source=False)
    return corpus


def collect_outpost_actor_corpus(
    config: OutpostActorConfig, *, base_corpus_id: str,
) -> dict[str, Any]:
    config.validate()
    if not _SHA256_RE.fullmatch(base_corpus_id):
        raise ValueError("base_corpus_id must be a lowercase SHA-256")
    _assert_collector_unchanged()
    collection_source_sha256 = source_signature()
    envs = VecScreepsEnv(
        config.num_envs, node=config.node, room=config.room,
        max_episode=config.resolved_max_episode, device=torch.device(config.device),
        curriculum=OUTPOST_CURRICULUM, lean_meta=False, seed=config.seed,
    )
    try:
        rows, sampling, collection = collect_outpost_actor_rows(envs, config)
    finally:
        envs.close()
    _assert_collector_unchanged()
    if source_signature() != collection_source_sha256:
        raise RuntimeError("executable RL source changed during supplement collection")
    meta = {
        **asdict(config),
        "max_episode": config.resolved_max_episode,
        "reservoir_seed": config.resolved_reservoir_seed,
        "env_map": config.env_map(),
        "supplement_type": OUTPOST_ACTOR_CORPUS_KIND,
        "base_corpus_sha256": base_corpus_id,
        "collection_source_sha256": collection_source_sha256,
        "collector_source_sha256": _collector_source_sha256(),
        "schema_sha256": SCHEMA_SHA256,
        "environment_schema_version": SCHEMA["version"],
        "contracts": dict(SCHEMA["artifact"]),
        "runtime": _runtime_provenance(config),
    }
    return assemble_outpost_actor_corpus(meta, rows, sampling, collection)


def _validate_coverage(
    value: Any, *, config: OutpostActorConfig,
    observed_counts: Mapping[str, Mapping[str, int]],
    observed_seeds: Mapping[str, Mapping[str, set[int]]],
) -> None:
    expected_keys = {
        "contract", "late_window_size", "readiness_targets",
        "seen_by_scope_phase", "retained_by_scope_phase",
        "seen_seeds_by_scope_phase", "retained_seeds_by_scope_phase",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("supplement outpost coverage keys differ from schema")
    if value["contract"] != OUTPOST_COVERAGE_CONTRACT:
        raise ValueError("supplement outpost coverage contract differs")
    if value["late_window_size"] != _late_window_size(config.steps):
        raise ValueError("supplement outpost late-window contract differs")
    targets = {
        "minimum_retained_overall": OUTPOST_MIN_RETAINED_OVERALL,
        "minimum_retained_late": OUTPOST_MIN_RETAINED_LATE,
        "minimum_distinct_late_seeds": OUTPOST_MIN_DISTINCT_LATE_SEEDS,
    }
    if value["readiness_targets"] != targets:
        raise ValueError("supplement readiness targets differ")
    allowed_seeds = {entry["seed"] for entry in config.env_map()}
    seen_counts = _validate_outpost_count_grid(
        value["seen_by_scope_phase"], location="supplement.seen_by_scope_phase",
    )
    retained_counts = _validate_outpost_count_grid(
        value["retained_by_scope_phase"], location="supplement.retained_by_scope_phase",
    )
    seen_seeds = _validate_outpost_seed_grid(
        value["seen_seeds_by_scope_phase"], location="supplement.seen_seeds_by_scope_phase",
        allowed_seeds=allowed_seeds,
    )
    retained_seeds = _validate_outpost_seed_grid(
        value["retained_seeds_by_scope_phase"],
        location="supplement.retained_seeds_by_scope_phase", allowed_seeds=allowed_seeds,
    )
    if retained_counts != observed_counts or retained_seeds != observed_seeds:
        raise ValueError("supplement retained outpost coverage differs from rows")
    for scope in OUTPOST_SCOPES:
        for phase in OUTPOST_PHASES:
            if (
                seen_counts[scope][phase] < retained_counts[scope][phase]
                or not seen_seeds[scope][phase] >= retained_seeds[scope][phase]
                or seen_counts[scope][phase] < len(seen_seeds[scope][phase])
            ):
                raise ValueError("supplement seen outpost coverage is inconsistent")
    failures: list[str] = []
    for scope in OUTPOST_REQUIRED_SCOPES:
        retained_overall = sum(retained_counts[scope].values())
        retained_late = retained_counts[scope]["late"]
        retained_late_seeds = len(retained_seeds[scope]["late"])
        if retained_overall < OUTPOST_MIN_RETAINED_OVERALL:
            failures.append(
                f"{scope}.overall retained={retained_overall} "
                f"seen={sum(seen_counts[scope].values())} "
                f"required={OUTPOST_MIN_RETAINED_OVERALL}"
            )
        if retained_late < OUTPOST_MIN_RETAINED_LATE:
            failures.append(
                f"{scope}.late retained={retained_late} "
                f"seen={seen_counts[scope]['late']} required={OUTPOST_MIN_RETAINED_LATE}"
            )
        if retained_late_seeds < OUTPOST_MIN_DISTINCT_LATE_SEEDS:
            failures.append(
                f"{scope}.late_seeds retained={retained_late_seeds} "
                f"seen={len(seen_seeds[scope]['late'])} "
                f"required={OUTPOST_MIN_DISTINCT_LATE_SEEDS}"
            )
    if failures:
        raise ValueError(
            "scripted supplement outpost readiness is insufficient: "
            + "; ".join(failures)
        )


def validate_outpost_actor_corpus(
    corpus: Mapping[str, Any], *, verify_hashes: bool = True,
    verify_source: bool = True, expected_base_corpus_id: str | None = None,
) -> Mapping[str, Any]:
    if not isinstance(corpus, dict) or set(corpus) != {
        "kind", "corpus_schema_version", "corpus_schema_sha256", "meta", "data",
        "integrity",
    }:
        raise ValueError("scripted supplement top-level schema invalid")
    if (
        corpus["kind"] != OUTPOST_ACTOR_CORPUS_KIND
        or corpus["corpus_schema_version"] != OUTPOST_ACTOR_CORPUS_VERSION
        or corpus["corpus_schema_sha256"] != OUTPOST_ACTOR_CORPUS_SCHEMA_SHA256
    ):
        raise ValueError("scripted supplement kind/schema is incompatible")
    meta = corpus["meta"]
    expected_meta = {
        "num_envs", "steps", "seed", "curriculum", "per_stratum", "device",
        "room", "node", "reservoir_seed", "max_episode", "env_map",
        "supplement_type", "base_corpus_sha256", "collection_source_sha256",
        "collector_source_sha256", "schema_sha256", "environment_schema_version",
        "contracts", "runtime",
    }
    if not isinstance(meta, dict) or set(meta) != expected_meta:
        raise ValueError("scripted supplement metadata keys differ from schema")
    config = OutpostActorConfig(**{
        key: meta[key] for key in (
            "num_envs", "steps", "seed", "curriculum", "per_stratum", "device",
            "room", "node", "reservoir_seed",
        )
    })
    config.validate()
    if meta["max_episode"] != config.resolved_max_episode or meta["env_map"] != config.env_map():
        raise ValueError("scripted supplement environment map/config differs")
    if meta["supplement_type"] != OUTPOST_ACTOR_CORPUS_KIND:
        raise ValueError("scripted supplement trajectory authority differs")
    for key in (
        "base_corpus_sha256", "collection_source_sha256", "collector_source_sha256",
        "schema_sha256",
    ):
        if not isinstance(meta[key], str) or not _SHA256_RE.fullmatch(meta[key]):
            raise ValueError(f"scripted supplement {key} is not a SHA-256")
    if (
        expected_base_corpus_id is not None
        and meta["base_corpus_sha256"] != expected_base_corpus_id
    ):
        raise ValueError("scripted supplement base corpus differs from expected")
    if (
        meta["schema_sha256"] != SCHEMA_SHA256
        or meta["environment_schema_version"] != SCHEMA["version"]
        or meta["contracts"] != SCHEMA["artifact"]
    ):
        raise ValueError("scripted supplement observation/action ABI differs")
    runtime = meta["runtime"]
    if (
        not isinstance(runtime, dict)
        or set(runtime) != _RUNTIME_KEYS
        or any(
            not isinstance(runtime[key], str)
            for key in (
                "python", "numpy", "torch", "platform", "node_executable",
                "node_version",
            )
        )
        or runtime.get("observation_protocol") != "XRL1"
        or runtime.get("response_protocol_version") != int(_RESPONSE_VERSION)
        or runtime.get("command_protocol") != _COMMAND_MAGIC.decode("ascii")
        or runtime.get("command_protocol_version") != int(_COMMAND_VERSION)
        or runtime.get("step_api") != "VecScreepsEnv.step_scripted/v1"
        or runtime.get("teacher_alignment")
        != "scripted_actions_apply_to_pre_action_host_obs"
        or runtime.get("trajectory_policy") != OUTPOST_ACTOR_CORPUS_KIND
        or runtime.get("lean_meta") is not False
    ):
        raise ValueError("scripted supplement teacher-forcing provenance differs")
    if verify_source and (
        meta["collection_source_sha256"] != source_signature()
        or meta["collector_source_sha256"] != _collector_source_sha256()
    ):
        raise ValueError("scripted supplement executable source fingerprint differs")
    data = corpus["data"]
    if not isinstance(data, dict) or set(data) != {"rows", "sampling", "collection"}:
        raise ValueError("scripted supplement data keys differ from schema")
    rows = data["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("scripted supplement rows are empty")
    sampling = data["sampling"]
    expected_sampling = {
        "algorithm", "capacity_per_stratum", "rng", "seed", "seen_by_stratum",
        "retained_by_stratum", "seen_by_row_kind", "outpost_coverage", "retained",
    }
    if (
        not isinstance(sampling, dict) or set(sampling) != expected_sampling
        or sampling["algorithm"] != "algorithm_r_per_semantic_stratum"
        or sampling["capacity_per_stratum"] != config.per_stratum
        or sampling["rng"] != "numpy.default_rng/PCG64"
        or sampling["seed"] != config.resolved_reservoir_seed
    ):
        raise ValueError("scripted supplement sampling provenance invalid")
    observed: dict[str, int] = {}
    row_kind_counts = {
        "exact_intent": 0, "spawn_positive": 0, "spawn_wait_legal": 0,
    }
    observed_counts = _empty_outpost_counts()
    observed_seeds = _empty_outpost_seed_sets()
    env_map = {entry["env_index"]: entry for entry in config.env_map()}
    for index, row in enumerate(rows):
        location = f"data.rows[{index}]"
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            raise ValueError(f"{location} keys differ from row ABI")
        kind = row["kind"]
        if kind not in row_kind_counts:
            raise ValueError(f"{location}.kind is invalid")
        timestep = int(row["timestep"])
        env_index = int(row["env_index"])
        actor_index = int(row["actor_index"])
        if not 0 <= timestep < config.steps or env_index not in env_map:
            raise ValueError(f"{location} trajectory reference invalid")
        _room_cap, actor_cap, _target_cap = _validate_obs(row["obs"], f"{location}.obs")
        _validate_action(row["action"], actor_cap, f"{location}.action")
        if not 0 <= actor_index < actor_cap or not bool(row["obs"]["actor_mask"][0, actor_index]):
            raise ValueError(f"{location}.actor_index is not live")
        semantics = _row_semantics(
            row["obs"], row["action"], curriculum=OUTPOST_CURRICULUM,
            timestep=timestep, steps=config.steps, env_index=0,
            actor_index=actor_index,
        )
        if semantics != (kind, row["stratum"]):
            raise ValueError(f"{location} semantic stratum does not match tensors")
        _validate_selected_action_legality(
            row["obs"], row["action"], actor_index=actor_index, location=location,
        )
        if kind == "spawn_positive":
            counts = row["action"]["body_counts"][0, actor_index, 0]
            budget = int(round(
                float(row["obs"]["actors"][
                    0, actor_index, ACTOR_FEATURE_INDEX["roomEnergyAvailable"]
                ]) * MAX_ROOM_ENERGY
            ))
            cost = sum(
                int(counts[part]) * BODY_PART_COSTS[part]
                for part in range(N_BODY_PART)
            )
            if int(counts.sum()) <= 0 or cost > budget:
                raise ValueError(f"{location} positive spawn body is invalid")
        observed[row["stratum"]] = observed.get(row["stratum"], 0) + 1
        row_kind_counts[kind] += 1
        phase, scope = classify_outpost_actor_row(
            row["obs"], row["action"], timestep=timestep, steps=config.steps,
            env_index=0, actor_index=actor_index,
        )
        observed_counts[scope][phase] += 1
        observed_seeds[scope][phase].add(int(env_map[env_index]["seed"]))
    retained_by_stratum = sampling["retained_by_stratum"]
    seen_by_stratum = sampling["seen_by_stratum"]
    if observed != retained_by_stratum:
        raise ValueError("supplement retained_by_stratum differs from rows")
    if not isinstance(seen_by_stratum, dict) or set(seen_by_stratum) != set(observed) or any(
        type(seen_by_stratum[key]) is not int
        or seen_by_stratum[key] < count
        or count > config.per_stratum
        for key, count in observed.items()
    ):
        raise ValueError("supplement reservoir counts are invalid")
    recorded_kinds = sampling["seen_by_row_kind"]
    if not isinstance(recorded_kinds, dict) or set(recorded_kinds) != set(row_kind_counts) or any(
        type(recorded_kinds[kind]) is not int or recorded_kinds[kind] < count
        for kind, count in row_kind_counts.items()
    ):
        raise ValueError("supplement seen_by_row_kind is invalid")
    if sampling["retained"] != len(rows):
        raise ValueError("supplement retained row count differs")
    _validate_coverage(
        sampling["outpost_coverage"], config=config,
        observed_counts=observed_counts, observed_seeds=observed_seeds,
    )
    collection = data["collection"]
    if not isinstance(collection, dict) or set(collection) != {
        "transitions", "teacher_intent_issued", "teacher_intent_invalid",
    }:
        raise ValueError("scripted supplement collection metrics differ")
    if (
        collection["transitions"] != config.steps * config.num_envs
        or type(collection["teacher_intent_issued"]) is not int
        or collection["teacher_intent_issued"] < 0
        or type(collection["teacher_intent_invalid"]) is not int
        or collection["teacher_intent_invalid"] != 0
    ):
        raise ValueError("scripted supplement collection metrics are invalid")
    if verify_hashes:
        unhashed = {key: value for key, value in corpus.items() if key != "integrity"}
        integrity = corpus["integrity"]
        if (
            not isinstance(integrity, dict)
            or integrity.get("algorithm") != "sha256-semantic-v1"
            or integrity.get("corpus_sha256") != content_sha256(unhashed)
        ):
            raise ValueError("scripted supplement semantic SHA-256 mismatch")
    return corpus


def save_outpost_actor_corpus(
    corpus: Mapping[str, Any], output_root: str | Path,
) -> Path:
    validate_outpost_actor_corpus(corpus, verify_source=False)
    content_id = str(corpus["integrity"]["corpus_sha256"])
    root = Path(output_root)
    destination = root / content_id
    if destination.exists():
        loaded = load_outpost_actor_corpus(destination, verify_source=False)
        if loaded["integrity"]["corpus_sha256"] != content_id:
            raise FileExistsError(f"non-identical supplement exists at {destination}")
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
        manifest["manifest_sha256"] = hashlib.sha256(
            _canonical_json(manifest)
        ).hexdigest()
        (temporary / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        os.rename(temporary, destination)
        for path in destination.rglob("*"):
            path.chmod(0o444 if path.is_file() else 0o555)
        destination.chmod(0o555)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_outpost_actor_corpus(
    directory: str | Path, *, verify_hashes: bool = True,
    verify_source: bool = True, expected_base_corpus_id: str | None = None,
) -> dict[str, Any]:
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("scripted supplement directory must not be a symlink")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("scripted supplement manifest must not be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {
        "storage_format", "content_id", "corpus", "shards", "manifest_sha256",
    }:
        raise ValueError("scripted supplement manifest schema invalid")
    recorded_manifest_sha256 = manifest.pop("manifest_sha256")
    if recorded_manifest_sha256 != hashlib.sha256(_canonical_json(manifest)).hexdigest():
        raise ValueError("scripted supplement manifest SHA-256 mismatch")
    if manifest["storage_format"] != "tensor-shards-v1":
        raise ValueError("scripted supplement storage format unsupported")
    records: dict[str, dict[str, Any]] = {}
    for record in manifest["shards"]:
        if not isinstance(record, dict) or set(record) != {
            "file", "sha256", "count", "dtype", "shape",
        }:
            raise ValueError("scripted supplement shard record schema invalid")
        relative = str(record["file"])
        path = Path(relative)
        if (
            len(path.parts) != 2 or path.parts[0] != "shards"
            or path.suffix != ".pt" or not _SHA256_RE.fullmatch(path.stem)
            or record["sha256"] != path.stem or relative in records
        ):
            raise ValueError("scripted supplement shard identity is invalid")
        records[relative] = record
    actual_shards = {
        path.relative_to(root).as_posix() for path in (root / "shards").glob("*.pt")
    }
    if actual_shards != set(records):
        raise ValueError("scripted supplement shard directory differs from manifest")
    actual_entries = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    if actual_entries != {"manifest.json", "shards", *records}:
        raise ValueError("scripted supplement directory contains undeclared entries")
    cache: dict[str, torch.Tensor] = {}
    referenced: set[str] = set()

    def decode(item: Any) -> Any:
        if isinstance(item, dict) and set(item) == {"$tensor", "$index"}:
            relative = str(item["$tensor"])
            record = records.get(relative)
            if record is None:
                raise ValueError("scripted supplement tensor reference is undeclared")
            referenced.add(relative)
            if relative not in cache:
                path = root / relative
                if path.is_symlink():
                    raise ValueError("scripted supplement shards must not be symlinks")
                if verify_hashes and _sha256_file(path) != record["sha256"]:
                    raise ValueError("scripted supplement tensor shard integrity failure")
                payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if (
                    not isinstance(payload, dict)
                    or set(payload) != {"tensor"}
                    or not torch.is_tensor(payload["tensor"])
                ):
                    raise ValueError("scripted supplement tensor shard schema failure")
                tensor = payload["tensor"]
                if (
                    int(tensor.shape[0]) != int(record["count"])
                    or str(tensor.dtype) != record["dtype"]
                    or list(tensor.shape[1:]) != record["shape"]
                ):
                    raise ValueError("scripted supplement tensor shard metadata failure")
                cache[relative] = tensor
            index = int(item["$index"])
            if not 0 <= index < cache[relative].shape[0]:
                raise ValueError("scripted supplement tensor index is out of range")
            return cache[relative][index]
        if isinstance(item, dict):
            return {key: decode(value) for key, value in item.items()}
        if isinstance(item, list):
            return [decode(value) for value in item]
        return item

    corpus = decode(manifest["corpus"])
    if referenced != set(records):
        raise ValueError("scripted supplement manifest has unreferenced shards")
    content_id = str(manifest["content_id"])
    if (
        root.name != content_id
        or corpus.get("integrity", {}).get("corpus_sha256") != content_id
    ):
        raise ValueError("scripted supplement content-addressed identity differs")
    validate_outpost_actor_corpus(
        corpus, verify_hashes=verify_hashes, verify_source=verify_source,
        expected_base_corpus_id=expected_base_corpus_id,
    )
    return corpus


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-corpus-id", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--curriculum", default=OUTPOST_CURRICULUM,
        choices=[OUTPOST_CURRICULUM],
    )
    parser.add_argument("--per-stratum", type=int, default=DEFAULT_PER_STRATUM)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Intra-op parallelism makes environment workers contend; see
    # vec_env.configure_host_threads.
    configure_host_threads()
    args = parse_args(argv)
    config = OutpostActorConfig(
        num_envs=args.num_envs, steps=args.steps, seed=args.seed,
        curriculum=args.curriculum, per_stratum=args.per_stratum,
        device=args.device,
    )
    corpus = collect_outpost_actor_corpus(
        config, base_corpus_id=args.base_corpus_id,
    )
    print(save_outpost_actor_corpus(corpus, args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
