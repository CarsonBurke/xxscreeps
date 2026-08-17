from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import torch

from .artifacts import source_signature
from .constants import (
    ACTOR_FEAT, ACTOR_FEATURE_INDEX, CONSTRUCTION_MASK_BYTES, GLOBAL_FEAT,
    INTENT_SLOTS, INTENT_TYPES, MAX_ROOMS, N_AMOUNT, N_BODY_PART,
    N_CONSTRUCTION_TYPE, PATCHES_PER_ROOM, PATCH_FLAT, SCHEMA, SCHEMA_SHA256,
    TARGET_FEAT,
)
from .dagger_corpus import (
    ACTION_KEYS, OUTPOST_COVERAGE_CONTRACT, OUTPOST_MIN_DISTINCT_LATE_SEEDS,
    OUTPOST_MIN_RETAINED_LATE, OUTPOST_MIN_RETAINED_OVERALL,
    TARGET_KIND_FEATURE_INDEX, TARGET_ROOM_FEATURE_INDEX, _clone_row,
    _empty_outpost_counts, _empty_outpost_seed_sets, _row_semantics,
    _serialized_seed_sets,
)
from .outpost_actor_corpus import (
    OUTPOST_ACTOR_CORPUS_KIND, OutpostActorConfig,
    _collector_source_sha256, _runtime_provenance, assemble_outpost_actor_corpus,
    classify_outpost_actor_row, collect_outpost_actor_rows,
    load_outpost_actor_corpus, save_outpost_actor_corpus,
    validate_outpost_actor_corpus,
)


def _obs() -> dict[str, torch.Tensor]:
    actors, targets, rooms = 8, 16, 2
    obs = {
        "patches": torch.zeros(
            1, rooms, PATCHES_PER_ROOM, PATCH_FLAT, dtype=torch.uint8,
        ),
        "room_mask": torch.ones(1, rooms, dtype=torch.uint8),
        "room_coords": torch.zeros(1, rooms, 2),
        "actors": torch.zeros(1, actors, ACTOR_FEAT),
        "actor_mask": torch.zeros(1, actors, dtype=torch.uint8),
        "actor_outcome": torch.zeros(1, actors, dtype=torch.uint8),
        "targets": torch.zeros(1, targets, TARGET_FEAT),
        "target_mask": torch.zeros(1, targets, dtype=torch.uint8),
        "intent_mask": torch.zeros(
            1, actors, INTENT_SLOTS, len(INTENT_TYPES), dtype=torch.uint8,
        ),
        "dir_mask": torch.ones(1, actors, INTENT_SLOTS, 8, dtype=torch.uint8),
        "target_select_mask": torch.zeros(
            1, len(INTENT_TYPES), targets, dtype=torch.uint8,
        ),
        "amount_mask": torch.ones(
            1, actors, INTENT_SLOTS, len(INTENT_TYPES), N_AMOUNT,
            dtype=torch.uint8,
        ),
        "construction_mask": torch.zeros(
            1, rooms, N_CONSTRUCTION_TYPE, CONSTRUCTION_MASK_BYTES,
            dtype=torch.uint8,
        ),
        "globals": torch.zeros(1, GLOBAL_FEAT),
    }
    obs["actor_mask"][0, 0] = 1
    obs["actors"][0, 0, ACTOR_FEATURE_INDEX["roomIndex"]] = 1 / (MAX_ROOMS - 1)
    obs["actors"][0, 0, ACTOR_FEATURE_INDEX["activeMove"]] = 1 / 50
    obs["target_mask"][0, 0] = 1
    obs["globals"][0, 6] = 2 / 16
    return obs


def _action(intent_name: str) -> dict[str, torch.Tensor]:
    actors = 8
    action = {
        key: torch.zeros(1, actors, INTENT_SLOTS, dtype=torch.long)
        for key in (
            "types", "dirs", "targets", "amounts", "construction_types",
            "construction_tiles",
        )
    }
    action["body_counts"] = torch.zeros(
        1, actors, INTENT_SLOTS, N_BODY_PART, dtype=torch.long,
    )
    action["body_order"] = torch.arange(N_BODY_PART).view(
        1, 1, 1, N_BODY_PART,
    ).expand(1, actors, INTENT_SLOTS, N_BODY_PART).clone()
    action["types"][0, 0, 0] = INTENT_TYPES.index(intent_name)
    return action


def _label(scope: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    obs = _obs()
    intent_name = "harvest" if scope == "remote_harvest" else "transfer"
    intent = INTENT_TYPES.index(intent_name)
    action = _action(intent_name)
    obs["intent_mask"][0, 0, 0, intent] = 1
    obs["target_select_mask"][0, intent, 0] = 1
    obs["targets"][0, 0, TARGET_KIND_FEATURE_INDEX] = (
        2 if scope == "remote_harvest" else 4
    ) / 6
    obs["targets"][0, 0, TARGET_ROOM_FEATURE_INDEX] = (
        1 / (MAX_ROOMS - 1) if scope == "remote_harvest" else 0
    )
    return obs, action


def _meta(config: OutpostActorConfig) -> dict:
    return {
        **config.__dict__,
        "max_episode": config.resolved_max_episode,
        "reservoir_seed": config.resolved_reservoir_seed,
        "env_map": config.env_map(),
        "supplement_type": OUTPOST_ACTOR_CORPUS_KIND,
        "base_corpus_sha256": "a" * 64,
        "collection_source_sha256": source_signature(),
        "collector_source_sha256": _collector_source_sha256(),
        "schema_sha256": SCHEMA_SHA256,
        "environment_schema_version": SCHEMA["version"],
        "contracts": dict(SCHEMA["artifact"]),
        "runtime": _runtime_provenance(config),
    }


def _valid_corpus() -> tuple[OutpostActorConfig, dict]:
    config = OutpostActorConfig(
        num_envs=3, steps=10, seed=40_000, per_stratum=64,
        reservoir_seed=31,
    )
    rows = []
    counts = _empty_outpost_counts()
    seeds = _empty_outpost_seed_sets()
    for scope in ("remote_harvest", "homebound_transfer"):
        obs, action = _label(scope)
        for phase, timestep in (("early", 0), ("late", 8)):
            for offset in range(32):
                env_index = offset % 3
                semantics = _row_semantics(
                    obs, action, curriculum="seed_outpost", timestep=timestep,
                    steps=config.steps, env_index=0, actor_index=0,
                )
                assert semantics is not None
                kind, stratum = semantics
                row = _clone_row(
                    obs, action, row_kind=kind, stratum=stratum,
                    timestep=timestep, env_index=0, actor_index=0,
                )
                row["env_index"] = env_index
                rows.append(row)
                counts[scope][phase] += 1
                seeds[scope][phase].add(config.seed + env_index)
    rows.sort(key=lambda row: (
        row["stratum"], row["timestep"], row["env_index"], row["actor_index"],
    ))
    retained_by_stratum = {
        stratum: sum(row["stratum"] == stratum for row in rows)
        for stratum in sorted({row["stratum"] for row in rows})
    }
    sampling = {
        "algorithm": "algorithm_r_per_semantic_stratum",
        "capacity_per_stratum": 64,
        "rng": "numpy.default_rng/PCG64",
        "seed": 31,
        "seen_by_stratum": retained_by_stratum,
        "retained_by_stratum": retained_by_stratum,
        "seen_by_row_kind": {
            "exact_intent": len(rows), "spawn_positive": 0,
            "spawn_wait_legal": 0,
        },
        "outpost_coverage": {
            "contract": OUTPOST_COVERAGE_CONTRACT,
            "late_window_size": 2,
            "readiness_targets": {
                "minimum_retained_overall": OUTPOST_MIN_RETAINED_OVERALL,
                "minimum_retained_late": OUTPOST_MIN_RETAINED_LATE,
                "minimum_distinct_late_seeds": OUTPOST_MIN_DISTINCT_LATE_SEEDS,
            },
            "seen_by_scope_phase": copy.deepcopy(counts),
            "retained_by_scope_phase": copy.deepcopy(counts),
            "seen_seeds_by_scope_phase": _serialized_seed_sets(seeds),
            "retained_seeds_by_scope_phase": _serialized_seed_sets(seeds),
        },
        "retained": len(rows),
    }
    collection = {
        "transitions": 30,
        "teacher_intent_issued": 128,
        "teacher_intent_invalid": 0,
    }
    return config, assemble_outpost_actor_corpus(
        _meta(config), rows, sampling, collection,
    )


def _sync(config: OutpostActorConfig, corpus: dict) -> None:
    rows = corpus["data"]["rows"]
    sampling = corpus["data"]["sampling"]
    counts = _empty_outpost_counts()
    seeds = _empty_outpost_seed_sets()
    for row in rows:
        phase, scope = classify_outpost_actor_row(
            row["obs"], row["action"], timestep=row["timestep"],
            steps=config.steps, env_index=0, actor_index=row["actor_index"],
        )
        counts[scope][phase] += 1
        seeds[scope][phase].add(config.seed + row["env_index"])
    retained = {
        stratum: sum(row["stratum"] == stratum for row in rows)
        for stratum in sorted({row["stratum"] for row in rows})
    }
    sampling["seen_by_stratum"] = retained
    sampling["retained_by_stratum"] = retained
    sampling["seen_by_row_kind"]["exact_intent"] = len(rows)
    sampling["retained"] = len(rows)
    coverage = sampling["outpost_coverage"]
    coverage["seen_by_scope_phase"] = copy.deepcopy(counts)
    coverage["retained_by_scope_phase"] = copy.deepcopy(counts)
    coverage["seen_seeds_by_scope_phase"] = _serialized_seed_sets(seeds)
    coverage["retained_seeds_by_scope_phase"] = _serialized_seed_sets(seeds)


def test_exact_dagger_v2_scope_and_phase_classification():
    remote_obs, remote_action = _label("remote_harvest")
    assert classify_outpost_actor_row(
        remote_obs, remote_action, timestep=7, steps=10,
        env_index=0, actor_index=0,
    ) == ("early", "remote_harvest")
    assert classify_outpost_actor_row(
        remote_obs, remote_action, timestep=8, steps=10,
        env_index=0, actor_index=0,
    ) == ("late", "remote_harvest")
    home_obs, home_action = _label("homebound_transfer")
    assert classify_outpost_actor_row(
        home_obs, home_action, timestep=8, steps=10,
        env_index=0, actor_index=0,
    ) == ("late", "homebound_transfer")
    home_obs["actors"][0, 0, ACTOR_FEATURE_INDEX["roomIndex"]] = 0
    assert classify_outpost_actor_row(
        home_obs, home_action, timestep=8, steps=10,
        env_index=0, actor_index=0,
    ) == ("late", "local_other")


def test_collection_uses_scripted_teacher_step_and_preserves_pre_state():
    class FakeVec:
        host_obs = None
        calls = 0

        def reset(self):
            self.host_obs, _action_value = _label("remote_harvest")
            return self.host_obs

        def step_scripted(self):
            pre_marker = float(self.host_obs["globals"][0, 1])
            next_obs, teacher = _label("remote_harvest")
            next_obs["globals"][0, 1] = pre_marker + 0.1
            self.host_obs = next_obs
            self.calls += 1
            return next_obs, torch.zeros(1), torch.zeros(1), [{}], teacher

    config = OutpostActorConfig(
        num_envs=1, steps=10, seed=41_000, reservoir_seed=37,
    )
    envs = FakeVec()
    rows, sampling, collection = collect_outpost_actor_rows(envs, config)
    assert envs.calls == 10
    assert collection["transitions"] == 10
    remote = sampling["outpost_coverage"]["seen_by_scope_phase"]["remote_harvest"]
    assert remote == {"early": 8, "late": 2}
    first = min(rows, key=lambda row: row["timestep"])
    assert float(first["obs"]["globals"][0, 1]) == 0


def test_collection_rejects_invalid_scripted_teacher_intent_with_provenance():
    class InvalidTeacherVec:
        host_obs = None

        def reset(self):
            self.host_obs, _action_value = _label("remote_harvest")
            return self.host_obs

        def step_scripted(self):
            next_obs, teacher = _label("remote_harvest")
            self.host_obs = next_obs
            info = {
                "curriculum": "seed_outpost",
                "intentInvalid": 1,
                "intentByType": {"harvest": 1},
                "intentResults": [{
                    "actor": 0, "type": "harvest", "code": -7,
                    "err": "ERR_INVALID_TARGET", "executed": False,
                }],
            }
            return next_obs, torch.zeros(1), torch.zeros(1), [info], teacher

    config = OutpostActorConfig(
        num_envs=1, steps=1, seed=41_500, reservoir_seed=39,
    )
    with pytest.raises(RuntimeError, match=r"invalid intent at step=1") as error:
        collect_outpost_actor_rows(InvalidTeacherVec(), config)
    assert '"seed":41500' in str(error.value)
    assert "ERR_INVALID_TARGET" in str(error.value)


def test_content_addressed_roundtrip_and_provenance_rejection():
    _config, corpus = _valid_corpus()
    assert corpus["kind"] == OUTPOST_ACTOR_CORPUS_KIND
    with tempfile.TemporaryDirectory() as temporary:
        destination = save_outpost_actor_corpus(corpus, Path(temporary))
        loaded = load_outpost_actor_corpus(
            destination, expected_base_corpus_id="a" * 64,
        )
        assert loaded["data"]["rows"][0].keys() == corpus["data"]["rows"][0].keys()
        assert loaded["integrity"] == corpus["integrity"]
        shard = next((destination / "shards").glob("*.pt"))
        shard.chmod(0o644)
        payload = bytearray(shard.read_bytes())
        payload[len(payload) // 2] ^= 1
        shard.write_bytes(payload)
        with pytest.raises(ValueError, match="shard integrity failure"):
            load_outpost_actor_corpus(destination)

    bad_base = copy.deepcopy(corpus)
    bad_base["meta"]["base_corpus_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="base corpus differs"):
        validate_outpost_actor_corpus(
            bad_base, verify_hashes=False, expected_base_corpus_id="a" * 64,
        )
    bad_authority = copy.deepcopy(corpus)
    bad_authority["meta"]["supplement_type"] = "dagger_correction_corpus"
    with pytest.raises(ValueError, match="trajectory authority"):
        validate_outpost_actor_corpus(bad_authority, verify_hashes=False)
    bad_source = copy.deepcopy(corpus)
    bad_source["meta"]["collection_source_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="source fingerprint"):
        validate_outpost_actor_corpus(bad_source, verify_hashes=False)


def test_corrupt_coverage_and_readiness_are_rejected():
    config, corpus = _valid_corpus()
    corrupt = copy.deepcopy(corpus)
    corrupt["data"]["sampling"]["outpost_coverage"][
        "retained_by_scope_phase"
    ]["remote_harvest"]["late"] += 1
    with pytest.raises(ValueError, match="coverage differs from rows"):
        validate_outpost_actor_corpus(corrupt, verify_hashes=False)

    invalid_metric = copy.deepcopy(corpus)
    invalid_metric["data"]["collection"]["teacher_intent_invalid"] = 1
    with pytest.raises(ValueError, match="collection metrics are invalid"):
        validate_outpost_actor_corpus(invalid_metric, verify_hashes=False)

    insufficient = copy.deepcopy(corpus)
    index = next(
        index for index, row in enumerate(insufficient["data"]["rows"])
        if "phase=late|scope=homebound_transfer" in row["stratum"]
    )
    insufficient["data"]["rows"].pop(index)
    _sync(config, insufficient)
    with pytest.raises(
        ValueError,
        match=r"homebound_transfer\.overall retained=63.*late retained=31",
    ):
        validate_outpost_actor_corpus(insufficient, verify_hashes=False)

    one_seed = copy.deepcopy(corpus)
    for row in one_seed["data"]["rows"]:
        if "phase=late|scope=remote_harvest" in row["stratum"]:
            row["env_index"] = 0
    _sync(config, one_seed)
    with pytest.raises(
        ValueError, match=r"remote_harvest\.late_seeds retained=1.*required=3",
    ):
        validate_outpost_actor_corpus(one_seed, verify_hashes=False)


def test_curriculum_is_fixed_to_seed_outpost():
    with pytest.raises(ValueError, match="must be seed_outpost"):
        OutpostActorConfig(
            num_envs=3, steps=10, seed=1, curriculum="empty",
        ).validate()
