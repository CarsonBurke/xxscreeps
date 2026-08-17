from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from .artifacts import artifact_meta, source_signature
from .constants import (
    ACTOR_FEAT,
    ACTOR_FEATURE_INDEX,
    CONSTRUCTION_MASK_BYTES,
    GLOBAL_FEAT,
    INTENT_SLOTS,
    INTENT_TYPES,
    MAX_ROOM_ENERGY,
    MAX_ROOMS,
    N_AMOUNT,
    N_BODY_PART,
    N_CONSTRUCTION_TYPE,
    PATCHES_PER_ROOM,
    PATCH_FLAT,
    SCHEMA,
    SCHEMA_SHA256,
    TARGET_FEAT,
)
from .dagger_corpus import (
    ACTION_KEYS,
    DaggerConfig,
    OUTPOST_COVERAGE_CONTRACT,
    OUTPOST_MIN_DISTINCT_LATE_SEEDS,
    OUTPOST_MIN_RETAINED_LATE,
    OUTPOST_MIN_RETAINED_OVERALL,
    TARGET_KIND_FEATURE_INDEX,
    TARGET_ROOM_FEATURE_INDEX,
    _POLICY_TEACHER_ABI_BRIDGE,
    _clone_row,
    _collector_source_sha256,
    _empty_outpost_counts,
    _empty_outpost_seed_sets,
    _load_joint_actor,
    _outpost_phase_scope,
    _row_semantics,
    _serialized_seed_sets,
    assemble_dagger_corpus,
    collect_rows,
    load_dagger_corpus,
    save_dagger_corpus,
    validate_dagger_corpus,
)
from .model import Actor, Critic


def _obs() -> dict[str, torch.Tensor]:
    actors, targets, rooms = 8, 16, 1
    obs = {
        "patches": torch.zeros(1, rooms, PATCHES_PER_ROOM, PATCH_FLAT, dtype=torch.uint8),
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
            1, actors, INTENT_SLOTS, len(INTENT_TYPES), N_AMOUNT, dtype=torch.uint8,
        ),
        "construction_mask": torch.zeros(
            1, rooms, N_CONSTRUCTION_TYPE, CONSTRUCTION_MASK_BYTES,
            dtype=torch.uint8,
        ),
        "globals": torch.zeros(1, GLOBAL_FEAT),
    }
    none = INTENT_TYPES.index("none")
    spawn = INTENT_TYPES.index("spawnCreep")
    upgrade = INTENT_TYPES.index("upgradeController")
    obs["actor_mask"][0, :2] = 1
    obs["actors"][0, 0, ACTOR_FEATURE_INDEX["isNonCreep"]] = 1
    obs["actors"][0, 0, ACTOR_FEATURE_INDEX["isSpawn"]] = 1
    obs["actors"][0, 0, ACTOR_FEATURE_INDEX["roomEnergyAvailable"]] = (
        650 / MAX_ROOM_ENERGY
    )
    obs["target_mask"][0, 0] = 1
    obs["targets"][0, 0, 0] = 2 / 6
    obs["intent_mask"][0, 0, 0, none] = 1
    obs["intent_mask"][0, 0, 0, spawn] = 1
    obs["intent_mask"][0, 1, 0, none] = 1
    obs["intent_mask"][0, 1, 0, upgrade] = 1
    obs["target_select_mask"][0, upgrade, 0] = 1
    obs["globals"][0, 0] = 2 / 8
    obs["globals"][0, 3] = 8 / 50
    obs["globals"][0, 6] = 1 / 16
    return obs


def _action(*, positive_spawn: bool) -> dict[str, torch.Tensor]:
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
    action["types"][0, 1, 0] = INTENT_TYPES.index("upgradeController")
    if positive_spawn:
        action["types"][0, 0, 0] = INTENT_TYPES.index("spawnCreep")
        action["body_counts"][0, 0, 0, 0] = 1
        action["body_counts"][0, 0, 0, 6] = 1
        action["body_order"][0, 0, 0] = torch.tensor([6, 0, 1, 2, 3, 4, 5, 7])
    return action


class _FakeActor:
    def __call__(self, obs, deterministic=False):
        del deterministic
        action = _action(positive_spawn=False)
        action = {key: value.expand(obs["actors"].shape[0], *value.shape[1:]) for key, value in action.items()}
        return SimpleNamespace(**action)


class _FakeVec:
    def __init__(self):
        self.host_obs = None
        self._step = 0

    def reset(self):
        self.host_obs = _obs()
        return self.host_obs

    def step_labeled(self, learner):
        assert set(learner) == ACTION_KEYS
        teacher = _action(positive_spawn=self._step == 1)
        self._step += 1
        self.host_obs = _obs()
        return (
            self.host_obs,
            torch.zeros(1),
            torch.zeros(1),
            [{"intentIssued": 2, "intentInvalid": 0}],
            teacher,
        )


def _meta(config: DaggerConfig) -> dict:
    fake_sha = "a" * 64
    return {
        **config.__dict__,
        "max_episode": config.resolved_max_episode,
        "reservoir_seed": config.resolved_reservoir_seed,
        "env_map": config.env_map(),
        "base_corpus_sha256": fake_sha,
        "checkpoint_file_sha256": "b" * 64,
        "checkpoint_kind": "joint_pretrain",
        "checkpoint_qualified": False,
        "checkpoint_partial": False,
        "checkpoint_global_epoch": 16,
        "checkpoint_global_epochs": 16,
        "checkpoint_schema_version": SCHEMA["version"],
        "checkpoint_schema_sha256": SCHEMA_SHA256,
        "checkpoint_contracts": dict(SCHEMA["artifact"]),
        "checkpoint_source_sha256": source_signature(),
        "source_sha256": source_signature(),
        "policy_source_mismatch": False,
        "policy_source_mismatch_allowed": False,
        "checkpoint_teacher_abi": SCHEMA["artifact"]["teacherAbi"],
        "current_teacher_abi": SCHEMA["artifact"]["teacherAbi"],
        "policy_teacher_abi_mismatch": False,
        "policy_teacher_abi_mismatch_allowed": False,
        "collector_source_sha256": _collector_source_sha256(),
        "schema_sha256": SCHEMA_SHA256,
        "environment_schema_version": SCHEMA["version"],
        "contracts": dict(SCHEMA["artifact"]),
        "runtime": {
            "step_api": "VecScreepsEnv.step_labeled/v1",
            "teacher_alignment": "teacher_actions_label_pre_action_host_obs",
        },
    }


def _synthetic_corpus():
    config = DaggerConfig(
        num_envs=1, steps=2, seed=20_003, max_episode=3,
        per_stratum=1, reservoir_seed=19,
    )
    rows, sampling, collection = collect_rows(_FakeActor(), _FakeVec(), config)
    return config, assemble_dagger_corpus(_meta(config), rows, sampling, collection)


def _outpost_label(scope: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return one legal mobile-creep label for a named outpost scope."""
    if scope not in {"remote_harvest", "homebound_transfer"}:
        raise ValueError(f"unsupported test scope: {scope}")
    obs = _obs()
    obs["patches"] = torch.zeros(
        1, 2, PATCHES_PER_ROOM, PATCH_FLAT, dtype=torch.uint8,
    )
    obs["room_mask"] = torch.ones(1, 2, dtype=torch.uint8)
    obs["room_coords"] = torch.zeros(1, 2, 2)
    obs["construction_mask"] = torch.zeros(
        1, 2, N_CONSTRUCTION_TYPE, CONSTRUCTION_MASK_BYTES, dtype=torch.uint8,
    )
    obs["globals"][0, 6] = 2 / 16
    obs["actors"][0, 1, ACTOR_FEATURE_INDEX["roomIndex"]] = 1 / (MAX_ROOMS - 1)
    obs["actors"][0, 1, ACTOR_FEATURE_INDEX["activeMove"]] = 1 / 50

    intent_name = "harvest" if scope == "remote_harvest" else "transfer"
    intent = INTENT_TYPES.index(intent_name)
    obs["intent_mask"][0, 1, 0].zero_()
    obs["intent_mask"][0, 1, 0, intent] = 1
    obs["target_select_mask"][0, intent, 0] = 1
    obs["targets"][0, 0, TARGET_KIND_FEATURE_INDEX] = (
        2 if scope == "remote_harvest" else 4
    ) / 6
    obs["targets"][0, 0, TARGET_ROOM_FEATURE_INDEX] = (
        1 / (MAX_ROOMS - 1) if scope == "remote_harvest" else 0
    )
    action = _action(positive_spawn=False)
    action["types"][0, 1, 0] = intent
    action["targets"][0, 1, 0] = 0
    return obs, action


def _outpost_corpus() -> tuple[DaggerConfig, dict]:
    config = DaggerConfig(
        num_envs=3, steps=10, seed=30_000, curriculum="seed_outpost",
        max_episode=11, per_stratum=64, reservoir_seed=23,
    )
    rows = []
    for scope in ("remote_harvest", "homebound_transfer"):
        obs, action = _outpost_label(scope)
        for phase, timestep in (("early", 0), ("late", 8)):
            for offset in range(32):
                env_index = offset % config.num_envs
                semantics = _row_semantics(
                    obs, action, curriculum="seed_outpost", timestep=timestep,
                    steps=config.steps, env_index=0, actor_index=1,
                )
                assert semantics is not None
                kind, stratum = semantics
                assert f"phase={phase}|scope={scope}" in stratum
                row = _clone_row(
                    obs, action, row_kind=kind, stratum=stratum,
                    timestep=timestep, env_index=0, actor_index=1,
                )
                row["env_index"] = env_index
                rows.append(row)
    rows.sort(key=lambda row: (
        row["stratum"], row["timestep"], row["env_index"], row["actor_index"],
    ))
    retained_by_stratum = {
        stratum: sum(row["stratum"] == stratum for row in rows)
        for stratum in sorted({row["stratum"] for row in rows})
    }
    counts = _empty_outpost_counts()
    seed_sets = _empty_outpost_seed_sets()
    env_map = {entry["env_index"]: entry for entry in config.env_map()}
    for row in rows:
        phase, scope = _outpost_phase_scope(
            row["obs"], row["action"], curriculum="seed_outpost",
            timestep=row["timestep"], steps=config.steps,
            env_index=0, actor_index=row["actor_index"],
        )
        counts[scope][phase] += 1
        seed_sets[scope][phase].add(env_map[row["env_index"]]["seed"])
    sampling = {
        "algorithm": "algorithm_r_per_semantic_stratum",
        "capacity_per_stratum": config.per_stratum,
        "rng": "numpy.default_rng/PCG64",
        "seed": config.resolved_reservoir_seed,
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
            "seen_seeds_by_scope_phase": _serialized_seed_sets(seed_sets),
            "retained_seeds_by_scope_phase": _serialized_seed_sets(seed_sets),
        },
        "retained": len(rows),
    }
    collection = {
        "transitions": config.steps * config.num_envs,
        "learner_intent_issued": 0,
        "learner_intent_invalid": 0,
    }
    return config, assemble_dagger_corpus(
        _meta(config), rows, sampling, collection,
    )


def _synchronize_outpost_sampling(config: DaggerConfig, corpus: dict) -> None:
    """Recompute exact sampling provenance after a deliberate test mutation."""
    rows = corpus["data"]["rows"]
    sampling = corpus["data"]["sampling"]
    retained_by_stratum = {
        stratum: sum(row["stratum"] == stratum for row in rows)
        for stratum in sorted({row["stratum"] for row in rows})
    }
    counts = _empty_outpost_counts()
    seed_sets = _empty_outpost_seed_sets()
    env_map = {entry["env_index"]: entry for entry in config.env_map()}
    for row in rows:
        classified = _outpost_phase_scope(
            row["obs"], row["action"],
            curriculum=env_map[row["env_index"]]["curriculum"],
            timestep=row["timestep"], steps=config.steps,
            env_index=0, actor_index=row["actor_index"],
        )
        assert classified is not None
        phase, scope = classified
        counts[scope][phase] += 1
        seed_sets[scope][phase].add(env_map[row["env_index"]]["seed"])
    sampling["seen_by_stratum"] = retained_by_stratum
    sampling["retained_by_stratum"] = retained_by_stratum
    sampling["seen_by_row_kind"]["exact_intent"] = len(rows)
    sampling["retained"] = len(rows)
    coverage = sampling["outpost_coverage"]
    coverage["seen_by_scope_phase"] = copy.deepcopy(counts)
    coverage["retained_by_scope_phase"] = copy.deepcopy(counts)
    coverage["seen_seeds_by_scope_phase"] = _serialized_seed_sets(seed_sets)
    coverage["retained_seeds_by_scope_phase"] = _serialized_seed_sets(seed_sets)


def test_collection_keeps_exact_intents_positive_spawns_and_legal_waits():
    config, corpus = _synthetic_corpus()
    rows = corpus["data"]["rows"]
    assert {row["kind"] for row in rows} == {
        "exact_intent", "spawn_positive", "spawn_wait_legal",
    }
    assert sum(row["kind"] == "exact_intent" for row in rows) == 1
    assert corpus["data"]["sampling"]["seen_by_row_kind"] == {
        "exact_intent": 2, "spawn_positive": 1, "spawn_wait_legal": 1,
    }
    assert all(count <= config.per_stratum for count in (
        corpus["data"]["sampling"]["retained_by_stratum"].values()
    ))
    assert corpus["meta"]["env_map"] == [
        {"env_index": 0, "seed": 20_003, "curriculum": "empty"},
    ]
    validate_dagger_corpus(corpus)


def test_outpost_scope_classification_is_teacher_and_actor_semantic():
    remote_obs, remote_action = _outpost_label("remote_harvest")
    assert _outpost_phase_scope(
        remote_obs, remote_action, curriculum="seed_outpost", timestep=0,
        steps=10, env_index=0, actor_index=1,
    ) == ("early", "remote_harvest")

    home_obs, home_action = _outpost_label("homebound_transfer")
    assert _outpost_phase_scope(
        home_obs, home_action, curriculum="seed_outpost", timestep=8,
        steps=10, env_index=0, actor_index=1,
    ) == ("late", "homebound_transfer")

    local_actor = copy.deepcopy(home_obs)
    local_actor["actors"][0, 1, ACTOR_FEATURE_INDEX["roomIndex"]] = 0
    assert _outpost_phase_scope(
        local_actor, home_action, curriculum="seed_outpost", timestep=8,
        steps=10, env_index=0, actor_index=1,
    ) == ("late", "local_other")

    local_target = copy.deepcopy(remote_obs)
    local_target["targets"][0, 0, TARGET_ROOM_FEATURE_INDEX] = 0
    assert _outpost_phase_scope(
        local_target, remote_action, curriculum="seed_outpost", timestep=0,
        steps=10, env_index=0, actor_index=1,
    ) == ("early", "local_other")
    assert _outpost_phase_scope(
        remote_obs, remote_action, curriculum="empty", timestep=0,
        steps=10, env_index=0, actor_index=1,
    ) is None


def test_outpost_phase_boundary_matches_training_late_window():
    obs, action = _outpost_label("remote_harvest")
    early = _row_semantics(
        obs, action, curriculum="seed_outpost", timestep=7, steps=10,
        env_index=0, actor_index=1,
    )
    late = _row_semantics(
        obs, action, curriculum="seed_outpost", timestep=8, steps=10,
        env_index=0, actor_index=1,
    )
    assert early is not None and "phase=early|scope=remote_harvest" in early[1]
    assert late is not None and "phase=late|scope=remote_harvest" in late[1]


def test_outpost_collection_persists_seen_and_retained_phase_scope_counts():
    class FakeOutpostVec:
        host_obs = None

        def reset(self):
            self.host_obs, _action_value = _outpost_label("remote_harvest")
            return self.host_obs

        def step_labeled(self, learner):
            assert set(learner) == ACTION_KEYS
            next_obs, teacher = _outpost_label("remote_harvest")
            self.host_obs = next_obs
            return next_obs, torch.zeros(1), torch.zeros(1), [{}], teacher

    config = DaggerConfig(
        num_envs=1, steps=10, seed=31_000, curriculum="seed_outpost",
        max_episode=11, per_stratum=64, reservoir_seed=29,
    )
    _rows, sampling, _collection = collect_rows(
        _FakeActor(), FakeOutpostVec(), config,
    )
    coverage = sampling["outpost_coverage"]
    assert coverage["seen_by_scope_phase"]["remote_harvest"] == {
        "early": 8, "late": 2,
    }
    assert coverage["retained_by_scope_phase"]["remote_harvest"] == {
        "early": 8, "late": 2,
    }
    assert coverage["seen_seeds_by_scope_phase"]["remote_harvest"] == {
        "early": [31_000], "late": [31_000],
    }
    assert coverage["retained_seeds_by_scope_phase"]["remote_harvest"] == {
        "early": [31_000], "late": [31_000],
    }


def test_outpost_v2_coverage_is_exact_and_corruption_is_rejected():
    _config, corpus = _outpost_corpus()
    validate_dagger_corpus(corpus)
    coverage = corpus["data"]["sampling"]["outpost_coverage"]
    assert coverage["retained_by_scope_phase"]["remote_harvest"] == {
        "early": 32, "late": 32,
    }
    assert coverage["retained_by_scope_phase"]["homebound_transfer"] == {
        "early": 32, "late": 32,
    }
    assert coverage["retained_seeds_by_scope_phase"]["remote_harvest"]["late"] == [
        30_000, 30_001, 30_002,
    ]

    corrupt_count = copy.deepcopy(corpus)
    corrupt_count["data"]["sampling"]["outpost_coverage"][
        "retained_by_scope_phase"
    ]["remote_harvest"]["late"] += 1
    with pytest.raises(ValueError, match="retained outpost coverage differs"):
        validate_dagger_corpus(corrupt_count, verify_hashes=False)

    corrupt_phase = copy.deepcopy(corpus)
    remote_late = next(
        row for row in corrupt_phase["data"]["rows"]
        if "phase=late|scope=remote_harvest" in row["stratum"]
    )
    remote_late["stratum"] = remote_late["stratum"].replace(
        "phase=late", "phase=early",
    )
    with pytest.raises(ValueError, match="semantic stratum"):
        validate_dagger_corpus(corrupt_phase, verify_hashes=False)

    missing = copy.deepcopy(corpus)
    del missing["data"]["sampling"]["outpost_coverage"]
    with pytest.raises(ValueError, match="sampling provenance invalid"):
        validate_dagger_corpus(missing, verify_hashes=False)


def test_outpost_v2_rejects_insufficient_rows_and_late_seed_diversity():
    config, corpus = _outpost_corpus()
    insufficient = copy.deepcopy(corpus)
    removed = next(
        index for index, row in enumerate(insufficient["data"]["rows"])
        if "phase=late|scope=remote_harvest" in row["stratum"]
    )
    insufficient["data"]["rows"].pop(removed)
    _synchronize_outpost_sampling(config, insufficient)
    with pytest.raises(
        ValueError,
        match=r"remote_harvest\.overall retained=63.*remote_harvest\.late retained=31",
    ):
        validate_dagger_corpus(insufficient, verify_hashes=False)

    one_late_seed = copy.deepcopy(corpus)
    for row in one_late_seed["data"]["rows"]:
        if "phase=late|scope=homebound_transfer" in row["stratum"]:
            row["env_index"] = 0
    _synchronize_outpost_sampling(config, one_late_seed)
    with pytest.raises(
        ValueError,
        match=r"homebound_transfer\.late_seeds retained=1.*required=3",
    ):
        validate_dagger_corpus(one_late_seed, verify_hashes=False)


def test_content_addressed_weights_only_roundtrip_and_corruption_rejection():
    _config, corpus = _synthetic_corpus()
    with tempfile.TemporaryDirectory() as temporary:
        destination = save_dagger_corpus(corpus, Path(temporary))
        assert destination.name == corpus["integrity"]["corpus_sha256"]
        loaded = load_dagger_corpus(
            destination,
            expected_base_corpus_id="a" * 64,
            expected_checkpoint_sha256="b" * 64,
        )
        assert loaded["data"]["sampling"] == corpus["data"]["sampling"]
        destination.chmod(0o755)
        extra = destination / "undeclared.txt"
        extra.write_text("not part of the artifact", encoding="utf-8")
        with pytest.raises(ValueError, match="undeclared entries"):
            load_dagger_corpus(destination)
        extra.unlink()
        shard = next((destination / "shards").glob("*.pt"))
        shard.chmod(0o644)
        payload = bytearray(shard.read_bytes())
        payload[len(payload) // 2] ^= 1
        shard.write_bytes(payload)
        with pytest.raises(ValueError, match="shard integrity failure"):
            load_dagger_corpus(destination)


def test_semantic_validation_rejects_non_spawn_none_rows():
    _config, corpus = _synthetic_corpus()
    broken = copy.deepcopy(corpus)
    row = next(row for row in broken["data"]["rows"] if row["kind"] == "exact_intent")
    row["action"]["types"][0, row["actor_index"], 0] = INTENT_TYPES.index("none")
    with pytest.raises(ValueError, match="semantic stratum"):
        validate_dagger_corpus(broken, verify_hashes=False)


def test_semantic_validation_rejects_illegal_active_factor_and_body_order():
    _config, corpus = _synthetic_corpus()
    illegal_target = copy.deepcopy(corpus)
    row = next(
        row for row in illegal_target["data"]["rows"]
        if row["kind"] == "exact_intent"
    )
    intent = int(row["action"]["types"][0, row["actor_index"], 0])
    target = int(row["action"]["targets"][0, row["actor_index"], 0])
    row["obs"]["target_select_mask"][0, intent, target] = 0
    with pytest.raises(ValueError, match="teacher target is illegal"):
        validate_dagger_corpus(illegal_target, verify_hashes=False)

    incompatible_target = copy.deepcopy(corpus)
    row = next(
        row for row in incompatible_target["data"]["rows"]
        if row["kind"] == "exact_intent"
    )
    row["obs"]["targets"][0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="actor-local policy compatibility"):
        validate_dagger_corpus(incompatible_target, verify_hashes=False)

    illegal_body = copy.deepcopy(corpus)
    row = next(
        row for row in illegal_body["data"]["rows"]
        if row["kind"] == "spawn_positive"
    )
    row["action"]["body_order"][0, row["actor_index"], 0] = torch.tensor(
        [6, 0, 2, 1, 3, 4, 5, 7],
    )
    with pytest.raises(ValueError, match="zero-count suffix"):
        validate_dagger_corpus(illegal_body, verify_hashes=False)


def test_joint_loader_accepts_complete_unqualified_current_source_checkpoint():
    base = "c" * 64
    actor, critic = Actor(), Critic()
    checkpoint = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "meta": artifact_meta(
            "joint_pretrain", actor, critic,
            partial=False, qualified=False, global_epoch=2, global_epochs=2,
            corpus_sha256=base,
        ),
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "joint.pt"
        torch.save(checkpoint, path)
        restored, meta, sha256 = _load_joint_actor(
            path, base_corpus_id=base, device=torch.device("cpu"),
        )
        assert isinstance(restored, Actor)
        assert meta["qualified"] is False
        assert len(sha256) == 64
        checkpoint["meta"]["source_sha256"] = "d" * 64
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="source fingerprint"):
            _load_joint_actor(path, base_corpus_id=base, device=torch.device("cpu"))
        restored, meta, _sha256 = _load_joint_actor(
            path, base_corpus_id=base, device=torch.device("cpu"),
            allow_policy_source_mismatch=True,
        )
        assert isinstance(restored, Actor)
        assert meta["source_sha256"] == "d" * 64
        checkpoint["meta"]["partial"] = True
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="complete joint checkpoint"):
            _load_joint_actor(
                path, base_corpus_id=base, device=torch.device("cpu"),
                allow_policy_source_mismatch=True,
            )


def test_source_mismatch_provenance_must_be_explicit_and_consistent():
    _config, corpus = _synthetic_corpus()
    mismatch = copy.deepcopy(corpus)
    mismatch["meta"]["checkpoint_source_sha256"] = "d" * 64
    mismatch["meta"]["policy_source_mismatch"] = True
    with pytest.raises(ValueError, match="not explicitly authorized"):
        validate_dagger_corpus(mismatch, verify_hashes=False)
    mismatch["meta"]["policy_source_mismatch_allowed"] = True
    validate_dagger_corpus(mismatch, verify_hashes=False)
    mismatch["meta"]["policy_source_mismatch"] = False
    with pytest.raises(ValueError, match="audit differs"):
        validate_dagger_corpus(mismatch, verify_hashes=False)


def test_joint_loader_allows_only_explicit_older_teacher_abi_mismatch():
    base = "e" * 64
    actor, critic = Actor(), Critic()
    checkpoint = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "meta": artifact_meta(
            "joint_pretrain", actor, critic,
            partial=False, qualified=False, global_epoch=2, global_epochs=2,
            corpus_sha256=base,
        ),
    }
    old_teacher_abi = _POLICY_TEACHER_ABI_BRIDGE["checkpoint_teacher_abi"]
    checkpoint["meta"]["contracts"] = dict(checkpoint["meta"]["contracts"])
    checkpoint["meta"]["contracts"]["teacherAbi"] = old_teacher_abi
    checkpoint["meta"]["schema_sha256"] = _POLICY_TEACHER_ABI_BRIDGE[
        "checkpoint_schema_sha256"
    ]
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "joint-old-teacher.pt"
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="teacher ABI differs"):
            _load_joint_actor(path, base_corpus_id=base, device=torch.device("cpu"))
        restored, meta, _sha256 = _load_joint_actor(
            path, base_corpus_id=base, device=torch.device("cpu"),
            allow_policy_teacher_abi_mismatch=True,
        )
        assert isinstance(restored, Actor)
        assert meta["contracts"]["teacherAbi"] == old_teacher_abi

        checkpoint["meta"]["source_sha256"] = "d" * 64
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="source fingerprint"):
            _load_joint_actor(
                path, base_corpus_id=base, device=torch.device("cpu"),
                allow_policy_teacher_abi_mismatch=True,
            )
        restored, _meta_value, _sha256 = _load_joint_actor(
            path, base_corpus_id=base, device=torch.device("cpu"),
            allow_policy_source_mismatch=True,
            allow_policy_teacher_abi_mismatch=True,
        )
        assert isinstance(restored, Actor)
        checkpoint["meta"]["source_sha256"] = source_signature()

        checkpoint["meta"]["schema_sha256"] = SCHEMA_SHA256
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="fingerprint is inconsistent"):
            _load_joint_actor(
                path, base_corpus_id=base, device=torch.device("cpu"),
                allow_policy_teacher_abi_mismatch=True,
            )
        checkpoint["meta"]["schema_sha256"] = _POLICY_TEACHER_ABI_BRIDGE[
            "checkpoint_schema_sha256"
        ]

        checkpoint["meta"]["contracts"]["actionAbi"] -= 1
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="non-teacher artifact ABIs differ"):
            _load_joint_actor(
                path, base_corpus_id=base, device=torch.device("cpu"),
                allow_policy_teacher_abi_mismatch=True,
            )

        checkpoint["meta"]["contracts"]["actionAbi"] += 1
        checkpoint["meta"]["contracts"]["teacherAbi"] = old_teacher_abi - 1
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="19-to-20 bridge"):
            _load_joint_actor(
                path, base_corpus_id=base, device=torch.device("cpu"),
                allow_policy_teacher_abi_mismatch=True,
            )

        checkpoint["meta"]["contracts"]["teacherAbi"] = (
            int(SCHEMA["artifact"]["teacherAbi"]) + 1
        )
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="19-to-20 bridge"):
            _load_joint_actor(
                path, base_corpus_id=base, device=torch.device("cpu"),
                allow_policy_teacher_abi_mismatch=True,
            )


def test_teacher_abi_mismatch_provenance_must_be_explicit_and_consistent():
    _config, corpus = _synthetic_corpus()
    mismatch = copy.deepcopy(corpus)
    mismatch["meta"]["checkpoint_contracts"]["teacherAbi"] -= 1
    mismatch["meta"]["checkpoint_teacher_abi"] -= 1
    mismatch["meta"]["checkpoint_schema_sha256"] = _POLICY_TEACHER_ABI_BRIDGE[
        "checkpoint_schema_sha256"
    ]
    mismatch["meta"]["policy_teacher_abi_mismatch"] = True
    with pytest.raises(ValueError, match="was not authorized"):
        validate_dagger_corpus(mismatch, verify_hashes=False)
    mismatch["meta"]["policy_teacher_abi_mismatch_allowed"] = True
    validate_dagger_corpus(mismatch, verify_hashes=False)
    wrong_version = copy.deepcopy(mismatch)
    wrong_version["meta"]["checkpoint_schema_version"] -= 1
    with pytest.raises(ValueError, match="schema versions differ"):
        validate_dagger_corpus(wrong_version, verify_hashes=False)
    mismatch["meta"]["policy_teacher_abi_mismatch"] = False
    with pytest.raises(ValueError, match="audit is inconsistent"):
        validate_dagger_corpus(mismatch, verify_hashes=False)
