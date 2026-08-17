"""Focused structural tests for immutable pretraining-corpus artifacts."""
from __future__ import annotations

import copy
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch

from .constants import (
    ACTOR_FEAT, ACTOR_FEATURE_INDEX, CONSTRUCTION_MASK_BYTES, GLOBAL_FEAT, INTENT_SLOTS,
    INTENT_TYPES, N_AMOUNT, N_BODY_PART, N_CONSTRUCTION_TYPE, PATCHES_PER_ROOM,
    MAX_ROOM_ENERGY, PATCH_FLAT, SCHEMA, SCHEMA_SHA256, TARGET_FEAT,
    MODEL_CFG,
)
from .pretrain_corpus import (
    ACTION_KEYS,
    CORPUS_SCHEMA_VERSION,
    TEMPORAL_OBS_KEYS,
    CorpusConfig,
    _append_temporal_replay,
    _invalid_scripted_intent_details,
    _same_state_counterfactual_action,
    _validate_temporal_replay,
    _record_teacher_step,
    assemble_corpus,
    load_corpus,
    save_corpus,
    validate_corpus,
    parse_args as parse_corpus_args,
)


def _obs() -> dict[str, torch.Tensor]:
    actor, target, room = 8, 16, 1
    return {
        "patches": torch.zeros(
            1, room, PATCHES_PER_ROOM, PATCH_FLAT, dtype=torch.uint8,
        ),
        "room_mask": torch.ones(1, room, dtype=torch.uint8),
        "room_coords": torch.zeros(1, room, 2),
        "actors": torch.zeros(1, actor, ACTOR_FEAT),
        "actor_mask": torch.ones(1, actor, dtype=torch.uint8),
        "actor_outcome": torch.zeros(1, actor, dtype=torch.uint8),
        "targets": torch.zeros(1, target, TARGET_FEAT),
        "target_mask": torch.ones(1, target, dtype=torch.uint8),
        "intent_mask": torch.ones(
            1, actor, INTENT_SLOTS, len(INTENT_TYPES), dtype=torch.uint8,
        ),
        "dir_mask": torch.ones(1, actor, INTENT_SLOTS, 8, dtype=torch.uint8),
        "target_select_mask": torch.ones(
            1, len(INTENT_TYPES), target, dtype=torch.uint8,
        ),
        "amount_mask": torch.ones(
            1, actor, INTENT_SLOTS, len(INTENT_TYPES), N_AMOUNT,
            dtype=torch.uint8,
        ),
        "construction_mask": torch.ones(
            1, room, N_CONSTRUCTION_TYPE, CONSTRUCTION_MASK_BYTES,
            dtype=torch.uint8,
        ),
        "globals": torch.zeros(1, GLOBAL_FEAT),
    }


def _action() -> dict[str, torch.Tensor]:
    actor = 8
    action = {
        "types": torch.zeros(1, actor, 1, dtype=torch.long),
        "dirs": torch.zeros(1, actor, 1, dtype=torch.long),
        "targets": torch.zeros(1, actor, 1, dtype=torch.long),
        "amounts": torch.zeros(1, actor, 1, dtype=torch.long),
        "body_counts": torch.zeros(1, actor, 1, N_BODY_PART, dtype=torch.long),
        "body_order": torch.arange(N_BODY_PART).view(1, 1, 1, -1).expand(
            1, actor, 1, -1,
        ).clone(),
        "construction_types": torch.zeros(1, actor, 1, dtype=torch.long),
        "construction_tiles": torch.zeros(1, actor, 1, dtype=torch.long),
    }
    assert set(action) == ACTION_KEYS
    return action


def _sampling(stratum: str) -> dict:
    return {
        "algorithm": "algorithm_r_per_stratum",
        "capacity_per_stratum": 64,
        "rng": "numpy.default_rng/PCG64",
        "seed": 7,
        "seen_by_stratum": {stratum: 1},
        "retained_by_stratum": {stratum: 1},
    }


def _corpus() -> dict:
    config = CorpusConfig(
        num_envs=1, steps=2, max_episode=2, curriculum="empty", seed=3,
        holdout_seed_offset=10, gamma=0.9, ti_actor_steps=2,
        ti_replay_capacity=4, ti_critic_replay_per_stratum=64,
    )
    rewards = torch.tensor([[1.0], [2.0]])
    dones = torch.zeros_like(rewards)
    returns = torch.tensor([[2.8], [2.0]])
    scripted_stratum = "empty:r1:p0_3:none"
    scripted_row = {
        "stratum": scripted_stratum, "timestep": 0, "env_index": 0,
        "return_target": returns[0, 0].clone(), "obs": _obs(), "action": _action(),
    }
    temporal_rows = []
    for timestep in range(2):
        current = _obs()
        current["globals"][0, 1] = float(timestep)
        future = _obs()
        future["globals"][0, 1] = float(timestep + 1)
        action = _action()
        action["types"][0, 0, 0] = INTENT_TYPES.index("move")
        action["dirs"][0, 0, 0] = timestep + 1
        counterfactual = _same_state_counterfactual_action(current, action, 0, 8)
        assert counterfactual is not None
        temporal_rows.append({
            "stratum": scripted_stratum,
            "timestep": timestep,
            "env_index": 0,
            "terminated": False,
            "truncated": False,
            "obs": {key: current[key] for key in TEMPORAL_OBS_KEYS},
            "action": action,
            "counterfactual_action": counterfactual,
            "next_obs": {key: future[key] for key in TEMPORAL_OBS_KEYS},
        })
    ti_stratum = "r1:rooms1:p0_3:k0"
    ti_row = {
        "stratum": ti_stratum, "timestep": 0, "env_index": 0,
        "return_target": returns[0, 0].clone(), "obs": _obs(),
    }
    scripted = {
        "lifecycle_replay": [scripted_row], "rewards_tn": rewards,
        "dones_tn": dones, "returns_tn": returns,
        "sampling": _sampling(scripted_stratum),
        "temporal_replay": temporal_rows,
        "temporal_sampling": {
            "algorithm": "algorithm_r_per_stratum_causal_pairs",
            "capacity_per_stratum": config.temporal_replay_per_stratum,
            "rng": "numpy.default_rng/PCG64",
            "seed": 101,
            "pairing": "same_env_same_tick_pre_action_to_post_action",
            "terminal_policy": "exclude_episode_end_before_vec_reset",
            "seen_by_stratum": {scripted_stratum: 2},
            "retained_by_stratum": {scripted_stratum: 2},
            "excluded_terminal": 0,
            "excluded_truncated": 0,
            "counterfactual_retained": 2,
        },
    }
    ti = {
        "critic_replay": [ti_row], "rewards_tn": rewards.clone(),
        "dones_tn": dones.clone(), "returns_tn": returns.clone(),
        "sampling": _sampling(ti_stratum),
    }
    ti_train = dict(ti)
    ti_train["factor_sampling"] = {
        "algorithm": "algorithm_r", "capacity": 4,
        "rng": "numpy.default_rng/PCG64", "seed": 9,
        "seen": 0, "retained": 0, "counts": {},
        "translator_policy": "exact_factors_only;macro_incompatible_move_rejected",
        "translator_abi": SCHEMA["artifact"]["teacherAbi"],
    }
    totals = {
        "transitions": 2, "harvest": 1.0, "control": 2.0, "skill": 3.0,
        "delivery": 0.0, "build": 0.0, "claims": 0, "max_creeps": 0,
        "spawn_success": 0, "issued": 0, "invalid": 0, "recovered": 0,
        "action_type_hist": {},
    }
    meta = {
        **config.__dict__, "holdout_seed": config.holdout_seed,
        "ti_bot_dir": str(config.resolved_ti_bot_dir),
        "ti_bot_source_sha256": "test", "ti_runtime_source_sha256": "test",
        "environment_schema_version": SCHEMA["version"],
        "schema_sha256": SCHEMA_SHA256, "collection_source_sha256": "test",
        "contracts": dict(SCHEMA["artifact"]), "runtime": {},
        "expert": "scripted+ti", "critic_target": "finite_horizon_discounted_return",
        "critic_endpoint": "zero_at_declared_lifecycle_horizon",
        "train_env_map": config.env_map(config.seed),
        "holdout_env_map": config.env_map(config.holdout_seed),
        "ti_env_map": [
            {"split": "train", "env_index": 0, "seed": config.seed, "expert": "ti"},
            {"split": "holdout", "env_index": 0, "seed": config.holdout_seed, "expert": "ti"},
        ],
    }
    holdout_scripted = copy.deepcopy(scripted)
    holdout_scripted["temporal_sampling"]["seed"] = 202
    data = {
        "train": scripted, "holdout": holdout_scripted,
        "ti_train": ti_train, "ti_holdout": ti,
        "spawn_replay": [], "ti_factor_replay": [],
        "teacher": {"train_totals": totals, "holdout_totals": dict(totals)},
    }
    return assemble_corpus(meta, data)


class CorpusArtifactTests(unittest.TestCase):
    def test_executable_provenance_binds_supplement_helper_dependency(self) -> None:
        from unittest.mock import patch

        from .artifacts import SOURCE_SIGNATURE_AGENT_NAMES, source_signature

        self.assertIn("outpost_actor_corpus.py", SOURCE_SIGNATURE_AGENT_NAMES)
        self.assertIn("dagger_corpus.py", SOURCE_SIGNATURE_AGENT_NAMES)
        dagger_path = Path(__file__).with_name("dagger_corpus.py")
        original_read_bytes = Path.read_bytes
        baseline = source_signature()

        def dependency_changed(path: Path) -> bytes:
            payload = original_read_bytes(path)
            return payload + b"\n# provenance probe" if path == dagger_path else payload

        with patch.object(Path, "read_bytes", dependency_changed):
            changed = source_signature()
        self.assertNotEqual(baseline, changed)

    def test_seed_sets_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "seeds overlap"):
            CorpusConfig(num_envs=4, holdout_seed_offset=2).validate()
        args = parse_corpus_args([])
        self.assertEqual(args.temporal_replay_per_stratum, 64)
        self.assertIn("seed_outpost", args.curriculum.split(","))
        from .pretrain_joint import parse_args as parse_joint_args
        joint_args = parse_joint_args(["--corpus", "unused"])
        self.assertEqual(joint_args.min_rare_intent_rows, 32)
        self.assertIsNone(joint_args.actor_supplement)

    def test_invalid_teacher_diagnostic_identifies_env_seed_stage_and_result(self) -> None:
        details = _invalid_scripted_intent_details(
            [{"intentInvalid": 0}, {
                "curriculum": "seed_outpost", "intentInvalid": 1,
                "intentByType": {"harvest": {"issued": 1, "invalid": 1}},
                "intentResults": [
                    {"actor": "remote", "type": "move", "code": -11},
                    {
                        "actor": "remote", "type": "harvest", "code": -6,
                        "executed": False, "targetKind": "source", "targetId": "s1",
                    },
                ],
            }],
            [
                {"env_index": 0, "seed": 3, "curriculum": "empty"},
                {"env_index": 1, "seed": 4, "curriculum": "seed_outpost"},
            ],
        )
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["env_index"], 1)
        self.assertEqual(details[0]["seed"], 4)
        self.assertEqual(details[0]["curriculum"], "seed_outpost")
        self.assertEqual(details[0]["results"], [{
            "actor": "remote", "type": "harvest", "code": -6,
            "executed": False, "targetKind": "source", "targetId": "s1",
        }])

    def test_metric_skill_is_counted_once(self) -> None:
        totals = {
            "transitions": 0, "harvest": 0.0, "control": 0.0, "skill": 0.0,
            "delivery": 0.0, "build": 0.0, "claims": 0, "max_creeps": 0,
            "spawn_success": 0, "issued": 0, "invalid": 0, "recovered": 0,
            "action_type_hist": {name: 0 for name in INTENT_TYPES},
        }
        infos = [{
            "curriculum": "empty", "harvestDelta": 2, "controlDelta": 3,
            "remoteHarvestDelta": 7, "remoteHomeDeliveryDelta": 5,
            "remoteRoomsStaffedPeak": 1, "remoteProductiveCreepsPeak": 1,
            "remoteRoomsStaffed": 1, "remoteProductiveCreeps": 1,
            "neutralOutpostRooms": 1,
        }]
        actions = _action()
        _record_teacher_step(
            totals, {}, _obs(), actions, infos, lambda *_args: None,
            late_window=True,
        )
        self.assertEqual(totals["skill"], 5.0)
        self.assertEqual(totals["skill"], totals["harvest"] + totals["control"])
        self.assertEqual(totals["remote_harvest"], 7.0)
        self.assertEqual(totals["remote_home_delivery"], 5.0)
        self.assertEqual(totals["late_remote_harvest"], 7.0)
        self.assertEqual(totals["late_remote_home_delivery"], 5.0)
        self.assertEqual(totals["late_remote_staffed_ticks"], 1)
        self.assertEqual(totals["late_remote_productive_ticks"], 1)
        self.assertEqual(totals["neutral_outposts"], 1.0)

    def test_outpost_corpus_readiness_is_fail_closed_for_both_splits(self) -> None:
        corpus = _corpus()
        corpus["meta"]["curriculum"] = "seed_outpost"
        config = CorpusConfig(**{
            key: corpus["meta"][key] for key in CorpusConfig.__dataclass_fields__
        })
        corpus["meta"]["train_env_map"] = config.env_map(config.seed)
        corpus["meta"]["holdout_env_map"] = config.env_map(config.holdout_seed)
        ready = {
            "late_transitions": 1,
            "late_remote_harvest": 1.0,
            "late_remote_home_delivery": 1.0,
            "late_remote_staffed_ticks": 1,
            "late_remote_productive_ticks": 1,
            "claims": 0,
            "remote_owned_peak": 0.0,
            "invalid": 0,
        }
        teacher = corpus["data"]["teacher"]
        teacher["train_by_curriculum"] = {"seed_outpost": dict(ready)}
        teacher["holdout_by_curriculum"] = {"seed_outpost": dict(ready)}
        validate_corpus(corpus, verify_hashes=False)

        failures = {
            "late_transitions": 0,
            "late_remote_harvest": 0.0,
            "late_remote_home_delivery": 0.0,
            "late_remote_staffed_ticks": 0,
            "late_remote_productive_ticks": 0,
            "claims": 1,
            "remote_owned_peak": 1.0,
            "invalid": 1,
        }
        for split in ("train", "holdout"):
            for key, value in failures.items():
                with self.subTest(split=split, key=key):
                    broken = copy.deepcopy(corpus)
                    broken["data"]["teacher"][f"{split}_by_curriculum"][
                        "seed_outpost"
                    ][key] = value
                    with self.assertRaisesRegex(
                        ValueError, f"teacher {split} seed_outpost",
                    ):
                        validate_corpus(broken, verify_hashes=False)

    def test_tensor_shard_roundtrip_is_weights_only_and_idempotent(self) -> None:
        corpus = _corpus()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = save_corpus(corpus, root)
            self.assertEqual(destination.name, corpus["integrity"]["corpus_sha256"])
            self.assertEqual(save_corpus(corpus, root), destination)
            loaded = load_corpus(destination)
            self.assertEqual(loaded["corpus_schema_version"], CORPUS_SCHEMA_VERSION)
            self.assertTrue(torch.equal(
                loaded["data"]["train"]["returns_tn"],
                corpus["data"]["train"]["returns_tn"],
            ))

    def test_tensor_shard_corruption_is_rejected(self) -> None:
        corpus = _corpus()
        with tempfile.TemporaryDirectory() as temporary:
            destination = save_corpus(corpus, Path(temporary))
            shard = next((destination / "shards").glob("*.pt"))
            shard.chmod(0o644)
            payload = bytearray(shard.read_bytes())
            payload[len(payload) // 2] ^= 0x01
            shard.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "shard integrity failure"):
                load_corpus(destination)

    def test_row_target_and_reservoir_contract_are_strict(self) -> None:
        corpus = _corpus()
        corpus["data"]["train"]["lifecycle_replay"][0]["return_target"] = torch.tensor(9.0)
        with self.assertRaisesRegex(ValueError, "return_target differs"):
            validate_corpus(corpus, verify_hashes=False)
        corpus = _corpus()
        corpus["data"]["train"]["sampling"]["retained_by_stratum"] = {}
        with self.assertRaisesRegex(ValueError, "retained counts mismatch"):
            validate_corpus(corpus, verify_hashes=False)

    def test_malformed_capacity_dtype_and_hash_are_rejected(self) -> None:
        corpus = _corpus()
        corpus["data"]["train"]["lifecycle_replay"][0]["obs"]["actors"] = (
            torch.zeros(1, 7, ACTOR_FEAT)
        )
        with self.assertRaisesRegex(ValueError, "unsupported compact capacity"):
            validate_corpus(corpus, verify_hashes=False)
        corpus = _corpus()
        corpus["meta"]["gamma"] = 0.8
        with self.assertRaisesRegex(ValueError, "semantic replay"):
            validate_corpus(corpus, verify_hashes=False)
        corpus = _corpus()
        corpus["data"]["train"]["temporal_replay"][0]["action"]["dirs"][0, 0, 0] = 8
        with self.assertRaisesRegex(ValueError, "dirs out of bounds"):
            validate_corpus(corpus, verify_hashes=False)

    def test_temporal_rows_are_same_tick_pairs_and_episode_ends_are_excluded(self) -> None:
        import numpy as np

        pre = _obs()
        pre["globals"][0, 1] = 11
        post = _obs()
        post["globals"][0, 1] = 12
        actions = _action()
        actions["dirs"][0, 0, 0] = 5
        rows: list[dict] = []
        seen: dict[str, int] = {}
        retained: dict[str, list[int]] = {}
        excluded = {"excluded_terminal": 0, "excluded_truncated": 0}
        _append_temporal_replay(
            rows, seen, retained, excluded, np.random.default_rng(1),
            pre, actions, post, torch.zeros(1), [{"curriculum": "empty"}],
            timestep=5, capacity_per_stratum=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["obs"]["globals"][0, 1]), 11)
        self.assertEqual(int(rows[0]["action"]["dirs"][0, 0, 0]), 5)
        self.assertEqual(float(rows[0]["next_obs"]["globals"][0, 1]), 12)

        reservoir_rng = np.random.default_rng(3)
        for timestep in range(6, 20):
            current = _obs()
            current["globals"][0, 1] = timestep
            future = _obs()
            future["globals"][0, 1] = timestep + 1
            marked_action = _action()
            marked_action["dirs"][0, 0, 0] = timestep % 8
            _append_temporal_replay(
                rows, seen, retained, excluded, reservoir_rng,
                current, marked_action, future, torch.zeros(1),
                [{"curriculum": "empty"}], timestep=timestep,
                capacity_per_stratum=1,
            )
        retained_timestep = rows[0]["timestep"]
        self.assertEqual(
            float(rows[0]["obs"]["globals"][0, 1]), retained_timestep,
        )
        self.assertEqual(
            int(rows[0]["action"]["dirs"][0, 0, 0]), retained_timestep % 8,
        )
        self.assertEqual(
            float(rows[0]["next_obs"]["globals"][0, 1]), retained_timestep + 1,
        )

        reset = _obs()
        reset["globals"][0, 1] = 99
        _append_temporal_replay(
            rows, seen, retained, excluded, np.random.default_rng(2),
            post, actions, reset, torch.ones(1),
            [{"curriculum": "empty", "episode_done": True, "truncated": True}],
            timestep=20, capacity_per_stratum=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(excluded["excluded_truncated"], 1)
        self.assertNotEqual(float(rows[0]["next_obs"]["globals"][0, 1]), 99)

        issued = _action()
        issued["types"][0, 0, 0] = INTENT_TYPES.index("move")
        counterfactual = _same_state_counterfactual_action(_obs(), issued, 0, 8)
        self.assertIsNotNone(counterfactual)
        assert counterfactual is not None
        self.assertEqual(int(counterfactual["types"][0, 0, 0]), 0)
        self.assertTrue(torch.equal(
            counterfactual["types"][0, 1:], issued["types"][0, 1:],
        ))

    def test_temporal_corruption_and_holdout_sampling_reuse_are_rejected(self) -> None:
        corpus = _corpus()
        self.assertEqual(
            set(corpus["data"]["train"]["temporal_replay"][0]["obs"]),
            TEMPORAL_OBS_KEYS,
        )
        corpus["data"]["train"]["temporal_replay"][0]["truncated"] = True
        with self.assertRaisesRegex(ValueError, "episode-ending transition"):
            validate_corpus(corpus, verify_hashes=False)

        corpus = _corpus()
        corpus["data"]["holdout"]["temporal_sampling"]["seed"] = (
            corpus["data"]["train"]["temporal_sampling"]["seed"]
        )
        with self.assertRaisesRegex(ValueError, "sampling RNG seeds overlap"):
            validate_corpus(corpus, verify_hashes=False)

        corpus = _corpus()
        corpus["data"]["train"]["temporal_replay"][0]["next_obs"]["globals"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "semantic SHA-256 mismatch"):
            validate_corpus(corpus, verify_hashes=True)

        corpus = _corpus()
        row = corpus["data"]["train"]["temporal_replay"][0]
        row["counterfactual_action"]["dirs"][0, 1, 0] = 1
        with self.assertRaisesRegex(ValueError, "change exactly one actor"):
            validate_corpus(corpus, verify_hashes=False)

        corpus = _corpus()
        row = corpus["data"]["train"]["temporal_replay"][0]
        row["counterfactual_action"] = copy.deepcopy(row["action"])
        row["counterfactual_action"]["dirs"][0, 0, 0] = 7
        with self.assertRaisesRegex(ValueError, "not canonical none"):
            validate_corpus(corpus, verify_hashes=False)

        corpus = _corpus()
        rows = copy.deepcopy(corpus["data"]["train"]["temporal_replay"])
        rows[1]["timestep"] = 2
        sampling = copy.deepcopy(corpus["data"]["train"]["temporal_sampling"])
        sampling["excluded_truncated"] = 1
        with self.assertRaisesRegex(ValueError, "episode-ending trajectory cell"):
            _validate_temporal_replay(
                rows, sampling, steps=3, envs=1,
                dones_tn=torch.tensor([[0.0], [0.0], [1.0]]),
                location="test",
            )

    def test_joint_epoch_uses_one_step_for_lifecycle_and_nextlat(self) -> None:
        from .model import Actor, Critic
        from .pretrain_joint import (
            DaggerSample,
            _evaluate_nextlat,
            _lifecycle_samples,
            _temporal_samples,
            _train_joint_lifecycle_epoch,
        )

        class CountingSGD(torch.optim.SGD):
            def __init__(self, parameters):
                super().__init__(parameters, lr=1e-3)
                self.step_calls = 0
                self.zero_grad_calls = 0

            def step(self, closure=None):
                self.step_calls += 1
                return super().step(closure)

            def zero_grad(self, set_to_none=True):
                self.zero_grad_calls += 1
                return super().zero_grad(set_to_none=set_to_none)

        cfg = dict(MODEL_CFG)
        cfg.update({"dModel": 32, "nHeads": 4, "spatialLayers": 1,
                    "entityLayers": 1, "ffMult": 2})
        actor = Actor(cfg)
        critic = Critic(cfg)
        corpus = _corpus()
        rows = copy.deepcopy(corpus["data"]["train"]["temporal_replay"])
        move = INTENT_TYPES.index("move")
        for index, row in enumerate(rows):
            row["action"]["types"][0, 0, 0] = move
            row["action"]["dirs"][0, 0, 0] = index + 1
            row["next_obs"]["globals"][0, 0] = 0.25 + 0.25 * index
        temporal = _temporal_samples(rows)
        for sample in temporal:
            sample.obs["globals"].requires_grad_()
            sample.next_obs["globals"].requires_grad_()
        lifecycle = _lifecycle_samples(
            corpus["data"]["train"]["lifecycle_replay"],
        )
        lifecycle = [copy.deepcopy(lifecycle[0]) for _ in range(6)]
        rare_replay = [copy.deepcopy(lifecycle[0]), copy.deepcopy(lifecycle[0])]
        rare_replay[0].action["types"][0, 0, 0] = INTENT_TYPES.index(
            "createConstructionSite"
        )
        rare_replay[0].obs["actors"][0, 0, ACTOR_FEATURE_INDEX["isRoom"]] = 1
        rare_replay[0].action["construction_types"][0, 0, 0] = 1
        rare_replay[1].action["types"][0, 0, 0] = INTENT_TYPES.index(
            "claimController"
        )
        dagger_replay = [
            DaggerSample(
                kind="correction", stratum=f"dagger:{index}", actor_index=0,
                obs=sample.obs, action=sample.action,
            )
            for index, sample in enumerate(rare_replay)
        ]
        spawn_action = copy.deepcopy(lifecycle[0].action)
        spawn_action["types"][0, 0, 0] = INTENT_TYPES.index("spawnCreep")
        spawn_action["body_counts"][0, 0, 0, 0] = 1
        spawn_obs = copy.deepcopy(lifecycle[0].obs)
        spawn_obs["actors"][0, 0, ACTOR_FEATURE_INDEX["isNonCreep"]] = 1
        spawn_obs["actors"][0, 0, ACTOR_FEATURE_INDEX["isSpawn"]] = 1
        spawn_obs["actors"][
            0, 0, ACTOR_FEATURE_INDEX["roomEnergyAvailable"]
        ] = 300 / MAX_ROOM_ENERGY
        spawn_replay = [
            ("spawn:test", 0, spawn_obs, spawn_action),
            ("waitlegal:test", 0, spawn_obs, lifecycle[0].action),
        ]
        actor_before = actor.latent_dynamics.dynamics_mlp[-1].weight.detach().clone()
        critic_before = critic.latent_dynamics.dynamics_mlp[-1].weight.detach().clone()
        head_before = critic.value_head[-1].weight.detach().clone()
        actor_optimizer = CountingSGD(actor.parameters())
        critic_optimizer = CountingSGD(critic.parameters())
        nll, legal, value_loss, trained, metrics = _train_joint_lifecycle_epoch(
            actor, critic, actor_optimizer, critic_optimizer,
            lifecycle, corpus["data"]["train"]["returns_tn"], temporal,
            device=torch.device("cpu"), use_bf16=False, minibatch=2,
            shuffle_generator=torch.Generator().manual_seed(11),
            nextlat_shuffle_generator=torch.Generator().manual_seed(12),
            correction_replay=dagger_replay,
            correction_shuffle_generator=torch.Generator().manual_seed(13),
            rare_replay=rare_replay,
            rare_refs_by_intent={
                "createConstructionSite": [(0, 0)],
                "claimController": [(1, 0)],
            },
            rare_shuffle_generator=torch.Generator().manual_seed(14),
            spawn_replay=spawn_replay,
            spawn_shuffle_generator=torch.Generator().manual_seed(15),
        )
        self.assertTrue(torch.isfinite(torch.tensor([nll, value_loss])).all())
        self.assertEqual(legal, 1.0)
        self.assertEqual(trained, 6)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        self.assertEqual(actor_optimizer.zero_grad_calls, 3)
        self.assertEqual(actor_optimizer.step_calls, 3)
        self.assertEqual(critic_optimizer.zero_grad_calls, 3)
        self.assertEqual(critic_optimizer.step_calls, 3)
        self.assertEqual(metrics["correction_exposures"], 2.0)
        self.assertEqual(metrics["rare_exposures"], 2.0)
        self.assertEqual(metrics["spawn_exposures"], 2.0)
        self.assertEqual(metrics["optimizer_steps"], 3.0)
        self.assertEqual(metrics["auxiliary_batches"], 3.0)
        self.assertEqual(metrics["auxiliary_minibatch"], 2.0)
        self.assertEqual(
            metrics["auxiliary_batches"],
            metrics["correction_batches"]
            + metrics["rare_batches"]
            + metrics["spawn_batches"],
        )
        self.assertFalse(torch.equal(
            actor_before, actor.latent_dynamics.dynamics_mlp[-1].weight,
        ))
        self.assertFalse(torch.equal(
            critic_before, critic.latent_dynamics.dynamics_mlp[-1].weight,
        ))
        self.assertFalse(torch.equal(critic.value_head[-1].weight, head_before))
        self.assertTrue(any(
            sample.obs["globals"].grad is not None for sample in temporal
        ))
        self.assertTrue(all(
            sample.next_obs["globals"].grad is None for sample in temporal
        ))
        evaluated = _evaluate_nextlat(
            actor, critic, temporal, device=torch.device("cpu"), minibatch=2,
        )
        for name in (
            "actor_mse", "actor_identity_mse", "actor_counterfactual_action_mse",
            "critic_mse", "critic_identity_mse", "critic_counterfactual_action_mse",
            "critic_kl",
        ):
            self.assertTrue(torch.isfinite(torch.tensor(evaluated[name])), name)

    def test_nextlat_checkpoint_metadata_carries_identity_and_metrics(self) -> None:
        from .pretrain_joint import (
            _nextlat_checkpoint_metadata, _precision_resume_identity,
        )

        metadata = _nextlat_checkpoint_metadata(
            SimpleNamespace(
                nextlat_train_rows=11, nextlat_holdout_rows=7,
                nextlat_actor_coef=1.0, nextlat_critic_coef=1.0,
                nextlat_critic_kl_coef=0.1,
                min_nextlat_relative_gap=0.01,
                min_nextlat_counterfactual_rows=128,
            ),
            {"holdout_actor_mse": 0.25},
        )
        self.assertEqual(metadata["nextlat"], SCHEMA["nextLat"])
        self.assertEqual(metadata["nextlat_train_rows"], 11)
        self.assertEqual(metadata["nextlat_holdout_rows"], 7)
        self.assertEqual(metadata["nextlat_holdout_actor_mse"], 0.25)
        self.assertEqual(metadata["nextlat_pretrain"]["actorCoef"], 1.0)
        self.assertEqual(
            metadata["nextlat_pretrain"]["optimizerAuthority"],
            "lifecycle_primary_fused_bc_value_nextlat_correction_rare_spawn_v3",
        )
        self.assertEqual(
            metadata["nextlat_pretrain"]["actionAblation"],
            "same_state_whole_joint_canonical_none_on_the_fly",
        )
        self.assertEqual(
            metadata["nextlat_pretrain"]["action_pooling"],
            "issued_sum_sqrt_count_v1",
        )
        self.assertEqual(
            metadata["nextlat_action_pooling"], "issued_sum_sqrt_count_v1",
        )
        from .pretrain_joint import _continuation_mismatches
        stale = copy.deepcopy(metadata)
        stale["nextlat_pretrain"]["action_pooling"] = "all_live_mean_v0"
        stale["nextlat_action_pooling"] = "all_live_mean_v0"
        mismatches = _continuation_mismatches(stale, metadata)
        self.assertIn("nextlat_action_pooling", mismatches)
        self.assertIn(
            "nextlat_pretrain",
            mismatches,
        )
        schedule_identity = {
            "actor_auxiliary_schedule": (
                "one_balanced_source_epoch_joint_collision_free_quantiles_v2"
            ),
        }
        self.assertIn(
            "actor_auxiliary_schedule",
            _continuation_mismatches(
                {"actor_auxiliary_schedule": "standalone_after_joint_v0"},
                schedule_identity,
            ),
        )
        bf16 = _precision_resume_identity(SimpleNamespace(
            effective_precision="bf16", training_device_type="cuda",
            visible_cuda_device_count=2,
        ))
        fp32 = _precision_resume_identity(SimpleNamespace(
            effective_precision="fp32", training_device_type="cuda",
            visible_cuda_device_count=2,
        ))
        self.assertNotEqual(bf16, fp32)
        one_gpu = dict(bf16, visible_cuda_device_count=1)
        self.assertNotEqual(bf16, one_gpu)

    def test_actor_supplement_identity_is_exact_resume_contract(self) -> None:
        from .pretrain_joint import (
            _actor_supplement_identity, _continuation_mismatches,
        )

        args = SimpleNamespace(
            actor_supplement_kind="scripted_teacher_supplement",
            actor_supplement_sha256="a" * 64,
            actor_supplement_schema_version=1,
            actor_supplement_schema_sha256="b" * 64,
            actor_supplement_source_sha256="c" * 64,
            actor_supplement_collector_sha256="d" * 64,
            actor_supplement_collection_seeds=[20_003, 20_004],
        )
        expected = _actor_supplement_identity(args)
        self.assertEqual(expected["actor_supplement_kind"], args.actor_supplement_kind)
        self.assertEqual(expected["actor_supplement_collection_seeds"], [20_003, 20_004])
        for key in expected:
            stale = copy.deepcopy(expected)
            stale[key] = None
            self.assertIn(key, _continuation_mismatches(stale, expected), key)

    def test_actor_supplement_routes_all_row_kinds_with_source_identity(self) -> None:
        from .pretrain_joint import _route_correction_rows

        lifecycle_row = copy.deepcopy(
            _corpus()["data"]["train"]["lifecycle_replay"][0]
        )
        rows = []
        for kind in ("exact_intent", "spawn_positive", "spawn_wait_legal"):
            rows.append({
                "kind": kind,
                "stratum": f"outpost:{kind}",
                "timestep": lifecycle_row["timestep"],
                "env_index": lifecycle_row["env_index"],
                "actor_index": 0,
                "obs": lifecycle_row["obs"],
                "action": lifecycle_row["action"],
            })
        tagged, exact, spawn = _route_correction_rows(
            rows, source="actor_supplement",
        )
        self.assertEqual(len(tagged), 3)
        self.assertEqual(len(exact), 1)
        self.assertEqual(len(spawn), 2)
        self.assertTrue(all(
            sample.stratum.startswith("actor_supplement:") for sample in tagged
        ))
        self.assertEqual(
            {row[0].split(":", 1)[0] for row in spawn},
            {"spawn", "waitlegal"},
        )
        bad = copy.deepcopy(rows[:1])
        bad[0]["kind"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported actor_supplement"):
            _route_correction_rows(bad, source="actor_supplement")

    def test_actor_supplement_seeds_are_independent_of_every_authority(self) -> None:
        from .pretrain_joint import _correction_seed_conflicts

        meta = {
            "seed": 3,
            "train_env_map": [{"seed": 3}, {"seed": 4}],
            "holdout_env_map": [{"seed": 10_003}],
        }
        conflicts = _correction_seed_conflicts(
            meta,
            dagger_seeds={30_000, 30_001},
            supplement_seeds={4, 23, 30_001},
            evaluation_offset=20,
            num_envs=2,
        )
        self.assertEqual(conflicts["dagger_base"], set())
        self.assertEqual(conflicts["supplement_base"], {4})
        self.assertEqual(conflicts["dagger_supplement"], {30_001})
        self.assertEqual(conflicts["evaluation"], {23})

    def test_joint_epoch_resume_matches_uninterrupted_training(self) -> None:
        from .model import Actor, Critic
        from .pretrain_joint import (
            DaggerSample, _lifecycle_samples, _temporal_samples,
            _train_joint_lifecycle_epoch,
        )

        cfg = dict(MODEL_CFG)
        cfg.update({"dModel": 16, "nHeads": 4, "spatialLayers": 1,
                    "entityLayers": 1, "ffMult": 2})
        corpus = _corpus()
        lifecycle = _lifecycle_samples(
            corpus["data"]["train"]["lifecycle_replay"],
        )
        lifecycle = [copy.deepcopy(lifecycle[0]) for _ in range(6)]
        temporal = _temporal_samples(
            corpus["data"]["train"]["temporal_replay"],
        )
        returns = corpus["data"]["train"]["returns_tn"]
        rare_replay = [copy.deepcopy(lifecycle[0]), copy.deepcopy(lifecycle[0])]
        rare_replay[0].action["types"][0, 0, 0] = INTENT_TYPES.index(
            "createConstructionSite"
        )
        rare_replay[0].action["construction_types"][0, 0, 0] = 1
        rare_replay[0].obs["actors"][0, 0, ACTOR_FEATURE_INDEX["isRoom"]] = 1
        rare_replay[1].action["types"][0, 0, 0] = INTENT_TYPES.index(
            "claimController"
        )
        dagger_replay = [
            DaggerSample(
                kind="exact_intent", stratum=f"dagger:{index}", actor_index=0,
                obs=sample.obs, action=sample.action,
            )
            for index, sample in enumerate(rare_replay)
        ]
        spawn_obs = copy.deepcopy(lifecycle[0].obs)
        spawn_obs["actors"][0, 0, ACTOR_FEATURE_INDEX["isNonCreep"]] = 1
        spawn_obs["actors"][0, 0, ACTOR_FEATURE_INDEX["isSpawn"]] = 1
        spawn_obs["actors"][
            0, 0, ACTOR_FEATURE_INDEX["roomEnergyAvailable"]
        ] = 300 / MAX_ROOM_ENERGY
        spawn_action = copy.deepcopy(lifecycle[0].action)
        spawn_action["types"][0, 0, 0] = INTENT_TYPES.index("spawnCreep")
        spawn_action["body_counts"][0, 0, 0, 0] = 1
        spawn_replay = [
            ("spawn:test", 0, spawn_obs, spawn_action),
            ("waitlegal:test", 0, spawn_obs, lifecycle[0].action),
        ]

        def initialized():
            torch.manual_seed(91)
            actor, critic = Actor(cfg), Critic(cfg)
            return (
                actor, critic,
                torch.optim.AdamW(actor.parameters(), lr=1e-4),
                torch.optim.AdamW(critic.parameters(), lr=1e-4),
                torch.Generator().manual_seed(12),
                torch.Generator().manual_seed(13),
                torch.Generator().manual_seed(14),
                torch.Generator().manual_seed(15),
                torch.Generator().manual_seed(16),
            )

        def train_epoch(parts):
            (
                actor, critic, actor_opt, critic_opt, lifecycle_rng, temporal_rng,
                dagger_rng, rare_rng, spawn_rng,
            ) = parts
            return _train_joint_lifecycle_epoch(
                actor, critic, actor_opt, critic_opt, lifecycle, returns, temporal,
                device=torch.device("cpu"), use_bf16=False, minibatch=2,
                shuffle_generator=lifecycle_rng,
                nextlat_shuffle_generator=temporal_rng,
                correction_replay=dagger_replay,
                correction_shuffle_generator=dagger_rng,
                rare_replay=rare_replay,
                rare_refs_by_intent={
                    "createConstructionSite": [(0, 0)],
                    "claimController": [(1, 0)],
                },
                rare_shuffle_generator=rare_rng,
                spawn_replay=spawn_replay,
                spawn_shuffle_generator=spawn_rng,
            )

        uninterrupted = initialized()
        uninterrupted_step = 0
        uninterrupted_result = None
        for _ in range(2):
            uninterrupted_result = train_epoch(uninterrupted)
            uninterrupted_step += uninterrupted_result[3]

        split = initialized()
        split_result = train_epoch(split)
        split_step = split_result[3]
        saved = {
            "actor": copy.deepcopy(split[0].state_dict()),
            "critic": copy.deepcopy(split[1].state_dict()),
            "actor_opt": copy.deepcopy(split[2].state_dict()),
            "critic_opt": copy.deepcopy(split[3].state_dict()),
            "lifecycle_rng": split[4].get_state().clone(),
            "temporal_rng": split[5].get_state().clone(),
            "dagger_rng": split[6].get_state().clone(),
            "rare_rng": split[7].get_state().clone(),
            "spawn_rng": split[8].get_state().clone(),
            "global_epoch": 1,
            "global_step": split_step,
        }
        resumed = initialized()
        resumed[0].load_state_dict(saved["actor"])
        resumed[1].load_state_dict(saved["critic"])
        resumed[2].load_state_dict(saved["actor_opt"])
        resumed[3].load_state_dict(saved["critic_opt"])
        resumed[4].set_state(saved["lifecycle_rng"])
        resumed[5].set_state(saved["temporal_rng"])
        resumed[6].set_state(saved["dagger_rng"])
        resumed[7].set_state(saved["rare_rng"])
        resumed[8].set_state(saved["spawn_rng"])
        resumed_result = train_epoch(resumed)
        resumed_step = saved["global_step"] + resumed_result[3]
        resumed_epoch = saved["global_epoch"] + 1

        for left, right in zip(uninterrupted[:2], resumed[:2], strict=True):
            for name, value in left.state_dict().items():
                self.assertTrue(torch.equal(value, right.state_dict()[name]), name)

        def assert_nested_equal(left, right):
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    assert_nested_equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for left_item, right_item in zip(left, right, strict=True):
                    assert_nested_equal(left_item, right_item)
            else:
                self.assertEqual(left, right)

        assert_nested_equal(uninterrupted[2].state_dict(), resumed[2].state_dict())
        assert_nested_equal(uninterrupted[3].state_dict(), resumed[3].state_dict())
        self.assertTrue(torch.equal(uninterrupted[4].get_state(), resumed[4].get_state()))
        self.assertTrue(torch.equal(uninterrupted[5].get_state(), resumed[5].get_state()))
        self.assertTrue(torch.equal(uninterrupted[6].get_state(), resumed[6].get_state()))
        self.assertTrue(torch.equal(uninterrupted[7].get_state(), resumed[7].get_state()))
        self.assertTrue(torch.equal(uninterrupted[8].get_state(), resumed[8].get_state()))
        self.assertEqual(uninterrupted_step, resumed_step)
        self.assertEqual(resumed_epoch, 2)
        self.assertIsNotNone(uninterrupted_result)
        for key in (
            "correction_exposures", "rare_exposures", "spawn_exposures",
            "optimizer_steps",
        ):
            self.assertEqual(
                uninterrupted_result[4][key], resumed_result[4][key], key,
            )

    def test_auxiliary_batches_are_evenly_scheduled_without_repetition(self) -> None:
        from .pretrain_joint import (
            _evenly_distribute_batches, _schedule_auxiliary_lanes,
        )

        batches = [[index] for index in range(11)]
        scheduled = _evenly_distribute_batches(batches, 4)
        flattened = [batch for slot in scheduled for batch in slot]
        self.assertEqual(flattened, batches)
        self.assertLessEqual(
            max(len(slot) for slot in scheduled)
            - min(len(slot) for slot in scheduled),
            1,
        )
        self.assertEqual(_evenly_distribute_batches([], 3), [[], [], []])
        with self.assertRaisesRegex(ValueError, "step count"):
            _evenly_distribute_batches([[0]], 0)
        lanes = {
            "dagger": [["d", index] for index in range(3)],
            "rare": [["r", index] for index in range(2)],
            "spawn": [["s", index] for index in range(4)],
        }
        lane_schedule = _schedule_auxiliary_lanes(lanes, 12)
        for name, source in lanes.items():
            self.assertEqual(
                [batch for slot in lane_schedule[name] for batch in slot],
                source,
            )
        self.assertTrue(all(
            sum(len(lane_schedule[name][slot]) for name in lanes) <= 1
            for slot in range(12)
        ))
        with self.assertRaisesRegex(
            ValueError, "exceeds collision-free lifecycle capacity",
        ):
            _schedule_auxiliary_lanes(lanes, 8)

    def test_persisted_auxiliary_geometry_is_qualified_exactly(self) -> None:
        from .pretrain_joint import _auxiliary_schedule_qualified

        geometry = {
            "nextlat_train_optimizer_steps": 475.0,
            "nextlat_train_correction_batches": 76.0,
            "nextlat_train_rare_batches": 6.0,
            "nextlat_train_spawn_batches": 180.0,
            "nextlat_train_auxiliary_batches": 262.0,
            "nextlat_train_auxiliary_minibatch": 32.0,
            "nextlat_train_correction_exposures": 2_432.0,
            "nextlat_train_rare_exposures": 166.0,
            "nextlat_train_spawn_exposures": 5_760.0,
        }
        self.assertTrue(_auxiliary_schedule_qualified(geometry, required=True))
        over_capacity = dict(
            geometry,
            nextlat_train_optimizer_steps=261.0,
        )
        self.assertFalse(_auxiliary_schedule_qualified(
            over_capacity, required=True,
        ))
        wrong_total = dict(
            geometry,
            nextlat_train_auxiliary_batches=261.0,
        )
        self.assertFalse(_auxiliary_schedule_qualified(
            wrong_total, required=True,
        ))
        self.assertFalse(_auxiliary_schedule_qualified({}, required=True))

    def test_production_preflight_rejects_oversize_before_training_init(self) -> None:
        import inspect

        from . import pretrain_joint
        from .pretrain_joint import DaggerSample, _preflight_joint_geometry

        corpus = _corpus()
        lifecycle = pretrain_joint._lifecycle_samples(
            corpus["data"]["train"]["lifecycle_replay"],
        )
        correction = [
            DaggerSample(
                kind="exact_intent", stratum=f"supplement:{index}",
                actor_index=0, obs=lifecycle[0].obs, action=lifecycle[0].action,
            )
            for index in range(2)
        ]
        with self.assertRaisesRegex(
            ValueError, "exceeds collision-free lifecycle capacity",
        ):
            _preflight_joint_geometry(
                lifecycle, correction,
                {
                    "createConstructionSite": [(0, 0)],
                    "claimController": [(1, 0)],
                },
                [], minibatch=1,
            )

        main_source = inspect.getsource(pretrain_joint.main)
        preflight_position = main_source.index("_preflight_joint_geometry(")
        self.assertLess(preflight_position, main_source.index("actor = Actor()"))
        self.assertLess(
            preflight_position,
            main_source.index("_should_run_ti_initialization("),
        )

    def test_dagger_build_supervises_type_not_arbitrary_site_identity(self) -> None:
        from .pretrain_joint import _dagger_factor_mask

        active = torch.ones(3, 14, dtype=torch.bool)
        selected = _dagger_factor_mask("build", active)
        self.assertTrue(selected[:, 0].all())
        self.assertFalse(selected[:, 1:].any())
        self.assertTrue(torch.equal(
            _dagger_factor_mask("harvest", active), active,
        ))

    def test_joint_epoch_critic_failure_is_transactional(self) -> None:
        from .model import Actor, Critic
        from .pretrain_joint import (
            _lifecycle_samples, _temporal_samples, _train_joint_lifecycle_epoch,
        )

        class CountingSGD(torch.optim.SGD):
            def __init__(self, parameters):
                super().__init__(parameters, lr=1e-3)
                self.step_calls = 0

            def step(self, closure=None):
                self.step_calls += 1
                return super().step(closure)

        cfg = dict(MODEL_CFG)
        cfg.update({"dModel": 16, "nHeads": 4, "spatialLayers": 1,
                    "entityLayers": 1, "ffMult": 2})
        corpus = _corpus()
        lifecycle = _lifecycle_samples(
            corpus["data"]["train"]["lifecycle_replay"],
        )
        temporal = _temporal_samples(
            corpus["data"]["train"]["temporal_replay"],
        )
        actor, critic = Actor(cfg), Critic(cfg)
        actor_before = copy.deepcopy(actor.state_dict())
        actor_opt = CountingSGD(actor.parameters())
        critic_opt = CountingSGD(critic.parameters())
        returns = corpus["data"]["train"]["returns_tn"].clone()
        returns[lifecycle[0].timestep, lifecycle[0].env_index] = float("nan")
        with self.assertRaises(FloatingPointError):
            _train_joint_lifecycle_epoch(
                actor, critic, actor_opt, critic_opt, lifecycle, returns, temporal,
                device=torch.device("cpu"), use_bf16=False, minibatch=2,
                shuffle_generator=torch.Generator().manual_seed(1),
                nextlat_shuffle_generator=torch.Generator().manual_seed(2),
            )
        self.assertEqual(actor_opt.step_calls, 0)
        self.assertEqual(critic_opt.step_calls, 0)
        for name, value in actor.state_dict().items():
            self.assertTrue(torch.equal(value, actor_before[name]), name)

    def test_nextlat_baselines_measure_action_conditioning(self) -> None:
        from .pretrain_joint import _evaluate_nextlat, _temporal_samples

        class ToyActor(torch.nn.Module):
            def encode_state(self, batch):
                return batch["globals"][:, :1]

            def predict_next_latent(self, state, _batch, action):
                return state + action["dirs"][:, :, 0].sum(
                    dim=-1, keepdim=True,
                ).to(state.dtype)

        class ToyCritic(ToyActor):
            def value_logits_and_latent(self, batch):
                latent = self.encode_state(batch)
                return torch.cat((latent, -latent), dim=-1), latent

            def detached_value_logits(self, latent):
                return torch.cat((latent, -latent), dim=-1)

        rows = copy.deepcopy(_corpus()["data"]["train"]["temporal_replay"])
        for index, row in enumerate(rows):
            state = float(index * 5)
            first_delta, second_delta = index + 1, 2
            row["obs"]["globals"][0, 0] = state
            row["action"]["types"][0, 1, 0] = INTENT_TYPES.index("move")
            row["action"]["dirs"][0, 0, 0] = first_delta
            row["action"]["dirs"][0, 1, 0] = second_delta
            row["next_obs"]["globals"][0, 0] = state + first_delta + second_delta
            # Evaluation must derive the whole-joint ablation, not trust the
            # collector's first-command counterfactual payload.
            row["counterfactual_action"] = None
        metrics = _evaluate_nextlat(
            ToyActor(), ToyCritic(), _temporal_samples(rows),
            device=torch.device("cpu"), minibatch=2,
        )
        self.assertEqual(metrics["actor_mse"], 0.0)
        self.assertEqual(metrics["critic_mse"], 0.0)
        self.assertEqual(metrics["counterfactual_rows"], 2.0)
        self.assertEqual(metrics["actor_counterfactual_action_mse"], 12.5)
        self.assertGreater(metrics["actor_identity_mse"], metrics["actor_mse"])
        self.assertGreater(
            metrics["actor_counterfactual_action_mse"], metrics["actor_mse"],
        )
        self.assertGreater(metrics["critic_identity_mse"], metrics["critic_mse"])
        self.assertGreater(
            metrics["critic_counterfactual_action_mse"], metrics["critic_mse"],
        )

    def test_nextlat_training_does_not_read_holdout_rows(self) -> None:
        from .model import Actor, Critic
        from .pretrain_joint import (
            _lifecycle_samples, _temporal_samples, _train_joint_lifecycle_epoch,
        )

        cfg = dict(MODEL_CFG)
        cfg.update({"dModel": 16, "nHeads": 4, "spatialLayers": 1,
                    "entityLayers": 1, "ffMult": 2})
        corpus = _corpus()
        train = _temporal_samples(corpus["data"]["train"]["temporal_replay"])
        holdout = _temporal_samples(corpus["data"]["holdout"]["temporal_replay"])
        for sample in holdout:
            sample.obs["globals"].requires_grad_()
            sample.next_obs["globals"].requires_grad_()
        actor, critic = Actor(cfg), Critic(cfg)
        lifecycle = _lifecycle_samples(
            corpus["data"]["train"]["lifecycle_replay"],
        )
        _train_joint_lifecycle_epoch(
            actor, critic, torch.optim.SGD(actor.parameters(), lr=1e-4),
            torch.optim.SGD(critic.parameters(), lr=1e-4), lifecycle,
            corpus["data"]["train"]["returns_tn"], train,
            device=torch.device("cpu"), use_bf16=False, minibatch=2,
            shuffle_generator=torch.Generator().manual_seed(3),
            nextlat_shuffle_generator=torch.Generator().manual_seed(4),
        )
        for sample in holdout:
            self.assertIsNone(sample.obs["globals"].grad)
            self.assertIsNone(sample.next_obs["globals"].grad)

    def test_joint_temporal_stream_cycles_deterministically(self) -> None:
        from .pretrain_joint import _independent_cyclic_order

        first = _independent_cyclic_order(
            3, 8, generator=torch.Generator().manual_seed(7),
        )
        second = _independent_cyclic_order(
            3, 8, generator=torch.Generator().manual_seed(7),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.numel(), 8)
        self.assertEqual(sorted(first[:3].tolist()), [0, 1, 2])
        self.assertEqual(sorted(first[3:6].tolist()), [0, 1, 2])

    def test_ti_warm_start_is_fresh_run_only(self) -> None:
        from .pretrain_joint import _should_run_ti_initialization

        self.assertTrue(_should_run_ti_initialization(global_epoch=0, resume=None))
        self.assertFalse(_should_run_ti_initialization(
            global_epoch=0, resume=Path("partial.pt"),
        ))
        self.assertFalse(_should_run_ti_initialization(global_epoch=1, resume=None))

    def test_auxiliary_lifecycle_retention_rejects_material_overwrite(self) -> None:
        from .pretrain_joint import _auxiliary_lifecycle_retained

        self.assertTrue(_auxiliary_lifecycle_retained(0.2, 0.21, max_ratio=1.1))
        self.assertFalse(_auxiliary_lifecycle_retained(0.2, 0.23, max_ratio=1.1))
        self.assertFalse(_auxiliary_lifecycle_retained(
            float("nan"), 0.2, max_ratio=1.1,
        ))

    def test_rare_intent_qualification_requires_independent_row_floor(self) -> None:
        from .pretrain_joint import _rare_intent_replay_qualified

        validation = {}
        for split, count in (("train", 48.0), ("holdout", 33.0)):
            validation.update({
                f"rare_{split}_createConstructionSite_count": count,
                f"rare_{split}_createConstructionSite_legal_frac": 1.0,
                f"rare_{split}_createConstructionSite_nll": 0.2,
                f"rare_{split}_createConstructionSite_accuracy": 0.9,
            })
        self.assertTrue(_rare_intent_replay_qualified(
            validation, "createConstructionSite",
            min_rows=32, max_nll=1.0, min_accuracy=0.8,
        ))
        validation["rare_holdout_createConstructionSite_count"] = 31.0
        self.assertFalse(_rare_intent_replay_qualified(
            validation, "createConstructionSite",
            min_rows=32, max_nll=1.0, min_accuracy=0.8,
        ))

    def test_correction_sources_qualify_independently(self) -> None:
        from .pretrain_joint import _correction_source_qualified

        metrics = {
            "_actor_supplement_actor_rows": 96.0,
            "_actor_supplement_actor_nll": 0.2,
            "_actor_supplement_actor_legal_frac": 1.0,
            "_actor_supplement_accuracy": 0.9,
            "_actor_supplement_min_intent_accuracy": 0.85,
        }
        self.assertTrue(_correction_source_qualified(
            metrics, prefix="actor_supplement",
            artifact_sha256="a" * 64, min_accuracy=0.8,
        ))
        self.assertTrue(_correction_source_qualified(
            {}, prefix="dagger", artifact_sha256=None, min_accuracy=0.8,
        ))
        metrics["_actor_supplement_min_intent_accuracy"] = 0.79
        self.assertFalse(_correction_source_qualified(
            metrics, prefix="actor_supplement",
            artifact_sha256="a" * 64, min_accuracy=0.8,
        ))


if __name__ == "__main__":
    unittest.main()
