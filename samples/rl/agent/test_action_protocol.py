"""Focused XAC1 binary command protocol tests (including real Node round trips)."""
from __future__ import annotations

import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .constants import (
    INTENT_SLOTS, INTENT_TYPES, MAX_ACTORS,
    N_BODY_PART, N_CONSTRUCTION_TILE, SCHEMA,
)
from .actions_util import pad_actions
from .env_client import (
    ScreepsEnv,
    _decode_teacher_actions,
    _encode_binary_command,
    _json_command,
    _teacher_actions_from_response,
    _teacher_action_payload_bytes,
)


class BinaryCommandEncodingTest(unittest.TestCase):
    def test_step_frame_layout(self) -> None:
        actions = {
            "types": np.array([[[1], [2]]], dtype=np.int64),
            "dirs": np.array([[[3], [4]]], dtype=np.int64),
            "targets": np.array([[[5], [6]]], dtype=np.int64),
            "amounts": np.array([[[7], [8]]], dtype=np.int64),
            "construction_types": np.array([[[2], [3]]], dtype=np.int64),
            "construction_tiles": np.array([[[513], [2499]]], dtype=np.int64),
            "body_counts": np.zeros((1, 2, 1, N_BODY_PART), dtype=np.int64),
            "body_order": np.tile(np.arange(N_BODY_PART), (1, 2, 1, 1)),
        }
        actions["body_counts"][0, 0, 0, :2] = [1, 1]
        actions["body_order"][0, 0, 0] = [1, 0, 2, 3, 4, 5, 6, 7]
        actions["body_counts"][0, 1, 0, 3] = 1
        actions["body_order"][0, 1, 0] = [3, 0, 1, 2, 4, 5, 6, 7]
        frame = _encode_binary_command({"cmd": "step", "actions": actions})
        header = struct.unpack("<4sBBHIHBB", frame[:16])
        # Command protocol 7 dropped the scripted-label opcode; snapshot and
        # restore moved down to 8 and 9.
        self.assertEqual(header, (b"XAC1", 7, 3, SCHEMA["version"], 46, 2, 1, 0))
        self.assertEqual(
            frame[16:26], bytes([1, 2, 3, 4, 5, 6, 7, 8, 2, 3]),
        )
        self.assertEqual(frame[26:30], bytes([1, 2, 195, 9]))
        self.assertEqual(frame[30:38], bytes([1, 1, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(frame[46:54], bytes([1, 0, 2, 3, 4, 5, 6, 7]))

    def test_rejects_mismatched_shapes_and_out_of_range_categories(self) -> None:
        actions = {
            name: np.zeros((MAX_ACTORS, INTENT_SLOTS), dtype=np.int64)
            for name in (
                "types", "dirs", "targets", "amounts", "construction_types",
                "construction_tiles",
            )
        }
        actions["body_counts"] = np.zeros((MAX_ACTORS, INTENT_SLOTS, N_BODY_PART), dtype=np.int64)
        actions["body_order"] = np.tile(np.arange(N_BODY_PART), (MAX_ACTORS, INTENT_SLOTS, 1))
        actions["dirs"] = np.zeros((MAX_ACTORS - 1, INTENT_SLOTS), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "plane shapes differ"):
            _encode_binary_command({"cmd": "step", "actions": actions})

        actions["dirs"] = np.zeros((MAX_ACTORS, INTENT_SLOTS), dtype=np.int64)
        actions["targets"][0, 0] = SCHEMA["maxTargets"]
        with self.assertRaisesRegex(ValueError, r"actions.targets values must be in"):
            _encode_binary_command({"cmd": "step", "actions": actions})

    def test_json_and_binary_share_validation_and_unbatched_shape(self) -> None:
        shape = (1, INTENT_SLOTS)
        actions = {
            name: np.zeros(shape, dtype=np.int64)
            for name in (
                "types", "dirs", "targets", "amounts", "construction_types",
                "construction_tiles",
            )
        }
        actions["body_counts"] = np.zeros((1, INTENT_SLOTS, N_BODY_PART), dtype=np.int64)
        actions["body_order"] = np.tile(np.arange(N_BODY_PART), (1, INTENT_SLOTS, 1))
        encoded = _json_command({"cmd": "step", "actions": actions})["actions"]
        self.assertEqual(np.asarray(encoded["types"]).shape, shape)
        self.assertEqual(np.asarray(encoded["bodyOrder"]).shape, (1, 1, N_BODY_PART))

        bad = dict(actions)
        bad["types"] = np.array([[1.5]])
        for encoder in (_encode_binary_command, _json_command):
            with self.assertRaisesRegex(ValueError, "finite integers"):
                encoder({"cmd": "step", "actions": bad})


class BinaryTeacherActionEncodingTest(unittest.TestCase):
    @staticmethod
    def _actions(rows: int = 2) -> dict[str, np.ndarray]:
        shape = (rows, INTENT_SLOTS)
        actions = {
            name: np.zeros(shape, dtype=np.int64)
            for name in (
                "types", "dirs", "targets", "amounts", "construction_types",
                "construction_tiles",
            )
        }
        actions["types"][0, 0] = INTENT_TYPES.index("spawnCreep")
        actions["dirs"][0, 0] = 3
        actions["targets"][0, 0] = 7
        actions["amounts"][0, 0] = 2
        actions["construction_types"][0, 0] = 4
        actions["construction_tiles"][0, 0] = 513
        actions["body_counts"] = np.zeros(
            (rows, INTENT_SLOTS, N_BODY_PART), dtype=np.int64,
        )
        actions["body_order"] = np.tile(
            np.arange(N_BODY_PART), (rows, INTENT_SLOTS, 1),
        )
        actions["body_counts"][0, 0, :2] = [1, 2]
        actions["body_order"][0, 0] = [1, 0, 2, 3, 4, 5, 6, 7]
        return actions

    def test_teacher_payload_matches_command_plane_layout_exactly(self) -> None:
        actions = self._actions()
        # XAC1 and the XRL1 teacher tail deliberately share their complete plane
        # ordering, so this also protects JSON-command/binary-response parity.
        payload = _encode_binary_command({"cmd": "step", "actions": actions})[16:]
        self.assertEqual(
            len(payload), _teacher_action_payload_bytes(2, INTENT_SLOTS),
        )
        decoded = _decode_teacher_actions(payload, rows=2, slots=INTENT_SLOTS)
        for key, expected in actions.items():
            torch.testing.assert_close(
                decoded[key], torch.as_tensor(expected, dtype=torch.long),
                rtol=0, atol=0,
            )

    def test_teacher_payload_rejects_malformed_length_and_body_order(self) -> None:
        payload = bytearray(
            _encode_binary_command({"cmd": "step", "actions": self._actions(1)})[16:]
        )
        with self.assertRaisesRegex(RuntimeError, "payload length"):
            _decode_teacher_actions(payload[:-1], rows=1, slots=INTENT_SLOTS)

        # Five scalar planes + uint16 tile + body-count plane precede body order.
        body_order_offset = 5 + 2 + N_BODY_PART
        payload[body_order_offset + 1] = payload[body_order_offset]
        with self.assertRaisesRegex(RuntimeError, "permutation"):
            _decode_teacher_actions(payload, rows=1, slots=INTENT_SLOTS)

        payload = bytearray(
            _encode_binary_command({"cmd": "step", "actions": self._actions(1)})[16:]
        )
        body_counts_offset = 5 + 2
        payload[body_counts_offset] = SCHEMA["maxBodyParts"] + 1
        with self.assertRaisesRegex(RuntimeError, "body_counts"):
            _decode_teacher_actions(payload, rows=1, slots=INTENT_SLOTS)

    def test_response_rejects_mismatched_teacher_descriptor_length(self) -> None:
        payload = bytes(_teacher_action_payload_bytes(1, INTENT_SLOTS))
        metadata = json.dumps({
            "ok": True,
            "done": False,
            "teacherActions": {
                "rows": 1,
                "slots": INTENT_SLOTS,
                "byteLength": len(payload) - 1,
            },
        }).encode("utf-8")
        frame = (
            struct.pack(
                "<4sBBHII", b"XRL1", 4, 0x09, SCHEMA["version"],
                len(metadata), len(payload),
            )
            + metadata
            + payload
        )
        stream = io.BytesIO(frame)
        env = object.__new__(ScreepsEnv)
        env._stderr_tail = []
        env._read_exact = stream.read
        with self.assertRaisesRegex(RuntimeError, "payload length descriptor"):
            env._read_bin_frame()

    def test_json_teacher_response_is_strictly_validated(self) -> None:
        wire = _json_command({"cmd": "step", "actions": self._actions(1)})["actions"]
        decoded = _teacher_actions_from_response(
            {}, {"actions": wire}, binary=False, command="step_scripted",
        )
        self.assertEqual(tuple(decoded["types"].shape), (1, INTENT_SLOTS))

        malformed = {**wire, "bodyOrder": []}
        with self.assertRaisesRegex(RuntimeError, "invalid JSON step_scripted teacher actions"):
            _teacher_actions_from_response(
                {}, {"actions": malformed}, binary=False, command="step_scripted",
            )

    def test_response_rejects_redundant_flag_and_observation_length_disagreement(self) -> None:
        def read_frame(flags: int, metadata: dict, blob: bytes = b"") -> None:
            encoded = json.dumps(metadata).encode("utf-8")
            frame = (
                struct.pack(
                    "<4sBBHII", b"XRL1", 4, flags, SCHEMA["version"],
                    len(encoded), len(blob),
                )
                + encoded
                + blob
            )
            env = object.__new__(ScreepsEnv)
            env._stderr_tail = []
            env._read_exact = io.BytesIO(frame).read
            env._read_bin_frame()

        with self.assertRaisesRegex(RuntimeError, "ok flag"):
            read_frame(0, {"ok": True, "done": False})
        with self.assertRaisesRegex(RuntimeError, "done flag"):
            read_frame(0x03, {"ok": True, "done": False})
        with self.assertRaisesRegex(RuntimeError, "observation flag"):
            read_frame(0x05, {"ok": True, "done": False})
        with self.assertRaisesRegex(RuntimeError, "observation payload length"):
            read_frame(0x05, {
                "ok": True,
                "done": False,
                "shapes": {
                    "patches": [1], "actors": [1], "targets": [1],
                    "roomCoords": [1], "intentMask": [1], "dirMask": [1],
                    "targetSelectMask": [1], "amountMask": [1],
                    "constructionMask": [1],
                },
            })


class BinaryCommandServerTest(unittest.TestCase):
    def _env(self, command_format: str) -> ScreepsEnv:
        return ScreepsEnv(max_episode=4, command_format=command_format)

    @staticmethod
    def _zero_actions(batch: int = 1) -> dict[str, torch.Tensor]:
        shape = (batch, MAX_ACTORS, INTENT_SLOTS)
        actions = {
            name: torch.zeros(shape, dtype=torch.long)
            for name in (
                "types", "dirs", "targets", "amounts", "construction_types",
                "construction_tiles",
            )
        }
        actions["body_counts"] = torch.zeros(
            batch, MAX_ACTORS, INTENT_SLOTS, N_BODY_PART, dtype=torch.long,
        )
        actions["body_order"] = torch.arange(N_BODY_PART).view(
            1, 1, 1, N_BODY_PART,
        ).expand(batch, MAX_ACTORS, INTENT_SLOTS, N_BODY_PART).clone()
        return actions

    def test_binary_schema_roundtrip_and_recoverable_validation_error(self) -> None:
        with patch.dict(os.environ, {"RL_OBS_FMT": "bin"}):
            env = self._env("bin")
            try:
                fragmented = _encode_binary_command({"cmd": "schema"})
                env._stdin.write(fragmented[:7])
                env._stdin.flush()
                env._stdin.write(fragmented[7:])
                env._stdin.flush()
                self.assertEqual(
                    env._read_bin_frame()["schema"]["version"], SCHEMA["version"]
                )

                invalid = bytearray(_encode_binary_command({"cmd": "schema"}))
                invalid[15] = 1  # reserved flags must remain zero
                env._stdin.write(invalid)
                env._stdin.flush()
                with self.assertRaisesRegex(RuntimeError, "unsupported command flags 1"):
                    env._read_bin_frame()

                actions = {
                    name: np.zeros((1, INTENT_SLOTS), dtype=np.int64)
                    for name in (
                        "types", "dirs", "targets", "amounts", "construction_types",
                        "construction_tiles",
                    )
                }
                actions["body_counts"] = np.zeros((1, INTENT_SLOTS, N_BODY_PART), dtype=np.int64)
                actions["body_order"] = np.tile(np.arange(N_BODY_PART), (1, INTENT_SLOTS, 1))
                invalid = bytearray(_encode_binary_command({"cmd": "step", "actions": actions}))
                invalid[16] = len(SCHEMA["intentTypes"])
                env._stdin.write(invalid)
                env._stdin.flush()
                with self.assertRaisesRegex(RuntimeError, r"actions.types\[0\].*outside"):
                    env._read_bin_frame()

                env.reset()
                invalid_order = bytearray(
                    _encode_binary_command({"cmd": "step", "actions": actions})
                )
                # Five scalar bytes, one uint16 tile, and eight counts precede order.
                invalid_order[16 + 5 + 2 + 8 + 1] = 0
                env._stdin.write(invalid_order)
                env._stdin.flush()
                with self.assertRaisesRegex(RuntimeError, "bodyOrder.*not a permutation"):
                    env._read_bin_frame()

                data = env._rpc({"cmd": "schema"})
                self.assertEqual(data["schema"]["version"], SCHEMA["version"])
            finally:
                env.close()

    def test_reset_step_scripted_and_json_debug_mode(self) -> None:
        actions = {
            name: torch.zeros((1, MAX_ACTORS, INTENT_SLOTS), dtype=torch.long)
            for name in (
                "types", "dirs", "targets", "amounts", "construction_types",
                "construction_tiles",
            )
        }
        actions["body_counts"] = torch.zeros(1, MAX_ACTORS, INTENT_SLOTS, N_BODY_PART, dtype=torch.long)
        actions["body_order"] = torch.arange(N_BODY_PART).view(1, 1, 1, N_BODY_PART).expand(
            1, MAX_ACTORS, INTENT_SLOTS, N_BODY_PART,
        )
        actions["types"][0, 0, 0] = SCHEMA["intentTypes"].index("spawnCreep")
        actions["body_counts"][0, 0, 0, 0] = 1
        with patch.dict(os.environ, {"RL_OBS_FMT": "bin"}):
            for command_format in ("bin", "json"):
                with self.subTest(command_format=command_format):
                    env = self._env(command_format)
                    try:
                        obs = env.reset()
                        self.assertEqual(tuple(obs["actors"].shape), (1, MAX_ACTORS, SCHEMA["actorFeat"]))
                        _, _, done, info = env.step(actions)
                        self.assertFalse(done)
                        self.assertEqual(info["step"], 1)
                        self.assertEqual(info["intentIssued"], 1)
                        self.assertIn("spawnCreep", info["intentByType"])
                        _, _, done, info, teacher = env.step_scripted()
                        self.assertFalse(done)
                        self.assertEqual(info["step"], 2)
                        self.assertEqual(
                            set(teacher),
                            {
                                "types", "dirs", "targets", "amounts",
                                "body_counts", "body_order",
                                "construction_types", "construction_tiles",
                            },
                        )
                    finally:
                        env.close()

    def test_binary_and_json_scripted_responses_have_exact_parity(self) -> None:
        results = {}
        for response_format in ("bin", "json"):
            with patch.dict(os.environ, {"RL_OBS_FMT": response_format}):
                env = ScreepsEnv(
                    max_episode=4, command_format="bin", lean_meta=True, seed=91,
                )
                try:
                    env.reset()
                    obs, reward, done, info, actions = env.step_scripted()
                    results[response_format] = (
                        obs, reward, done, info, pad_actions(actions),
                    )
                finally:
                    env.close()

        bin_obs, bin_reward, bin_done, bin_info, bin_actions = results["bin"]
        json_obs, json_reward, json_done, json_info, json_actions = results["json"]
        self.assertEqual((bin_reward, bin_done), (json_reward, json_done))
        self.assertNotIn("actions", bin_info)
        self.assertIn("actions", json_info)
        for key in bin_actions:
            torch.testing.assert_close(
                bin_actions[key], json_actions[key], rtol=0, atol=0,
            )
        for key, value in bin_obs.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, json_obs[key], rtol=0, atol=0)

    def test_binary_and_json_materialize_the_same_grouped_spawn_body(self) -> None:
        observed: dict[str, list[int]] = {}
        with patch.dict(os.environ, {"RL_OBS_FMT": "bin"}):
            for command_format in ("bin", "json"):
                env = ScreepsEnv(
                    max_episode=4, command_format=command_format, lean_meta=False,
                )
                try:
                    obs = env.reset()
                    actor = next(
                        i for i, meta in enumerate(obs["_actor_meta"])
                        if meta.get("kind") == "structure"
                    )
                    shape = (1, MAX_ACTORS, INTENT_SLOTS)
                    actions = {
                        name: torch.zeros(shape, dtype=torch.long)
                        for name in (
                            "types", "dirs", "targets", "amounts",
                            "construction_types", "construction_tiles",
                        )
                    }
                    actions["body_counts"] = torch.zeros(
                        1, MAX_ACTORS, INTENT_SLOTS, N_BODY_PART, dtype=torch.long,
                    )
                    actions["body_order"] = torch.arange(N_BODY_PART).view(
                        1, 1, 1, N_BODY_PART,
                    ).expand(1, MAX_ACTORS, INTENT_SLOTS, N_BODY_PART).clone()
                    actions["types"][0, actor, 0] = INTENT_TYPES.index("spawnCreep")
                    actions["body_counts"][0, actor, 0, :3] = torch.tensor([2, 1, 2])
                    actions["body_order"][0, actor, 0] = torch.tensor(
                        [1, 2, 0, 3, 4, 5, 6, 7],
                    )
                    _, _, _, info = env.step(actions)
                    result = next(
                        row for row in info["intentResults"]
                        if row["type"] == "spawnCreep"
                    )
                    self.assertEqual(result["code"], 0)
                    observed[command_format] = result["spawnBodyParts"]
                finally:
                    env.close()
        self.assertEqual(observed, {
            "bin": [1, 2, 2, 0, 0],
            "json": [1, 2, 2, 0, 0],
        })

    def test_construction_mask_matches_real_engine_execution(self) -> None:
        with patch.dict(os.environ, {"RL_OBS_FMT": "bin"}):
            env = ScreepsEnv(max_episode=5, command_format="bin", lean_meta=False)
            try:
                obs = env.reset()
                actor = next(
                    i for i, meta in enumerate(obs["_actor_meta"])
                    if meta.get("kind") == "room"
                )
                self.assertFalse(any(
                    meta.get("kind") == "position" for meta in obs["_target_meta"]
                ))
                packed = obs["construction_mask"][0, 0]
                legal_pair = next(
                    (type_index, tile)
                    for type_index in range(len(SCHEMA["constructionTypes"]))
                    for tile in range(N_CONSTRUCTION_TILE)
                    if int(packed[type_index, tile // 8]) & (1 << (tile % 8))
                )
                type_index, legal_tile = legal_pair
                illegal_tile = next(
                    tile for tile in range(N_CONSTRUCTION_TILE)
                    if not int(packed[type_index, tile // 8]) & (1 << (tile % 8))
                )

                shape = (1, MAX_ACTORS, INTENT_SLOTS)
                actions = {
                    name: torch.zeros(shape, dtype=torch.long)
                    for name in (
                        "types", "dirs", "targets", "amounts",
                        "construction_types", "construction_tiles",
                    )
                }
                actions["body_counts"] = torch.zeros(
                    1, MAX_ACTORS, INTENT_SLOTS, N_BODY_PART, dtype=torch.long,
                )
                actions["body_order"] = torch.arange(N_BODY_PART).view(
                    1, 1, 1, N_BODY_PART,
                ).expand(1, MAX_ACTORS, INTENT_SLOTS, N_BODY_PART)
                actions["types"][0, actor, 0] = INTENT_TYPES.index(
                    "createConstructionSite"
                )
                actions["construction_types"][0, actor, 0] = type_index
                actions["construction_tiles"][0, actor, 0] = illegal_tile
                _, _, _, info = env.step(actions)
                rejected = next(
                    row for row in info["intentResults"]
                    if row["type"] == "createConstructionSite"
                )
                self.assertNotEqual(rejected["code"], 0)

                actions["construction_tiles"][0, actor, 0] = legal_tile
                next_obs, _, _, info = env.step(actions)
                accepted = next(
                    row for row in info["intentResults"]
                    if row["type"] == "createConstructionSite"
                )
                self.assertEqual(accepted["code"], 0)
                self.assertEqual(accepted["constructionType"], type_index)
                self.assertEqual(accepted["constructionTile"], legal_tile)
                self.assertFalse(
                    int(next_obs["construction_mask"][0, 0, type_index, legal_tile // 8])
                    & (1 << (legal_tile % 8))
                )
            finally:
                env.close()

    def test_bad_magic_replies_then_terminates_instead_of_hanging(self) -> None:
        with patch.dict(os.environ, {"RL_OBS_FMT": "bin"}):
            env = self._env("bin")
            try:
                env._stdin.write(b"NOPE" + bytes(12))
                env._stdin.flush()
                with self.assertRaisesRegex(RuntimeError, "refusing unsafe resynchronization"):
                    env._read_bin_frame()
                self.assertNotEqual(env.proc.wait(timeout=2), 0)
            finally:
                env.close()

    def test_restored_world_keeps_its_own_scenario_rooms(self) -> None:
        """A reservoir lane restores several curricula into one env process.

        Room exposure and stage attribution are properties of the restored world,
        not of the process's `RL_CURRICULUM`. `spawn_claimer_650` carries the
        connected expansion room and `spawn_flexible_300` does not, so restoring
        both into a flexible process is what catches a process-level rule.
        """
        with tempfile.TemporaryDirectory() as tmp:
            captured: dict[str, tuple[str, int]] = {}
            paths: dict[str, Path] = {}
            for stage in ("spawn_flexible_300", "spawn_claimer_650"):
                path = Path(tmp) / f"{stage}.xsnp"
                paths[stage] = path
                env = ScreepsEnv(max_episode=8, curriculum=stage)
                try:
                    obs = env.reset()
                    env.snapshot(str(path))
                    captured[stage] = (
                        str(env.last_info["curriculum"]),
                        int(obs["room_mask"].sum()),
                    )
                finally:
                    env.close()
            self.assertEqual(captured["spawn_flexible_300"][1], 1)
            self.assertEqual(captured["spawn_claimer_650"][1], 2)

            env = ScreepsEnv(max_episode=8, curriculum="spawn_flexible_300")
            try:
                env.reset()
                for stage in (
                    "spawn_flexible_300", "spawn_claimer_650",
                    "spawn_flexible_300", "spawn_claimer_650",
                ):
                    obs = env.restore(str(paths[stage]))
                    self.assertEqual(
                        (str(env.last_info["curriculum"]), int(obs["room_mask"].sum())),
                        captured[stage],
                        f"{stage} lost state when restored after another scenario",
                    )
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
