"""Engine-backed contracts for start-state snapshots.

A start-state reservoir is only usable if a restored world is the same MDP
state the capturing environment observed. These tests establish that with the
strongest available evidence: byte-identical observations and byte-identical
continuations under an identical action stream, in a different process.

Executor route caches are per-process operational state and are deliberately
cold at every episode and segment start, so the capturing environment restores
its own snapshot before continuing. Any residual divergence would mean engine
state was lost by capture or restore, which is what these tests must catch.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .env_client import ScreepsEnv

_CURRICULUM = "seed_outpost"


def _obs_digest(obs: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(obs):
        value = obs[key]
        if not torch.is_tensor(value):
            continue
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class StateSnapshotEngineTest(unittest.TestCase):
    def test_restored_world_continues_the_captured_trajectory_exactly(self) -> None:
        with patch.dict(os.environ, {"RL_OBS_FMT": "bin"}), tempfile.TemporaryDirectory() as work:
            path = str(Path(work) / "mid.xsnp")
            source = ScreepsEnv(
                max_episode=20000, curriculum=_CURRICULUM, lean_meta=True, seed=3,
            )
            try:
                source.reset()
                for _ in range(400):
                    source.step_scripted()
                descriptor = source.snapshot(path, events=["periodic"])
                self.assertEqual(descriptor["step"], 400)
                self.assertGreater(descriptor["bytes"], 0)
                self.assertIn("W7N3", descriptor["rooms"])
                self.assertEqual(descriptor["curriculum"], _CURRICULUM)
                self.assertEqual(descriptor["events"], ["periodic"])
                captured = source.restore(path)
                baseline = []
                for _ in range(30):
                    obs, reward, _, info, _ = source.step_scripted()
                    baseline.append((_obs_digest(obs), round(float(reward), 6), info["time"]))
            finally:
                source.close()

            replay = ScreepsEnv(
                max_episode=20000, curriculum=_CURRICULUM, lean_meta=True, seed=3,
            )
            try:
                replay.reset()
                restored = replay.restore(path)
                self.assertEqual(_obs_digest(restored), _obs_digest(captured))
                self.assertTrue(replay.last_info["restored"])
                self.assertEqual(replay.last_info["step"], 400)
                self.assertEqual(replay.last_info["time"], descriptor["tick"])
                continued = []
                for _ in range(30):
                    obs, reward, _, info, _ = replay.step_scripted()
                    continued.append((_obs_digest(obs), round(float(reward), 6), info["time"]))
            finally:
                replay.close()

        self.assertEqual(baseline, continued)

    def test_restore_is_exact_from_several_economy_phases(self) -> None:
        """Boundary crossings, spawning, and replacement must all restore exactly."""
        with patch.dict(os.environ, {"RL_OBS_FMT": "bin"}), tempfile.TemporaryDirectory() as work:
            checkpoints = {}
            source = ScreepsEnv(
                max_episode=20000, curriculum=_CURRICULUM, lean_meta=True, seed=5,
            )
            try:
                for index in range(4):
                    if index == 0:
                        source.reset()
                    for _ in range(211):
                        source.step_scripted()
                    path = str(Path(work) / f"phase{index}.xsnp")
                    source.snapshot(path)
                    # Compare against a cold-cache continuation of the same state.
                    source.restore(path)
                    trace = []
                    for _ in range(12):
                        obs, reward, _, _, _ = source.step_scripted()
                        trace.append((_obs_digest(obs), round(float(reward), 6)))
                    checkpoints[path] = trace
                    source.restore(path)
            finally:
                source.close()

            replay = ScreepsEnv(
                max_episode=20000, curriculum=_CURRICULUM, lean_meta=True, seed=5,
            )
            try:
                replay.reset()
                for path, expected in checkpoints.items():
                    replay.restore(path)
                    trace = []
                    for _ in range(12):
                        obs, reward, _, _, _ = replay.step_scripted()
                        trace.append((_obs_digest(obs), round(float(reward), 6)))
                    self.assertEqual(trace, expected, f"divergence after restoring {path}")
            finally:
                replay.close()

    def test_incompatible_snapshot_is_rejected(self) -> None:
        with patch.dict(os.environ, {"RL_OBS_FMT": "bin"}), tempfile.TemporaryDirectory() as work:
            path = Path(work) / "foreign.xsnp"
            source = ScreepsEnv(max_episode=64, curriculum="empty", lean_meta=True, seed=7)
            try:
                source.reset()
                source.step_scripted()
                source.snapshot(str(path))
            finally:
                source.close()
            payload = bytearray(path.read_bytes())
            meta_length = int.from_bytes(payload[8:12], "little")
            meta = payload[16 : 16 + meta_length].decode("utf-8")
            self.assertIn('"room":"W7N3"', meta.replace(" ", ""))
            path.write_bytes(bytes(payload).replace(b'"room":"W7N3"', b'"room":"W1N1"'))
            target = ScreepsEnv(max_episode=64, curriculum="empty", lean_meta=True, seed=7)
            try:
                target.reset()
                with self.assertRaises(RuntimeError) as error:
                    target.restore(str(path))
                self.assertIn("does not match environment", str(error.exception))
            finally:
                target.close()

    def test_restore_preserves_the_claim_budget(self) -> None:
        """A restored world must not refresh its global-control-level room cap.

        The controlled-room set lives in processor scratch, which a fresh shard
        flushes. Without capturing it, every restored segment could claim past
        the cap that a fresh evaluation world enforces.
        """
        with patch.dict(os.environ, {"RL_OBS_FMT": "bin"}), tempfile.TemporaryDirectory() as work:
            path = Path(work) / "claimed.xsnp"
            source = ScreepsEnv(
                max_episode=20000, curriculum="seed_claimer", lean_meta=True, seed=3,
            )
            try:
                info = source.reset()
                self.assertEqual(source.last_info["controlledRooms"], 0)
                owned = 1
                for _ in range(2000):
                    _, _, _, info, _ = source.step_scripted()
                    owned = int(info["ownedRooms"])
                    if owned >= 2:
                        break
                self.assertGreaterEqual(owned, 2, "scripted claimer never claimed")
                source.snapshot(str(path))
            finally:
                source.close()

            payload = path.read_bytes()
            meta_length = int.from_bytes(payload[8:12], "little")
            meta = json.loads(payload[16 : 16 + meta_length].decode("utf-8"))
            controlled = meta["world"]["scratch"]["controlledRooms"]["100"]
            self.assertEqual(len(controlled), 1, controlled)

            target = ScreepsEnv(
                max_episode=20000, curriculum="seed_claimer", lean_meta=True, seed=3,
            )
            try:
                target.reset()
                target.restore(str(path))
                self.assertEqual(target.last_info["controlledRooms"], 1)
                _, _, _, resumed, _ = target.step_scripted()
                self.assertEqual(int(resumed["ownedRooms"]), 2)
                self.assertEqual(int(resumed["claimDelta"]), 0)
            finally:
                target.close()

    def test_teacher_state_replays_identically_in_a_learner_session(self) -> None:
        """A bridge-lane start must be reproducible even though the teacher is not.

        The teacher's own decisions vary between identical runs, so a corpus of
        teacher trajectories is a sample rather than a replay. The reservoir does
        not inherit that: what it stores is a world, and a stored world reopened
        without expert code must continue identically every time. If this fails,
        bridge starts are unusable for matched comparisons.
        """
        with patch.dict(os.environ, {"RL_OBS_FMT": "bin"}), tempfile.TemporaryDirectory() as work:
            path = str(Path(work) / "teacher.xsnp")
            teacher = ScreepsEnv(
                max_episode=20000, curriculum="seed_outpost", lean_meta=True,
                seed=77, expert=True,
            )
            try:
                teacher.reset()
                for _ in range(300):
                    teacher.step()
                descriptor = teacher.snapshot(path, events=["periodic"])
                self.assertGreater(descriptor["bytes"], 0)
            finally:
                teacher.close()

            traces = []
            for _ in range(2):
                learner = ScreepsEnv(
                    max_episode=20000, curriculum="empty", lean_meta=True, seed=5,
                )
                try:
                    learner.reset()
                    learner.restore(path)
                    self.assertTrue(learner.last_info["restored"])
                    self.assertFalse(learner.expert)
                    trace = []
                    for _ in range(60):
                        obs, reward, _, _, _ = learner.step_scripted()
                        trace.append((_obs_digest(obs), round(float(reward), 6)))
                    traces.append(trace)
                finally:
                    learner.close()

        self.assertEqual(traces[0], traces[1])

    def test_event_tags_cover_the_stratification_contract(self) -> None:
        """The environment must emit pre-decision tags, not only outcomes."""
        with patch.dict(os.environ, {"RL_OBS_FMT": "bin"}):
            env = ScreepsEnv(
                max_episode=1200, curriculum="seed_outpost", lean_meta=True, seed=3,
            )
            seen: set[str] = set()
            try:
                env.reset()
                for _ in range(1100):
                    _, _, done, info, _ = env.step_scripted()
                    seen.update(info.get("events") or [])
                    if done:
                        break
            finally:
                env.close()
        for tag in ("pre_spawn", "post_spawn", "remote_outbound", "remote_at_source"):
            self.assertIn(tag, seen)


if __name__ == "__main__":
    unittest.main()
