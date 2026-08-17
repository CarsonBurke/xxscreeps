"""Contracts for the event-stratified start-state reservoir.

These tests defend the properties the reservoir exists for: balanced coverage of
economy phases and decision events, recency inside a stratum, both successful
and failed worlds, teacher scaffolding that retires, exact resume, and no
orphaned snapshot files.
"""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from .state_reservoir import (
    LANE_POLICY,
    LANE_TEACHER,
    OUTCOME_FAILURE,
    OUTCOME_SUCCESS,
    OUTCOME_TEACHER,
    PERIODIC_EVENT,
    LaneMixture,
    ReservoirConfig,
    SnapshotRecord,
    StartStateController,
    StartStateReservoir,
    import_teacher_snapshots,
    phase_for_tick,
    primary_event,
    relative_entropy_balance,
)


def _record(
    reservoir: StartStateReservoir,
    *,
    event: str = "pre_spawn",
    phase: str = "mid",
    outcome: str = OUTCOME_SUCCESS,
    lane: str = LANE_POLICY,
    tick: int = 4000,
    env: int = 0,
) -> SnapshotRecord:
    destination = reservoir.destination(
        lane=lane, event=event, phase=phase, outcome=outcome, tick=tick, env=env,
    )
    destination.write_bytes(b"snapshot")
    return SnapshotRecord(
        path=str(destination),
        lane=lane,
        event=event,
        phase=phase,
        tick=tick,
        step=tick - 1,
        outcome=outcome,
        curriculum="seed_outpost",
        bytes=8,
        creeps=12,
        owned_rooms=1,
        remote_staffed=1,
        skill_rate=6.0,
        update=3,
        env=env,
        seed=3,
        events=(event,),
        created=0.0,
    )


class EventStratificationTest(unittest.TestCase):
    def test_pre_decision_and_rare_tags_win_the_stratum(self) -> None:
        self.assertEqual(primary_event(["periodic", "pre_spawn"]), "pre_spawn")
        self.assertEqual(primary_event(["pre_spawn", "pre_claim"]), "pre_claim")
        self.assertEqual(primary_event(["post_spawn", "remote_at_source"]), "remote_at_source")
        self.assertEqual(primary_event([]), PERIODIC_EVENT)
        self.assertEqual(primary_event(["not_a_tag"]), PERIODIC_EVENT)

    def test_phases_partition_the_declared_horizon(self) -> None:
        self.assertEqual(phase_for_tick(0), "early")
        self.assertEqual(phase_for_tick(1_999), "early")
        self.assertEqual(phase_for_tick(2_000), "mid")
        self.assertEqual(phase_for_tick(7_999), "mid")
        self.assertEqual(phase_for_tick(8_000), "late")
        self.assertEqual(phase_for_tick(14_999), "late")
        self.assertEqual(phase_for_tick(15_000), "endgame")
        self.assertEqual(phase_for_tick(20_000), "endgame")


class ReservoirPopulationTest(unittest.TestCase):
    def test_full_stratum_evicts_the_oldest_and_deletes_its_file(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            reservoir = StartStateReservoir(root, config=ReservoirConfig(per_stratum=2))
            first = _record(reservoir, tick=1)
            second = _record(reservoir, tick=2)
            self.assertIsNone(reservoir.admit(first))
            self.assertIsNone(reservoir.admit(second))
            third = _record(reservoir, tick=3)
            evicted = reservoir.admit(third)
            self.assertIsNotNone(evicted)
            self.assertEqual(evicted.tick, 1)
            self.assertFalse(Path(first.path).exists())
            self.assertTrue(Path(third.path).exists())
            self.assertEqual(reservoir.size, 2)

    def test_sampling_is_uniform_over_strata_not_over_records(self) -> None:
        """A stratum with one record must not be buried by a crowded one."""
        with tempfile.TemporaryDirectory() as root:
            reservoir = StartStateReservoir(
                root, config=ReservoirConfig(per_stratum=32), rng=random.Random(11),
            )
            for index in range(32):
                reservoir.admit(_record(reservoir, event="pre_spawn", tick=index))
            reservoir.admit(_record(reservoir, event="pre_claim", phase="late", tick=99))
            draws = [reservoir.sample(LANE_POLICY).event for _ in range(400)]
            rare = draws.count("pre_claim") / len(draws)
            self.assertGreater(rare, 0.35)
            self.assertLess(rare, 0.65)

    def test_success_and_failure_are_separate_strata(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            reservoir = StartStateReservoir(root, rng=random.Random(5))
            for index in range(6):
                reservoir.admit(_record(reservoir, outcome=OUTCOME_SUCCESS, tick=index))
            reservoir.admit(_record(reservoir, outcome=OUTCOME_FAILURE, tick=50))
            outcomes = {reservoir.sample(LANE_POLICY).outcome for _ in range(60)}
            self.assertEqual(outcomes, {OUTCOME_SUCCESS, OUTCOME_FAILURE})

    def test_admission_fills_empty_strata_before_refreshing_full_ones(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            reservoir = StartStateReservoir(
                root,
                config=ReservoirConfig(per_stratum=1, refresh_probability=0.0),
                rng=random.Random(3),
            )
            self.assertTrue(
                reservoir.should_admit(
                    lane=LANE_POLICY, event="rcl_up", phase="late", outcome=OUTCOME_SUCCESS,
                )
            )
            reservoir.admit(_record(reservoir, event="rcl_up", phase="late"))
            self.assertFalse(
                reservoir.should_admit(
                    lane=LANE_POLICY, event="rcl_up", phase="late", outcome=OUTCOME_SUCCESS,
                )
            )
            self.assertTrue(
                reservoir.should_admit(
                    lane=LANE_POLICY, event="rcl_up", phase="endgame", outcome=OUTCOME_SUCCESS,
                )
            )


class TeacherLaneTest(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        snapshot = root / "teacher" / "pre_claim" / "late" / "teacher" / "t009000.xsnp"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"teacher-state")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "format": 1,
            "kind": "teacher_start_states",
            "teacher": "ti",
            "records": [{
                "path": str(snapshot.relative_to(root)),
                "event": "pre_claim",
                "phase": "late",
                "tick": 9000,
                "step": 8999,
                "curriculum": "seed_outpost",
                "bytes": 13,
                "creeps": 18,
                "owned_rooms": 1,
                "remote_staffed": 2,
                "skill_rate": 9.5,
                "env": 0,
                "seed": 201,
                "events": ["pre_claim"],
                "created": 0.0,
            }],
        }), encoding="utf-8")
        return manifest

    def test_teacher_states_import_into_their_own_lane(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            manifest = self._manifest(root)
            reservoir = StartStateReservoir(root / "reservoir")
            self.assertEqual(import_teacher_snapshots(reservoir, manifest), 1)
            # Re-importing the same manifest must not duplicate the state.
            self.assertEqual(import_teacher_snapshots(reservoir, manifest), 0)
            drawn = reservoir.sample(LANE_TEACHER)
            self.assertEqual(drawn.lane, LANE_TEACHER)
            self.assertEqual(drawn.outcome, OUTCOME_TEACHER)
            self.assertEqual(drawn.phase, "late")

    def test_teacher_lane_retires_per_phase_once_the_policy_covers_it(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            manifest = self._manifest(root)
            reservoir = StartStateReservoir(
                root / "reservoir",
                config=ReservoirConfig(
                    per_stratum=32, teacher_retire_records=4, teacher_retire_events=2,
                ),
                rng=random.Random(2),
            )
            import_teacher_snapshots(reservoir, manifest)
            self.assertFalse(reservoir.teacher_retired("late"))
            self.assertEqual(reservoir.sample(LANE_TEACHER).lane, LANE_TEACHER)
            for index in range(3):
                reservoir.admit(_record(reservoir, event="rcl_up", phase="late", tick=index))
            self.assertFalse(reservoir.teacher_retired("late"))
            for index in range(3):
                reservoir.admit(
                    _record(reservoir, event="remote_at_source", phase="late", tick=100 + index)
                )
            self.assertTrue(reservoir.teacher_retired("late"))
            # A retired teacher slot becomes a policy slot, never a tick-zero start.
            self.assertEqual(reservoir.sample(LANE_TEACHER).lane, LANE_POLICY)

    def test_teacher_files_published_under_the_reservoir_root_survive(self) -> None:
        """The hostile layout: the bridge set lives inside the pruned root."""
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            manifest = self._manifest(root / "teacher-set")
            reservoir = StartStateReservoir(root, config=ReservoirConfig(per_stratum=2))
            import_teacher_snapshots(reservoir, manifest)
            teacher_record = reservoir.sample(LANE_TEACHER)
            teacher_file = Path(teacher_record.path)
            self.assertTrue(teacher_file.is_file())
            # Overfill a policy stratum that shares the teacher's event and phase.
            for tick in range(4):
                reservoir.admit(_record(reservoir, event="pre_claim", phase="late", tick=tick))
            reservoir.prune_orphans()
            reservoir.save()
            restored = StartStateReservoir.load(root)
            self.assertTrue(teacher_file.is_file())
            self.assertEqual(restored.lane_size(LANE_TEACHER), 1)
            self.assertEqual(restored.lane_size(LANE_POLICY), 2)


class PersistenceTest(unittest.TestCase):
    def test_index_round_trip_restores_the_population(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            reservoir = StartStateReservoir(root, config=ReservoirConfig(per_stratum=3))
            reservoir.admit(_record(reservoir, event="pre_spawn", phase="early", tick=100))
            reservoir.admit(_record(reservoir, event="rcl_up", phase="late", tick=9000))
            reservoir.save()
            restored = StartStateReservoir.load(root)
            self.assertEqual(restored.size, 2)
            self.assertEqual(restored.config.per_stratum, 3)
            self.assertEqual(
                sorted(record.event for record in restored.records()),
                ["pre_spawn", "rcl_up"],
            )

    def test_missing_files_and_orphans_are_reconciled_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            reservoir = StartStateReservoir(root)
            kept = _record(reservoir, tick=1)
            lost = _record(reservoir, tick=2)
            reservoir.admit(kept)
            reservoir.admit(lost)
            reservoir.save()
            Path(lost.path).unlink()
            orphan = Path(root) / "policy" / "pre_spawn" / "mid" / "success" / "orphan.xsnp"
            orphan.write_bytes(b"stale")
            restored = StartStateReservoir.load(root)
            self.assertEqual(restored.size, 1)
            self.assertEqual(restored.records()[0].tick, 1)
            self.assertFalse(orphan.exists())


class ControllerTest(unittest.TestCase):
    def _controller(self, **config_kwargs) -> StartStateController:
        self._work = tempfile.TemporaryDirectory()
        reservoir = StartStateReservoir(
            self._work.name, config=ReservoirConfig(**config_kwargs), rng=random.Random(7),
        )
        return StartStateController(
            reservoir,
            mixture=LaneMixture(fresh=2, policy=2, teacher=0),
            num_envs=4,
            max_episode=20000,
        )

    def tearDown(self) -> None:
        work = getattr(self, "_work", None)
        if work is not None:
            work.cleanup()

    def test_fresh_lane_is_never_restored_or_truncated(self) -> None:
        controller = self._controller(segment_ticks=8)
        self.assertEqual(controller.lanes, ("fresh", "fresh", "policy", "policy"))
        for index in (0, 1):
            self.assertEqual(controller.envs[index].segment_ticks, 0)
            self.assertIsNone(controller.start_path(index))
        for step in range(20):
            controller.note_step(0, {"step": step, "harvestDelta": 10, "controlDelta": 1})
            self.assertFalse(controller.restart_due(0))

    def test_snapshot_lane_restarts_on_its_segment_boundary(self) -> None:
        controller = self._controller(segment_ticks=4)
        reservoir = controller.reservoir
        reservoir.admit(_record(reservoir, tick=7000, phase="mid"))
        for step in range(4):
            controller.note_step(2, {"step": step, "harvestDelta": 8, "controlDelta": 0})
        self.assertEqual(controller.due_restarts(), [2])
        path = controller.start_path(2)
        self.assertIsNotNone(path)
        self.assertEqual(controller.envs[2].origin_tick, 7000)
        self.assertFalse(controller.restart_due(2))

    def test_empty_reservoir_falls_back_to_a_fresh_start(self) -> None:
        controller = self._controller(segment_ticks=4)
        self.assertIsNone(controller.start_path(3))
        self.assertEqual(controller.envs[3].origin, "fresh")

    def test_capture_respects_the_gap_and_skips_restarted_envs(self) -> None:
        controller = self._controller(min_capture_gap=16, per_stratum=8)
        info = {
            "step": 320, "time": 321, "events": ["pre_spawn"], "creeps": 9,
            "harvestDelta": 12, "controlDelta": 1, "ownedRooms": 1, "remoteRoomsStaffed": 1,
        }
        for index in range(4):
            controller.note_step(index, info)
        requests = controller.capture_requests([info] * 4)
        self.assertEqual(len(requests), 4)
        self.assertEqual({fields["event"] for _i, _p, _t, fields in requests}, {"pre_spawn"})
        # A failed capture must not burn the window, so the gap is charged only by
        # the descriptors that came back from the environment.
        self.assertEqual(len(controller.capture_requests([info] * 4)), 4)
        descriptors = []
        for index, path, _tags, _fields in requests:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"x")
            descriptors.append({
                "env": index, "path": path, "tick": 321, "step": 320,
                "bytes": 1, "rooms": ["W7N3"], "curriculum": "seed_outpost",
            })
        controller.commit_captures(
            descriptors,
            {i: fields for i, _p, _t, fields in requests},
            seeds=(3, 4, 5, 6),
        )
        near = {**info, "step": 328, "time": 329}
        self.assertEqual(controller.capture_requests([near] * 4), [])
        # An env that just restarted holds a different world than the tick reported.
        boundary = {**info, "step": 400, "time": 401, "segment_boundary": True}
        self.assertEqual(controller.capture_requests([boundary] * 4), [])

    def test_only_full_lifecycle_envs_book_episodes(self) -> None:
        """A restored world reaching the horizon is a fragment, not an episode."""
        controller = self._controller(segment_ticks=64)
        self.assertTrue(controller.books_full_lifecycle(0))
        self.assertTrue(controller.books_full_lifecycle(1))
        self.assertFalse(controller.books_full_lifecycle(2))
        self.assertFalse(controller.books_full_lifecycle(3))

    def test_outcome_labels_split_productive_and_collapsed_worlds(self) -> None:
        controller = self._controller(success_skill_rate=4.0)
        productive = {"step": 100, "creeps": 12, "harvestDelta": 20, "controlDelta": 2}
        for _ in range(4000):
            controller.note_step(0, productive)
        self.assertEqual(controller.outcome_for(0, productive), OUTCOME_SUCCESS)
        collapsed = {"step": 100, "creeps": 1, "harvestDelta": 0, "controlDelta": 0}
        self.assertEqual(controller.outcome_for(0, collapsed), OUTCOME_FAILURE)
        for _ in range(4000):
            controller.note_step(1, {"step": 100, "creeps": 6, "harvestDelta": 0, "controlDelta": 0})
        self.assertEqual(
            controller.outcome_for(1, {"step": 100, "creeps": 6}), OUTCOME_FAILURE,
        )

    def test_productivity_label_does_not_measure_segment_age(self) -> None:
        """A young but productive segment must not be labelled a failure."""
        controller = self._controller(success_skill_rate=4.0)
        productive = {"step": 40, "creeps": 10, "harvestDelta": 8, "controlDelta": 1}
        for _ in range(40):
            controller.note_step(0, productive)
        self.assertAlmostEqual(controller.envs[0].skill_rate, 9.0, places=6)
        self.assertLess(controller.envs[0].skill_ema, 4.0)
        self.assertEqual(controller.outcome_for(0, productive), OUTCOME_SUCCESS)

    def test_captures_start_one_gap_into_a_new_world(self) -> None:
        controller = self._controller(min_capture_gap=64, per_stratum=8)
        early = {
            "step": 3, "time": 4, "events": ["post_spawn"], "creeps": 2,
            "harvestDelta": 2, "controlDelta": 0,
        }
        self.assertEqual(controller.capture_requests([early] * 4), [])
        later = {**early, "step": 64, "time": 65}
        self.assertEqual(len(controller.capture_requests([later] * 4)), 4)

    def test_restored_segment_keeps_absolute_capture_and_skill_state(self) -> None:
        controller = self._controller(min_capture_gap=64, segment_ticks=64)
        reservoir = controller.reservoir
        reservoir.admit(_record(reservoir, tick=9001, phase="late"))
        self.assertIsNotNone(controller.start_path(2))
        state = controller.envs[2]
        self.assertEqual(state.last_capture_step, 9000)
        self.assertAlmostEqual(state.skill_rate, 6.0, places=6)
        # A tick right after the restore is inside the gap of the snapshot itself.
        soon = {
            "step": 9010, "time": 9011, "events": ["pre_spawn"], "creeps": 12,
            "harvestDelta": 6, "controlDelta": 1,
        }
        self.assertEqual(controller.capture_requests([soon] * 4)[0][0], 0)
        self.assertNotIn(2, [index for index, *_ in controller.capture_requests([soon] * 4)])

    def test_commit_captures_records_env_provenance(self) -> None:
        controller = self._controller(min_capture_gap=1, per_stratum=4)
        info = {
            "step": 9000, "time": 9001, "events": ["rcl_up"], "creeps": 20,
            "harvestDelta": 30, "controlDelta": 3, "ownedRooms": 2, "remoteRoomsStaffed": 2,
        }
        requests = controller.capture_requests([info, info, info, info])
        self.assertTrue(requests)
        index, path, _tags, _fields = requests[0]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"x")
        descriptors = [{
            "env": index, "path": path, "tick": 9001, "step": 9000,
            "bytes": 1, "rooms": ["W7N3"], "curriculum": "seed_outpost",
        }]
        admitted = controller.commit_captures(
            descriptors,
            {i: fields for i, _p, _t, fields in requests},
            seeds=(3, 4, 5, 6),
        )
        self.assertEqual(admitted, 1)
        record = controller.reservoir.records()[0]
        self.assertEqual(record.event, "rcl_up")
        self.assertEqual(record.phase, "late")
        self.assertEqual(record.seed, 3 + index)
        self.assertEqual(record.update, controller.update)

    def test_metrics_report_phase_coverage_of_training_transitions(self) -> None:
        controller = self._controller()
        for step in (10, 20, 3_000, 9_000, 9_100, 18_000):
            controller.note_step(0, {"step": step, "harvestDelta": 1, "controlDelta": 0})
        metrics = controller.metrics()
        self.assertAlmostEqual(metrics["train_phase_fraction_early"], 2 / 6)
        self.assertAlmostEqual(metrics["train_phase_fraction_late"], 2 / 6)
        self.assertAlmostEqual(metrics["train_phase_fraction_endgame"], 1 / 6)
        self.assertEqual(metrics["start_mix_fresh"], 2.0)

    def test_resume_rejects_a_changed_mixture_and_keeps_live_segments(self) -> None:
        controller = self._controller(segment_ticks=64)
        controller.update = 12
        controller.note_step(2, {"step": 5, "harvestDelta": 1, "controlDelta": 0})
        payload = controller.state_dict()
        fresh = StartStateController(
            controller.reservoir,
            mixture=LaneMixture(fresh=2, policy=2, teacher=0),
            num_envs=4,
            max_episode=20000,
        )
        fresh.note_step(2, {"step": 0, "harvestDelta": 0, "controlDelta": 0})
        fresh.load_state_dict(payload)
        self.assertEqual(fresh.update, 12)
        self.assertEqual(fresh.envs[2].ticks_in_segment, 1)
        other = StartStateController(
            controller.reservoir,
            mixture=LaneMixture(fresh=3, policy=1, teacher=0),
            num_envs=4,
            max_episode=20000,
        )
        with self.assertRaises(ValueError):
            other.load_state_dict(payload)


class MixtureTest(unittest.TestCase):
    def test_parse_requires_a_full_and_non_degenerate_fleet(self) -> None:
        mixture = LaneMixture.parse("fresh=12,policy=8,teacher=4", 24)
        self.assertEqual((mixture.fresh, mixture.policy, mixture.teacher), (12, 8, 4))
        with self.assertRaises(ValueError):
            LaneMixture.parse("fresh=12,policy=8,teacher=4", 16)
        with self.assertRaises(ValueError):
            LaneMixture.parse("fresh=0,policy=24,teacher=0", 24)
        with self.assertRaises(ValueError):
            LaneMixture.parse("fresh=12,bogus=12", 24)

    def test_assignment_interleaves_snapshot_lanes(self) -> None:
        lanes = LaneMixture(fresh=12, policy=8, teacher=4).assign(24)
        self.assertEqual(len(lanes), 24)
        self.assertEqual(lanes[:12], ("fresh",) * 12)
        self.assertEqual(lanes.count("policy"), 8)
        self.assertEqual(lanes.count("teacher"), 4)
        # Teacher slots must not be contiguous, so they cover several curricula.
        tail = lanes[12:]
        self.assertNotEqual(tail[:4], ("teacher",) * 4)

    def test_entropy_balance_reports_coverage(self) -> None:
        from collections import Counter

        support = ["early", "mid", "late", "endgame"]
        self.assertAlmostEqual(
            relative_entropy_balance(Counter({name: 5 for name in support}), support), 1.0,
        )
        self.assertEqual(relative_entropy_balance(Counter({"early": 5}), support), 0.0)
        self.assertEqual(relative_entropy_balance(Counter(), support), 0.0)


if __name__ == "__main__":
    unittest.main()
