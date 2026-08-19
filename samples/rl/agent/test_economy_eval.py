"""Focused unit and real-engine tests for economy telemetry/evaluation."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from .env_client import ScreepsEnv
from .eval_economy import EconomyAggregate, economy_stage


def _economy(
    *, rcl: int = 1, owned: int = 1,
    remote_productive: int = 0, remote_staffed: int = 0,
) -> dict:
    return {
        "homeRoom": "W7N3",
        "rooms": {
            "W7N3": {
                "isHome": True,
                "owned": True,
                "controller": {"level": rcl},
                "creeps": {"active": 2},
                "sources": {
                    "harvestedThisTick": 10,
                    "sustainableRate": 20,
                    "remaining": 2500,
                },
            }
        },
        "totals": {
            "ownedRooms": owned,
            "remoteCreeps": 0,
            "remoteRoomsStaffed": 0,
            "remoteProductiveCreeps": remote_productive,
            "remoteEconomyRoomsStaffed": remote_staffed,
            "droppedEnergy": 4,
            "creeps": {
                "active": 2, "productive": 2, "spawning": 1,
                "byRole": {"miner": 1},
            },
            "sinks": {
                "energy": 100, "capacity": 300, "starvation": 200,
                "spawnExtensionFillFraction": 1 / 3,
                "towerFillFraction": 1,
            },
            "construction": {
                "sites": 1,
                "remaining": 90,
                "createdThisTick": 1,
                "completedThisTick": 0,
            },
            "controller": {"rclMax": rcl},
        },
        "spawn": {
            "intentMetricsAvailable": True,
            "attempts": 2,
            "accepted": 1,
            "failures": {"busy": 1},
            "startedThisTick": 1,
            "completedThisTick": 0,
            "cancelledThisTick": 0,
        },
        "overflow": {
            "rooms": {"overflow": 0},
            "actors": {"overflow": 1},
            "targets": {"overflow": 0},
        },
    }


class EconomyAggregationTest(unittest.TestCase):
    def test_stages_are_non_overlapping_and_expansion_wins(self) -> None:
        self.assertEqual(economy_stage(_economy(rcl=1)), "bootstrap_rcl1")
        self.assertEqual(economy_stage(_economy(rcl=2)), "buildout_rcl2")
        self.assertEqual(economy_stage(_economy(rcl=4)), "mature_rcl3plus")
        self.assertEqual(
            economy_stage(_economy(rcl=2, remote_productive=1)),
            "remote_outpost",
        )
        self.assertEqual(
            economy_stage(_economy(rcl=3, remote_staffed=1)),
            "remote_outpost",
        )
        self.assertEqual(economy_stage(_economy(rcl=1, owned=2)), "expansion")

    def test_aggregate_separates_event_totals_from_state_means(self) -> None:
        aggregate = EconomyAggregate()
        info = {
            "harvestDelta": 10,
            "remoteHarvestDelta": 4,
            "remoteHomeDeliveryDelta": 3,
            "controlDelta": 3,
            "transferDelta": 5,
            "buildDelta": 2,
            "claimDelta": 0,
            "rclUp": 0,
        }
        aggregate.update(1, info, _economy())
        aggregate.update(2, info, _economy())
        result = aggregate.result()
        self.assertEqual(result["events"]["harvested"], 20)
        self.assertEqual(result["events"]["remote_harvested"], 8)
        self.assertEqual(result["events"]["remote_home_delivered"], 6)
        self.assertEqual(result["events"]["spawn_attempts"], 4)
        self.assertEqual(result["state_means"]["source_utilization"], 0.5)
        self.assertEqual(result["state_means"]["sink_starvation"], 200)
        self.assertEqual(result["role_live_tick_means"], {"miner": 1.0})
        self.assertEqual(result["rooms"]["W7N3"]["state_means"]["source_utilization"], 0.5)
        self.assertEqual(
            result["rooms"]["W7N3"]["role_live_tick_means"], {}
        )
        self.assertEqual(result["spawn_failures"], {"busy": 2.0})
        self.assertEqual(result["overflow_ticks"]["actors"], 2)

    def test_unavailable_expert_intents_are_not_counted_as_measured_zeros(self) -> None:
        aggregate = EconomyAggregate()
        economy = _economy()
        economy["spawn"] = {
            "intentMetricsAvailable": False,
            "attempts": None,
            "accepted": None,
            "failures": None,
            "startedThisTick": 1,
            "completedThisTick": 0,
            "cancelledThisTick": 0,
        }
        aggregate.update(1, {}, economy)
        result = aggregate.result()
        self.assertEqual(result["spawn_intent_metric_ticks"], 0)
        self.assertEqual(result["events"]["spawn_started"], 1)
        self.assertEqual(result["spawn_failures"], {})


class EconomyTelemetryEngineTest(unittest.TestCase):
    def test_seed_outpost_harvests_and_hauls_without_claiming(self) -> None:
        with patch.dict(
            os.environ,
            {"RL_OBS_FMT": "bin", "RL_ECONOMY_TELEMETRY": "1"},
        ):
            env = ScreepsEnv(
                max_episode=130,
                lean_meta=True,
                curriculum="seed_outpost",
            )
            try:
                env.reset()
                remote_harvest = 0
                remote_delivery = 0
                claims = 0
                last_info = {}
                for _ in range(110):
                    _, _, _, last_info, _ = env.step_scripted()
                    remote_harvest += int(last_info["remoteHarvestDelta"])
                    remote_delivery += int(last_info["remoteHomeDeliveryDelta"])
                    claims += int(last_info["claimDelta"])
                    if remote_delivery > 0:
                        break
                self.assertGreater(remote_harvest, 0)
                self.assertGreater(remote_delivery, 0)
                self.assertEqual(claims, 0)
                self.assertEqual(last_info["neutralOutpostRooms"], 1)
                self.assertGreaterEqual(last_info["remoteRoomsStaffedPeak"], 1)
                self.assertGreaterEqual(last_info["remoteProductiveCreepsPeak"], 1)
                self.assertEqual(last_info["remoteOwnedRoomsPeak"], 0)
            finally:
                env.close()

    @unittest.expectedFailure  # ROADMAP blocker 5: the baseline's aging finisher
    def test_outpost_route_finisher_and_replacement_remain_productive(self) -> None:
        """Route cargo survives replacement and the successor works after TTL 1500.

        The hand-written planner is now the evaluation baseline, not a teacher, and
        it fails this at seed 7: the seeded outpost worker survives to tick 1,497
        but its successor never delivers home again. The assertions describe the
        behavior a baseline must have, so they stay executable and expected-failing
        rather than deleted; fixing the planner turns this into an unexpected
        success, which fails the suite and forces the marker off.
        """
        with patch.dict(
            os.environ,
            {"RL_OBS_FMT": "bin", "RL_ECONOMY_TELEMETRY": "1"},
        ):
            env = ScreepsEnv(
                max_episode=1810,
                lean_meta=False,
                curriculum="seed_outpost",
                seed=7,
            )
            try:
                env.reset()
                invalid = 0
                claims = 0
                finisher_delivered = False
                replacement_spawned = False
                late_harvest = 0
                late_delivery = 0
                late_staffed_ticks = 0
                late_productive_ticks = 0
                remote_owned_peak = 0
                for timestep in range(1800):
                    _, _, _, info, _ = env.step_scripted()
                    invalid += int(info["intentInvalid"])
                    claims += int(info["claimDelta"])
                    remote_owned_peak = max(
                        remote_owned_peak, int(info["remoteOwnedRoomsPeak"]),
                    )
                    if timestep >= 1200:
                        replacement_spawned |= int(info["spawnSuccess"]) > 0
                        if int(info["remoteHomeDeliveryDelta"]) > 0:
                            finisher_delivered |= any(
                                result.get("actor") == "seed_worker"
                                and result.get("type") == "transfer"
                                and int(result.get("code", -1)) == 0
                                and bool(result.get("executed"))
                                for result in info["intentResults"]
                            )
                    if timestep >= 1600:
                        late_harvest += int(info["remoteHarvestDelta"])
                        late_delivery += int(info["remoteHomeDeliveryDelta"])
                        late_staffed_ticks += int(info["remoteRoomsStaffed"] > 0)
                        late_productive_ticks += int(
                            info["remoteProductiveCreeps"] > 0
                        )
                self.assertEqual(invalid, 0)
                self.assertTrue(finisher_delivered)
                self.assertTrue(replacement_spawned)
                self.assertGreater(late_harvest, 0)
                self.assertGreater(late_delivery, 0)
                self.assertGreater(late_staffed_ticks, 0)
                self.assertGreater(late_productive_ticks, 0)
                self.assertEqual(claims, 0)
                self.assertEqual(remote_owned_peak, 0)
            finally:
                env.close()

    def test_scripted_step_returns_consistent_post_tick_economy(self) -> None:
        with patch.dict(
            os.environ,
            {"RL_OBS_FMT": "bin", "RL_ECONOMY_TELEMETRY": "1"},
        ):
            env = ScreepsEnv(max_episode=8, lean_meta=True)
            try:
                env.reset()
                saw_spawn_attempt = False
                saw_spawn_start = False
                for _ in range(4):
                    _, _, _, info, _ = env.step_scripted()
                    economy = info["economy"]
                    self.assertEqual(economy["version"], 1)
                    self.assertIn("W7N3", economy["rooms"])
                    self.assertEqual(economy["homeRoom"], "W7N3")
                    self.assertEqual(
                        set(economy["overflow"]), {"rooms", "actors", "targets"}
                    )
                    totals = economy["totals"]
                    room_total = sum(
                        room["creeps"]["total"] for room in economy["rooms"].values()
                    )
                    self.assertEqual(totals["creeps"]["total"], room_total)
                    self.assertEqual(
                        totals["sources"]["harvestedThisTick"], info["harvestDelta"]
                    )
                    home_sources = economy["rooms"]["W7N3"]["sources"]["items"]
                    self.assertEqual(len(home_sources), 2)
                    self.assertTrue(
                        all(source["capacity"] == 3000 for source in home_sources)
                    )
                    for room in economy["rooms"].values():
                        for source in room["sources"]["items"]:
                            self.assertGreaterEqual(source["remaining"], 0)
                            self.assertLessEqual(source["remaining"], source["capacity"])
                        self.assertGreaterEqual(room["sinks"]["starvation"], 0)
                        self.assertGreaterEqual(room["construction"]["remaining"], 0)
                    spawn = economy["spawn"]
                    self.assertTrue(spawn["intentMetricsAvailable"])
                    saw_spawn_attempt |= spawn["attempts"] > 0
                    saw_spawn_start |= spawn["startedThisTick"] > 0
                    if spawn["accepted"]:
                        self.assertEqual(spawn["acceptedWithoutObservedStart"], 0)
                self.assertTrue(saw_spawn_attempt)
                self.assertTrue(saw_spawn_start)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
