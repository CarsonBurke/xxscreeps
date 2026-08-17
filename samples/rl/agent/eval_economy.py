#!/usr/bin/env python3
"""Run a real-engine economy evaluation and emit JSON Lines diagnostics.

The final record reports RCL1, RCL2, RCL3+, neutral-outpost, and owned-expansion
periods separately so long mature periods cannot hide lifecycle failures.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_RL_ROOT = Path(__file__).resolve().parents[1]
_REPO = _RL_ROOT.parents[1]
for _path in (_REPO, _RL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from samples.rl.agent.env_client import ScreepsEnv
except ImportError:
    from agent.env_client import ScreepsEnv


STAGE_ORDER = (
    "bootstrap_rcl1", "buildout_rcl2", "mature_rcl3plus",
    "remote_outpost", "expansion",
)
EVENT_KEYS = (
    "harvested",
    "remote_harvested",
    "remote_home_delivered",
    "controller_progress",
    "delivered",
    "advanced_deposits",
    "advanced_withdrawals",
    "tower_refills",
    "build_progress",
    "construction_created",
    "construction_completed",
    "spawn_attempts",
    "spawn_accepted",
    "spawn_started",
    "spawn_completed",
    "spawn_cancelled",
    "claims",
    "rcl_ups",
)
STATE_KEYS = (
    "source_utilization",
    "source_remaining",
    "dropped_energy",
    "sink_energy",
    "sink_capacity",
    "sink_starvation",
    "spawn_extension_fill_fraction",
    "tower_fill_fraction",
    "creeps_active",
    "creeps_productive",
    "creeps_spawning",
    "remote_creeps",
    "remote_productive_creeps",
    "remote_rooms_staffed",
    "remote_economy_rooms_staffed",
    "remote_owned_rooms",
    "neutral_outpost_rooms",
    "owned_rooms",
    "construction_sites",
    "construction_remaining",
    "rcl_max",
)


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def economy_stage(economy: dict[str, Any]) -> str:
    """Classify a post-tick economy state into a non-overlapping lifecycle stage."""
    totals = economy.get("totals") or {}
    if int(totals.get("ownedRooms") or 0) >= 2:
        return "expansion"
    if (
        int(totals.get("remoteEconomyRoomsStaffed") or 0) > 0
        or int(totals.get("remoteProductiveCreeps") or 0) > 0
    ):
        return "remote_outpost"
    home_name = economy.get("homeRoom")
    rooms = economy.get("rooms") or {}
    home = rooms.get(home_name) or next(
        (room for room in rooms.values() if room.get("isHome")), {}
    )
    rcl = int((home.get("controller") or {}).get("level") or 0)
    if rcl <= 1:
        return "bootstrap_rcl1"
    if rcl == 2:
        return "buildout_rcl2"
    return "mature_rcl3plus"


def _managed_source_state(economy: dict[str, Any]) -> tuple[float, float, float]:
    """Return harvested, sustainable rate, and remaining for economically active rooms."""
    harvested = sustainable = remaining = 0.0
    for room in (economy.get("rooms") or {}).values():
        creeps = room.get("creeps") or {}
        if not (room.get("isHome") or room.get("owned") or _number(creeps.get("active")) > 0):
            continue
        sources = room.get("sources") or {}
        harvested += _number(sources.get("harvestedThisTick"))
        sustainable += _number(sources.get("sustainableRate"))
        remaining += _number(sources.get("remaining"))
    return harvested, sustainable, remaining


class EconomyAggregate:
    """Accumulate event totals and state-time means without mixing their semantics."""

    def __init__(self) -> None:
        self.ticks = 0
        self.first_tick: int | None = None
        self.last_tick: int | None = None
        self.events: Counter[str] = Counter()
        self.states: Counter[str] = Counter()
        self.peaks: Counter[str] = Counter()
        self.role_ticks: Counter[str] = Counter()
        self.role_peaks: Counter[str] = Counter()
        self.room_ticks: Counter[str] = Counter()
        self.room_states: dict[str, Counter[str]] = {}
        self.room_role_ticks: dict[str, Counter[str]] = {}
        self.room_role_peaks: dict[str, Counter[str]] = {}
        self.spawn_failures: Counter[str] = Counter()
        self.spawn_intent_metric_ticks = 0
        self.overflow_ticks: Counter[str] = Counter()

    def update(self, tick: int, info: dict[str, Any], economy: dict[str, Any]) -> None:
        totals = economy.get("totals") or {}
        spawn = economy.get("spawn") or {}
        creeps = totals.get("creeps") or {}
        sinks = totals.get("sinks") or {}
        construction = totals.get("construction") or {}
        controller = totals.get("controller") or {}
        harvested, sustainable, source_remaining = _managed_source_state(economy)

        event_values = {
            "harvested": _number(info.get("harvestDelta")),
            "remote_harvested": _number(info.get("remoteHarvestDelta")),
            "remote_home_delivered": _number(info.get("remoteHomeDeliveryDelta")),
            "controller_progress": _number(info.get("controlDelta")),
            "delivered": _number(info.get("transferDelta")),
            "advanced_deposits": _number(info.get("advancedDepositDelta")),
            "advanced_withdrawals": _number(info.get("advancedWithdrawDelta")),
            "tower_refills": _number(info.get("towerRefillDelta")),
            "build_progress": _number(info.get("buildDelta")),
            "construction_created": _number(construction.get("createdThisTick")),
            "construction_completed": _number(construction.get("completedThisTick")),
            "spawn_attempts": _number(spawn.get("attempts")),
            "spawn_accepted": _number(spawn.get("accepted")),
            "spawn_started": _number(spawn.get("startedThisTick")),
            "spawn_completed": _number(spawn.get("completedThisTick")),
            "spawn_cancelled": _number(spawn.get("cancelledThisTick")),
            "claims": _number(info.get("claimDelta")),
            "rcl_ups": _number(info.get("rclUp")),
        }
        state_values = {
            "source_utilization": harvested / sustainable if sustainable > 0 else 0.0,
            "source_remaining": source_remaining,
            "dropped_energy": _number(totals.get("droppedEnergy")),
            "sink_energy": _number(sinks.get("energy")),
            "sink_capacity": _number(sinks.get("capacity")),
            "sink_starvation": _number(sinks.get("starvation")),
            "spawn_extension_fill_fraction": _number(
                sinks.get("spawnExtensionFillFraction")
            ),
            "tower_fill_fraction": _number(sinks.get("towerFillFraction")),
            "creeps_active": _number(creeps.get("active")),
            "creeps_productive": _number(creeps.get("productive")),
            "creeps_spawning": _number(creeps.get("spawning")),
            "remote_creeps": _number(totals.get("remoteCreeps")),
            "remote_productive_creeps": _number(totals.get("remoteProductiveCreeps")),
            "remote_rooms_staffed": _number(totals.get("remoteRoomsStaffed")),
            "remote_economy_rooms_staffed": _number(
                totals.get("remoteEconomyRoomsStaffed")
            ),
            "remote_owned_rooms": _number(
                info.get("remoteOwnedRooms", totals.get("remoteOwnedRooms"))
            ),
            "neutral_outpost_rooms": _number(
                info.get("neutralOutpostRooms", totals.get("neutralOutpostRooms"))
            ),
            "owned_rooms": _number(totals.get("ownedRooms")),
            "construction_sites": _number(construction.get("sites")),
            "construction_remaining": _number(construction.get("remaining")),
            "rcl_max": _number(controller.get("rclMax")),
        }
        self.ticks += 1
        self.first_tick = tick if self.first_tick is None else min(self.first_tick, tick)
        self.last_tick = tick if self.last_tick is None else max(self.last_tick, tick)
        self.events.update(event_values)
        self.states.update(state_values)
        for key, value in state_values.items():
            self.peaks[key] = max(self.peaks[key], value)
        for role, count in (creeps.get("byRole") or {}).items():
            count = _number(count)
            self.role_ticks[str(role)] += count
            self.role_peaks[str(role)] = max(self.role_peaks[str(role)], count)
        for room_name, room in (economy.get("rooms") or {}).items():
            room_name = str(room_name)
            room_creeps = room.get("creeps") or {}
            room_sources = room.get("sources") or {}
            room_sinks = room.get("sinks") or {}
            room_construction = room.get("construction") or {}
            room_controller = room.get("controller") or {}
            room_sustainable = _number(room_sources.get("sustainableRate"))
            room_values = {
                "source_utilization": (
                    _number(room_sources.get("harvestedThisTick")) / room_sustainable
                    if room_sustainable > 0 else 0.0
                ),
                "source_remaining": _number(room_sources.get("remaining")),
                "dropped_energy": _number(room.get("droppedEnergy")),
                "sink_starvation": _number(room_sinks.get("starvation")),
                "creeps_active": _number(room_creeps.get("active")),
                "creeps_spawning": _number(room_creeps.get("spawning")),
                "construction_sites": _number(room_construction.get("sites")),
                "rcl": _number(room_controller.get("level")),
                "owned": 1.0 if room.get("owned") else 0.0,
            }
            self.room_ticks[room_name] += 1
            self.room_states.setdefault(room_name, Counter()).update(room_values)
            role_ticks = self.room_role_ticks.setdefault(room_name, Counter())
            role_peaks = self.room_role_peaks.setdefault(room_name, Counter())
            for role, count in (room_creeps.get("byRole") or {}).items():
                count = _number(count)
                role_ticks[str(role)] += count
                role_peaks[str(role)] = max(role_peaks[str(role)], count)
        if spawn.get("intentMetricsAvailable"):
            self.spawn_intent_metric_ticks += 1
            self.spawn_failures.update(
                {
                    str(key): _number(value)
                    for key, value in (spawn.get("failures") or {}).items()
                }
            )
        for category, row in (economy.get("overflow") or {}).items():
            if _number((row or {}).get("overflow")) > 0:
                self.overflow_ticks[str(category)] += 1

    def result(self) -> dict[str, Any]:
        ticks = max(1, self.ticks)
        rooms = {}
        for room_name in sorted(self.room_ticks):
            room_ticks = max(1, self.room_ticks[room_name])
            rooms[room_name] = {
                "ticks_visible": self.room_ticks[room_name],
                "state_means": {
                    key: value / room_ticks
                    for key, value in sorted(self.room_states[room_name].items())
                },
                "role_live_tick_means": {
                    key: value / room_ticks
                    for key, value in sorted(self.room_role_ticks[room_name].items())
                },
                "role_live_peaks": dict(sorted(self.room_role_peaks[room_name].items())),
            }
        return {
            "ticks": self.ticks,
            "first_tick": self.first_tick,
            "last_tick": self.last_tick,
            "events": {key: self.events[key] for key in EVENT_KEYS},
            "event_rates": {f"{key}_per_tick": self.events[key] / ticks for key in EVENT_KEYS},
            "state_means": {key: self.states[key] / ticks for key in STATE_KEYS},
            "state_peaks": {key: self.peaks[key] for key in STATE_KEYS},
            "role_live_tick_means": {
                key: value / ticks for key, value in sorted(self.role_ticks.items())
            },
            "role_live_peaks": dict(sorted(self.role_peaks.items())),
            "rooms": rooms,
            "spawn_failures": dict(sorted(self.spawn_failures.items())),
            "spawn_intent_metric_ticks": self.spawn_intent_metric_ticks,
            "overflow_ticks": {
                category: self.overflow_ticks[category]
                for category in ("rooms", "actors", "targets")
            },
        }


def _emit(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    old_telemetry = os.environ.get("RL_ECONOMY_TELEMETRY")
    os.environ["RL_ECONOMY_TELEMETRY"] = "1"
    try:
        env = ScreepsEnv(
            node=args.node,
            room=args.room,
            max_episode=args.ticks + 5,
            expert=args.mode == "expert",
            bot_dir=args.bot_dir,
            curriculum=args.curriculum,
            lean_meta=True,
            seed=args.seed,
        )
    finally:
        if old_telemetry is None:
            os.environ.pop("RL_ECONOMY_TELEMETRY", None)
        else:
            os.environ["RL_ECONOMY_TELEMETRY"] = old_telemetry

    overall = EconomyAggregate()
    stage_aggregates = {name: EconomyAggregate() for name in STAGE_ORDER}
    window = EconomyAggregate()
    transitions: list[dict[str, Any]] = []
    previous_stage: str | None = None
    previous_rcl: int | None = None
    previous_owned: int | None = None
    final_economy: dict[str, Any] | None = None
    started_at = time.monotonic()
    completed_ticks = 0
    try:
        env.reset()
        for tick in range(1, args.ticks + 1):
            if args.mode == "scripted":
                _, _, done, info, _ = env.step_scripted()
            else:
                _, _, done, info = env.step()
            economy = info.get("economy")
            if not isinstance(economy, dict):
                raise RuntimeError(
                    "server did not return economy telemetry; ensure server.mjs supports "
                    "RL_ECONOMY_TELEMETRY=1"
                )
            stage = economy_stage(economy)
            overall.update(tick, info, economy)
            stage_aggregates[stage].update(tick, info, economy)
            window.update(tick, info, economy)
            completed_ticks = tick
            final_economy = economy

            totals = economy.get("totals") or {}
            rcl = int(((totals.get("controller") or {}).get("rclMax")) or 0)
            owned = int(totals.get("ownedRooms") or 0)
            if stage != previous_stage or rcl != previous_rcl or owned != previous_owned:
                transitions.append(
                    {"tick": tick, "stage": stage, "rcl_max": rcl, "owned_rooms": owned}
                )
                previous_stage, previous_rcl, previous_owned = stage, rcl, owned

            if args.interval > 0 and (tick % args.interval == 0 or tick == args.ticks):
                if not args.quiet:
                    _emit(
                        {
                            "type": "economy_window",
                            "version": 1,
                            "mode": args.mode,
                            "curriculum": args.curriculum,
                            "stage": stage,
                            **window.result(),
                        }
                    )
                window = EconomyAggregate()
            if done and tick < args.ticks:
                raise RuntimeError(f"environment ended early at tick {tick}")
    finally:
        env.close()

    elapsed = time.monotonic() - started_at
    stages = {name: aggregate.result() for name, aggregate in stage_aggregates.items()}
    summary = {
        "type": "economy_summary",
        "version": 1,
        "mode": args.mode,
        "curriculum": args.curriculum,
        "seed": env.seed,
        "command_format": env.command_format,
        "room": args.room,
        "ticks_requested": args.ticks,
        "ticks_completed": completed_ticks,
        "elapsed_seconds": elapsed,
        "ticks_per_second": completed_ticks / elapsed if elapsed > 0 else 0,
        "stage_order": list(STAGE_ORDER),
        "stages_observed": [name for name in STAGE_ORDER if stages[name]["ticks"] > 0],
        "stage_coverage": {
            name: stages[name]["ticks"] > 0 for name in STAGE_ORDER
        },
        "transitions": transitions,
        "overall": overall.result(),
        "stages": stages,
        "final_economy": final_economy,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=6000)
    parser.add_argument("--interval", type=int, default=250, help="JSONL window size; 0 disables")
    parser.add_argument("--mode", choices=("scripted", "expert"), default="scripted")
    parser.add_argument("--curriculum", default="empty")
    parser.add_argument("--room", default="W7N3")
    parser.add_argument("--node", default=None)
    parser.add_argument("--bot-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="write final summary JSON")
    parser.add_argument("--quiet", action="store_true", help="emit only the final summary")
    args = parser.parse_args()
    if args.ticks < 1:
        parser.error("--ticks must be positive")
    if args.interval < 0:
        parser.error("--interval cannot be negative")

    summary = evaluate(args)
    _emit(summary)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
