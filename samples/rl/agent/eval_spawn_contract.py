#!/usr/bin/env python3
"""Engine-backed contract for the teacher's own free-form spawn bodies.

The International is the only behaviour-cloning teacher, and it ignores a
world's archetype tag: it sizes a bootstrap body to the budget it can see. This
contract therefore holds no table of expected bodies. For every spawn world it
measures what TI actually does -- the tick it decides on, the body it decides,
and the budget it decided against -- snapshots that exact pre-decision world,
and replays the measured body through the learner action ABI so the engine
itself certifies the decision. The asserted quantity is "the action ABI
reproduces the teacher's body in the teacher's own state", plus the supervision
breadth these worlds owe the spawn factors.

The expert wire carries no intent summary (`stepExpert` omits
`summarizeIntentResults`), so engine legality of the teacher's decision is read
from `spawnSuccess` while the exact per-part engine outcome comes from the
learner replay's `intentResults`.
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from .constants import (
    BODY_PART_COSTS,
    BODY_PART_TYPES,
    INTENT_TYPES,
    MAX_ACTORS,
    MAX_BODY_PARTS,
    MAX_ROOM_ENERGY,
    N_BODY_PART,
    SCHEMA,
)
from .actions_util import pad_actions
from .env_client import ScreepsEnv
from .pretrain_joint import (
    SPAWN_BUDGET_BUCKETS,
    SPAWN_CURRICULA,
    SPAWN_LENGTH_BUCKETS,
    SPAWN_LENGTH_REQUIRED_BUCKETS,
    SPAWN_LENGTH_UNREACHED_BUCKETS,
    _parts_to_count_order,
    _spawn_budget_bucket,
    _spawn_length_bucket,
)
from .ti_intents import translate_ti_intents


MAX_DECISION_TICKS = 64
_BUDGET_FEATURE = SCHEMA["actorFeatures"].index("roomEnergyAvailable")


@dataclass(frozen=True)
class TeacherSpawn:
    """One measured teacher body decision and the state it was decided in."""

    scenario: str
    tick: int
    actor_index: int
    raw_parts: tuple[int, ...]
    grouped_parts: tuple[int, ...]
    order_exact: bool
    counts: torch.Tensor
    order: torch.Tensor
    budget: int
    cost: int
    obs: dict[str, torch.Tensor]
    context: dict[str, int]
    snapshot: str

    def describe(self) -> str:
        return " ".join(
            f"{BODY_PART_TYPES[part]}*{int(self.counts[part])}"
            for part in self.order.tolist()
            if int(self.counts[part])
        )


def _policy_observation_signature(obs: dict) -> str:
    """Digest policy inputs while treating the target table as an entity set."""
    digest = hashlib.sha256()
    target_rows: list[bytes] = []
    for target_index in obs["target_mask"][0].bool().nonzero().flatten().tolist():
        target_rows.append(b"".join((
            obs["targets"][0, target_index].contiguous().numpy().tobytes(),
            obs["target_select_mask"][..., target_index].contiguous().numpy().tobytes(),
        )))
    for key in sorted(obs):
        if key.startswith("_") or key in {"targets", "target_mask", "target_select_mask"}:
            continue
        value = obs[key]
        if hasattr(value, "contiguous"):
            digest.update(key.encode())
            digest.update(value.contiguous().numpy().tobytes())
    for row in sorted(target_rows):
        digest.update(row)
    return digest.hexdigest()


def _spawn_labels(info: dict) -> list:
    """Return the exactly representable spawn decisions of one expert tick."""
    return [
        label for label in translate_ti_intents(
            info.get("expertIntents"),
            info.get("expertActorMeta") or [],
            info.get("expertTargetMeta") or [],
            info.get("expertRoomNames") or [],
        )
        if label.rejection is None
        and label.body_parts is not None
        and label.actor_index is not None
    ]


def _assert_decision_state(
    scenario: str, obs: dict, actor_index: int,
) -> dict[str, int]:
    """Check the labelled state is a trainable spawn row, and describe it.

    The world's archetype tag buys nothing here: measured over all six worlds,
    The International removes every seeded context creep and construction site
    within two ticks and only then decides, so at the labelled tick each world
    is one idle spawn holding the energy its name advertises. What must hold for
    the row to teach anything is therefore that the spawn decision is legal and
    the advertised budget is funded -- the same two properties the corpus
    collector refuses a row without. The returned counts are telemetry, so a
    curriculum that starts keeping its context is visible instead of asserted.
    """
    if not bool(obs["intent_mask"][0, actor_index, 0, INTENT_TYPES.index("spawnCreep")]):
        raise AssertionError(
            f"{scenario}: spawn is masked illegal in the state the teacher labelled"
        )
    advertised = int(scenario.rsplit("_", 1)[-1])
    budget = round(
        float(obs["actors"][0, actor_index, _BUDGET_FEATURE]) * MAX_ROOM_ENERGY
    )
    if budget != advertised:
        raise AssertionError(
            f"{scenario}: world funds {budget} energy, its name advertises {advertised}"
        )
    return {
        "creeps": sum(
            1 for meta in obs.get("_actor_meta", ()) if meta.get("kind") == "creep"
        ),
        "sites": sum(
            1 for meta in obs.get("_target_meta", ()) if meta.get("kind") == "site"
        ),
    }


def _measure_teacher_spawn(
    scenario: str, *, node: str | None, seed: int, snapshot: Path,
) -> TeacherSpawn:
    """Run TI in one spawn world and measure the body decision it makes.

    The world is snapshotted before every expert tick, so the snapshot left
    behind when the decision lands is exactly the state the teacher decided in.
    """
    env = ScreepsEnv(
        node=node,
        curriculum=scenario,
        max_episode=MAX_DECISION_TICKS + 5,
        expert=True,
        lean_meta=False,
        capture_expert_intents=True,
        seed=seed,
    )
    try:
        obs = env.reset()
        for tick in range(MAX_DECISION_TICKS):
            env.snapshot(snapshot)
            next_obs, _reward, done, info = env.step()
            labels = _spawn_labels(info)
            if not labels:
                if done:
                    break
                obs = next_obs
                continue
            if len(labels) != 1:
                raise AssertionError(
                    f"{scenario}: {len(labels)} spawn decisions on one tick; a "
                    "contract world owns exactly one spawn"
                )
            label = labels[0]
            actor_index = int(label.actor_index)
            raw_parts = tuple(int(part) for part in label.body_parts)
            counts, order, order_exact = _parts_to_count_order(raw_parts)
            length = int(counts.sum())
            if not (1 <= length <= MAX_BODY_PARTS):
                raise AssertionError(f"{scenario}: invalid body length {length}")
            if not label.full_action:
                raise AssertionError(
                    f"{scenario}: teacher spawn is not an exact action, so the "
                    "intent factor cannot be supervised"
                )
            cost = sum(BODY_PART_COSTS[part] for part in raw_parts)
            budget = round(
                float(obs["actors"][0, actor_index, _BUDGET_FEATURE]) * MAX_ROOM_ENERGY
            )
            if cost > budget:
                raise AssertionError(
                    f"{scenario}: teacher body cost {cost} exceeds budget {budget}"
                )
            if float(info.get("spawnSuccess") or 0) < 1:
                raise AssertionError(
                    f"{scenario}: engine did not execute the teacher's body "
                    f"{raw_parts}"
                )
            context = _assert_decision_state(scenario, obs, actor_index)
            return TeacherSpawn(
                scenario=scenario,
                tick=tick,
                actor_index=actor_index,
                raw_parts=raw_parts,
                grouped_parts=tuple(
                    int(part)
                    for part in order.tolist()
                    for _ in range(int(counts[part]))
                ),
                order_exact=order_exact,
                counts=counts,
                order=order,
                budget=budget,
                cost=cost,
                obs={
                    key: value.clone()
                    for key, value in obs.items()
                    if hasattr(value, "clone")
                } | {
                    key: value for key, value in obs.items() if key.startswith("_")
                },
                context=context,
                snapshot=str(snapshot),
            )
        raise AssertionError(
            f"{scenario}: teacher issued no spawn within {MAX_DECISION_TICKS} ticks"
        )
    finally:
        env.close()


def _teacher_action(measured: TeacherSpawn) -> dict[str, torch.Tensor]:
    """Express the measured teacher body as one learner action batch."""
    actions = {
        key: torch.zeros(MAX_ACTORS, 1, dtype=torch.long)
        for key in (
            "types", "dirs", "targets", "amounts",
            "construction_types", "construction_tiles",
        )
    }
    actions["body_counts"] = torch.zeros(MAX_ACTORS, 1, N_BODY_PART, dtype=torch.long)
    actions["body_order"] = torch.arange(N_BODY_PART).view(1, 1, N_BODY_PART).expand(
        MAX_ACTORS, 1, -1,
    ).clone()
    actions["types"][measured.actor_index, 0] = INTENT_TYPES.index("spawnCreep")
    actions["body_counts"][measured.actor_index, 0] = measured.counts
    actions["body_order"][measured.actor_index, 0] = measured.order
    return pad_actions(actions)


def _replay_teacher_body(learner: ScreepsEnv, measured: TeacherSpawn) -> list[int]:
    """Replay the measured body in the teacher's own state and return it as built.

    The restored world must present the same policy inputs the teacher decided
    against; otherwise the replay would certify a different decision.
    """
    scenario = measured.scenario
    restored = learner.restore(measured.snapshot)
    if _policy_observation_signature(restored) != _policy_observation_signature(
        measured.obs
    ):
        raise AssertionError(
            f"{scenario}: restored world differs from the teacher's decision state"
        )
    next_obs, _reward, _done, info = learner.step(_teacher_action(measured))
    spawn_results = [
        result for result in info.get("intentResults", ())
        if result.get("type") == "spawnCreep"
    ]
    if len(spawn_results) != 1 or int(spawn_results[0].get("code", -1)) != 0:
        raise AssertionError(f"{scenario}: engine rejected the teacher body: {spawn_results}")
    built = [int(part) for part in spawn_results[0].get("spawnBodyParts", ())]
    if built != list(measured.grouped_parts):
        raise AssertionError(
            f"{scenario}: engine built {built}, teacher decided "
            f"{list(measured.grouped_parts)}"
        )
    if int(spawn_results[0].get("spawnBodyLength", 0)) != len(built):
        raise AssertionError(f"{scenario}: engine body length disagrees with its parts")
    # Unlike the expert wire, the learner step does summarize its intents, so
    # this counter is measured rather than structurally absent.
    if int(info.get("intentInvalid") or 0):
        raise AssertionError(f"{scenario}: learner replay produced an invalid intent")
    spawned_names = [
        meta.get("id", "") for meta in next_obs.get("_actor_meta", ())
        if meta.get("kind") == "creep" and meta.get("id", "").startswith("rl_")
    ]
    if len(spawned_names) != 1:
        raise AssertionError(f"{scenario}: unexpected spawned names {spawned_names}")
    role = scenario.split("_", 2)[1]
    if role in spawned_names[0] or spawned_names[0].startswith("rlr"):
        raise AssertionError(f"{scenario}: spawn name persists role {spawned_names[0]!r}")
    return built


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenario", action="append", default=None,
        help="restrict the contract to one spawn world (repeatable)",
    )
    args = parser.parse_args()
    scenarios = tuple(args.scenario or SPAWN_CURRICULA)
    unknown = [stage for stage in scenarios if stage not in SPAWN_CURRICULA]
    if unknown:
        raise AssertionError(f"unknown spawn scenarios {unknown}")

    budget_counts = {bucket: 0 for bucket in SPAWN_BUDGET_BUCKETS}
    length_counts = {bucket: 0 for bucket in SPAWN_LENGTH_BUCKETS}
    measured_by_signature: dict[str, TeacherSpawn] = {}
    with tempfile.TemporaryDirectory(prefix="rl-spawn-contract-") as tmp:
        snapshot = Path(tmp) / "decision.snap"
        for scenario in scenarios:
            measured = _measure_teacher_spawn(
                scenario, node=args.node, seed=args.seed, snapshot=snapshot,
            )
            signature = _policy_observation_signature(measured.obs)
            twin = measured_by_signature.get(signature)
            if twin is not None and twin.grouped_parts != measured.grouped_parts:
                raise AssertionError(
                    "contradictory teacher bodies for policy-equivalent states: "
                    f"{twin.scenario} chose {list(twin.grouped_parts)} and "
                    f"{scenario} chose {list(measured.grouped_parts)}"
                )
            measured_by_signature[signature] = measured
            # A learner session per world, because a second restore into a live
            # session inherits the first world's room visibility: measured on
            # 2026-08-17, restoring `spawn_claimer_650` after any other world
            # drops its expansion room from the observation.
            learner = ScreepsEnv(
                node=args.node, curriculum=scenario,
                max_episode=MAX_DECISION_TICKS + 5, lean_meta=False, seed=args.seed,
            )
            try:
                built = _replay_teacher_body(learner, measured)
            finally:
                learner.close()
            budget_counts[_spawn_budget_bucket(measured.budget)] += 1
            length_counts[_spawn_length_bucket(len(built))] += 1
            print(
                f"PASS {scenario}: teacher tick={measured.tick} "
                f"body={measured.describe()} parts={built} "
                f"cost={measured.cost} budget={measured.budget} "
                f"order_supervised={measured.order_exact} "
                f"decision_state={measured.context}"
                + (f" policy_equivalent_to={twin.scenario}" if twin else ""),
                flush=True,
            )

    if scenarios != tuple(SPAWN_CURRICULA):
        # Coverage is a property of the whole world set, so a `--scenario` subset
        # reports its strata instead of asserting a population it cannot observe.
        print(
            f"PARTIAL {len(scenarios)}/{len(SPAWN_CURRICULA)} worlds: "
            f"budget={budget_counts} length={length_counts}",
            flush=True,
        )
        return 0
    missing_budget = [
        bucket for bucket in SPAWN_BUDGET_BUCKETS if budget_counts[bucket] == 0
    ]
    if missing_budget:
        raise AssertionError(
            f"teacher energy-budget strata missing {missing_budget} counts={budget_counts}"
        )
    missing_length = [
        bucket for bucket in SPAWN_LENGTH_REQUIRED_BUCKETS
        if length_counts[bucket] == 0
    ]
    if missing_length:
        raise AssertionError(
            f"teacher body-length strata missing {missing_length} counts={length_counts}"
        )
    for bucket in SPAWN_LENGTH_UNREACHED_BUCKETS:
        print(
            f"COVERAGE-GAP body_length_{bucket}={length_counts[bucket]}: this "
            "economy cannot fund the bucket (>=50 energy per part), so it is "
            "recorded, not required; a higher-RCL curriculum promotes it back",
            flush=True,
        )
    print("PASS spawn pretraining contract", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
