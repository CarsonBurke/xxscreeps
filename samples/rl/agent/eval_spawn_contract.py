#!/usr/bin/env python3
"""Engine-backed contract for exact free-form spawn-body scenarios."""
from __future__ import annotations

import argparse
import hashlib

from .constants import (
    BODY_PART_COSTS,
    BODY_PART_TYPES,
    INTENT_TYPES,
    MAX_BODY_PARTS,
    MAX_ROOM_ENERGY,
    SCHEMA,
)
from .actions_util import pad_actions
from .env_client import ScreepsEnv


SCENARIOS = (
    "spawn_flexible_300",
    "spawn_miner_450",
    "spawn_hauler_3000",
    "spawn_builder_650",
    "spawn_upgrader_550",
    "spawn_claimer_650",
)

PART = {name: index for index, name in enumerate(BODY_PART_TYPES)}
EXPECTED_BODIES = {
    "spawn_flexible_300": [PART["work"], PART["carry"], PART["move"]],
    "spawn_miner_450": [PART["work"]] * 4 + [PART["move"]],
    "spawn_hauler_3000": [PART["carry"], PART["move"]] * 25,
    "spawn_builder_650": [
        PART["work"], PART["carry"], PART["carry"], PART["move"], PART["move"],
    ] * 2,
    "spawn_upgrader_550": [
        PART["work"], PART["work"], PART["carry"], PART["move"],
        PART["work"], PART["work"], PART["carry"],
    ],
    "spawn_claimer_650": [PART["claim"], PART["move"]],
}


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default=None)
    args = parser.parse_args()
    spawn_type = INTENT_TYPES.index("spawnCreep")
    seen_lengths: list[int] = []
    seen_budgets: list[int] = []
    policy_signatures: dict[str, str] = {}

    for scenario in SCENARIOS:
        env = ScreepsEnv(
            node=args.node,
            curriculum=scenario,
            max_episode=200,
            lean_meta=False,
        )
        try:
            obs = env.reset()
            signature = _policy_observation_signature(obs)
            previous = policy_signatures.get(signature)
            if previous is not None and EXPECTED_BODIES[previous] != EXPECTED_BODIES[scenario]:
                raise AssertionError(
                    "contradictory body labels for policy-equivalent states: "
                    f"{previous} and {scenario}"
                )
            policy_signatures[signature] = scenario
            if scenario == "spawn_builder_650":
                sites = [
                    meta for meta in obs.get("_target_meta", ())
                    if meta.get("kind") == "site"
                ]
                if not sites:
                    raise AssertionError(f"{scenario}: observable construction demand missing")
            if scenario == "spawn_claimer_650":
                active = obs["actor_mask"][0].bool()
                creep_rows = obs["actors"][0, active]
                work = SCHEMA["actorFeatures"].index("activeWork")
                carry = SCHEMA["actorFeatures"].index("activeCarry")
                if not bool((creep_rows[:, work] > 0).any()) or not bool(
                    (creep_rows[:, carry] > 0).any()
                ):
                    raise AssertionError(
                        f"{scenario}: observable mining/logistics coverage missing"
                    )
            next_obs, _reward, _done, info, raw_actions = env.step_scripted()
            actions = pad_actions(raw_actions)
            rows = (actions["types"][:, 0] == spawn_type).nonzero().flatten()
            if rows.numel() != 1:
                raise AssertionError(f"{scenario}: expected one spawn label, got {rows.numel()}")
            actor = int(rows[0])
            counts = actions["body_counts"][actor, 0]
            order = actions["body_order"][actor, 0]
            length = int(counts.sum())
            if not (1 <= length <= MAX_BODY_PARTS):
                raise AssertionError(f"{scenario}: invalid body length {length}")
            parts = [
                int(part)
                for part in order.tolist()
                for _ in range(int(counts[part]))
            ]
            expected = EXPECTED_BODIES[scenario]
            expected_grouped = [
                part for part in dict.fromkeys(expected) for _ in range(expected.count(part))
            ]
            if parts != expected_grouped:
                raise AssertionError(
                    f"{scenario}: body {parts} != expected {expected_grouped}"
                )
            cost = sum(BODY_PART_COSTS[part] for part in parts)
            budget_feature = SCHEMA["actorFeatures"].index("roomEnergyAvailable")
            budget = round(float(obs["actors"][0, actor, budget_feature]) * MAX_ROOM_ENERGY)
            if cost > budget:
                raise AssertionError(f"{scenario}: body cost {cost} exceeds budget {budget}")
            spawn_results = [
                result for result in info.get("intentResults", ())
                if result.get("type") == "spawnCreep"
            ]
            if len(spawn_results) != 1 or int(spawn_results[0].get("code", -1)) != 0:
                raise AssertionError(f"{scenario}: engine rejected spawn: {spawn_results}")
            expected_role = scenario.split("_", 2)[1]
            spawned_names = [
                meta.get("id", "") for meta in next_obs.get("_actor_meta", ())
                if meta.get("kind") == "creep" and meta.get("id", "").startswith("rl_")
            ]
            if len(spawned_names) != 1 or not spawned_names[0].startswith("rl_"):
                raise AssertionError(f"{scenario}: unexpected spawned name {spawned_names}")
            if expected_role in spawned_names[0] or spawned_names[0].startswith("rlr"):
                raise AssertionError(f"{scenario}: spawn name persists role {spawned_names[0]!r}")
            seen_lengths.append(length)
            seen_budgets.append(budget)
            print(
                f"PASS {scenario}: body={parts} cost={cost} budget={budget}",
                flush=True,
            )
        finally:
            env.close()

    if not (any(length <= 6 for length in seen_lengths)
            and any(7 <= length <= 15 for length in seen_lengths)
            and any(length >= 16 for length in seen_lengths)):
        raise AssertionError(f"body-length strata missing: {seen_lengths}")
    if not (any(budget <= 300 for budget in seen_budgets)
            and any(301 <= budget <= 549 for budget in seen_budgets)
            and any(550 <= budget <= 649 for budget in seen_budgets)
            and any(budget >= 650 for budget in seen_budgets)):
        raise AssertionError(f"budget strata missing: {seen_budgets}")
    print("PASS spawn pretraining contract", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
