#!/usr/bin/env python3
"""End-to-end connected-room claim contract using the real engine.

This is an environment/action validation scenario, not learned-policy evidence.
It seeds one CLAIM+MOVE creep, reserves the remote controller, verifies that its
own reservation remains renewable/claimable but not attackable, and converts it
to ownership. A second pass requires the scripted teacher to label and execute
the claim macro itself.
"""
from __future__ import annotations

import argparse

import torch

from .constants import INTENT_SLOTS, INTENT_TYPES, MAX_ACTORS, SCHEMA
from .env_client import ScreepsEnv


def validate_scripted_claim(args: argparse.Namespace, claim: int) -> int:
    """The seeded claimer must receive labels through the real teacher path."""
    env = ScreepsEnv(
        node=args.node,
        room=args.room,
        max_episode=args.ticks + 10,
        curriculum="seed_claimer",
        lean_meta=False,
    )
    saw_claim_label = False
    try:
        env.reset()
        for tick in range(1, args.ticks + 1):
            _, reward, done, info, actions = env.step_scripted()
            saw_claim_label = saw_claim_label or bool(
                (torch.as_tensor(actions["types"]) == claim).any().item()
            )
            if int(info.get("ownedRooms") or 0) >= 2:
                if not saw_claim_label:
                    raise RuntimeError("teacher claimed without exposing a claim label")
                if int(info.get("claimDelta") or 0) != 1:
                    raise RuntimeError(f"teacher claim missing claimDelta=1: {info}")
                if reward < float(SCHEMA["reward"]["roomClaim"]):
                    raise RuntimeError(f"teacher claim reward={reward} below contract")
                return tick
            if done:
                break
    finally:
        env.close()
    raise RuntimeError(
        f"scripted seed_claimer did not claim within {args.ticks} ticks; "
        f"saw_claim_label={saw_claim_label}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate cross-room claim execution")
    parser.add_argument("--room", default="W7N3")
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--node", default=None)
    args = parser.parse_args()

    env = ScreepsEnv(
        node=args.node,
        room=args.room,
        max_episode=args.ticks + 10,
        curriculum="seed_claimer",
        lean_meta=False,
    )
    claim = INTENT_TYPES.index("claimController")
    reserve = INTENT_TYPES.index("reserveController")
    attack_controller = INTENT_TYPES.index("attackController")
    saw_own_reservation = False
    try:
        obs = env.reset()
        for tick in range(1, args.ticks + 1):
            actor_meta = obs.get("_actor_meta") or []
            target_meta = obs.get("_target_meta") or []
            actor_idx = next(
                (i for i, meta in enumerate(actor_meta) if meta.get("id") == "seed_claimer"),
                None,
            )
            target_idx = next(
                (
                    i for i, meta in enumerate(target_meta)
                    if meta.get("room") != args.room
                    and meta.get("structureType") == "controller"
                ),
                None,
            )
            if actor_idx is None or target_idx is None:
                raise RuntimeError(
                    f"missing expansion actor/target at tick={tick}: "
                    f"actors={actor_meta} targets={target_meta[:8]}"
                )

            own_reservation = bool(target_meta[target_idx].get("myReservation"))
            if own_reservation:
                saw_own_reservation = True
                if int(obs["target_select_mask"][0, claim, target_idx]) != 1:
                    raise RuntimeError("own reservation does not remain claimable")
                if int(obs["target_select_mask"][0, reserve, target_idx]) != 1:
                    raise RuntimeError("own reservation cannot be renewed")
                if int(obs["target_select_mask"][0, attack_controller, target_idx]) != 0:
                    raise RuntimeError("own reservation is incorrectly attackable")

            shape = (1, MAX_ACTORS, INTENT_SLOTS)
            actions = {
                key: torch.zeros(shape, dtype=torch.long)
                for key in ("types", "dirs", "targets", "amounts")
            }
            actions["types"][0, actor_idx, 0] = claim if own_reservation else reserve
            actions["targets"][0, actor_idx, 0] = target_idx
            obs, reward, done, info = env.step(actions)
            if int(info.get("ownedRooms") or 0) >= 2:
                if not saw_own_reservation:
                    raise RuntimeError("claim bypassed the reserve-to-claim contract")
                if int(info.get("claimDelta") or 0) != 1:
                    raise RuntimeError(f"claim did not emit claimDelta=1: {info}")
                if reward < float(SCHEMA["reward"]["roomClaim"]):
                    raise RuntimeError(f"claim reward={reward} below contract")
                teacher_tick = validate_scripted_claim(args, claim)
                print(
                    f"[eval_expansion] PASS executor_claim_tick={tick} "
                    f"teacher_claim_tick={teacher_tick}",
                    flush=True,
                )
                return 0
            if tick % 100 == 0:
                pos = (obs.get("_actor_meta") or [])[actor_idx]
                result = next(
                    (
                        item for item in info.get("intentResults", [])
                        if item.get("actor") == "seed_claimer"
                    ),
                    None,
                )
                print(
                    f"[eval_expansion] t={tick} room={pos.get('room')} "
                    f"xy=({pos.get('x')},{pos.get('y')}) result={result}",
                    flush=True,
                )
            if done:
                break
        print(f"[eval_expansion] FAIL no claim after {args.ticks} ticks", flush=True)
        return 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
