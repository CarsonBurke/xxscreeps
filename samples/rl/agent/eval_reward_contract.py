#!/usr/bin/env python3
"""Prove productive-delivery reward cannot be recycled from a rewarded sink.

This drives the real simulator, not a mocked reward function. It spends spawn
energy, delivers the seeded worker's energy once, then verifies both the policy
mask and executor reject withdrawing that energy back from the spawn.
"""
from __future__ import annotations

import argparse

import torch

from .constants import INTENT_SLOTS, INTENT_TYPES, MAX_ACTORS
from .env_client import ScreepsEnv


def blank_actions() -> dict[str, torch.Tensor]:
    shape = (1, MAX_ACTORS, INTENT_SLOTS)
    return {
        key: torch.zeros(shape, dtype=torch.long)
        for key in ("types", "dirs", "targets", "amounts")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate non-cyclic delivery reward")
    parser.add_argument("--room", default="W7N3")
    parser.add_argument("--ticks", type=int, default=80)
    parser.add_argument("--node", default=None)
    args = parser.parse_args()

    spawn_intent = INTENT_TYPES.index("spawnCreep")
    transfer_intent = INTENT_TYPES.index("transfer")
    withdraw_intent = INTENT_TYPES.index("withdraw")
    env = ScreepsEnv(
        node=args.node,
        room=args.room,
        max_episode=args.ticks + 10,
        curriculum="seed_full",
        lean_meta=False,
    )
    try:
        obs = env.reset()
        actor_meta = obs.get("_actor_meta") or []
        spawn_actor = next(
            (i for i, meta in enumerate(actor_meta) if meta.get("kind") == "structure"),
            None,
        )
        if spawn_actor is None:
            raise RuntimeError(f"spawn actor missing: {actor_meta}")
        actions = blank_actions()
        actions["types"][0, spawn_actor, 0] = spawn_intent
        obs, _, _, info = env.step(actions)
        spawn_result = next(
            (r for r in info.get("intentResults", []) if r.get("type") == "spawnCreep"),
            None,
        )
        if not spawn_result or int(spawn_result.get("code", -1)) != 0:
            raise RuntimeError(f"could not spend spawn energy: {spawn_result}")

        delivered = 0.0
        for _ in range(args.ticks):
            actor_meta = obs.get("_actor_meta") or []
            target_meta = obs.get("_target_meta") or []
            worker = next(
                (i for i, meta in enumerate(actor_meta) if meta.get("id") == "seed_worker"),
                None,
            )
            spawn_target = next(
                (
                    i for i, meta in enumerate(target_meta)
                    if meta.get("structureType") == "spawn"
                ),
                None,
            )
            if worker is None or spawn_target is None:
                raise RuntimeError("seed worker or spawn target disappeared")
            actions = blank_actions()
            actions["types"][0, worker, 0] = transfer_intent
            actions["targets"][0, worker, 0] = spawn_target
            obs, _, done, info = env.step(actions)
            delivered += float(info.get("transferDelta") or 0)
            if delivered > 0:
                break
            if done:
                break
        if delivered <= 0:
            print("[eval_reward_contract] FAIL no initial productive delivery", flush=True)
            return 2

        actor_meta = obs.get("_actor_meta") or []
        target_meta = obs.get("_target_meta") or []
        worker = next(i for i, meta in enumerate(actor_meta) if meta.get("id") == "seed_worker")
        spawn_target = next(
            i for i, meta in enumerate(target_meta) if meta.get("structureType") == "spawn"
        )
        if int(obs["target_select_mask"][0, withdraw_intent, spawn_target]) != 0:
            print("[eval_reward_contract] FAIL withdraw mask exposes rewarded spawn", flush=True)
            return 2
        if int(obs["intent_mask"][0, worker, 0, withdraw_intent]) != 0:
            print("[eval_reward_contract] FAIL withdraw intent has no valid source", flush=True)
            return 2

        # Deliberately bypass the policy mask: the executor must enforce the same
        # contract, otherwise stale/corrupt actions could still farm the reward.
        actions = blank_actions()
        actions["types"][0, worker, 0] = withdraw_intent
        actions["targets"][0, worker, 0] = spawn_target
        _, reward, _, info = env.step(actions)
        result = next(
            (r for r in info.get("intentResults", []) if r.get("type") == "withdraw"),
            None,
        )
        if float(info.get("transferDelta") or 0) != 0 or reward != 0 or not result:
            print(
                f"[eval_reward_contract] FAIL reward={reward} info={info} result={result}",
                flush=True,
            )
            return 2
        if int(result.get("code", 0)) == 0:
            print(f"[eval_reward_contract] FAIL executor accepted spawn withdraw: {result}", flush=True)
            return 2

        print(
            f"[eval_reward_contract] PASS delivered={delivered:.0f} "
            f"withdraw_code={result.get('code')} reward_after_reject={reward:.1f}",
            flush=True,
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
