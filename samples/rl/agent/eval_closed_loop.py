#!/usr/bin/env python3
"""Score a learned actor on fresh, untouched worlds.

This is the only admissible comparison between two training runs. Start-state
reservoirs change which states PPO trains on, so a run can look better on its
own restored starts while being no better at building an economy from nothing.
Evaluation therefore always resets full-length episodes at tick zero, on a seed
family that neither training nor teacher collection used, and reports the
per-curriculum late-window remote metrics that decay first.

    python3 -m samples.rl.agent.eval_closed_loop \\
      --checkpoint samples/rl/runs/policy_reservoirA.pt \\
      --ticks 20000 --num-envs 10 --seed 900 \\
      --output samples/rl/runs/eval_A.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_RL_ROOT = Path(__file__).resolve().parents[1]
_REPO = _RL_ROOT.parents[1]
for _path in (_REPO, _RL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from samples.rl.agent.artifacts import load_full_state, validate_artifact
from samples.rl.agent.model import Agent
from samples.rl.agent.pretrain_joint import _validate_closed_loop
from samples.rl.agent.vec_env import VecScreepsEnv, configure_host_threads

DEFAULT_CURRICULA = "empty,seed_creep,seed_full,seed_claimer,seed_outpost"
# Training uses seed 3 and teacher collection used 201/301; evaluation must not
# reuse either, or a "held-out" world is one the policy already started from.
DEFAULT_EVAL_SEED = 900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--curriculum", type=str, default=DEFAULT_CURRICULA)
    parser.add_argument("--room", type=str, default="W7N3")
    parser.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--node", type=str, default=None)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="sample actions instead of greedy decoding; required to observe "
             "behaviour that survives only as low-probability policy mass",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    # Intra-op parallelism makes environment workers contend; see
    # vec_env.configure_host_threads.
    configure_host_threads()
    args = parse_args()
    if args.ticks < 1:
        raise SystemExit("[eval] --ticks must be positive")
    if args.num_envs < 1:
        raise SystemExit("[eval] --num-envs must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "actor" not in checkpoint:
        raise SystemExit(f"[eval] {args.checkpoint} is not a current-schema artifact")
    agent = Agent()
    agent.to(device)
    try:
        meta = validate_artifact(
            checkpoint, agent.actor, agent.critic,
            kinds=("joint_pretrain", "ppo"),
            evaluation_only_source_mismatch=True,
        )
    except ValueError as error:
        raise SystemExit(f"[eval] incompatible checkpoint: {error}") from error
    load_full_state(agent.actor, checkpoint["actor"], name="actor")

    envs = VecScreepsEnv(
        args.num_envs,
        node=args.node,
        room=args.room,
        max_episode=args.ticks,
        device=device,
        curriculum=args.curriculum,
        lean_meta=True,
        seed=args.seed,
    )
    print(
        f"[eval] {args.checkpoint.name} kind={meta.get('kind')} "
        f"update={checkpoint.get('update')} global_step={checkpoint.get('global_step')} "
        f"envs={args.num_envs} ticks={args.ticks} seed={args.seed}",
        flush=True,
    )
    try:
        result = _validate_closed_loop(
            agent.actor, envs, steps=args.ticks, device=device,
            deterministic=not args.sample,
        )
    finally:
        envs.close()

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "kind": meta.get("kind"),
        "update": checkpoint.get("update"),
        "global_step": checkpoint.get("global_step"),
        "eval_seed": args.seed,
        "eval_ticks": args.ticks,
        "eval_num_envs": args.num_envs,
        "decode": "sampled" if args.sample else "greedy",
        "curriculum": args.curriculum,
        "initialization_source_sha256": meta.get("initialization_source_sha256"),
        **result,
    }
    for key in sorted(summary):
        if isinstance(summary[key], (int, float, str, type(None))):
            print(f"[eval] {key}={summary[key]}", flush=True)
    stages = summary.get("closed_loop_by_curriculum") or {}
    for stage in sorted(stages):
        row = stages[stage]
        print(
            f"[eval] {stage}: skill_rate={row.get('skill_rate', 0):.3f} "
            f"control_rate={row.get('control_rate', 0):.3f} "
            f"remote_harvest={row.get('remote_harvest', 0):.0f} "
            f"remote_home_delivery={row.get('remote_home_delivery', 0):.0f} "
            f"late_remote_harvest={row.get('late_remote_harvest', 0):.0f} "
            f"late_remote_staffed_ticks={row.get('late_remote_staffed_ticks', 0):.0f} "
            f"claims={row.get('claims', 0):.0f}",
            flush=True,
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"[eval] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
