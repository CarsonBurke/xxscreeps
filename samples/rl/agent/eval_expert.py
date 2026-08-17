#!/usr/bin/env python3
"""Measure The International (TI) expert competence floor (CPU, 1 env — light).

  RL_NODE="$(mise exec node@24 -- which node)" \\
  python3 -m samples.rl.agent.eval_expert --ticks 1000

Requires TI dist at ../The-International-Open-Source/dist (or --bot-dir / RL_EXPERT_BOT).
Does not use GPU. TI has a long cold start — report skill after --skip-warmup ticks
separately so early zeros do not dominate the floor used for critic scale judgment.

Not a BC teacher (TI-BC deferred). Critic/obs skill scale only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_RL_ROOT = Path(__file__).resolve().parents[1]
_REPO = _RL_ROOT.parents[1]
for p in (_REPO, _RL_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    from samples.rl.agent.env_client import ScreepsEnv
    from samples.rl.agent.ti_intents import summarize_ti_labels, translate_ti_intents
except ImportError:
    from agent.env_client import ScreepsEnv
    from agent.ti_intents import summarize_ti_labels, translate_ti_intents


def main() -> int:
    p = argparse.ArgumentParser(description="TI expert skill floor (H+C), CPU-only")
    p.add_argument("--ticks", type=int, default=1000)
    p.add_argument("--room", type=str, default="W7N3")
    p.add_argument("--node", type=str, default=None)
    p.add_argument("--max-episode", type=int, default=20_000)
    p.add_argument("--bot-dir", type=str, default=None)
    p.add_argument(
        "--skip-warmup",
        type=int,
        default=500,
        help="report post-warmup skill excluding first N ticks (matches critic pretrain default)",
    )
    p.add_argument("--curriculum", type=str, default="empty")
    p.add_argument(
        "--capture-intents", action="store_true",
        help="capture exact raw TI intent payloads at the runner boundary",
    )
    p.add_argument(
        "--intent-samples", type=int, default=0,
        help="print the first N non-empty captured payloads as compact JSON",
    )
    args = p.parse_args()

    env = ScreepsEnv(
        node=args.node,
        room=args.room,
        max_episode=args.max_episode,
        expert=True,
        bot_dir=args.bot_dir,
        curriculum=args.curriculum,
        capture_expert_intents=args.capture_intents,
    )
    try:
        env.reset()
        H = C = S = R = 0.0
        H_w = C_w = 0.0
        creeps_peak = 0
        first_creep = None
        step_skill: list[float] = []
        intent_ticks = intent_rooms = intent_objects = 0
        intent_samples = 0
        translated = []
        for t in range(args.ticks):
            _, r, done, info = env.step(None)
            h = float(info.get("harvestDelta") or 0)
            c = float(info.get("controlDelta") or 0)
            H += h
            C += c
            step_skill.append(h + c)
            raw_intents = info.get("expertIntents")
            if isinstance(raw_intents, dict):
                intent_ticks += 1
                # TickResult.intentPayloads is the room-name map itself. Accept
                # a future explicit {room: ...} wrapper without miscounting the
                # current wire shape.
                rooms = raw_intents.get("room") if "room" in raw_intents else raw_intents
                rooms = rooms or {}
                intent_rooms += len(rooms)
                intent_objects += sum(
                    len((payload or {}).get("object") or {})
                    for payload in rooms.values()
                    if isinstance(payload, dict)
                )
                if rooms and intent_samples < args.intent_samples:
                    import json
                    print(
                        "[eval_expert] intent_sample="
                        + json.dumps(raw_intents, separators=(",", ":"), sort_keys=True),
                        flush=True,
                    )
                    intent_samples += 1
                translated.extend(translate_ti_intents(
                    raw_intents,
                    info.get("expertActorMeta") or [],
                    info.get("expertTargetMeta") or [],
                    info.get("expertRoomNames") or [],
                ))
            if t >= args.skip_warmup:
                H_w += h
                C_w += c
            S += float(info.get("spawnSuccess") or 0)
            R += float(r)
            creeps = int(info.get("creeps") or 0)
            creeps_peak = max(creeps_peak, creeps)
            if first_creep is None and creeps > 0:
                first_creep = t + 1
            if done:
                env.reset()
            if (t + 1) % 100 == 0:
                print(
                    f"[eval_expert] t={t+1} skill={(H+C)/(t+1):.3f} "
                    f"H={H:.0f} C={C:.0f} spawns={S:.0f} ret={R:.2f} creeps={creeps}",
                    flush=True,
                )
        T = max(1, args.ticks)
        n = len(step_skill)
        skill_first200 = sum(step_skill[:200]) / max(1, min(200, n))
        skill_last500 = sum(step_skill[-500:]) / max(1, min(500, n))
        post = max(1, args.ticks - args.skip_warmup)
        skill_post_warmup = (H_w + C_w) / post
        print(
            f"[eval_expert] DONE ticks={args.ticks} skill_rate={(H+C)/T:.4f} "
            f"skill_post_warmup({args.skip_warmup}..)={skill_post_warmup:.4f} "
            f"skill_first200={skill_first200:.4f} skill_last500={skill_last500:.4f} "
            f"harvest={H:.1f} control={C:.1f} spawnSuccess={S:.0f} "
            f"return={R:.2f} first_creep_tick={first_creep} creeps_peak={creeps_peak}",
            flush=True,
        )
        if args.capture_intents:
            print(
                f"[eval_expert] raw_intents ticks={intent_ticks}/{args.ticks} "
                f"rooms={intent_rooms} object_rows={intent_objects}",
                flush=True,
            )
            import json
            print(
                "[eval_expert] translated="
                + json.dumps(summarize_ti_labels(translated), sort_keys=True),
                flush=True,
            )
        # Gates: TI cold start is expected — G1 may fail early; post-warmup skill is the signal.
        ok = True
        if first_creep is None:
            print("[gate] G1 FAIL never spawned creep (TI cold start or bot broken)", flush=True)
            ok = False
        elif first_creep > 500:
            print(f"[gate] G1 SLOW first creep @ {first_creep} (TI cold start)", flush=True)
        else:
            print(f"[gate] G1 PASS first_creep={first_creep}", flush=True)
        if H <= 0:
            print("[gate] G2 FAIL harvest (whole run)", flush=True)
            ok = False
        else:
            print("[gate] G2 PASS harvest", flush=True)
        if H_w + C_w <= 0 and args.ticks > args.skip_warmup:
            print("[gate] G2b FAIL post-warmup H+C (critic teacher useless)", flush=True)
            ok = False
        else:
            print(f"[gate] G2b PASS post-warmup skill≈{skill_post_warmup:.3f} e/t", flush=True)
        if C <= 0:
            print("[gate] G4 WARN control=0 (may need longer ticks / RCL path)", flush=True)
        else:
            print("[gate] G4 PASS control", flush=True)
        print(
            f"[compare] scripted dual-source skill≈1.42 e/t @1k (H1108 C307); "
            f"TI full={(H+C)/T:.3f} post_warmup={skill_post_warmup:.3f}",
            flush=True,
        )
        return 0 if ok else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
