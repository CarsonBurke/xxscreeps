#!/usr/bin/env python3
"""Measure scripted RCL1 baseline competence (CPU, 1 env — light).

  RL_NODE="$(mise exec node@24 -- which node)" \\
  python3 -m samples.rl.agent.eval_scripted --ticks 2000

Does not use GPU. Keep ticks modest unless authorized for longer runs.
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
except ImportError:
    from agent.env_client import ScreepsEnv


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=500)
    p.add_argument("--room", type=str, default="W7N3")
    p.add_argument("--node", type=str, default=None)
    p.add_argument("--max-episode", type=int, default=2000)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument(
        "--curriculum",
        type=str,
        default="empty",
        help="empty|seed_creep|seed_full|seed_claimer",
    )
    args = p.parse_args()

    env = ScreepsEnv(
        node=args.node,
        room=args.room,
        max_episode=args.max_episode,
        expert=False,
        curriculum=args.curriculum,
        lean_meta=False,
        seed=args.seed,
    )
    try:
        env.reset()
        H = C = S = R = 0.0
        delivered = built = rcl_ups = claims = 0.0
        max_owned_rooms = 0
        max_energy = max_sites = 0
        creeps_peak = 0
        first_creep = None
        invalid_results = 0
        issued_results = 0
        invalid_by_type: dict[str, int] = {}
        invalid_by_type_code: dict[str, int] = {}
        advanced_deposits = 0.0
        advanced_withdrawals = 0.0
        tower_refills = 0.0
        # Per-tick H+C for windowed skill (full mean hides ramp / late collapse)
        step_skill: list[float] = []
        for t in range(args.ticks):
            _, r, done, info, _ = env.step_scripted()
            for result in info.get("intentResults") or ():
                issued_results += 1
                if int(result.get("code", -1)) not in (0, -2, -4, -11):
                    invalid_results += 1
                    intent = str(result.get("type") or "unknown")
                    invalid_by_type[intent] = invalid_by_type.get(intent, 0) + 1
                    key = f"{intent}:{int(result.get('code', -1))}"
                    invalid_by_type_code[key] = invalid_by_type_code.get(key, 0) + 1
            h = float(info.get("harvestDelta") or 0)
            c = float(info.get("controlDelta") or 0)
            H += h
            C += c
            step_skill.append(h + c)
            S += float(info.get("spawnSuccess") or 0)
            delivered += float(info.get("transferDelta") or 0)
            built += float(info.get("buildDelta") or 0)
            rcl_ups += float(info.get("rclUp") or 0)
            claims += float(info.get("claimDelta") or 0)
            advanced_deposits += float(info.get("advancedDepositDelta") or 0)
            advanced_withdrawals += float(info.get("advancedWithdrawDelta") or 0)
            tower_refills += float(info.get("towerRefillDelta") or 0)
            max_owned_rooms = max(max_owned_rooms, int(info.get("ownedRooms") or 0))
            max_energy = max(max_energy, int(info.get("energyAvailable") or 0))
            max_sites = max(max_sites, int(info.get("sites") or 0))
            R += float(r)
            creeps = int(info.get("creeps") or 0)
            creeps_peak = max(creeps_peak, creeps)
            if first_creep is None and creeps > 0:
                first_creep = t + 1
            if done:
                env.reset()
            if (t + 1) % 100 == 0:
                print(
                    f"[eval_scripted] t={t+1} skill={(H+C)/(t+1):.3f} "
                    f"H={H:.0f} C={C:.0f} spawns={S:.0f} ret={R:.2f} creeps={creeps}",
                    flush=True,
                )
        T = max(1, args.ticks)
        n = len(step_skill)
        skill_first200 = sum(step_skill[:200]) / max(1, min(200, n))
        skill_last500 = sum(step_skill[-500:]) / max(1, min(500, n))
        print(
            f"[eval_scripted] DONE seed={args.seed} ticks={args.ticks} "
            f"skill_rate={(H+C)/T:.4f} "
            f"skill_first200={skill_first200:.4f} skill_last500={skill_last500:.4f} "
            f"harvest={H:.1f} control={C:.1f} spawnSuccess={S:.0f} "
            f"delivered={delivered:.1f} built={built:.1f} rclUps={rcl_ups:.0f} "
            f"claims={claims:.0f} ownedRoomsPeak={max_owned_rooms} "
            f"maxEnergy={max_energy} maxSites={max_sites} return={R:.2f} "
            f"first_creep_tick={first_creep} creeps_peak={creeps_peak} "
            f"advancedDeposits={advanced_deposits:.0f} advancedWithdrawals={advanced_withdrawals:.0f} "
            f"towerRefills={tower_refills:.0f} invalid={invalid_results}/{issued_results}",
            flush=True,
        )
        if invalid_by_type:
            print(f"[eval_scripted] invalid_by_type={invalid_by_type}", flush=True)
            print(f"[eval_scripted] invalid_by_type_code={invalid_by_type_code}", flush=True)
        # Gates (docs/05 + JOB0-REPORT)
        ok = True
        if first_creep is None or first_creep > 200:
            print("[gate] G1 FAIL first creep", flush=True)
            ok = False
        else:
            print("[gate] G1 PASS", flush=True)
        if H <= 0:
            print("[gate] G2 FAIL harvest", flush=True)
            ok = False
        else:
            print("[gate] G2 PASS", flush=True)
        # G3: prove the logistics loop directly; the initial spawn is pre-funded.
        if delivered <= 0 or S < 2:
            print(f"[gate] G3 FAIL delivered={delivered:.0f} spawnSuccess={S:.0f}", flush=True)
            ok = False
        else:
            print("[gate] G3 PASS", flush=True)
        if C <= 0:
            print("[gate] G4 FAIL control (may need longer ticks)", flush=True)
            ok = False
        else:
            print("[gate] G4 PASS", flush=True)
        if args.ticks >= 2000:
            if rcl_ups <= 0 or built <= 0 or creeps_peak < 5:
                print(
                    f"[gate] G5 FAIL rclUps={rcl_ups:.0f} built={built:.0f} "
                    f"creepsPeak={creeps_peak}",
                    flush=True,
                )
                ok = False
            else:
                print("[gate] G5 PASS RCL2 construction + population growth", flush=True)
        if invalid_results:
            print(f"[gate] G6 FAIL invalid teacher results={invalid_results}", flush=True)
            ok = False
        else:
            print("[gate] G6 PASS zero invalid teacher results", flush=True)
        if args.ticks >= 14000:
            if claims < 1 or max_owned_rooms < 2:
                print(
                    f"[gate] G7 FAIL claims={claims:.0f} ownedRoomsPeak={max_owned_rooms}",
                    flush=True,
                )
                ok = False
            else:
                print("[gate] G7 PASS economy-produced cross-room claim", flush=True)
        if args.ticks >= 20000:
            if advanced_deposits < 1 or advanced_withdrawals < 1 or tower_refills < 1:
                print(
                    "[gate] G8 FAIL advanced logistics "
                    f"deposits={advanced_deposits} withdrawals={advanced_withdrawals} "
                    f"towerRefills={tower_refills}",
                    flush=True,
                )
                ok = False
            else:
                print("[gate] G8 PASS storage/container operation + tower fueling", flush=True)
        return 0 if ok else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
