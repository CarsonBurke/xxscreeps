#!/usr/bin/env python3
"""
Watch a trained policy play xxscreeps (optionally in the Screeps client).

Examples (repo root):

  # Headful Screeps map (browser) + terminal HUD
  RL_NODE="$(mise exec node@24 -- which node)" \\
  python3 -m samples.rl.agent.watch \\
    --checkpoint samples/rl/runs/policy.pt --headful --ticks 2000

  # Terminal-only (fast)
  RL_NODE="$(mise exec node@24 -- which node)" \\
  python3 -m samples.rl.agent.watch --checkpoint samples/rl/runs/policy.pt --ticks 500
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

_RL_ROOT = Path(__file__).resolve().parents[1]
_REPO = _RL_ROOT.parents[1]
for p in (_REPO, _RL_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    from samples.rl.agent.constants import AMOUNT_BINS, INTENT_TYPES, MAX_ACTORS
    from samples.rl.agent.env_client import ScreepsEnv
    from samples.rl.agent.model import Agent, count_params
except ImportError:
    from agent.constants import AMOUNT_BINS, INTENT_TYPES, MAX_ACTORS
    from agent.env_client import ScreepsEnv
    from agent.model import Agent, count_params


DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch RL policy in xxscreeps")
    p.add_argument("--checkpoint", type=Path, default=_RL_ROOT / "runs" / "policy.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--room", type=str, default="W7N3")
    p.add_argument("--ticks", type=int, default=2000, help="episode length / max steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--node", type=str, default=None)
    p.add_argument("--deterministic", action="store_true", help="argmax actions (default: sample)")
    p.add_argument("--sample", action="store_true", help="sample actions (default)")
    p.add_argument("--headful", action="store_true", help="open Screeps client (RL_HEADFUL)")
    p.add_argument("--tick-ms", type=int, default=None, help="delay between ticks (default 150 if headful)")
    p.add_argument("--no-open", action="store_true", help="do not auto-open browser")
    p.add_argument("--password", type=str, default="rlwatch")
    p.add_argument("--print-every", type=int, default=1, help="HUD print interval (ticks)")
    p.add_argument("--no-compile", action="store_true")
    return p.parse_args()


def load_agent(path: Path, device: torch.device) -> Agent:
    agent = Agent().to(device)
    if not path.is_file():
        print(f"[watch] no checkpoint at {path}; random init weights", flush=True)
        return agent
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "actor" in ckpt:
        missing_a, unexp_a = agent.actor.load_state_dict(ckpt["actor"], strict=False)
        missing_c, unexp_c = agent.critic.load_state_dict(ckpt["critic"], strict=False)
        print(
            f"[watch] loaded {path} update={ckpt.get('update')} "
            f"global_step={ckpt.get('global_step')} "
            f"actor_missing={len(missing_a)} critic_missing={len(missing_c)}",
            flush=True,
        )
    else:
        agent.load_state_dict(ckpt, strict=False)
        print(f"[watch] loaded raw state_dict from {path}", flush=True)
    agent.eval()
    return agent


def _to_device(obs: dict, device: torch.device) -> dict:
    out = {}
    for k, v in obs.items():
        if k.startswith("_"):
            out[k] = v
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def describe_actions(out, actor_meta: list, target_meta: list) -> list[str]:
    lines = []
    types = out.types[0].cpu()
    dirs = out.dirs[0].cpu()
    tgts = out.targets[0].cpu()
    amts = out.amounts[0].cpu()
    n = min(MAX_ACTORS, len(actor_meta) if actor_meta else types.shape[0])
    for ai in range(n):
        meta = actor_meta[ai] if ai < len(actor_meta) else {"id": f"a{ai}", "kind": "?"}
        slots = []
        for s in range(types.shape[1]):
            ti = int(types[ai, s].item())
            name = INTENT_TYPES[ti] if 0 <= ti < len(INTENT_TYPES) else str(ti)
            if name == "none":
                continue
            di = int(dirs[ai, s].item())
            tg = int(tgts[ai, s].item())
            am = int(amts[ai, s].item())
            dname = DIRS[di] if 0 <= di < len(DIRS) else str(di)
            tmeta = target_meta[tg] if target_meta and tg < len(target_meta) else {"id": tg}
            tid = tmeta.get("id", tg) if isinstance(tmeta, dict) else tg
            ab = AMOUNT_BINS[am] if 0 <= am < len(AMOUNT_BINS) else am
            slots.append(f"{name}({dname},tgt={tid},amt={ab})")
        if slots:
            aid = meta.get("id", ai)
            kind = meta.get("kind", "?")
            xy = f"{meta.get('x', '?')},{meta.get('y', '?')}"
            lines.append(f"  [{kind} {aid} @{xy}] " + " | ".join(slots))
    return lines


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    deterministic = bool(args.deterministic) and not args.sample

    # Propagate headful settings into the Node env server
    if args.headful:
        os.environ["RL_HEADFUL"] = "1"
        os.environ["RL_HEADFUL_PASSWORD"] = args.password
        if args.no_open:
            os.environ["RL_NO_OPEN"] = "1"
        tick_ms = args.tick_ms if args.tick_ms is not None else 150
        os.environ["RL_TICK_MS"] = str(tick_ms)
    elif args.tick_ms is not None:
        os.environ["RL_TICK_MS"] = str(args.tick_ms)

    os.environ["RL_ROOM"] = args.room
    os.environ["RL_MAX_EPISODE"] = str(args.ticks)

    agent = load_agent(args.checkpoint, device)
    print(
        f"[watch] device={device} actor_params={count_params(agent.actor):,} "
        f"deterministic={deterministic} headful={args.headful}",
        flush=True,
    )

    env = ScreepsEnv(
        node=args.node,
        room=args.room,
        max_episode=args.ticks,
        device="cpu",
    )
    try:
        obs = env.reset()
        info = env.last_info or {}
        if info.get("headfulUrl"):
            print(f"[watch] client URL: {info['headfulUrl']}", flush=True)
            print(f"[watch] login Player 1 / {args.password} → room {args.room}", flush=True)
        elif args.headful:
            print("[watch] headful requested; check Node stderr for URL / port errors", flush=True)

        ep_ret = 0.0
        t0 = time.time()
        for step in range(args.ticks):
            batch = _to_device(obs, device)
            with torch.inference_mode():
                out = agent.act(batch, deterministic=deterministic)
            actions = {
                "types": out.types.cpu(),
                "dirs": out.dirs.cpu(),
                "targets": out.targets.cpu(),
                "amounts": out.amounts.cpu(),
            }
            obs, reward, done, info = env.step(actions)
            ep_ret += reward
            g = info.get("globals") or (obs.get("_time") and {}) or {}
            if isinstance(g, dict) and not g and hasattr(env, "last_info"):
                g = (env.last_info or {}).get("globals") or {}
            # globals may be on obs batch
            if not g and "globals" in (env.last_info or {}):
                g = env.last_info["globals"]

            if step % max(1, args.print_every) == 0 or done:
                creeps = g.get("creeps", "?") if isinstance(g, dict) else "?"
                energy = g.get("storedEnergy", "?") if isinstance(g, dict) else "?"
                rcl = g.get("rclMax", "?") if isinstance(g, dict) else "?"
                ctrl = g.get("controlProgress", "?") if isinstance(g, dict) else "?"
                v = float(out.value[0].item()) if out.value.numel() else float("nan")
                print(
                    f"[t={step+1:4d}] r={reward:+.4f} ep_ret={ep_ret:.3f} V={v:.3f} "
                    f"harvest={info.get('harvestDelta', 0)} control={info.get('controlDelta', 0)} "
                    f"creeps={creeps} storeE={energy} rcl={rcl} ctrl={ctrl}",
                    flush=True,
                )
                actor_meta = obs.get("_actor_meta") or []
                target_meta = obs.get("_target_meta") or []
                # batch dim unwrap
                if actor_meta and isinstance(actor_meta, list) and actor_meta and isinstance(actor_meta[0], list):
                    actor_meta = actor_meta[0]
                if target_meta and isinstance(target_meta, list) and target_meta and isinstance(target_meta[0], list):
                    target_meta = target_meta[0]
                for line in describe_actions(out, actor_meta, target_meta)[:12]:
                    print(line, flush=True)

            if done:
                print(f"[watch] episode done at step={step+1} ep_ret={ep_ret:.3f}", flush=True)
                break

        dt = time.time() - t0
        print(f"[watch] finished ticks={step+1} ep_ret={ep_ret:.3f} wall={dt:.1f}s", flush=True)
        if args.headful:
            print("[watch] client still open until process exits — Ctrl+C when done watching", flush=True)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("[watch] closing", flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
