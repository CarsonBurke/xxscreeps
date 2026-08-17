#!/usr/bin/env python3
"""
Watch a trained policy play xxscreeps (optionally in the Screeps client).

Examples (repo root):

  # Headful Screeps map (browser) + terminal HUD
  RL_NODE="$(mise exec node@24 -- which node)" \\
  python3 -m samples.rl.agent.watch \\
    --checkpoint samples/rl/runs/policy.pt --headful --ticks 2000

  # Qualified joint-pretrain artifact (complete actor + critic contract)
  RL_NODE="$(mise exec node@24 -- which node)" \\
  python3 -m samples.rl.agent.watch \\
    --checkpoint samples/rl/runs/joint_pretrain_v2.pt --deterministic --ticks 500

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
    from samples.rl.agent.artifacts import load_full_state, validate_artifact
    from samples.rl.agent.constants import (
        AMOUNT_BINS, BODY_PART_TYPES, CONSTRUCTION_TYPES, INTENT_TYPES,
        MAX_ACTORS,
    )
    from samples.rl.agent.env_client import ScreepsEnv
    from samples.rl.agent.model import Agent, count_params
except ImportError:
    from agent.artifacts import load_full_state, validate_artifact
    from agent.constants import (
        AMOUNT_BINS, BODY_PART_TYPES, CONSTRUCTION_TYPES, INTENT_TYPES,
        MAX_ACTORS,
    )
    from agent.env_client import ScreepsEnv
    from agent.model import Agent, count_params


DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch RL policy in xxscreeps")
    p.add_argument("--checkpoint", type=Path, default=_RL_ROOT / "runs" / "policy.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--room", type=str, default="W7N3")
    p.add_argument("--curriculum", type=str, default="empty")
    p.add_argument("--ticks", type=int, default=2000, help="episode length / max steps")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--node", type=str, default=None)
    p.add_argument("--deterministic", action="store_true", help="argmax actions (default: sample)")
    p.add_argument("--sample", action="store_true", help="sample actions (default)")
    p.add_argument("--headful", action="store_true", help="open Screeps client (RL_HEADFUL)")
    p.add_argument("--tick-ms", type=int, default=None, help="delay between ticks (default 150 if headful)")
    p.add_argument("--no-open", action="store_true", help="do not auto-open browser")
    p.add_argument("--password", type=str, default="rlwatch")
    p.add_argument(
        "--allow-unqualified-joint", action="store_true",
        help="watch a complete but unqualified joint-pretrain artifact",
    )
    p.add_argument("--print-every", type=int, default=1, help="HUD print interval (ticks)")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument(
        "--allow-source-mismatch", action="store_true",
        help=(
            "evaluation only: load a schema/state-compatible local artifact "
            "whose recorded executable source differs"
        ),
    )
    return p.parse_args()


def load_agent(
    path: Path, device: torch.device, *,
    allow_source_mismatch: bool = False,
    allow_unqualified_joint: bool = False,
) -> Agent:
    agent = Agent().to(device)
    if not path.is_file():
        raise SystemExit(f"[watch] checkpoint does not exist: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "actor" not in ckpt or "critic" not in ckpt:
        raise SystemExit(
            f"[watch] checkpoint {path} is not a complete current-schema artifact"
        )
    try:
        meta = validate_artifact(
            ckpt, agent.actor, agent.critic, kinds=("joint_pretrain", "ppo"),
            evaluation_only_source_mismatch=allow_source_mismatch,
        )
        if meta["kind"] == "joint_pretrain" and bool(meta.get("partial")):
            raise ValueError("joint-pretrain artifact is partial")
        if meta["kind"] == "joint_pretrain" and not bool(meta.get("qualified")):
            if not allow_unqualified_joint:
                raise ValueError(
                    "joint-pretrain artifact is unqualified; pass "
                    "--allow-unqualified-joint to watch it anyway"
                )
            print(
                "[watch] viewing an unqualified joint-pretrain artifact; this is a "
                "viewing override, not a promotion",
                flush=True,
            )
        load_full_state(agent.actor, ckpt["actor"], name="actor")
        load_full_state(agent.critic, ckpt["critic"], name="critic")
    except ValueError as error:
        raise SystemExit(f"[watch] incompatible checkpoint: {error}") from error
    print(
        f"[watch] loaded {path} update={ckpt.get('update')} "
        f"global_step={ckpt.get('global_step')} kind={meta['kind']}"
        f" source_mismatch_override={allow_source_mismatch}",
        flush=True,
    )
    agent.eval()
    return agent


def _to_device(obs: dict, device: torch.device) -> dict:
    try:
        from samples.rl.agent.vec_env import promote_obs_device
    except ImportError:
        from agent.vec_env import promote_obs_device
    return promote_obs_device(obs, device, non_blocking=False)


def describe_actions(out, actor_meta: list, target_meta: list) -> list[str]:
    lines = []
    types = out.types[0].cpu()
    dirs = out.dirs[0].cpu()
    tgts = out.targets[0].cpu()
    amts = out.amounts[0].cpu()
    body_counts = out.body_counts[0].cpu()
    body_order = out.body_order[0].cpu()
    construction_types = out.construction_types[0].cpu()
    construction_tiles = out.construction_tiles[0].cpu()
    # `_actor_meta` lists every encoded actor while the action tensors carry the
    # compacted actor bucket, which counts only maskable actors - a creep that is
    # still spawning appears in the meta and not in the bucket. Bound by the
    # tensor, or a colony crossing a bucket boundary indexes past the end and
    # kills the run.
    meta_rows = len(actor_meta) if actor_meta else types.shape[0]
    n = min(MAX_ACTORS, types.shape[0], meta_rows)
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
            if name == "spawnCreep":
                counts = body_counts[ai, s]
                body = []
                for part in body_order[ai, s].tolist():
                    name_part = (
                        BODY_PART_TYPES[part]
                        if 0 <= part < len(BODY_PART_TYPES) else str(part)
                    )
                    body.extend([name_part] * int(counts[part]))
                slots.append(f"spawnCreep(body={body})")
            elif name == "createConstructionSite":
                ci = int(construction_types[ai, s].item())
                structure = (
                    CONSTRUCTION_TYPES[ci]
                    if 0 <= ci < len(CONSTRUCTION_TYPES) else str(ci)
                )
                tile = int(construction_tiles[ai, s].item())
                slots.append(
                    f"createConstructionSite(type={structure},x={tile % 50},y={tile // 50})"
                )
            else:
                slots.append(f"{name}({dname},tgt={tid},amt={ab})")
        if slots:
            aid = meta.get("id", ai)
            kind = meta.get("kind", "?")
            role = meta.get("role")
            outcome = meta.get("outcome")
            xy = f"{meta.get('x', '?')},{meta.get('y', '?')}"
            state = f" role={role}" if role else ""
            state += f" prev={outcome}" if outcome and outcome != "none" else ""
            lines.append(f"  [{kind} {aid} @{xy}{state}] " + " | ".join(slots))
    return lines


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    deterministic = bool(args.deterministic) and not args.sample

    # Optional env inheritance; ScreepsEnv constructor args are authoritative.
    # Watch needs full meta for HUD + pathfinder nav quality (train uses lean/cheap).
    os.environ["RL_LEAN_META"] = "0"
    os.environ.setdefault("RL_NAV", "pathfinder")
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

    agent = load_agent(
        args.checkpoint, device,
        allow_source_mismatch=args.allow_source_mismatch,
        allow_unqualified_joint=args.allow_unqualified_joint,
    )
    print(
        f"[watch] device={device} actor_params={count_params(agent.actor):,} "
        f"deterministic={deterministic} headful={args.headful}",
        flush=True,
    )

    tick_ms = args.tick_ms
    if tick_ms is None and args.headful:
        tick_ms = 150

    env = ScreepsEnv(
        node=args.node,
        room=args.room,
        max_episode=args.ticks,
        device="cpu",
        headful=args.headful,
        headful_password=args.password,
        tick_ms=tick_ms,
        no_open=args.no_open,
        seed=args.seed,
        curriculum=args.curriculum,
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
        total_harvest = total_control = total_delivery = total_build = 0.0
        total_claims = total_invalid = total_issued = 0.0
        max_creeps = 0
        t0 = time.time()
        step = -1
        for step in range(args.ticks):
            batch = _to_device(obs, device)
            with torch.inference_mode():
                out = agent.act(batch, deterministic=deterministic)
            actions = {
                "types": out.types.cpu(),
                "dirs": out.dirs.cpu(),
                "targets": out.targets.cpu(),
                "amounts": out.amounts.cpu(),
                "body_counts": out.body_counts.cpu(),
                "body_order": out.body_order.cpu(),
                "construction_types": out.construction_types.cpu(),
                "construction_tiles": out.construction_tiles.cpu(),
            }
            obs, reward, done, info = env.step(actions)
            ep_ret += reward
            total_harvest += float(info.get("harvestDelta") or 0)
            total_control += float(info.get("controlDelta") or 0)
            total_delivery += float(info.get("transferDelta") or 0)
            total_build += float(info.get("buildDelta") or 0)
            total_claims += float(info.get("claimDelta") or 0)
            total_invalid += float(info.get("intentInvalid") or 0)
            total_issued += float(info.get("intentIssued") or 0)
            max_creeps = max(max_creeps, int(info.get("creeps") or 0))
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
        print(
            f"[watch] finished curriculum={args.curriculum} seed={args.seed} "
            f"decode={'deterministic' if deterministic else 'sampled'} ticks={step+1} "
            f"ep_ret={ep_ret:.3f} H={total_harvest:.0f} C={total_control:.0f} "
            f"delivery={total_delivery:.0f} build={total_build:.0f} "
            f"claims={total_claims:.0f} max_creeps={max_creeps} "
            f"invalid={total_invalid:.0f}/{total_issued:.0f} wall={dt:.1f}s",
            flush=True,
        )
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
