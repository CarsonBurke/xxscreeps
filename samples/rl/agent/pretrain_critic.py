#!/usr/bin/env python3
"""
VAPO-style critic pretraining on The International (fixed expert).

Runs N parallel TI envs for T steps (default 8×20k). Observations are not all
kept in RAM — every `chunk` steps we compute truncated MC returns (λ=1, bootstrap
with the current critic) and fit the critic, then free the chunk.

  RL_NODE="$(mise exec node@24 -- which node)" \\
  python3 -m samples.rl.agent.pretrain_critic \\
    --num-envs 8 --steps 20000 --chunk 512 --epochs-per-chunk 4 --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_RL_ROOT = Path(__file__).resolve().parents[1]
_REPO = _RL_ROOT.parents[1]
for p in (_REPO, _RL_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    from samples.rl.agent.constants import PPO_CFG
    from samples.rl.agent.env_client import ScreepsEnv, stack_batches
    from samples.rl.agent.gae import compute_gae_tn
    from samples.rl.agent.model import Critic, auto_minibatch_size, count_params, maybe_compile
except ImportError:
    from agent.constants import PPO_CFG
    from agent.env_client import ScreepsEnv, stack_batches
    from agent.gae import compute_gae_tn
    from agent.model import Critic, auto_minibatch_size, count_params, maybe_compile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain critic on The International")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=20_000, help="expert steps per env")
    p.add_argument("--chunk", type=int, default=512, help="train every this many steps (RAM)")
    p.add_argument("--epochs-per-chunk", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lr", type=float, default=float(PPO_CFG["lr"]) * 2)
    p.add_argument("--gamma", type=float, default=float(PPO_CFG["gamma"]))
    p.add_argument("--minibatch", type=int, default=None)
    p.add_argument("--vram-total-gb", type=float, default=32.0)
    p.add_argument("--vram-reserved-gb", type=float, default=12.0)
    p.add_argument("--max-episode", type=int, default=20_000)
    p.add_argument("--bot-dir", type=str, default=None)
    p.add_argument("--node", type=str, default=None)
    p.add_argument("--save", type=Path, default=_RL_ROOT / "runs" / "critic_pretrained.pt")
    p.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help="TB log dir (default: runs/tb-critic-pretrain/<timestamp> so runs do not merge)",
    )
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _obs_cpu(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in obs.items() if not k.startswith("_")}


def _step_expert_one(env: ScreepsEnv):
    try:
        o, r, d, info = env.step(None)
        if d:
            o = env.reset()
        return o, r, float(d), info
    except Exception as err:  # noqa: BLE001
        # Recover a dead TI sim by killing Node and reopening
        print(f"[pretrain] env step failed ({err!s:.120}); hard_reset", flush=True)
        try:
            o = env.hard_reset()
        except Exception as err2:  # noqa: BLE001
            print(f"[pretrain] hard_reset failed ({err2!s:.120})", flush=True)
            raise
        return o, 0.0, 1.0, {"harvestDelta": 0, "controlDelta": 0, "recovered": True}


def _train_chunk(
    critic: nn.Module,
    critic_c: nn.Module,
    opt: torch.optim.Optimizer,
    buf_obs: list[dict[str, torch.Tensor]],
    rewards_tn: torch.Tensor,
    dones_tn: torch.Tensor,
    *,
    gamma: float,
    epochs: int,
    mb: int,
    device: torch.device,
    use_bf16: bool,
    writer,
    global_step: int,
) -> tuple[float, float, int]:
    """Fit critic on one chunk with λ=1 returns (bootstrap last value)."""
    T, N = rewards_tn.shape
    keys = list(buf_obs[0].keys())
    obs_tn = {
        k: torch.stack([buf_obs[t][k] for t in range(T)], dim=0)  # T,N,...
        for k in keys
    }
    # bootstrap with current critic on last obs
    with torch.no_grad():
        last = {k: v[-1].to(device) for k, v in obs_tn.items()}  # N,...
        boot = critic_c(last).float().cpu()  # N
        boot = boot * (1.0 - dones_tn[-1])

    values_dummy = torch.zeros(T, N)
    # Use zeros as V for GAE λ=1 construction of returns from rewards + boot
    # Manual MC backward with bootstrap:
    returns = torch.zeros(T, N)
    running = boot.clone()
    for t in reversed(range(T)):
        running = rewards_tn[t] + gamma * running * (1.0 - dones_tn[t])
        returns[t] = running

    obs_flat = {k: v.reshape(T * N, *v.shape[2:]).to(device) for k, v in obs_tn.items()}
    ret_flat = returns.reshape(T * N).to(device)
    B = T * N
    idx = torch.arange(B, device=device)
    losses = []

    critic.train()
    for _ in range(epochs):
        perm = idx[torch.randperm(B, device=device)]
        for start in range(0, B, mb):
            inds = perm[start : start + mb]
            batch = {k: v[inds] for k, v in obs_flat.items()}
            target = ret_flat[inds]
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
            with ctx:
                pred = critic_c(batch)
            loss = 0.5 * (pred.float() - target).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(critic.parameters(), float(PPO_CFG.get("maxGradNorm", 0.5)))
            opt.step()
            losses.append(loss.item())
            global_step += inds.numel()
            if writer is not None:
                writer.add_scalar("pretrain/value_loss", loss.item(), global_step)
                writer.add_scalar(
                    "pretrain/grad_norm",
                    float(gn if not torch.is_tensor(gn) else gn.item()),
                    global_step,
                )

    with torch.no_grad():
        sub = idx[torch.randperm(B, device=device)[: min(4096, B)]]
        pred = critic({k: v[sub] for k, v in obs_flat.items()}).float().cpu().numpy()
        y = ret_flat[sub].cpu().numpy()
        var_y = np.var(y)
        ev = float("nan") if var_y == 0 else 1.0 - float(np.var(y - pred) / var_y)

    return float(np.mean(losses)), ev, global_step


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    # Unique run dir so TensorBoard does not merge with older event files
    # (shared dirs reuse step indices and hide the latest run under old curves).
    if args.logdir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.logdir = _RL_ROOT / "runs" / "tb-critic-pretrain" / stamp
    args.logdir.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(args.logdir))
        writer.add_text("run/logdir", str(args.logdir))
        writer.flush()
    except Exception as err:  # noqa: BLE001
        print(f"[pretrain] TB disabled: {err}", flush=True)

    print(
        f"[pretrain] expert=TI  envs={args.num_envs} steps={args.steps} "
        f"chunk={args.chunk} → {args.num_envs * args.steps} env-steps  device={device}",
        flush=True,
    )
    print(f"[pretrain] tensorboard logdir={args.logdir}", flush=True)

    envs = [
        ScreepsEnv(
            node=args.node,
            max_episode=args.max_episode,
            device="cpu",  # obs stay on CPU until train chunk
            expert=True,
            bot_dir=args.bot_dir,
        )
        for _ in range(args.num_envs)
    ]

    critic = Critic().to(device)
    print(f"[pretrain] critic params={count_params(critic):,}", flush=True)

    # probe minibatch after first reset
    try:
        sample_list = list(ThreadPoolExecutor(max_workers=len(envs)).map(lambda e: e.reset(), envs))
        sample = stack_batches(sample_list)
        if args.minibatch:
            mb = args.minibatch
            print(f"[pretrain] minibatch={mb} (manual)", flush=True)
        else:
            mb = auto_minibatch_size(
                sample,
                critic,
                critic,
                total_vram_gb=args.vram_total_gb,
                reserved_gb=args.vram_reserved_gb,
            )
        mb = max(8, mb)
    except Exception as err:
        for e in envs:
            e.close()
        raise RuntimeError(f"expert reset failed: {err}") from err

    critic_c = maybe_compile(critic, not args.no_compile, "critic")
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, eps=1e-5)
    use_bf16 = (not args.no_bf16) and device.type == "cuda" and torch.cuda.is_bf16_supported()

    obs = sample
    buf_obs: list[dict[str, torch.Tensor]] = []
    rewards_rows: list[torch.Tensor] = []
    dones_rows: list[torch.Tensor] = []
    total_h = total_c = 0.0
    global_step = 0
    chunk_i = 0
    t0 = time.time()
    N = len(envs)

    try:
        with ThreadPoolExecutor(max_workers=N) as pool:
            for t in range(args.steps):
                buf_obs.append(_obs_cpu(obs))
                results = list(pool.map(_step_expert_one, envs))
                next_batches, r_row, d_row = [], [], []
                for o, r, d, info in results:
                    total_h += float(info.get("harvestDelta") or 0)
                    total_c += float(info.get("controlDelta") or 0)
                    next_batches.append(o)
                    r_row.append(r)
                    d_row.append(d)
                obs = stack_batches(next_batches)
                rewards_rows.append(torch.tensor(r_row, dtype=torch.float32))
                dones_rows.append(torch.tensor(d_row, dtype=torch.float32))

                # flush chunk
                if len(buf_obs) >= args.chunk or t == args.steps - 1:
                    rewards_tn = torch.stack(rewards_rows, dim=0)
                    dones_tn = torch.stack(dones_rows, dim=0)
                    loss, ev, global_step = _train_chunk(
                        critic, critic_c, opt, buf_obs, rewards_tn, dones_tn,
                        gamma=args.gamma,
                        epochs=args.epochs_per_chunk,
                        mb=mb,
                        device=device,
                        use_bf16=use_bf16,
                        writer=writer,
                        global_step=global_step,
                    )
                    chunk_i += 1
                    # per-env e/t (not sum across envs)
                    rate = (total_h + total_c) / max(1, (t + 1) * N)
                    sps = (t + 1) / max(1e-6, time.time() - t0)
                    print(
                        f"[pretrain] step {t+1}/{args.steps} chunk={chunk_i} "
                        f"loss={loss:.5f} ev={ev:.3f} skill≈{rate:.2f}e/t "
                        f"sps≈{sps:.2f} ({time.time()-t0:.0f}s)",
                        flush=True,
                    )
                    if writer:
                        writer.add_scalar("pretrain/chunk_loss", loss, chunk_i)
                        writer.add_scalar("pretrain/explained_variance", ev, chunk_i)
                        writer.add_scalar("pretrain/skill_rate_et", rate, chunk_i)
                        writer.add_scalar("charts/SPS", sps * N, chunk_i)
                    buf_obs.clear()
                    rewards_rows.clear()
                    dones_rows.clear()
                    # checkpoint each chunk so long runs are resumable-ish
                    if chunk_i % 5 == 0:
                        torch.save({"critic": critic.state_dict(), "step": t + 1}, args.save)
                        print(f"[pretrain] checkpoint → {args.save}", flush=True)

        ckpt = {
            "critic": critic.state_dict(),
            "meta": {
                "num_envs": args.num_envs,
                "steps": args.steps,
                "chunk": args.chunk,
                "epochs_per_chunk": args.epochs_per_chunk,
                "gamma": args.gamma,
                "total_harvest": total_h,
                "total_control": total_c,
                "skill_rate": (total_h + total_c) / max(1, args.steps * N),
                "wall_s": time.time() - t0,
                "bot": "The-International",
            },
        }
        torch.save(ckpt, args.save)
        print(f"[pretrain] saved → {args.save}  meta={ckpt['meta']}", flush=True)
        return 0
    finally:
        for e in envs:
            try:
                e.close()
            except Exception:
                pass
        if writer:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
