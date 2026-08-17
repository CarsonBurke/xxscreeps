#!/usr/bin/env python3
"""Pretrain the entity-aware critic on authoritative TI trajectories.

TI is substantially stronger than the scripted policy, but its concurrent raw
intent language is not identical to the one-slot macro actor ABI.  This command
therefore uses every TI transition for value/representation learning and reports
the exact representable actor-label subset without fabricating missing labels.
"""
from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .artifacts import (
    artifact_meta, atomic_torch_save, directory_signature, source_signature,
)
from .constants import PPO_CFG, VALUE_CFG
from .gae import discounted_returns_tn
from .model import Actor, Critic, count_params
from .rollout_buffer import HostRolloutBuffer
from .ti_intents import summarize_ti_labels, translate_ti_intents
from .vec_env import _clone_host_obs, promote_obs_device
from .env_client import ScreepsEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--epochs-per-chunk", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=float(PPO_CFG["lr"]) * 2)
    parser.add_argument("--gamma", type=float, default=float(PPO_CFG["gamma"]))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--validation-steps", type=int, default=6000)
    parser.add_argument("--min-validation-ev", type=float, default=0.01)
    parser.add_argument("--max-validation-mae-ratio", type=float, default=0.9)
    parser.add_argument("--room", default="W7N3")
    parser.add_argument("--node", default=None)
    parser.add_argument("--bot-dir", default=None)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument(
        "--save", type=Path,
        default=Path(__file__).resolve().parents[1] / "runs" / "ti_critic.pt",
    )
    return parser.parse_args()


def _train_chunk(
    critic: Critic,
    optimizer: torch.optim.Optimizer,
    rollout: HostRolloutBuffer,
    rewards: list[torch.Tensor],
    dones: list[torch.Tensor],
    *,
    next_obs: dict[str, torch.Tensor],
    device: torch.device,
    gamma: float,
    epochs: int,
    minibatch: int,
    use_bf16: bool,
) -> tuple[float, float, int]:
    t = len(rollout)
    rewards_tn = torch.stack(rewards)
    dones_tn = torch.stack(dones)
    with torch.no_grad():
        boot = critic(promote_obs_device(next_obs, device, non_blocking=False)).float().cpu()
    returns = discounted_returns_tn(
        rewards_tn, dones_tn, gamma=gamma, next_value=boot,
        truncations=dones_tn,
    ).reshape(t)
    critic.support.validate_targets(returns)
    observations = rollout.as_flat_obs()
    order = torch.arange(t)
    losses: list[float] = []
    updates = 0
    critic.train()
    for _ in range(epochs):
        perm = order[torch.randperm(t)]
        for start in range(0, t, minibatch):
            indices = perm[start : start + minibatch]
            obs = promote_obs_device(
                observations.gather_minibatch(indices), device, non_blocking=False,
            )
            target = returns[indices].to(device)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_bf16 else nullcontext()
            )
            with context:
                logits = critic(obs, return_logits=True)
                loss = critic.support.cross_entropy(logits.float(), target).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite TI critic loss {float(loss)}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                critic.parameters(), float(VALUE_CFG["criticMaxGradNorm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
    critic.eval()
    with torch.no_grad():
        pred = []
        for start in range(0, t, minibatch):
            indices = order[start : start + minibatch]
            obs = promote_obs_device(
                observations.gather_minibatch(indices), device, non_blocking=False,
            )
            pred.append(critic(obs).float().cpu())
        prediction = torch.cat(pred)
    variance = float(torch.var(returns, unbiased=False))
    ev = float("nan") if variance == 0 else 1.0 - float(
        torch.var(returns - prediction, unbiased=False) / variance
    )
    return float(np.mean(losses)), ev, updates


def _evaluate_critic(
    critic: Critic,
    *,
    steps: int,
    seed: int,
    node: str | None,
    room: str,
    bot_dir: str | None,
    device: torch.device,
    gamma: float,
    minibatch: int,
) -> dict[str, float]:
    """Evaluate state-dependent value prediction on a fresh TI trajectory."""
    if steps <= 1:
        raise ValueError("TI critic validation requires at least two steps")
    env = ScreepsEnv(
        node=node, room=room, max_episode=steps + 5, device="cpu",
        expert=True, bot_dir=bot_dir, lean_meta=True,
        capture_expert_intents=False, seed=seed,
    )
    rollout = HostRolloutBuffer(steps, 1)
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    try:
        obs = env.reset()
        for step in range(steps):
            pre = _clone_host_obs(obs)
            obs, reward, done, _info = env.step()
            reward_row = torch.tensor([reward], dtype=torch.float32)
            done_row = torch.tensor([float(done)], dtype=torch.float32)
            rollout.write_step(
                host_obs=pre,
                types=torch.zeros(1, 1, 1, dtype=torch.uint8),
                dirs=torch.zeros(1, 1, 1, dtype=torch.uint8),
                targets=torch.zeros(1, 1, 1, dtype=torch.uint8),
                amounts=torch.zeros(1, 1, 1, dtype=torch.uint8),
                logprob=torch.zeros(1, 1), value=torch.zeros(1),
                reward=reward_row, done=done_row, trunc=done_row,
            )
            rewards.append(reward_row)
            dones.append(done_row)
            if done:
                raise RuntimeError(f"TI validation environment ended early at {step + 1}")
        with torch.no_grad():
            bootstrap = critic(
                promote_obs_device(obs, device, non_blocking=False),
            ).float().cpu()
        rewards_tn = torch.stack(rewards)
        dones_tn = torch.stack(dones)
        target = discounted_returns_tn(
            rewards_tn, dones_tn, gamma=gamma, next_value=bootstrap,
            truncations=dones_tn,
        ).reshape(steps)
        observations = rollout.as_flat_obs()
        predictions: list[torch.Tensor] = []
        critic.eval()
        with torch.no_grad():
            for start in range(0, steps, minibatch):
                indices = torch.arange(start, min(steps, start + minibatch))
                batch = promote_obs_device(
                    observations.gather_minibatch(indices), device,
                    non_blocking=False,
                )
                predictions.append(critic(batch).float().cpu())
        prediction = torch.cat(predictions)
    finally:
        env.close()
    variance = float(torch.var(target, unbiased=False))
    residual_variance = float(torch.var(target - prediction, unbiased=False))
    ev = float("nan") if variance == 0 else 1.0 - residual_variance / variance
    # The median is the optimal constant under absolute error.
    baseline = torch.full_like(target, float(target.median()))
    mae = float(torch.mean(torch.abs(target - prediction)))
    constant_mae = float(torch.mean(torch.abs(target - baseline)))
    return {
        "validation_ev": ev,
        "validation_mae": mae,
        "validation_constant_mae": constant_mae,
        "validation_mae_ratio": mae / max(constant_mae, 1e-12),
        "validation_prediction_std": float(torch.std(prediction, unbiased=False)),
        "validation_target_std": float(torch.std(target, unbiased=False)),
        "validation_prediction_min": float(prediction.min()),
        "validation_prediction_max": float(prediction.max()),
    }


def main() -> int:
    args = parse_args()
    runtime_source_sha256 = source_signature()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda" and not args.no_bf16
    env = ScreepsEnv(
        node=args.node, room=args.room, max_episode=args.steps + 5,
        device="cpu", expert=True, bot_dir=args.bot_dir, lean_meta=False,
        capture_expert_intents=True, seed=args.seed,
    )
    actor_signature = Actor().to("cpu")
    critic = Critic().to(device)
    optimizer = torch.optim.AdamW(
        critic.parameters(), lr=args.lr, eps=1e-5, weight_decay=0.0,
        fused=device.type == "cuda",
    )
    print(
        f"[ti] steps={args.steps} chunk={args.chunk} device={device} "
        f"critic_params={count_params(critic):,}", flush=True,
    )
    rollout = HostRolloutBuffer(args.chunk, 1)
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    labels = []
    total_h = total_c = 0.0
    last_loss = last_ev = float("nan")
    update_count = 0
    started = time.monotonic()
    try:
        obs = env.reset()
        for step in range(args.steps):
            pre = _clone_host_obs(obs)
            obs, reward, done, info = env.step()
            labels.extend(translate_ti_intents(
                info.get("expertIntents"),
                info.get("expertActorMeta") or [],
                info.get("expertTargetMeta") or [],
                info.get("expertRoomNames") or [],
            ))
            total_h += float(info.get("harvestDelta") or 0)
            total_c += float(info.get("controlDelta") or 0)
            rollout.write_step(
                host_obs=pre,
                types=torch.zeros(1, 1, 1, dtype=torch.uint8),
                dirs=torch.zeros(1, 1, 1, dtype=torch.uint8),
                targets=torch.zeros(1, 1, 1, dtype=torch.uint8),
                amounts=torch.zeros(1, 1, 1, dtype=torch.uint8),
                logprob=torch.zeros(1, 1), value=torch.zeros(1),
                reward=torch.tensor([reward]), done=torch.tensor([float(done)]),
                trunc=torch.tensor([float(done)]),
            )
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            dones.append(torch.tensor([float(done)], dtype=torch.float32))
            flush = len(rollout) == args.chunk or step + 1 == args.steps
            if flush:
                last_loss, last_ev, updates = _train_chunk(
                    critic, optimizer, rollout, rewards, dones,
                    next_obs=obs, device=device, gamma=args.gamma,
                    epochs=args.epochs_per_chunk, minibatch=args.minibatch,
                    use_bf16=use_bf16,
                )
                update_count += updates
                rollout.reset()
                rewards.clear()
                dones.clear()
                print(
                    f"[ti] step={step + 1}/{args.steps} loss={last_loss:.5f} "
                    f"ev={last_ev:.3f} skill={(total_h + total_c)/(step + 1):.3f}",
                    flush=True,
                )
            if done and step + 1 < args.steps:
                raise RuntimeError(f"TI environment ended early at {step + 1}")
    finally:
        env.close()
    label_summary = summarize_ti_labels(labels)
    bot_directory = Path(args.bot_dir).resolve() if args.bot_dir else (
        Path(__file__).resolve().parents[4]
        / "The-International-Open-Source" / "dist"
    )
    validation = _evaluate_critic(
        critic, steps=args.validation_steps, seed=args.seed + 1,
        node=args.node, room=args.room, bot_dir=args.bot_dir, device=device,
        gamma=args.gamma, minibatch=args.minibatch,
    )
    qualified = bool(
        np.isfinite(validation["validation_ev"])
        and validation["validation_ev"] >= args.min_validation_ev
        and np.isfinite(validation["validation_mae_ratio"])
        and validation["validation_mae_ratio"] <= args.max_validation_mae_ratio
    )
    checkpoint = {
        "critic": critic.state_dict(),
        "critic_opt": optimizer.state_dict(),
        "meta": artifact_meta(
            "ti_critic", actor_signature, critic,
            source_sha256=runtime_source_sha256,
            partial=False, qualified=qualified,
            steps=args.steps, seed=args.seed, room=args.room, gamma=args.gamma,
            value_loss=last_loss, explained_variance=last_ev,
            skill_rate=(total_h + total_c) / max(1, args.steps),
            total_harvest=total_h, total_control=total_c,
            exact_label_summary=label_summary, optimizer_updates=update_count,
            ti_bot_source_sha256=directory_signature(bot_directory),
            validation_seed=args.seed + 1,
            validation_steps=args.validation_steps,
            min_validation_ev=args.min_validation_ev,
            max_validation_mae_ratio=args.max_validation_mae_ratio,
            **validation,
            wall_seconds=time.monotonic() - started,
        ),
    }
    args.save.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(checkpoint, args.save)
    print(f"[ti] exact_labels={label_summary}", flush=True)
    print(f"[ti] heldout={validation} qualified={qualified}", flush=True)
    print(f"[ti] checkpoint → {args.save}", flush=True)
    return 0 if checkpoint["meta"]["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
