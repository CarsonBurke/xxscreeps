#!/usr/bin/env python3
"""Same-expert joint BC ∥ critic pretrain.

One env fleet, one teacher stream, both heads optimized on the same (s, a, r):

  L = L_BC + value_coef · L_V
  L_BC = −mean log π(a|s)   (type-gated, masked)
  L_V  = 0.5 · MSE(V(s), R^{λ=1 MC})

v1 expert = scripted only (exact factorized action labels). TI is not a joint
expert until labels in this action contract exist; do not pair scripted BC with
a TI critic.

  python3 -m samples.rl.agent.pretrain_joint \\
    --num-envs 4 --steps 6000 --chunk 512 --device cpu \\
    --save samples/rl/runs/joint_pretrain_v2.pt

Do NOT queue via mlq until readiness gates (docs) pass and user authorizes.
"""
from __future__ import annotations

import argparse
import sys
import time
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
    from samples.rl.agent.artifacts import (
        artifact_meta,
        atomic_torch_save,
        load_full_state,
        validate_artifact,
    )
    from samples.rl.agent.actions_util import safe_bc_nll
    from samples.rl.agent.constants import INTENT_TYPES, PPO_CFG, SCHEMA
    from samples.rl.agent.gae import mc_returns_tn
    from samples.rl.agent.model import Actor, Critic, count_params, maybe_compile
    from samples.rl.agent.vec_env import VecScreepsEnv, _clone_host_obs, promote_obs_device
except ImportError:
    from agent.artifacts import artifact_meta, atomic_torch_save, load_full_state, validate_artifact
    from agent.actions_util import safe_bc_nll
    from agent.constants import INTENT_TYPES, PPO_CFG, SCHEMA
    from agent.gae import mc_returns_tn
    from agent.model import Actor, Critic, count_params, maybe_compile
    from agent.vec_env import VecScreepsEnv, _clone_host_obs, promote_obs_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Same-expert joint BC+critic pretrain (schema v2)",
    )
    p.add_argument(
        "--expert",
        choices=("scripted",),
        default="scripted",
        help="joint expert (scripted only; TI blocked without action labels)",
    )
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=6000, help="total steps per env")
    p.add_argument("--chunk", type=int, default=512, help="collect this many steps then train both heads")
    p.add_argument("--epochs-per-chunk", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=float(PPO_CFG["lr"]) * 2)
    p.add_argument("--gamma", type=float, default=float(PPO_CFG["gamma"]))
    p.add_argument(
        "--value-coef",
        type=float,
        default=1.0,
        help="weight on L_V relative to L_BC (both optimized each mb)",
    )
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--max-episode", type=int, default=6000,
                   help="episode length; must cover empty-economy expansion")
    p.add_argument("--room", type=str, default="W7N3")
    p.add_argument(
        "--curriculum",
        type=str,
        default="empty,seed_creep,seed_full,seed_claimer",
        help="comma-separated curriculum stages distributed across env workers",
    )
    p.add_argument("--node", type=str, default=None)
    p.add_argument("--save", type=Path, default=_RL_ROOT / "runs" / "joint_pretrain_v2.pt")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--validation-steps", type=int, default=256)
    p.add_argument("--closed-loop-steps", type=int, default=6000)
    p.add_argument("--min-closed-loop-rate", type=float, default=0.1)
    p.add_argument("--min-validation-ev", type=float, default=0.0)
    p.add_argument("--min-closed-loop-creeps", type=int, default=24)
    p.add_argument("--min-closed-loop-claims", type=int, default=1)
    p.add_argument("--min-teacher-delivery", type=float, default=1.0)
    p.add_argument("--min-teacher-build", type=float, default=1.0)
    p.add_argument("--min-teacher-claims", type=int, default=1)
    p.add_argument("--min-teacher-creeps", type=int, default=24)
    p.add_argument("--logdir", type=Path, default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _train_chunk(
    actor: nn.Module,
    critic: nn.Module,
    actor_c: nn.Module,
    critic_c: nn.Module,
    opt_a: torch.optim.Optimizer,
    opt_c: torch.optim.Optimizer,
    buf_obs: list[dict[str, torch.Tensor]],
    buf_act: list[dict[str, torch.Tensor]],
    rewards_tn: torch.Tensor,
    dones_tn: torch.Tensor,
    *,
    next_obs: dict[str, torch.Tensor] | None,
    term_obs_rows: list[list[dict[str, torch.Tensor] | None]],
    bc_valid_tn: torch.Tensor | None = None,
    gamma: float,
    epochs: int,
    mb: int,
    value_coef: float,
    device: torch.device,
    use_bf16: bool,
    writer,
    global_step: int,
) -> tuple[float, float, float, int, dict[str, float]]:
    """Joint CE + MC value fit on one chunk.

    `bc_valid_tn` optional [T, N] float/bool — 0 drops that transition from BC only
    (e.g. env crash recovery with empty action labels). Critic still sees r/done.
    """
    T, N = rewards_tn.shape
    keys = [k for k in buf_obs[0].keys() if not k.startswith("_")]
    # list of length T; each entry stacked over N → stack to [T, N, ...]
    obs_tn = {k: torch.stack([buf_obs[t][k] for t in range(T)], dim=0) for k in keys}
    act_tn = {
        k: torch.stack([buf_act[t][k] for t in range(T)], dim=0)
        for k in ("types", "dirs", "targets", "amounts")
    }
    if bc_valid_tn is None:
        bc_valid_flat = torch.ones(T * N, dtype=torch.bool)
    else:
        bc_valid_flat = bc_valid_tn.reshape(T * N).cpu() > 0.5

    try:
        from samples.rl.agent.env_client import stack_batches
    except ImportError:
        from agent.env_client import stack_batches

    with torch.no_grad():
        if next_obs is not None:
            raw = {
                k: v for k, v in next_obs.items()
                if not str(k).startswith("_") and torch.is_tensor(v)
            }
            boot_in = promote_obs_device(raw, device, non_blocking=False)
        else:
            boot_in = promote_obs_device(
                {k: v[-1] for k, v in obs_tn.items()}, device, non_blocking=False,
            )
        boot = critic_c(boot_in).float().cpu()

        next_values_tn = torch.zeros(T, N)
        next_values_tn[-1] = boot
        term_batches: list[dict[str, torch.Tensor]] = []
        term_slots: list[tuple[int, int]] = []
        for t_i, row in enumerate(term_obs_rows):
            for i, term in enumerate(row):
                if term is None:
                    continue
                batch = {
                    k: (v if v.dim() > 0 and v.shape[0] == 1 else v.unsqueeze(0))
                    for k, v in term.items()
                    if torch.is_tensor(v)
                }
                term_batches.append(batch)
                term_slots.append((t_i, i))
        if term_batches:
            stacked = promote_obs_device(stack_batches(term_batches), device, non_blocking=False)
            vals = critic_c(stacked).float().cpu()
            for j, (t_i, i) in enumerate(term_slots):
                next_values_tn[t_i, i] = vals[j]

    # Time-limit env: every done is a truncation today.
    trunc = dones_tn.clone()
    returns = mc_returns_tn(
        rewards_tn,
        dones_tn,
        gamma=gamma,
        next_value=boot,
        truncations=trunc,
        next_values_tn=next_values_tn,
    )

    # Keep the corpus typed on host. Promote only the active minibatch; promoting
    # a documented 512×4 patch chunk at once consumed multiple GiB of device RAM.
    obs_flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in obs_tn.items()}
    act_flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in act_tn.items()}
    ret_flat = returns.reshape(T * N)
    B = T * N
    idx = torch.arange(B)

    nlls: list[float] = []
    vlosses: list[float] = []
    legal_fracs: list[float] = []
    factor_nll_sum = torch.zeros(4, dtype=torch.float64)
    factor_count = torch.zeros(4, dtype=torch.float64)
    with torch.no_grad():
        # Count live-actor primary slot types only (ignore padded none mass).
        am = obs_flat.get("actor_mask")
        types = act_flat["types"]  # [B, A, S]
        if am is not None and types.dim() == 3:
            live = am > 0.5  # [B, A]
            chosen = types[..., 0][live].long()
        else:
            chosen = types.reshape(-1).long()
        chosen = chosen[(chosen >= 0) & (chosen < len(INTENT_TYPES))]
        type_hist = torch.bincount(chosen, minlength=len(INTENT_TYPES)).cpu()

    actor.train()
    critic.train()
    for _ in range(epochs):
        perm = idx[torch.randperm(B)]
        for start in range(0, B, mb):
            inds = perm[start : start + mb]
            if inds.numel() == 0:
                continue
            batch_obs = promote_obs_device(
                {k: v[inds] for k, v in obs_flat.items()}, device, non_blocking=False,
            )
            batch_act = {k: v[inds].to(device) for k, v in act_flat.items()}
            target = ret_flat[inds].to(device)
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with ctx:
                out = actor_c(batch_obs, action=batch_act)
                # Validate each active conditional factor. One bad target/amount must
                # not silently discard an otherwise useful actor label.
                lp = out.factor_logprob
                bc_ok = bc_valid_flat[inds].to(device)
                eligible = bc_ok[:, None, None] & out.factor_active
                nll, frac_legal = safe_bc_nll(lp, eligible, strict=True)
                with torch.no_grad():
                    factor_nll_sum += (
                        -torch.where(eligible, lp, torch.zeros_like(lp)).sum(dim=(0, 1))
                    ).double().cpu()
                    factor_count += eligible.sum(dim=(0, 1)).double().cpu()
                pred = critic_c(batch_obs)
                vloss = 0.5 * (pred.float() - target).pow(2).mean()
                loss = nll + value_coef * vloss
            if not torch.isfinite(loss).all():
                print(
                    f"[joint] skip non-finite loss nll={float(nll.detach())} "
                    f"vloss={float(vloss.detach())} legal_frac={frac_legal:.3f}",
                    flush=True,
                )
                continue
            opt_a.zero_grad(set_to_none=True)
            opt_c.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), float(PPO_CFG.get("maxGradNorm", 0.5)))
            nn.utils.clip_grad_norm_(critic.parameters(), float(PPO_CFG.get("maxGradNorm", 0.5)))
            opt_a.step()
            opt_c.step()
            nlls.append(float(nll.detach()))
            vlosses.append(float(vloss.detach()))
            legal_fracs.append(frac_legal)
            global_step += int(inds.numel())
            if writer is not None:
                writer.add_scalar("joint/bc_nll", nlls[-1], global_step)
                writer.add_scalar("joint/value_loss", vlosses[-1], global_step)
                writer.add_scalar("joint/bc_legal_frac", frac_legal, global_step)

    with torch.no_grad():
        sub = idx[torch.randperm(B)[: min(4096, B)]]
        eval_obs = promote_obs_device(
            {k: v[sub] for k, v in obs_flat.items()}, device, non_blocking=False,
        )
        pred = critic(eval_obs).float().cpu().numpy()
        y = ret_flat[sub].cpu().numpy()
        var_y = float(np.var(y))
        ev = float("nan") if var_y == 0 else 1.0 - float(np.var(y - pred) / var_y)

    hist = {
        INTENT_TYPES[i]: int(type_hist[i])
        for i in range(len(INTENT_TYPES))
        if int(type_hist[i]) > 0
    }
    if legal_fracs:
        hist["_bc_legal_frac"] = float(np.mean(legal_fracs))
    for index, name in enumerate(("type", "direction", "target", "amount")):
        if factor_count[index] > 0:
            hist[f"_bc_{name}_nll"] = float(factor_nll_sum[index] / factor_count[index])
            hist[f"_bc_{name}_count"] = int(factor_count[index])
    return (
        float(np.mean(nlls)) if nlls else float("nan"),
        float(np.mean(vlosses)) if vlosses else float("nan"),
        ev,
        global_step,
        hist,
    )


@torch.inference_mode()
def _validate_teacher_forced(
    actor: Actor,
    critic: Critic,
    envs: VecScreepsEnv,
    *,
    steps: int,
    device: torch.device,
    gamma: float,
) -> dict[str, float]:
    """Fresh teacher-forced actor legality plus critic return validation."""
    envs.reset()
    total_nll = 0.0
    total_factors = 0
    total_skill = 0.0
    total_delivery = 0.0
    total_build = 0.0
    total_claims = 0.0
    max_creeps = 0
    values_rows: list[torch.Tensor] = []
    rewards_rows: list[torch.Tensor] = []
    dones_rows: list[torch.Tensor] = []
    type_hist = torch.zeros(len(INTENT_TYPES), dtype=torch.long)
    actor.eval()
    critic.eval()
    for _ in range(max(1, steps)):
        assert envs.host_obs is not None
        obs = promote_obs_device(envs.host_obs, device, non_blocking=False)
        values_rows.append(critic(obs).float().cpu())
        _next, reward, done, infos, actions = envs.step_scripted()
        rewards_rows.append(reward.float().cpu())
        dones_rows.append(done.float().cpu())
        action_device = {key: value.to(device) for key, value in actions.items()}
        out = actor(obs, action=action_device)
        eligible = out.factor_active
        nll, legal = safe_bc_nll(out.factor_logprob, eligible, strict=True)
        count = int(eligible.sum().item())
        total_nll += float(nll.item()) * count
        total_factors += count
        if legal != 1.0:
            raise RuntimeError(f"teacher-forced legal coverage={legal:.6f}")
        total_skill += sum(
            float((info or {}).get("harvestDelta") or 0)
            + float((info or {}).get("controlDelta") or 0)
            for info in infos
        )
        total_delivery += sum(float((info or {}).get("transferDelta") or 0) for info in infos)
        total_build += sum(float((info or {}).get("buildDelta") or 0) for info in infos)
        total_claims += sum(float((info or {}).get("claimDelta") or 0) for info in infos)
        teacher_invalid = sum(int((info or {}).get("intentInvalid") or 0) for info in infos)
        if teacher_invalid:
            raise RuntimeError(
                f"teacher-forced validation produced {teacher_invalid} engine-invalid intents"
            )
        max_creeps = max(
            [max_creeps, *(int((info or {}).get("creeps") or 0) for info in infos)],
        )
        live = obs["actor_mask"].bool()
        chosen = actions["types"][..., 0][live.cpu()].long()
        chosen = chosen[(chosen >= 0) & (chosen < len(INTENT_TYPES))]
        type_hist += torch.bincount(chosen, minlength=len(INTENT_TYPES))
    assert envs.host_obs is not None
    final_obs = promote_obs_device(envs.host_obs, device, non_blocking=False)
    next_value = critic(final_obs).float().cpu()
    rewards_tn = torch.stack(rewards_rows)
    dones_tn = torch.stack(dones_rows)
    returns = mc_returns_tn(
        rewards_tn,
        dones_tn,
        gamma=gamma,
        next_value=next_value,
        truncations=dones_tn,
    )
    predicted = torch.stack(values_rows)
    target_np = returns.numpy()
    pred_np = predicted.numpy()
    variance = float(np.var(target_np))
    validation_ev = (
        float("nan") if variance == 0
        else 1.0 - float(np.var(target_np - pred_np) / variance)
    )
    denominator = max(1, steps * envs.n)
    metrics = {
        "validation_factor_nll": total_nll / max(1, total_factors),
        "validation_legal_frac": 1.0,
        "validation_teacher_skill_rate": total_skill / denominator,
        "validation_value_ev": validation_ev,
        "validation_delivery": total_delivery,
        "validation_build": total_build,
        "validation_claims": total_claims,
        "validation_max_creeps": float(max_creeps),
    }
    for index, name in enumerate(INTENT_TYPES):
        metrics[f"validation_intent_{name}_count"] = float(type_hist[index])
    return metrics


@torch.inference_mode()
def _validate_closed_loop(
    actor: Actor,
    envs: VecScreepsEnv,
    *,
    steps: int,
    device: torch.device,
) -> dict[str, object]:
    """Run the learned deterministic actor in fresh environments."""
    obs = envs.reset()
    actor.eval()
    total_skill = 0.0
    invalid = 0
    issued = 0
    max_creeps = 0
    total_claims = 0
    by_curriculum: dict[str, dict[str, float]] = {}
    for _ in range(max(1, steps)):
        out = actor(obs, deterministic=True)
        obs, _reward, _done, infos = envs.step({
            "types": out.types,
            "dirs": out.dirs,
            "targets": out.targets,
            "amounts": out.amounts,
        })
        for info in infos:
            info = info or {}
            harvest = float(info.get("harvestDelta") or 0)
            control = float(info.get("controlDelta") or 0)
            delivered = float(info.get("transferDelta") or 0)
            built = float(info.get("buildDelta") or 0)
            claims = int(info.get("claimDelta") or 0)
            creeps = int(info.get("creeps") or 0)
            total_skill += harvest + control
            max_creeps = max(max_creeps, creeps)
            total_claims += claims
            stage = str(info.get("curriculum") or "unknown")
            stage_metrics = by_curriculum.setdefault(stage, {
                "transitions": 0.0,
                "skill": 0.0,
                "delivery": 0.0,
                "build": 0.0,
                "claims": 0.0,
                "max_creeps": 0.0,
                "issued": 0.0,
                "invalid": 0.0,
            })
            stage_metrics["transitions"] += 1.0
            stage_metrics["skill"] += harvest + control
            stage_metrics["delivery"] += delivered
            stage_metrics["build"] += built
            stage_metrics["claims"] += claims
            stage_metrics["max_creeps"] = max(stage_metrics["max_creeps"], float(creeps))
            for result in info.get("intentResults") or ():
                issued += 1
                stage_metrics["issued"] += 1.0
                if int(result.get("code", -1)) not in (C_OK, -4, -11):
                    invalid += 1
                    stage_metrics["invalid"] += 1.0
    for stage_metrics in by_curriculum.values():
        transitions = max(1.0, stage_metrics["transitions"])
        stage_metrics["skill_rate"] = stage_metrics["skill"] / transitions
        stage_metrics["invalid_frac"] = (
            stage_metrics["invalid"] / max(1.0, stage_metrics["issued"])
        )
    denominator = max(1, steps * envs.n)
    return {
        "closed_loop_skill_rate": total_skill / denominator,
        "closed_loop_invalid_frac": invalid / max(1, issued),
        "closed_loop_max_creeps": float(max_creeps),
        "closed_loop_claims": float(total_claims),
        "closed_loop_by_curriculum": by_curriculum,
    }


# Screeps OK is numerically zero; named locally to keep validation independent of JS imports.
C_OK = 0


def _joint_checkpoint(
    actor: Actor,
    critic: Critic,
    opt_a: torch.optim.Optimizer,
    opt_c: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    step: int,
    chunk_index: int,
    global_step: int,
    total_harvest: float,
    total_control: float,
    total_delivered: float,
    total_built: float,
    total_claims: int,
    max_creeps: int,
    spawn_success: int,
    nll: float,
    value_loss: float,
    explained_variance: float,
    action_hist: dict,
    teacher_by_curriculum: dict[str, dict[str, float]],
    partial: bool,
    wall_s: float,
) -> dict:
    skill = (total_harvest + total_control) / max(1, step * args.num_envs)
    return {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_opt": opt_a.state_dict(),
        "critic_opt": opt_c.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "meta": artifact_meta(
            "joint_pretrain",
            actor,
            critic,
            partial=bool(partial),
            qualified=False,
            expert=args.expert,
            curriculum=args.curriculum,
            room=args.room,
            num_envs=args.num_envs,
            steps=args.steps,
            step=int(step),
            chunk_index=int(chunk_index),
            chunk=args.chunk,
            epochs_per_chunk=args.epochs_per_chunk,
            minibatch=args.minibatch,
            max_episode=args.max_episode,
            gamma=args.gamma,
            value_coef=args.value_coef,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            seed=args.seed,
            validation_steps=args.validation_steps,
            closed_loop_steps=args.closed_loop_steps,
            min_validation_ev=args.min_validation_ev,
            min_closed_loop_rate=args.min_closed_loop_rate,
            min_closed_loop_creeps=args.min_closed_loop_creeps,
            min_closed_loop_claims=args.min_closed_loop_claims,
            min_teacher_delivery=args.min_teacher_delivery,
            min_teacher_build=args.min_teacher_build,
            min_teacher_claims=args.min_teacher_claims,
            min_teacher_creeps=args.min_teacher_creeps,
            total_harvest=float(total_harvest),
            total_control=float(total_control),
            total_delivered=float(total_delivered),
            total_built=float(total_built),
            total_claims=int(total_claims),
            max_creeps=int(max_creeps),
            spawn_success=int(spawn_success),
            skill_rate=float(skill),
            final_bc_nll=float(nll),
            final_value_loss=float(value_loss),
            final_ev=float(explained_variance),
            action_type_hist=action_hist,
            teacher_by_curriculum=teacher_by_curriculum,
            reward=SCHEMA["reward"],
            critic_trunk_only_for_ppo=True,
            wall_s=float(wall_s),
        ),
        "update": -1,
        "global_step": int(global_step),
    }


def _qualification_failures(
    args: argparse.Namespace,
    *,
    last_nll: float,
    last_vloss: float,
    corpus_hist: dict[str, float],
    skill: float,
    total_delivered: float,
    total_built: float,
    total_claims: int,
    max_creeps: int,
    teacher_by_curriculum: dict[str, dict[str, float]],
    validation: dict[str, float],
    closed_loop: dict[str, object],
) -> list[str]:
    empty_teacher = teacher_by_curriculum.get("empty", {})
    teacher_invalid = sum(
        float(metrics.get("invalid", 0))
        for metrics in teacher_by_curriculum.values()
    )
    closed_loop_stages = closed_loop.get("closed_loop_by_curriculum", {})
    empty_closed = (
        closed_loop_stages.get("empty", {})
        if isinstance(closed_loop_stages, dict)
        else {}
    )
    checks = {
        "finite_actor_loss": np.isfinite(last_nll),
        "finite_critic_loss": np.isfinite(last_vloss),
        "teacher_factor_legality": float(corpus_hist.get("_bc_legal_frac", 0.0)) == 1.0,
        "teacher_engine_legality": teacher_invalid == 0,
        "teacher_skill": skill > 0,
        "teacher_delivery": total_delivered >= args.min_teacher_delivery,
        "teacher_build": total_built >= args.min_teacher_build,
        "teacher_claim": total_claims >= args.min_teacher_claims,
        "teacher_population": max_creeps >= args.min_teacher_creeps,
        # Seeded curricula are useful data augmentation, but cannot certify the
        # end-to-end objective. The empty worker must bootstrap, build, scale,
        # and claim from its own production in both teacher and learned runs.
        "empty_teacher_coverage": float(empty_teacher.get("transitions", 0)) > 0,
        "empty_teacher_engine_legality": float(empty_teacher.get("invalid", 0)) == 0,
        "empty_teacher_skill": float(empty_teacher.get("skill", 0)) > 0,
        "empty_teacher_delivery": (
            float(empty_teacher.get("delivery", 0)) >= args.min_teacher_delivery
        ),
        "empty_teacher_build": float(empty_teacher.get("build", 0)) >= args.min_teacher_build,
        "empty_teacher_claim": (
            float(empty_teacher.get("claims", 0)) >= args.min_teacher_claims
        ),
        "empty_teacher_population": (
            float(empty_teacher.get("max_creeps", 0)) >= args.min_teacher_creeps
        ),
        "validation_actor_loss": np.isfinite(validation["validation_factor_nll"]),
        "validation_legality": validation["validation_legal_frac"] == 1.0,
        "validation_critic_ev": (
            np.isfinite(validation["validation_value_ev"])
            and validation["validation_value_ev"] >= args.min_validation_ev
        ),
        "closed_loop_skill": (
            closed_loop["closed_loop_skill_rate"] >= args.min_closed_loop_rate
        ),
        "closed_loop_legality": closed_loop["closed_loop_invalid_frac"] <= 0.01,
        "closed_loop_population": (
            closed_loop["closed_loop_max_creeps"] >= args.min_closed_loop_creeps
        ),
        "closed_loop_claim": (
            float(closed_loop["closed_loop_claims"]) >= args.min_closed_loop_claims
        ),
        "empty_closed_loop_coverage": float(empty_closed.get("transitions", 0)) > 0,
        "empty_closed_loop_skill": (
            float(empty_closed.get("skill_rate", 0)) >= args.min_closed_loop_rate
        ),
        "empty_closed_loop_legality": float(empty_closed.get("invalid_frac", 1)) <= 0.01,
        "empty_closed_loop_delivery": (
            float(empty_closed.get("delivery", 0)) >= args.min_teacher_delivery
        ),
        "empty_closed_loop_build": (
            float(empty_closed.get("build", 0)) >= args.min_teacher_build
        ),
        "empty_closed_loop_population": (
            float(empty_closed.get("max_creeps", 0)) >= args.min_closed_loop_creeps
        ),
        "empty_closed_loop_claim": (
            float(empty_closed.get("claims", 0)) >= args.min_closed_loop_claims
        ),
    }
    return [name for name, passed in checks.items() if not bool(passed)]


def main() -> int:
    args = parse_args()
    if args.expert != "scripted":
        print(
            f"[joint] expert={args.expert!r} not supported for joint pretrain "
            "(need factorized action labels; v1=scripted only)",
            flush=True,
        )
        return 2
    if args.validation_steps >= args.max_episode:
        raise SystemExit(
            "[joint] validation-steps must be shorter than max-episode so fresh "
            "critic returns do not cross an auto-reset boundary"
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    if args.logdir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.logdir = _RL_ROOT / "runs" / "tb-joint-pretrain" / stamp
    args.logdir.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(args.logdir))
        writer.add_text("run/expert", args.expert)
        writer.add_text("run/curriculum", args.curriculum)
    except Exception as err:  # noqa: BLE001
        print(f"[joint] TB disabled: {err}", flush=True)

    print(
        f"[joint] expert={args.expert} curriculum={args.curriculum} "
        f"envs={args.num_envs} steps={args.steps} chunk={args.chunk} "
        f"max_episode={args.max_episode} device={device} reward=productive-economy",
        flush=True,
    )
    print(f"[joint] tensorboard logdir={args.logdir}", flush=True)
    print(
        "[joint] same-expert stream: BC NLL + critic MC on identical (s,a,r) — "
        "do not pair with TI critic separately",
        flush=True,
    )

    # Host-side collect; train promotes to device.
    envs = VecScreepsEnv(
        args.num_envs,
        node=args.node,
        room=args.room,
        max_episode=args.max_episode,
        device="cpu",
        curriculum=args.curriculum,
    )
    actor = Actor().to(device)
    critic = Critic().to(device)
    print(
        f"[joint] actor_params={count_params(actor):,} critic_params={count_params(critic):,}",
        flush=True,
    )
    actor_c = maybe_compile(actor, not args.no_compile and device.type == "cuda", "actor")
    critic_c = maybe_compile(critic, not args.no_compile and device.type == "cuda", "critic")
    opt_a = torch.optim.AdamW(actor.parameters(), lr=args.lr_actor, eps=1e-5)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=args.lr_critic, eps=1e-5)
    start_step = 0
    total_h = total_c = total_delivered = total_built = 0.0
    total_claims = 0
    max_creeps = 0
    total_spawn = 0
    corpus_hist: dict[str, float] = {}
    teacher_by_curriculum: dict[str, dict[str, float]] = {}
    global_step = 0
    chunk_i = 0
    resume_rng: tuple[torch.Tensor | None, tuple | None] = (None, None)
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if not isinstance(state, dict):
            raise SystemExit("[joint] resume is not a complete artifact")
        try:
            meta = validate_artifact(state, actor, critic, kinds=("joint_pretrain",))
        except ValueError as error:
            raise SystemExit(f"[joint] incompatible resume: {error}") from error
        continuation = {
            "expert": args.expert,
            "curriculum": args.curriculum,
            "room": args.room,
            "num_envs": args.num_envs,
            "chunk": args.chunk,
            "epochs_per_chunk": args.epochs_per_chunk,
            "minibatch": args.minibatch,
            "max_episode": args.max_episode,
            "gamma": args.gamma,
            "value_coef": args.value_coef,
            "lr_actor": args.lr_actor,
            "lr_critic": args.lr_critic,
            "seed": args.seed,
            "validation_steps": args.validation_steps,
            "closed_loop_steps": args.closed_loop_steps,
            "min_validation_ev": args.min_validation_ev,
            "min_closed_loop_rate": args.min_closed_loop_rate,
            "min_closed_loop_creeps": args.min_closed_loop_creeps,
            "min_closed_loop_claims": args.min_closed_loop_claims,
            "min_teacher_delivery": args.min_teacher_delivery,
            "min_teacher_build": args.min_teacher_build,
            "min_teacher_claims": args.min_teacher_claims,
            "min_teacher_creeps": args.min_teacher_creeps,
        }
        mismatched = {
            key: (meta.get(key), value)
            for key, value in continuation.items()
            if meta.get(key) != value
        }
        if mismatched:
            raise SystemExit(f"[joint] continuation parameters differ: {mismatched}")
        if "actor_opt" not in state or "critic_opt" not in state:
            raise SystemExit("[joint] continuation requires both optimizer states")
        load_full_state(actor, state["actor"], name="actor")
        load_full_state(critic, state["critic"], name="critic")
        opt_a.load_state_dict(state["actor_opt"])
        opt_c.load_state_dict(state["critic_opt"])
        start_step = int(meta.get("step", 0))
        if start_step >= args.steps:
            raise SystemExit(
                f"[joint] resume step={start_step} has already reached requested steps={args.steps}"
            )
        total_h = float(meta.get("total_harvest", 0))
        total_c = float(meta.get("total_control", 0))
        total_delivered = float(meta.get("total_delivered", 0))
        total_built = float(meta.get("total_built", 0))
        total_claims = int(meta.get("total_claims", 0))
        max_creeps = int(meta.get("max_creeps", 0))
        total_spawn = int(meta.get("spawn_success", 0))
        corpus_hist = {
            str(key): float(value)
            for key, value in (meta.get("action_type_hist") or {}).items()
        }
        teacher_by_curriculum = {
            str(stage): {str(key): float(value) for key, value in metrics.items()}
            for stage, metrics in (meta.get("teacher_by_curriculum") or {}).items()
        }
        global_step = int(state.get("global_step", start_step * args.num_envs))
        chunk_i = int(meta.get("chunk_index", 0))
        resume_rng = (state.get("torch_rng_state"), state.get("numpy_rng_state"))
        print(f"[joint] resumed {args.resume} at step={start_step} chunk={chunk_i}", flush=True)
    use_bf16 = (
        (not args.no_bf16)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    mb = max(8, int(args.minibatch))

    envs.reset()
    assert envs.host_obs is not None
    if start_step:
        print(
            f"[joint] deterministically replaying {start_step} teacher steps to restore env stage",
            flush=True,
        )
        for replay_step in range(start_step):
            envs.step_scripted()
            if (replay_step + 1) % 500 == 0:
                print(f"[joint] replay {replay_step + 1}/{start_step}", flush=True)
        torch_state, numpy_state = resume_rng
        if torch_state is not None:
            torch.set_rng_state(torch_state.cpu())
        if numpy_state is not None:
            np.random.set_state(numpy_state)

    buf_obs: list[dict[str, torch.Tensor]] = []
    buf_act: list[dict[str, torch.Tensor]] = []
    rewards_rows: list[torch.Tensor] = []
    dones_rows: list[torch.Tensor] = []
    bc_valid_rows: list[torch.Tensor] = []
    term_obs_rows: list[list[dict[str, torch.Tensor] | None]] = []
    last_nll = last_vloss = last_ev = float("nan")
    last_hist: dict[str, float] = {}
    t0 = time.time()
    N = args.num_envs

    try:
        for t in range(start_step, args.steps):
            # Pre-step decision state (post-tick of previous / post-reset).
            pre = _clone_host_obs(envs.host_obs)
            # Store stacked [N, ...] rows for the chunk.
            buf_obs.append(pre)

            _obs, rew, done, infos, acts = envs.step_scripted()
            buf_act.append(acts)
            rewards_rows.append(rew.detach().cpu().float())
            dones_rows.append(done.detach().cpu().float())

            term_row: list[dict[str, torch.Tensor] | None] = []
            bc_row = torch.ones(N, dtype=torch.float32)
            for i, info in enumerate(infos):
                info = info or {}
                harvest = float(info.get("harvestDelta") or 0)
                control = float(info.get("controlDelta") or 0)
                delivered = float(info.get("transferDelta") or 0)
                built = float(info.get("buildDelta") or 0)
                claims = int(info.get("claimDelta") or 0)
                creeps = int(info.get("creeps") or 0)
                spawn_success = int(info.get("spawnSuccess") or 0)
                intent_issued = int(info.get("intentIssued") or 0)
                intent_invalid = int(info.get("intentInvalid") or 0)
                total_h += harvest
                total_c += control
                total_delivered += delivered
                total_built += built
                total_claims += claims
                max_creeps = max(max_creeps, creeps)
                total_spawn += spawn_success
                stage = str(info.get("curriculum") or "unknown")
                stage_metrics = teacher_by_curriculum.setdefault(stage, {
                    "transitions": 0.0,
                    "skill": 0.0,
                    "delivery": 0.0,
                    "build": 0.0,
                    "claims": 0.0,
                    "max_creeps": 0.0,
                    "spawn_success": 0.0,
                    "issued": 0.0,
                    "invalid": 0.0,
                })
                stage_metrics["transitions"] += 1.0
                stage_metrics["skill"] += harvest + control
                stage_metrics["delivery"] += delivered
                stage_metrics["build"] += built
                stage_metrics["claims"] += claims
                stage_metrics["max_creeps"] = max(stage_metrics["max_creeps"], float(creeps))
                stage_metrics["spawn_success"] += spawn_success
                stage_metrics["issued"] += intent_issued
                stage_metrics["invalid"] += intent_invalid
                if intent_invalid:
                    raise RuntimeError(
                        f"teacher curriculum={stage!r} produced {intent_invalid} "
                        f"engine-invalid intents at step={t + 1}"
                    )
                if info.get("invalid_demo") or info.get("recovered"):
                    bc_row[i] = 0.0
                term = info.get("terminal_observation")
                if term is not None:
                    term_row.append({
                        k: v.detach().cpu() if torch.is_tensor(v) else v
                        for k, v in term.items()
                        if torch.is_tensor(v) and not str(k).startswith("_")
                    })
                else:
                    term_row.append(None)
            bc_valid_rows.append(bc_row)
            term_obs_rows.append(term_row)

            flush = len(buf_obs) >= args.chunk or t == args.steps - 1
            if flush and buf_obs:
                rewards_tn = torch.stack(rewards_rows, dim=0)
                dones_tn = torch.stack(dones_rows, dim=0)
                bc_valid_tn = torch.stack(bc_valid_rows, dim=0)
                next_host = envs.host_obs  # s_T after last step
                last_nll, last_vloss, last_ev, global_step, last_hist = _train_chunk(
                    actor,
                    critic,
                    actor_c,
                    critic_c,
                    opt_a,
                    opt_c,
                    buf_obs,
                    buf_act,
                    rewards_tn,
                    dones_tn,
                    next_obs=next_host,
                    term_obs_rows=term_obs_rows,
                    bc_valid_tn=bc_valid_tn,
                    gamma=args.gamma,
                    epochs=args.epochs_per_chunk,
                    mb=mb,
                    value_coef=args.value_coef,
                    device=device,
                    use_bf16=use_bf16,
                    writer=writer,
                    global_step=global_step,
                )
                for name, value in last_hist.items():
                    if name.startswith("_"):
                        corpus_hist[name] = float(value)
                    else:
                        corpus_hist[name] = corpus_hist.get(name, 0.0) + float(value)
                chunk_i += 1
                rate = (total_h + total_c) / max(1, (t + 1) * N)
                sps = (t + 1) / max(1e-6, time.time() - t0)
                print(
                    f"[joint] step {t + 1}/{args.steps} chunk={chunk_i} "
                    f"nll={last_nll:.4f} vloss={last_vloss:.5f} ev={last_ev:.3f} "
                    f"skill≈{rate:.2f}e/t spawnΣ={total_spawn} sps≈{sps:.2f}",
                    flush=True,
                )
                if writer:
                    writer.add_scalar("joint/chunk_nll", last_nll, chunk_i)
                    writer.add_scalar("joint/chunk_vloss", last_vloss, chunk_i)
                    writer.add_scalar("joint/explained_variance", last_ev, chunk_i)
                    writer.add_scalar("joint/skill_rate_et", rate, chunk_i)
                    writer.add_scalar("joint/spawn_success_cum", total_spawn, chunk_i)
                buf_obs.clear()
                buf_act.clear()
                rewards_rows.clear()
                dones_rows.clear()
                bc_valid_rows.clear()
                term_obs_rows.clear()
                if chunk_i % 5 == 0:
                    snapshot = _joint_checkpoint(
                        actor,
                        critic,
                        opt_a,
                        opt_c,
                        args=args,
                        step=t + 1,
                        chunk_index=chunk_i,
                        global_step=global_step,
                        total_harvest=total_h,
                        total_control=total_c,
                        total_delivered=total_delivered,
                        total_built=total_built,
                        total_claims=total_claims,
                        max_creeps=max_creeps,
                        spawn_success=total_spawn,
                        nll=last_nll,
                        value_loss=last_vloss,
                        explained_variance=last_ev,
                        action_hist=corpus_hist,
                        teacher_by_curriculum=teacher_by_curriculum,
                        partial=True,
                        wall_s=time.time() - t0,
                    )
                    atomic_torch_save(snapshot, args.save)
                    print(f"[joint] checkpoint → {args.save}", flush=True)

            if (t + 1) % 50 == 0 and not flush:
                print(
                    f"[joint] collect {t + 1}/{args.steps} "
                    f"buf={len(buf_obs)} HΣ={total_h:.0f} CΣ={total_c:.0f}",
                    flush=True,
                )

        skill = (total_h + total_c) / max(1, args.steps * N)
        validation_envs = VecScreepsEnv(
            args.num_envs,
            node=args.node,
            room=args.room,
            max_episode=args.max_episode,
            device=device,
            curriculum=args.curriculum,
            lean_meta=False,
        )
        try:
            validation = _validate_teacher_forced(
                actor,
                critic,
                validation_envs,
                steps=args.validation_steps,
                device=device,
                gamma=args.gamma,
            )
            closed_loop = _validate_closed_loop(
                actor,
                validation_envs,
                steps=args.closed_loop_steps,
                device=device,
            )
        finally:
            validation_envs.close()
        print(f"[joint] validation={validation} closed_loop={closed_loop}", flush=True)
        ckpt = _joint_checkpoint(
            actor,
            critic,
            opt_a,
            opt_c,
            args=args,
            step=args.steps,
            chunk_index=chunk_i,
            global_step=global_step,
            total_harvest=total_h,
            total_control=total_c,
            total_delivered=total_delivered,
            total_built=total_built,
            total_claims=total_claims,
            max_creeps=max_creeps,
            spawn_success=total_spawn,
            nll=last_nll,
            value_loss=last_vloss,
            explained_variance=last_ev,
            action_hist=corpus_hist,
            teacher_by_curriculum=teacher_by_curriculum,
            partial=False,
            wall_s=time.time() - t0,
        )
        qualification_failures = _qualification_failures(
            args,
            last_nll=last_nll,
            last_vloss=last_vloss,
            corpus_hist=corpus_hist,
            skill=skill,
            total_delivered=total_delivered,
            total_built=total_built,
            total_claims=total_claims,
            max_creeps=max_creeps,
            teacher_by_curriculum=teacher_by_curriculum,
            validation=validation,
            closed_loop=closed_loop,
        )
        qualified = not qualification_failures
        ckpt["meta"].update(validation)
        ckpt["meta"].update(closed_loop)
        ckpt["meta"]["qualified"] = qualified
        ckpt["meta"]["qualification_failures"] = qualification_failures
        atomic_torch_save(ckpt, args.save)
        print(f"[joint] saved → {args.save}", flush=True)
        print(f"[joint] meta={ckpt['meta']}", flush=True)
        if not qualified:
            print(
                "[joint] FAILED qualification: legality, fresh critic EV, economy "
                "coverage, closed-loop skill/population/claim, and finite losses required; "
                f"failed={qualification_failures}",
                flush=True,
            )
            return 1
        return 0
    finally:
        envs.close()
        if writer:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
