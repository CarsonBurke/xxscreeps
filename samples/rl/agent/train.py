#!/usr/bin/env python3
"""
Train ViT-PPO on xxscreeps with:
  · 8 parallel rollouts
  · one bounded adaptive-horizon extension for competent rollouts
  · decoupled GAE: critic λ=1.0 (MC), policy λ=0.97
  · discounted-return reward RMS + rollout-level advantage normalization
  · Flash-Attention via SDPA + bf16 autocast
  · CleanRL-style TensorBoard charts/losses
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_RL_ROOT = Path(__file__).resolve().parents[1]
_REPO = _RL_ROOT.parents[1]
for p in (_REPO, _RL_ROOT, str(_RL_ROOT)):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    from samples.rl.agent.artifacts import (
        artifact_meta,
        atomic_torch_save,
        load_critic_trunk,
        load_full_state,
        validate_artifact,
    )
    from samples.rl.agent.constants import PPO_CFG, SCHEMA
    from samples.rl.agent.env_client import stack_batches
    from samples.rl.agent.gae import decoupled_gae
    from samples.rl.agent.metrics_log import MetricsLog
    from samples.rl.agent.model import Agent, count_params
    from samples.rl.agent.ppo import PPOTrainer, RolloutBatch
    from samples.rl.agent.rollout_buffer import HostRolloutBuffer
    from samples.rl.agent.running_stats import RewardNormalizer
    from samples.rl.agent.vec_env import VecScreepsEnv, promote_obs_device
except ImportError:
    from agent.artifacts import (
        artifact_meta,
        atomic_torch_save,
        load_critic_trunk,
        load_full_state,
        validate_artifact,
    )
    from agent.constants import PPO_CFG, SCHEMA
    from agent.env_client import stack_batches
    from agent.gae import decoupled_gae
    from agent.metrics_log import MetricsLog
    from agent.model import Agent, count_params
    from agent.ppo import PPOTrainer, RolloutBatch
    from agent.rollout_buffer import HostRolloutBuffer
    from agent.running_stats import RewardNormalizer
    from agent.vec_env import VecScreepsEnv, promote_obs_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screeps ViT-PPO trainer")
    p.add_argument("--steps", type=int, default=int(PPO_CFG.get("rolloutSteps", 256)),
                   help="base rollout length per env (extended adaptively)")
    p.add_argument("--max-rollout-steps", type=int, default=int(PPO_CFG.get("maxRolloutSteps", 1024)))
    p.add_argument("--extend-steps", type=int, default=int(PPO_CFG.get("extendSteps", 256)))
    p.add_argument("--extend-rate", type=float, default=float(PPO_CFG.get("extendRateThreshold", 5.0)),
                   help="extend when (harvest+control)/tick > this")
    p.add_argument("--num-envs", type=int, default=int(PPO_CFG.get("numEnvs", 8)))
    p.add_argument("--updates", type=int, default=50)
    p.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help=(
            "soft wall-clock budget; finish the active update, save a complete "
            "checkpoint, then stop"
        ),
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--room", type=str, default="W7N3")
    p.add_argument("--max-episode", type=int, default=2000)
    p.add_argument("--lr", type=float, default=float(PPO_CFG["lr"]))
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument(
        "--minibatch",
        type=int,
        default=int(PPO_CFG.get("minibatch", 128)),
        help="transitions per optimizer step (default from schema; capped by steps×envs)",
    )
    p.add_argument("--save", type=Path, default=_RL_ROOT / "runs" / "policy.pt")
    p.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help="TB log dir (default: runs/tb-ppo/<timestamp> so runs do not merge)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--node", type=str, default=None)
    p.add_argument(
        "--no-reward-norm",
        action="store_true",
        help="disable CleanRL/Gymnasium discounted-return reward normalization",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="joint_pretrain.pt or PPO policy (actor+critic; opts if present)",
    )
    p.add_argument(
        "--headful",
        action="store_true",
        help="serve Screeps client on env 0 (http://127.0.0.1:21025/) to watch training",
    )
    p.add_argument("--headful-password", type=str, default="rlwatch")
    p.add_argument(
        "--tick-ms",
        type=int,
        default=None,
        help="delay on headful env ticks (default 100 if --headful); slows the whole batch",
    )
    p.add_argument("--no-open", action="store_true", help="do not auto-open browser for headful")
    p.add_argument(
        "--curriculum",
        type=str,
        default=None,
        help=(
            "comma-separated empty|seed_creep|seed_full|seed_claimer stages "
            "distributed across envs (default: empty)"
        ),
    )
    return p.parse_args()


def collect_rollout(
    envs: VecScreepsEnv,
    trainer: PPOTrainer,
    obs: dict[str, torch.Tensor],
    *,
    base_steps: int,
    max_steps: int,
    extend_steps: int,
    extend_rate: float,
    device: torch.device,
    ep_returns: list[float],
    ep_lengths: list[int],
    ep_trackers: list[dict],
    reward_normalizer: RewardNormalizer | None = None,
    buf: HostRolloutBuffer | None = None,
) -> tuple[RolloutBatch, dict[str, torch.Tensor], dict[str, float], int]:
    """
    Adaptive-length rollout across num_envs.
    Extends once by `extend_steps` when the base segment is competent, up to
    max_steps. Repeated extension in one collection used to run straight to the
    cap and turn a 1.27 GiB buffer into a multi-dozen-GiB allocation.

    Episode stats use **raw** rewards. GAE / value targets use CleanRL reward norm
    (discounted-return RMS) when `reward_normalizer` is set.

    Performance:
      · HostRolloutBuffer [T_max,N,...] — no list+stack alloc thrash
      · Host-side obs mirror (`envs.host_obs`) — no per-step full-obs D2H
      · Terminal V always B=N (compile-stable critic graph; unused slots masked)
    """
    n_env = envs.n
    if buf is None:
        buf = HostRolloutBuffer(max_steps, n_env)
    else:
        if buf.n != n_env:
            buf = HostRolloutBuffer(base_steps, n_env)
        else:
            buf.reset()
            buf.ensure_capacity(base_steps)

    target = base_steps
    total_harvest = 0.0
    total_control = 0.0
    total_delivered = 0.0
    total_built = 0.0
    total_claims = 0.0
    total_intent_issued = 0
    total_intent_invalid = 0
    intent_totals: dict[str, dict[str, int]] = {}
    stage_totals: dict[str, dict[str, float]] = {}
    max_creeps = 0
    overflow_steps = 0
    extensions = 0
    # Scratch for compile-stable terminal V: always N slots.
    term_mask = torch.zeros(n_env, dtype=torch.bool)
    term_slots: list[dict[str, torch.Tensor] | None] = [None] * n_env

    while buf.t < target:
        with torch.no_grad():
            out = trainer.act(obs)

        if envs.host_obs is not None:
            host = envs.host_obs
        else:
            host = {k: v.detach().cpu() for k, v in obs.items() if not k.startswith("_")}

        next_obs, reward, done, infos = envs.step(
            {
                "types": out.types,
                "dirs": out.dirs,
                "targets": out.targets,
                "amounts": out.amounts,
            }
        )

        trunc_row = torch.zeros(n_env, dtype=torch.float32)
        term_mask.zero_()
        for i in range(n_env):
            term_slots[i] = None
        reward_host = reward.detach().cpu()
        for i, info in enumerate(infos):
            h = float(info.get("harvestDelta") or 0.0)
            c = float(info.get("controlDelta") or 0.0)
            total_harvest += h
            total_control += c
            total_delivered += float(info.get("transferDelta") or 0.0)
            total_built += float(info.get("buildDelta") or 0.0)
            total_claims += float(info.get("claimDelta") or 0.0)
            total_intent_issued += int(info.get("intentIssued") or 0)
            total_intent_invalid += int(info.get("intentInvalid") or 0)
            stage = str(info.get("curriculum") or "unknown")
            stage_row = stage_totals.setdefault(
                stage,
                {
                    "transitions": 0.0,
                    "skill": 0.0,
                    "reward": 0.0,
                    "invalid": 0.0,
                    "issued": 0.0,
                    "max_creeps": 0.0,
                },
            )
            stage_row["transitions"] += 1.0
            stage_row["skill"] += h + c
            stage_row["reward"] += float(reward_host[i].item())
            stage_row["invalid"] += float(info.get("intentInvalid") or 0)
            stage_row["issued"] += float(info.get("intentIssued") or 0)
            stage_row["max_creeps"] = max(stage_row["max_creeps"], float(info.get("creeps") or 0))
            for name, counts in (info.get("intentByType") or {}).items():
                row = intent_totals.setdefault(str(name), {"issued": 0, "invalid": 0})
                row["issued"] += int((counts or {}).get("issued") or 0)
                row["invalid"] += int((counts or {}).get("invalid") or 0)
            max_creeps = max(max_creeps, int(info.get("creeps") or 0))
            g = info.get("globals") or {}
            if g.get("roomOverflow") or g.get("actorOverflow") or g.get("targetOverflow"):
                overflow_steps += 1
            ep_trackers[i]["ret"] += float(reward_host[i].item())
            ep_trackers[i]["len"] += 1
            ep_trackers[i]["harvest"] += h
            ep_trackers[i]["control"] += c
            if info.get("episode_done"):
                ep_returns.append(ep_trackers[i]["ret"])
                ep_lengths.append(ep_trackers[i]["len"])
                ep_trackers[i] = {"ret": 0.0, "len": 0, "harvest": 0.0, "control": 0.0}
            if info.get("truncated") or info.get("episode_done"):
                trunc_row[i] = 1.0
            term_obs = info.get("terminal_observation")
            if term_obs is not None:
                term_mask[i] = True
                term_slots[i] = {
                    k: (v if v.dim() > 0 and v.shape[0] == 1 else v.unsqueeze(0))
                    for k, v in term_obs.items()
                    if torch.is_tensor(v)
                }

        t_idx = buf.write_step(
            host_obs=host if not any(k.startswith("_") for k in host) else {
                k: v for k, v in host.items() if not str(k).startswith("_")
            },
            types=out.types,
            dirs=out.dirs,
            targets=out.targets,
            amounts=out.amounts,
            logprob=out.actor_logprob,
            value=out.value,
            reward=reward,
            done=done,
            trunc=trunc_row,
        )

        # Terminal V: always critic B=N (pad missing with zeros) — no variable B recompile.
        if bool(term_mask.any()):
            batches = []
            # Build N-length list: real term obs or zeros from first available template
            template = next((s for s in term_slots if s is not None), None)
            assert template is not None
            for i in range(n_env):
                if term_slots[i] is not None:
                    batches.append(term_slots[i])
                else:
                    batches.append({k: torch.zeros_like(v) for k, v in template.items()})
            with torch.no_grad():
                stacked = promote_obs_device(stack_batches(batches), device)
                vals = trainer.value_only(stacked)  # [N] fixed
            # Only mark real terminals
            buf.set_term_values(t_idx, vals.cpu(), term_mask)

        obs = next_obs

        if buf.t >= target and target < max_steps and extensions == 0:
            rate = (total_harvest + total_control) / max(1, buf.t * n_env)
            if rate > extend_rate:
                target = min(max_steps, target + extend_steps)
                buf.ensure_capacity(target)
                extensions += 1

    T = buf.t
    N = n_env
    rewards_raw = buf.tn("reward").to(device)
    values_tn = buf.tn("value").to(device)
    dones_tn = buf.tn("done").to(device)
    logprob_tn = buf.tn("logprob").to(device)
    trunc_tn = buf.tn("trunc").to(device=device, dtype=rewards_raw.dtype)

    if reward_normalizer is not None:
        rewards_tn = reward_normalizer.normalize(rewards_raw, dones_tn)
    else:
        rewards_tn = rewards_raw

    with torch.no_grad():
        bootstrap = trainer.value_only(obs)  # [N]
    next_values_tn = torch.zeros(T, N, device=device, dtype=values_tn.dtype)
    if T > 1:
        next_values_tn[:-1] = values_tn[1:]
    next_values_tn[-1] = bootstrap
    # Splice terminal V where has_term
    has_term = buf.has_term[:T].to(device)
    term_v = buf.term_value[:T].to(device=device, dtype=values_tn.dtype)
    trunc_b = trunc_tn > 0
    splice = has_term & trunc_b
    next_values_tn = torch.where(splice, term_v, next_values_tn)

    adv, ret, gae_info = decoupled_gae(
        rewards_tn,
        values_tn,
        dones_tn,
        gamma=float(PPO_CFG["gamma"]),
        policy_lambda=float(PPO_CFG.get("gaeLambda", 0.95)),
        next_value=bootstrap,
        truncations=trunc_tn,
        next_values_tn=next_values_tn,
    )

    obs_flat = buf.as_flat_obs()
    types_tn = buf.tn("types")
    dirs_tn = buf.tn("dirs")
    tgts_tn = buf.tn("targets")
    amts_tn = buf.tn("amounts")

    rollout = RolloutBatch(
        obs=obs_flat,
        actions={
            "types": types_tn.reshape(T * N, *types_tn.shape[2:]),
            "dirs": dirs_tn.reshape(T * N, *dirs_tn.shape[2:]),
            "targets": tgts_tn.reshape(T * N, *tgts_tn.shape[2:]),
            "amounts": amts_tn.reshape(T * N, *amts_tn.shape[2:]),
        },
        logprob=logprob_tn.reshape(T * N, *logprob_tn.shape[2:]).cpu(),
        value=values_tn.reshape(T * N).cpu(),
        reward=rewards_tn.reshape(T * N).cpu(),
        done=dones_tn.reshape(T * N).cpu(),
        advantage=adv.reshape(T * N).cpu(),
        ret=ret.reshape(T * N).cpu(),
        batch_size=T * N,
    )

    rate = (total_harvest + total_control) / max(1, T * N)
    meta = {
        "rollout_steps": float(T),
        "rollout_env_steps": float(T * N),
        "extensions": float(extensions),
        "skill_rate": float(rate),
        "total_harvest": float(total_harvest),
        "total_control": float(total_control),
        "total_delivered": float(total_delivered),
        "total_built": float(total_built),
        "total_claims": float(total_claims),
        "intent_issued": float(total_intent_issued),
        "intent_invalid": float(total_intent_invalid),
        "intent_invalid_fraction": float(
            total_intent_invalid / max(1, total_intent_issued)
        ),
        "max_creeps": float(max_creeps),
        "overflow_step_fraction": float(overflow_steps / max(1, T * N)),
        "mean_reward": float(rewards_raw.mean().item()),
        "mean_reward_norm": float(rewards_tn.mean().item()),
        "ret_mean": float(ret.mean().item()),
        "ret_std": float(ret.std().item()),
        "v_mean": float(values_tn.mean().item()),
        "v_std": float(values_tn.std().item()),
        **gae_info,
    }
    for name, counts in intent_totals.items():
        safe_name = "".join(char if char.isalnum() else "_" for char in name)
        meta[f"intent_{safe_name}_issued"] = float(counts["issued"])
        meta[f"intent_{safe_name}_invalid"] = float(counts["invalid"])
    for name, totals in stage_totals.items():
        safe_name = "".join(char if char.isalnum() else "_" for char in name)
        prefix = f"stage_{safe_name}"
        transitions = max(1.0, totals["transitions"])
        meta[f"{prefix}_skill_rate"] = totals["skill"] / transitions
        meta[f"{prefix}_mean_reward"] = totals["reward"] / transitions
        meta[f"{prefix}_invalid_fraction"] = (
            totals["invalid"] / max(1.0, totals["issued"])
        )
        meta[f"{prefix}_max_creeps"] = totals["max_creeps"]
    if reward_normalizer is not None:
        meta.update(reward_normalizer.stats())
    return rollout, obs, meta, T


def _ppo_checkpoint(
    trainer: PPOTrainer,
    *,
    update: int,
    global_step: int,
    minibatch: int,
    args: argparse.Namespace,
    reward_normalizer: RewardNormalizer | None,
) -> dict:
    payload = {
        "actor": trainer.actor.state_dict(),
        "critic": trainer.critic.state_dict(),
        "actor_opt": trainer.actor_opt.state_dict(),
        "critic_opt": trainer.critic_opt.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "update": int(update),
        "global_step": int(global_step),
        "minibatch": int(minibatch),
        "args": vars(args),
        "meta": artifact_meta(
            "ppo",
            trainer.actor,
            trainer.critic,
            reward=SCHEMA["reward"],
            normalize_reward=reward_normalizer is not None,
        ),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if reward_normalizer is not None:
        payload["reward_normalizer"] = reward_normalizer.state_dict()
    return payload


def main() -> int:
    args = parse_args()
    if args.updates < 1:
        raise SystemExit("[train] --updates must be at least 1")
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        raise SystemExit("[train] --max-wall-seconds must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    if args.logdir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.logdir = _RL_ROOT / "runs" / "tb-ppo" / stamp
    args.logdir.mkdir(parents=True, exist_ok=True)

    # Throughput knobs for compiled rollout + large PPO updates
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(args.logdir))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join(f"|{k}|{v}|" for k, v in sorted(vars(args).items(), key=lambda x: x[0]))),
        )
        writer.flush()
        print(f"[train] tensorboard logdir={args.logdir}", flush=True)
    except Exception as err:  # noqa: BLE001
        print(f"[train] TB disabled: {err}", flush=True)

    if args.headful:
        print(
            f"[train] headful ON env0 → http://127.0.0.1:21025/ "
            f"(login Player 1 / {args.headful_password} → {args.room}); "
            f"tick delay slows all envs to the slowest",
            flush=True,
        )
    agent = Agent()
    # Eager params on device first; torch.compile wraps in PPOTrainer
    agent.to(device)
    resume_update = 0
    resume_global_step = 0
    resume_ckpt: dict | None = None
    resume_kind: str | None = None
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if not isinstance(resume_ckpt, dict) or "actor" not in resume_ckpt or "critic" not in resume_ckpt:
            raise SystemExit(
                f"[train] checkpoint {args.resume} is not a complete schema-v2 artifact"
            )
        try:
            meta = validate_artifact(
                resume_ckpt, agent.actor, agent.critic, kinds=("joint_pretrain", "ppo"),
            )
        except ValueError as error:
            raise SystemExit(f"[train] incompatible checkpoint: {error}") from error
        resume_kind = str(meta["kind"])
        load_full_state(agent.actor, resume_ckpt["actor"], name="actor")
        if resume_kind == "joint_pretrain":
            if not bool(meta.get("qualified")):
                raise SystemExit(
                    "[train] joint pretrain is an unqualified snapshot; pass all validation gates first"
                )
            load_critic_trunk(agent.critic, resume_ckpt["critic"])
            print(
                f"[train] initialized from qualified joint pretrain "
                f"curriculum={meta.get('curriculum')!r} skill={meta.get('skill_rate')}",
                flush=True,
            )
        else:
            requested_norm = bool(PPO_CFG.get("normalizeReward", True)) and not args.no_reward_norm
            if bool(meta.get("normalize_reward")) != requested_norm:
                raise SystemExit(
                    "[train] PPO reward-normalization mode differs from checkpoint; "
                    "resume with the original mode"
                )
            load_full_state(agent.critic, resume_ckpt["critic"], name="critic")
            resume_update = int(resume_ckpt["update"]) + 1
            resume_global_step = int(resume_ckpt["global_step"])
            print(
                f"[train] resuming PPO update={resume_ckpt['update']} "
                f"global_step={resume_global_step}",
                flush=True,
            )
    print(
        f"[train] spawning {args.num_envs} envs (parallel ThreadPool step/reset) …",
        flush=True,
    )
    envs = VecScreepsEnv(
        args.num_envs,
        node=args.node,
        room=args.room,
        max_episode=args.max_episode,
        device=device,
        headful=args.headful,
        headful_password=args.headful_password,
        tick_ms=args.tick_ms if args.tick_ms is not None else (100 if args.headful else None),
        no_open=args.no_open,
        curriculum=args.curriculum,
    )
    print(
        f"[train] device={device} envs={args.num_envs} headful={args.headful} "
        f"actor_params={count_params(agent.actor):,} critic_params={count_params(agent.critic):,} "
        f"(separate models) base_steps={args.steps} max_rollout={args.max_rollout_steps} "
        f"compile_reduce_overhead={not args.no_compile} "
        f"bf16={not args.no_bf16 and device.type=='cuda'}",
        flush=True,
    )

    obs0 = envs.reset()
    if args.headful:
        info0 = envs.envs[0].last_info or {}
        url = info0.get("headfulUrl")
        if url:
            print(f"[train] Screeps client: {url}", flush=True)
        else:
            print("[train] headful requested — check Node stderr if client did not bind", flush=True)
    # Minibatch = transitions per optimizer step (not "# of minibatches").
    # The entity transformer's activation cost makes the schema default (128)
    # intentionally much smaller than the old room-only model's minibatch.
    # Always capped by actual rollout size B = T×N in PPOTrainer.update.
    mb = max(1, int(args.minibatch))
    print(
        f"[train] minibatch_size={mb} transitions/step  "
        f"(rollout base B≈{args.steps * args.num_envs}; mb capped to B) "
        f"policy_clip=[{1.0 - float(PPO_CFG['clip']):.2f}, {1.0 + float(PPO_CFG.get('clipHigh', PPO_CFG['clip'])):.2f}]",
        flush=True,
    )

    trainer = PPOTrainer(
        agent,
        lr=args.lr,
        device=device,
        compile_model=not args.no_compile,
        use_bf16=not args.no_bf16,
        minibatch=mb,
    )
    # Restore optimizers AFTER trainer construction (E8 resume contract).
    if resume_kind == "ppo":
        if "actor_opt" not in resume_ckpt or "critic_opt" not in resume_ckpt:
            raise SystemExit("[train] PPO resume requires both optimizer states")
        try:
            trainer.actor_opt.load_state_dict(resume_ckpt["actor_opt"])
            trainer.critic_opt.load_state_dict(resume_ckpt["critic_opt"])
        except Exception as error:  # noqa: BLE001
            raise SystemExit(f"[train] optimizer state is incompatible: {error}") from error
        print("[train] restored AdamW optimizer states", flush=True)
    # Capture CUDA graphs for reduce-overhead act before timed rollouts
    if not args.no_compile and device.type == "cuda":
        trainer.warmup(obs0, steps=5)

    # Warmup and model construction consume RNG. Restore continuation state only
    # after both so PPO resumes with the same sampling/minibatch streams.
    if resume_kind == "ppo":
        required_rng = ("torch_rng_state", "numpy_rng_state")
        missing_rng = [key for key in required_rng if key not in resume_ckpt]
        if device.type == "cuda" and "cuda_rng_state_all" not in resume_ckpt:
            missing_rng.append("cuda_rng_state_all")
        if missing_rng:
            raise SystemExit(
                "[train] PPO resume lacks exact RNG state: " + ", ".join(missing_rng)
            )
        torch.set_rng_state(resume_ckpt["torch_rng_state"].cpu())
        np.random.set_state(resume_ckpt["numpy_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_ckpt["cuda_rng_state_all"]]
            )
        print("[train] restored CPU/CUDA/NumPy RNG states", flush=True)

    global_step = resume_global_step
    start_time = time.time()
    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    ep_trackers = [{"ret": 0.0, "len": 0, "harvest": 0.0, "control": 0.0} for _ in range(args.num_envs)]

    use_reward_norm = bool(PPO_CFG.get("normalizeReward", True)) and not args.no_reward_norm
    reward_normalizer: RewardNormalizer | None = None
    if use_reward_norm:
        clip = PPO_CFG.get("rewardNormClip", 10.0)
        reward_normalizer = RewardNormalizer(
            gamma=float(PPO_CFG["gamma"]),
            clip=float(clip) if clip is not None else None,
        )
        if resume_kind == "ppo":
            if "reward_normalizer" not in resume_ckpt:
                raise SystemExit("[train] normalized PPO resume lacks reward normalizer state")
            reward_normalizer.load_state_dict(
                resume_ckpt["reward_normalizer"], restore_returns=False,
            )
            print(
                f"[train] restored reward RMS with fresh per-env traces "
                f"std={reward_normalizer.stats().get('reward_rms_std', 0):.4f}",
                flush=True,
            )
        print(
            f"[train] CleanRL reward norm ON (discounted-return RMS, clip={clip})",
            flush=True,
        )
    else:
        print("[train] reward norm OFF", flush=True)

    # Host-side metrics (polars parquet when available) — never on act hot path
    metrics: MetricsLog | None = MetricsLog(args.logdir / "metrics.jsonl", flush_every=5)

    # Allocate the normal horizon and grow only when competence actually triggers
    # adaptive extension. Preallocating the old 20k cap consumed tens of GiB.
    rollout_buf = HostRolloutBuffer(args.steps, args.num_envs)
    print(
        f"[train] host rollout buffer T_initial={args.steps} N={args.num_envs} "
        f"(lazy growth to cap={args.max_rollout_steps})",
        flush=True,
    )

    value_warmup = int(PPO_CFG.get("valueWarmupUpdates", 0))
    # Skip warmup when resuming past the warmup window (avoid re-freezing a trained actor).
    if resume_update >= value_warmup:
        value_warmup = 0
    if value_warmup > 0:
        print(
            f"[train] value warmup: first {value_warmup} updates critic-only (actor frozen)",
            flush=True,
        )

    try:
        obs = obs0
        completed_update = resume_update - 1
        for update in range(resume_update, resume_update + args.updates):
            t0 = time.time()
            # fresh episode lists this update (cleanrl logs completed eps)
            finished_returns: list[float] = []
            finished_lengths: list[int] = []
            # wrap trackers: capture finished into finished_* via shared lists
            # (collect_rollout appends to ep_returns/ep_lengths)
            n_before_r = len(ep_returns)
            n_before_l = len(ep_lengths)

            rollout, obs, meta, T = collect_rollout(
                envs,
                trainer,
                obs,
                base_steps=args.steps,
                max_steps=args.max_rollout_steps,
                extend_steps=args.extend_steps,
                extend_rate=args.extend_rate,
                device=device,
                ep_returns=ep_returns,
                ep_lengths=ep_lengths,
                ep_trackers=ep_trackers,
                reward_normalizer=reward_normalizer,
                buf=rollout_buf,
            )
            finished_returns = ep_returns[n_before_r:]
            finished_lengths = ep_lengths[n_before_l:]

            global_step += int(meta["rollout_env_steps"])
            # VAPO value warmup: critic-only for first N absolute updates.
            in_warmup = value_warmup > 0 and update < value_warmup
            stats = trainer.update(rollout, critic_only=in_warmup)
            if in_warmup:
                stats["value_warmup"] = 1.0
            # VRAM stats without empty_cache thrash (red-team: empty_cache every update kills SPS).
            if device.type == "cuda":
                if (update + 1) % 20 == 0:
                    torch.cuda.empty_cache()
                free_b, total_b = torch.cuda.mem_get_info()
                stats["vram_alloc_gb"] = torch.cuda.memory_allocated() / (1024**3)
                stats["vram_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
                stats["vram_free_gb"] = free_b / (1024**3)
            dt = time.time() - t0
            sps = int(meta["rollout_env_steps"] / max(dt, 1e-6))

            # True episodic stats = completed episodes only.
            # Never use in-progress tracker len/ret as "episodic_*" — that climbs
            # 512,1024,1536… across updates until max_episode and looks like
            # extension (ext) even when skill ≪ extend_rate.
            ongoing_ret = float(np.mean([t["ret"] for t in ep_trackers]))
            ongoing_len = float(np.mean([t["len"] for t in ep_trackers]))
            if finished_returns:
                mean_ep_ret = float(np.mean(finished_returns))
                mean_ep_len = float(np.mean(finished_lengths))
            elif ep_returns:
                # Hold last completed so TB has a point every update (same cadence
                # as mean_step_reward) without faking unfinished episodes.
                mean_ep_ret = float(ep_returns[-1])
                mean_ep_len = float(ep_lengths[-1]) if ep_lengths else float("nan")
            else:
                mean_ep_ret = float("nan")
                mean_ep_len = float("nan")

            vram_s = ""
            if "vram_free_gb" in stats:
                vram_s = (
                    f" vram_free={stats['vram_free_gb']:.1f}G"
                    f"/alloc={stats['vram_alloc_gb']:.1f}G"
                )
            ep_s = (
                f"ep_ret={mean_ep_ret:.3f} ep_len={mean_ep_len:.1f}"
                if finished_returns or ep_returns
                else f"ongoing_ret={ongoing_ret:.3f} ongoing_len={ongoing_len:.1f}"
            )
            wu = " WARMUP" if in_warmup else ""
            kl_stop = " KL-stop" if stats.get("early_stop", 0) else ""
            print(
                f"[update {update+1}]{wu}{kl_stop} "
                f"T={int(meta['rollout_steps'])} mb={int(stats.get('minibatch', mb))} "
                f"ext={int(meta['extensions'])} "
                f"skill={meta['skill_rate']:.2f}e/t mean_r={meta['mean_reward']:.4f} "
                f"{ep_s} "
                f"pi={stats['policy_loss']:.4f} v={stats['value_loss']:.4f} "
                f"H={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
                f"gnA={stats['grad_norm_actor']:.3f} gnC={stats['grad_norm_critic']:.3f} "
                f"sps={sps} {dt:.1f}s{vram_s}",
                flush=True,
            )

            if writer:
                # CleanRL-style charts/ — same global_step cadence for reward & return
                writer.add_scalar("charts/learning_rate", trainer.actor_opt.param_groups[0]["lr"], global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/rollout_length", meta["rollout_steps"], global_step)
                writer.add_scalar("charts/rollout_env_steps", meta["rollout_env_steps"], global_step)
                writer.add_scalar("charts/rollout_extensions", meta["extensions"], global_step)
                writer.add_scalar("charts/skill_rate_et", meta["skill_rate"], global_step)
                writer.add_scalar("charts/mean_step_reward", meta["mean_reward"], global_step)
                if "mean_reward_norm" in meta:
                    writer.add_scalar("charts/mean_step_reward_norm", meta["mean_reward_norm"], global_step)
                if "reward_rms_std" in meta:
                    writer.add_scalar("charts/reward_rms_std", meta["reward_rms_std"], global_step)
                    writer.add_scalar("charts/ret_mean", meta.get("ret_mean", 0.0), global_step)
                    writer.add_scalar("charts/ret_std", meta.get("ret_std", 0.0), global_step)
                    writer.add_scalar("charts/v_mean", meta.get("v_mean", 0.0), global_step)
                    writer.add_scalar("charts/v_std", meta.get("v_std", 0.0), global_step)
                writer.add_scalar("charts/total_harvest", meta["total_harvest"], global_step)
                writer.add_scalar("charts/total_control", meta["total_control"], global_step)
                writer.add_scalar("charts/total_delivered", meta["total_delivered"], global_step)
                writer.add_scalar("charts/total_built", meta["total_built"], global_step)
                writer.add_scalar("charts/total_claims", meta["total_claims"], global_step)
                writer.add_scalar(
                    "charts/intent_invalid_fraction", meta["intent_invalid_fraction"], global_step,
                )
                for key, value in meta.items():
                    if key.startswith("intent_") and key.endswith(("_issued", "_invalid")):
                        writer.add_scalar(f"actions/{key}", value, global_step)
                    elif key.startswith("stage_"):
                        writer.add_scalar(f"stages/{key}", value, global_step)
                writer.add_scalar("charts/max_creeps", meta["max_creeps"], global_step)
                writer.add_scalar("charts/overflow_step_fraction", meta["overflow_step_fraction"], global_step)
                # Always: unfinished episode progress (not true episodic length)
                writer.add_scalar("charts/ongoing_return", ongoing_ret, global_step)
                writer.add_scalar("charts/ongoing_length", ongoing_len, global_step)
                # True episodic_* only after ≥1 completed ep; then every update (hold-last)
                if finished_returns or ep_returns:
                    writer.add_scalar("charts/episodic_return", mean_ep_ret, global_step)
                    writer.add_scalar("charts/episodic_length", mean_ep_len, global_step)
                    writer.add_scalar(
                        "charts/episodic_return_avg100",
                        float(np.mean(ep_returns[-100:])),
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_length_avg100",
                        float(np.mean(ep_lengths[-100:])),
                        global_step,
                    )
                if finished_returns:
                    for er, el in zip(finished_returns, finished_lengths):
                        writer.add_scalar("charts/episodic_return_samples", er, global_step)
                        writer.add_scalar("charts/episodic_length_samples", el, global_step)

                # losses/ (cleanrl names)
                writer.add_scalar("losses/value_loss", stats["value_loss"], global_step)
                writer.add_scalar("losses/policy_loss", stats["policy_loss"], global_step)
                writer.add_scalar("losses/entropy", stats["entropy"], global_step)
                writer.add_scalar("losses/old_approx_kl", stats["old_approx_kl"], global_step)
                writer.add_scalar("losses/approx_kl", stats["approx_kl"], global_step)
                writer.add_scalar("losses/clipfrac", stats["clipfrac"], global_step)
                writer.add_scalar("losses/explained_variance", stats["explained_variance"], global_step)
                writer.add_scalar("losses/grad_norm", stats["grad_norm"], global_step)
                writer.add_scalar("losses/grad_norm_actor", stats["grad_norm_actor"], global_step)
                writer.add_scalar("losses/grad_norm_critic", stats["grad_norm_critic"], global_step)
                writer.add_scalar("charts/minibatch", stats.get("minibatch", mb), global_step)
                writer.add_scalar("charts/learning_rate_actor", trainer.actor_opt.param_groups[0]["lr"], global_step)
                writer.add_scalar("charts/learning_rate_critic", trainer.critic_opt.param_groups[0]["lr"], global_step)

                # VAPO GAE diagnostics
                writer.add_scalar("gae/lambda_policy_mean", meta["lambda_policy_mean"], global_step)
                writer.add_scalar("gae/lambda_policy_min", meta["lambda_policy_min"], global_step)
                writer.add_scalar("gae/lambda_policy_max", meta["lambda_policy_max"], global_step)
                writer.add_scalar("gae/lambda_critic", meta["lambda_critic"], global_step)
                # Flush every 5 updates — per-update flush stalls the train loop
                if (update + 1) % 5 == 0:
                    writer.flush()

            metrics.add(
                update=update,
                global_step=global_step,
                skill_rate=meta["skill_rate"],
                mean_reward=meta["mean_reward"],
                mean_reward_norm=meta.get("mean_reward_norm", float("nan")),
                policy_loss=stats["policy_loss"],
                value_loss=stats["value_loss"],
                entropy=stats["entropy"],
                approx_kl=stats["approx_kl"],
                sps=sps,
                rollout_steps=meta["rollout_steps"],
                extensions=meta["extensions"],
                delivered=meta["total_delivered"],
                built=meta["total_built"],
                claims=meta["total_claims"],
                intent_issued=meta["intent_issued"],
                intent_invalid=meta["intent_invalid"],
                intent_invalid_fraction=meta["intent_invalid_fraction"],
                max_creeps=meta["max_creeps"],
                overflow_step_fraction=meta["overflow_step_fraction"],
                ep_ret=mean_ep_ret if mean_ep_ret == mean_ep_ret else None,  # NaN → None
                early_stop=float(stats.get("early_stop", 0)),
                **{key: value for key, value in meta.items() if key.startswith("stage_")},
            )

            if (update + 1) % 5 == 0:
                ckpt = _ppo_checkpoint(
                    trainer,
                    update=update,
                    global_step=global_step,
                    minibatch=mb,
                    args=args,
                    reward_normalizer=reward_normalizer,
                )
                atomic_torch_save(ckpt, args.save)
                print(f"[train] saved {args.save}", flush=True)

            completed_update = update
            elapsed = time.time() - start_time
            if args.max_wall_seconds is not None and elapsed >= args.max_wall_seconds:
                print(
                    f"[train] wall budget reached after update {update + 1}: "
                    f"{elapsed:.1f}s >= {args.max_wall_seconds:.1f}s",
                    flush=True,
                )
                break

        final = _ppo_checkpoint(
            trainer,
            update=completed_update,
            global_step=global_step,
            minibatch=mb,
            args=args,
            reward_normalizer=reward_normalizer,
        )
        atomic_torch_save(final, args.save)
        elapsed = time.time() - start_time
        print(f"[train] done → {args.save}  global_step={global_step} elapsed={elapsed:.1f}s", flush=True)
        return 0
    finally:
        if metrics is not None:
            try:
                metrics.close()
            except Exception:
                pass
        envs.close()
        if writer:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
