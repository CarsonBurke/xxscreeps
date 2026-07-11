#!/usr/bin/env python3
"""
Train ViT-PPO on xxscreeps with:
  · 8 parallel rollouts
  · adaptive horizon (+500 when harvest+control > 5 e/t, cap 20k)
  · VAPO decoupled GAE: critic λ=1.0 (MC), policy length-adaptive λ
  · CleanRL reward norm (discounted-return RMS) + per-mb adv norm
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
    from samples.rl.agent.constants import PPO_CFG
    from samples.rl.agent.gae import decoupled_gae
    from samples.rl.agent.model import Agent, count_params
    from samples.rl.agent.ppo import PPOTrainer, RolloutBatch
    from samples.rl.agent.running_stats import RewardNormalizer
    from samples.rl.agent.vec_env import VecScreepsEnv
except ImportError:
    from agent.constants import PPO_CFG
    from agent.gae import decoupled_gae
    from agent.model import Agent, count_params
    from agent.ppo import PPOTrainer, RolloutBatch
    from agent.running_stats import RewardNormalizer
    from agent.vec_env import VecScreepsEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screeps ViT-PPO trainer")
    p.add_argument("--steps", type=int, default=int(PPO_CFG.get("rolloutSteps", 256)),
                   help="base rollout length per env (extended adaptively)")
    p.add_argument("--max-rollout-steps", type=int, default=int(PPO_CFG.get("maxRolloutSteps", 20000)))
    p.add_argument("--extend-steps", type=int, default=int(PPO_CFG.get("extendSteps", 500)))
    p.add_argument("--extend-rate", type=float, default=float(PPO_CFG.get("extendRateThreshold", 5.0)),
                   help="extend when (harvest+control)/tick > this")
    p.add_argument("--num-envs", type=int, default=int(PPO_CFG.get("numEnvs", 8)))
    p.add_argument("--updates", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--room", type=str, default="W7N3")
    p.add_argument("--max-episode", type=int, default=2000)
    p.add_argument("--lr", type=float, default=float(PPO_CFG["lr"]))
    p.add_argument("--lambda-alpha", type=float, default=float(PPO_CFG.get("lambdaAlpha", 0.05)))
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument(
        "--minibatch",
        type=int,
        default=int(PPO_CFG.get("minibatch", 2048)),
        help="transitions per optimizer step (default 2048; capped by steps×envs)",
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
        "--critic-pretrain",
        type=Path,
        default=None,
        help="load critic weights from pretrain_critic.py (VAPO value pretraining)",
    )
    p.add_argument(
        "--no-reward-norm",
        action="store_true",
        help="disable CleanRL/Gymnasium discounted-return reward normalization",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="load actor+critic (+opts if present) from a policy checkpoint and continue",
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
    return p.parse_args()


def _obs_cpu(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in obs.items() if not k.startswith("_")}


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
) -> tuple[RolloutBatch, dict[str, torch.Tensor], dict[str, float], int]:
    """
    Adaptive-length rollout across num_envs.
    Extends by `extend_steps` when about to finish if mean (harvest+control)/tick > extend_rate,
    up to max_steps.

    Episode stats use **raw** rewards. GAE / value targets use CleanRL reward norm
    (discounted-return RMS) when `reward_normalizer` is set.
    """
    target = base_steps
    obs_buf: list[dict[str, torch.Tensor]] = []
    act_types, act_dirs, act_tgts, act_amts = [], [], [], []
    logprobs, values, rewards, dones = [], [], [], []

    total_harvest = 0.0
    total_control = 0.0
    t = 0
    extensions = 0

    while t < target:
        with torch.no_grad():
            out = trainer.act(obs)

        obs_buf.append(_obs_cpu(obs))
        act_types.append(out.types.detach().cpu())
        act_dirs.append(out.dirs.detach().cpu())
        act_tgts.append(out.targets.detach().cpu())
        act_amts.append(out.amounts.detach().cpu())
        logprobs.append(out.logprob.detach().cpu())
        values.append(out.value.detach().cpu())

        next_obs, reward, done, infos = envs.step(
            {
                "types": out.types,
                "dirs": out.dirs,
                "targets": out.targets,
                "amounts": out.amounts,
            }
        )
        rewards.append(reward.detach().cpu())
        dones.append(done.detach().cpu())

        for i, info in enumerate(infos):
            h = float(info.get("harvestDelta") or 0.0)
            c = float(info.get("controlDelta") or 0.0)
            total_harvest += h
            total_control += c
            # Raw reward for human-readable episodic return
            ep_trackers[i]["ret"] += float(reward[i].item())
            ep_trackers[i]["len"] += 1
            ep_trackers[i]["harvest"] += h
            ep_trackers[i]["control"] += c
            if info.get("episode_done"):
                ep_returns.append(ep_trackers[i]["ret"])
                ep_lengths.append(ep_trackers[i]["len"])
                ep_trackers[i] = {"ret": 0.0, "len": 0, "harvest": 0.0, "control": 0.0}
                if reward_normalizer is not None:
                    reward_normalizer.reset_env(i)

        obs = next_obs
        t += 1

        # About to finish this target horizon → maybe extend
        if t >= target and target < max_steps:
            rate = (total_harvest + total_control) / max(1, t)
            if rate > extend_rate:
                target = min(max_steps, target + extend_steps)
                extensions += 1

    # Stack [T, N, ...]
    def stack_tn(xs: list[torch.Tensor]) -> torch.Tensor:
        # each x is [N] or [N, ...]
        return torch.stack(xs, dim=0)

    rewards_raw = stack_tn(rewards).to(device)          # T,N  (logging / episode stats)
    values_tn = stack_tn(values).to(device)
    dones_tn = stack_tn(dones).to(device)
    logprob_tn = stack_tn(logprobs).to(device)
    types_tn = stack_tn(act_types).to(device)
    dirs_tn = stack_tn(act_dirs).to(device)
    tgts_tn = stack_tn(act_tgts).to(device)
    amts_tn = stack_tn(act_amts).to(device)

    T, N = rewards_raw.shape
    # CleanRL / Gymnasium NormalizeReward: r ← r / √Var(discounted return)
    # so GAE advantages + critic returns stay O(1). Values are learned in this space.
    if reward_normalizer is not None:
        rewards_tn = reward_normalizer.normalize(rewards_raw, dones_tn)
    else:
        rewards_tn = rewards_raw

    # Bootstrap value for truncated envs (same units as rewards_tn / critic)
    with torch.no_grad():
        bootstrap = trainer.act(obs).value.float()  # [N]
        # zero bootstrap where last step was terminal
        bootstrap = bootstrap * (1.0 - dones_tn[-1])

    # Prefer actual collected T as sequence length (VAPO uses response length)
    lengths = torch.full((N,), float(T), device=device)

    adv, ret, adv_mc, gae_info = decoupled_gae(
        rewards_tn,
        values_tn,
        dones_tn,
        gamma=float(PPO_CFG["gamma"]),
        alpha=float(PPO_CFG.get("lambdaAlpha", 0.05)),
        next_value=bootstrap,
        episode_lengths=lengths,
    )

    keys = [k for k in obs_buf[0] if not k.startswith("_")]
    # obs_buf items are [N,...] each step → stack to [T,N,...] then flatten.
    # Keep full rollout on CPU; PPOTrainer.update streams minibatches to GPU.
    # Full B=2048 patches alone is ~2.3GiB and competed with dual-trunk activations.
    obs_tn = {k: torch.stack([o[k] for o in obs_buf], dim=0) for k in keys}
    obs_flat = {k: v.reshape(T * N, *v.shape[2:]).contiguous().cpu() for k, v in obs_tn.items()}

    rollout = RolloutBatch(
        obs=obs_flat,
        actions={
            "types": types_tn.reshape(T * N, *types_tn.shape[2:]).cpu(),
            "dirs": dirs_tn.reshape(T * N, *dirs_tn.shape[2:]).cpu(),
            "targets": tgts_tn.reshape(T * N, *tgts_tn.shape[2:]).cpu(),
            "amounts": amts_tn.reshape(T * N, *amts_tn.shape[2:]).cpu(),
        },
        logprob=logprob_tn.reshape(T * N).cpu(),
        value=values_tn.reshape(T * N).cpu(),
        reward=rewards_tn.reshape(T * N).cpu(),  # normalized if enabled (for diagnostics)
        done=dones_tn.reshape(T * N).cpu(),
        advantage=adv.reshape(T * N).cpu(),
        ret=ret.reshape(T * N).cpu(),
        batch_size=T * N,
    )

    rate = (total_harvest + total_control) / max(1, T)
    meta = {
        "rollout_steps": float(T),
        "rollout_env_steps": float(T * N),
        "extensions": float(extensions),
        "skill_rate": float(rate),
        "total_harvest": float(total_harvest),
        "total_control": float(total_control),
        "mean_reward": float(rewards_raw.mean().item()),  # always raw for charts
        "mean_reward_norm": float(rewards_tn.mean().item()),
        "ret_mean": float(ret.mean().item()),
        "ret_std": float(ret.std().item()),
        "v_mean": float(values_tn.mean().item()),
        "v_std": float(values_tn.std().item()),
        **gae_info,
    }
    if reward_normalizer is not None:
        meta.update(reward_normalizer.stats())
    return rollout, obs, meta, T


def main() -> int:
    args = parse_args()
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
    )
    agent = Agent()
    # Eager params on device first; torch.compile wraps in PPOTrainer
    agent.to(device)
    resume_update = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "actor" in ckpt:
            ma, ua = agent.actor.load_state_dict(ckpt["actor"], strict=False)
            mc, uc = agent.critic.load_state_dict(ckpt["critic"], strict=False)
            resume_update = int(ckpt.get("update", 0)) + 1
            print(
                f"[train] resumed {args.resume} update={ckpt.get('update')} "
                f"global_step={ckpt.get('global_step')} "
                f"actor_miss={len(ma)} critic_miss={len(mc)}",
                flush=True,
            )
        else:
            agent.load_state_dict(ckpt, strict=False)
            print(f"[train] resumed raw state_dict from {args.resume}", flush=True)
    elif args.critic_pretrain is not None:
        ckpt = torch.load(args.critic_pretrain, map_location=device, weights_only=False)
        state = ckpt["critic"] if isinstance(ckpt, dict) and "critic" in ckpt else ckpt
        # strict=False: architecture may gain QK-Norm / room_pool_score after pretrain
        missing, unexpected = agent.critic.load_state_dict(state, strict=False)
        print(
            f"[train] loaded pretrained critic from {args.critic_pretrain} "
            f"(missing={len(missing)} unexpected={len(unexpected)})",
            flush=True,
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
    # Default 2048 with steps×envs = 512×8 = 4096 → two optimizer steps per epoch.
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
    # Capture CUDA graphs for reduce-overhead act before timed rollouts
    if not args.no_compile and device.type == "cuda":
        trainer.warmup(obs0, steps=5)

    global_step = 0
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
        print(
            f"[train] CleanRL reward norm ON (discounted-return RMS, clip={clip})",
            flush=True,
        )
    else:
        print("[train] reward norm OFF", flush=True)

    try:
        obs = obs0
        for update in range(args.updates):
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
            )
            finished_returns = ep_returns[n_before_r:]
            finished_lengths = ep_lengths[n_before_l:]

            global_step += int(meta["rollout_env_steps"])
            stats = trainer.update(rollout)
            # Release peak activation cache so residual VRAM doesn't look pegged
            # after the big full-batch step (mb can be thousands of transitions).
            if device.type == "cuda":
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
            print(
                f"[update {update+1}/{args.updates}] "
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
                writer.flush()

            if (update + 1) % 5 == 0:
                ckpt = {
                    "actor": trainer.actor.state_dict(),
                    "critic": trainer.critic.state_dict(),
                    "actor_opt": trainer.actor_opt.state_dict(),
                    "critic_opt": trainer.critic_opt.state_dict(),
                    "update": update,
                    "global_step": global_step,
                    "minibatch": mb,
                    "args": vars(args),
                }
                torch.save(ckpt, args.save)
                print(f"[train] saved {args.save}", flush=True)

        torch.save({
            "actor": trainer.actor.state_dict(),
            "critic": trainer.critic.state_dict(),
            "global_step": global_step,
            "minibatch": mb,
        }, args.save)
        elapsed = time.time() - start_time
        print(f"[train] done → {args.save}  global_step={global_step} elapsed={elapsed:.1f}s", flush=True)
        return 0
    finally:
        envs.close()
        if writer:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
