"""PPO with monolithic torch.compile(reduce-overhead) of Agent / Actor / Critic."""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .constants import INTENT_SLOTS, MAX_ACTORS, PPO_CFG
from .model import Actor, Agent, Critic, maybe_compile


@dataclass
class RolloutBatch:
    obs: dict[str, Tensor]
    actions: dict[str, Tensor]
    logprob: Tensor
    value: Tensor
    reward: Tensor
    done: Tensor
    advantage: Tensor
    ret: Tensor
    batch_size: int = 0


class PPOTrainer:
    def __init__(
        self,
        agent: Agent | None = None,
        *,
        actor: Actor | None = None,
        critic: Critic | None = None,
        lr: float | None = None,
        critic_lr: float | None = None,
        clip: float | None = None,
        clip_high: float | None = None,
        entropy_coef: float | None = None,
        value_coef: float | None = None,
        max_grad_norm: float | None = None,
        epochs: int | None = None,
        minibatch: int | None = None,
        device: torch.device | str = "cpu",
        compile_model: bool = True,
        use_bf16: bool = True,
    ):
        self.cfg = PPO_CFG
        self.device = torch.device(device)
        self.compile_model = bool(compile_model)

        if agent is not None:
            self.actor = agent.actor
            self.critic = agent.critic
            self.agent = agent
        else:
            assert actor is not None and critic is not None
            self.actor = actor
            self.critic = critic
            self.agent = Agent()
            self.agent.actor = actor  # type: ignore[assignment]
            self.agent.critic = critic  # type: ignore[assignment]

        self.agent.to(self.device)
        self.actor = self.agent.actor
        self.critic = self.agent.critic

        # Monolithic compiles (whole modules — not subgraph pieces):
        #   agent_c  — rollout act (actor+critic in one graph, reduce-overhead)
        #   actor_c  — full Actor for PPO policy update
        #   critic_c — full Critic for PPO value update
        # Eager refs kept for optim / state_dict; compiled handles call.
        self.agent_c = maybe_compile(self.agent, compile_model, "agent-monolith")
        self.actor_c = maybe_compile(self.actor, compile_model, "actor-monolith")
        self.critic_c = maybe_compile(self.critic, compile_model, "critic-monolith")

        self.use_bf16 = bool(use_bf16 and self.device.type == "cuda" and torch.cuda.is_bf16_supported())
        actor_lr = lr if lr is not None else float(self.cfg["lr"])
        # VAPO: critic learns faster
        c_lr = critic_lr if critic_lr is not None else actor_lr * 2.0

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=c_lr, eps=1e-5)

        # Asymmetric policy clip: ratio ∈ [1 − clip, 1 + clipHigh]
        # Low side stays conservative (0.2); high side allows more probability growth (0.28).
        self.clip = clip if clip is not None else float(self.cfg["clip"])
        self.clip_high = (
            clip_high if clip_high is not None else float(self.cfg.get("clipHigh", self.clip))
        )
        self.entropy_coef = entropy_coef if entropy_coef is not None else float(self.cfg["entropyCoef"])
        self.value_coef = value_coef if value_coef is not None else float(self.cfg["valueCoef"])
        self.max_grad_norm = max_grad_norm if max_grad_norm is not None else float(self.cfg["maxGradNorm"])
        self.epochs = epochs if epochs is not None else int(self.cfg["epochs"])
        self.minibatch = minibatch if minibatch is not None else int(self.cfg["minibatch"])
        self._warmed = False

    @property
    def policy(self) -> Agent:
        """Back-compat for train.py checkpointing."""
        return self.agent

    @torch.inference_mode()
    def act(self, obs: dict[str, Tensor], deterministic: bool = False):
        """Rollout via monolithic agent_c (policy + value, reduce-overhead)."""
        self.agent.eval()
        model_obs = {k: v for k, v in obs.items() if not k.startswith("_")}
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_bf16 else nullcontext()
        with ctx:
            out = self.agent_c(model_obs, deterministic=deterministic)
        out.logprob = out.logprob.float()
        out.entropy = out.entropy.float()
        out.value = out.value.float()
        return out

    def warmup(self, obs: dict[str, Tensor], *, steps: int = 5) -> None:
        """Freeze static shapes, then capture reduce-overhead CUDA graphs."""
        if self.device.type != "cuda":
            return
        model_obs = {k: v for k, v in obs.items() if not k.startswith("_")}
        # Freeze room pack on eager modules before any compiled call
        r = self.agent.freeze_room_pack(model_obs["room_mask"])
        print(
            f"[ppo] monolithic compile warmup ×{steps} "
            f"(B={model_obs['patches'].shape[0]} R_pack={r} compile={self.compile_model}) …",
            flush=True,
        )
        t0 = time.perf_counter()
        # Rollout graph (agent-monolith)
        for _ in range(steps):
            _ = self.act(model_obs)
        # Update graphs (actor/critic monoliths) — fixed mb-sized dummy if needed later
        if self.compile_model:
            self.actor.train()
            self.critic.train()
            B = model_obs["patches"].shape[0]
            # Tiny train-mode trace so update path is compiled before first real update
            n_act = int(model_obs["actors"].shape[1]) if model_obs["actors"].dim() > 1 else MAX_ACTORS
            dummy_act = {
                "types": torch.zeros(B, n_act, INTENT_SLOTS, dtype=torch.long, device=self.device),
                "dirs": torch.zeros(B, n_act, INTENT_SLOTS, dtype=torch.long, device=self.device),
                "targets": torch.zeros(B, n_act, INTENT_SLOTS, dtype=torch.long, device=self.device),
                "amounts": torch.zeros(B, n_act, INTENT_SLOTS, dtype=torch.long, device=self.device),
            }
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_bf16 else nullcontext()
            with ctx:
                out = self.actor_c(model_obs, action=dummy_act)
                v = self.critic_c(model_obs)
            (out.logprob.float().mean() + v.float().mean()).backward()
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            self.actor.eval()
            self.critic.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._warmed = True
        print(f"[ppo] warmup done in {time.perf_counter() - t0:.2f}s", flush=True)

    def update(self, rollout: RolloutBatch) -> dict[str, float]:
        self.actor.train()
        self.critic.train()
        B = rollout.reward.reshape(-1).shape[0]
        # Index on CPU so full-rollout tensors can stay host-side (stream mb → GPU).
        idx = torch.arange(B)
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "old_approx_kl": 0.0,
            "clipfrac": 0.0,
            "grad_norm": 0.0,
            "grad_norm_actor": 0.0,
            "grad_norm_critic": 0.0,
            "explained_variance": 0.0,
        }
        n_updates = 0

        obs_flat = {k: v for k, v in rollout.obs.items() if not k.startswith("_")}
        for k, v in list(obs_flat.items()):
            if v.shape[0] != B:
                obs_flat[k] = v.reshape(B, *v.shape[1:])

        act_flat = {}
        for k, v in rollout.actions.items():
            if v.dim() >= 3 and v.shape[0] * v.shape[1] == B:
                act_flat[k] = v.reshape(B, *v.shape[2:])
            elif v.shape[0] != B:
                act_flat[k] = v.reshape(B, *v.shape[1:])
            else:
                act_flat[k] = v

        old_lp = rollout.logprob.reshape(B)
        old_v = rollout.value.reshape(B)
        adv_all = rollout.advantage.reshape(B)
        ret_all = rollout.ret.reshape(B)

        y_pred = old_v.detach().cpu().numpy()
        y_true = ret_all.detach().cpu().numpy()
        var_y = np.var(y_true)
        stats["explained_variance"] = float(
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )

        # Minibatch size = #transitions per step; never larger than full batch B.
        # No gradient accumulation: each mb does zero_grad → backward → step once.
        mb = max(1, min(int(self.minibatch), B))
        if self.minibatch > B:
            # e.g. want 2048 but rollout only collected 1024
            print(
                f"[ppo] minibatch clamped {self.minibatch} → {mb} (full batch B={B})",
                flush=True,
            )

        for _ in range(self.epochs):
            perm = idx[torch.randperm(B)]
            for start in range(0, B, mb):
                inds = perm[start : start + mb]
                if inds.numel() == 0:
                    continue
                # Move only this mb to GPU (full rollout may live on CPU).
                obs_mb = {
                    k: v[inds].to(self.device, non_blocking=True)
                    for k, v in obs_flat.items()
                    if not k.startswith("_")
                }
                act_mb = {k: v[inds].to(self.device, non_blocking=True) for k, v in act_flat.items()}
                old_lp_mb = old_lp[inds].to(self.device, non_blocking=True)
                old_v_mb = old_v[inds].to(self.device, non_blocking=True)
                # CleanRL: norm_adv=True — per-minibatch advantage normalize
                # Reward norm (discounted-return RMS) is applied in train.collect_rollout
                # before GAE so `ret` / values share that scale; critic trains on those.
                adv = adv_all[inds].to(self.device, non_blocking=True)
                ret = ret_all[inds].to(self.device, non_blocking=True)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_bf16 else nullcontext()

                # Sequential actor → critic so only one network's activations are live.
                # Dual trunks + B·R patch attention dominates VRAM; concurrent graphs OOM'd at mb=2048.
                with ctx:
                    out = self.actor_c(obs_mb, action=act_mb)
                new_lp = out.logprob.float()
                entropy = out.entropy.float().mean()

                logratio = new_lp - old_lp_mb
                ratio = logratio.exp()
                ratio_lo = 1.0 - self.clip
                ratio_hi = 1.0 + self.clip_high
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # Fraction of samples outside the asymmetric clip band
                    clipfrac = ((ratio < ratio_lo) | (ratio > ratio_hi)).float().mean()

                # Asymmetric policy clip (ε_low=clip, ε_high=clipHigh):
                #   ratio clamped to [1−ε_low, 1+ε_high]
                #   pg_loss = −min(A·r, A·clip(r)).mean()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, ratio_lo, ratio_hi) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # No grad accumulation: zero → backward → clip → step, once per minibatch
                self.actor_opt.zero_grad(set_to_none=True)
                (policy_loss - self.entropy_coef * entropy).backward()
                gn_a = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                with ctx:
                    new_v = self.critic_c(obs_mb)
                new_v = new_v.float()

                # CleanRL clip_vloss=True
                v_loss_unclipped = (new_v - ret) ** 2
                v_clipped = old_v_mb + torch.clamp(new_v - old_v_mb, -self.clip, self.clip)
                v_loss_clipped = (v_clipped - ret) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                self.critic_opt.zero_grad(set_to_none=True)
                (self.value_coef * value_loss).backward()
                gn_c = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                gn_a_f = float(gn_a if not torch.is_tensor(gn_a) else gn_a.item())
                gn_c_f = float(gn_c if not torch.is_tensor(gn_c) else gn_c.item())

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["approx_kl"] += float(approx_kl.item())
                stats["old_approx_kl"] += float(old_approx_kl.item())
                stats["clipfrac"] += float(clipfrac.item())
                stats["grad_norm_actor"] += gn_a_f
                stats["grad_norm_critic"] += gn_c_f
                stats["grad_norm"] += max(gn_a_f, gn_c_f)
                n_updates += 1

        if n_updates:
            for k in list(stats.keys()):
                if k != "explained_variance":
                    stats[k] /= n_updates
        stats["minibatch"] = float(mb)
        stats["batch_size"] = float(B)
        stats["clip"] = float(self.clip)
        stats["clip_high"] = float(self.clip_high)
        return stats
