"""PPO with torch.compile(reduce-overhead) of Actor / Critic (no third agent graph).

GPU focus:
  · one compiled graph per trunk (actor + critic) — not a third agent monolith
  · static last-mb pad so reduce-overhead CUDA graphs stay captured
  · warmup at real minibatch B (not num_envs) so update path does not recompile
  · sequential actor→critic update (only one trunk's activations live)
  · TF32 + cudnn.benchmark; contiguous H2D
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .constants import (
    INTENT_SLOTS,
    MAX_ACTORS,
    MAX_TARGETS,
    N_AMOUNT,
    N_DIR,
    N_INTENT,
    PPO_CFG,
)
from .model import Actor, Agent, Critic, configure_cuda_backends, maybe_compile


def _mean_actor_then_transition(values: Tensor, live: Tensor) -> Tensor:
    """Equal-weight team states regardless of their current population.

    A shared team advantage is one sample per transition. Averaging every actor
    globally would make a 64-creep colony count 64 times more than a bootstrap
    state with one creep.
    """
    live_f = live.to(values.dtype)
    per_state = (values * live_f).sum(dim=-1) / live_f.sum(dim=-1).clamp_min(1.0)
    return per_state.mean()


@dataclass
class RolloutBatch:
    obs: dict[str, Tensor]
    actions: dict[str, Tensor]
    logprob: Tensor  # [B, A] per-actor factors; PPO never clips the team product
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

        if self.device.type == "cuda":
            configure_cuda_backends()

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

        # Two compiled graphs only (actor + critic). Avoid a third agent-monolith
        # CUDA graph — that duplicated capture VRAM without saving launches.
        self.actor_c = maybe_compile(self.actor, compile_model, "actor-monolith")
        self.critic_c = maybe_compile(self.critic, compile_model, "critic-monolith")

        self.use_bf16 = bool(use_bf16 and self.device.type == "cuda" and torch.cuda.is_bf16_supported())
        actor_lr = lr if lr is not None else float(self.cfg["lr"])
        c_lr = critic_lr if critic_lr is not None else actor_lr * 2.0

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=c_lr, eps=1e-5)

        self.clip = clip if clip is not None else float(self.cfg["clip"])
        self.clip_high = (
            clip_high if clip_high is not None else float(self.cfg.get("clipHigh", self.clip))
        )
        self.entropy_coef = entropy_coef if entropy_coef is not None else float(self.cfg["entropyCoef"])
        self.value_coef = value_coef if value_coef is not None else float(self.cfg["valueCoef"])
        self.max_grad_norm = max_grad_norm if max_grad_norm is not None else float(self.cfg["maxGradNorm"])
        self.epochs = epochs if epochs is not None else int(self.cfg["epochs"])
        self.minibatch = minibatch if minibatch is not None else int(self.cfg["minibatch"])
        self.target_kl = float(self.cfg.get("targetKl", 0.02))
        self._warmed = False
        # Pinned host scratch for mb staging (set in warmup / first update).
        self._pin_hold: list[torch.Tensor] = []

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.inference_mode()
    def act(self, obs: dict[str, Tensor], deterministic: bool = False):
        """Rollout: actor then critic (two graphs; no third agent monolith)."""
        self.actor.eval()
        self.critic.eval()
        model_obs = {k: v for k, v in obs.items() if not k.startswith("_")}
        with self._autocast():
            out = self.actor_c(model_obs, deterministic=deterministic)
            out.value = self.critic_c(model_obs)
        out.logprob = out.logprob.float()
        out.entropy = out.entropy.float()
        out.actor_logprob = out.actor_logprob.float()
        out.actor_entropy = out.actor_entropy.float()
        out.value = out.value.float()
        return out

    @torch.inference_mode()
    def value_only(self, obs: dict[str, Tensor]) -> Tensor:
        """Critic-only forward (bootstrap / terminal V) — skips actor trunk."""
        self.critic.eval()
        model_obs = {k: v for k, v in obs.items() if not k.startswith("_")}
        with self._autocast():
            v = self.critic_c(model_obs)
        return v.float()

    def _expand_obs_to_b(self, obs: dict[str, Tensor], B: int) -> dict[str, Tensor]:
        """Tile env-batch obs to fixed B for compile capture (update path)."""
        n0 = next(v.shape[0] for k, v in obs.items() if torch.is_tensor(v) and not k.startswith("_"))
        if n0 == B:
            return {k: v for k, v in obs.items() if not k.startswith("_")}
        out: dict[str, Tensor] = {}
        reps = (B + n0 - 1) // n0
        for k, v in obs.items():
            if k.startswith("_") or not torch.is_tensor(v):
                continue
            t = v.repeat((reps,) + (1,) * (v.dim() - 1))[:B]
            out[k] = t.contiguous()
        return out

    def warmup(self, obs: dict[str, Tensor], *, steps: int = 5) -> None:
        """Freeze room pack; capture reduce-overhead graphs at *update* mb size."""
        if self.device.type != "cuda":
            return
        model_obs = {k: v for k, v in obs.items() if not k.startswith("_")}
        r = self.agent.freeze_room_pack(model_obs["room_mask"])
        mb = max(1, int(self.minibatch))
        n_env = int(model_obs["patches"].shape[0])
        print(
            f"[ppo] compile warmup ×{steps} "
            f"(rollout B={n_env} update_mb={mb} R_pack={r} compile={self.compile_model}) …",
            flush=True,
        )
        t0 = time.perf_counter()

        # Capture the ONLY shapes we allow at runtime (dynamic=False):
        #   act / value_only: B = n_env  (never variable n_term)
        #   update:           B = mb     (last mb padded to mb)
        for _ in range(steps):
            _ = self.act(model_obs)
            _ = self.value_only(model_obs)  # same B=n_env as act critic path

        if self.compile_model:
            # Update: train-mode + action=dict (different specialization than act sample).
            self.actor.train()
            self.critic.train()
            mb_obs = self._expand_obs_to_b(model_obs, mb)
            n_act = int(mb_obs["actors"].shape[1]) if mb_obs["actors"].dim() > 1 else MAX_ACTORS
            dummy_act = {
                "types": torch.zeros(mb, n_act, INTENT_SLOTS, dtype=torch.long, device=self.device),
                "dirs": torch.zeros(mb, n_act, INTENT_SLOTS, dtype=torch.long, device=self.device),
                "targets": torch.zeros(mb, n_act, INTENT_SLOTS, dtype=torch.long, device=self.device),
                "amounts": torch.zeros(mb, n_act, INTENT_SLOTS, dtype=torch.long, device=self.device),
            }
            with self._autocast():
                out = self.actor_c(mb_obs, action=dummy_act)
                v = self.critic_c(mb_obs)
            (out.logprob.float().mean() + v.float().mean()).backward()
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            del out, v, dummy_act, mb_obs
            self.actor.eval()
            self.critic.eval()
            # Re-hit eval act after train capture so both mode specializations exist.
            for _ in range(2):
                _ = self.act(model_obs)
                _ = self.value_only(model_obs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self._warmed = True
        print(
            f"[ppo] warmup done in {time.perf_counter() - t0:.2f}s "
            f"(graphs: act@B={n_env}, value_only@B={n_env}, update@B={mb})",
            flush=True,
        )

    def update(self, rollout: RolloutBatch, *, critic_only: bool = False) -> dict[str, float]:
        self.actor.train()
        self.critic.train()
        B = rollout.reward.reshape(-1).shape[0]
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
        action_bounds = {
            "types": N_INTENT,
            "dirs": N_DIR,
            "targets": MAX_TARGETS,
            "amounts": N_AMOUNT,
        }
        for name, upper in action_bounds.items():
            values = rollout.actions[name]
            lo = int(values.min().item())
            hi = int(values.max().item())
            if lo < 0 or hi >= upper:
                raise RuntimeError(
                    f"rollout action {name} outside [0,{upper}): min={lo} max={hi}"
                )
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

        old_lp = rollout.logprob.reshape(B, MAX_ACTORS)
        old_v = rollout.value.reshape(B)
        adv_all = rollout.advantage.reshape(B)
        ret_all = rollout.ret.reshape(B)

        # One stable normalization contract for the rollout. Re-normalizing each
        # shuffled minibatch changes rare economy advantages by batch composition.
        if adv_all.numel() >= 2:
            adv_all = (adv_all - adv_all.mean()) / (adv_all.std(unbiased=False) + 1e-8)
        else:
            adv_all = adv_all * 0

        # Host-side EV (no GPU sync on full batch)
        y_pred = old_v.detach().cpu().numpy()
        y_true = ret_all.detach().cpu().numpy()
        var_y = np.var(y_true)
        stats["explained_variance"] = float(
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )

        mb = max(1, min(int(self.minibatch), B))
        if self.minibatch > B:
            print(
                f"[ppo] minibatch clamped {self.minibatch} → {mb} (full batch B={B})",
                flush=True,
            )

        early_stop = False
        stopped_epoch = -1
        pad_last_mb = self.device.type == "cuda" and self.compile_model and mb > 1
        pin = self.device.type == "cuda"
        acc: dict[str, Tensor] | None = None

        for epoch in range(self.epochs):
            if early_stop:
                break
            perm = idx[torch.randperm(B)]
            for start in range(0, B, mb):
                inds = perm[start : start + mb]
                if inds.numel() == 0:
                    continue
                n_real = int(inds.numel())
                if pad_last_mb and n_real < mb:
                    pad = inds[-1:].repeat(mb - n_real)
                    inds = torch.cat([inds, pad], dim=0)

                # Stream only this mb to GPU; promote uint8→float32 here.
                self._pin_hold.clear()
                from .vec_env import promote_obs_device

                obs_mb = promote_obs_device(
                    {k: v[inds] for k, v in obs_flat.items() if not k.startswith("_")},
                    self.device,
                    pin_hold=self._pin_hold,
                    non_blocking=pin,
                )

                act_mb = {
                    k: v[inds].to(self.device, non_blocking=pin)
                    for k, v in act_flat.items()
                }
                old_lp_mb = old_lp[inds].to(self.device, non_blocking=pin)
                old_v_mb = old_v[inds].to(self.device, non_blocking=pin)
                adv = adv_all[inds].to(self.device, non_blocking=pin)
                ret = ret_all[inds].to(self.device, non_blocking=pin)

                if n_real < inds.numel():
                    adv[n_real:] = 0

                def _real(t: Tensor) -> Tensor:
                    return t[:n_real] if n_real < t.shape[0] else t

                policy_loss = torch.zeros((), device=self.device)
                entropy = torch.zeros((), device=self.device)
                approx_kl = torch.zeros((), device=self.device)
                old_approx_kl = torch.zeros((), device=self.device)
                clipfrac = torch.zeros((), device=self.device)
                gn_a = 0.0

                # --- Actor update (activations freed before critic) ---
                if not critic_only:
                    with self._autocast():
                        out = self.actor_c(obs_mb, action=act_mb)
                    new_lp = _real(out.actor_logprob.float())
                    actor_ent = _real(out.actor_entropy.float())
                    old_lp_r = _real(old_lp_mb)
                    adv_r = _real(adv)
                    live = (_real(obs_mb["actor_mask"]) > 0).float()

                    logratio = new_lp - old_lp_r
                    ratio = logratio.exp()
                    ratio_lo = 1.0 - self.clip
                    ratio_hi = 1.0 + self.clip_high
                    with torch.no_grad():
                        old_approx_kl = _mean_actor_then_transition(-logratio, live)
                        approx_kl = _mean_actor_then_transition((ratio - 1) - logratio, live)
                        clipped = ((ratio < ratio_lo) | (ratio > ratio_hi)).float()
                        clipfrac = _mean_actor_then_transition(clipped, live)

                    actor_adv = adv_r.unsqueeze(-1).expand_as(ratio)
                    surr1 = ratio * actor_adv
                    surr2 = torch.clamp(ratio, ratio_lo, ratio_hi) * actor_adv
                    policy_loss = -_mean_actor_then_transition(torch.min(surr1, surr2), live)
                    entropy = _mean_actor_then_transition(actor_ent, live)

                    self.actor_opt.zero_grad(set_to_none=True)
                    (policy_loss - self.entropy_coef * entropy).backward()
                    gn_a = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_opt.step()
                    # Drop actor activations before critic forward (VRAM peak).
                    del out, new_lp, logratio, ratio, surr1, surr2, adv_r, old_lp_r, actor_adv, actor_ent, live

                # --- Critic update ---
                critic_mod = self.critic
                use_hl = bool(getattr(critic_mod, "use_hl_gauss", False))
                ret_r = _real(ret)
                old_v_r = _real(old_v_mb)
                if use_hl:
                    with self._autocast():
                        logits = critic_mod.value_logits(obs_mb)
                    logits_r = _real(logits.float())
                    value_loss = critic_mod.support.cross_entropy(logits_r, ret_r).mean()
                    del logits, logits_r
                else:
                    with self._autocast():
                        new_v = self.critic_c(obs_mb)
                    new_v = _real(new_v.float())
                    use_vclip = bool(self.cfg.get("clipValueLoss", False))
                    v_loss_unclipped = (new_v - ret_r) ** 2
                    if use_vclip:
                        v_clipped = old_v_r + torch.clamp(new_v - old_v_r, -self.clip, self.clip)
                        v_loss_clipped = (v_clipped - ret_r) ** 2
                        value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                        del v_clipped, v_loss_clipped
                    else:
                        value_loss = 0.5 * v_loss_unclipped.mean()
                    del new_v, v_loss_unclipped

                self.critic_opt.zero_grad(set_to_none=True)
                (self.value_coef * value_loss).backward()
                gn_c = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                # Free mb device tensors before next slice
                del obs_mb, act_mb, old_lp_mb, old_v_mb, adv, ret, ret_r, old_v_r

                gna_t = gn_a if torch.is_tensor(gn_a) else torch.as_tensor(float(gn_a), device=self.device)
                gnc_t = gn_c if torch.is_tensor(gn_c) else torch.as_tensor(float(gn_c), device=self.device)
                # Accumulate on device; flush to host once per update.
                if acc is None:
                    acc = {
                        "policy_loss": policy_loss.detach(),
                        "value_loss": value_loss.detach(),
                        "entropy": entropy.detach(),
                        "approx_kl": approx_kl.detach(),
                        "old_approx_kl": old_approx_kl.detach(),
                        "clipfrac": clipfrac.detach(),
                        "grad_norm_actor": gna_t.detach().float(),
                        "grad_norm_critic": gnc_t.detach().float(),
                    }
                else:
                    acc["policy_loss"] = acc["policy_loss"] + policy_loss.detach()
                    acc["value_loss"] = acc["value_loss"] + value_loss.detach()
                    acc["entropy"] = acc["entropy"] + entropy.detach()
                    acc["approx_kl"] = acc["approx_kl"] + approx_kl.detach()
                    acc["old_approx_kl"] = acc["old_approx_kl"] + old_approx_kl.detach()
                    acc["clipfrac"] = acc["clipfrac"] + clipfrac.detach()
                    acc["grad_norm_actor"] = acc["grad_norm_actor"] + gna_t.detach().float()
                    acc["grad_norm_critic"] = acc["grad_norm_critic"] + gnc_t.detach().float()
                n_updates += 1

                # KL early-stop still needs a sync for control flow when enabled.
                stop_now = False
                if (not critic_only) and self.target_kl > 0:
                    stop_now = bool((approx_kl.detach() > self.target_kl).item())
                del policy_loss, value_loss, entropy, approx_kl, old_approx_kl, clipfrac

                if stop_now:
                    early_stop = True
                    stopped_epoch = epoch
                    break

        if n_updates and acc is not None:
            for k, v in acc.items():
                stats[k] = float(v.detach().float().mean().cpu() if v.dim() else v.detach().float().cpu()) / n_updates
            stats["grad_norm"] = max(stats["grad_norm_actor"], stats["grad_norm_critic"])
        stats["minibatch"] = float(mb)
        stats["batch_size"] = float(B)
        stats["clip"] = float(self.clip)
        stats["clip_high"] = float(self.clip_high)
        stats["early_stop"] = float(early_stop)
        stats["stopped_epoch"] = float(stopped_epoch)
        stats["target_kl"] = float(self.target_kl)
        return stats
