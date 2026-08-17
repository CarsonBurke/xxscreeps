"""PPO with torch.compile(reduce-overhead) of Actor / Critic (no third agent graph).

GPU focus:
  · one compiled graph per trunk (actor + critic) — not a third agent monolith
  · static last-mb pad so reduce-overhead CUDA graphs stay captured
  · warmup at real minibatch B (not num_envs) so update path does not recompile
  · sequential actor→critic update (only one trunk's activations live)
  · TF32 + cudnn.benchmark; contiguous H2D
"""
from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .constants import (
    MAX_ACTORS,
    MAX_TARGETS,
    NEXTLAT_CFG,
    N_AMOUNT,
    N_BODY_PART,
    N_CONSTRUCTION_TILE,
    N_CONSTRUCTION_TYPE,
    N_DIR,
    N_INTENT,
    PPO_CFG,
    VALUE_CFG,
)
from .model import Actor, Agent, Critic, configure_cuda_backends, maybe_compile
from .muon import PPO_MUON_LR, HybridMuonAdamW
from .rollout_buffer import FlatRolloutObservations


def _own(tensor: Tensor) -> Tensor:
    """Return a tensor whose storage survives the next compiled replay.

    `torch.compile(mode="reduce-overhead")` returns tensors backed by a CUDA
    graph's capture pool; replaying the graph overwrites them. Anything the
    trainer keeps past the next forward has to own its memory first, and
    `.float()` is not a copy when the tensor is already float32.
    """
    return tensor.clone()


def _mean_actor_then_transition(values: Tensor, live: Tensor) -> Tensor:
    """Equal-weight team states regardless of their current population.

    A shared team advantage is one sample per transition. Averaging every actor
    globally would make a 64-creep colony count 64 times more than a bootstrap
    state with one creep.
    """
    live_f = live.to(values.dtype)
    per_state = (values * live_f).sum(dim=-1) / live_f.sum(dim=-1).clamp_min(1.0)
    return per_state.mean()


def _masked_latent_loss(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Smooth-L1 over valid transitions, averaged per state and latent feature."""
    per_state = torch.nn.functional.smooth_l1_loss(
        prediction.float(), target.detach().float(), reduction="none",
    ).mean(dim=-1)
    weights = valid.to(per_state.dtype)
    return (per_state * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_categorical_kl(
    student_logits: Tensor, teacher_logits: Tensor, valid: Tensor,
) -> Tensor:
    """KL(teacher || student), with no gradient through the teacher distribution."""
    teacher_logp = torch.nn.functional.log_softmax(teacher_logits.detach().float(), dim=-1)
    student_logp = torch.nn.functional.log_softmax(student_logits.float(), dim=-1)
    per_state = torch.nn.functional.kl_div(
        student_logp, teacher_logp, log_target=True, reduction="none",
    ).sum(dim=-1)
    weights = valid.to(per_state.dtype)
    return (per_state * weights).sum() / weights.sum().clamp_min(1.0)


@dataclass
class RolloutBatch:
    obs: dict[str, Tensor] | FlatRolloutObservations
    actions: dict[str, Tensor]
    logprob: Tensor  # [B, A] per-actor factors; PPO never clips the team product
    value: Tensor
    reward: Tensor
    done: Tensor
    advantage: Tensor
    ret: Tensor
    actor_latent: Tensor | None = None  # [B,D], frozen behavior encoder state
    critic_latent: Tensor | None = None  # [B,D], frozen pre-update encoder state
    critic_logits: Tensor | None = None  # [B,K], frozen pre-update value teacher
    next_indices: Tensor | None = None  # [B], time-major next-state row
    nextlat_valid: Tensor | None = None  # [B], excludes reset/terminal/last rows
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
        muon_lr: float | None = None,
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

        # Compile exactly the tick path. `act` and `value_only` run once per
        # simulated tick at B=n_env - thousands of launch-bound calls per update
        # - which is what `reduce-overhead` CUDA graphs remove, and their
        # capture pools are a couple of GB.
        #
        # The minibatch path stays eager. It is compute-bound, so compiling it
        # saved about 1 s of a 13 s update, while a training forward and
        # backward at minibatch 1536 with four room slots already peaks near
        # 22 GB: inductor's workspace on top of that exhausts a 32 GB device
        # during warmup. Measured, not assumed - both failures are in
        # PERFORMANCE.md.
        self.actor_c = maybe_compile(self.actor, compile_model, "actor-rollout")
        self.critic_c = maybe_compile(self.critic, compile_model, "critic-rollout")

        self.use_bf16 = bool(use_bf16 and self.device.type == "cuda" and torch.cuda.is_bf16_supported())
        actor_lr = lr if lr is not None else float(self.cfg["lr"])
        c_lr = critic_lr if critic_lr is not None else actor_lr * 2.0

        # Hidden trunk matrices go to Muon (Polar Express + NorMuon); embeddings,
        # norms, biases and every head stay on fused AdamW. `muon_lr` defaults to
        # the value that matches the RMS coordinate step AdamW was taking on
        # those matrices, so the geometry changes and the step size does not.
        self.muon_lr = float(muon_lr) if muon_lr is not None else PPO_MUON_LR
        self.actor_opt = HybridMuonAdamW(
            self.actor, adam_lr=actor_lr, muon_lr=self.muon_lr,
        )
        self.critic_opt = HybridMuonAdamW(
            self.critic, adam_lr=c_lr, muon_lr=self.muon_lr * 2.0,
        )

        self.clip = clip if clip is not None else float(self.cfg["clip"])
        self.clip_high = (
            clip_high if clip_high is not None else float(self.cfg.get("clipHigh", self.clip))
        )
        self.entropy_coef = entropy_coef if entropy_coef is not None else float(self.cfg["entropyCoef"])
        self.value_coef = value_coef if value_coef is not None else float(self.cfg["valueCoef"])
        self.max_grad_norm = max_grad_norm if max_grad_norm is not None else float(self.cfg["maxGradNorm"])
        self.critic_max_grad_norm = float(VALUE_CFG["criticMaxGradNorm"])
        self.epochs = epochs if epochs is not None else int(self.cfg["epochs"])
        self.minibatch = minibatch if minibatch is not None else int(self.cfg["minibatch"])
        self.target_kl = float(self.cfg.get("targetKl", 0.02))
        self.nextlat_actor_mse_coef = float(NEXTLAT_CFG["actorMseCoef"])
        self.nextlat_critic_mse_coef = float(NEXTLAT_CFG["criticMseCoef"])
        self.nextlat_critic_kl_coef = float(NEXTLAT_CFG["criticKlCoef"])
        if int(NEXTLAT_CFG.get("horizon", 1)) != 1:
            raise ValueError("current PPO temporal pairing supports nextLat.horizon=1")
        self._warmed = False
        # Pinned host scratch for mb staging (set in warmup / first update).
        self._pin_hold: list[torch.Tensor] = []
        # Every distinct (call site, shape) a compiled graph has been asked for.
        # `dynamic=False` mints a graph and a CUDA-graph capture pool per entry,
        # so a new one appearing after warmup is a mid-run recompile stall and
        # has to be attributable rather than inferred from a slow update.
        self._compiled_shapes: set[tuple[str, tuple[int, ...]]] = set()
        self._late_shape_mints = 0

    def _note_shape(self, site: str, obs: dict[str, Tensor]) -> None:
        if not self.compile_model:
            return
        rooms = obs.get("room_mask")
        actors = obs.get("actor_mask")
        targets = obs.get("target_mask")
        key = (
            site,
            (
                int(rooms.shape[0]) if rooms is not None else -1,
                int(rooms.shape[1]) if rooms is not None else -1,
                int(actors.shape[1]) if actors is not None else -1,
                int(targets.shape[1]) if targets is not None else -1,
            ),
        )
        if key in self._compiled_shapes:
            return
        self._compiled_shapes.add(key)
        if self._warmed:
            self._late_shape_mints += 1
            batch, rooms_n, actors_n, targets_n = key[1]
            print(
                f"[ppo] new compiled shape after warmup: {site} "
                f"B={batch} rooms={rooms_n} actors={actors_n} targets={targets_n} "
                f"(total late mints {self._late_shape_mints})",
                flush=True,
            )

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.inference_mode()
    def act_actor(self, obs: dict[str, Tensor], deterministic: bool = False):
        """Sample actions without putting the independent critic on the tick path."""
        self.actor.eval()
        model_obs = {k: v for k, v in obs.items() if not k.startswith("_")}
        self._note_shape("act", model_obs)
        with self._autocast():
            out = self.actor_c(model_obs, deterministic=deterministic)
        out.logprob = out.logprob.float()
        out.entropy = out.entropy.float()
        out.actor_logprob = out.actor_logprob.float()
        out.actor_entropy = out.actor_entropy.float()
        out.value = out.value.float()
        return out

    @torch.inference_mode()
    def act(self, obs: dict[str, Tensor], deterministic: bool = False):
        """Actor+critic convenience path for evaluation and compatibility."""
        out = self.act_actor(obs, deterministic=deterministic)
        out.value = self.value_only(obs)
        return out

    @torch.inference_mode()
    def value_only(self, obs: dict[str, Tensor]) -> Tensor:
        """Critic-only forward (bootstrap / terminal V) — skips actor trunk."""
        self.critic.eval()
        model_obs = {k: v for k, v in obs.items() if not k.startswith("_")}
        self._note_shape("value_only", model_obs)
        with self._autocast():
            v = self.critic_c(model_obs)
        # A reduce-overhead graph owns its output buffer and overwrites it on the
        # next replay, and `.float()` on a float32 tensor returns the same
        # storage. Callers keep this value past the next forward, so it must be
        # copied out of the capture pool.
        return _own(v.float())

    @torch.no_grad()
    def rollout_values(
        self,
        observations,
        *,
        batch_size: int | None = None,
        return_nextlat_targets: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Evaluate stored rollout states and retain compact temporal targets on GPU."""
        from .vec_env import promote_obs_device

        total = int(getattr(observations, "batch_size", 0))
        if total <= 0:
            values = torch.empty(0, dtype=torch.float32)
            if not return_nextlat_targets:
                return values
            latent = torch.empty(
                0, self.actor.d_model, dtype=torch.float32, device=self.device,
            )
            logits = torch.empty(
                0, self.critic.support.num_bins,
                dtype=torch.float32, device=self.device,
            )
            return values, latent, logits
        chunk = max(1, int(batch_size or self.minibatch))
        values: list[Tensor] = []
        latents: list[Tensor] = []
        logits_rows: list[Tensor] = []
        self.critic.eval()
        for start in range(0, total, chunk):
            indices = torch.arange(start, min(total, start + chunk))
            host = (
                observations.gather_minibatch(indices)
                if hasattr(observations, "gather_minibatch")
                else {key: value[indices] for key, value in observations.items()}
            )
            model_obs = promote_obs_device(
                host, self.device, non_blocking=self.device.type == "cuda",
            )
            with self._autocast():
                if return_nextlat_targets:
                    logits, latent = self.critic(
                        model_obs, return_logits=True, return_latent=True,
                    )
                    value = self.critic.support.to_expected_scalar(logits.float())
                else:
                    value = self.critic(model_obs)
            values.append(value.float())
            if return_nextlat_targets:
                # These compact tables are consumed by the immediately following
                # PPO update. Keeping them device-resident avoids a pointless
                # D2H followed by H2D round trip (~17 MiB at B=8192).
                latents.append(latent.float())
                logits_rows.append(logits.float())
        # One synchronization for all scalar values, after all critic chunks
        # have been enqueued. Compact latent/logit targets remain on device.
        values_all = torch.cat(values, dim=0).cpu()
        if not return_nextlat_targets:
            return values_all
        return values_all, torch.cat(latents, dim=0), torch.cat(logits_rows, dim=0)

    @torch.no_grad()
    def nextlat_rollout_targets(
        self, observations, *, batch_size: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode every rollout state once with frozen pre-update weights.

        PPO reuses each transition for several epochs.  Re-encoding the same
        future state inside every shuffled minibatch both wastes trunk compute
        and makes the target drift after earlier optimizer steps.  One compact
        pre-update target table matches NextLat's stop-gradient teacher and is
        only ~21 MiB for B=8192 (two 128-d latents + 409 value logits).
        """
        from .vec_env import promote_obs_device

        total = int(getattr(observations, "batch_size", 0))
        if total <= 0 and isinstance(observations, dict):
            total = int(next(iter(observations.values())).shape[0])
        if total <= 0:
            empty_latent = torch.empty(0, self.actor.d_model, dtype=torch.float32)
            empty_logits = torch.empty(
                0, self.critic.support.num_bins, dtype=torch.float32,
            )
            return empty_latent, empty_latent.clone(), empty_logits
        chunk = max(1, int(batch_size or self.minibatch))
        actor_targets: list[Tensor] = []
        critic_targets: list[Tensor] = []
        critic_logits: list[Tensor] = []
        actor_training = self.actor.training
        critic_training = self.critic.training
        self.actor.eval()
        self.critic.eval()
        try:
            for start in range(0, total, chunk):
                indices = torch.arange(start, min(total, start + chunk))
                host = (
                    observations.gather_minibatch(indices)
                    if hasattr(observations, "gather_minibatch")
                    else {key: value[indices] for key, value in observations.items()}
                )
                model_obs = promote_obs_device(
                    host, self.device, non_blocking=self.device.type == "cuda",
                )
                with self._autocast():
                    actor_latent = self.actor.encode_state(model_obs)
                    logits, critic_latent = self.critic.value_logits_and_latent(model_obs)
                actor_targets.append(actor_latent.float())
                critic_targets.append(critic_latent.float())
                critic_logits.append(logits.float())
                del model_obs, host, actor_latent, critic_latent, logits
        finally:
            self.actor.train(actor_training)
            self.critic.train(critic_training)
        actor_all = torch.cat(actor_targets, dim=0)
        critic_all = torch.cat(critic_targets, dim=0)
        logits_all = torch.cat(critic_logits, dim=0)
        return actor_all, critic_all, logits_all

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

    def _resize_room_pack(self, obs: dict[str, Tensor], rooms: int) -> dict[str, Tensor]:
        """Return `obs` with exactly `rooms` room slots; padding is masked out."""
        out: dict[str, Tensor] = {}
        for key, value in obs.items():
            if (
                key not in ("patches", "room_mask", "room_coords", "construction_mask")
                or not torch.is_tensor(value)
                or value.dim() < 2
            ):
                out[key] = value
                continue
            current = value.shape[1]
            if current == rooms:
                out[key] = value
            elif current > rooms:
                out[key] = value[:, :rooms].contiguous()
            else:
                pad = value.new_zeros(
                    (value.shape[0], rooms - current, *value.shape[2:]),
                )
                out[key] = torch.cat((value, pad), dim=1).contiguous()
        return out

    def warmup(self, obs: dict[str, Tensor], *, steps: int = 5) -> None:
        """Capture the compiled tick-path graphs before any timed rollout.

        Only `act` and `value_only` are compiled, at `B=n_env`, and
        `dynamic=False` specializes per shape, so both room packs are captured
        here: an episode starts at the seed room and reaches `MAX_ROOMS` once
        expansion exposes neighbors. A pack first seen mid-rollout would stall
        the tick loop while it records.

        `_note_shape` records what is captured, so a shape appearing later shows
        up as a late mint instead of an unexplained slow update.
        """
        if self.device.type != "cuda" or not self.compile_model:
            return
        from .vec_env import ROOM_BUCKETS

        model_obs = {k: v for k, v in obs.items() if not k.startswith("_")}
        r = self.agent.freeze_room_pack(model_obs["room_mask"])
        n_env = int(model_obs["patches"].shape[0])
        room_packs = sorted({int(pack) for pack in ROOM_BUCKETS})
        print(
            f"[ppo] compile warmup ×{steps} "
            f"(tick path B={n_env} reduce-overhead, room packs={room_packs}, "
            f"start_pack={r}; minibatch path eager) …",
            flush=True,
        )
        t0 = time.perf_counter()

        for rooms in sorted(room_packs, reverse=True):
            room_obs = self._resize_room_pack(model_obs, rooms)
            for _ in range(steps):
                _ = self.act(room_obs)
                _ = self.value_only(room_obs)  # same B=n_env as the act critic path
            del room_obs

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self._warmed = True
        print(
            f"[ppo] warmup done in {time.perf_counter() - t0:.2f}s "
            f"(captured {len(self._compiled_shapes)} tick-path shapes at B={n_env}, "
            f"room packs {room_packs})",
            flush=True,
        )

    def graph_stats(self) -> dict[str, float]:
        """Compiled-graph inventory, so the tick path's shapes stay measured.

        Only `act` and `value_only` are compiled. `dynamic=False` specializes per
        shape and per call signature, so the expected inventory is four graphs:
        actor and critic, each at both room packs, all captured by `warmup`.
        `compile_late_shape_mints` above zero means a shape appeared after
        warmup and paid a recompile inside a timed rollout.
        """
        if not self.compile_model:
            return {}
        try:
            from torch._dynamo.utils import counters
        except Exception:  # noqa: BLE001 - diagnostics must never break training
            return {}
        frames = counters.get("frames", {})
        breaks = counters.get("graph_break", {})
        return {
            "compile_unique_graphs": float(counters.get("stats", {}).get("unique_graphs", 0)),
            "compile_frames_ok": float(frames.get("ok", 0)),
            "compile_frames_total": float(frames.get("total", 0)),
            "compile_graph_breaks": float(sum(breaks.values())),
            "compile_graph_break_reasons": float(len(breaks)),
            "compile_distinct_shapes": float(len(self._compiled_shapes)),
            "compile_late_shape_mints": float(self._late_shape_mints),
        }

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
            "value_target_min": 0.0,
            "value_target_max": 0.0,
            "value_target_overflow_fraction": 0.0,
            "value_target_saturation_fraction": 0.0,
            "nextlat_actor_mse": 0.0,
            "nextlat_critic_mse": 0.0,
            "nextlat_critic_kl": 0.0,
            "nextlat_valid_fraction": 0.0,
            "nextlat_actor_delta_rms": 0.0,
            "nextlat_actor_state_rms": 0.0,
            "nextlat_critic_delta_rms": 0.0,
            "nextlat_critic_state_rms": 0.0,
        }
        n_updates = 0
        perf_detail = os.environ.get("RL_PERF_DETAIL") == "1"
        perf_sparse_gather_seconds = 0.0
        perf_cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            "obs_h2d": [],
            "actor_step": [],
            "critic_step": [],
        }

        def cuda_event_pair(name: str):
            if not (perf_detail and self.device.type == "cuda"):
                return nullcontext()

            class _CudaTimer:
                def __enter__(timer_self):
                    timer_self.start = torch.cuda.Event(enable_timing=True)
                    timer_self.end = torch.cuda.Event(enable_timing=True)
                    timer_self.start.record()
                    return timer_self

                def __exit__(timer_self, exc_type, exc, tb):
                    timer_self.end.record()
                    perf_cuda_events[name].append((timer_self.start, timer_self.end))

            return _CudaTimer()

        lazy_obs = getattr(rollout.obs, "gather_minibatch", None)
        if lazy_obs is not None:
            obs_flat = None
            observed_batch = int(getattr(rollout.obs, "batch_size", -1))
            if observed_batch != B:
                raise RuntimeError(
                    f"lazy rollout observations B={observed_batch} expected {B}"
                )
        else:
            obs_flat = {k: v for k, v in rollout.obs.items() if not k.startswith("_")}
        action_bounds = {
            "types": N_INTENT,
            "dirs": N_DIR,
            "targets": MAX_TARGETS,
            "amounts": N_AMOUNT,
            "body_counts": 51,
            "body_order": N_BODY_PART,
            "construction_types": N_CONSTRUCTION_TYPE,
            "construction_tiles": N_CONSTRUCTION_TILE,
        }
        for name, upper in action_bounds.items():
            values = rollout.actions.get(name)
            if values is None:
                continue
            lo = int(values.min().item())
            hi = int(values.max().item())
            if lo < 0 or hi >= upper:
                raise RuntimeError(
                    f"rollout action {name} outside [0,{upper}): min={lo} max={hi}"
                )
        if obs_flat is not None:
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
        next_indices_all = (
            rollout.next_indices.reshape(B).long()
            if rollout.next_indices is not None else torch.arange(B)
        )
        nextlat_valid_all = (
            rollout.nextlat_valid.reshape(B).bool()
            if rollout.nextlat_valid is not None else torch.zeros(B, dtype=torch.bool)
        )
        if next_indices_all.numel() != B or nextlat_valid_all.numel() != B:
            raise RuntimeError("NextLat rollout pairing must have exactly B rows")
        if next_indices_all.numel() and (
            int(next_indices_all.min()) < 0 or int(next_indices_all.max()) >= B
        ):
            raise RuntimeError("NextLat next-state index outside rollout batch")
        stats["nextlat_valid_fraction"] = float(nextlat_valid_all.float().mean())

        target_diag = self.critic.support.target_diagnostics(ret_all)
        stats["value_target_min"] = float(target_diag["target_min"].item())
        stats["value_target_max"] = float(target_diag["target_max"].item())
        stats["value_target_overflow_fraction"] = float(
            target_diag["overflow_fraction"].item()
        )
        stats["value_target_saturation_fraction"] = float(
            target_diag["saturation_fraction"].item()
        )
        self.critic.support.validate_targets(ret_all)

        # One stable normalization contract for the rollout. Re-normalizing each
        # shuffled minibatch changes rare economy advantages by batch composition.
        if adv_all.numel() >= 2:
            adv_all = (adv_all - adv_all.mean()) / (adv_all.std(unbiased=False) + 1e-8)
        else:
            adv_all = adv_all * 0

        # Compact rollout tables are reused by every epoch.  Upload each once,
        # then gather minibatches on-device.  Observations remain in the sparse
        # host page store and are materialized per minibatch below.
        act_device = {
            key: value.to(self.device, non_blocking=self.device.type == "cuda")
            for key, value in act_flat.items()
        }
        old_lp_device = old_lp.to(
            self.device, non_blocking=self.device.type == "cuda",
        )
        adv_device = adv_all.to(
            self.device, non_blocking=self.device.type == "cuda",
        )
        ret_device = ret_all.to(
            self.device, non_blocking=self.device.type == "cuda",
        )
        next_indices_device = next_indices_all.to(
            self.device, non_blocking=self.device.type == "cuda",
        )
        nextlat_valid_device = nextlat_valid_all.to(
            self.device, non_blocking=self.device.type == "cuda",
        )

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

        if (
            rollout.actor_latent is not None
            and rollout.critic_latent is not None
            and rollout.critic_logits is not None
        ):
            target_actor_all = rollout.actor_latent.reshape(B, -1).to(
                self.device, non_blocking=self.device.type == "cuda",
            )
            target_critic_all = rollout.critic_latent.reshape(B, -1).to(
                self.device, non_blocking=self.device.type == "cuda",
            )
            target_value_logits_all = rollout.critic_logits.reshape(B, -1).to(
                self.device, non_blocking=self.device.type == "cuda",
            )
        else:
            target_actor_all, target_critic_all, target_value_logits_all = (
                self.nextlat_rollout_targets(rollout.obs, batch_size=mb)
            )

        early_stop = False
        stopped_epoch = -1
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

                # Stream only this mb to GPU; promote uint8→float32 here.
                self._pin_hold.clear()
                from .vec_env import promote_obs_device

                gather_started = time.perf_counter()
                host_obs_mb = (
                    lazy_obs(inds)
                    if lazy_obs is not None
                    else {
                        k: v[inds]
                        for k, v in obs_flat.items()
                        if not k.startswith("_")
                    }
                )
                perf_sparse_gather_seconds += time.perf_counter() - gather_started
                with cuda_event_pair("obs_h2d"):
                    obs_mb = promote_obs_device(
                        host_obs_mb,
                        self.device,
                        pin_hold=self._pin_hold,
                        non_blocking=pin,
                    )

                inds_device = inds.to(self.device, non_blocking=pin)
                act_mb = {
                    key: value.index_select(0, inds_device)
                    for key, value in act_device.items()
                }
                old_lp_mb = old_lp_device.index_select(0, inds_device)
                adv = adv_device.index_select(0, inds_device)
                ret = ret_device.index_select(0, inds_device)
                next_inds_device = next_indices_device.index_select(0, inds_device)
                nextlat_valid = nextlat_valid_device.index_select(0, inds_device)

                if n_real < inds.numel():
                    adv[n_real:] = 0
                    nextlat_valid[n_real:] = False

                next_actor_latent = target_actor_all.index_select(0, next_inds_device)
                next_critic_latent = target_critic_all.index_select(0, next_inds_device)
                next_critic_logits = target_value_logits_all.index_select(
                    0, next_inds_device,
                )

                def _real(t: Tensor) -> Tensor:
                    return t[:n_real] if n_real < t.shape[0] else t

                policy_loss = torch.zeros((), device=self.device)
                entropy = torch.zeros((), device=self.device)
                approx_kl = torch.zeros((), device=self.device)
                old_approx_kl = torch.zeros((), device=self.device)
                clipfrac = torch.zeros((), device=self.device)
                gn_a = 0.0
                nextlat_actor_mse = torch.zeros((), device=self.device)
                nextlat_actor_delta_rms = torch.zeros((), device=self.device)
                nextlat_actor_state_rms = torch.zeros((), device=self.device)

                # --- Actor update (activations freed before critic) ---
                if not critic_only:
                    actor_timer = cuda_event_pair("actor_step")
                    actor_timer.__enter__()
                    with self._autocast():
                        out = self.actor(obs_mb, action=act_mb)
                    new_lp = _real(out.actor_logprob.float())
                    actor_ent = _real(out.actor_entropy.float())
                    old_lp_r = _real(old_lp_mb)[:, : new_lp.shape[1]]
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
                    with self._autocast():
                        predicted_actor_latent = self.actor.predict_next_latent(
                            out.state_latent, obs_mb, act_mb,
                        )
                        nextlat_actor_mse = _masked_latent_loss(
                            predicted_actor_latent, next_actor_latent, nextlat_valid,
                        )
                    nextlat_actor_delta_rms = (
                        predicted_actor_latent.float() - out.state_latent.float()
                    ).square().mean().sqrt()
                    nextlat_actor_state_rms = out.state_latent.float().square().mean().sqrt()

                    self.actor_opt.zero_grad(set_to_none=True)
                    (
                        policy_loss
                        - self.entropy_coef * entropy
                        + self.nextlat_actor_mse_coef * nextlat_actor_mse
                    ).backward()
                    gn_a = nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    self.actor_opt.step()
                    actor_timer.__exit__(None, None, None)
                    # Drop actor activations before critic forward (VRAM peak).
                    del out, new_lp, logratio, ratio, surr1, surr2, adv_r, old_lp_r, actor_adv, actor_ent, live, predicted_actor_latent

                # --- Critic update ---
                critic_timer = cuda_event_pair("critic_step")
                critic_timer.__enter__()
                critic_mod = self.critic
                ret_r = _real(ret)
                with self._autocast():
                    logits, critic_latent = self.critic(
                        obs_mb, return_logits=True, return_latent=True,
                    )
                logits_r = _real(logits.float())
                value_loss = critic_mod.support.cross_entropy(
                    logits_r, ret_r, validate=False
                ).mean()
                with self._autocast():
                    predicted_critic_latent = critic_mod.predict_next_latent(
                        critic_latent, obs_mb, act_mb,
                    )
                    nextlat_critic_mse = _masked_latent_loss(
                        predicted_critic_latent, next_critic_latent, nextlat_valid,
                    )
                predicted_value_logits = critic_mod.detached_value_logits(
                    predicted_critic_latent,
                )
                nextlat_critic_kl = _masked_categorical_kl(
                    predicted_value_logits, next_critic_logits, nextlat_valid,
                )
                nextlat_critic_delta_rms = (
                    predicted_critic_latent.float() - critic_latent.float()
                ).square().mean().sqrt()
                nextlat_critic_state_rms = critic_latent.float().square().mean().sqrt()
                del logits, logits_r, critic_latent, predicted_value_logits

                self.critic_opt.zero_grad(set_to_none=True)
                (
                    self.value_coef * value_loss
                    + self.nextlat_critic_mse_coef * nextlat_critic_mse
                    + self.nextlat_critic_kl_coef * nextlat_critic_kl
                ).backward()
                gn_c = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.critic_max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.critic_opt.step()
                critic_timer.__exit__(None, None, None)

                # Free mb device tensors before next slice
                del obs_mb, act_mb, old_lp_mb, adv, ret, ret_r
                del predicted_critic_latent, next_actor_latent, next_critic_latent
                del next_critic_logits, nextlat_valid

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
                        "nextlat_actor_mse": nextlat_actor_mse.detach(),
                        "nextlat_critic_mse": nextlat_critic_mse.detach(),
                        "nextlat_critic_kl": nextlat_critic_kl.detach(),
                        "nextlat_actor_delta_rms": nextlat_actor_delta_rms.detach(),
                        "nextlat_actor_state_rms": nextlat_actor_state_rms.detach(),
                        "nextlat_critic_delta_rms": nextlat_critic_delta_rms.detach(),
                        "nextlat_critic_state_rms": nextlat_critic_state_rms.detach(),
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
                    acc["nextlat_actor_mse"] = acc["nextlat_actor_mse"] + nextlat_actor_mse.detach()
                    acc["nextlat_critic_mse"] = acc["nextlat_critic_mse"] + nextlat_critic_mse.detach()
                    acc["nextlat_critic_kl"] = acc["nextlat_critic_kl"] + nextlat_critic_kl.detach()
                    acc["nextlat_actor_delta_rms"] = acc["nextlat_actor_delta_rms"] + nextlat_actor_delta_rms.detach()
                    acc["nextlat_actor_state_rms"] = acc["nextlat_actor_state_rms"] + nextlat_actor_state_rms.detach()
                    acc["nextlat_critic_delta_rms"] = acc["nextlat_critic_delta_rms"] + nextlat_critic_delta_rms.detach()
                    acc["nextlat_critic_state_rms"] = acc["nextlat_critic_state_rms"] + nextlat_critic_state_rms.detach()
                n_updates += 1

                # KL early-stop still needs a sync for control flow when enabled.
                stop_now = False
                if (not critic_only) and self.target_kl > 0:
                    stop_now = bool((approx_kl.detach() > self.target_kl).item())
                del policy_loss, value_loss, entropy, approx_kl, old_approx_kl, clipfrac
                del nextlat_actor_mse, nextlat_critic_mse, nextlat_critic_kl
                del nextlat_actor_delta_rms, nextlat_actor_state_rms
                del nextlat_critic_delta_rms, nextlat_critic_state_rms

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
        stats["critic_max_grad_norm"] = float(self.critic_max_grad_norm)
        stats["optimizer_steps"] = float(n_updates)
        if perf_detail:
            stats["perf_sparse_gather_seconds"] = perf_sparse_gather_seconds
            if self.device.type == "cuda":
                # The scalar stats flush above has already synchronized the
                # queued update. Resolve events once, never inside a minibatch.
                for name, pairs in perf_cuda_events.items():
                    stats[f"perf_{name}_seconds"] = 1e-3 * sum(
                        start.elapsed_time(end) for start, end in pairs
                    )
        return stats
