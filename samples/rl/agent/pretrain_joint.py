#!/usr/bin/env python3
"""Corpus-backed full-lifecycle joint BC ∥ critic pretrain.

Load a validated immutable corpus, then optimize both heads over globally
shuffled, stage-balanced rows. The corpus owns collection and return semantics;
this trainer never regenerates its train, holdout, spawn, or TI rows:

  L_actor = L_BC + actor_nextlat_coef · L_actor_nextlat
  L_critic = value_coef · L_V + critic_nextlat_coef · L_critic_nextlat
             + critic_kl_coef · L_probe_KL

Lifecycle and independently shuffled temporal rows share each actor/critic
optimizer step; NextLat never advances either model on an auxiliary-only step.

The scripted planner supplies complete actor labels. TI contributes exact
representable actor factors plus independent train/holdout critic splits.
Rare `createConstructionSite` and `claimController` actors are reindexed from
the same lifecycle rows into an intent-balanced actor-only auxiliary lane. It
imitates intent+structure type+tile for construction and intent+target for
claims, so a construction demonstration is cloned as a structure at a position.

  python3 -m samples.rl.agent.pretrain_joint \\
    --corpus samples/rl/runs/pretrain-corpora/<corpus-sha256> \\
    --global-epochs 16 --device cuda \\
    --save samples/rl/runs/joint_pretrain_v4.pt

Do NOT queue via mlq until readiness gates (docs) pass and user authorizes.
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        source_signature,
        validate_artifact,
    )
    from samples.rl.agent.actions_util import safe_bc_nll
    from samples.rl.agent.constants import (
        ACTOR_FEATURE_INDEX, BODY_PART_COSTS, INTENT_TYPES, MAX_ACTORS,
        MAX_BODY_PARTS, MAX_ROOM_ENERGY, N_BODY_PART,
        PPO_CFG, SCHEMA, VALUE_CFG,
    )
    from samples.rl.agent.model import Actor, Critic, count_params
    from samples.rl.agent.muon import (
        MUON_MOMENTUM_MAX, MUON_MOMENTUM_MIN, MUON_MOMENTUM_WARMUP_STEPS,
        MUON_WEIGHT_DECAY, OPTIMIZER_KIND, PRETRAIN_MUON_LR,
        HybridMuonAdamW, optimizer_parameter_counts,
    )
    from samples.rl.agent.ti_intents import translate_ti_intents
    from samples.rl.agent.vec_env import (
        VecScreepsEnv, _compact_entity_prefixes, promote_obs_device,
    )
except ImportError:
    from agent.artifacts import (
        artifact_meta, atomic_torch_save,
        load_full_state, source_signature, validate_artifact,
    )
    from agent.actions_util import safe_bc_nll
    from agent.constants import (
        ACTOR_FEATURE_INDEX, BODY_PART_COSTS, INTENT_TYPES, MAX_ACTORS,
        MAX_BODY_PARTS, MAX_ROOM_ENERGY, N_BODY_PART,
        PPO_CFG, SCHEMA, VALUE_CFG,
    )
    from agent.model import Actor, Critic, count_params
    from agent.muon import (
        MUON_MOMENTUM_MAX, MUON_MOMENTUM_MIN, MUON_MOMENTUM_WARMUP_STEPS,
        MUON_WEIGHT_DECAY, OPTIMIZER_KIND, PRETRAIN_MUON_LR,
        HybridMuonAdamW, optimizer_parameter_counts,
    )
    from agent.ti_intents import translate_ti_intents
    from agent.vec_env import (
        VecScreepsEnv, _compact_entity_prefixes, promote_obs_device,
    )


SPAWN_CURRICULA = (
    "spawn_flexible_300",
    "spawn_miner_450",
    "spawn_hauler_3000",
    "spawn_builder_650",
    "spawn_upgrader_550",
    "spawn_claimer_650",
)
SPAWN_REPLAY_PER_STRATUM = 16
# Spawn supervision is audited in buckets of the two quantities a body decision
# is made against: the energy the room can spend right now, and the part count
# that energy buys.  Every bucket count in this file is measured from the
# teacher's own decisions; none of them encodes an expected body.
SPAWN_BUDGET_BUCKETS = ("le300", "301_549", "550_649", "ge650")
SPAWN_LENGTH_BUCKETS = ("le6", "7_15", "ge16")
# Body length the simulated economy cannot fund.  The engine charges at least
# 50 energy per part, so a 16-part body needs 800+ energy in a single spawn,
# which is an RCL3 spawn plus ten extensions plus a full refill; the curricula
# this pipeline trains on settle around RCL1-RCL2, and The International spends
# its budget on expensive work parts rather than on part count.  The bucket is
# therefore recorded as a named, non-fatal coverage gap
# (`*_length_unreached_<bucket>`) instead of being deleted, so a higher-RCL
# curriculum that starts filling it can promote it back to a fatal requirement.
SPAWN_LENGTH_UNREACHED_BUCKETS = ("ge16",)
SPAWN_LENGTH_REQUIRED_BUCKETS = tuple(
    bucket for bucket in SPAWN_LENGTH_BUCKETS
    if bucket not in SPAWN_LENGTH_UNREACHED_BUCKETS
)
TEACHER_REPLAY_PER_STRATUM = 64
_SPAWN_ENERGY_FEATURE = ACTOR_FEATURE_INDEX["roomEnergyAvailable"]
RARE_ACTOR_INTENTS = ("createConstructionSite", "claimController")
# Factors the rare lane imitates and therefore requires a teacher to have
# labelled: the intent plus its structure type, or the intent plus its exact
# controller pointer.  A factor-masked row naming the intent without them
# carries no rare demonstration at all.
RARE_INTENT_REQUIRED_FACTORS = {
    "createConstructionSite": (0, 4),
    "claimController": (0, 2),
}
MAX_CLOSED_LOOP_INVALID_FRAC = 0.01
OUTPOST_LATE_WINDOW_STEPS = 1_000


SpawnReplayRow = tuple[
    str, int, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor,
]


@dataclass
class LifecycleSample:
    """One compact teacher row plus the factors its labels actually supervise.

    The teacher is The International, which is exactly representable only for a
    subset of factors per tick, so every row carries its own eligibility mask of
    shape ``[1, actors, 6 + 2 * N_BODY_PART]`` in ``factor_logprob`` order.
    """

    stratum: str
    timestep: int
    env_index: int
    obs: dict[str, torch.Tensor]
    action: dict[str, torch.Tensor]
    eligible: torch.Tensor


@dataclass
class CriticSample:
    """One compact observation with a full-trajectory return lookup."""

    stratum: str
    timestep: int
    env_index: int
    obs: dict[str, torch.Tensor]


@dataclass
class TemporalSample:
    """One causal scripted transition retained directly at collection time."""

    stratum: str
    timestep: int
    env_index: int
    obs: dict[str, torch.Tensor]
    action: dict[str, torch.Tensor]
    counterfactual_action: dict[str, torch.Tensor] | None
    next_obs: dict[str, torch.Tensor]


TI_ACTOR_TRAINING = "one_time_initialization_before_joint_epochs"
JOINT_OBJECTIVE_AUTHORITY = (
    "lifecycle_primary_fused_bc_value_nextlat_rare_spawn_tile_v5"
)
ACTOR_AUXILIARY_SCHEDULE = (
    "one_balanced_source_epoch_joint_collision_free_quantiles_v2"
)
NEXTLAT_ACTION_ABLATION = "same_state_whole_joint_canonical_none_on_the_fly"
NEXTLAT_ACTION_POOLING = "issued_sum_sqrt_count_v1"


def _should_run_ti_initialization(*, global_epoch: int, resume: Path | None) -> bool:
    """TI warm starts belong only to a fresh optimizer/model trajectory."""
    return int(global_epoch) == 0 and resume is None


def _continuation_mismatches(
    saved: dict[str, Any], expected: dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return every exact continuation-contract mismatch."""
    return {
        key: (saved.get(key), value)
        for key, value in expected.items()
        if saved.get(key) != value
    }


def _lifecycle_samples(rows: list[dict]) -> list[LifecycleSample]:
    return [
        LifecycleSample(
            stratum=str(row["stratum"]),
            timestep=int(row["timestep"]),
            env_index=int(row["env_index"]),
            obs=row["obs"],
            action=row["action"],
            eligible=row["eligible"],
        )
        for row in rows
    ]


def _critic_samples(rows: list[dict]) -> list[CriticSample]:
    return [
        CriticSample(
            stratum=str(row["stratum"]),
            timestep=int(row["timestep"]),
            env_index=int(row["env_index"]),
            obs=row["obs"],
        )
        for row in rows
    ]


def _temporal_samples(rows: list[dict]) -> list[TemporalSample]:
    samples: list[TemporalSample] = []
    for row in rows:
        if bool(row["terminated"]) or bool(row["truncated"]):
            raise ValueError("episode-ending row reached temporal training")
        samples.append(TemporalSample(
            stratum=str(row["stratum"]),
            timestep=int(row["timestep"]),
            env_index=int(row["env_index"]),
            obs=row["obs"],
            action=row["action"],
            counterfactual_action=row["counterfactual_action"],
            next_obs=row["next_obs"],
        ))
    return samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-lifecycle joint BC+critic pretrain (current schema)",
    )
    p.add_argument(
        "--corpus", type=Path, required=True,
        help="validated immutable corpus artifact; training never recollects it",
    )
    p.add_argument(
        "--global-epochs", type=int, default=16,
        help="globally shuffled epochs over the stage-balanced lifecycle corpus",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="training device (the loaded corpus remains host-side)",
    )
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=float(PPO_CFG["lr"]) * 2)
    p.add_argument(
        "--ti-critic-pretrain-epochs", type=int, default=1,
        help=(
            "one-time TI value initialization epochs before actor-aligned "
            "scripted critic training; never repeated between lifecycle epochs"
        ),
    )
    p.add_argument("--muon-lr", type=float, default=PRETRAIN_MUON_LR)
    p.add_argument(
        "--muon-weight-decay", type=float, default=MUON_WEIGHT_DECAY,
        help="cautious decay on hidden Muon matrices; AdamW auxiliaries use zero decay",
    )
    p.add_argument(
        "--muon-momentum-warmup-steps", type=int,
        default=MUON_MOMENTUM_WARMUP_STEPS,
    )
    p.add_argument(
        "--value-coef",
        type=float,
        default=1.0,
        help="weight on L_V relative to L_BC (both optimized each mb)",
    )
    p.add_argument(
        "--nextlat-actor-coef", type=float, default=1.0,
        help="pretraining-only actor latent smooth-L1 coefficient",
    )
    p.add_argument(
        "--nextlat-critic-coef", type=float, default=1.0,
        help="pretraining-only critic latent smooth-L1 coefficient",
    )
    p.add_argument(
        "--nextlat-critic-kl-coef", type=float, default=0.1,
        help="pretraining-only detached value-head probe KL coefficient",
    )
    p.add_argument(
        "--min-nextlat-relative-gap", type=float, default=0.01,
        help=(
            "minimum relative identity improvement and same-state legal "
            "counterfactual penalty"
        ),
    )
    p.add_argument(
        "--min-nextlat-counterfactual-rows", type=int, default=128,
        help=(
            "minimum held-out rows with at least one issued command for the "
            "same-state whole-joint canonical-none ablation"
        ),
    )
    p.add_argument(
        "--max-aux-lifecycle-nll-ratio", type=float, default=1.1,
        help=(
            "maximum final lifecycle train NLL divided by the NLL immediately "
            "after the final joint lifecycle+NextLat pass"
        ),
    )
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--save", type=Path, default=_RL_ROOT / "runs" / "joint_pretrain_v4.pt")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument(
        "--critic-init", type=Path, default=None,
        help="initialize the raw-H+C critic from a qualified TI critic artifact",
    )
    p.add_argument("--closed-loop-steps", type=int, default=20000)
    p.add_argument(
        "--evaluation-seed-offset", type=int, default=20_000,
        help="third, never-collected seed family used for learned closed-loop gates",
    )
    p.add_argument("--min-closed-loop-rate", type=float, default=0.1)
    p.add_argument("--min-validation-ev", type=float, default=0.01)
    p.add_argument(
        "--min-spawn-replay-accuracy", type=float, default=0.8,
        help=(
            "minimum final positive intent, exact body, and per-body-stratum "
            "accuracy on the balanced spawn replay"
        ),
    )
    p.add_argument(
        "--max-rare-intent-nll", type=float, default=1.0,
        help=(
            "maximum actor-balanced semantic-factor NLL for every required rare "
            "intent on both lifecycle train and holdout rows"
        ),
    )
    p.add_argument(
        "--min-rare-intent-accuracy", type=float, default=0.8,
        help=(
            "minimum deterministic exact semantic-factor accuracy for every "
            "required rare intent on both lifecycle train and holdout rows"
        ),
    )
    p.add_argument(
        "--min-rare-intent-rows", type=int, default=32,
        help=(
            "minimum independently retained train and holdout actors for every "
            "required rare intent"
        ),
    )
    p.add_argument("--min-closed-loop-creeps", type=int, default=24)
    p.add_argument("--min-closed-loop-claims", type=int, default=1)
    p.add_argument(
        "--min-outpost-closed-loop-success-rate", type=float, default=0.5,
        help=(
            "fraction of unseen seed_outpost environments that must remotely "
            "harvest, return energy home, and staff a productive neutral "
            "outpost without claiming or owning it"
        ),
    )
    p.add_argument("--min-teacher-delivery", type=float, default=1.0)
    p.add_argument("--min-teacher-build", type=float, default=1.0)
    p.add_argument("--min-teacher-claims", type=int, default=1)
    p.add_argument("--min-teacher-creeps", type=int, default=24)
    p.add_argument("--logdir", type=Path, default=None)
    p.add_argument(
        "--compile", action="store_true",
        help="opt into experimental torch.compile graphs (eager is trusted)",
    )
    p.add_argument("--no-compile", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--seed", type=int, default=3)
    return p.parse_args(argv)


@torch.inference_mode()
def _validate_closed_loop(
    actor: Actor,
    envs: VecScreepsEnv,
    *,
    steps: int,
    device: torch.device,
    deterministic: bool = True,
) -> dict[str, object]:
    """Run the learned actor in fresh environments.

    `deterministic` selects the decode. Greedy decoding is the qualification
    setting, but it cannot see behaviour that survives only as low-probability
    policy mass, construction being the clearest case, so a sampled run is the
    comparable measurement when behaviour counts are the question.
    """
    obs = envs.reset()
    actor.eval()
    total_skill = 0.0
    total_reward = 0.0
    total_harvest = 0.0
    total_control = 0.0
    invalid = 0
    issued = 0
    max_creeps = 0
    total_claims = 0
    by_curriculum: dict[str, dict[str, object]] = {}
    by_env: dict[str, dict[str, object]] = {}
    by_intent: dict[str, dict[str, int]] = {}
    late_window_size = min(OUTPOST_LATE_WINDOW_STEPS, max(1, steps // 5))
    for timestep in range(max(1, steps)):
        late_window = timestep >= max(1, steps) - late_window_size
        out = actor(obs, deterministic=deterministic)
        obs, reward, _done, infos = envs.step({
            "types": out.types,
            "dirs": out.dirs,
            "targets": out.targets,
            "amounts": out.amounts,
            "body_counts": out.body_counts,
            "body_order": out.body_order,
            "construction_types": out.construction_types,
            "construction_tiles": out.construction_tiles,
        })
        for env_index, info in enumerate(infos):
            info = info or {}
            harvest = float(info.get("harvestDelta") or 0)
            control = float(info.get("controlDelta") or 0)
            delivered = float(info.get("transferDelta") or 0)
            built = float(info.get("buildDelta") or 0)
            claims = int(info.get("claimDelta") or 0)
            remote_harvest = float(info.get("remoteHarvestDelta") or 0)
            remote_home_delivery = float(info.get("remoteHomeDeliveryDelta") or 0)
            remote_staffed_peak = float(info.get("remoteRoomsStaffedPeak") or 0)
            remote_productive_peak = float(
                info.get("remoteProductiveCreepsPeak") or 0
            )
            remote_staffed = float(info.get("remoteRoomsStaffed") or 0)
            remote_productive = float(info.get("remoteProductiveCreeps") or 0)
            remote_owned_peak = float(info.get("remoteOwnedRoomsPeak") or 0)
            neutral_outposts = float(info.get("neutralOutpostRooms") or 0)
            creeps = int(info.get("creeps") or 0)
            step_reward = float(reward[env_index])
            total_skill += harvest + control
            total_reward += step_reward
            total_harvest += harvest
            total_control += control
            max_creeps = max(max_creeps, creeps)
            total_claims += claims
            stage = str(info.get("curriculum") or "unknown")
            stage_metrics = by_curriculum.setdefault(stage, {
                "transitions": 0.0,
                "skill": 0.0,
                "reward": 0.0,
                "harvest": 0.0,
                "control": 0.0,
                "delivery": 0.0,
                "build": 0.0,
                "claims": 0.0,
                "remote_harvest": 0.0,
                "remote_home_delivery": 0.0,
                "remote_staffed_peak": 0.0,
                "remote_productive_peak": 0.0,
                "remote_owned_peak": 0.0,
                "neutral_outposts": 0.0,
                "late_transitions": 0.0,
                "late_remote_harvest": 0.0,
                "late_remote_home_delivery": 0.0,
                "late_remote_staffed_ticks": 0.0,
                "late_remote_productive_ticks": 0.0,
                "late_spawn_success": 0.0,
                "max_creeps": 0.0,
                "issued": 0.0,
                "invalid": 0.0,
            })
            stage_metrics["transitions"] += 1.0
            stage_metrics["skill"] += harvest + control
            stage_metrics["reward"] += step_reward
            stage_metrics["harvest"] += harvest
            stage_metrics["control"] += control
            stage_metrics["delivery"] += delivered
            stage_metrics["build"] += built
            stage_metrics["claims"] += claims
            stage_metrics["remote_harvest"] += remote_harvest
            stage_metrics["remote_home_delivery"] += remote_home_delivery
            if late_window:
                stage_metrics["late_transitions"] += 1.0
                stage_metrics["late_remote_harvest"] += remote_harvest
                stage_metrics["late_remote_home_delivery"] += remote_home_delivery
                stage_metrics["late_remote_staffed_ticks"] += float(remote_staffed > 0)
                stage_metrics["late_remote_productive_ticks"] += float(
                    remote_productive > 0
                )
                stage_metrics["late_spawn_success"] += float(
                    info.get("spawnSuccess") or 0
                )
            stage_metrics["remote_staffed_peak"] = max(
                stage_metrics["remote_staffed_peak"], remote_staffed_peak,
            )
            stage_metrics["remote_productive_peak"] = max(
                stage_metrics["remote_productive_peak"], remote_productive_peak,
            )
            stage_metrics["remote_owned_peak"] = max(
                stage_metrics["remote_owned_peak"], remote_owned_peak,
            )
            stage_metrics["neutral_outposts"] = max(
                stage_metrics["neutral_outposts"], neutral_outposts,
            )
            stage_metrics["max_creeps"] = max(stage_metrics["max_creeps"], float(creeps))
            env_metrics = by_env.setdefault(str(env_index), {
                "curriculum": stage,
                "seed": int(envs.envs[env_index].seed),
                "transitions": 0.0,
                "skill": 0.0,
                "reward": 0.0,
                "harvest": 0.0,
                "control": 0.0,
                "delivery": 0.0,
                "build": 0.0,
                "claims": 0.0,
                "remote_harvest": 0.0,
                "remote_home_delivery": 0.0,
                "remote_staffed_peak": 0.0,
                "remote_productive_peak": 0.0,
                "remote_owned_peak": 0.0,
                "neutral_outposts": 0.0,
                "late_transitions": 0.0,
                "late_remote_harvest": 0.0,
                "late_remote_home_delivery": 0.0,
                "late_remote_staffed_ticks": 0.0,
                "late_remote_productive_ticks": 0.0,
                "late_spawn_success": 0.0,
                "max_creeps": 0.0,
                "issued": 0.0,
                "invalid": 0.0,
            })
            env_metrics["transitions"] += 1.0
            env_metrics["skill"] += harvest + control
            env_metrics["reward"] += step_reward
            env_metrics["harvest"] += harvest
            env_metrics["control"] += control
            env_metrics["delivery"] += delivered
            env_metrics["build"] += built
            env_metrics["claims"] += claims
            env_metrics["remote_harvest"] += remote_harvest
            env_metrics["remote_home_delivery"] += remote_home_delivery
            if late_window:
                env_metrics["late_transitions"] += 1.0
                env_metrics["late_remote_harvest"] += remote_harvest
                env_metrics["late_remote_home_delivery"] += remote_home_delivery
                env_metrics["late_remote_staffed_ticks"] += float(remote_staffed > 0)
                env_metrics["late_remote_productive_ticks"] += float(
                    remote_productive > 0
                )
                env_metrics["late_spawn_success"] += float(
                    info.get("spawnSuccess") or 0
                )
            env_metrics["remote_staffed_peak"] = max(
                env_metrics["remote_staffed_peak"], remote_staffed_peak,
            )
            env_metrics["remote_productive_peak"] = max(
                env_metrics["remote_productive_peak"], remote_productive_peak,
            )
            env_metrics["remote_owned_peak"] = max(
                env_metrics["remote_owned_peak"], remote_owned_peak,
            )
            env_metrics["neutral_outposts"] = max(
                env_metrics["neutral_outposts"], neutral_outposts,
            )
            env_metrics["max_creeps"] = max(env_metrics["max_creeps"], float(creeps))
            stage_metrics["spawn_success"] = stage_metrics.get("spawn_success", 0.0) + float(
                info.get("spawnSuccess") or 0
            )
            stage_intents = stage_metrics.setdefault("intent_by_type", {})
            for intent_name, intent_counts in (info.get("intentByType") or {}).items():
                if not isinstance(intent_counts, dict):
                    continue
                intent_issued = int(intent_counts.get("issued") or 0)
                intent_invalid = int(intent_counts.get("invalid") or 0)
                aggregate = by_intent.setdefault(
                    str(intent_name), {"issued": 0, "invalid": 0},
                )
                aggregate["issued"] += intent_issued
                aggregate["invalid"] += intent_invalid
                stage_aggregate = stage_intents.setdefault(
                    str(intent_name), {"issued": 0, "invalid": 0},
                )
                stage_aggregate["issued"] += intent_issued
                stage_aggregate["invalid"] += intent_invalid
            for result in info.get("intentResults") or ():
                issued += 1
                stage_metrics["issued"] += 1.0
                env_metrics["issued"] += 1.0
                if result.get("type") == "spawnCreep" and int(result.get("code", -1)) == C_OK:
                    parts = [int(part) for part in result.get("spawnBodyParts", ())]
                    body_cost = float(sum(
                        BODY_PART_COSTS[part]
                        for part in parts
                        if 0 <= part < len(BODY_PART_COSTS)
                    ))
                    if "spawn_body_parts" not in stage_metrics:
                        stage_metrics["spawn_body_length"] = float(
                            result.get("spawnBodyLength", 0)
                        )
                        stage_metrics["spawn_body_parts"] = parts
                        stage_metrics["spawn_body_cost"] = body_cost
                    stage_metrics.setdefault("spawn_body_parts_all", []).append(parts)
                    stage_metrics["spawn_body_cost_total"] = (
                        float(stage_metrics.get("spawn_body_cost_total", 0.0)) + body_cost
                    )
                if int(result.get("code", -1)) not in (C_OK, -2, -4, -11):
                    invalid += 1
                    stage_metrics["invalid"] += 1.0
                    env_metrics["invalid"] += 1.0
    for stage_metrics in by_curriculum.values():
        transitions = max(1.0, stage_metrics["transitions"])
        stage_metrics["skill_rate"] = stage_metrics["skill"] / transitions
        stage_metrics["reward_rate"] = stage_metrics["reward"] / transitions
        stage_metrics["harvest_rate"] = stage_metrics["harvest"] / transitions
        stage_metrics["control_rate"] = stage_metrics["control"] / transitions
        stage_metrics["invalid_frac"] = (
            stage_metrics["invalid"] / max(1.0, stage_metrics["issued"])
        )
    for env_metrics in by_env.values():
        transitions = max(1.0, env_metrics["transitions"])
        env_metrics["skill_rate"] = env_metrics["skill"] / transitions
        env_metrics["reward_rate"] = env_metrics["reward"] / transitions
        env_metrics["harvest_rate"] = env_metrics["harvest"] / transitions
        env_metrics["control_rate"] = env_metrics["control"] / transitions
        env_metrics["invalid_frac"] = (
            env_metrics["invalid"] / max(1.0, env_metrics["issued"])
        )
    denominator = max(1, steps * envs.n)
    return {
        "closed_loop_skill_rate": total_skill / denominator,
        "closed_loop_reward_rate": total_reward / denominator,
        "closed_loop_harvest_rate": total_harvest / denominator,
        "closed_loop_control_rate": total_control / denominator,
        "closed_loop_invalid_frac": invalid / max(1, issued),
        "closed_loop_max_creeps": float(max_creeps),
        "closed_loop_claims": float(total_claims),
        "closed_loop_intent_by_type": by_intent,
        "closed_loop_by_curriculum": by_curriculum,
        "closed_loop_by_env": by_env,
    }


# Screeps OK is numerically zero; named locally to keep validation independent of JS imports.
C_OK = 0


def _parts_to_count_order(
    parts: list[int] | tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Convert a body to counts/order and report whether ordering is exact.

    The count representation is exact for every body. Its order represents only
    contiguous type blocks, so an interleaved raw body such as CARRY,MOVE,CARRY
    must not supervise the order likelihood as though grouping preserved it.
    """
    raw_parts = [int(part) for part in parts]
    counts = torch.zeros(N_BODY_PART, dtype=torch.long)
    positive_order: list[int] = []
    seen: set[int] = set()
    for part in raw_parts:
        if not 0 <= part < N_BODY_PART:
            raise RuntimeError(f"teacher emitted out-of-range spawn part {part}")
        counts[part] += 1
        if part not in seen:
            positive_order.append(part)
            seen.add(part)
    if not positive_order or int(counts.sum()) > MAX_BODY_PARTS:
        raise RuntimeError(f"teacher emitted invalid spawn body length={int(counts.sum())}")
    full_order = positive_order + [part for part in range(N_BODY_PART) if part not in seen]
    grouped = [
        part for part in positive_order for _ in range(int(counts[part]))
    ]
    return counts, torch.tensor(full_order, dtype=torch.long), grouped == raw_parts


def _body_label_factor_mask(body_counts: torch.Tensor, order_exact: bool) -> torch.Tensor:
    """Eligibility for an exact composition and conditionally exact body order."""
    eligible = torch.zeros(6 + 2 * N_BODY_PART, dtype=torch.bool)
    eligible[6 : 6 + N_BODY_PART] = True
    if order_exact:
        positive_types = int((body_counts > 0).sum())
        order_start = 6 + N_BODY_PART
        eligible[order_start : order_start + max(1, positive_types - 1)] = True
    return eligible


def _spawn_budget_bucket(budget: int) -> str:
    """Bucket the energy a spawn decision could actually spend."""
    if budget <= 300:
        return "le300"
    if budget < 550:
        return "301_549"
    if budget < 650:
        return "550_649"
    return "ge650"


def _spawn_length_bucket(length: int) -> str:
    """Bucket the part count a spawn decision bought."""
    if length <= 6:
        return "le6"
    if length <= 15:
        return "7_15"
    return "ge16"


def _record_spawn_labels(
    metrics: dict[str, float], actions: dict[str, torch.Tensor], obs: dict[str, torch.Tensor], env_index: int,
) -> None:
    """Audit body-only spawn supervision and its observed energy budget."""
    spawn_type = INTENT_TYPES.index("spawnCreep")
    types = actions["types"][env_index, :, 0]
    for actor_index in torch.nonzero(types == spawn_type, as_tuple=False).flatten().tolist():
        counts = actions["body_counts"][env_index, actor_index, 0].long()
        length = int(counts.sum())
        if not (1 <= length <= MAX_BODY_PARTS):
            raise RuntimeError(f"teacher emitted invalid spawn length={length}")
        body_cost = sum(
            BODY_PART_COSTS[part] * int(counts[part]) for part in range(N_BODY_PART)
        )
        budget = int(round(
            float(obs["actors"][env_index, actor_index, _SPAWN_ENERGY_FEATURE])
            * MAX_ROOM_ENERGY
        ))
        if body_cost > budget:
            raise RuntimeError(
                f"teacher emitted unaffordable body cost={body_cost} budget={budget}"
            )
        metrics["spawn_labels"] = metrics.get("spawn_labels", 0.0) + 1.0
        metrics["spawn_body_cost_min"] = min(
            metrics.get("spawn_body_cost_min", float(body_cost)), float(body_cost),
        )
        metrics["spawn_body_cost_max"] = max(
            metrics.get("spawn_body_cost_max", 0.0), float(body_cost),
        )
        metrics["spawn_body_length_max"] = max(
            metrics.get("spawn_body_length_max", 0.0), float(length),
        )
        length_key = f"spawn_length_{_spawn_length_bucket(length)}"
        metrics[length_key] = metrics.get(length_key, 0.0) + 1.0
        budget_key = f"spawn_budget_{_spawn_budget_bucket(budget)}"
        metrics[budget_key] = metrics.get(budget_key, 0.0) + 1.0


def _append_spawn_replay(
    replay: list[SpawnReplayRow],
    obs: dict[str, torch.Tensor],
    actions: dict[str, torch.Tensor],
    eligible: torch.Tensor,
) -> None:
    """Retain bounded positive *and wait* decisions for every available spawn.

    Positive-only replay makes the intent head learn "spawn whenever possible"
    and erases the teacher's economically essential save-up decisions.  Body
    factors remain active only on positive rows; wait rows supervise the intent
    choice and are stratified by budget, capacity, and population phase.

    ``eligible`` is the teacher's per-factor supervision mask, laid out over
    ``[batch, actors, 6 + 2 * N_BODY_PART]`` exactly like ``actions``.  The two
    row kinds supervise disjoint factors, so each is retained only when its own
    factors carry a label: a positive row needs every body-count factor, a wait
    row needs the intent factor.  Anything else would contribute zero gradient.
    """
    spawn_type = INTENT_TYPES.index("spawnCreep")
    none_type = INTENT_TYPES.index("none")
    batch = actions["types"].shape[0]
    for env_index in range(batch):
        live = obs["actor_mask"][env_index] > 0.5
        spawn_actor = obs["actors"][env_index, :, ACTOR_FEATURE_INDEX["isSpawn"]] > 0.5
        spawning = obs["actors"][env_index, :, ACTOR_FEATURE_INDEX["spawning"]] > 0.5
        decision_rows = torch.nonzero(
            live & spawn_actor & ~spawning, as_tuple=False,
        ).flatten()
        for actor_tensor in decision_rows:
            actor_index = int(actor_tensor)
            decision = int(actions["types"][env_index, actor_index, 0])
            if decision not in (none_type, spawn_type):
                continue
            budget = int(round(
                float(obs["actors"][env_index, actor_index, _SPAWN_ENERGY_FEATURE])
                * MAX_ROOM_ENERGY
            ))
            capacity = int(round(
                float(obs["actors"][env_index, actor_index, ACTOR_FEATURE_INDEX["roomEnergyCapacity"]])
                * MAX_ROOM_ENERGY
            ))
            budget_bucket = _spawn_budget_bucket(budget)
            if decision == spawn_type:
                if not bool(
                    eligible[env_index, actor_index, 6 : 6 + N_BODY_PART].all()
                ):
                    # This row exists to supervise the exact composition, so a
                    # body this ABI cannot express has nothing to teach here.
                    continue
                counts = actions["body_counts"][env_index, actor_index, 0].long()
                length = int(counts.sum())
                length_bucket = _spawn_length_bucket(length)
                body_signature = "_".join(str(int(count)) for count in counts.tolist())
                stratum = f"spawn:{budget_bucket}:{length_bucket}:{body_signature}"
            else:
                wait_kind = (
                    "waitlegal"
                    if float(obs["intent_mask"][env_index, actor_index, 0, spawn_type]) > 0.5
                    else "waitforced"
                )
                # With spawn masked, none is the only legal type and contributes
                # exactly zero gradient.  Such rows are useful telemetry but would
                # merely dilute the balanced decision loss, so replay only genuine
                # economic waits where spawning was a legal alternative.
                if wait_kind == "waitforced":
                    continue
                if not bool(eligible[env_index, actor_index, 0]):
                    # The intent is the only factor a wait row ever supervises.
                    continue
                capacity_bucket = (
                    "cap300" if capacity <= 300 else "cap301_549" if capacity < 550
                    else "cap550_649" if capacity < 650 else "capge650"
                )
                population = int((
                    (obs["actor_mask"][env_index] > 0.5)
                    & (obs["actors"][env_index, :, ACTOR_FEATURE_INDEX["isNonCreep"]] < 0.5)
                ).sum())
                population_bucket = (
                    "pop0_3" if population <= 3 else "pop4_7" if population <= 7
                    else "pop8_15" if population <= 15 else "popge16"
                )
                stratum = f"{wait_kind}:{budget_bucket}:{capacity_bucket}:{population_bucket}"
            existing = sum(item[0] == stratum for item in replay)
            if existing >= SPAWN_REPLAY_PER_STRATUM:
                continue
            compact_obs = _compact_entity_prefixes({
                key: value[env_index : env_index + 1]
                for key, value in obs.items()
            })
            actor_cap = compact_obs["actors"].shape[1]
            replay.append((
                stratum,
                actor_index,
                {key: value.clone() for key, value in compact_obs.items()},
                {
                    key: value[env_index : env_index + 1, :actor_cap].clone()
                    for key, value in actions.items()
                },
                eligible[env_index : env_index + 1, :actor_cap].clone(),
            ))


def _spawn_replay_coverage(
    replay: list[SpawnReplayRow],
) -> dict[str, float]:
    """Measure positive body coverage from tensors, never diagnostic names.

    Contract rows intentionally use stable scenario strata such as
    ``spawn:contract:spawn_hauler_3000``.  Inferring body length or available
    energy from those labels made valid contract refreshes silently disappear
    from qualification.  The observation and demonstrated action are the
    authoritative data.
    """
    coverage = {
        f"budget_{bucket}": 0.0 for bucket in SPAWN_BUDGET_BUCKETS
    } | {
        f"length_{bucket}": 0.0 for bucket in SPAWN_LENGTH_BUCKETS
    }
    spawn_type = INTENT_TYPES.index("spawnCreep")
    for _stratum, actor_index, obs, action, _eligible in replay:
        if int(action["types"][0, actor_index, 0]) != spawn_type:
            continue
        budget = int(round(
            float(obs["actors"][0, actor_index, _SPAWN_ENERGY_FEATURE])
            * MAX_ROOM_ENERGY
        ))
        length = int(action["body_counts"][0, actor_index, 0].sum())
        budget_bucket = _spawn_budget_bucket(budget)
        length_bucket = _spawn_length_bucket(length)
        coverage[f"budget_{budget_bucket}"] += 1.0
        coverage[f"length_{length_bucket}"] += 1.0
    return coverage


def _spawn_length_coverage_gap(counts: Mapping[str, float], prefix: str) -> dict[str, float]:
    """Name every body-length bucket this economy cannot fund, with its count.

    The gate above requires only the reachable buckets, so the unreached ones
    would otherwise vanish from the record.  Emitting them under an explicit
    ``<prefix>_length_unreached_<bucket>`` key keeps the gap visible in the
    checkpoint: the day a higher-RCL curriculum starts filling one, it shows up
    here first and can be promoted back into `SPAWN_LENGTH_REQUIRED_BUCKETS`.
    """
    return {
        f"{prefix}_length_unreached_{bucket}": float(counts.get(bucket, 0.0))
        for bucket in SPAWN_LENGTH_UNREACHED_BUCKETS
    }


def _teacher_spawn_coverage(
    teacher_by_curriculum: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Total the teacher's measured spawn supervision breadth per bucket.

    Every value is a count of the teacher's own labelled decisions, summed over
    curricula, plus the named body-length coverage gap.  Nothing here is an
    expectation about which body the teacher should have chosen.
    """
    def total(key: str) -> float:
        return sum(
            float(metrics.get(key, 0.0))
            for metrics in teacher_by_curriculum.values()
        )

    coverage = {"teacher_spawn_labels": total("spawn_labels")}
    coverage.update({
        f"teacher_spawn_budget_{bucket}": total(f"spawn_budget_{bucket}")
        for bucket in SPAWN_BUDGET_BUCKETS
    })
    coverage.update({
        f"teacher_spawn_length_{bucket}": total(f"spawn_length_{bucket}")
        for bucket in SPAWN_LENGTH_BUCKETS
    })
    coverage.update(_spawn_length_coverage_gap(
        {
            bucket: coverage[f"teacher_spawn_length_{bucket}"]
            for bucket in SPAWN_LENGTH_BUCKETS
        },
        "teacher_spawn",
    ))
    return coverage


def _teacher_contract_bodies(
    spawn_replay: Sequence[SpawnReplayRow],
) -> dict[str, dict[str, Any]]:
    """Read each spawn world's teacher body decision out of the corpus rows.

    The contract rows are The International's own labels, collected in those
    worlds and carried with the corpus `ti_bot_source_sha256` provenance, so
    qualification can compare a learned policy against the measured teacher
    instead of against a frozen archetype table.  Each row also carries the
    eligibility mask that decided which factors were supervised: body order is
    representable only for contiguous type blocks, so an interleaved teacher
    body is comparable by composition alone and says nothing about part order.
    """
    spawn_type = INTENT_TYPES.index("spawnCreep")
    order_start = 6 + N_BODY_PART
    prefix = "spawn:contract:"
    bodies: dict[str, dict[str, Any]] = {}
    for stratum, actor_index, _obs, action, eligible in spawn_replay:
        if not stratum.startswith(prefix):
            continue
        stage = stratum[len(prefix):]
        if int(action["types"][0, actor_index, 0]) != spawn_type:
            raise SystemExit(
                f"[joint] corpus spawn contract row {stratum!r} is not a spawn decision"
            )
        counts = action["body_counts"][0, actor_index, 0].long()
        order = action["body_order"][0, actor_index, 0].long()
        measured = {
            "parts": [
                int(part)
                for part in order.tolist()
                for _ in range(int(counts[part]))
            ],
            "order_supervised": bool(eligible[0, actor_index, order_start:].any()),
        }
        previous = bodies.setdefault(stage, measured)
        if previous != measured:
            raise SystemExit(
                f"[joint] corpus carries two different teacher bodies for {stage!r}: "
                f"{previous} and {measured}"
            )
    return bodies


def _spawn_body_matches(observed: Any, measured: Mapping[str, Any]) -> bool:
    """Compare a policy body against the teacher's, over supervised factors only."""
    if not isinstance(observed, (list, tuple)):
        return False
    parts = [int(part) for part in observed]
    teacher = [int(part) for part in measured["parts"]]
    if measured["order_supervised"]:
        return parts == teacher
    return sorted(parts) == sorted(teacher)


def _evaluation_seed_overlap(
    meta: dict[str, Any], *, offset: int, num_envs: int,
) -> set[int]:
    """Return evaluation seeds already present in either corpus split."""
    evaluation = {
        int(meta["seed"]) + int(offset) + index
        for index in range(int(num_envs))
    }
    collected = {
        int(entry["seed"])
        for key in ("train_env_map", "holdout_env_map")
        for entry in meta.get(key, [])
    }
    return evaluation & collected


def _ti_spawn_contract_label(
    obs: dict[str, torch.Tensor], info: dict[str, Any], *, stage: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool] | None:
    """Label the teacher's body decision for one spawn-world tick, if it made one.

    Mirrors `pretrain_corpus.TiLabeller` for the single factor group these worlds
    exist to cover.  The International's spawn command is deliberately not a full
    action today: it carries an energy-structure order and a spawn direction this
    ABI does not express, so the intent factor is eligible only when the shared
    translator declares the command exact, and the supervised factors are the
    body composition plus, when the body is ordered by block, its part order.
    Returns the batch-one action tensors, their eligibility mask, and whether the
    order factors were exact; `None` when the teacher did not spawn.
    """
    labels = [
        label for label in translate_ti_intents(
            info.get("expertIntents"),
            info.get("expertActorMeta") or [],
            info.get("expertTargetMeta") or [],
            info.get("expertRoomNames") or [],
        )
        if label.rejection is None
        and label.body_parts is not None
        and label.actor_index is not None
    ]
    if not labels:
        return None
    if len(labels) > 1:
        raise RuntimeError(
            f"spawn contract {stage!r} produced {len(labels)} spawn decisions in one "
            "tick; a contract world owns exactly one spawn"
        )
    label = labels[0]
    actor_index = int(label.actor_index)
    actor_cap = obs["actors"].shape[1]
    spawn_type = INTENT_TYPES.index("spawnCreep")
    if actor_index >= actor_cap:
        raise RuntimeError(
            f"spawn contract {stage!r} spawned from actor {actor_index} truncated "
            f"out of the {actor_cap}-row observation"
        )
    if not bool(obs["intent_mask"][0, actor_index, 0, spawn_type]):
        # `safe_bc_nll(..., strict=True)` refuses a masked label, and these worlds
        # are built with an idle spawn and a funded budget, so a masked spawn is a
        # world/ABI disagreement rather than a row to drop.
        raise RuntimeError(
            f"spawn contract {stage!r} spawn intent is masked illegal for actor "
            f"{actor_index}"
        )
    body_counts, body_order, order_exact = _parts_to_count_order(label.body_parts)
    actions = {
        "types": torch.zeros(1, actor_cap, 1, dtype=torch.long),
        "dirs": torch.zeros(1, actor_cap, 1, dtype=torch.long),
        "targets": torch.zeros(1, actor_cap, 1, dtype=torch.long),
        "amounts": torch.zeros(1, actor_cap, 1, dtype=torch.long),
        "construction_types": torch.zeros(1, actor_cap, 1, dtype=torch.long),
        "construction_tiles": torch.zeros(1, actor_cap, 1, dtype=torch.long),
        "body_counts": torch.zeros(1, actor_cap, 1, N_BODY_PART, dtype=torch.long),
        "body_order": torch.arange(N_BODY_PART).view(1, 1, 1, N_BODY_PART).expand(
            1, actor_cap, 1, -1,
        ).clone(),
    }
    actions["types"][0, actor_index, 0] = spawn_type
    actions["body_counts"][0, actor_index, 0] = body_counts
    actions["body_order"][0, actor_index, 0] = body_order
    eligible = torch.zeros(1, actor_cap, 6 + 2 * N_BODY_PART, dtype=torch.bool)
    if label.full_action:
        eligible[0, actor_index, 0] = True
    eligible[0, actor_index] |= _body_label_factor_mask(body_counts, order_exact)
    return actions, eligible, order_exact


def _collect_spawn_contract_replay(
    replay: list[SpawnReplayRow],
    teacher_by_curriculum: dict[str, dict[str, float]],
    *,
    node: str | None,
    room: str,
    bot_dir: str | None = None,
    seed: int = 0,
    max_ticks: int = 64,
) -> None:
    """Collect one exact, observable body decision per dedicated spawn world.

    These scenarios exist to cover rare 2..50-part bodies, not to occupy 60% of
    critic training for 14k artificial ticks.  Each world contributes precisely
    one teacher body decision; the long economy streams remain the critic
    distribution.

    The teacher is The International, which does not act on its opening ticks: a
    measured healthy episode issued no engine intent for its first twelve.  Each
    world is therefore stepped until its spawn decision lands and stops
    contributing afterwards, rather than being read once after reset.

    `bot_dir` of `None` defers to `ScreepsEnv`'s canonical resolution
    (`RL_EXPERT_BOT`, else the sibling TI dist), which is what
    `pretrain_corpus.CorpusConfig.resolved_ti_bot_dir` names.
    """
    envs = VecScreepsEnv(
        len(SPAWN_CURRICULA), node=node, room=room, max_episode=max_ticks + 5,
        device="cpu", curriculum=",".join(SPAWN_CURRICULA), lean_meta=False,
        expert=True, bot_dir=bot_dir, capture_expert_intents=True, seed=seed,
    )
    pending = set(range(len(SPAWN_CURRICULA)))
    try:
        envs.reset()
        for tick in range(max_ticks):
            if not pending:
                break
            assert envs.host_obs is not None
            # Two host stacks alternate, so the pre-step batch stays valid across
            # exactly one step; the retained rows clone what they keep.
            pre = envs.host_obs
            _obs, _reward, _done, infos = envs.step_expert()
            for env_index in sorted(pending):
                stage = SPAWN_CURRICULA[env_index]
                info = infos[env_index] or {}
                metrics = teacher_by_curriculum.setdefault(stage, {
                    "transitions": 0.0, "skill": 0.0, "delivery": 0.0,
                    "build": 0.0, "claims": 0.0, "max_creeps": 0.0,
                    "spawn_success": 0.0,
                })
                metrics["transitions"] += 1.0
                # `stepExpert` reports no intent summary at all, so an issued /
                # invalid counter here would report a measurement that does not
                # exist.  Spawn execution is on the expert wire, and the label is
                # checked against it below.
                metrics["spawn_success"] += float(info.get("spawnSuccess") or 0)
                single_pre = _compact_entity_prefixes({
                    key: value[env_index : env_index + 1] for key, value in pre.items()
                })
                labelled = _ti_spawn_contract_label(single_pre, info, stage=stage)
                if labelled is None:
                    continue
                actions, eligible, order_exact = labelled
                if float(info.get("spawnSuccess") or 0) < 1.0:
                    raise RuntimeError(
                        f"spawn contract {stage!r} decided a body the engine did not "
                        "execute"
                    )
                replay_start = len(replay)
                _append_spawn_replay(replay, single_pre, actions, eligible)
                if len(replay) - replay_start != 1:
                    raise RuntimeError(
                        f"spawn contract {stage!r} retained "
                        f"{len(replay) - replay_start} rows for one decision"
                    )
                _stratum, actor_index, row_obs, row_action, row_eligible = replay[replay_start]
                replay[replay_start] = (
                    f"spawn:contract:{stage}", actor_index, row_obs, row_action,
                    row_eligible,
                )
                _record_spawn_labels(metrics, actions, single_pre, 0)
                metrics["spawn_label_tick"] = float(tick)
                if not order_exact:
                    # Same accounting as the corpus labeller: an interleaved body
                    # keeps its exact composition and drops the order factors.
                    metrics["spawn_body_order_interleaved"] = 1.0
                pending.discard(env_index)
    finally:
        envs.close()
    if pending:
        raise RuntimeError(
            "spawn contract teacher issued no spawn within "
            f"{max_ticks} ticks for "
            + ", ".join(repr(SPAWN_CURRICULA[index]) for index in sorted(pending))
        )


def _append_teacher_lifecycle_replay(
    replay: list[LifecycleSample],
    seen_by_stratum: dict[str, int],
    retained_by_stratum: dict[str, list[int]],
    rng: np.random.Generator,
    obs: dict[str, torch.Tensor],
    actions: dict[str, torch.Tensor],
    eligible: torch.Tensor,
    infos: list[dict],
    *,
    timestep: int,
    env_labels: list[int] | None = None,
) -> None:
    """Retain one bounded row per stage/state/action signature.

    A team tick may contain harvest, logistics, construction, and control at
    once. Preserve that joint label once under its complete category signature
    instead of cloning it or letting common harvest hide the rare actions.

    ``eligible`` is the teacher's per-factor supervision mask over
    ``[batch, actors, 6 + 2 * N_BODY_PART]``.  A tick the teacher did not label
    at all is not a candidate: it never counts as seen and never draws from the
    reservoir RNG, so eligibility cannot perturb the retained sampling of ticks
    that do carry labels.

    ``env_labels`` renames each batch row's recorded ``env_index``.  Callers that
    step one environment at a time still need the row to name its true fleet
    index, because the return lookup is indexed by ``(timestep, env_index)``.
    """
    if env_labels is not None and len(env_labels) != len(infos):
        raise ValueError(
            f"env_labels labels {len(env_labels)} rows for {len(infos)} infos"
        )
    groups = {
        "spawn": {INTENT_TYPES.index("spawnCreep")},
        "harvest": {INTENT_TYPES.index("harvest")},
        "logistics": {
            INTENT_TYPES.index("transfer"), INTENT_TYPES.index("withdraw"),
            INTENT_TYPES.index("pickup"), INTENT_TYPES.index("drop"),
        },
        "construction": {
            INTENT_TYPES.index("build"), INTENT_TYPES.index("repair"),
            INTENT_TYPES.index("createConstructionSite"),
        },
        "control": {
            INTENT_TYPES.index("upgradeController"),
            INTENT_TYPES.index("claimController"),
            INTENT_TYPES.index("reserveController"),
        },
    }
    for env_index, info in enumerate(infos):
        if (info or {}).get("invalid_demo") or (info or {}).get("recovered"):
            continue
        if not bool(eligible[env_index].any()):
            continue
        chosen = set(int(value) for value in actions["types"][env_index, :, 0].tolist())
        categories = [name for name, members in groups.items() if chosen & members]
        if not categories:
            categories = ["other" if chosen - {0} else "none"]
        rcl = int(round(float(obs["globals"][env_index, 0]) * 8))
        population = int(round(float(obs["globals"][env_index, 3]) * 50))
        pop_bucket = "p0_3" if population <= 3 else "p4_11" if population <= 11 else "p12p"
        stage = str((info or {}).get("curriculum") or "unknown")
        signature = "+".join(categories)
        stratum = f"{stage}:r{rcl}:{pop_bucket}:{signature}"
        seen = seen_by_stratum.get(stratum, 0) + 1
        seen_by_stratum[stratum] = seen
        stratum_indices = retained_by_stratum.setdefault(stratum, [])
        replacement_index: int | None = None
        if len(stratum_indices) >= TEACHER_REPLAY_PER_STRATUM:
            replacement = int(rng.integers(0, seen))
            if replacement >= TEACHER_REPLAY_PER_STRATUM:
                continue
            replacement_index = stratum_indices[replacement]
        compact_obs = _compact_entity_prefixes({
            key: value[env_index : env_index + 1] for key, value in obs.items()
        })
        actor_cap = compact_obs["actors"].shape[1]
        sample = LifecycleSample(
            stratum=stratum,
            timestep=timestep,
            env_index=(
                int(env_labels[env_index]) if env_labels is not None else env_index
            ),
            obs={key: value.clone() for key, value in compact_obs.items()},
            action={
                key: value[env_index : env_index + 1, :actor_cap].clone()
                for key, value in actions.items()
            },
            eligible=eligible[env_index : env_index + 1, :actor_cap].clone(),
        )
        if replacement_index is None:
            replay.append(sample)
            stratum_indices.append(len(replay) - 1)
        else:
            replay[replacement_index] = sample


def _balanced_lifecycle_indices(replay: list[LifecycleSample]) -> list[int]:
    """Build the exact stage/stratum-balanced lifecycle multiset."""
    if not replay:
        raise ValueError("lifecycle replay is empty")
    by_stratum: dict[str, list[int]] = {}
    for index, sample in enumerate(replay):
        by_stratum.setdefault(sample.stratum, []).append(index)
    largest = max(len(indices) for indices in by_stratum.values())
    by_stage: dict[str, list[int]] = {}
    for indices in by_stratum.values():
        repeats = (largest + len(indices) - 1) // len(indices)
        balanced_stratum = (indices * repeats)[:largest]
        stage = replay[indices[0]].stratum.split(":", 1)[0]
        by_stage.setdefault(stage, []).extend(balanced_stratum)
    largest_stage = max(len(indices) for indices in by_stage.values())
    balanced: list[int] = []
    for indices in by_stage.values():
        repeats = (largest_stage + len(indices) - 1) // len(indices)
        balanced.extend((indices * repeats)[:largest_stage])
    return balanced


def _balanced_lifecycle_order(
    replay: list[LifecycleSample],
    *,
    shuffle_generator: torch.Generator | None,
) -> torch.Tensor:
    """Balance semantic strata and curricula before one deployment-policy epoch."""
    balanced = _balanced_lifecycle_indices(replay)
    order = torch.tensor(balanced, dtype=torch.long)
    return order[torch.randperm(order.numel(), generator=shuffle_generator)]


def _independent_cyclic_order(
    size: int,
    count: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw a full-coverage independent shuffled stream of exactly ``count`` rows."""
    if size <= 0:
        raise ValueError("independent replay is empty")
    chunks: list[torch.Tensor] = []
    remaining = int(count)
    while remaining > 0:
        permutation = torch.randperm(size, generator=generator)
        take = min(remaining, size)
        chunks.append(permutation[:take])
        remaining -= take
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.long)


def _evenly_distribute_batches(
    batches: list[Any], slots: int,
) -> list[list[Any]]:
    """Assign every auxiliary batch exactly once across primary optimizer steps."""
    if slots <= 0:
        raise ValueError("primary optimizer-step count must be positive")
    scheduled: list[list[Any]] = [[] for _ in range(slots)]
    if not batches:
        return scheduled
    # Place batch centers at evenly spaced primary-step quantiles. This keeps
    # each independently shuffled lane interleaved without changing exposure.
    for index, batch in enumerate(batches):
        slot = min(slots - 1, ((2 * index + 1) * slots) // (2 * len(batches)))
        scheduled[slot].append(batch)
    return scheduled


def _schedule_auxiliary_lanes(
    lanes: dict[str, list[Any]], slots: int,
) -> dict[str, list[list[Any]]]:
    """Jointly interleave lanes with exactly zero clip-coupled collisions."""
    if slots <= 0:
        raise ValueError("primary optimizer-step count must be positive")
    total_batches = sum(len(batches) for batches in lanes.values())
    if total_batches > slots:
        counts = {name: len(batches) for name, batches in lanes.items()}
        raise ValueError(
            "auxiliary replay exceeds collision-free lifecycle capacity: "
            f"batches={total_batches} primary_steps={slots} lanes={counts}"
        )
    scheduled = {
        name: [[] for _ in range(slots)]
        for name in lanes
    }
    ranked = sorted(
        (
            ((2 * index + 1) / (2 * len(batches)), name, index, batch)
            for name, batches in lanes.items()
            for index, batch in enumerate(batches)
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    for merged_index, (_progress, name, _lane_index, batch) in enumerate(ranked):
        slot = min(
            slots - 1,
            ((2 * merged_index + 1) * slots) // (2 * max(1, len(ranked))),
        )
        scheduled[name][slot].append(batch)
    collisions = [
        slot for slot in range(slots)
        if sum(len(scheduled[name][slot]) for name in lanes) > 1
    ]
    if collisions:
        raise AssertionError(
            f"collision-free auxiliary scheduler produced collisions: {collisions}"
        )
    return scheduled


def _preflight_joint_geometry(
    lifecycle_replay: list[LifecycleSample],
    rare_refs_by_intent: dict[str, list[tuple[int, int]]],
    spawn_replay: list[SpawnReplayRow],
    *,
    minibatch: int,
) -> dict[str, int]:
    """Validate exact fused geometry using host-only disposable RNG streams."""
    width = max(1, int(minibatch))
    lifecycle_rows = len(_balanced_lifecycle_indices(lifecycle_replay))
    primary_steps = (lifecycle_rows + width - 1) // width
    rare_batches = _rare_intent_actor_batches(
        rare_refs_by_intent, minibatch=min(32, width),
        shuffle_generator=torch.Generator().manual_seed(0x52415245),
    )
    spawn_batches = _spawn_actor_batches(
        spawn_replay, minibatch=min(32, width),
        shuffle_generator=torch.Generator().manual_seed(0x53504157),
    )
    lanes = {
        "rare": rare_batches,
        "spawn": spawn_batches,
    }
    _schedule_auxiliary_lanes(lanes, primary_steps)
    counts = {f"{name}_batches": len(batches) for name, batches in lanes.items()}
    return counts | {
        "auxiliary_batches": sum(counts.values()),
        "primary_steps": primary_steps,
        "lifecycle_rows": lifecycle_rows,
    }


def _train_joint_lifecycle_epoch(
    actor: Actor,
    critic: Critic,
    optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    replay: list[LifecycleSample],
    returns_tn: torch.Tensor,
    temporal_replay: list[TemporalSample],
    *,
    device: torch.device,
    use_bf16: bool,
    minibatch: int,
    value_coef: float = 1.0,
    shuffle_generator: torch.Generator | None = None,
    nextlat_shuffle_generator: torch.Generator,
    nextlat_actor_coef: float = 1.0,
    nextlat_critic_coef: float = 1.0,
    nextlat_critic_kl_coef: float = 0.1,
    rare_replay: list[LifecycleSample] | None = None,
    rare_refs_by_intent: dict[str, list[tuple[int, int]]] | None = None,
    rare_shuffle_generator: torch.Generator | None = None,
    spawn_replay: list[SpawnReplayRow] | None = None,
    spawn_shuffle_generator: torch.Generator | None = None,
) -> tuple[float, float, float, int, dict[str, float]]:
    """Optimize lifecycle authority and NextLat in the same model updates.

    Lifecycle rows define the number and size of optimizer steps. Temporal
    rows are independently shuffled and cycled to the same exposure count.
    Rare-intent and spawn rows each contribute exactly one of their
    independently balanced source epochs, evenly interleaved over those same
    steps. Every actor loss is combined before one clip/update, so no narrow
    replay owns a later Muon step that can overwrite the broad policy.
    """
    if not replay:
        raise ValueError("joint lifecycle replay is empty")
    if not temporal_replay:
        raise ValueError("joint temporal replay is empty")
    if rare_replay and rare_shuffle_generator is None:
        raise ValueError("fused rare-intent replay requires its dedicated shuffle RNG")
    if spawn_replay and spawn_shuffle_generator is None:
        raise ValueError("fused spawn replay requires its dedicated shuffle RNG")
    order = _balanced_lifecycle_order(
        replay, shuffle_generator=shuffle_generator,
    )
    temporal_order = _independent_cyclic_order(
        len(temporal_replay), order.numel(),
        generator=nextlat_shuffle_generator,
    )
    primary_steps = (order.numel() + max(1, minibatch) - 1) // max(1, minibatch)
    rare_batches = _rare_intent_actor_batches(
        rare_refs_by_intent or {}, minibatch=min(32, minibatch),
        shuffle_generator=rare_shuffle_generator,
    ) if rare_replay and rare_shuffle_generator is not None else []
    spawn_batches = _spawn_actor_batches(
        spawn_replay or [], minibatch=min(32, minibatch),
        shuffle_generator=spawn_shuffle_generator,
    ) if spawn_replay else []
    auxiliary_schedule = _schedule_auxiliary_lanes({
        "rare": rare_batches,
        "spawn": spawn_batches,
    }, primary_steps)
    rare_schedule = auxiliary_schedule["rare"]
    spawn_schedule = auxiliary_schedule["spawn"]
    losses: list[float] = []
    value_losses: list[float] = []
    legal: list[float] = []
    actor_latent_losses: list[float] = []
    critic_latent_losses: list[float] = []
    critic_kls: list[float] = []
    auxiliary_sums = {
        "rare_nll": 0.0, "rare_legal": 0.0, "rare_exposures": 0,
        "spawn_nll": 0.0, "spawn_legal": 0.0, "spawn_exposures": 0,
    }
    actor.train()
    critic.train()
    updates = 0
    for primary_step, start in enumerate(range(0, order.numel(), max(1, minibatch))):
        selected = order[start : start + max(1, minibatch)].tolist()
        temporal_selected = temporal_order[
            start : start + len(selected)
        ].tolist()
        obs_rows = [replay[index].obs for index in selected]
        action_rows = [replay[index].action for index in selected]
        actor_cap = max(row["actors"].shape[1] for row in obs_rows)
        target_cap = max(row["targets"].shape[1] for row in obs_rows)
        room_cap = max(row["room_mask"].shape[1] for row in obs_rows)
        obs_device = promote_obs_device(_pad_replay_tensors(
            obs_rows, actor_cap=actor_cap, target_cap=target_cap, room_cap=room_cap,
        ), device, non_blocking=False)
        action_device = {
            key: value.to(device) for key, value in _pad_replay_tensors(
                action_rows, actor_cap=actor_cap, target_cap=target_cap,
                room_cap=room_cap, actions=True,
            ).items()
        }
        eligible_device = _pad_eligible_masks(
            [replay[index].eligible for index in selected], actor_cap=actor_cap,
        ).to(device)
        target = torch.tensor(
            [
                float(returns_tn[replay[index].timestep, replay[index].env_index])
                for index in selected
            ],
            dtype=torch.float32,
            device=device,
        )
        temporal_obs, temporal_action, temporal_next_obs = _temporal_minibatch(
            temporal_replay, temporal_selected, device=device,
        )
        with (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_bf16 else nullcontext()
        ):
            out = actor(obs_device, action=action_device)
            # Only factors this teacher actually labelled on this row may be
            # supervised; the model's own active-factor mask narrows it further.
            nll, fraction = safe_bc_nll(
                out.factor_logprob,
                _supervised_factor_mask(eligible_device, out.factor_active),
                strict=True,
            )
            current_actor_latent = actor.encode_state(temporal_obs)
            with torch.no_grad():
                next_actor_latent = actor.encode_state(temporal_next_obs).detach()
            predicted_actor_latent = actor.predict_next_latent(
                current_actor_latent, temporal_obs, temporal_action,
            )
            actor_latent_loss = F.smooth_l1_loss(
                predicted_actor_latent.float(), next_actor_latent.float(),
            )
            actor_total = nll + float(nextlat_actor_coef) * actor_latent_loss
        if not torch.isfinite(actor_total):
            raise FloatingPointError(f"non-finite joint actor loss {actor_total}")
        optimizer.zero_grad(set_to_none=True)
        actor_total.backward()
        for rare_selected in rare_schedule[primary_step]:
            rare_loss, rare_legal, rare_per_actor = _rare_intent_batch_loss(
                actor, rare_replay or [], rare_selected,
                device=device, use_bf16=use_bf16,
            )
            rare_loss.backward()
            auxiliary_sums["rare_nll"] += float(rare_per_actor.detach().sum())
            auxiliary_sums["rare_legal"] += float(rare_legal) * len(rare_selected)
            auxiliary_sums["rare_exposures"] += len(rare_selected)
        for spawn_selected in spawn_schedule[primary_step]:
            spawn_loss, spawn_legal = _spawn_actor_batch_loss(
                actor, spawn_replay or [], spawn_selected,
                device=device, use_bf16=use_bf16,
            )
            spawn_loss.backward()
            auxiliary_sums["spawn_nll"] += float(spawn_loss.detach()) * len(spawn_selected)
            auxiliary_sums["spawn_legal"] += float(spawn_legal) * len(spawn_selected)
            auxiliary_sums["spawn_exposures"] += len(spawn_selected)
        nn.utils.clip_grad_norm_(
            actor.parameters(), float(PPO_CFG.get("maxGradNorm", 0.5)),
            error_if_nonfinite=True,
        )
        with (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_bf16 else nullcontext()
        ):
            value_logits = critic(obs_device, return_logits=True)
            value_loss = critic.support.cross_entropy(
                value_logits.float(), target, validate=False,
            ).mean()
            _current_logits, current_critic_latent = (
                critic.value_logits_and_latent(temporal_obs)
            )
            with torch.no_grad():
                next_critic_logits, next_critic_latent = (
                    critic.value_logits_and_latent(temporal_next_obs)
                )
                next_critic_logits = next_critic_logits.detach()
                next_critic_latent = next_critic_latent.detach()
            predicted_critic_latent = critic.predict_next_latent(
                current_critic_latent, temporal_obs, temporal_action,
            )
            critic_latent_loss = F.smooth_l1_loss(
                predicted_critic_latent.float(), next_critic_latent.float(),
            )
        # Match PPO's fp32 distributional probe.  The head parameters remain
        # detached while the probe keeps gradients into the predicted latent.
        predicted_logits = critic.detached_value_logits(
            predicted_critic_latent.float(),
        )
        teacher_logp = F.log_softmax(next_critic_logits.float(), dim=-1)
        student_logp = F.log_softmax(predicted_logits.float(), dim=-1)
        critic_kl = F.kl_div(
            student_logp, teacher_logp, log_target=True,
            reduction="batchmean",
        )
        critic_auxiliary = (
            float(nextlat_critic_coef) * critic_latent_loss
            + float(nextlat_critic_kl_coef) * critic_kl
        )
        critic_total = float(value_coef) * value_loss + critic_auxiliary
        if not torch.isfinite(critic_total):
            raise FloatingPointError(f"non-finite joint critic loss {critic_total}")

        # Retain only actor gradients while building the critic graphs.  Both
        # models step only after every objective and gradient is validated.
        critic_optimizer.zero_grad(set_to_none=True)
        critic_total.backward()
        nn.utils.clip_grad_norm_(
            critic.parameters(), float(VALUE_CFG["criticMaxGradNorm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        critic_optimizer.step()
        losses.append(float(nll.detach()))
        value_losses.append(float(value_loss.detach()))
        actor_latent_losses.append(float(actor_latent_loss.detach()))
        critic_latent_losses.append(float(critic_latent_loss.detach()))
        critic_kls.append(float(critic_kl.detach()))
        legal.append(float(fraction))
        updates += len(selected)
    return (
        float(np.mean(losses)), float(np.mean(legal)),
        float(np.mean(value_losses)), updates,
        {
            "actor_smooth_l1": float(np.mean(actor_latent_losses)),
            "critic_smooth_l1": float(np.mean(critic_latent_losses)),
            "critic_kl": float(np.mean(critic_kls)),
            "rows": float(updates),
            **{
                f"{lane}_{metric}": (
                    float(auxiliary_sums[f"{lane}_{metric}"])
                    / max(1, int(auxiliary_sums[f"{lane}_exposures"]))
                )
                for lane in ("rare", "spawn")
                for metric in ("nll", "legal")
            },
            **{
                f"{lane}_exposures": float(auxiliary_sums[f"{lane}_exposures"])
                for lane in ("rare", "spawn")
            },
            "rare_batches": float(len(rare_batches)),
            "spawn_batches": float(len(spawn_batches)),
            "auxiliary_batches": float(len(rare_batches) + len(spawn_batches)),
            "auxiliary_minibatch": float(min(32, max(1, minibatch))),
            "optimizer_steps": float(primary_steps),
        },
    )


def _train_global_lifecycle_epoch(
    actor: Actor,
    critic: Critic,
    optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    replay: list[LifecycleSample],
    returns_tn: torch.Tensor,
    *,
    device: torch.device,
    use_bf16: bool,
    minibatch: int,
    value_coef: float = 1.0,
    shuffle_generator: torch.Generator | None = None,
) -> tuple[float, float, float, int]:
    """Lifecycle-only compatibility path for focused component tests.

    Production joint pretraining calls :func:`_train_joint_lifecycle_epoch` so
    there is no standalone NextLat optimizer phase.
    """
    if not replay:
        return float("nan"), 0.0, float("nan"), 0
    # A zero-coefficient self-transition keeps this compatibility path on the
    # same optimizer implementation without allowing an auxiliary-only step.
    temporal = [TemporalSample(
        stratum=sample.stratum,
        timestep=sample.timestep,
        env_index=sample.env_index,
        obs=sample.obs,
        action=sample.action,
        counterfactual_action=None,
        next_obs=sample.obs,
    ) for sample in replay]
    nll, legal, value_loss, updates, _metrics = _train_joint_lifecycle_epoch(
        actor, critic, optimizer, critic_optimizer, replay, returns_tn, temporal,
        device=device, use_bf16=use_bf16, minibatch=minibatch,
        value_coef=value_coef, shuffle_generator=shuffle_generator,
        nextlat_shuffle_generator=torch.Generator().manual_seed(0),
        nextlat_actor_coef=0.0, nextlat_critic_coef=0.0,
        nextlat_critic_kl_coef=0.0,
    )
    return nll, legal, value_loss, updates


@torch.inference_mode()
def _evaluate_global_lifecycle(
    actor: Actor,
    critic: Critic,
    replay: list[LifecycleSample],
    returns_tn: torch.Tensor,
    *,
    device: torch.device,
    minibatch: int,
) -> dict[str, float]:
    """Report final-model metrics on each retained row exactly once."""
    if not replay:
        return {
            "nll": float("nan"), "legal_frac": 0.0,
            "value_loss": float("nan"), "ev": float("nan"),
        }
    actor.eval()
    critic.eval()
    nll_sum = 0.0
    factor_count = 0
    target_rows: list[torch.Tensor] = []
    prediction_rows: list[torch.Tensor] = []
    stage_rows: list[str] = []
    value_loss_sum = 0.0
    for start in range(0, len(replay), max(1, minibatch)):
        selected = list(range(start, min(len(replay), start + max(1, minibatch))))
        obs_rows = [replay[index].obs for index in selected]
        action_rows = [replay[index].action for index in selected]
        actor_cap = max(row["actors"].shape[1] for row in obs_rows)
        target_cap = max(row["targets"].shape[1] for row in obs_rows)
        room_cap = max(row["room_mask"].shape[1] for row in obs_rows)
        obs_device = promote_obs_device(_pad_replay_tensors(
            obs_rows, actor_cap=actor_cap, target_cap=target_cap, room_cap=room_cap,
        ), device, non_blocking=False)
        action_device = {
            key: value.to(device) for key, value in _pad_replay_tensors(
                action_rows, actor_cap=actor_cap, target_cap=target_cap,
                room_cap=room_cap, actions=True,
            ).items()
        }
        eligible_device = _pad_eligible_masks(
            [replay[index].eligible for index in selected], actor_cap=actor_cap,
        ).to(device)
        target = torch.tensor(
            [
                float(returns_tn[replay[index].timestep, replay[index].env_index])
                for index in selected
            ], dtype=torch.float32, device=device,
        )
        out = actor(obs_device, action=action_device)
        active = _supervised_factor_mask(eligible_device, out.factor_active)
        nll, legal = safe_bc_nll(out.factor_logprob, active, strict=True)
        if legal != 1.0:
            raise RuntimeError(f"global lifecycle legal fraction={legal}")
        count = int(active.sum())
        nll_sum += float(nll) * count
        factor_count += count
        logits = critic(obs_device, return_logits=True)
        value_loss_sum += float(
            critic.support.cross_entropy(logits.float(), target, validate=False).sum()
        )
        prediction_rows.append(critic.support.to_expected_scalar(logits.float()).cpu())
        target_rows.append(target.cpu())
        stage_rows.extend(replay[index].stratum.split(":", 1)[0] for index in selected)
    prediction = torch.cat(prediction_rows)
    target = torch.cat(target_rows)
    variance = float(torch.var(target, unbiased=False))
    ev = float("nan") if variance == 0 else 1.0 - float(
        torch.var(target - prediction, unbiased=False) / variance
    )
    actor.train()
    critic.train()
    metrics = {
        "nll": nll_sum / max(1, factor_count),
        "legal_frac": 1.0,
        "value_loss": value_loss_sum / max(1, len(replay)),
        "ev": ev,
        "target_min": float(target.min()),
        "target_max": float(target.max()),
    }
    stage_array = np.asarray(stage_rows)
    for stage in sorted(set(stage_rows)):
        indices = torch.from_numpy(np.flatnonzero(stage_array == stage))
        stage_target = target[indices]
        stage_prediction = prediction[indices]
        stage_variance = float(torch.var(stage_target, unbiased=False))
        metrics[f"stage_{stage}_ev"] = (
            float("nan") if stage_variance == 0 else 1.0 - float(
                torch.var(stage_target - stage_prediction, unbiased=False)
                / stage_variance
            )
        )
        metrics[f"stage_{stage}_count"] = float(indices.numel())
    return metrics


@torch.inference_mode()
def _evaluate_lifecycle_actor_nll(
    actor: Actor,
    replay: list[LifecycleSample],
    *,
    device: torch.device,
    minibatch: int,
) -> float:
    """Score broad-policy retention without an unnecessary critic forward."""
    if not replay:
        return float("nan")
    actor.eval()
    nll_sum = 0.0
    factor_count = 0
    for start in range(0, len(replay), max(1, minibatch)):
        selected = list(range(start, min(len(replay), start + max(1, minibatch))))
        obs_rows = [replay[index].obs for index in selected]
        action_rows = [replay[index].action for index in selected]
        actor_cap = max(row["actors"].shape[1] for row in obs_rows)
        target_cap = max(row["targets"].shape[1] for row in obs_rows)
        room_cap = max(row["room_mask"].shape[1] for row in obs_rows)
        obs_device = promote_obs_device(_pad_replay_tensors(
            obs_rows, actor_cap=actor_cap, target_cap=target_cap,
            room_cap=room_cap,
        ), device, non_blocking=False)
        action_device = {
            key: value.to(device) for key, value in _pad_replay_tensors(
                action_rows, actor_cap=actor_cap, target_cap=target_cap,
                room_cap=room_cap, actions=True,
            ).items()
        }
        eligible_device = _pad_eligible_masks(
            [replay[index].eligible for index in selected], actor_cap=actor_cap,
        ).to(device)
        out = actor(obs_device, action=action_device)
        eligible = _supervised_factor_mask(eligible_device, out.factor_active)
        nll, legal = safe_bc_nll(out.factor_logprob, eligible, strict=True)
        if legal != 1.0:
            raise RuntimeError(f"lifecycle retention legal fraction={legal}")
        count = int(eligible.sum())
        nll_sum += float(nll) * count
        factor_count += count
    actor.train()
    return nll_sum / max(1, factor_count)


def _auxiliary_lifecycle_retained(
    after_joint_nll: float,
    final_nll: float,
    *,
    max_ratio: float,
) -> bool:
    """Reject a final narrow-lane actor that materially forgot broad BC."""
    return (
        np.isfinite(after_joint_nll)
        and np.isfinite(final_nll)
        and final_nll <= max(abs(after_joint_nll), 1e-8) * float(max_ratio)
    )


def _rare_intent_actor_refs(
    replay: list[LifecycleSample],
) -> dict[str, list[tuple[int, int]]]:
    """Index rare-intent actors whose semantic factors are actually labelled.

    A teacher row may name a rare intent while supervising none of the factors
    this lane imitates, which is routine for factor-masked TI rows.  Admitting
    such an actor would clone a fabricated structure type or controller pointer,
    so require exactly what :data:`RARE_INTENT_REQUIRED_FACTORS` names.
    """
    intent_ids = {name: INTENT_TYPES.index(name) for name in RARE_ACTOR_INTENTS}
    refs = {name: [] for name in RARE_ACTOR_INTENTS}
    for sample_index, sample in enumerate(replay):
        types = sample.action["types"]
        if types.ndim != 3 or types.shape[0] != 1 or types.shape[2] != 1:
            raise ValueError(
                "lifecycle action types must have shape [1, actors, 1], got "
                f"{tuple(types.shape)}"
            )
        chosen = types[0, :, 0]
        eligible = sample.eligible[0]
        for name, intent_id in intent_ids.items():
            required = list(RARE_INTENT_REQUIRED_FACTORS[name])
            refs[name].extend(
                (sample_index, int(actor_index))
                for actor_index in (chosen == intent_id).nonzero().flatten().tolist()
                if bool(eligible[actor_index, required].all())
            )
    return refs


def _ti_rare_intent_samples(
    rows: list[dict[str, Any]],
) -> tuple[list[LifecycleSample], dict[str, list[tuple[int, int]]]]:
    """Adapt exact TI rare factors into the balanced semantic actor lane.

    The general TI reservoir is intentionally multi-source and factor-masked.
    Rare construction/controller labels would otherwise be overwhelmed by
    harvest ticks.  Only rows carrying every semantic factor required by the
    rare objective are admitted here.
    """
    samples: list[LifecycleSample] = []
    refs = {name: [] for name in RARE_ACTOR_INTENTS}
    for row_index, row in enumerate(rows):
        types = row["action"]["types"][0, :, 0]
        eligible = row["eligible"][0]
        for name, factors in RARE_INTENT_REQUIRED_FACTORS.items():
            intent_id = INTENT_TYPES.index(name)
            candidates = torch.nonzero(types == intent_id, as_tuple=False).flatten()
            for actor_tensor in candidates:
                actor_index = int(actor_tensor)
                if not bool(eligible[actor_index, list(factors)].all()):
                    continue
                sample_index = len(samples)
                samples.append(LifecycleSample(
                    stratum=f"ti:{name}",
                    timestep=int(row.get("timestep", row_index)),
                    env_index=0,
                    obs=row["obs"],
                    action=row["action"],
                    eligible=row["eligible"],
                ))
                refs[name].append((sample_index, actor_index))
    return samples, refs


def _rare_actor_batch(
    replay: list[LifecycleSample],
    selected: list[tuple[int, int]],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    sample_indices = [sample_index for sample_index, _actor_index in selected]
    obs_rows = [replay[index].obs for index in sample_indices]
    action_rows = [replay[index].action for index in sample_indices]
    actor_cap = max(row["actors"].shape[1] for row in obs_rows)
    target_cap = max(row["targets"].shape[1] for row in obs_rows)
    room_cap = max(row["room_mask"].shape[1] for row in obs_rows)
    obs_device = promote_obs_device(_pad_replay_tensors(
        obs_rows, actor_cap=actor_cap, target_cap=target_cap, room_cap=room_cap,
    ), device, non_blocking=False)
    action_device = {
        key: value.to(device) for key, value in _pad_replay_tensors(
            action_rows, actor_cap=actor_cap, target_cap=target_cap,
            room_cap=room_cap, actions=True,
        ).items()
    }
    # No eligibility here: callers that supervise masked teacher factors narrow
    # the mask themselves through _rare_eligible_batch.
    actor_indices = torch.tensor(
        [actor_index for _sample_index, actor_index in selected],
        dtype=torch.long,
        device=device,
    )
    return obs_device, action_device, actor_indices


def _rare_eligible_batch(
    replay: list[LifecycleSample],
    selected: list[tuple[int, int]],
    *,
    actor_cap: int,
    device: torch.device,
) -> torch.Tensor:
    """Batch the rare lane's teacher masks at its already padded actor cap."""
    return _pad_eligible_masks(
        [replay[sample_index].eligible for sample_index, _actor_index in selected],
        actor_cap=actor_cap,
    ).to(device)


def _action_factor_values(action) -> torch.Tensor:
    """Materialize factors in the exact order used by ``factor_logprob``."""
    def field(name: str) -> torch.Tensor:
        return action[name] if isinstance(action, dict) else getattr(action, name)

    return torch.cat((
        field("types"),
        field("dirs"),
        field("targets"),
        field("amounts"),
        field("construction_types"),
        field("construction_tiles"),
        field("body_counts")[..., 0, :],
        field("body_order")[..., 0, :],
    ), dim=-1)


def _actor_balanced_factor_nll(
    factor_logprob: torch.Tensor,
    factor_active: torch.Tensor,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Average active factors per actor, then weight every actor equally."""
    _factor_nll, legal = safe_bc_nll(
        factor_logprob, factor_active, strict=True,
    )
    active_count = factor_active.sum(dim=-1)
    if bool((active_count <= 0).any()):
        raise RuntimeError("rare-intent actor has no active action factors")
    per_actor = -torch.where(
        factor_active, factor_logprob, torch.zeros_like(factor_logprob),
    ).sum(dim=-1) / active_count
    return per_actor.mean(), legal, per_actor


def _rare_intent_factor_mask(
    intent: str,
    factor_active: torch.Tensor,
) -> torch.Tensor:
    """Select the semantic factors worth imitating for a rare intent.

    A construction demonstration is a structure type at a position, and cloning
    only the type teaches half of it. Construction is rare enough (a few hundred
    rows against tens of thousands) that the main corpus mean cannot carry the
    placement on its own, so the rare lane imitates the teacher's tile exactly.
    Claim examples retain their exact controller pointer.
    """
    selected = torch.zeros_like(factor_active)
    selected[..., 0] = factor_active[..., 0]
    if intent == "createConstructionSite":
        selected[..., 4] = factor_active[..., 4]
        selected[..., 5] = factor_active[..., 5]
    elif intent == "claimController":
        selected[..., 2] = factor_active[..., 2]
    else:
        raise ValueError(f"unsupported rare actor intent {intent!r}")
    if bool((selected.sum(dim=-1) <= 0).any()):
        raise RuntimeError(f"rare intent {intent!r} has no selected factors")
    return selected


def _rare_intent_actor_batches(
    refs_by_intent: dict[str, list[tuple[int, int]]],
    *,
    minibatch: int,
    shuffle_generator: torch.Generator,
) -> list[list[tuple[str, int, int]]]:
    """Build one exact intent-balanced rare-lane epoch without optimizing."""
    missing = [name for name in RARE_ACTOR_INTENTS if not refs_by_intent.get(name)]
    if missing:
        raise ValueError(f"rare-intent training replay is missing {missing}")
    largest = max(len(refs_by_intent[name]) for name in RARE_ACTOR_INTENTS)
    balanced: list[tuple[str, int, int]] = []
    for name in RARE_ACTOR_INTENTS:
        refs = refs_by_intent[name]
        complete_repeats, remainder = divmod(largest, len(refs))
        balanced.extend((name, *ref) for ref in refs * complete_repeats)
        if remainder:
            extra = torch.randperm(
                len(refs), generator=shuffle_generator,
            )[:remainder].tolist()
            balanced.extend((name, *refs[index]) for index in extra)
    order = torch.randperm(len(balanced), generator=shuffle_generator).tolist()
    width = max(1, int(minibatch))
    return [
        [balanced[index] for index in order[start : start + width]]
        for start in range(0, len(order), width)
    ]


def _rare_intent_batch_loss(
    actor: Actor,
    replay: list[LifecycleSample],
    selected_named: list[tuple[str, int, int]],
    *,
    device: torch.device,
    use_bf16: bool,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    selected = [
        (sample_index, actor_index)
        for _name, sample_index, actor_index in selected_named
    ]
    obs_device, action_device, actor_indices = _rare_actor_batch(
        replay, selected, device=device,
    )
    eligible_device = _rare_eligible_batch(
        replay, selected, actor_cap=action_device["types"].shape[1], device=device,
    )
    row = torch.arange(len(selected), device=device)
    with (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 else nullcontext()
    ):
        out = actor(obs_device, action=action_device)
        active = (
            out.factor_active[row, actor_indices]
            & eligible_device[row, actor_indices]
        )
        selected_factors = torch.stack([
            _rare_intent_factor_mask(name, active[index])
            for index, (name, _sample_index, _actor_index)
            in enumerate(selected_named)
        ])
        nll, legal, per_actor = _actor_balanced_factor_nll(
            out.factor_logprob[row, actor_indices], selected_factors,
        )
    if not torch.isfinite(nll):
        raise FloatingPointError(f"non-finite rare-intent actor loss {nll}")
    return nll, legal, per_actor


def _train_rare_intent_actor_epoch(
    actor: Actor,
    optimizer: torch.optim.Optimizer,
    replay: list[LifecycleSample],
    refs_by_intent: dict[str, list[tuple[int, int]]],
    *,
    device: torch.device,
    use_bf16: bool,
    minibatch: int,
    shuffle_generator: torch.Generator,
) -> tuple[float, float, int]:
    """One intent-balanced actor-only epoch over semantic rare factors."""
    batches_for_epoch = _rare_intent_actor_batches(
        refs_by_intent, minibatch=minibatch,
        shuffle_generator=shuffle_generator,
    )
    actor.train()
    loss_sum = 0.0
    actor_count = 0
    legal_sum = 0.0
    batches = 0
    for selected_named in batches_for_epoch:
        nll, legal, per_actor = _rare_intent_batch_loss(
            actor, replay, selected_named, device=device, use_bf16=use_bf16,
        )
        optimizer.zero_grad(set_to_none=True)
        nll.backward()
        nn.utils.clip_grad_norm_(
            actor.parameters(), float(PPO_CFG.get("maxGradNorm", 0.5)),
            error_if_nonfinite=True,
        )
        optimizer.step()
        loss_sum += float(per_actor.detach().sum())
        actor_count += len(selected_named)
        legal_sum += float(legal)
        batches += 1
    return (
        loss_sum / max(1, actor_count),
        legal_sum / max(1, batches),
        actor_count,
    )


@torch.inference_mode()
def _evaluate_rare_intent_actors(
    actor: Actor,
    replay: list[LifecycleSample],
    refs_by_intent: dict[str, list[tuple[int, int]]],
    *,
    device: torch.device,
    minibatch: int,
) -> dict[str, float]:
    """Per-intent semantic-factor NLL and deterministic exact accuracy."""
    actor.eval()
    metrics: dict[str, float] = {}
    for name in RARE_ACTOR_INTENTS:
        refs = refs_by_intent.get(name, [])
        nll_sum = 0.0
        factor_accuracy_sum = 0.0
        exact_correct = 0
        type_correct = 0
        count = 0
        legal_sum = 0.0
        batches = 0
        for start in range(0, len(refs), max(1, minibatch)):
            selected = refs[start : start + max(1, minibatch)]
            obs_device, action_device, actor_indices = _rare_actor_batch(
                replay, selected, device=device,
            )
            eligible_device = _rare_eligible_batch(
                replay, selected,
                actor_cap=action_device["types"].shape[1], device=device,
            )
            row = torch.arange(len(selected), device=device)
            teacher = actor(obs_device, action=action_device)
            selected_active = _rare_intent_factor_mask(
                name,
                teacher.factor_active[row, actor_indices]
                & eligible_device[row, actor_indices],
            )
            _nll, legal, per_actor = _actor_balanced_factor_nll(
                teacher.factor_logprob[row, actor_indices], selected_active,
            )
            predicted = actor(obs_device, deterministic=True)
            predicted_values = _action_factor_values(predicted)[row, actor_indices]
            teacher_values = _action_factor_values(action_device)[row, actor_indices]
            factor_matches = predicted_values == teacher_values
            per_actor_factor_accuracy = (
                (factor_matches & selected_active).sum(dim=-1)
                / selected_active.sum(dim=-1)
            )
            exact = (factor_matches | ~selected_active).all(dim=-1)
            nll_sum += float(per_actor.sum())
            factor_accuracy_sum += float(per_actor_factor_accuracy.sum())
            exact_correct += int(exact.sum())
            type_correct += int(
                (predicted.types[row, actor_indices, 0]
                 == action_device["types"][row, actor_indices, 0]).sum()
            )
            count += len(selected)
            legal_sum += float(legal)
            batches += 1
        metrics.update({
            f"{name}_count": float(count),
            f"{name}_nll": nll_sum / max(1, count) if count else float("nan"),
            f"{name}_accuracy": exact_correct / max(1, count),
            f"{name}_factor_accuracy": factor_accuracy_sum / max(1, count),
            f"{name}_type_accuracy": type_correct / max(1, count),
            f"{name}_legal_frac": legal_sum / max(1, batches),
        })
    actor.train()
    return metrics


def _train_critic_replay_epoch(
    critic: Critic,
    optimizer: torch.optim.Optimizer,
    replay: list[CriticSample],
    returns_tn: torch.Tensor,
    *,
    device: torch.device,
    use_bf16: bool,
    minibatch: int,
    value_coef: float = 1.0,
    shuffle_generator: torch.Generator | None = None,
) -> tuple[float, int]:
    """Fit critic-only rows without inventing actor labels."""
    if not replay:
        return float("nan"), 0
    by_stratum: dict[str, list[int]] = {}
    for index, sample in enumerate(replay):
        by_stratum.setdefault(sample.stratum, []).append(index)
    largest = max(len(indices) for indices in by_stratum.values())
    balanced: list[int] = []
    for indices in by_stratum.values():
        repeats = (largest + len(indices) - 1) // len(indices)
        balanced.extend((indices * repeats)[:largest])
    order = torch.tensor(balanced, dtype=torch.long)
    order = order[torch.randperm(order.numel(), generator=shuffle_generator)]
    losses: list[float] = []
    critic.train()
    for start in range(0, order.numel(), max(1, minibatch)):
        selected = order[start : start + max(1, minibatch)].tolist()
        obs_rows = [replay[index].obs for index in selected]
        obs_device = promote_obs_device(_pad_replay_tensors(
            obs_rows,
            actor_cap=max(row["actors"].shape[1] for row in obs_rows),
            target_cap=max(row["targets"].shape[1] for row in obs_rows),
            room_cap=max(row["room_mask"].shape[1] for row in obs_rows),
        ), device, non_blocking=False)
        target = torch.tensor([
            float(returns_tn[replay[index].timestep, replay[index].env_index])
            for index in selected
        ], dtype=torch.float32, device=device)
        with (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_bf16 else nullcontext()
        ):
            logits = critic(obs_device, return_logits=True)
            loss = critic.support.cross_entropy(
                logits.float(), target, validate=False,
            ).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite TI critic loss {loss}")
        optimizer.zero_grad(set_to_none=True)
        (float(value_coef) * loss).backward()
        nn.utils.clip_grad_norm_(
            critic.parameters(), float(VALUE_CFG["criticMaxGradNorm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)), int(order.numel())


@torch.inference_mode()
def _evaluate_critic_replay(
    critic: Critic,
    replay: list[CriticSample],
    returns_tn: torch.Tensor,
    *,
    device: torch.device,
    minibatch: int,
) -> dict[str, float]:
    if not replay:
        return {"value_loss": float("nan"), "ev": float("nan"), "count": 0.0}
    critic.eval()
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    loss_sum = 0.0
    for start in range(0, len(replay), max(1, minibatch)):
        selected = list(range(start, min(len(replay), start + max(1, minibatch))))
        obs_rows = [replay[index].obs for index in selected]
        obs_device = promote_obs_device(_pad_replay_tensors(
            obs_rows,
            actor_cap=max(row["actors"].shape[1] for row in obs_rows),
            target_cap=max(row["targets"].shape[1] for row in obs_rows),
            room_cap=max(row["room_mask"].shape[1] for row in obs_rows),
        ), device, non_blocking=False)
        target = torch.tensor([
            float(returns_tn[replay[index].timestep, replay[index].env_index])
            for index in selected
        ], dtype=torch.float32, device=device)
        logits = critic(obs_device, return_logits=True).float()
        loss_sum += float(critic.support.cross_entropy(
            logits, target, validate=False,
        ).sum())
        predictions.append(critic.support.to_expected_scalar(logits).cpu())
        targets.append(target.cpu())
    target = torch.cat(targets)
    prediction = torch.cat(predictions)
    variance = float(torch.var(target, unbiased=False))
    critic.train()
    return {
        "value_loss": loss_sum / len(replay),
        "ev": float("nan") if variance == 0 else 1.0 - float(
            torch.var(target - prediction, unbiased=False) / variance
        ),
        "count": float(len(replay)),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
    }


def _train_ti_actor_replay(
    actor: Actor,
    optimizer: torch.optim.Optimizer,
    replay: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]],
    *,
    device: torch.device,
    use_bf16: bool,
    minibatch: int,
    shuffle_generator: torch.Generator | None = None,
) -> tuple[float, float]:
    """Train one shuffled epoch over exactly representable TI factors."""
    if not replay:
        return float("nan"), 0.0
    order = torch.randperm(len(replay), generator=shuffle_generator)
    losses: list[float] = []
    coverage: list[float] = []
    actor.train()
    for start in range(0, len(replay), max(1, minibatch)):
        selected = order[start : start + max(1, minibatch)].tolist()
        obs_rows = [replay[index][0] for index in selected]
        action_rows = [replay[index][1] for index in selected]
        actor_cap = max(row["actors"].shape[1] for row in obs_rows)
        target_cap = max(row["targets"].shape[1] for row in obs_rows)
        room_cap = max(row["room_mask"].shape[1] for row in obs_rows)
        obs_host = _pad_replay_tensors(
            obs_rows, actor_cap=actor_cap, target_cap=target_cap, room_cap=room_cap,
        )
        action_host = _pad_replay_tensors(
            action_rows, actor_cap=actor_cap, target_cap=target_cap,
            room_cap=room_cap, actions=True,
        )
        requested = _pad_eligible_masks(
            [replay[index][2] for index in selected], actor_cap=actor_cap,
        ).to(device)
        obs_device = promote_obs_device(obs_host, device, non_blocking=False)
        action_device = {key: value.to(device) for key, value in action_host.items()}
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_bf16 else nullcontext()
        )
        with context:
            out = actor(obs_device, action=action_device)
            # An inactive factor's logprob is a constant zero, so requesting one
            # would silently dilute the mean instead of supervising anything.
            eligible = _supervised_factor_mask(requested, out.factor_active)
            nll, fraction = safe_bc_nll(
                out.factor_logprob, eligible, strict=True,
            )
        if not bool(eligible.any()):
            continue
        optimizer.zero_grad(set_to_none=True)
        nll.backward()
        nn.utils.clip_grad_norm_(
            actor.parameters(), float(PPO_CFG.get("maxGradNorm", 0.5)),
            error_if_nonfinite=True,
        )
        optimizer.step()
        losses.append(float(nll.detach()))
        coverage.append(float(fraction))
    return (
        float(np.mean(losses)) if losses else float("nan"),
        float(np.mean(coverage)) if coverage else 0.0,
    )


@torch.inference_mode()
def _evaluate_ti_actor_replay(
    actor: Actor,
    replay: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]],
    *,
    device: torch.device,
    minibatch: int,
) -> tuple[float, float]:
    """Score TI initialization factors on the final actor, never a stale warm start."""
    if not replay:
        return float("nan"), 0.0
    actor.eval()
    nll_sum = 0.0
    requested_count = 0
    legal_sum = 0.0
    for start in range(0, len(replay), max(1, minibatch)):
        selected = list(range(start, min(len(replay), start + max(1, minibatch))))
        obs_rows = [replay[index][0] for index in selected]
        action_rows = [replay[index][1] for index in selected]
        actor_cap = max(row["actors"].shape[1] for row in obs_rows)
        target_cap = max(row["targets"].shape[1] for row in obs_rows)
        room_cap = max(row["room_mask"].shape[1] for row in obs_rows)
        obs_device = promote_obs_device(_pad_replay_tensors(
            obs_rows, actor_cap=actor_cap, target_cap=target_cap,
            room_cap=room_cap,
        ), device, non_blocking=False)
        action_device = {
            key: value.to(device) for key, value in _pad_replay_tensors(
                action_rows, actor_cap=actor_cap, target_cap=target_cap,
                room_cap=room_cap, actions=True,
            ).items()
        }
        requested = _pad_eligible_masks(
            [replay[index][2] for index in selected], actor_cap=actor_cap,
        ).to(device)
        if not bool(requested.any()):
            continue
        out = actor(obs_device, action=action_device)
        eligible = _supervised_factor_mask(requested, out.factor_active)
        nll, legal = safe_bc_nll(
            out.factor_logprob, eligible, strict=True,
        )
        count = int(eligible.sum())
        nll_sum += float(nll) * count
        legal_sum += float(legal) * count
        requested_count += count
    actor.train()
    return (
        nll_sum / max(1, requested_count),
        legal_sum / max(1, requested_count),
    )


def _pad_replay_tensors(
    rows: list[dict[str, torch.Tensor]],
    *,
    actor_cap: int,
    target_cap: int,
    room_cap: int,
    actions: bool = False,
) -> dict[str, torch.Tensor]:
    """Batch compact replay rows by padding only to this minibatch's maxima."""
    room_keys = {"patches", "room_mask", "room_coords", "construction_mask"}
    actor_keys = {
        "actors", "actor_mask", "actor_outcome", "intent_mask",
        "dir_mask", "amount_mask",
    }
    target_keys = {"targets", "target_mask"}

    def padded(key: str, value: torch.Tensor) -> torch.Tensor:
        if actions or key in actor_keys:
            axis, capacity = 1, actor_cap
        elif key in room_keys:
            axis, capacity = 1, room_cap
        elif key in target_keys:
            axis, capacity = 1, target_cap
        elif key == "target_select_mask":
            axis, capacity = value.dim() - 1, target_cap
        else:
            return value
        if value.shape[axis] == capacity:
            return value
        shape = list(value.shape)
        shape[axis] = capacity
        out = value.new_zeros(shape)
        slices = [slice(None)] * value.dim()
        slices[axis] = slice(0, value.shape[axis])
        out[tuple(slices)].copy_(value)
        return out

    return {
        key: torch.cat([padded(key, row[key]) for row in rows], dim=0)
        for key in rows[0]
    }


def _pad_eligible_masks(
    rows: list[torch.Tensor], *, actor_cap: int,
) -> torch.Tensor:
    """Batch per-row factor-eligibility masks on the shared replay padding path.

    Masks are ``[1, actors, 6 + 2 * N_BODY_PART]`` bool tensors whose actor axis
    is compacted per row exactly like that row's observation and action tensors,
    so they must pad to the same minibatch maximum.  ``actions=True`` pads every
    key on axis 1, which is why the target/room capacities are irrelevant here.
    """
    return _pad_replay_tensors(
        [{"eligible": row} for row in rows],
        actor_cap=actor_cap, target_cap=0, room_cap=0, actions=True,
    )["eligible"]


def _supervised_factor_mask(
    eligible: torch.Tensor, factor_active: torch.Tensor,
) -> torch.Tensor:
    """Factors this teacher labelled that this action also actually consumes.

    Row masks are padded to the minibatch's actor cap, while the promoted batch is
    compacted to a live-actor capacity bucket, which is a prefix of that cap; the
    model trims the action tensors it is handed exactly the same way.  Trim
    identically and then AND, so a factor is supervised only where the teacher and
    the ABI agree.
    """
    return eligible[:, : factor_active.shape[1]] & factor_active


def _temporal_minibatch(
    replay: list[TemporalSample],
    selected: list[int],
    *,
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]
]:
    """Pad current/future states independently and preserve complete actions."""
    obs_rows = [replay[index].obs for index in selected]
    next_rows = [replay[index].next_obs for index in selected]
    action_rows = [replay[index].action for index in selected]
    actor_cap = max(
        max(row["actors"].shape[1] for row in obs_rows),
        max(row["types"].shape[1] for row in action_rows),
    )
    target_cap = max(row["targets"].shape[1] for row in obs_rows)
    room_cap = max(row["room_mask"].shape[1] for row in obs_rows)
    next_actor_cap = max(row["actors"].shape[1] for row in next_rows)
    next_target_cap = max(row["targets"].shape[1] for row in next_rows)
    next_room_cap = max(row["room_mask"].shape[1] for row in next_rows)
    obs = promote_obs_device(_pad_replay_tensors(
        obs_rows, actor_cap=actor_cap, target_cap=target_cap, room_cap=room_cap,
    ), device, non_blocking=False)
    action = {
        key: value.to(device)
        for key, value in _pad_replay_tensors(
            action_rows, actor_cap=actor_cap, target_cap=target_cap,
            room_cap=room_cap, actions=True,
        ).items()
    }
    next_obs = promote_obs_device(_pad_replay_tensors(
        next_rows, actor_cap=next_actor_cap, target_cap=next_target_cap,
        room_cap=next_room_cap,
    ), device, non_blocking=False)
    return obs, action, next_obs


def _whole_joint_none_action(
    obs: dict[str, torch.Tensor],
    action: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Ablate every issued live-actor command to canonical legal ``none``.

    This is derived from the complete action in the evaluated state.  It avoids
    certifying action dependence from one collector-selected command while the
    rest of the joint action remains factual.
    """
    result = {key: value.clone() for key, value in action.items()}
    live = obs["actor_mask"][:, : action["types"].shape[1]].bool()
    issued = live & action["types"][:, :, 0].ne(0)
    for key in (
        "types", "dirs", "targets", "amounts",
        "construction_types", "construction_tiles",
    ):
        result[key].zero_()
    result["body_counts"].zero_()
    identity = torch.arange(
        N_BODY_PART, dtype=result["body_order"].dtype,
        device=result["body_order"].device,
    ).view(1, 1, 1, N_BODY_PART)
    result["body_order"].copy_(identity.expand_as(result["body_order"]))
    return result, issued.any(dim=-1)


@torch.inference_mode()
def _evaluate_nextlat(
    actor: Actor,
    critic: Critic,
    replay: list[TemporalSample],
    *,
    device: torch.device,
    minibatch: int,
) -> dict[str, float]:
    """Evaluate learned, identity, and whole-joint-none predictors exactly once."""
    if len(replay) < 2:
        raise ValueError("NextLat evaluation requires at least two transitions")
    actor.eval()
    critic.eval()
    totals = {
        "actor_mse": 0.0,
        "actor_identity_mse": 0.0,
        "actor_counterfactual_action_mse": 0.0,
        "actor_counterfactual_reference_mse": 0.0,
        "critic_mse": 0.0,
        "critic_identity_mse": 0.0,
        "critic_counterfactual_action_mse": 0.0,
        "critic_counterfactual_reference_mse": 0.0,
        "critic_kl": 0.0,
    }
    count = 0
    counterfactual_count = 0
    for start in range(0, len(replay), max(1, minibatch)):
        selected = list(range(start, min(len(replay), start + max(1, minibatch))))
        obs, action, next_obs = _temporal_minibatch(
            replay, selected, device=device,
        )
        current_actor = actor.encode_state(obs)
        next_actor = actor.encode_state(next_obs).detach()
        predicted_actor = actor.predict_next_latent(current_actor, obs, action)
        _current_logits, current_critic = critic.value_logits_and_latent(obs)
        next_critic_logits, next_critic = critic.value_logits_and_latent(next_obs)
        next_critic = next_critic.detach()
        next_critic_logits = next_critic_logits.detach()
        predicted_critic = critic.predict_next_latent(current_critic, obs, action)
        predicted_logits = critic.detached_value_logits(predicted_critic)
        per_batch = len(selected)
        totals["actor_mse"] += float(F.mse_loss(
            predicted_actor.float(), next_actor.float(), reduction="sum",
        )) / predicted_actor.shape[-1]
        totals["actor_identity_mse"] += float(F.mse_loss(
            current_actor.float(), next_actor.float(), reduction="sum",
        )) / current_actor.shape[-1]
        totals["critic_mse"] += float(F.mse_loss(
            predicted_critic.float(), next_critic.float(), reduction="sum",
        )) / predicted_critic.shape[-1]
        totals["critic_identity_mse"] += float(F.mse_loss(
            current_critic.float(), next_critic.float(), reduction="sum",
        )) / current_critic.shape[-1]
        teacher_logp = F.log_softmax(next_critic_logits.float(), dim=-1)
        student_logp = F.log_softmax(predicted_logits.float(), dim=-1)
        totals["critic_kl"] += float(F.kl_div(
            student_logp, teacher_logp, log_target=True, reduction="sum",
        ))
        counter_action, counter_rows = _whole_joint_none_action(obs, action)
        if bool(counter_rows.any()):
            positions = torch.nonzero(counter_rows, as_tuple=False).flatten()
            counter_actor = actor.predict_next_latent(
                current_actor, obs, counter_action,
            )[positions]
            counter_critic = critic.predict_next_latent(
                current_critic, obs, counter_action,
            )[positions]
            counter_next_actor = next_actor[positions]
            counter_next_critic = next_critic[positions]
            totals["actor_counterfactual_action_mse"] += float(F.mse_loss(
                counter_actor.float(), counter_next_actor.float(), reduction="sum",
            )) / counter_actor.shape[-1]
            totals["actor_counterfactual_reference_mse"] += float(F.mse_loss(
                predicted_actor[positions].float(), counter_next_actor.float(),
                reduction="sum",
            )) / counter_actor.shape[-1]
            totals["critic_counterfactual_action_mse"] += float(F.mse_loss(
                counter_critic.float(), counter_next_critic.float(), reduction="sum",
            )) / counter_critic.shape[-1]
            totals["critic_counterfactual_reference_mse"] += float(F.mse_loss(
                predicted_critic[positions].float(), counter_next_critic.float(),
                reduction="sum",
            )) / counter_critic.shape[-1]
            counterfactual_count += int(counter_rows.sum())
        count += per_batch
    actor.train()
    critic.train()
    counterfactual_keys = {
        "actor_counterfactual_action_mse", "actor_counterfactual_reference_mse",
        "critic_counterfactual_action_mse", "critic_counterfactual_reference_mse",
    }
    result = {
        key: (
            value / counterfactual_count
            if key in counterfactual_keys and counterfactual_count
            else float("nan") if key in counterfactual_keys else value / count
        )
        for key, value in totals.items()
    }
    return result | {
        "rows": float(count),
        "counterfactual_rows": float(counterfactual_count),
        "counterfactual_fraction": counterfactual_count / count,
    }


def _spawn_actor_batches(
    replay: list[SpawnReplayRow],
    *,
    minibatch: int,
    shuffle_generator: torch.Generator | None,
) -> list[list[int]]:
    """Build one exact stratum- and decision-balanced spawn epoch."""
    if not replay:
        return []
    by_stratum: dict[str, list[int]] = {}
    for index, (stratum, *_row) in enumerate(replay):
        by_stratum.setdefault(stratum, []).append(index)
    largest = max(len(indices) for indices in by_stratum.values())
    balanced: list[int] = []
    for indices in by_stratum.values():
        repeats = (largest + len(indices) - 1) // len(indices)
        balanced.extend((indices * repeats)[:largest])
    positive_indices = [
        index for index in balanced if replay[index][0].startswith("spawn:")
    ]
    wait_indices = [
        index for index in balanced if replay[index][0].startswith("waitlegal:")
    ]
    if positive_indices and wait_indices:
        class_size = max(len(positive_indices), len(wait_indices))
        positive_indices = (
            positive_indices
            * ((class_size + len(positive_indices) - 1) // len(positive_indices))
        )[:class_size]
        wait_indices = (
            wait_indices
            * ((class_size + len(wait_indices) - 1) // len(wait_indices))
        )[:class_size]
        balanced = positive_indices + wait_indices
    order = torch.tensor(balanced, dtype=torch.long)
    order = order[torch.randperm(order.numel(), generator=shuffle_generator)]
    width = max(1, int(minibatch))
    return [
        order[start : start + width].tolist()
        for start in range(0, order.numel(), width)
    ]


def _spawn_actor_batch_loss(
    actor: Actor,
    replay: list[SpawnReplayRow],
    selected: list[int],
    *,
    device: torch.device,
    use_bf16: bool,
) -> tuple[torch.Tensor, float]:
    obs_rows = [replay[index][2] for index in selected]
    action_rows = [replay[index][3] for index in selected]
    actor_cap = max(row["actors"].shape[1] for row in obs_rows)
    target_cap = max(row["targets"].shape[1] for row in obs_rows)
    room_cap = max(row["room_mask"].shape[1] for row in obs_rows)
    obs_host = _pad_replay_tensors(
        obs_rows, actor_cap=actor_cap, target_cap=target_cap, room_cap=room_cap,
    )
    action_host = _pad_replay_tensors(
        action_rows, actor_cap=actor_cap, target_cap=target_cap,
        room_cap=room_cap, actions=True,
    )
    obs_device = promote_obs_device(obs_host, device, non_blocking=False)
    action_device = {key: value.to(device) for key, value in action_host.items()}
    eligible_device = _pad_eligible_masks(
        [replay[index][4] for index in selected], actor_cap=actor_cap,
    ).to(device)
    with (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else nullcontext()
    ):
        out = actor(obs_device, action=action_device)
        selected_actor = torch.tensor(
            [replay[index][1] for index in selected],
            dtype=torch.long, device=device,
        )
        row = torch.arange(len(selected), device=device)
        selected_factor_lp = out.factor_logprob[row, selected_actor]
        # Supervise a factor only where this teacher labelled it and the chosen
        # intent actually consumes it.
        selected_eligible = (
            out.factor_active[row, selected_actor]
            & eligible_device[row, selected_actor]
        )
        _raw_nll, fraction = safe_bc_nll(
            selected_factor_lp, selected_eligible, strict=True,
        )
        group_losses: list[torch.Tensor] = []
        type_eligible = selected_eligible[:, 0]
        if bool(type_eligible.any()):
            group_losses.append(-selected_factor_lp[type_eligible, 0].mean())
        chosen_type = action_device["types"][row, selected_actor, 0]
        positive = chosen_type == INTENT_TYPES.index("spawnCreep")
        if bool(positive.any()):
            count_lp = selected_factor_lp[positive, 6 : 6 + N_BODY_PART]
            count_eligible = selected_eligible[positive, 6 : 6 + N_BODY_PART]
            group_losses.append(-torch.where(
                count_eligible, count_lp, torch.zeros_like(count_lp),
            ).sum(dim=-1).mean())
            order_eligible = selected_eligible[positive, 6 + N_BODY_PART :]
            order_lp = selected_factor_lp[positive, 6 + N_BODY_PART :]
            group_losses.append(-torch.where(
                order_eligible, order_lp, torch.zeros_like(order_lp),
            ).sum(dim=-1).mean())
        if not group_losses:
            raise RuntimeError(
                "spawn batch supervises no eligible factor; retention keeps only "
                "rows carrying a labelled body or intent factor"
            )
        nll = torch.stack(group_losses).mean()
    if not torch.isfinite(nll):
        raise FloatingPointError(f"non-finite spawn actor loss {nll}")
    return nll, fraction


def _train_spawn_replay(
    actor: Actor,
    optimizer: torch.optim.Optimizer,
    replay: list[SpawnReplayRow],
    *,
    device: torch.device,
    use_bf16: bool,
    minibatch: int,
    shuffle_generator: torch.Generator | None = None,
) -> tuple[float, float, dict[str, float]]:
    """One balanced spawn-decision epoch; body CE applies only to spawn rows."""
    if not replay:
        return float("nan"), 1.0, {
            "spawn_type_nll": float("nan"), "wait_legal_type_nll": float("nan"),
            "spawn_type_accuracy": float("nan"),
            "wait_legal_type_accuracy": float("nan"),
            "spawn_body_accuracy": float("nan"),
            "spawn_min_stratum_body_accuracy": float("nan"),
        }
    batches_for_epoch = _spawn_actor_batches(
        replay, minibatch=minibatch, shuffle_generator=shuffle_generator,
    )
    losses: list[float] = []
    legal: list[float] = []
    actor.train()
    for selected in batches_for_epoch:
        nll, fraction = _spawn_actor_batch_loss(
            actor, replay, selected, device=device, use_bf16=use_bf16,
        )
        optimizer.zero_grad(set_to_none=True)
        nll.backward()
        nn.utils.clip_grad_norm_(
            actor.parameters(),
            float(PPO_CFG.get("maxGradNorm", 0.5)),
            error_if_nonfinite=True,
        )
        optimizer.step()
        losses.append(float(nll.detach()))
        legal.append(float(fraction))

    # Qualification must describe the final replay-trained policy. Measuring
    # rows immediately after their minibatch can be invalidated by later
    # optimizer steps, especially for the rare legal-wait states.
    type_nll_sum = {"spawn": 0.0, "wait_legal": 0.0}
    type_correct = {"spawn": 0, "wait_legal": 0}
    type_count = {"spawn": 0, "wait_legal": 0}
    spawn_body_correct = 0
    spawn_body_count = 0
    spawn_body_by_stratum: dict[str, list[int]] = {}
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(replay), max(1, int(minibatch))):
            selected = list(range(start, min(len(replay), start + max(1, int(minibatch)))))
            obs_rows = [replay[index][2] for index in selected]
            action_rows = [replay[index][3] for index in selected]
            actor_cap = max(row["actors"].shape[1] for row in obs_rows)
            target_cap = max(row["targets"].shape[1] for row in obs_rows)
            room_cap = max(row["room_mask"].shape[1] for row in obs_rows)
            obs_host = _pad_replay_tensors(
                obs_rows, actor_cap=actor_cap, target_cap=target_cap,
                room_cap=room_cap,
            )
            action_host = _pad_replay_tensors(
                action_rows, actor_cap=actor_cap, target_cap=target_cap,
                room_cap=room_cap, actions=True,
            )
            obs_device = promote_obs_device(obs_host, device, non_blocking=False)
            action_device = {key: value.to(device) for key, value in action_host.items()}
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16 else nullcontext()
            )
            with context:
                out = actor(obs_device, action=action_device)
            selected_actor = torch.tensor(
                [replay[index][1] for index in selected],
                dtype=torch.long, device=device,
            )
            row = torch.arange(len(selected), device=device)
            chosen_type = action_device["types"][row, selected_actor, 0]
            deterministic = actor(obs_device, deterministic=True)
            predicted = deterministic.types[row, selected_actor, 0]
            strata = [replay[index][0] for index in selected]
            selectors = {
                "spawn": chosen_type == INTENT_TYPES.index("spawnCreep"),
                "wait_legal": torch.tensor(
                    [stratum.startswith("waitlegal:") for stratum in strata],
                    dtype=torch.bool, device=device,
                ),
            }
            selected_type_lp = out.factor_logprob[row, selected_actor, 0]
            positive = selectors["spawn"]
            if bool(positive.any()):
                exact_body = (
                    deterministic.body_counts[row, selected_actor, 0]
                    == action_device["body_counts"][row, selected_actor, 0]
                ).all(dim=-1) & (
                    deterministic.body_order[row, selected_actor, 0]
                    == action_device["body_order"][row, selected_actor, 0]
                ).all(dim=-1)
                for local_index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
                    correct, total = spawn_body_by_stratum.setdefault(
                        strata[local_index], [0, 0],
                    )
                    is_correct = int(exact_body[local_index])
                    spawn_body_by_stratum[strata[local_index]] = [
                        correct + is_correct, total + 1,
                    ]
                    spawn_body_correct += is_correct
                    spawn_body_count += 1
            for name, selector in selectors.items():
                count = int(selector.sum())
                if count:
                    type_nll_sum[name] += float(-selected_type_lp[selector].sum())
                    type_correct[name] += int(
                        (predicted[selector] == chosen_type[selector]).sum()
                    )
                    type_count[name] += count
    diagnostics = {
        f"{name}_type_{metric}": (
            type_nll_sum[name] / max(1, type_count[name])
            if metric == "nll" else type_correct[name] / max(1, type_count[name])
        )
        for name in ("spawn", "wait_legal")
        for metric in ("nll", "accuracy")
    }
    diagnostics.update({
        "spawn_body_accuracy": spawn_body_correct / max(1, spawn_body_count),
        "spawn_min_stratum_body_accuracy": min(
            (correct / total for correct, total in spawn_body_by_stratum.values()),
            default=float("nan"),
        ),
    })
    actor.train()
    return float(np.mean(losses)), float(np.mean(legal)), diagnostics


@torch.inference_mode()
def _evaluate_spawn_replay(
    actor: Actor,
    replay: list[SpawnReplayRow],
    *,
    device: torch.device,
    use_bf16: bool,
    minibatch: int,
) -> dict[str, float]:
    """Evaluate spawn semantics after the complete fused actor epoch."""
    if not replay:
        return {
            "spawn_type_nll": float("nan"),
            "wait_legal_type_nll": float("nan"),
            "spawn_type_accuracy": float("nan"),
            "wait_legal_type_accuracy": float("nan"),
            "spawn_body_accuracy": float("nan"),
            "spawn_min_stratum_body_accuracy": float("nan"),
        }
    type_nll_sum = {"spawn": 0.0, "wait_legal": 0.0}
    type_correct = {"spawn": 0, "wait_legal": 0}
    type_count = {"spawn": 0, "wait_legal": 0}
    spawn_body_correct = 0
    spawn_body_count = 0
    spawn_body_by_stratum: dict[str, list[int]] = {}
    actor.eval()
    for start in range(0, len(replay), max(1, int(minibatch))):
        selected = list(range(start, min(len(replay), start + max(1, int(minibatch)))))
        obs_rows = [replay[index][2] for index in selected]
        action_rows = [replay[index][3] for index in selected]
        actor_cap = max(row["actors"].shape[1] for row in obs_rows)
        target_cap = max(row["targets"].shape[1] for row in obs_rows)
        room_cap = max(row["room_mask"].shape[1] for row in obs_rows)
        obs_device = promote_obs_device(_pad_replay_tensors(
            obs_rows, actor_cap=actor_cap, target_cap=target_cap, room_cap=room_cap,
        ), device, non_blocking=False)
        action_device = {
            key: value.to(device) for key, value in _pad_replay_tensors(
                action_rows, actor_cap=actor_cap, target_cap=target_cap,
                room_cap=room_cap, actions=True,
            ).items()
        }
        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16 else nullcontext()
        ):
            out = actor(obs_device, action=action_device)
            deterministic = actor(obs_device, deterministic=True)
        selected_actor = torch.tensor(
            [replay[index][1] for index in selected],
            dtype=torch.long, device=device,
        )
        row = torch.arange(len(selected), device=device)
        chosen_type = action_device["types"][row, selected_actor, 0]
        predicted = deterministic.types[row, selected_actor, 0]
        strata = [replay[index][0] for index in selected]
        selectors = {
            "spawn": chosen_type == INTENT_TYPES.index("spawnCreep"),
            "wait_legal": torch.tensor(
                [stratum.startswith("waitlegal:") for stratum in strata],
                dtype=torch.bool, device=device,
            ),
        }
        selected_type_lp = out.factor_logprob[row, selected_actor, 0]
        positive = selectors["spawn"]
        if bool(positive.any()):
            exact_body = (
                deterministic.body_counts[row, selected_actor, 0]
                == action_device["body_counts"][row, selected_actor, 0]
            ).all(dim=-1) & (
                deterministic.body_order[row, selected_actor, 0]
                == action_device["body_order"][row, selected_actor, 0]
            ).all(dim=-1)
            for local_index in torch.nonzero(
                positive, as_tuple=False,
            ).flatten().tolist():
                correct, total = spawn_body_by_stratum.setdefault(
                    strata[local_index], [0, 0],
                )
                is_correct = int(exact_body[local_index])
                spawn_body_by_stratum[strata[local_index]] = [
                    correct + is_correct, total + 1,
                ]
                spawn_body_correct += is_correct
                spawn_body_count += 1
        for name, selector in selectors.items():
            count = int(selector.sum())
            if count:
                type_nll_sum[name] += float(-selected_type_lp[selector].sum())
                type_correct[name] += int(
                    (predicted[selector] == chosen_type[selector]).sum()
                )
                type_count[name] += count
    actor.train()
    diagnostics = {
        f"{name}_type_nll": type_nll_sum[name] / max(1, type_count[name])
        for name in ("spawn", "wait_legal")
    }
    diagnostics.update({
        f"{name}_type_accuracy": type_correct[name] / max(1, type_count[name])
        for name in ("spawn", "wait_legal")
    })
    diagnostics.update({
        "spawn_body_accuracy": spawn_body_correct / max(1, spawn_body_count),
        "spawn_min_stratum_body_accuracy": min(
            (correct / total for correct, total in spawn_body_by_stratum.values()),
            default=float("nan"),
        ),
    })
    return diagnostics


def _nextlat_checkpoint_metadata(
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "nextlat": dict(SCHEMA["nextLat"]),
        "nextlat_pretrain": {
            "actorCoef": float(args.nextlat_actor_coef),
            "criticCoef": float(args.nextlat_critic_coef),
            "criticKlCoef": float(args.nextlat_critic_kl_coef),
            "loss": "smooth_l1_latent+detached_value_probe_kl",
            "target": "detached_next_observation_encoding",
            "optimizerAuthority": JOINT_OBJECTIVE_AUTHORITY,
            "temporalSampling": "independent_cyclic_shuffle_per_joint_minibatch",
            "action_pooling": NEXTLAT_ACTION_POOLING,
            "actionAblation": NEXTLAT_ACTION_ABLATION,
            "storedCounterfactualUse": "provenance_only_not_evaluation",
            "minRelativeGap": float(args.min_nextlat_relative_gap),
            "minCounterfactualRows": int(args.min_nextlat_counterfactual_rows),
        },
        "nextlat_action_pooling": NEXTLAT_ACTION_POOLING,
        "nextlat_train_rows": int(getattr(args, "nextlat_train_rows", 0)),
        "nextlat_holdout_rows": int(getattr(args, "nextlat_holdout_rows", 0)),
        **{f"nextlat_{key}": float(value) for key, value in metrics.items()},
    }


def _precision_resume_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "effective_precision": str(args.effective_precision),
        "training_device_type": str(args.training_device_type),
        "visible_cuda_device_count": int(args.visible_cuda_device_count),
    }


def _joint_checkpoint(
    actor: Actor,
    critic: Critic,
    opt_a: torch.optim.Optimizer,
    opt_c: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    step: int,
    global_epoch: int,
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
    corpus_sha256: str,
    shuffle_generator: torch.Generator,
    rare_shuffle_generator: torch.Generator,
    nextlat_shuffle_generator: torch.Generator,
    spawn_shuffle_generator: torch.Generator,
    nextlat_metrics: dict[str, float],
) -> dict:
    skill = (total_harvest + total_control) / max(1, step * args.num_envs)
    return {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_opt": opt_a.state_dict(),
        "critic_opt": opt_c.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if args.training_device_type == "cuda" else None
        ),
        "shuffle_rng_state": shuffle_generator.get_state(),
        "rare_shuffle_rng_state": rare_shuffle_generator.get_state(),
        "nextlat_shuffle_rng_state": nextlat_shuffle_generator.get_state(),
        "spawn_shuffle_rng_state": spawn_shuffle_generator.get_state(),
        "meta": artifact_meta(
            "joint_pretrain",
            actor,
            critic,
            source_sha256=args.runtime_source_sha256,
            partial=bool(partial),
            qualified=False,
            corpus_sha256=str(corpus_sha256),
            corpus_schema_sha256=getattr(args, "corpus_schema_sha256", None),
            corpus_collection_source_sha256=getattr(
                args, "corpus_collection_source_sha256", None,
            ),
            corpus_spawn_refresh_source_sha256=getattr(
                args, "corpus_spawn_refresh_source_sha256", None,
            ),
            **_nextlat_checkpoint_metadata(args, nextlat_metrics),
            **_precision_resume_identity(args),
            expert=args.expert,
            curriculum=args.curriculum,
            room=args.room,
            num_envs=args.num_envs,
            steps=args.steps,
            step=int(step),
            global_epoch=int(global_epoch),
            chunk=args.chunk,
            global_epochs=args.global_epochs,
            minibatch=args.minibatch,
            max_episode=args.max_episode,
            gamma=args.gamma,
            critic_target="finite_horizon_discounted_return",
            critic_endpoint="zero_at_declared_lifecycle_horizon",
            value_coef=args.value_coef,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            optimizer=OPTIMIZER_KIND,
            muon_orthogonalization="polar_express_5",
            muon_variance_reduction="normuon_low_rank_beta2",
            muon_lr=args.muon_lr,
            muon_weight_decay=args.muon_weight_decay,
            muon_decay="cautious_update_agreement",
            muon_momentum_min=MUON_MOMENTUM_MIN,
            muon_momentum_max=MUON_MOMENTUM_MAX,
            muon_momentum_warmup_steps=args.muon_momentum_warmup_steps,
            adamw_weight_decay=0.0,
            seed=args.seed,
            collection_seed=getattr(args, "collection_seed", None),
            ti_actor_steps=args.ti_actor_steps,
            ti_replay_capacity=getattr(args, "ti_replay_capacity", None),
            ti_critic_replay_per_stratum=getattr(
                args, "ti_critic_replay_per_stratum", None,
            ),
            ti_critic_pretrain_epochs=args.ti_critic_pretrain_epochs,
            ti_critic_training="initialization_only_before_scripted_value_epochs",
            ti_actor_training=TI_ACTOR_TRAINING,
            actor_auxiliary_schedule=ACTOR_AUXILIARY_SCHEDULE,
            ti_bot_dir=getattr(args, "ti_bot_dir", None),
            ti_bot_source_sha256=getattr(args, "ti_bot_source_sha256", None),
            holdout_seed_offset=args.holdout_seed_offset,
            closed_loop_steps=args.closed_loop_steps,
            evaluation_seed_offset=args.evaluation_seed_offset,
            evaluation_seed=int(args.collection_seed) + args.evaluation_seed_offset,
            min_validation_ev=args.min_validation_ev,
            rare_actor_intents=list(RARE_ACTOR_INTENTS),
            rare_actor_objective=(
                "intent_balanced_actor_mean_semantic_factor_nll;"
                "construction=intent+structure_type+tile;claim=intent+target"
            ),
            max_rare_intent_nll=args.max_rare_intent_nll,
            min_rare_intent_accuracy=args.min_rare_intent_accuracy,
            min_rare_intent_rows=args.min_rare_intent_rows,
            min_spawn_replay_accuracy=args.min_spawn_replay_accuracy,
            max_aux_lifecycle_nll_ratio=args.max_aux_lifecycle_nll_ratio,
            min_closed_loop_rate=args.min_closed_loop_rate,
            min_closed_loop_creeps=args.min_closed_loop_creeps,
            min_closed_loop_claims=args.min_closed_loop_claims,
            min_outpost_closed_loop_success_rate=(
                args.min_outpost_closed_loop_success_rate
            ),
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
            resume_env_mode="immutable_corpus",
            wall_s=float(wall_s),
        ),
        "update": -1,
        "global_step": int(global_step),
    }


def _rare_intent_replay_qualified(
    validation: dict[str, float],
    intent: str,
    *,
    min_rows: int,
    max_nll: float,
    min_accuracy: float,
) -> bool:
    """Require independent train and holdout support plus semantic quality."""
    return all(
        float(validation.get(f"rare_{split}_{intent}_count", 0.0)) >= min_rows
        and float(validation.get(f"rare_{split}_{intent}_legal_frac", 0.0)) == 1.0
        and np.isfinite(float(validation.get(
            f"rare_{split}_{intent}_nll", float("nan"),
        )))
        and float(validation[f"rare_{split}_{intent}_nll"]) <= max_nll
        and float(validation.get(f"rare_{split}_{intent}_accuracy", 0.0))
        >= min_accuracy
        for split in ("train", "holdout")
    )


def _auxiliary_schedule_qualified(validation: dict[str, float]) -> bool:
    """Validate persisted exact fused-lane geometry and collision-free capacity.

    Every joint epoch persists this geometry, so an artifact that lacks it was
    not produced by the fused schedule and cannot be qualified against it.
    """
    prefix = "nextlat_train_"
    names = ("rare", "spawn")
    required_keys = {
        f"{prefix}optimizer_steps", f"{prefix}auxiliary_batches",
        f"{prefix}auxiliary_minibatch",
        *(f"{prefix}{name}_batches" for name in names),
        *(f"{prefix}{name}_exposures" for name in names),
    }
    if not required_keys.issubset(validation):
        return False
    values = {key: float(validation[key]) for key in required_keys}
    if not all(np.isfinite(value) and value >= 0 and value.is_integer()
               for value in values.values()):
        return False
    primary_steps = int(values[f"{prefix}optimizer_steps"])
    total_batches = int(values[f"{prefix}auxiliary_batches"])
    auxiliary_minibatch = int(values[f"{prefix}auxiliary_minibatch"])
    lane_batches = [int(values[f"{prefix}{name}_batches"]) for name in names]
    if (
        primary_steps <= 0 or auxiliary_minibatch <= 0
        or total_batches != sum(lane_batches)
    ):
        return False
    if total_batches > primary_steps:
        return False
    for name, batches in zip(names, lane_batches, strict=True):
        exposures = int(values[f"{prefix}{name}_exposures"])
        if batches == 0:
            if exposures != 0:
                return False
        elif not (
            (batches - 1) * auxiliary_minibatch
            < exposures <= batches * auxiliary_minibatch
        ):
            return False
    return True


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
    teacher_spawn_bodies: dict[str, dict[str, Any]],
    validation: dict[str, float],
    closed_loop: dict[str, object],
) -> list[str]:
    empty_teacher = teacher_by_curriculum.get("empty", {})
    outpost_teacher = teacher_by_curriculum.get("seed_outpost", {})
    teacher_spawn_coverage = _teacher_spawn_coverage(teacher_by_curriculum)

    def relative_improvement(value: str, baseline: str) -> bool:
        learned = float(validation.get(value, float("nan")))
        reference = float(validation.get(baseline, float("nan")))
        return (
            np.isfinite(learned) and np.isfinite(reference)
            and (reference - learned) / max(abs(reference), 1e-8)
            >= args.min_nextlat_relative_gap
        )

    def relative_action_gap(counterfactual: str, reference: str) -> bool:
        counterfactual_value = float(validation.get(counterfactual, float("nan")))
        reference_value = float(validation.get(reference, float("nan")))
        return (
            np.isfinite(counterfactual_value) and np.isfinite(reference_value)
            and (counterfactual_value - reference_value) / max(abs(reference_value), 1e-8)
            >= args.min_nextlat_relative_gap
        )

    def rare_intent_passes(intent: str) -> bool:
        return _rare_intent_replay_qualified(
            validation, intent,
            min_rows=int(getattr(args, "min_rare_intent_rows", 1)),
            max_nll=float(args.max_rare_intent_nll),
            min_accuracy=float(args.min_rare_intent_accuracy),
        )
    closed_loop_stages = closed_loop.get("closed_loop_by_curriculum", {})
    empty_closed = (
        closed_loop_stages.get("empty", {})
        if isinstance(closed_loop_stages, dict)
        else {}
    )
    outpost_closed = (
        closed_loop_stages.get("seed_outpost", {})
        if isinstance(closed_loop_stages, dict)
        else {}
    )
    claimer_closed = (
        closed_loop_stages.get("seed_claimer", {})
        if isinstance(closed_loop_stages, dict)
        else {}
    )
    closed_loop_envs = closed_loop.get("closed_loop_by_env", {})
    outpost_envs = [
        metrics for metrics in (
            closed_loop_envs.values() if isinstance(closed_loop_envs, dict) else ()
        )
        if metrics.get("curriculum") == "seed_outpost"
    ]
    outpost_successes = sum(
        float(metrics.get("remote_harvest", 0.0)) > 0
        and float(metrics.get("remote_home_delivery", 0.0)) > 0
        and float(metrics.get("remote_staffed_peak", 0.0)) >= 1
        and float(metrics.get("remote_productive_peak", 0.0)) >= 1
        and float(metrics.get("claims", 0.0)) == 0
        and float(metrics.get("remote_owned_peak", 0.0)) == 0
        and float(metrics.get("late_remote_harvest", 0.0)) > 0
        and float(metrics.get("late_remote_home_delivery", 0.0)) > 0
        and float(metrics.get("late_remote_staffed_ticks", 0.0)) > 0
        and float(metrics.get("late_remote_productive_ticks", 0.0)) > 0
        and float(metrics.get("invalid_frac", 1.0)) <= MAX_CLOSED_LOOP_INVALID_FRAC
        for metrics in outpost_envs
    )
    outpost_success_rate = outpost_successes / max(1, len(outpost_envs))
    # What the spawn worlds evaluate is "does the policy choose the teacher's
    # body in this world", so the reference is the body The International was
    # measured choosing there, read from the corpus rows it was labelled into.
    # The policy is probed over a window that reaches the teacher's own decision
    # tick, and its first body in that window is the decision under test; later
    # bodies are spawned into an economy the teacher never demonstrated and
    # therefore have no measured reference at all.
    closed_spawn_ok = (
        isinstance(closed_loop_stages, dict)
        and set(teacher_spawn_bodies) == set(SPAWN_CURRICULA)
        and all(
            float(closed_loop_stages.get(stage, {}).get("spawn_success", 0.0)) > 0
            and _spawn_body_matches(
                closed_loop_stages.get(stage, {}).get("spawn_body_parts"),
                teacher_spawn_bodies[stage],
            )
            for stage in SPAWN_CURRICULA
        )
    )
    if hasattr(args, "max_aux_lifecycle_nll_ratio"):
        # Production records this immediately before the final fused epoch, so
        # the gate detects broad-policy regression caused inside that epoch.
        # The old key remains a compatibility fallback for focused callers.
        after_joint_nll = float(validation.get(
            "lifecycle_before_final_joint_train_nll",
            validation.get("lifecycle_after_joint_train_nll", float("nan")),
        ))
        final_lifecycle_nll = float(validation.get(
            "lifecycle_final_train_nll", float("nan"),
        ))
        max_aux_nll_ratio = float(args.max_aux_lifecycle_nll_ratio)
    else:
        # Compatibility for focused callers predating this qualification field;
        # production parse_args always supplies it and therefore stays fail-closed.
        after_joint_nll = final_lifecycle_nll = float(
            corpus_hist.get("_lifecycle_replay_nll", float("nan"))
        )
        max_aux_nll_ratio = 1.1
    checks = {
        "finite_actor_loss": np.isfinite(last_nll),
        "finite_critic_loss": np.isfinite(last_vloss),
        "teacher_factor_legality": float(corpus_hist.get("_bc_legal_frac", 0.0)) == 1.0,
        "spawn_replay_legality": (
            float(corpus_hist.get("_spawn_replay_legal_frac", 0.0)) == 1.0
        ),
        "spawn_replay_semantics": (
            float(corpus_hist.get("_spawn_replay_size", 0.0)) >= len(SPAWN_CURRICULA)
            and np.isfinite(float(corpus_hist.get("_spawn_replay_nll", float("nan"))))
            and float(corpus_hist.get("_spawn_replay_wait_legal_size", 0.0)) >= 3.0
            and float(corpus_hist.get("_spawn_replay_wait_legal_strata", 0.0)) >= 3.0
            and np.isfinite(float(
                corpus_hist.get("_spawn_replay_wait_legal_type_nll", float("nan"))
            ))
            and float(
                corpus_hist.get("_spawn_replay_wait_legal_type_accuracy", 0.0)
            ) >= 0.8
            and float(
                corpus_hist.get("_spawn_replay_spawn_type_accuracy", 0.0)
            ) >= args.min_spawn_replay_accuracy
            and float(
                corpus_hist.get("_spawn_replay_spawn_body_accuracy", 0.0)
            ) >= args.min_spawn_replay_accuracy
            and float(corpus_hist.get(
                "_spawn_replay_spawn_min_stratum_body_accuracy", 0.0,
            )) >= args.min_spawn_replay_accuracy
            and all(
                float(corpus_hist.get(f"_spawn_replay_budget_{bucket}", 0.0)) > 0
                for bucket in SPAWN_BUDGET_BUCKETS
            )
            # Body length is required only where this economy can pay for it;
            # `_spawn_replay_length_unreached_*` records the rest (see
            # SPAWN_LENGTH_UNREACHED_BUCKETS).
            and all(
                float(corpus_hist.get(f"_spawn_replay_length_{bucket}", 0.0)) > 0
                for bucket in SPAWN_LENGTH_REQUIRED_BUCKETS
            )
        ),
        "ti_actor_replay": (
            getattr(args, "ti_actor_steps", 0) <= 0
            or (
                float(corpus_hist.get("_ti_actor_replay_rows", 0.0)) > 0
                and np.isfinite(float(corpus_hist.get("_ti_actor_nll", float("nan"))))
                and float(corpus_hist.get("_ti_actor_legal_coverage", 0.0)) >= 0.9
            )
        ),
        "auxiliary_schedule_capacity": _auxiliary_schedule_qualified(validation),
        "scripted_lifecycle_replay": (
            float(corpus_hist.get("_lifecycle_replay_size", 0.0)) > 0
            and float(corpus_hist.get("_lifecycle_holdout_size", 0.0)) > 0
            and np.isfinite(float(corpus_hist.get("_lifecycle_replay_nll", float("nan"))))
            and float(corpus_hist.get("_lifecycle_replay_legal_frac", 0.0)) == 1.0
            and float(corpus_hist.get("_lifecycle_holdout_legal_frac", 0.0)) == 1.0
            and all(
                float(corpus_hist.get(f"_lifecycle_replay_{category}", 0.0)) > 0
                for category in ("spawn", "harvest", "logistics", "construction", "control")
            )
        ),
        **{
            f"rare_intent_{intent}": rare_intent_passes(intent)
            for intent in RARE_ACTOR_INTENTS
        },
        "teacher_skill": skill > 0,
        "teacher_delivery": total_delivered >= args.min_teacher_delivery,
        "teacher_build": total_built >= args.min_teacher_build,
        "teacher_claim": total_claims >= args.min_teacher_claims,
        "teacher_population": max_creeps >= args.min_teacher_creeps,
        # The empty teacher certifies autonomous bootstrap economy. Claiming and
        # neutral-room remoting have separate observable curricula below; do
        # not conflate ownership expansion with ordinary remote harvesting.
        "empty_teacher_coverage": float(empty_teacher.get("transitions", 0)) > 0,
        # `stepExpert` carries no intent summary, so teacher engine legality is
        # not measurable from the expert wire: it is asserted at collection time
        # instead (the labeller validates every factor against that tick's
        # candidate masks, and the spawn contract requires the engine to have
        # executed each labelled body).
        "empty_teacher_skill": float(empty_teacher.get("skill", 0)) > 0,
        "empty_teacher_delivery": (
            float(empty_teacher.get("delivery", 0)) >= args.min_teacher_delivery
        ),
        "empty_teacher_build": float(empty_teacher.get("build", 0)) >= args.min_teacher_build,
        "outpost_teacher_coverage": float(outpost_teacher.get("transitions", 0)) > 0,
        "outpost_teacher_remote_harvest": (
            float(outpost_teacher.get("remote_harvest", 0)) > 0
        ),
        "outpost_teacher_home_delivery": (
            float(outpost_teacher.get("remote_home_delivery", 0)) > 0
        ),
        "outpost_teacher_productive": (
            float(outpost_teacher.get("remote_staffed_peak", 0)) >= 1
            and float(outpost_teacher.get("remote_productive_peak", 0)) >= 1
        ),
        "outpost_teacher_late_activity": (
            float(outpost_teacher.get("late_transitions", 0)) > 0
            and float(outpost_teacher.get("late_remote_harvest", 0)) > 0
            and float(outpost_teacher.get("late_remote_home_delivery", 0)) > 0
            and float(outpost_teacher.get("late_remote_staffed_ticks", 0)) > 0
            and float(outpost_teacher.get("late_remote_productive_ticks", 0)) > 0
        ),
        "outpost_teacher_stays_neutral": (
            float(outpost_teacher.get("claims", 0)) == 0
            and float(outpost_teacher.get("remote_owned_peak", 0)) == 0
            and float(outpost_teacher.get("neutral_outposts", 0)) >= 1
        ),
        # Dedicated spawn scenarios broaden body supervision, while the empty
        # gates above still independently require an economy-funded expansion.
        # Every count below is measured from the teacher's own labelled bodies.
        # Energy-budget breadth is required outright; body-length breadth is
        # required for the buckets this economy can fund, and the unreached ones
        # are recorded in the checkpoint as a named coverage gap.
        "teacher_spawn_body_coverage": (
            teacher_spawn_coverage["teacher_spawn_labels"] >= len(SPAWN_CURRICULA)
            and all(
                teacher_spawn_coverage[f"teacher_spawn_budget_{bucket}"] > 0
                for bucket in SPAWN_BUDGET_BUCKETS
            )
            and all(
                teacher_spawn_coverage[f"teacher_spawn_length_{bucket}"] > 0
                for bucket in SPAWN_LENGTH_REQUIRED_BUCKETS
            )
        ),
        # Sourced from the TI factor-row evaluation of the final actor; the
        # per-stage spawn body evidence now lives in `spawn_replay_semantics`
        # (corpus rows) and `closed_loop_spawn_scenarios` (engine execution).
        "validation_actor_loss": np.isfinite(validation["validation_factor_nll"]),
        "validation_critic_loss": np.isfinite(
            validation["lifecycle_holdout_value_loss"]
        ),
        "validation_legality": validation["validation_legal_frac"] == 1.0,
        "nextlat_actor_beats_identity": (
            relative_improvement(
                "nextlat_holdout_actor_mse",
                "nextlat_holdout_actor_identity_mse",
            )
        ),
        "nextlat_actor_uses_action": (
            relative_action_gap(
                "nextlat_holdout_actor_counterfactual_action_mse",
                "nextlat_holdout_actor_counterfactual_reference_mse",
            )
        ),
        "nextlat_critic_beats_identity": (
            relative_improvement(
                "nextlat_holdout_critic_mse",
                "nextlat_holdout_critic_identity_mse",
            )
        ),
        "nextlat_critic_uses_action": (
            relative_action_gap(
                "nextlat_holdout_critic_counterfactual_action_mse",
                "nextlat_holdout_critic_counterfactual_reference_mse",
            )
        ),
        "nextlat_action_baseline_coverage": (
            float(validation.get("nextlat_holdout_counterfactual_rows", 0.0))
            >= args.min_nextlat_counterfactual_rows
        ),
        "auxiliary_lifecycle_retention": _auxiliary_lifecycle_retained(
            after_joint_nll, final_lifecycle_nll,
            max_ratio=max_aux_nll_ratio,
        ),
        "lifecycle_holdout_critic_ev": (
            np.isfinite(validation["lifecycle_holdout_value_ev"])
            and validation["lifecycle_holdout_value_ev"] >= args.min_validation_ev
        ),
        "lifecycle_holdout_stage_ev": all(
            np.isfinite(float(validation.get(
                f"lifecycle_holdout_stage_{stage}_ev", float("nan")
            )))
            and float(validation[f"lifecycle_holdout_stage_{stage}_ev"])
            >= args.min_validation_ev
            for stage in (part.strip() for part in args.curriculum.split(","))
            if stage
        ),
        "closed_loop_skill": (
            closed_loop["closed_loop_skill_rate"] >= args.min_closed_loop_rate
        ),
        "closed_loop_legality": (
            closed_loop["closed_loop_invalid_frac"] <= MAX_CLOSED_LOOP_INVALID_FRAC
        ),
        "empty_closed_loop_coverage": float(empty_closed.get("transitions", 0)) > 0,
        "empty_closed_loop_skill": (
            float(empty_closed.get("skill_rate", 0)) >= args.min_closed_loop_rate
        ),
        "empty_closed_loop_control": float(empty_closed.get("control", 0)) > 0,
        "empty_closed_loop_population": (
            float(empty_closed.get("max_creeps", 0)) >= args.min_closed_loop_creeps
        ),
        "empty_closed_loop_legality": (
            float(empty_closed.get("invalid_frac", 1))
            <= MAX_CLOSED_LOOP_INVALID_FRAC
        ),
        "empty_closed_loop_delivery": (
            float(empty_closed.get("delivery", 0)) >= args.min_teacher_delivery
        ),
        "empty_closed_loop_build": (
            float(empty_closed.get("build", 0)) >= args.min_teacher_build
        ),
        "outpost_closed_loop_coverage": (
            float(outpost_closed.get("transitions", 0)) > 0
        ),
        "outpost_closed_loop_remote_harvest": (
            float(outpost_closed.get("remote_harvest", 0)) > 0
        ),
        "outpost_closed_loop_home_delivery": (
            float(outpost_closed.get("remote_home_delivery", 0)) > 0
        ),
        "outpost_closed_loop_productive": (
            float(outpost_closed.get("remote_productive_peak", 0)) >= 1
            and float(outpost_closed.get("remote_staffed_peak", 0)) >= 1
        ),
        "outpost_closed_loop_late_activity": (
            float(outpost_closed.get("late_transitions", 0)) > 0
            and float(outpost_closed.get("late_remote_harvest", 0)) > 0
            and float(outpost_closed.get("late_remote_home_delivery", 0)) > 0
            and float(outpost_closed.get("late_remote_staffed_ticks", 0)) > 0
            and float(outpost_closed.get("late_remote_productive_ticks", 0)) > 0
        ),
        "outpost_closed_loop_stays_neutral": (
            float(outpost_closed.get("claims", 0)) == 0
            and float(outpost_closed.get("remote_owned_peak", 0)) == 0
        ),
        "outpost_closed_loop_legality": (
            float(outpost_closed.get("invalid_frac", 1))
            <= MAX_CLOSED_LOOP_INVALID_FRAC
        ),
        "outpost_closed_loop_success_rate": (
            bool(outpost_envs)
            and outpost_success_rate >= args.min_outpost_closed_loop_success_rate
        ),
        "seed_claimer_closed_loop_claim": (
            float(claimer_closed.get("claims", 0)) >= args.min_closed_loop_claims
            and float(claimer_closed.get("remote_owned_peak", 0)) >= 1
        ),
        "closed_loop_spawn_scenarios": closed_spawn_ok,
    }
    return [name for name, passed in checks.items() if not bool(passed)]



def _merge_teacher_metrics(*sources: dict) -> dict[str, dict[str, float]]:
    """Merge independently collected teacher lanes with their original reducers."""
    merged: dict[str, dict[str, float]] = {}
    for source in sources:
        for stage, raw_metrics in (source or {}).items():
            metrics = merged.setdefault(str(stage), {})
            for key, raw_value in raw_metrics.items():
                if not isinstance(raw_value, (int, float, np.number)):
                    continue
                value = float(raw_value)
                if key.endswith("_min"):
                    metrics[key] = min(metrics.get(key, value), value)
                elif key.endswith("_max") or key == "max_creeps":
                    metrics[key] = max(metrics.get(key, value), value)
                else:
                    metrics[key] = metrics.get(key, 0.0) + value
    return merged


def _split_with_alias(data: dict, *names: str) -> dict | None:
    for name in names:
        value = data.get(name)
        if isinstance(value, dict):
            return value
    return None


def main() -> int:
    # Intra-op parallelism makes environment workers contend; see
    # vec_env.configure_host_threads.
    configure_host_threads()
    args = parse_args()
    args.runtime_source_sha256 = source_signature()
    try:
        from samples.rl.agent.pretrain_corpus import load_corpus
    except ImportError:
        from agent.pretrain_corpus import load_corpus
    try:
        corpus = load_corpus(args.corpus)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"[joint] invalid corpus {args.corpus}: {error}") from error
    meta = corpus["meta"]
    data = corpus["data"]
    corpus_sha256 = str(corpus["integrity"]["corpus_sha256"])
    # The immutable corpus is an offline dataset, not executable code. Its
    # content hash, tensor/action schema, returns, and collection provenance are
    # validated by load_corpus; later collector refactors must not invalidate
    # already-audited training data. Fresh learned-policy qualification still
    # runs against the current environment implementation.
    args.corpus_collection_source_sha256 = meta.get("collection_source_sha256")
    refresh = meta.get("spawn_contract_refresh")
    args.corpus_spawn_refresh_source_sha256 = (
        refresh.get("collection_source_sha256")
        if isinstance(refresh, dict) else None
    )

    # Collection facts have exactly one authority: the validated corpus. They
    # are attached to args only for the existing evaluation/qualification ABI.
    args.expert = str(meta["expert"])
    args.corpus_schema_sha256 = corpus.get("corpus_schema_sha256")
    args.collection_seed = int(meta["seed"])
    args.num_envs = int(meta["num_envs"])
    args.steps = int(meta["steps"])
    args.max_episode = int(meta["max_episode"])
    args.curriculum = str(meta["curriculum"])
    args.room = str(meta["room"])
    args.node = meta.get("node")
    args.gamma = float(meta["gamma"])
    args.holdout_seed_offset = int(meta["holdout_seed_offset"])
    args.ti_actor_steps = int(meta.get("ti_actor_steps", 0))
    args.ti_replay_capacity = int(meta.get("ti_replay_capacity", 0))
    args.ti_critic_replay_per_stratum = int(
        meta.get("ti_critic_replay_per_stratum", 0)
    )
    args.ti_bot_dir = meta.get("ti_bot_dir")
    args.ti_bot_source_sha256 = meta.get("ti_bot_source_sha256")
    args.chunk = 0

    if args.global_epochs <= 0:
        raise SystemExit("[joint] global-epochs must be positive")
    if args.ti_critic_pretrain_epochs < 0:
        raise SystemExit("[joint] ti-critic-pretrain-epochs must be non-negative")
    if (
        not np.isfinite(args.nextlat_actor_coef)
        or not np.isfinite(args.nextlat_critic_coef)
        or not np.isfinite(args.nextlat_critic_kl_coef)
        or args.nextlat_actor_coef <= 0
        or args.nextlat_critic_coef <= 0
        or args.nextlat_critic_kl_coef < 0
    ):
        raise SystemExit("[joint] NextLat pretrain coefficients are invalid")
    if not 0.0 < args.min_nextlat_relative_gap < 1.0:
        raise SystemExit("[joint] min-nextlat-relative-gap must be in (0, 1)")
    if args.min_nextlat_counterfactual_rows <= 0:
        raise SystemExit("[joint] min-nextlat-counterfactual-rows must be positive")
    if (
        not np.isfinite(args.max_aux_lifecycle_nll_ratio)
        or args.max_aux_lifecycle_nll_ratio < 1.0
    ):
        raise SystemExit("[joint] max-aux-lifecycle-nll-ratio must be finite and >= 1")
    if args.max_rare_intent_nll <= 0:
        raise SystemExit("[joint] max-rare-intent-nll must be positive")
    if args.min_rare_intent_rows <= 0:
        raise SystemExit("[joint] min-rare-intent-rows must be positive")
    if not 0.0 <= args.min_spawn_replay_accuracy <= 1.0:
        raise SystemExit("[joint] min-spawn-replay-accuracy must be in [0, 1]")
    if not 0.0 <= args.min_rare_intent_accuracy <= 1.0:
        raise SystemExit("[joint] min-rare-intent-accuracy must be in [0, 1]")
    overlapping_evaluation_seeds = _evaluation_seed_overlap(
        meta, offset=args.evaluation_seed_offset, num_envs=args.num_envs,
    )
    if overlapping_evaluation_seeds:
        raise SystemExit(
            "[joint] evaluation seeds overlap collected train/holdout worlds: "
            f"{sorted(overlapping_evaluation_seeds)}"
        )
    if not 0.0 <= args.min_outpost_closed_loop_success_rate <= 1.0:
        raise SystemExit(
            "[joint] min-outpost-closed-loop-success-rate must be in [0, 1]"
        )
    if args.compile and args.no_compile:
        raise SystemExit("[joint] --compile and --no-compile are mutually exclusive")
    if args.compile:
        raise SystemExit(
            "[joint] compilation is unsupported for variable-capacity corpus replay"
        )

    train_split = data["train"]
    holdout_split = data["holdout"]
    lifecycle_train = _lifecycle_samples(train_split["lifecycle_replay"])
    lifecycle_holdout = _lifecycle_samples(holdout_split["lifecycle_replay"])
    temporal_train = _temporal_samples(train_split["temporal_replay"])
    temporal_holdout = _temporal_samples(holdout_split["temporal_replay"])
    args.nextlat_train_rows = len(temporal_train)
    args.nextlat_holdout_rows = len(temporal_holdout)
    ti_factor_rows = data.get("ti_factor_replay", data.get("ti_actor_replay", []))
    ti_rare_samples, ti_rare_refs = _ti_rare_intent_samples(ti_factor_rows)
    rare_train_replay = [*lifecycle_train, *ti_rare_samples]
    rare_train_refs = _rare_intent_actor_refs(lifecycle_train)
    ti_rare_offset = len(lifecycle_train)
    for name in RARE_ACTOR_INTENTS:
        rare_train_refs[name].extend(
            (ti_rare_offset + sample_index, actor_index)
            for sample_index, actor_index in ti_rare_refs[name]
        )
    rare_holdout_refs = _rare_intent_actor_refs(lifecycle_holdout)
    missing_rare = {
        split: [name for name in RARE_ACTOR_INTENTS if not refs.get(name)]
        for split, refs in (
            ("train", rare_train_refs), ("holdout", rare_holdout_refs),
        )
    }
    missing_rare = {split: names for split, names in missing_rare.items() if names}
    if missing_rare:
        raise SystemExit(
            f"[joint] corpus lacks required rare-intent actor rows: {missing_rare}"
        )
    returns_tn = train_split["returns_tn"]
    holdout_returns_tn = holdout_split["returns_tn"]
    spawn_replay = [
        (
            str(row["stratum"]), int(row["actor_index"]), row["obs"], row["action"],
            row["eligible"],
        )
        for row in data["spawn_replay"]
    ]
    mb = max(8, int(args.minibatch))
    try:
        preflight_geometry = _preflight_joint_geometry(
            lifecycle_train, rare_train_refs, spawn_replay, minibatch=mb,
        )
    except ValueError as error:
        raise SystemExit(f"[joint] auxiliary schedule preflight failed: {error}") from error
    ti_actor_replay = [
        (row["obs"], row["action"], row["eligible"])
        for row in ti_factor_rows
    ]
    ti_train_split = _split_with_alias(data, "ti_train", "ti_critic_train")
    ti_holdout_split = _split_with_alias(data, "ti_holdout", "ti_critic_holdout")
    ti_critic_train = _critic_samples(
        (ti_train_split or {}).get(
            "critic_replay", (ti_train_split or {}).get("lifecycle_replay", []),
        )
    )
    ti_critic_holdout = _critic_samples(
        (ti_holdout_split or {}).get(
            "critic_replay", (ti_holdout_split or {}).get("lifecycle_replay", []),
        )
    )
    ti_train_returns = (
        ti_train_split["returns_tn"] if ti_train_split is not None else torch.empty(0, 1)
    )
    ti_holdout_returns = (
        ti_holdout_split["returns_tn"]
        if ti_holdout_split is not None else torch.empty(0, 1)
    )
    if not lifecycle_train or not lifecycle_holdout:
        raise SystemExit("[joint] corpus lifecycle train/holdout replay is empty")
    if len(temporal_train) < 2 or len(temporal_holdout) < 2:
        raise SystemExit("[joint] corpus temporal train/holdout replay is insufficient")

    teacher = data["teacher"]
    totals = teacher["train_totals"]
    total_h = float(totals.get("harvest", 0.0))
    total_c = float(totals.get("control", 0.0))
    total_delivered = float(totals.get("delivery", 0.0))
    total_built = float(totals.get("build", 0.0))
    total_claims = int(totals.get("claims", 0))
    max_creeps = int(totals.get("max_creeps", 0))
    total_spawn = int(totals.get("spawn_success", 0))
    teacher_by_curriculum = _merge_teacher_metrics(
        teacher.get("spawn_contract_by_curriculum", {}),
        teacher.get("train_by_curriculum", {}),
    )
    corpus_hist = {
        str(key): float(value)
        for key, value in (teacher.get("corpus_hist") or {}).items()
        if isinstance(value, (int, float, np.number))
    }
    corpus_hist["_ti_actor_replay_rows"] = float(len(ti_actor_replay))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(args.seed ^ 0x53485546)
    rare_shuffle_generator = torch.Generator(device="cpu")
    rare_shuffle_generator.manual_seed(args.seed ^ 0x52415245)
    nextlat_shuffle_generator = torch.Generator(device="cpu")
    nextlat_shuffle_generator.manual_seed(args.seed ^ 0x4E455854)
    spawn_shuffle_generator = torch.Generator(device="cpu")
    spawn_shuffle_generator.manual_seed(args.seed ^ 0x53504157)
    device = torch.device(args.device)
    args.training_device_type = device.type
    args.visible_cuda_device_count = (
        torch.cuda.device_count() if device.type == "cuda" else 0
    )
    args.effective_precision = (
        "bf16"
        if not args.no_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
        else "fp32"
    )
    args.save.parent.mkdir(parents=True, exist_ok=True)
    if args.logdir is None:
        args.logdir = _RL_ROOT / "runs" / "tb-joint-pretrain" / datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
    args.logdir.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(args.logdir))
        writer.add_text("run/corpus_sha256", corpus_sha256)
    except Exception as error:  # noqa: BLE001
        print(f"[joint] TB disabled: {error}", flush=True)

    actor = Actor().to(device)
    critic = Critic().to(device)
    opt_a = HybridMuonAdamW(
        actor, adam_lr=args.lr_actor, muon_lr=args.muon_lr,
        muon_weight_decay=args.muon_weight_decay,
        muon_momentum_warmup_steps=args.muon_momentum_warmup_steps,
    )
    opt_c = HybridMuonAdamW(
        critic, adam_lr=args.lr_critic, muon_lr=args.muon_lr,
        muon_weight_decay=args.muon_weight_decay,
        muon_momentum_warmup_steps=args.muon_momentum_warmup_steps,
    )
    critic.support.validate_targets(returns_tn)
    critic.support.validate_targets(holdout_returns_tn)
    if ti_train_split is not None:
        critic.support.validate_targets(ti_train_returns)
    if ti_holdout_split is not None:
        critic.support.validate_targets(ti_holdout_returns)

    global_epoch = 0
    global_step = 0
    if args.resume is not None and args.critic_init is not None:
        raise SystemExit("[joint] --resume and --critic-init are mutually exclusive")
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        try:
            resume_meta = validate_artifact(
                state, actor, critic, kinds=("joint_pretrain",),
            )
        except ValueError as error:
            raise SystemExit(f"[joint] incompatible resume: {error}") from error
        continuation = {
            "corpus_sha256": corpus_sha256,
            "minibatch": args.minibatch,
            "value_coef": args.value_coef,
            "lr_actor": args.lr_actor,
            "lr_critic": args.lr_critic,
            "ti_critic_pretrain_epochs": args.ti_critic_pretrain_epochs,
            "muon_lr": args.muon_lr,
            "muon_weight_decay": args.muon_weight_decay,
            "muon_momentum_warmup_steps": args.muon_momentum_warmup_steps,
            "rare_actor_intents": list(RARE_ACTOR_INTENTS),
            "max_rare_intent_nll": args.max_rare_intent_nll,
            "min_rare_intent_accuracy": args.min_rare_intent_accuracy,
            "min_rare_intent_rows": args.min_rare_intent_rows,
            "min_spawn_replay_accuracy": args.min_spawn_replay_accuracy,
            "evaluation_seed_offset": args.evaluation_seed_offset,
            "min_outpost_closed_loop_success_rate": (
                args.min_outpost_closed_loop_success_rate
            ),
            "nextlat": dict(SCHEMA["nextLat"]),
            "nextlat_action_pooling": NEXTLAT_ACTION_POOLING,
            "nextlat_pretrain": {
                "actorCoef": args.nextlat_actor_coef,
                "criticCoef": args.nextlat_critic_coef,
                "criticKlCoef": args.nextlat_critic_kl_coef,
                "loss": "smooth_l1_latent+detached_value_probe_kl",
                "target": "detached_next_observation_encoding",
                "optimizerAuthority": JOINT_OBJECTIVE_AUTHORITY,
                "temporalSampling": "independent_cyclic_shuffle_per_joint_minibatch",
                "action_pooling": NEXTLAT_ACTION_POOLING,
                "actionAblation": NEXTLAT_ACTION_ABLATION,
                "storedCounterfactualUse": "provenance_only_not_evaluation",
                "minRelativeGap": args.min_nextlat_relative_gap,
                "minCounterfactualRows": args.min_nextlat_counterfactual_rows,
            },
            "ti_actor_training": TI_ACTOR_TRAINING,
            "actor_auxiliary_schedule": ACTOR_AUXILIARY_SCHEDULE,
            "max_aux_lifecycle_nll_ratio": args.max_aux_lifecycle_nll_ratio,
            "nextlat_train_rows": len(temporal_train),
            "nextlat_holdout_rows": len(temporal_holdout),
            **_precision_resume_identity(args),
            "seed": args.seed,
        }
        mismatched = _continuation_mismatches(resume_meta, continuation)
        if mismatched:
            raise SystemExit(f"[joint] continuation parameters differ: {mismatched}")
        if "shuffle_rng_state" not in state:
            raise SystemExit("[joint] continuation lacks dedicated shuffle RNG state")
        if "rare_shuffle_rng_state" not in state:
            raise SystemExit(
                "[joint] continuation lacks dedicated rare-intent shuffle RNG state"
            )
        if "nextlat_shuffle_rng_state" not in state:
            raise SystemExit(
                "[joint] continuation lacks dedicated NextLat shuffle RNG state"
            )
        if "spawn_shuffle_rng_state" not in state:
            raise SystemExit(
                "[joint] continuation lacks dedicated spawn shuffle RNG state"
            )
        load_full_state(actor, state["actor"], name="actor")
        load_full_state(critic, state["critic"], name="critic")
        opt_a.load_state_dict(state["actor_opt"])
        opt_c.load_state_dict(state["critic_opt"])
        shuffle_generator.set_state(state["shuffle_rng_state"].cpu())
        rare_shuffle_generator.set_state(state["rare_shuffle_rng_state"].cpu())
        nextlat_shuffle_generator.set_state(state["nextlat_shuffle_rng_state"].cpu())
        spawn_shuffle_generator.set_state(state["spawn_shuffle_rng_state"].cpu())
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if state.get("numpy_rng_state") is not None:
            np.random.set_state(state["numpy_rng_state"])
        if state.get("cuda_rng_state_all") is not None:
            if not torch.cuda.is_available():
                raise SystemExit("[joint] continuation requires CUDA RNG state")
            cuda_rng_states = [
                rng_state.cpu() for rng_state in state["cuda_rng_state_all"]
            ]
            if len(cuda_rng_states) != args.visible_cuda_device_count:
                raise SystemExit(
                    "[joint] continuation CUDA RNG state count differs from "
                    "visible CUDA device count"
                )
            torch.cuda.set_rng_state_all(cuda_rng_states)
        global_epoch = int(resume_meta.get("global_epoch", 0))
        global_step = int(state.get("global_step", 0))
        resumed_hist = resume_meta.get("action_type_hist", {})
        if isinstance(resumed_hist, dict):
            for key in ("_ti_actor_nll", "_ti_actor_legal_coverage"):
                value = resumed_hist.get(key)
                if isinstance(value, (int, float, np.number)):
                    corpus_hist[key] = float(value)
        if global_epoch >= args.global_epochs:
            raise SystemExit("[joint] resume already reached requested global epochs")
    elif args.critic_init is not None:
        state = torch.load(args.critic_init, map_location=device, weights_only=False)
        try:
            init_meta = validate_artifact(state, actor, critic, kinds=("ti_critic",))
        except ValueError as error:
            raise SystemExit(f"[joint] incompatible TI critic init: {error}") from error
        if not bool(init_meta.get("qualified")):
            raise SystemExit("[joint] TI critic init is not qualified")
        load_full_state(critic, state["critic"], name="critic")

    print(
        f"[joint] corpus={corpus_sha256} envs={args.num_envs} steps={args.steps} "
        f"scripted_train={len(lifecycle_train)} scripted_holdout={len(lifecycle_holdout)} "
        f"ti_factors={len(ti_actor_replay)} ti_critic_train={len(ti_critic_train)} "
        f"ti_critic_holdout={len(ti_critic_holdout)} "
        f"rare_train={{{', '.join(f'{name}:{len(rare_train_refs[name])}' for name in RARE_ACTOR_INTENTS)}}} "
        f"rare_holdout={{{', '.join(f'{name}:{len(rare_holdout_refs[name])}' for name in RARE_ACTOR_INTENTS)}}} "
        f"nextlat_train={len(temporal_train)} nextlat_holdout={len(temporal_holdout)} "
        f"geometry={preflight_geometry} epochs={args.global_epochs}",
        flush=True,
    )
    use_bf16 = args.effective_precision == "bf16"
    t0 = time.time()
    spawn_nll = spawn_legal = float("nan")
    ti_actor_nll = float(corpus_hist.get("_ti_actor_nll", float("nan")))
    ti_actor_coverage = float(
        corpus_hist.get("_ti_actor_legal_coverage", float("nan"))
    )
    rare_actor_nll = rare_actor_legal = float("nan")
    spawn_diagnostics: dict[str, float] = {}
    epoch_nll = epoch_vloss = float("nan")
    ti_critic_loss = float("nan")
    nextlat_metrics: dict[str, float] = {}
    phase_metrics: dict[str, float] = {}

    def checkpoint(*, partial: bool, explained_variance: float) -> dict:
        return _joint_checkpoint(
            actor, critic, opt_a, opt_c, args=args, step=args.steps,
            global_epoch=global_epoch, global_step=global_step,
            total_harvest=total_h, total_control=total_c,
            total_delivered=total_delivered, total_built=total_built,
            total_claims=total_claims, max_creeps=max_creeps,
            spawn_success=total_spawn, nll=epoch_nll, value_loss=epoch_vloss,
            explained_variance=explained_variance, action_hist=corpus_hist,
            teacher_by_curriculum=teacher_by_curriculum, partial=partial,
            wall_s=time.time() - t0, corpus_sha256=corpus_sha256,
            shuffle_generator=shuffle_generator,
            rare_shuffle_generator=rare_shuffle_generator,
            nextlat_shuffle_generator=nextlat_shuffle_generator,
            spawn_shuffle_generator=spawn_shuffle_generator,
            nextlat_metrics=nextlat_metrics,
        )

    try:
        if _should_run_ti_initialization(
            global_epoch=global_epoch, resume=args.resume,
        ):
            ti_updates = 0
            for ti_epoch in range(args.ti_critic_pretrain_epochs):
                ti_critic_loss, ti_trained = _train_critic_replay_epoch(
                    critic, opt_c, ti_critic_train, ti_train_returns,
                    device=device, use_bf16=use_bf16, minibatch=mb,
                    value_coef=args.value_coef, shuffle_generator=shuffle_generator,
                )
                ti_updates += ti_trained
                print(
                    f"[joint] TI critic init {ti_epoch + 1}/"
                    f"{args.ti_critic_pretrain_epochs} vloss={ti_critic_loss:.5f}",
                    flush=True,
                )
                if writer:
                    writer.add_scalar(
                        "joint/ti_critic_initialization_value_loss",
                        ti_critic_loss,
                        ti_epoch + 1,
                    )
            ti_actor_nll, ti_actor_coverage = _train_ti_actor_replay(
                actor, opt_a, ti_actor_replay, device=device, use_bf16=use_bf16,
                minibatch=min(32, mb), shuffle_generator=shuffle_generator,
            )
            corpus_hist["_ti_actor_nll"] = float(ti_actor_nll)
            corpus_hist["_ti_actor_legal_coverage"] = float(ti_actor_coverage)
            global_step += ti_updates + len(ti_actor_replay)
            print(
                f"[joint] TI actor init nll={ti_actor_nll:.4f} "
                f"coverage={ti_actor_coverage:.4f}",
                flush=True,
            )
            if writer:
                writer.add_scalar(
                    "joint/ti_actor_initialization_nll", ti_actor_nll, 0,
                )
        for epoch in range(global_epoch, args.global_epochs):
            # The deployment-distribution scripted pass owns the final critic.
            # Repeating off-policy TI value epochs here erased seed-claimer value
            # structure because V(s) is behavior-policy dependent.
            final_training_epoch = epoch + 1 == args.global_epochs
            if final_training_epoch:
                phase_metrics["lifecycle_before_final_joint_train_nll"] = (
                    _evaluate_lifecycle_actor_nll(
                        actor, lifecycle_train, device=device, minibatch=mb,
                    )
                )
            (
                epoch_nll, _legal, epoch_vloss, trained, epoch_nextlat,
            ) = _train_joint_lifecycle_epoch(
                actor, critic, opt_a, opt_c, lifecycle_train, returns_tn,
                temporal_train,
                device=device, use_bf16=use_bf16, minibatch=mb,
                value_coef=args.value_coef, shuffle_generator=shuffle_generator,
                nextlat_shuffle_generator=nextlat_shuffle_generator,
                nextlat_actor_coef=args.nextlat_actor_coef,
                nextlat_critic_coef=args.nextlat_critic_coef,
                nextlat_critic_kl_coef=args.nextlat_critic_kl_coef,
                rare_replay=rare_train_replay,
                rare_refs_by_intent=rare_train_refs,
                rare_shuffle_generator=rare_shuffle_generator,
                spawn_replay=spawn_replay,
                spawn_shuffle_generator=spawn_shuffle_generator,
            )
            nextlat_metrics = {
                f"train_{key}": float(value)
                for key, value in epoch_nextlat.items()
            }
            rare_actor_nll = float(epoch_nextlat["rare_nll"])
            rare_actor_legal = float(epoch_nextlat["rare_legal"])
            spawn_nll = float(epoch_nextlat["spawn_nll"])
            spawn_legal = float(epoch_nextlat["spawn_legal"])
            global_step += trained
            global_epoch = epoch + 1
            print(
                f"[joint] epoch {global_epoch}/{args.global_epochs} nll={epoch_nll:.4f} "
                f"scripted_vloss={epoch_vloss:.5f} "
                f"spawn_nll={spawn_nll:.4f} ti_actor_nll={ti_actor_nll:.4f} "
                f"rare_actor_nll={rare_actor_nll:.4f} "
                f"nextlat={epoch_nextlat['actor_smooth_l1']:.4f}/"
                f"{epoch_nextlat['critic_smooth_l1']:.4f}",
                flush=True,
            )
            if writer:
                writer.add_scalar("joint/global_actor_nll", epoch_nll, global_epoch)
                writer.add_scalar("joint/global_value_loss", epoch_vloss, global_epoch)
                writer.add_scalar("joint/rare_actor_nll", rare_actor_nll, global_epoch)
                for name, value in epoch_nextlat.items():
                    writer.add_scalar(f"nextlat/train_{name}", value, global_epoch)
                if final_training_epoch:
                    for name, value in phase_metrics.items():
                        writer.add_scalar(f"joint/{name}", value, global_epoch)
            if global_epoch < args.global_epochs and (
                global_epoch == 1 or global_epoch % 4 == 0
            ):
                atomic_torch_save(checkpoint(partial=True, explained_variance=float("nan")), args.save)

        ti_actor_nll, ti_actor_coverage = _evaluate_ti_actor_replay(
            actor, ti_actor_replay, device=device, minibatch=mb,
        )
        lifecycle_train_metrics = _evaluate_global_lifecycle(
            actor, critic, lifecycle_train, returns_tn, device=device, minibatch=mb,
        )
        phase_metrics["lifecycle_final_train_nll"] = float(
            lifecycle_train_metrics["nll"]
        )
        phase_metrics["lifecycle_after_joint_train_nll"] = float(
            lifecycle_train_metrics["nll"]
        )
        if writer:
            writer.add_scalar(
                "joint/lifecycle_final_train_nll",
                phase_metrics["lifecycle_final_train_nll"],
                global_epoch,
            )
        lifecycle_metrics = _evaluate_global_lifecycle(
            actor, critic, lifecycle_holdout, holdout_returns_tn,
            device=device, minibatch=mb,
        )
        ti_train_metrics = _evaluate_critic_replay(
            critic, ti_critic_train, ti_train_returns, device=device, minibatch=mb,
        )
        ti_holdout_metrics = _evaluate_critic_replay(
            critic, ti_critic_holdout, ti_holdout_returns, device=device, minibatch=mb,
        )
        rare_train_metrics = _evaluate_rare_intent_actors(
            actor, rare_train_replay, rare_train_refs, device=device, minibatch=mb,
        )
        rare_holdout_metrics = _evaluate_rare_intent_actors(
            actor, lifecycle_holdout, rare_holdout_refs, device=device, minibatch=mb,
        )
        nextlat_train_metrics = _evaluate_nextlat(
            actor, critic, temporal_train, device=device, minibatch=mb,
        )
        nextlat_holdout_metrics = _evaluate_nextlat(
            actor, critic, temporal_holdout, device=device, minibatch=mb,
        )
        nextlat_metrics = {
            **{
                f"train_{key}": float(value)
                for key, value in nextlat_train_metrics.items()
            },
            **{
                f"holdout_{key}": float(value)
                for key, value in nextlat_holdout_metrics.items()
            },
            "train_optimizer_exposures_per_epoch": float(trained),
            "train_replay_rows": float(len(temporal_train)),
            "train_exposure_ratio": float(trained) / max(1, len(temporal_train)),
            **{
                f"train_{key}": float(epoch_nextlat[key])
                for key in (
                    "optimizer_steps", "rare_exposures", "spawn_exposures",
                    "rare_nll", "rare_legal", "spawn_nll", "spawn_legal",
                    "rare_batches", "spawn_batches",
                    "auxiliary_batches", "auxiliary_minibatch",
                )
            },
        }
        spawn_diagnostics = _evaluate_spawn_replay(
            actor, spawn_replay, device=device, use_bf16=use_bf16,
            minibatch=mb,
        )
        if writer:
            for split, metrics in (
                ("train", rare_train_metrics), ("holdout", rare_holdout_metrics),
            ):
                for name, value in metrics.items():
                    writer.add_scalar(f"rare_intent/{split}_{name}", value, global_epoch)
            for name, value in nextlat_metrics.items():
                writer.add_scalar(f"nextlat/{name}", value, global_epoch)
        epoch_nll = float(lifecycle_metrics["nll"])
        epoch_vloss = float(lifecycle_metrics["value_loss"])
        last_ev = float(lifecycle_metrics["ev"])
        corpus_hist.update({
            "_bc_legal_frac": lifecycle_metrics["legal_frac"],
            "_spawn_replay_nll": float(spawn_nll),
            "_spawn_replay_legal_frac": float(spawn_legal),
            "_spawn_replay_size": float(len(spawn_replay)),
            "_ti_actor_nll": float(ti_actor_nll),
            "_ti_actor_legal_coverage": float(ti_actor_coverage),
            "_rare_actor_train_nll": float(rare_actor_nll),
            "_rare_actor_train_legal_frac": float(rare_actor_legal),
            "_lifecycle_replay_nll": float(lifecycle_train_metrics["nll"]),
            "_lifecycle_replay_legal_frac": lifecycle_train_metrics["legal_frac"],
            "_lifecycle_replay_size": float(len(lifecycle_train)),
            "_lifecycle_holdout_legal_frac": lifecycle_metrics["legal_frac"],
            "_lifecycle_holdout_size": float(len(lifecycle_holdout)),
        })
        for name, value in spawn_diagnostics.items():
            corpus_hist[f"_spawn_replay_{name}"] = float(value)
        lifecycle_strata = [sample.stratum for sample in lifecycle_train]
        for category in ("spawn", "harvest", "logistics", "construction", "control"):
            corpus_hist[f"_lifecycle_replay_{category}"] = float(sum(
                category in stratum.rsplit(":", 1)[-1].split("+")
                for stratum in lifecycle_strata
            ))
        all_spawn_strata = [row[0] for row in spawn_replay]
        wait_strata = {s for s in all_spawn_strata if s.startswith("waitlegal:")}
        corpus_hist["_spawn_replay_wait_legal_size"] = float(sum(
            s.startswith("waitlegal:") for s in all_spawn_strata
        ))
        corpus_hist["_spawn_replay_wait_legal_strata"] = float(len(wait_strata))
        spawn_coverage = _spawn_replay_coverage(spawn_replay)
        for name, count in spawn_coverage.items():
            corpus_hist[f"_spawn_replay_{name}"] = count
        corpus_hist.update(_spawn_length_coverage_gap(
            {
                bucket: spawn_coverage[f"length_{bucket}"]
                for bucket in SPAWN_LENGTH_BUCKETS
            },
            "_spawn_replay",
        ))

        evaluation_seed = int(meta["seed"]) + args.evaluation_seed_offset
        validation_curricula = [
            stage.strip() for stage in args.curriculum.split(",") if stage.strip()
        ]
        if "seed_outpost" not in validation_curricula:
            validation_curricula.append("seed_outpost")
        teacher_spawn_bodies = _teacher_contract_bodies(spawn_replay)
        missing_contracts = sorted(set(SPAWN_CURRICULA) - set(teacher_spawn_bodies))
        if missing_contracts:
            raise SystemExit(
                f"[joint] corpus lacks teacher spawn contract rows for {missing_contracts}"
            )
        # The policy is probed over a window that reaches the teacher's own
        # decision tick, because that is the state its supervision came from:
        # The International does not act on a world's opening ticks, so a
        # single-tick probe would ask for a decision no row ever demonstrated.
        contract_ticks = [
            teacher_by_curriculum.get(stage, {}).get("spawn_label_tick")
            for stage in SPAWN_CURRICULA
        ]
        if any(tick is None for tick in contract_ticks):
            raise SystemExit(
                "[joint] corpus spawn contract metrics lack the teacher's "
                "spawn_label_tick; recollect the corpus with the TI contract collector"
            )
        spawn_probe_steps = 1 + max(int(tick) for tick in contract_ticks)
        validation_envs = VecScreepsEnv(
            args.num_envs, node=args.node, room=args.room,
            max_episode=args.max_episode, device=device,
            curriculum=",".join(validation_curricula), lean_meta=False,
            seed=evaluation_seed,
        )
        # Retained for the closed-loop spawn probe: these worlds certify that the
        # learned policy itself spawns each contract body in the engine.
        spawn_validation_envs = VecScreepsEnv(
            len(SPAWN_CURRICULA), node=args.node, room=args.room,
            max_episode=spawn_probe_steps + 1,
            device=device, curriculum=",".join(SPAWN_CURRICULA), lean_meta=False,
            seed=evaluation_seed,
        )
        try:
            # The scripted planner is an evaluation baseline only (see
            # agent/eval_scripted.py); no scripted quantity may enter a loss or a
            # qualification gate.  Teacher-forced actor validation is therefore the
            # TI factor-row measurement on the final actor, which is the same
            # measurement against the only teacher this pipeline has.
            validation = {
                "validation_factor_nll": float(ti_actor_nll),
                "validation_legal_frac": float(ti_actor_coverage),
            }
            validation.update({
                "lifecycle_train_value_ev": float(lifecycle_train_metrics["ev"]),
                "lifecycle_train_value_loss": float(lifecycle_train_metrics["value_loss"]),
                "lifecycle_holdout_factor_nll": epoch_nll,
                "lifecycle_holdout_legal_frac": lifecycle_metrics["legal_frac"],
                "lifecycle_holdout_value_ev": last_ev,
                "lifecycle_holdout_value_loss": epoch_vloss,
                "ti_critic_train_value_ev": float(ti_train_metrics["ev"]),
                "ti_critic_train_value_loss": float(ti_train_metrics["value_loss"]),
                "ti_critic_holdout_value_ev": float(ti_holdout_metrics["ev"]),
                "ti_critic_holdout_value_loss": float(ti_holdout_metrics["value_loss"]),
            })
            validation.update({
                f"nextlat_{key}": float(value)
                for key, value in nextlat_metrics.items()
            })
            validation.update({
                f"rare_train_{key}": float(value)
                for key, value in rare_train_metrics.items()
            })
            validation.update({
                f"rare_holdout_{key}": float(value)
                for key, value in rare_holdout_metrics.items()
            })
            validation.update({
                f"lifecycle_holdout_{key}": float(value)
                for key, value in lifecycle_metrics.items() if key.startswith("stage_")
            })
            validation.update(phase_metrics)
            closed_loop = _validate_closed_loop(
                actor, validation_envs, steps=args.closed_loop_steps, device=device,
            )
            spawn_closed_loop = _validate_closed_loop(
                actor, spawn_validation_envs, steps=spawn_probe_steps, device=device,
            )
            closed_loop.setdefault("closed_loop_by_curriculum", {}).update(
                spawn_closed_loop.get("closed_loop_by_curriculum", {})
            )
        finally:
            validation_envs.close()
            spawn_validation_envs.close()

        result = checkpoint(partial=False, explained_variance=last_ev)
        failures = _qualification_failures(
            args, last_nll=epoch_nll, last_vloss=epoch_vloss,
            corpus_hist=corpus_hist,
            skill=(total_h + total_c) / max(1, args.steps * args.num_envs),
            total_delivered=total_delivered, total_built=total_built,
            total_claims=total_claims, max_creeps=max_creeps,
            teacher_by_curriculum=teacher_by_curriculum,
            teacher_spawn_bodies=teacher_spawn_bodies,
            validation=validation, closed_loop=closed_loop,
        )
        result["meta"].update(validation)
        result["meta"].update(closed_loop)
        result["meta"].update(_teacher_spawn_coverage(teacher_by_curriculum))
        result["meta"]["teacher_spawn_bodies"] = teacher_spawn_bodies
        result["meta"]["spawn_probe_steps"] = spawn_probe_steps
        result["meta"]["qualified"] = not failures
        result["meta"]["qualification_failures"] = failures
        atomic_torch_save(result, args.save)
        print(
            f"[joint] saved {args.save} qualified={not failures} failures={failures}",
            flush=True,
        )
        return 0 if not failures else 1
    finally:
        if writer:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
