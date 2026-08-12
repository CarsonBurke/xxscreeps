"""CPU-only unit tests for redesign contracts (no env, no GPU).

  python3 -m samples.rl.agent.test_latent_unit
"""
from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path

import torch

_RL = Path(__file__).resolve().parents[1]
_REPO = _RL.parents[1]
for p in (_REPO, _RL):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from samples.rl.agent.constants import (
    ACTOR_FEAT,
    GLOBAL_FEAT,
    INTENT_SLOTS,
    INTENT_TYPES,
    MAX_ACTORS,
    MAX_ROOMS,
    MAX_TARGETS,
    N_AMOUNT,
    N_DIR,
    N_INTENT,
    PATCH_FLAT,
    PATCHES_PER_ROOM,
    PPO_CFG,
    SCHEMA,
    SCHEMA_SHA256,
    SCHEMA_PATH,
    TARGET_FEAT,
)
from samples.rl.agent.gae import compute_gae_tn
from samples.rl.agent.hl_gauss import HLGaussSupport
from samples.rl.agent.model import Actor, Agent, Critic
from samples.rl.agent.running_stats import RewardNormalizer


def test_gae_truncation_bootstraps():
    T, N = 4, 1
    rewards = torch.ones(T, N)
    values = torch.zeros(T, N)
    dones = torch.tensor([[0.0], [0.0], [0.0], [1.0]])
    # Without trunc: terminal → bootstrap 0
    adv_term, ret_term = compute_gae_tn(
        rewards, values, dones, gamma=0.99, lam=1.0, next_value=torch.tensor([100.0]),
        truncations=torch.zeros_like(dones),
    )
    # With trunc: bootstrap next_value
    adv_trunc, ret_trunc = compute_gae_tn(
        rewards, values, dones, gamma=0.99, lam=1.0, next_value=torch.tensor([100.0]),
        truncations=dones.clone(),
    )
    assert ret_trunc[-1, 0] > ret_term[-1, 0] + 50  # truncation includes bootstrap


def test_gae_cuts_chain_on_done():
    """Post-reset rewards must not leak into pre-boundary advantages."""
    T, N = 4, 1
    rewards = torch.tensor([[1.0], [1.0], [0.0], [100.0]])  # big reward after done
    values = torch.zeros(T, N)
    dones = torch.tensor([[0.0], [1.0], [0.0], [0.0]])  # episode ends at t=1
    trunc = dones.clone()
    # next values: bootstrap 0 at t=1 (terminal obs V)
    nv = torch.zeros(T, N)
    adv, ret = compute_gae_tn(
        rewards, values, dones, gamma=0.99, lam=1.0,
        next_value=torch.zeros(N), truncations=trunc, next_values_tn=nv,
    )
    # t=0,1 must not include the 100 at t=3
    assert float(ret[0, 0]) < 5.0, ret
    assert float(ret[1, 0]) < 5.0, ret


def test_type_gated_logprob_none_low_entropy():
    B = 2
    actor = Actor()
    obs = {
        "patches": torch.zeros(B, MAX_ROOMS, PATCHES_PER_ROOM, PATCH_FLAT),
        "room_mask": torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
        "room_coords": torch.zeros(B, MAX_ROOMS, 2),
        "actors": torch.zeros(B, MAX_ACTORS, ACTOR_FEAT),
        "actor_mask": torch.ones(B, MAX_ACTORS),  # all live — worst case
        "targets": torch.zeros(B, MAX_TARGETS, TARGET_FEAT),
        "target_mask": torch.zeros(B, MAX_TARGETS),
        "intent_mask": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, N_INTENT),
        "dir_mask": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, N_DIR),
        "target_select_mask": torch.zeros(B, N_INTENT, MAX_TARGETS),
        "amount_mask": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, N_INTENT, N_AMOUNT),
        "globals": torch.zeros(B, GLOBAL_FEAT),
    }
    # Only none legal
    obs["intent_mask"][..., INTENT_TYPES.index("none")] = 1
    obs["amount_mask"][..., 0] = 1
    out = actor(obs)
    # Entropy should not be huge (old bug: 4 heads × 24 actors × 2 slots)
    assert float(out.entropy.mean().detach()) < 5.0, out.entropy.mean()


def test_hl_gauss_support():
    s = HLGaussSupport(101, -10, 10, 2.0)
    t = torch.tensor([0.0, 5.0, -10.0])
    p = s.project(t)
    assert torch.allclose(p.sum(-1), torch.ones(3), atol=1e-4)
    ce = s.cross_entropy(torch.zeros(3, 101), t)
    assert ce.shape == (3,)


def test_reward_normalizer_roundtrip():
    rn = RewardNormalizer(gamma=0.99, clip=10.0)
    r = torch.randn(8, 2)
    d = torch.zeros(8, 2)
    d[-1] = 1
    out = rn.normalize(r, d)
    sd = rn.state_dict()
    rn2 = RewardNormalizer()
    rn2.load_state_dict(sd)
    assert abs(rn2.stats()["reward_rms_std"] - rn.stats()["reward_rms_std"]) < 1e-9
    assert rn2._returns is None  # fresh simulator episodes must not inherit live traces


def test_critic_scalar_default():
    c = Critic()
    assert c.use_hl_gauss is False
    B = 1
    batch = {
        "patches": torch.zeros(B, MAX_ROOMS, PATCHES_PER_ROOM, PATCH_FLAT),
        "room_mask": torch.tensor([[1.0, 0, 0, 0]]),
        "room_coords": torch.zeros(B, MAX_ROOMS, 2),
        "actors": torch.zeros(B, MAX_ACTORS, ACTOR_FEAT),
        "actor_mask": torch.zeros(B, MAX_ACTORS),
        "targets": torch.zeros(B, MAX_TARGETS, TARGET_FEAT),
        "target_mask": torch.zeros(B, MAX_TARGETS),
        "globals": torch.zeros(B, GLOBAL_FEAT),
    }
    v = c(batch)
    assert v.shape == (1,)


def test_schema_reward_values_productive_economy():
    r = SCHEMA["reward"]
    assert 0 < float(r["energyHarvested"]) < float(r["controlPoints"])
    assert float(r["controlPoints"]) == 1.0
    assert float(r["energyDelivered"]) > 0
    assert float(r["buildProgress"]) > 0
    assert float(r["roomClaim"]) > 0
    assert "spawn" not in r  # avoid rewarding body churn directly


def test_intent_enum_parity():
    raw = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert raw["intentTypes"] == INTENT_TYPES
    assert N_INTENT == len(INTENT_TYPES)
    assert "createConstructionSite" in INTENT_TYPES
    assert "spawnCreep" in INTENT_TYPES


def test_logprob_sample_matches_evaluate():
    """Importance-sampling contract: act then re-evaluate same action → same logπ."""
    torch.manual_seed(0)
    B = 2
    actor = Actor()
    obs = {
        "patches": torch.zeros(B, MAX_ROOMS, PATCHES_PER_ROOM, PATCH_FLAT),
        "room_mask": torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
        "room_coords": torch.zeros(B, MAX_ROOMS, 2),
        "actors": torch.zeros(B, MAX_ACTORS, ACTOR_FEAT),
        "actor_mask": torch.zeros(B, MAX_ACTORS),
        "targets": torch.zeros(B, MAX_TARGETS, TARGET_FEAT),
        "target_mask": torch.zeros(B, MAX_TARGETS),
        "intent_mask": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, N_INTENT),
        "dir_mask": torch.ones(B, MAX_ACTORS, INTENT_SLOTS, N_DIR),
        "target_select_mask": torch.zeros(B, N_INTENT, MAX_TARGETS),
        "amount_mask": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, N_INTENT, N_AMOUNT),
        "globals": torch.zeros(B, GLOBAL_FEAT),
    }
    # One live actor; only none + move legal
    obs["actor_mask"][:, 0] = 1
    obs["intent_mask"][:, 0, :, INTENT_TYPES.index("none")] = 1
    obs["intent_mask"][:, 0, :, INTENT_TYPES.index("move")] = 1
    obs["amount_mask"][..., 0] = 1
    with torch.no_grad():
        sampled = actor(obs, deterministic=False)
        action = {
            "types": sampled.types,
            "dirs": sampled.dirs,
            "targets": sampled.targets,
            "amounts": sampled.amounts,
        }
        evaled = actor(obs, action=action)
    assert torch.allclose(sampled.logprob, evaled.logprob, atol=1e-4, rtol=1e-4), (
        sampled.logprob,
        evaled.logprob,
    )


def test_entity_context_changes_peer_policy_and_critic():
    """Actor coordination and centralized value must consume peer entity state."""
    torch.manual_seed(7)
    actor = Actor().eval()
    critic = Critic().eval()
    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, :2] = 1
    none = INTENT_TYPES.index("none")
    move = INTENT_TYPES.index("move")
    obs["intent_mask"][0, :2, :, none] = 1
    obs["intent_mask"][0, :2, :, move] = 1
    obs["amount_mask"][..., 0] = 1
    action = {
        k: torch.zeros(1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long)
        for k in ("types", "dirs", "targets", "amounts")
    }
    with torch.no_grad():
        before_actor = actor(obs, action=action).actor_logprob[0, 0]
        before_value_state = critic._backbone(obs)[0]
        changed = {k: v.clone() for k, v in obs.items()}
        changed["actors"][0, 1, 1] = 0.9
        changed["actors"][0, 1, 11] = 1.0
        after_actor = actor(changed, action=action).actor_logprob[0, 0]
        after_value_state = critic._backbone(changed)[0]
    assert not torch.allclose(before_actor, after_actor, atol=1e-7, rtol=1e-7)
    assert not torch.allclose(before_value_state, after_value_state, atol=1e-7, rtol=1e-7)


def test_per_actor_logprob_factorization():
    obs = _dummy_obs_batch(2)
    obs["actor_mask"][:, :3] = 1
    obs["intent_mask"][:, :3, :, INTENT_TYPES.index("none")] = 1
    obs["intent_mask"][:, :3, :, INTENT_TYPES.index("move")] = 1
    obs["amount_mask"][..., 0] = 1
    with torch.no_grad():
        out = Actor()(obs)
    assert out.actor_logprob.shape == (2, MAX_ACTORS)
    assert torch.allclose(out.logprob, out.actor_logprob.sum(dim=-1))
    assert torch.count_nonzero(out.actor_logprob[:, 3:]) == 0


def test_structure_cannot_select_remote_target_intent():
    """Compact candidates must still close types with no actor-local target."""
    obs = _dummy_obs_batch(1)
    obs["room_mask"][0, 1] = 1
    obs["actor_mask"][0, 0] = 1
    obs["actors"][0, 0, 0] = 1  # structure, not a mobile creep
    obs["actors"][0, 0, 3] = 1.0 / (MAX_ROOMS - 1)  # room slot 1
    obs["target_mask"][0, 0] = 1
    obs["targets"][0, 0, 3] = 0  # candidate is in room slot 0
    none = INTENT_TYPES.index("none")
    repair = INTENT_TYPES.index("repair")
    obs["intent_mask"][0, 0, :, none] = 1
    obs["intent_mask"][0, 0, :, repair] = 1
    obs["target_select_mask"][0, repair, 0] = 1
    obs["amount_mask"][..., 0] = 1

    actor = Actor().eval()
    with torch.no_grad():
        actor.type_head.bias[none] = -50
        actor.type_head.bias[repair] = 50
        out = actor(obs, deterministic=True)
    assert int(out.types[0, 0, 0]) == none


def test_creep_construction_target_is_same_room_only():
    obs = _dummy_obs_batch(1)
    obs["room_mask"][0, 1] = 1
    obs["actor_mask"][0, 0] = 1  # mobile creep in room 0
    obs["target_mask"][0, 0] = 1
    obs["targets"][0, 0, 0] = 1.0  # explicit build position
    obs["targets"][0, 0, 3] = 1.0 / (MAX_ROOMS - 1)  # remote room 1
    none = INTENT_TYPES.index("none")
    construct = INTENT_TYPES.index("createConstructionSite")
    obs["intent_mask"][0, 0, :, none] = 1
    obs["intent_mask"][0, 0, :, construct] = 1
    obs["target_select_mask"][0, construct, 0] = 1
    obs["amount_mask"][0, 0, :, construct, 0] = 1

    actor = Actor().eval()
    with torch.no_grad():
        actor.type_head.bias[none] = -50
        actor.type_head.bias[construct] = 50
        out = actor(obs, deterministic=True)
    assert int(out.types[0, 0, 0]) == none


def test_construction_has_no_direction_factor():
    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    obs["target_mask"][0, 0] = 1
    obs["targets"][0, 0, 0] = 1.0  # position target
    construct = INTENT_TYPES.index("createConstructionSite")
    obs["intent_mask"][0, 0, 0, construct] = 1
    obs["target_select_mask"][0, construct, 0] = 1
    obs["amount_mask"][0, 0, 0, construct, 0] = 1
    with torch.no_grad():
        out = Actor()(obs, deterministic=True)
    assert int(out.types[0, 0, 0]) == construct
    assert out.factor_active[0, 0].tolist() == [True, False, True, True]


def test_resource_amount_bins_are_unique_for_selected_target():
    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    obs["actors"][0, 0, 20] = 50.0 / 2000.0
    obs["actors"][0, 0, 21] = 50.0 / 2000.0
    obs["target_mask"][0, 0] = 1
    obs["targets"][0, 0, 0] = 2.0 / 6.0
    obs["targets"][0, 0, 13] = 49.0 / 1_000_000.0
    obs["targets"][0, 0, 14] = 50.0 / 1_000_000.0
    transfer = INTENT_TYPES.index("transfer")
    obs["intent_mask"][0, 0, :, transfer] = 1
    obs["target_select_mask"][0, transfer, 0] = 1
    obs["amount_mask"][0, 0, :, transfer, :] = 1

    actor = Actor().eval()
    with torch.no_grad():
        actor.type_head.bias.fill_(-50)
        actor.type_head.bias[transfer] = 50
        actor.amount_head.bias.fill_(-50)
        actor.amount_head.bias[-1] = 50  # would choose 1000 without dynamic legality
        out = actor(obs, deterministic=True)
    assert int(out.types[0, 0, 0]) == transfer
    assert int(out.amounts[0, 0, 0]) == 0  # exact 1 available; bin 1 would alias


def test_creep_cannot_transfer_to_itself():
    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    obs["actors"][0, 0, 1:3] = torch.tensor([0.2, 0.3])
    obs["actors"][0, 0, 20:22] = torch.tensor([25.0 / 2000.0, 50.0 / 2000.0])
    obs["target_mask"][0, :2] = 1
    obs["targets"][0, :, 0] = 4.0 / 6.0  # creep targets
    obs["targets"][0, 0, 1:3] = obs["actors"][0, 0, 1:3]  # self
    obs["targets"][0, 1, 1:3] = torch.tensor([0.4, 0.3])  # peer
    obs["targets"][0, :, 14] = 50.0 / 2000.0
    none = INTENT_TYPES.index("none")
    transfer = INTENT_TYPES.index("transfer")
    obs["intent_mask"][0, 0].zero_()
    obs["intent_mask"][0, 0, :, none] = 1
    obs["intent_mask"][0, 0, :, transfer] = 1
    obs["target_select_mask"][0, transfer, :2] = 1
    obs["amount_mask"][0, 0, :, transfer, 0] = 1

    actor = Actor().eval()
    with torch.no_grad():
        actor.type_head.bias.fill_(-50)
        actor.type_head.bias[transfer] = 50
        out = actor(obs, deterministic=True)
    assert int(out.types[0, 0, 0]) == transfer
    assert int(out.targets[0, 0, 0]) == 1

    # With only self available, the transfer type itself must close.
    only_self = {key: value.clone() for key, value in obs.items()}
    only_self["target_mask"][0, 1] = 0
    only_self["target_select_mask"][0, transfer, 1] = 0
    with torch.no_grad():
        out = actor(only_self, deterministic=True)
    assert int(out.types[0, 0, 0]) == none


def test_tower_attack_requires_creep_target():
    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    obs["actors"][0, 0, 0] = 1  # structure actor
    obs["actors"][0, 0, 5] = 1  # tower
    obs["target_mask"][0, 0] = 1
    obs["targets"][0, 0, 0] = 2.0 / 6.0  # hostile structure, not creep
    none = INTENT_TYPES.index("none")
    attack = INTENT_TYPES.index("attack")
    obs["intent_mask"][0, 0].zero_()
    obs["intent_mask"][0, 0, :, none] = 1
    obs["intent_mask"][0, 0, :, attack] = 1
    obs["target_select_mask"][0, attack, 0] = 1

    actor = Actor().eval()
    with torch.no_grad():
        actor.type_head.bias.fill_(-50)
        actor.type_head.bias[attack] = 50
        out = actor(obs, deterministic=True)
    assert int(out.types[0, 0, 0]) == none


def test_factorized_ppo_update_finite():
    """Exercise the real per-actor PPO update path on a complete synthetic batch."""
    from samples.rl.agent.ppo import PPOTrainer, RolloutBatch

    torch.manual_seed(11)
    batch_size = 4
    obs = _dummy_obs_batch(batch_size)
    obs["actor_mask"][:, :3] = 1
    obs["intent_mask"][:, :3, :, INTENT_TYPES.index("none")] = 1
    obs["intent_mask"][:, :3, :, INTENT_TYPES.index("move")] = 1
    obs["amount_mask"][..., 0] = 1
    agent = Agent()
    with torch.no_grad():
        out = agent.actor(obs)
        values = agent.critic(obs)
    rollout = RolloutBatch(
        obs=obs,
        actions={
            "types": out.types,
            "dirs": out.dirs,
            "targets": out.targets,
            "amounts": out.amounts,
        },
        logprob=out.actor_logprob,
        value=values,
        reward=torch.ones(batch_size),
        done=torch.zeros(batch_size),
        advantage=torch.tensor([1.0, -0.5, 0.25, 2.0]),
        ret=torch.tensor([1.0, 0.5, 1.5, 2.0]),
        batch_size=batch_size,
    )
    trainer = PPOTrainer(
        agent,
        device="cpu",
        compile_model=False,
        epochs=1,
        minibatch=2,
    )
    stats = trainer.update(rollout)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert torch.isfinite(torch.tensor(stats[key])), (key, stats[key])


def test_ppo_population_does_not_reweight_transitions():
    from samples.rl.agent.ppo import _mean_actor_then_transition

    values = torch.zeros(2, MAX_ACTORS)
    live = torch.zeros_like(values)
    values[0, 0] = 2
    live[0, 0] = 1
    live[1, :] = 1  # 64-actor state has value 0
    got = _mean_actor_then_transition(values, live)
    assert torch.allclose(got, torch.tensor(1.0)), got


def test_room_pack_keeps_expansion_capacity():
    agent = Agent()
    initial_mask = torch.tensor([[1.0] + [0.0] * (MAX_ROOMS - 1)])
    assert agent.freeze_room_pack(initial_mask) == MAX_ROOMS
    assert agent.actor.trunk._static_r == MAX_ROOMS
    assert agent.critic.trunk._static_r == MAX_ROOMS


def test_trunk_only_filter_for_reward_norm():
    """Joint/critic pretrain under reward-norm must not load raw value head."""
    critic = Critic()
    sd = critic.state_dict()
    trunk = {k: v for k, v in sd.items() if k.startswith("trunk.")}
    assert trunk, "expected trunk.* keys"
    head = {k: v for k, v in sd.items() if not k.startswith("trunk.")}
    assert head, "expected value-head keys"
    # Loading trunk only must leave head keys missing (strict=False)
    c2 = Critic()
    missing, unexpected = c2.load_state_dict(trunk, strict=False)
    assert any("value" in m or "head" in m or "out" in m or "mlp" in m or "proj" in m for m in missing) or len(missing) > 0
    assert not unexpected
    assert bool(PPO_CFG.get("normalizeReward", True)) is True


def _dummy_obs_batch(B: int) -> dict[str, torch.Tensor]:
    return {
        "patches": torch.zeros(B, MAX_ROOMS, PATCHES_PER_ROOM, PATCH_FLAT),
        "room_mask": torch.tensor([[1.0, 0, 0, 0]] * B),
        "room_coords": torch.zeros(B, MAX_ROOMS, 2),
        "actors": torch.zeros(B, MAX_ACTORS, ACTOR_FEAT),
        "actor_mask": torch.zeros(B, MAX_ACTORS),
        "targets": torch.zeros(B, MAX_TARGETS, TARGET_FEAT),
        "target_mask": torch.zeros(B, MAX_TARGETS),
        "intent_mask": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, N_INTENT),
        "dir_mask": torch.ones(B, MAX_ACTORS, INTENT_SLOTS, N_DIR),
        "target_select_mask": torch.zeros(B, N_INTENT, MAX_TARGETS),
        "amount_mask": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, N_INTENT, N_AMOUNT),
        "globals": torch.zeros(B, GLOBAL_FEAT),
    }


def test_safe_bc_nll_skips_neginf():
    from samples.rl.agent.actions_util import safe_bc_nll

    lp = torch.tensor([0.0, float("-inf"), -1.0])
    nll, frac = safe_bc_nll(lp)
    assert torch.isfinite(nll)
    assert abs(frac - 2.0 / 3.0) < 1e-5
    assert abs(float(nll) - 0.5) < 1e-5  # -mean([0, -1]) = 0.5


def test_safe_bc_nll_skips_finfo_min():
    """Masked logits use finfo.min — must not dominate BC loss."""
    from samples.rl.agent.actions_util import safe_bc_nll

    tiny = torch.finfo(torch.float32).min
    lp = torch.tensor([-0.5, tiny, -1.0])
    nll, frac = safe_bc_nll(lp)
    assert torch.isfinite(nll)
    assert abs(frac - 2.0 / 3.0) < 1e-5
    # Only -0.5 and -1.0 contribute → -mean = 0.75
    assert abs(float(nll) - 0.75) < 1e-4


def test_safe_bc_nll_strict_rejects_masked_teacher_factor():
    from samples.rl.agent.actions_util import safe_bc_nll

    lp = torch.tensor([[-0.5, torch.finfo(torch.float32).min]])
    try:
        safe_bc_nll(lp, torch.ones_like(lp, dtype=torch.bool), strict=True)
    except ValueError as error:
        assert "teacher contract violation" in str(error)
    else:
        raise AssertionError("strict BC accepted a masked teacher factor")


def test_mc_returns_tn_shapes():
    from samples.rl.agent.gae import mc_returns_tn

    T, N = 3, 2
    r = torch.ones(T, N)
    d = torch.zeros(T, N)
    d[-1] = 1
    boot = torch.zeros(N)
    ret = mc_returns_tn(r, d, gamma=0.99, next_value=boot, truncations=d)
    assert ret.shape == (T, N)
    assert float(ret[0, 0]) > 0


def test_host_rollout_buffer_write_and_flat():
    from samples.rl.agent.constants import MAX_ACTORS, INTENT_SLOTS
    from samples.rl.agent.rollout_buffer import HostRolloutBuffer

    T_max, N = 8, 2
    buf = HostRolloutBuffer(T_max, N)
    host = {k: torch.zeros((N, *shape), dtype=dtype) for k, (shape, dtype) in
            __import__("samples.rl.agent.rollout_buffer", fromlist=["_OBS_SPEC"])._OBS_SPEC.items()}
    host["room_mask"][:, 0] = 1
    types = torch.zeros(N, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long)
    for _ in range(3):
        buf.write_step(
            host_obs=host,
            types=types,
            dirs=types,
            targets=types,
            amounts=types,
            logprob=torch.zeros(N, MAX_ACTORS),
            value=torch.ones(N),
            reward=torch.ones(N) * 0.5,
            done=torch.zeros(N),
            trunc=torch.zeros(N),
        )
    assert len(buf) == 3
    flat = buf.as_flat_obs()
    assert flat["patches"].shape[0] == 3 * N
    assert float(buf.tn("reward").sum()) == 3.0
    buf.set_term_values(0, torch.tensor([1.0, 2.0]), torch.tensor([True, False]))
    assert float(buf.term_value[0, 0]) == 1.0
    assert bool(buf.has_term[0, 0]) and not bool(buf.has_term[0, 1])
    old_capacity = buf.t_max
    buf.ensure_capacity(old_capacity + 3)
    assert buf.t_max >= old_capacity + 3
    assert buf.obs["patches"].dtype == torch.uint8
    assert buf.obs["target_select_mask"].shape[-2:] == (N_INTENT, MAX_TARGETS)


def test_joint_train_chunk_shapes_and_finite():
    """_train_chunk on synthetic [T,N] data: finite NLL/vloss, no env."""
    from samples.rl.agent.pretrain_joint import _train_chunk

    torch.manual_seed(1)
    T, N = 4, 2
    device = torch.device("cpu")
    actor = Actor()
    critic = Critic()
    opt_a = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=1e-3)

    # per-t stacked host obs [N,...]
    buf_obs = []
    buf_act = []
    for _ in range(T):
        o = _dummy_obs_batch(N)
        # Match the real host contract: quantized spatial data and byte masks.
        o["patches"] = o["patches"].to(torch.uint8)
        for key in (
            "room_mask", "actor_mask", "target_mask", "intent_mask",
            "dir_mask", "target_select_mask", "amount_mask",
        ):
            o[key] = o[key].to(torch.uint8)
        o["actor_mask"][:, 0] = 1
        o["intent_mask"][:, 0, :, INTENT_TYPES.index("none")] = 1
        o["intent_mask"][:, 0, :, INTENT_TYPES.index("move")] = 1
        o["amount_mask"][..., 0] = 1
        buf_obs.append(o)
        buf_act.append({
            "types": torch.zeros(N, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long),
            "dirs": torch.zeros(N, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long),
            "targets": torch.zeros(N, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long),
            "amounts": torch.zeros(N, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long),
        })
        # legal move on live actor
        buf_act[-1]["types"][:, 0, 0] = INTENT_TYPES.index("move")

    rewards = torch.ones(T, N)
    dones = torch.zeros(T, N)
    dones[-1] = 1.0
    term_rows: list[list] = [[None] * N for _ in range(T)]
    # last step terminal obs for env 0
    term = {k: v[0:1].clone() for k, v in buf_obs[-1].items()}
    term_rows[-1][0] = term

    next_obs = {k: v.clone() for k, v in buf_obs[-1].items()}
    nll, vloss, ev, gs, hist = _train_chunk(
        actor, critic, actor, critic, opt_a, opt_c,
        buf_obs, buf_act, rewards, dones,
        next_obs=next_obs,
        term_obs_rows=term_rows,
        gamma=0.99,
        epochs=1,
        mb=4,
        value_coef=1.0,
        device=device,
        use_bf16=False,
        writer=None,
        global_step=0,
    )
    assert nll == nll and vloss == vloss, "NaN losses"
    assert gs > 0
    assert isinstance(hist, dict)


def test_joint_ckpt_meta_contract():
    """Checkpoint provenance survives a real torch serialization round-trip."""
    from samples.rl.agent.artifacts import artifact_meta, validate_artifact

    actor = Actor()
    critic = Critic()
    payload = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "meta": artifact_meta(
            "joint_pretrain", actor, critic,
            expert="scripted", curriculum="empty", reward=SCHEMA["reward"],
            critic_trunk_only_for_ppo=True,
        ),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "joint.pt"
        torch.save(payload, path)
        restored = torch.load(path, map_location="cpu", weights_only=False)
    assert restored["meta"]["schema_sha256"] == SCHEMA_SHA256
    assert restored["meta"]["reward"] == SCHEMA["reward"]
    assert restored["meta"]["expert"] == "scripted"
    validate_artifact(restored, actor, critic, kinds=("joint_pretrain",))


def test_joint_qualification_requires_critic_expansion_and_scale():
    from types import SimpleNamespace

    from samples.rl.agent.pretrain_joint import _qualification_failures

    args = SimpleNamespace(
        min_teacher_delivery=1.0,
        min_teacher_build=1.0,
        min_teacher_claims=1,
        min_teacher_creeps=8,
        min_validation_ev=0.0,
        min_closed_loop_rate=0.1,
        min_closed_loop_creeps=4,
        min_closed_loop_claims=1,
    )
    validation = {
        "validation_factor_nll": 0.5,
        "validation_legal_frac": 1.0,
        "validation_value_ev": 0.2,
    }
    closed_loop = {
        "closed_loop_skill_rate": 0.2,
        "closed_loop_invalid_frac": 0.0,
        "closed_loop_max_creeps": 4.0,
        "closed_loop_claims": 1.0,
        "closed_loop_by_curriculum": {
            "empty": {
                "transitions": 1000.0,
                "skill_rate": 0.2,
                "invalid_frac": 0.0,
                "delivery": 50.0,
                "build": 5.0,
                "claims": 1.0,
                "max_creeps": 8.0,
            },
        },
    }
    teacher_by_curriculum = {
        "empty": {
            "transitions": 6000.0,
            "skill": 100.0,
            "delivery": 50.0,
            "build": 5.0,
            "claims": 1.0,
            "max_creeps": 8.0,
            "issued": 100.0,
            "invalid": 0.0,
        },
    }
    common = dict(
        last_nll=0.5,
        last_vloss=1.0,
        corpus_hist={"_bc_legal_frac": 1.0},
        skill=0.2,
        total_delivered=50.0,
        total_built=5.0,
        total_claims=1,
        max_creeps=8,
        teacher_by_curriculum=teacher_by_curriculum,
        validation=validation,
        closed_loop=closed_loop,
    )
    assert _qualification_failures(args, **common) == []

    validation["validation_value_ev"] = -0.1
    closed_loop["closed_loop_claims"] = 0.0
    closed_loop["closed_loop_max_creeps"] = 1.0
    failures = _qualification_failures(args, **common)
    assert "validation_critic_ev" in failures
    assert "closed_loop_claim" in failures
    assert "closed_loop_population" in failures

    # Aggregate success assembled from seeded stages must not certify a model
    # that cannot bootstrap and expand from the empty curriculum.
    validation["validation_value_ev"] = 0.2
    closed_loop["closed_loop_claims"] = 1.0
    closed_loop["closed_loop_max_creeps"] = 8.0
    closed_loop["closed_loop_by_curriculum"]["empty"] = {
        "transitions": 1000.0,
        "skill_rate": 0.2,
        "invalid_frac": 0.0,
        "delivery": 0.0,
        "build": 0.0,
        "claims": 0.0,
        "max_creeps": 1.0,
        "issued": 100.0,
        "invalid": 1.0,
    }
    teacher_by_curriculum["empty"] = {
        "transitions": 6000.0,
        "skill": 10.0,
        "delivery": 0.0,
        "build": 0.0,
        "claims": 0.0,
        "max_creeps": 1.0,
        "issued": 100.0,
        "invalid": 1.0,
    }
    failures = _qualification_failures(args, **common)
    assert "empty_teacher_claim" in failures
    assert "empty_teacher_population" in failures
    assert "teacher_engine_legality" in failures
    assert "empty_teacher_engine_legality" in failures
    assert "empty_closed_loop_claim" in failures
    assert "empty_closed_loop_population" in failures


def test_artifact_rejects_partial_actor_state():
    from samples.rl.agent.artifacts import artifact_meta, load_full_state

    actor = Actor()
    critic = Critic()
    state = dict(actor.state_dict())
    state.pop(next(iter(state)))
    payload = {"actor": state, "critic": critic.state_dict(), "meta": artifact_meta(
        "joint_pretrain", actor, critic,
    )}
    try:
        load_full_state(actor, payload["actor"], name="actor")
    except ValueError as error:
        assert "incomplete or incompatible" in str(error)
    else:
        raise AssertionError("partial actor state loaded without error")


def test_pack_field_order_documented():
    """Encoder, binary frame writer, and Python decoder must share one order."""
    order = (
        "patches",
        "actors",
        "targets",
        "room_coords",
        "room_mask",
        "actor_mask",
        "target_mask",
        "intent_mask",
        "dir_mask",
        "target_select_mask",
        "amount_mask",
    )
    import re

    encode_source = (_RL / "env" / "encode.mjs").read_text(encoding="utf-8")
    encode_match = re.search(
        r"const tensorParts = \[(.*?)\];", encode_source, flags=re.DOTALL,
    )
    assert encode_match is not None
    encode_names = tuple(re.findall(
        r"\b(?:patchPacked|actors|targets|roomCoords|rm|am|tm|im|dm|tsm|amm)\b",
        encode_match.group(1),
    ))
    assert encode_names == (
        "patchPacked", "actors", "targets", "roomCoords", "rm", "am", "tm", "im", "dm", "tsm", "amm",
    )

    server_source = (_RL / "env" / "server.mjs").read_text(encoding="utf-8")
    server_match = re.search(r"parts = \[(.*?)\];", server_source, flags=re.DOTALL)
    assert server_match is not None
    server_names = tuple(re.findall(r"raw\.(\w+)", server_match.group(1)))
    assert server_names == (
        "patches", "actors", "targets", "roomCoords", "roomMask", "actorMask",
        "targetMask", "intentMask", "dirMask", "targetSelectMask", "amountMask",
    )

    client_source = (_RL / "agent" / "env_client.py").read_text(encoding="utf-8")
    client_match = re.search(
        r"patches = take_u8\(.*?amount_mask = take_u8\(.*?\n",
        client_source,
        flags=re.DOTALL,
    )
    assert client_match is not None
    client_names = tuple(re.findall(
        r"^\s*(patches|actors|targets|room_coords|room_mask|actor_mask|target_mask|"
        r"intent_mask|dir_mask|target_select|amount_mask)\s*=\s*take_",
        client_match.group(0),
        flags=re.MULTILINE,
    ))
    assert client_names == (
        "patches", "actors", "targets", "room_coords", "room_mask", "actor_mask",
        "target_mask", "intent_mask", "dir_mask", "target_select", "amount_mask",
    )
    assert tuple("target_select_mask" if name == "target_select" else name for name in client_names) == order


def main() -> int:
    tests = [
        test_gae_truncation_bootstraps,
        test_gae_cuts_chain_on_done,
        test_type_gated_logprob_none_low_entropy,
        test_hl_gauss_support,
        test_reward_normalizer_roundtrip,
        test_critic_scalar_default,
        test_schema_reward_values_productive_economy,
        test_intent_enum_parity,
        test_logprob_sample_matches_evaluate,
        test_entity_context_changes_peer_policy_and_critic,
        test_per_actor_logprob_factorization,
        test_structure_cannot_select_remote_target_intent,
        test_creep_construction_target_is_same_room_only,
        test_construction_has_no_direction_factor,
        test_resource_amount_bins_are_unique_for_selected_target,
        test_creep_cannot_transfer_to_itself,
        test_tower_attack_requires_creep_target,
        test_factorized_ppo_update_finite,
        test_ppo_population_does_not_reweight_transitions,
        test_room_pack_keeps_expansion_capacity,
        test_trunk_only_filter_for_reward_norm,
        test_safe_bc_nll_skips_neginf,
        test_safe_bc_nll_skips_finfo_min,
        test_safe_bc_nll_strict_rejects_masked_teacher_factor,
        test_mc_returns_tn_shapes,
        test_host_rollout_buffer_write_and_flat,
        test_joint_train_chunk_shapes_and_finite,
        test_joint_ckpt_meta_contract,
        test_joint_qualification_requires_critic_expansion_and_scale,
        test_artifact_rejects_partial_actor_state,
        test_pack_field_order_documented,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
