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
    ACTOR_FEATURE_INDEX,
    BODY_PART_COSTS,
    CONSTRUCTION_MASK_BYTES,
    GLOBAL_FEAT,
    INTENT_SLOTS,
    INTENT_TYPES,
    MAX_ACTORS,
    MAX_BODY_PARTS,
    MAX_ROOM_ENERGY,
    MAX_ROOMS,
    MAX_TARGETS,
    N_AMOUNT,
    N_BODY_PART,
    N_CONSTRUCTION_TYPE,
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
from samples.rl.agent.gae import (
    compute_gae_tn,
    decoupled_gae,
    length_adaptive_lambda,
    segment_lengths_tn,
    segment_returns_per_env,
    self_imitation_mask,
)
from samples.rl.agent.hl_gauss import HLGaussSupport
from samples.rl.agent.model import ActionConditionedDynamics, Actor, Agent, Critic
from samples.rl.agent.muon import (
    MUON_BETA2,
    HybridMuonAdamW,
    normuon_group_step,
    optimizer_parameter_counts,
    polar_express,
    split_hidden_matrices,
)
from samples.rl.agent.running_stats import RewardNormalizer

_AF = ACTOR_FEATURE_INDEX


def test_muon_partition_covers_hidden_matrices_only():
    actor = Actor()
    critic = Critic()
    actor_muon, actor_adam, actor_muon_names, actor_adam_names = (
        split_hidden_matrices(actor)
    )
    critic_muon, critic_adam, critic_muon_names, critic_adam_names = (
        split_hidden_matrices(critic)
    )
    assert sum(parameter.numel() for parameter in actor_muon) == 983_040
    assert sum(parameter.numel() for parameter in critic_muon) == 983_040
    assert sum(parameter.numel() for parameter in (*actor_muon, *actor_adam)) == sum(
        parameter.numel() for parameter in actor.parameters()
    )
    assert sum(parameter.numel() for parameter in (*critic_muon, *critic_adam)) == sum(
        parameter.numel() for parameter in critic.parameters()
    )
    assert all(
        (".blocks." in name or ".entity_blocks." in name)
        and name.endswith(".weight")
        for name in actor_muon_names
    )
    assert all(
        (".blocks." in name or ".entity_blocks." in name)
        and name.endswith(".weight")
        for name in critic_muon_names
    )
    assert "body_count_head.weight" in actor_adam_names
    assert "value_head.4.weight" in critic_adam_names
    assert "trunk.body_count_embed.weight" in actor_adam_names


def test_polar_express_beats_newton_schulz_on_small_singular_values():
    """Why Polar Express replaced Newton-Schulz: the same five rounds, closer to
    the polar factor. Mean singular value must reach ~1 instead of ~0.87, and
    the smallest direction of an ill-conditioned matrix must be lifted further.
    """
    from torch.optim._muon import _zeropower_via_newtonschulz as newton_schulz

    generator = torch.Generator().manual_seed(0)
    for rows, cols in ((64, 64), (48, 12), (12, 48)):
        rank = min(rows, cols)
        base = torch.randn(3, rows, cols, generator=generator)
        left, _, right = torch.linalg.svd(base, full_matrices=False)
        # Condition number 1e3: the hard case for a fixed-coefficient iteration.
        spectrum = torch.linspace(1e-3, 1.0, rank).flip(0).expand(3, -1)
        matrices = left @ torch.diag_embed(spectrum) @ right

        singular = torch.linalg.svdvals(polar_express(matrices).float())
        assert singular.shape == (3, rank)
        assert abs(float(singular.mean()) - 1.0) < 0.05, singular
        assert float(singular.max()) < 1.2, singular

        reference = torch.stack([
            newton_schulz(matrices[i], (3.4445, -4.7750, 2.0315), 5, 1e-7).float()
            for i in range(matrices.shape[0])
        ])
        reference_singular = torch.linalg.svdvals(reference)
        assert float(singular.min()) > float(reference_singular.min())
        assert float(singular.mean()) > float(reference_singular.mean())

        # The polar factor keeps the original singular vectors, so the update
        # still points along the gradient it came from.
        assert bool(((polar_express(matrices) * matrices).sum(dim=(-2, -1)) > 0).all())


def test_normuon_step_preserves_update_norm():
    """NorMuon reweights rows but renormalizes, so `muon_lr` keeps its meaning."""
    generator = torch.Generator().manual_seed(1)
    grads = torch.randn(2, 8, 8, generator=generator)
    params = torch.randn(2, 8, 8, generator=generator)
    before = params.clone()
    momentum_buffer = torch.zeros_like(params)
    second_moment = torch.zeros(2, 8, 1)
    momentum, lr = 0.85, 0.01
    normuon_group_step(
        params, grads.clone(), momentum_buffer, second_moment,
        torch.tensor(momentum), torch.tensor(lr), torch.tensor(0.0),
        MUON_BETA2, torch.float32,
    )
    assert torch.isfinite(params).all()
    # Reconstruct the orthogonalized direction the step must have taken.
    buffer = torch.zeros_like(before)
    buffer.lerp_(grads, 1.0 - momentum)
    direction = polar_express(grads.clone().lerp_(buffer, momentum))
    step = before - params
    assert abs(float(step.norm()) / float(direction.norm() * lr) - 1.0) < 1e-4
    # The reweighting is real: individual rows move by different multiples of
    # the orthogonalized direction even though the total norm is unchanged.
    row_ratio = (step / (direction * lr)).mean(dim=-1)
    assert float(row_ratio.max() - row_ratio.min()) > 1e-3
    assert float(second_moment.max()) > 0.0
    assert torch.allclose(momentum_buffer, buffer, atol=1e-7)


def test_normuon_decay_is_cautious():
    """Weight decay applies only where the learned step already shrinks the weight."""
    generator = torch.Generator().manual_seed(2)
    grads = torch.randn(1, 8, 8, generator=generator)
    params = torch.randn(1, 8, 8, generator=generator)
    lr, weight_decay = 0.01, 0.5
    plain = params.clone()
    decayed = params.clone()
    normuon_group_step(
        plain, grads.clone(), torch.zeros_like(params), torch.zeros(1, 8, 1),
        torch.tensor(0.85), torch.tensor(lr), torch.tensor(0.0),
        MUON_BETA2, torch.float32,
    )
    normuon_group_step(
        decayed, grads.clone(), torch.zeros_like(params), torch.zeros(1, 8, 1),
        torch.tensor(0.85), torch.tensor(lr), torch.tensor(lr * weight_decay),
        MUON_BETA2, torch.float32,
    )
    learned_step = params - plain
    shrinking = (learned_step * params) > 0
    assert bool(shrinking.any()) and bool((~shrinking).any())
    # Untouched where the step grows the weight; pulled toward zero elsewhere.
    assert torch.allclose(decayed[~shrinking], plain[~shrinking], atol=1e-7)
    extra = plain[shrinking] - decayed[shrinking]
    # Decay is one power of the learning rate, matching the rate this stack
    # tuned `muon_weight_decay` against.
    expected = params[shrinking] * (lr * weight_decay)
    assert torch.allclose(extra, expected, atol=1e-7)


def test_muon_second_moment_reweights_anisotropic_rows():
    """Rows with persistently large updates must be damped relative to quiet rows."""
    params = torch.zeros(1, 4, 4)
    grads = torch.zeros(1, 4, 4)
    grads[0, 0] = 4.0
    grads[0, 1] = 0.05
    momentum_buffer = torch.zeros_like(params)
    second_moment = torch.zeros(1, 4, 1)
    for _ in range(6):
        normuon_group_step(
            params, grads.clone(), momentum_buffer, second_moment,
            torch.tensor(0.95), torch.tensor(0.01), torch.tensor(0.0),
            MUON_BETA2, torch.float32,
        )
    assert float(second_moment[0, 0]) > float(second_moment[0, 1])


def test_muon_batched_group_matches_one_matrix_at_a_time():
    """Stacking matrices must not couple them: no statistic may span the group."""
    generator = torch.Generator().manual_seed(3)
    params = torch.randn(3, 12, 8, generator=generator)
    grads = torch.randn(3, 12, 8, generator=generator)
    momentum, lr, decay = 0.9, 0.02, 0.01

    batched = params.clone()
    batched_momentum = torch.randn(3, 12, 8, generator=generator)
    batched_second = torch.rand(3, 12, 1, generator=generator)
    single_momentum = batched_momentum.clone()
    single_second = batched_second.clone()
    normuon_group_step(
        batched, grads.clone(), batched_momentum, batched_second,
        torch.tensor(momentum), torch.tensor(lr), torch.tensor(decay),
        MUON_BETA2, torch.float32,
    )

    for index in range(params.shape[0]):
        single = params[index : index + 1].clone()
        normuon_group_step(
            single, grads[index : index + 1].clone(),
            single_momentum[index : index + 1],
            single_second[index : index + 1],
            torch.tensor(momentum), torch.tensor(lr), torch.tensor(decay),
            MUON_BETA2, torch.float32,
        )
        assert torch.allclose(batched[index], single[0], atol=1e-6), index
    assert torch.allclose(batched_momentum, single_momentum, atol=1e-6)
    assert torch.allclose(batched_second, single_second, atol=1e-6)


def test_ppo_trainer_uses_hybrid_muon_with_rms_matched_rate():
    """PPO's trunk rate is derived from the AdamW step it replaced, not inherited."""
    from samples.rl.agent.muon import PPO_MUON_LR
    from samples.rl.agent.ppo import PPOTrainer

    trainer = PPOTrainer(actor=Actor(), critic=Critic(), device="cpu")
    assert isinstance(trainer.actor_opt, HybridMuonAdamW)
    assert isinstance(trainer.critic_opt, HybridMuonAdamW)
    assert trainer.actor_opt.muon_lr == PPO_MUON_LR
    assert trainer.critic_opt.muon_lr == PPO_MUON_LR * 2.0
    assert trainer.critic_opt.adam_lr == trainer.actor_opt.adam_lr * 2.0
    # An orthogonalized step moves each coordinate of these matrices by roughly
    # muon_lr * sqrt(min(R, C)) * sqrt(max(1, R/C)) / sqrt(R*C); that must land
    # within a factor of two of the AdamW rate it replaced.
    for group in trainer.actor_opt.groups:
        rows, cols = group.shape
        rms = (
            trainer.actor_opt.muon_lr * group.lr_scale
            * (min(rows, cols) ** 0.5) / ((rows * cols) ** 0.5)
        )
        assert 0.5 <= rms / trainer.actor_opt.adam_lr <= 2.0, (group.shape, rms)

    trainer.actor_opt.set_learning_rates(adam_lr=5e-5, muon_lr=6e-4)
    assert trainer.actor_opt.adam_lr == 5e-5
    assert trainer.actor_opt.muon_lr == 6e-4


def test_hybrid_muon_optimizer_state_roundtrip():
    class TinyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Module()
            self.trunk.room_enc = torch.nn.Module()
            self.trunk.room_enc.blocks = torch.nn.ModuleList([
                torch.nn.ModuleDict({"wq": torch.nn.Linear(4, 4)})
            ])
            self.head = torch.nn.Linear(4, 2)

        def forward(self, value):
            value = self.trunk.room_enc.blocks[0]["wq"](value)
            return self.head(value)

    model = TinyTransformer()
    optimizer = HybridMuonAdamW(
        model, adam_lr=3e-4, muon_lr=1e-2, muon_weight_decay=2.5e-2,
    )
    assert optimizer_parameter_counts(optimizer) == {"muon": 16, "adamw": 14}
    loss = model(torch.randn(8, 4)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert optimizer.step_count == 1
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    # One step into a 300-step ramp from 0.85 to 0.95.
    assert abs(optimizer.momentum - (0.85 + (0.95 - 0.85) / 300)) < 1e-9

    restored_model = TinyTransformer()
    restored = HybridMuonAdamW(
        restored_model, adam_lr=3e-4, muon_lr=1e-2, muon_weight_decay=2.5e-2,
    )
    restored.load_state_dict(optimizer.state_dict())
    assert restored.step_count == 1
    assert torch.equal(
        restored.groups[0].momentum_buffer, optimizer.groups[0].momentum_buffer,
    )
    assert torch.equal(
        restored.groups[0].second_moment, optimizer.groups[0].second_moment,
    )

    # A stale format or a different hidden-matrix layout must fail loudly, not
    # silently restore a mismatched population.
    stale = optimizer.state_dict()
    stale["format"] = 1
    try:
        restored.load_state_dict(stale)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("stale optimizer format must be rejected")


def test_muon_group_batches_same_shape_matrices():
    """Equal-shape matrices share one stacked update; distinct shapes do not."""
    actor = Actor()
    optimizer = HybridMuonAdamW(actor, adam_lr=3e-4)
    assert [group.shape for group in optimizer.groups] == [
        (128, 128), (128, 512), (512, 128),
    ]
    assert [len(group.params) for group in optimizer.groups] == [20, 5, 5]
    assert optimizer.groups[0].lr_scale == 1.0
    # Muon's original aspect-ratio adjustment: sqrt(max(1, rows/cols)).
    assert optimizer.groups[1].lr_scale == 1.0
    assert optimizer.groups[2].lr_scale == 2.0


def test_muon_rejects_partial_gradients():
    """A half-populated group is a bug in the loss, not something to average over."""
    actor = Actor()
    optimizer = HybridMuonAdamW(actor, adam_lr=3e-4)
    group = optimizer.groups[0]
    for parameter in group.params:
        parameter.grad = torch.zeros_like(parameter)
    group.params[3].grad = None
    try:
        optimizer.step()
    except RuntimeError as error:
        assert "partial gradients" in str(error)
    else:
        raise AssertionError("a partially populated Muon group must be rejected")


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


def test_gae_cuts_chain_on_truncation_without_done():
    """A start-state segment boundary truncates without ending the episode."""
    T, N = 4, 1
    rewards = torch.tensor([[1.0], [1.0], [0.0], [100.0]])
    values = torch.zeros(T, N)
    # Segment boundary at t=1: the environment episode continues in a different
    # world, so `done` stays 0 while the trajectory is cut and bootstrapped.
    dones = torch.zeros(T, N)
    trunc = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    next_values = torch.zeros(T, N)
    adv, ret = compute_gae_tn(
        rewards, values, dones, gamma=0.99, lam=1.0,
        next_value=torch.zeros(N), truncations=trunc, next_values_tn=next_values,
    )
    # t=1 sees only its own reward plus the spliced terminal value.
    assert abs(float(adv[1, 0]) - 1.0) < 1e-6, adv
    # t=0 sees t=1 but nothing from the restored world at t>=2.
    assert abs(float(adv[0, 0]) - (1.0 + 0.99)) < 1e-6, adv
    assert float(adv[2, 0]) > 90.0, adv


def test_decoupled_gae_pairs_policy_lambda_advantages_with_lambda_one_targets():
    """One pass, two estimators: independent recurrence for each λ."""
    generator = torch.Generator().manual_seed(17)
    T, N = 37, 5
    rewards = torch.randn(T, N, generator=generator)
    values = torch.randn(T, N, generator=generator)
    dones = torch.zeros(T, N)
    dones[8, 0] = dones[13, 2] = dones[31, 4] = 1
    trunc = torch.zeros_like(dones)
    trunc[8, 0] = trunc[31, 4] = 1
    # Segment boundaries truncate without ending the episode.
    trunc[20, 1] = trunc[5, 3] = 1
    next_values = torch.randn(T, N, generator=generator)
    lambda_policy = torch.tensor([0.95, 0.5, 0.99, 0.0, 0.8])
    gamma = 0.9995

    def recurrence(lam: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(rewards)
        last_gae = torch.zeros(N)
        for t in reversed(range(T)):
            terminated = dones[t] * (1.0 - trunc[t])
            delta = rewards[t] + gamma * next_values[t] * (1.0 - terminated) - values[t]
            cut = torch.clamp(dones[t] + trunc[t], max=1.0)
            last_gae = delta + gamma * lam * (1.0 - cut) * last_gae
            out[t] = last_gae
        return out

    expected_adv = recurrence(lambda_policy)
    expected_lambda_one = recurrence(torch.ones(N))
    actual_adv, actual_ret, info = decoupled_gae(
        rewards, values, dones, gamma=gamma, lambda_policy=lambda_policy,
        truncations=trunc, next_values_tn=next_values,
    )
    assert torch.equal(actual_adv, expected_adv)
    # The critic target is the λ=1 return, not `policy advantage + value`.
    assert torch.equal(actual_ret, expected_lambda_one + values)
    assert not torch.equal(actual_ret, actual_adv + values)
    assert info["gamma"] == gamma
    assert info["gae_lambda_critic"] == 1.0
    assert abs(info["gae_lambda_policy_mean"] - float(lambda_policy.mean())) < 1e-9
    assert info["gae_lambda_policy_min"] == 0.0
    assert abs(
        info["critic_effective_horizon"] - 1 / (1 - gamma)
    ) < 1e-6
    # The reported policy horizon is the mean of the per-λ windows, not the
    # window of the mean λ; the two differ by Jensen once λ varies, and the
    # former is what the rollout's transitions actually got.
    per_lambda = 1.0 / (1.0 - gamma * lambda_policy)
    assert abs(info["policy_effective_horizon"] - float(per_lambda.mean())) < 1e-4
    assert abs(info["policy_effective_horizon_min"] - float(per_lambda.min())) < 1e-4
    assert info["policy_effective_horizon"] > 1 / (
        1 - gamma * float(lambda_policy.mean())
    )


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
        "construction_mask": torch.zeros(
            B, MAX_ROOMS, N_CONSTRUCTION_TYPE, CONSTRUCTION_MASK_BYTES,
            dtype=torch.uint8,
        ),
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


def test_hl_gauss_symlog_high_return_geometry_and_decode():
    critic = Critic()
    support = critic.support
    assert support.num_bins == 409
    assert support.centers[support.num_bins // 2] == 0
    targets = torch.tensor([
        -1_000_000_000.0,
        -1_000_000.0,
        -8_648.0,  # representative large shaped return within the support
        0.0,
        8_648.0,
        1_000_000.0,
        1_000_000_000.0,
    ])
    labels = support.project(targets)
    assert torch.allclose(labels.sum(-1), torch.ones_like(targets), atol=1e-6)
    decoded = support.to_expected_scalar(labels.clamp_min(1e-30).log())
    # HL-Gauss is Gaussian in symlog coordinates, where these anchored targets
    # decode exactly up to float32 precision.
    assert torch.allclose(
        torch.log1p(decoded.abs()), torch.log1p(targets.abs()), atol=2e-5,
    )
    diagnostics = support.target_diagnostics(targets)
    assert diagnostics["overflow_count"].item() == 0
    assert diagnostics["saturation_fraction"].item() == 0


def test_hl_gauss_rejects_instead_of_clamping_overflow():
    support = Critic().support
    try:
        support.project(torch.tensor([2_000_000_000.0]))
    except ValueError as error:
        assert "outside declared raw-return support" in str(error)
        assert "overflow_count=1" in str(error)
    else:
        raise AssertionError("out-of-support return was silently projected")


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
    assert c.use_hl_gauss is True
    B = 1
    batch = {
        "patches": torch.zeros(B, MAX_ROOMS, PATCHES_PER_ROOM, PATCH_FLAT),
        "room_mask": torch.tensor([[1.0, 0, 0, 0]]),
        "room_coords": torch.zeros(B, MAX_ROOMS, 2),
        "actors": torch.zeros(B, MAX_ACTORS, ACTOR_FEAT),
        "actor_mask": torch.zeros(B, MAX_ACTORS),
        "actor_outcome": torch.zeros(B, MAX_ACTORS),
        "targets": torch.zeros(B, MAX_TARGETS, TARGET_FEAT),
        "target_mask": torch.zeros(B, MAX_TARGETS),
        "globals": torch.zeros(B, GLOBAL_FEAT),
    }
    v = c(batch)
    assert v.shape == (1,)
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-5)


def test_critic_value_head_stays_fp32_under_autocast():
    critic = Critic()
    obs = _dummy_obs_batch(1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = critic(obs, return_logits=True)
    assert logits.dtype == torch.float32


def test_value_contract_has_no_clipping_and_fixed_critic_grad_clip():
    source = (_RL / "agent" / "ppo.py").read_text(encoding="utf-8")
    assert "clipValueLoss" not in source
    assert "v_clipped" not in source
    assert float(SCHEMA["value"]["criticMaxGradNorm"]) == 0.5
    assert SCHEMA["value"]["loss"] == "hlGauss"
    assert SCHEMA["value"]["transform"] == "symlog"
    assert int(SCHEMA["artifact"]["learningAbi"]) >= 12
    assert float(SCHEMA["ppo"]["gamma"]) == 0.9995
    assert "gaeLambda" not in SCHEMA["ppo"]
    assert float(SCHEMA["ppo"]["gaeLambdaCritic"]) == 1.0
    assert float(SCHEMA["ppo"]["gaeLambdaPolicyAlpha"]) == 0.5
    assert float(SCHEMA["ppo"]["selfImitationCoef"]) == 0.1
    assert float(SCHEMA["ppo"]["selfImitationQuantile"]) == 0.8
    assert int(SCHEMA["ppo"]["groupStartsPerState"]) == 2
    assert int(SCHEMA["ppo"]["numEnvs"]) == 24
    # Launch budget is a measured ceiling, not taste: minibatch 1536 OOMs and
    # 24x4096 exceeds host RAM once environments hold four live rooms
    # (docs/PERFORMANCE.md#the-measured-ceiling-is-at-capacity-not-at-typical-occupancy).
    assert int(SCHEMA["ppo"]["rolloutSteps"]) == 2048
    assert int(SCHEMA["ppo"]["maxRolloutSteps"]) == 2048
    assert int(SCHEMA["ppo"]["minibatch"]) == 1024
    assert int(SCHEMA["ppo"]["epochs"]) == 3
    assert 2048 * 24 // 1024 == 48
    assert int(SCHEMA["nextLat"]["horizon"]) == 1


def test_schema_reward_values_productive_economy():
    r = SCHEMA["reward"]
    assert set(key for key in r if not key.startswith("_")) == {
        "energyHarvested", "controlPoints",
    }
    assert 0 < float(r["energyHarvested"]) < float(r["controlPoints"])
    assert float(r["controlPoints"]) == 1.0
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
        "construction_mask": torch.zeros(
            B, MAX_ROOMS, N_CONSTRUCTION_TYPE, CONSTRUCTION_MASK_BYTES,
            dtype=torch.uint8,
        ),
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
    # The production prior intentionally initializes the intent projection at a
    # tiny gain; use an ordinary trained-head scale so this test probes whether
    # peer context can affect policy rather than initialization magnitude.
    with torch.no_grad():
        actor.type_head.weight.normal_(std=0.2)
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


def test_creep_token_contains_exact_body_and_storage_state():
    actor = Actor().eval()
    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    obs["actors"][0, 0, _AF["totalWork"]] = 1.0 / MAX_BODY_PARTS
    obs["actors"][0, 0, _AF["activeWork"]] = 1.0 / MAX_BODY_PARTS
    with torch.no_grad():
        before = actor.trunk(
            obs["patches"], obs["room_mask"], obs["room_coords"],
            obs["actors"], obs["actor_mask"], obs["actor_outcome"],
            obs["targets"], obs["target_mask"], obs["globals"],
        )[0]
        changed = {key: value.clone() for key, value in obs.items()}
        changed["actors"][0, 0, _AF["totalWork"]] = 2.0 / MAX_BODY_PARTS
        changed["actors"][0, 0, _AF["activeWork"]] = 2.0 / MAX_BODY_PARTS
        changed["actors"][0, 0, _AF["storedEnergy"]] = 50.0 / MAX_ROOM_ENERGY
        changed["actors"][0, 0, _AF["storeCapacity"]] = 100.0 / MAX_ROOM_ENERGY
        after = actor.trunk(
            changed["patches"], changed["room_mask"], changed["room_coords"],
            changed["actors"], changed["actor_mask"], changed["actor_outcome"],
            changed["targets"], changed["target_mask"], changed["globals"],
        )[0]
    assert not torch.allclose(before[0, 0], after[0, 0])


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


def test_immobile_creep_can_only_select_targets_already_in_primitive_range():
    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    obs["actors"][0, 0, 1:3] = torch.tensor([0.2, 0.3])
    obs["target_mask"][0, 0] = 1
    obs["targets"][0, 0, 0] = 0.0  # source
    obs["targets"][0, 0, 1:3] = torch.tensor([0.5, 0.3])
    none = INTENT_TYPES.index("none")
    harvest = INTENT_TYPES.index("harvest")
    obs["intent_mask"][0, 0].zero_()
    obs["intent_mask"][0, 0, :, none] = 1
    obs["intent_mask"][0, 0, :, harvest] = 1
    obs["target_select_mask"][0, harvest, 0] = 1

    actor = Actor().eval()
    with torch.no_grad():
        actor.type_head.bias.fill_(-50)
        actor.type_head.bias[harvest] = 50
        distant = actor(obs, deterministic=True)
    assert int(distant.types[0, 0, 0]) == none

    obs["targets"][0, 0, 1] = obs["actors"][0, 0, 1] + 1.0 / 49.0
    with torch.no_grad():
        adjacent = actor(obs, deterministic=True)
    assert int(adjacent.types[0, 0, 0]) == harvest


def test_creep_cannot_issue_room_construction():
    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    none = INTENT_TYPES.index("none")
    construct = INTENT_TYPES.index("createConstructionSite")
    obs["intent_mask"][0, 0, :, none] = 1

    actor = Actor().eval()
    with torch.no_grad():
        actor.type_head.bias[none] = -50
        actor.type_head.bias[construct] = 50
        out = actor(obs, deterministic=True)
    assert int(out.types[0, 0, 0]) == none


def test_construction_uses_exact_type_and_tile_factors():
    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    obs["actors"][0, 0, 0] = 1
    obs["actors"][0, 0, _AF["isRoom"]] = 1
    construct = INTENT_TYPES.index("createConstructionSite")
    obs["intent_mask"][0, 0, 0, construct] = 1
    type_index = 2
    tile = 17 * 50 + 23
    obs["construction_mask"][0, 0, type_index, tile // 8] |= 1 << (tile % 8)
    with torch.no_grad():
        out = Actor()(obs, deterministic=True)
    assert int(out.types[0, 0, 0]) == construct
    assert int(out.construction_types[0, 0, 0]) == type_index
    assert int(out.construction_tiles[0, 0, 0]) == tile
    assert out.factor_active[0, 0, :7].tolist() == [
        True, False, False, False, True, True, False,
    ]
    assert not bool(out.factor_active[0, 0, 7:].any())


def test_construction_sample_and_reevaluate_logprob_match():
    from unittest import mock

    torch.manual_seed(19)
    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    obs["actors"][0, 0, 0] = 1
    obs["actors"][0, 0, _AF["isRoom"]] = 1
    construct = INTENT_TYPES.index("createConstructionSite")
    obs["intent_mask"][0, 0, 0, construct] = 1
    for type_index, tiles in ((0, (101, 777)), (3, (456, 2048))):
        for tile in tiles:
            obs["construction_mask"][0, 0, type_index, tile // 8] |= 1 << (tile % 8)
    actor = Actor().eval()
    original_einsum = torch.einsum

    def reject_integer_einsum(equation, *operands):
        assert all(operand.dtype != torch.long for operand in operands), (
            f"categorical indices reached einsum {equation}"
        )
        return original_einsum(equation, *operands)

    with torch.no_grad():
        with mock.patch.object(torch, "einsum", side_effect=reject_integer_einsum):
            sampled = actor(obs)
            action = {
                key: getattr(sampled, key)
                for key in (
                    "types", "dirs", "targets", "amounts", "construction_types",
                    "construction_tiles", "body_counts", "body_order",
                )
            }
            evaluated = actor(obs, action=action)
    assert torch.allclose(
        sampled.actor_logprob, evaluated.actor_logprob, atol=1e-5, rtol=1e-5,
    )


def test_spawn_count_chain_is_exactly_affordable_and_reevaluable():
    obs = _dummy_obs_batch(1)
    spawn = INTENT_TYPES.index("spawnCreep")
    none = INTENT_TYPES.index("none")
    obs["actor_mask"][0, 0] = 1
    obs["actors"][0, 0, _AF["isNonCreep"]] = 1
    obs["actors"][0, 0, _AF["isSpawn"]] = 1
    obs["actors"][0, 0, _AF["roomEnergyAvailable"]] = 300.0 / MAX_ROOM_ENERGY
    obs["intent_mask"][0, 0, 0, none] = 1
    obs["intent_mask"][0, 0, 0, spawn] = 1
    actor = Actor().eval()
    with torch.no_grad():
        actor.type_head.bias.fill_(-50)
        actor.type_head.bias[spawn] = 50
        for _ in range(4):
            out = actor(obs)
            assert int(out.types[0, 0, 0]) == spawn
            counts = out.body_counts[0, 0, 0]
            order = out.body_order[0, 0, 0]
            length = int(counts.sum())
            assert 1 <= length <= MAX_BODY_PARTS
            assert sorted(order.tolist()) == list(range(N_BODY_PART))
            positive_count = int((counts > 0).sum())
            assert bool((counts[order[:positive_count]] > 0).all())
            assert order[positive_count:].tolist() == sorted(order[positive_count:].tolist())
            cost = sum(
                BODY_PART_COSTS[part] * int(counts[part]) for part in range(N_BODY_PART)
            )
            assert cost <= 300
            action = {
                key: getattr(out, key)
                for key in (
                    "types", "dirs", "targets", "amounts",
                    "body_counts", "body_order",
                )
            }
            evaluated = actor(obs, action=action)
            assert torch.allclose(
                out.actor_logprob[0, 0], evaluated.actor_logprob[0, 0], atol=2e-5,
            )

        # Below the cheapest body cost, spawn itself closes and none remains.
        obs["actors"][0, 0, _AF["roomEnergyAvailable"]] = 0
        masked = actor(obs, deterministic=True)
        assert int(masked.types[0, 0, 0]) == none
        assert masked.body_counts[0, 0, 0].tolist() == [0] * N_BODY_PART
        assert masked.body_order[0, 0, 0].tolist() == list(range(N_BODY_PART))

    actor.zero_grad(set_to_none=True)
    affordable_obs = {key: value.clone() for key, value in obs.items()}
    affordable_obs["actors"][0, 0, _AF["roomEnergyAvailable"]] = 300.0 / MAX_ROOM_ENERGY
    differentiable = actor(affordable_obs)
    spawn_loss = -differentiable.actor_logprob[0, 0]
    assert torch.isfinite(spawn_loss)
    spawn_loss.backward()
    actor_grads = [parameter.grad for parameter in actor.parameters() if parameter.grad is not None]
    assert actor_grads
    assert all(torch.isfinite(gradient).all() for gradient in actor_grads)


def test_spawn_count_chain_has_full_support_without_forced_cheap_suffix():
    actor = Actor().eval()
    logits = torch.zeros(2, N_BODY_PART, MAX_BODY_PARTS + 1, requires_grad=True)
    counts = torch.zeros(2, N_BODY_PART, dtype=torch.long)
    counts[0, 0] = 6  # six MOVE, exactly 300 energy
    counts[1, -1] = 30  # thirty TOUGH, also exactly 300 energy
    sampled, logprob, entropy = actor._budget_conditioned_counts(
        logits, torch.tensor([300, 300]), False, counts,
    )
    assert torch.equal(sampled, counts)
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()
    (-logprob.mean()).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_spawn_order_canonicalizes_zero_aliases_and_rejects_bad_wire_order():
    from samples.rl.agent.actions_util import safe_bc_nll

    actor = Actor().eval()
    logits = torch.randn(1, N_BODY_PART, requires_grad=True)
    counts = torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0]])
    canonical = torch.arange(N_BODY_PART).view(1, -1)
    order, lp, entropy, active = actor._positive_type_order(
        logits, counts, False, canonical,
    )
    assert torch.equal(order, canonical)
    assert int(active.sum()) == 1  # contract sentinel; no stochastic order choice
    assert float(lp.detach().sum()) == 0.0
    assert float(entropy.detach().sum()) == 0.0

    aliased = canonical.clone()
    aliased[0, 1], aliased[0, 2] = aliased[0, 2].clone(), aliased[0, 1].clone()
    _, bad_lp, _, bad_active = actor._positive_type_order(
        logits, counts, False, aliased,
    )
    try:
        safe_bc_nll(bad_lp, bad_active, strict=True)
    except ValueError:
        pass
    else:
        raise AssertionError("noncanonical zero-count order alias was accepted")


def test_spawn_factor_budget_is_eight_counts_plus_at_most_seven_order_choices():
    actor = Actor().eval()
    logits = torch.zeros(1, N_BODY_PART)
    counts = torch.ones(1, N_BODY_PART, dtype=torch.long)
    order = torch.arange(N_BODY_PART).view(1, -1)
    _, _, _, active = actor._positive_type_order(logits, counts, True, order)
    assert int(active.sum()) == N_BODY_PART - 1
    assert 2 * N_BODY_PART == 16
    assert 2 * N_BODY_PART < 1 + MAX_BODY_PARTS  # replaces old length + 50-slot scan


def test_ti_interleaved_spawn_supervises_counts_but_not_grouped_order():
    from samples.rl.agent.pretrain_joint import (
        _body_label_factor_mask,
        _parts_to_count_order,
    )

    carry, move = 2, 0
    interleaved = [carry, move] * 4
    counts, order, order_exact = _parts_to_count_order(interleaved)
    assert counts.tolist() == [4, 0, 4, 0, 0, 0, 0, 0]
    assert order[:2].tolist() == [carry, move]
    assert not order_exact
    eligible = _body_label_factor_mask(counts, order_exact)
    assert bool(eligible[6 : 6 + N_BODY_PART].all())
    assert not bool(eligible[6 + N_BODY_PART :].any())

    grouped = [carry] * 4 + [move] * 4
    grouped_counts, grouped_order, grouped_exact = _parts_to_count_order(grouped)
    assert torch.equal(grouped_counts, counts)
    assert torch.equal(grouped_order, order)
    assert grouped_exact
    grouped_eligible = _body_label_factor_mask(grouped_counts, grouped_exact)
    # Two positive types have exactly one stochastic relative-order choice.
    assert int(grouped_eligible[6 + N_BODY_PART :].sum()) == 1


def test_resource_amount_bins_are_unique_for_selected_target():
    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    obs["actors"][0, 0, _AF["storedEnergy"]] = 50.0 / MAX_ROOM_ENERGY
    obs["actors"][0, 0, _AF["storeCapacity"]] = 50.0 / MAX_ROOM_ENERGY
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
    obs["actors"][0, 0, _AF["activeMove"]] = 1.0 / MAX_BODY_PARTS
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
    obs["actors"][0, 0, _AF["isTower"]] = 1
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
        critic_logits, critic_latent = agent.critic.value_logits_and_latent(obs)
        values = agent.critic.support.to_expected_scalar(critic_logits)
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
        actor_latent=out.state_latent,
        critic_latent=critic_latent,
        critic_logits=critic_logits,
        batch_size=batch_size,
    )
    trainer = PPOTrainer(
        agent,
        device="cpu",
        compile_model=False,
        max_grad_norm=0.125,
        epochs=1,
        minibatch=2,
    )
    clip_calls: list[float] = []
    original_clip = torch.nn.utils.clip_grad_norm_
    def recording_clip(parameters, max_norm, *args, **kwargs):
        clip_calls.append(float(max_norm))
        return original_clip(parameters, max_norm, *args, **kwargs)
    torch.nn.utils.clip_grad_norm_ = recording_clip
    try:
        stats = trainer.update(rollout)
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert torch.isfinite(torch.tensor(stats[key])), (key, stats[key])
    assert stats["critic_max_grad_norm"] == 0.5
    assert stats["value_target_overflow_fraction"] == 0.0
    assert clip_calls.count(0.125) == 2  # actor: two minibatches
    assert clip_calls.count(0.5) == 2  # critic: independently fixed at 0.5


def test_factorized_ppo_update_accepts_sparse_rollout_observations():
    """Exercise page reconstruction through PPO's actual minibatch staging."""
    from samples.rl.agent.ppo import PPOTrainer, RolloutBatch
    from samples.rl.agent.rollout_buffer import HostRolloutBuffer

    torch.manual_seed(12)
    batch_size = 4
    obs = _dummy_obs_batch(batch_size)
    obs["actor_mask"][:, :2] = 1
    obs["intent_mask"][:, :2, :, INTENT_TYPES.index("none")] = 1
    obs["intent_mask"][:, :2, :, INTENT_TYPES.index("move")] = 1
    obs["amount_mask"][..., 0] = 1
    agent = Agent()
    with torch.no_grad():
        out = agent.actor(obs)
        critic_logits, critic_latent = agent.critic.value_logits_and_latent(obs)
        values = agent.critic.support.to_expected_scalar(critic_logits)

    buf = HostRolloutBuffer(batch_size, 1)
    byte_keys = {
        "room_mask", "actor_mask", "target_mask", "intent_mask",
        "dir_mask", "target_select_mask", "amount_mask",
        "construction_mask",
    }
    for index in range(batch_size):
        host = {
            key: (
                value[index : index + 1].to(torch.uint8)
                if key == "patches" or key in byte_keys
                else value[index : index + 1]
            )
            for key, value in obs.items()
        }
        buf.write_step(
            host_obs=host,
            types=out.types[index : index + 1],
            dirs=out.dirs[index : index + 1],
            targets=out.targets[index : index + 1],
            amounts=out.amounts[index : index + 1],
            logprob=out.actor_logprob[index : index + 1],
            value=values[index : index + 1],
            reward=torch.ones(1),
            done=torch.zeros(1),
            trunc=torch.zeros(1),
        )
    rollout = RolloutBatch(
        obs=buf.as_flat_obs(),
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
        agent, device="cpu", compile_model=False, epochs=1, minibatch=2,
    )
    stats = trainer.update(rollout)
    assert buf.patch_pages.count == batch_size
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        assert torch.isfinite(torch.tensor(stats[key])), (key, stats[key])


def test_nextlat_ppo_pairs_are_causal_masked_and_train_both_trunks():
    """Future rows are detached targets; terminals and rollout tails contribute zero."""
    from samples.rl.agent.ppo import PPOTrainer, RolloutBatch, _masked_latent_loss

    prediction = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[2.0, 2.0], [100.0, 100.0]], requires_grad=True)
    loss = _masked_latent_loss(prediction, target, torch.tensor([True, False]))
    assert torch.allclose(loss, torch.tensor(1.5)), loss
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert target.grad is None

    torch.manual_seed(31)
    batch_size = 4
    obs = _dummy_obs_batch(batch_size)
    obs["actor_mask"][:, :2] = 1
    obs["intent_mask"][:, :2, :, INTENT_TYPES.index("none")] = 1
    obs["intent_mask"][:, :2, :, INTENT_TYPES.index("move")] = 1
    obs["dir_mask"][:, :2] = 1
    obs["amount_mask"][..., 0] = 1
    # Make each actual next state observably different while keeping table shape fixed.
    obs["globals"][:, 0] = torch.arange(batch_size, dtype=torch.float32) / batch_size
    agent = Agent()
    with torch.no_grad():
        out = agent.actor(obs)
        critic_logits, critic_latent = agent.critic.value_logits_and_latent(obs)
        values = agent.critic.support.to_expected_scalar(critic_logits)
    actions = {
        "types": out.types,
        "dirs": out.dirs,
        "targets": out.targets,
        "amounts": out.amounts,
        "body_counts": out.body_counts,
        "body_order": out.body_order,
        "construction_types": out.construction_types,
        "construction_tiles": out.construction_tiles,
    }
    actor_before = agent.actor.latent_dynamics.dynamics_mlp[-1].weight.detach().clone()
    critic_before = agent.critic.latent_dynamics.dynamics_mlp[-1].weight.detach().clone()
    rollout = RolloutBatch(
        obs=obs,
        actions=actions,
        logprob=out.actor_logprob,
        value=values,
        reward=torch.ones(batch_size),
        done=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        advantage=torch.tensor([1.0, -0.5, 0.25, 2.0]),
        ret=torch.tensor([1.0, 0.5, 1.5, 2.0]),
        actor_latent=out.state_latent,
        critic_latent=critic_latent,
        critic_logits=critic_logits,
        next_indices=torch.tensor([1, 2, 3, 3]),
        nextlat_valid=torch.tensor([True, False, True, False]),
        batch_size=batch_size,
    )
    trainer = PPOTrainer(
        agent, device="cpu", compile_model=False, epochs=1, minibatch=4,
    )
    stats = trainer.update(rollout)
    assert stats["nextlat_valid_fraction"] == 0.5
    for key in ("nextlat_actor_mse", "nextlat_critic_mse", "nextlat_critic_kl"):
        assert torch.isfinite(torch.tensor(stats[key])), (key, stats[key])
        assert stats[key] >= 0
    assert not torch.equal(
        actor_before, agent.actor.latent_dynamics.dynamics_mlp[-1].weight,
    )
    assert not torch.equal(
        critic_before, agent.critic.latent_dynamics.dynamics_mlp[-1].weight,
    )


def test_nextlat_time_major_pairing_cuts_done_truncation_and_tail():
    from samples.rl.agent.train import _nextlat_pair_indices

    done = torch.zeros(4, 3)
    trunc = torch.zeros_like(done)
    done[1, 0] = 1
    trunc[2, 2] = 1
    next_indices, valid = _nextlat_pair_indices(done, trunc)
    expected = torch.tensor([
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [9, 10, 11],
    ])
    assert torch.equal(next_indices, expected)
    assert torch.equal(
        valid,
        torch.tensor([
            [True, True, True],
            [False, True, True],
            [True, True, False],
            [False, False, False],
        ]),
    )


def test_nextlat_action_encoding_uses_target_features_not_table_identity():
    torch.manual_seed(37)
    actor = Actor().eval()
    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    obs["target_mask"][0, :2] = 1
    obs["targets"][0, 0, 0:4] = torch.tensor([3 / 6, 0.1, 0.2, 0.0])
    obs["targets"][0, 1, 0:4] = torch.tensor([3 / 6, 0.8, 0.7, 0.0])
    pickup = INTENT_TYPES.index("pickup")
    actions = {
        "types": torch.full((1, MAX_ACTORS, 1), pickup),
        "dirs": torch.zeros(1, MAX_ACTORS, 1, dtype=torch.long),
        "targets": torch.zeros(1, MAX_ACTORS, 1, dtype=torch.long),
        "amounts": torch.zeros(1, MAX_ACTORS, 1, dtype=torch.long),
    }
    state = torch.randn(1, actor.d_model)
    with torch.no_grad():
        baseline = actor.predict_next_latent(state, obs, actions)
        permuted_obs = {key: value.clone() for key, value in obs.items()}
        permuted_obs["targets"][:, [0, 1]] = permuted_obs["targets"][:, [1, 0]]
        permuted_actions = {key: value.clone() for key, value in actions.items()}
        permuted_actions["targets"][:, 0, 0] = 1
        permuted = actor.predict_next_latent(state, permuted_obs, permuted_actions)
    assert torch.allclose(baseline, permuted, atol=1e-6, rtol=1e-6)


def _nextlat_actions(batch_size: int, actor_count: int) -> dict[str, torch.Tensor]:
    shape = (batch_size, actor_count, INTENT_SLOTS)
    return {
        "types": torch.full(shape, INTENT_TYPES.index("none"), dtype=torch.long),
        "dirs": torch.zeros(shape, dtype=torch.long),
        "targets": torch.zeros(shape, dtype=torch.long),
        "amounts": torch.zeros(shape, dtype=torch.long),
    }


def test_nextlat_none_actions_have_zero_context_and_ignore_idle_state():
    torch.manual_seed(41)
    dynamics = ActionConditionedDynamics(24).eval()
    state = torch.randn(1, 24)
    batch = {
        "actors": torch.randn(1, 4, ACTOR_FEAT),
        "actor_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        "targets": torch.randn(1, 3, TARGET_FEAT),
    }
    actions = _nextlat_actions(1, 4)

    with torch.no_grad():
        context = dynamics.action_context(state, batch, actions)
        prediction = dynamics(state, batch, actions)
    assert torch.equal(context, torch.zeros_like(context))

    changed_batch = {key: value.clone() for key, value in batch.items()}
    changed_batch["actors"].normal_(mean=50.0, std=10.0)
    changed_batch["targets"].normal_(mean=-50.0, std=10.0)
    changed_actions = {key: value.clone() for key, value in actions.items()}
    changed_actions["dirs"].fill_(N_DIR - 1)
    changed_actions["targets"].fill_(batch["targets"].shape[1] - 1)
    changed_actions["amounts"].fill_(N_AMOUNT - 1)
    permutation = torch.tensor([2, 0, 3, 1])
    changed_batch["actors"] = changed_batch["actors"][:, permutation]
    changed_batch["actor_mask"] = changed_batch["actor_mask"][:, permutation]
    changed_actions = {
        key: value[:, permutation] for key, value in changed_actions.items()
    }
    with torch.no_grad():
        changed_context = dynamics.action_context(state, changed_batch, changed_actions)
        changed_prediction = dynamics(state, changed_batch, changed_actions)
    assert torch.equal(changed_context, torch.zeros_like(changed_context))
    assert torch.equal(prediction, changed_prediction)


def test_nextlat_issued_actions_are_sensitive_and_permutation_invariant():
    torch.manual_seed(43)
    dynamics = ActionConditionedDynamics(24).eval()
    state = torch.randn(1, 24)
    batch = {
        "actors": torch.randn(1, 4, ACTOR_FEAT),
        "actor_mask": torch.ones(1, 4),
        "targets": torch.randn(1, 3, TARGET_FEAT),
    }
    actions = _nextlat_actions(1, 4)
    move = INTENT_TYPES.index("move")
    pickup = INTENT_TYPES.index("pickup")
    actions["types"][0, 0, 0] = move
    actions["dirs"][0, 0, 0] = 1
    actions["types"][0, 2, 0] = pickup
    actions["targets"][0, 2, 0] = 1

    with torch.no_grad():
        none_actions = _nextlat_actions(1, 4)
        none_context = dynamics.action_context(state, batch, none_actions)
        none_prediction = dynamics(state, batch, none_actions)
        context = dynamics.action_context(state, batch, actions)
        prediction = dynamics(state, batch, actions)
        changed_direction = {key: value.clone() for key, value in actions.items()}
        changed_direction["dirs"][0, 0, 0] = 2
        changed_context = dynamics.action_context(state, batch, changed_direction)
        changed_prediction = dynamics(state, batch, changed_direction)
    assert torch.equal(none_context, torch.zeros_like(none_context))
    assert not torch.allclose(none_prediction, prediction)
    assert not torch.allclose(context, changed_context)
    assert not torch.allclose(prediction, changed_prediction)

    idle_changed_batch = {key: value.clone() for key, value in batch.items()}
    idle_changed_batch["actors"][0, [1, 3]].normal_(mean=100.0, std=20.0)
    idle_changed_actions = {key: value.clone() for key, value in actions.items()}
    idle_changed_actions["dirs"][0, [1, 3], 0] = N_DIR - 1
    idle_changed_actions["targets"][0, [1, 3], 0] = 2
    idle_changed_actions["amounts"][0, [1, 3], 0] = N_AMOUNT - 1
    with torch.no_grad():
        idle_changed_context = dynamics.action_context(
            state, idle_changed_batch, idle_changed_actions,
        )
        idle_changed_prediction = dynamics(
            state, idle_changed_batch, idle_changed_actions,
        )
    assert torch.equal(context, idle_changed_context)
    assert torch.equal(prediction, idle_changed_prediction)

    permutation = torch.tensor([2, 3, 0, 1])
    permuted_batch = {
        "actors": batch["actors"][:, permutation],
        "actor_mask": batch["actor_mask"][:, permutation],
        "targets": batch["targets"],
    }
    permuted_actions = {
        key: value[:, permutation] for key, value in actions.items()
    }
    with torch.no_grad():
        permuted_context = dynamics.action_context(
            state, permuted_batch, permuted_actions,
        )
        permuted_prediction = dynamics(state, permuted_batch, permuted_actions)
    assert torch.allclose(context, permuted_context, atol=1e-7, rtol=1e-6)
    assert torch.allclose(prediction, permuted_prediction, atol=1e-7, rtol=1e-6)


def test_nextlat_action_context_gradients_are_finite_and_mask_none_exactly():
    torch.manual_seed(47)
    dynamics = ActionConditionedDynamics(24)
    state = torch.randn(1, 24, requires_grad=True)
    actors = torch.randn(1, 3, ACTOR_FEAT, requires_grad=True)
    batch = {
        "actors": actors,
        "actor_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "targets": torch.randn(1, 2, TARGET_FEAT, requires_grad=True),
    }
    actions = _nextlat_actions(1, 3)
    move = INTENT_TYPES.index("move")
    none = INTENT_TYPES.index("none")
    actions["types"][0, 0, 0] = move
    actions["dirs"][0, 0, 0] = 1

    prediction = dynamics(state, batch, actions)
    loss = prediction.square().mean()
    loss.backward()

    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert torch.count_nonzero(state.grad) > 0
    assert actors.grad is not None and torch.isfinite(actors.grad).all()
    assert torch.count_nonzero(actors.grad[0, 0]) > 0
    assert torch.count_nonzero(actors.grad[0, 1:]) == 0
    type_grad = dynamics.type_embed.weight.grad
    assert type_grad is not None and torch.isfinite(type_grad).all()
    assert torch.count_nonzero(type_grad[move]) > 0
    assert torch.count_nonzero(type_grad[none]) == 0
    for name, parameter in dynamics.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_ratio_diagnostics_equal_weight_team_states():
    """The KL and clip-fraction diagnostics stay per team state, not per actor."""
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
    # The compatibility hook must not freeze reset-time visibility; host-side
    # 1/2/4 buckets grow with later expansion.
    assert not hasattr(agent.actor.trunk, "_static_r")
    assert not hasattr(agent.critic.trunk, "_static_r")


def test_entity_capacity_buckets_preserve_live_policy_and_value():
    from samples.rl.agent.vec_env import promote_obs_device

    torch.manual_seed(23)
    dense = _dummy_obs_batch(2)
    dense["actor_mask"][:, :2] = 1
    dense["target_mask"][:, :9] = 1
    dense["intent_mask"][:, :2, :, INTENT_TYPES.index("none")] = 1
    dense["amount_mask"][:, :2, :, :, 0] = 1
    compact = promote_obs_device(dense, "cpu", non_blocking=False)
    assert compact["actors"].shape[1] == 8
    assert compact["targets"].shape[1] == 16

    actor = Actor().eval()
    critic = Critic().eval()
    with torch.no_grad():
        dense_out = actor(dense, deterministic=True)
        compact_out = actor(compact, deterministic=True)
        dense_value = critic(dense)
        compact_value = critic(compact)
    assert torch.equal(compact_out.types, dense_out.types[:, :8])
    assert torch.allclose(
        compact_out.actor_logprob, dense_out.actor_logprob[:, :8], atol=2e-5,
    )
    assert torch.allclose(compact_value, dense_value, atol=2e-5)


def test_rollout_zero_pads_compact_action_prefix():
    from samples.rl.agent.rollout_buffer import HostRolloutBuffer

    buf = HostRolloutBuffer(1, 1)
    host = _rollout_host_obs(1)
    compact = torch.ones(1, 8, INTENT_SLOTS, dtype=torch.uint8)
    buf.write_step(
        host_obs=host,
        types=compact,
        dirs=compact,
        targets=compact,
        amounts=compact,
        logprob=torch.ones(1, 8),
        value=torch.zeros(1),
        reward=torch.zeros(1),
        done=torch.zeros(1),
        trunc=torch.zeros(1),
    )
    assert torch.equal(buf.types[0, 0, :8], compact[0])
    assert not bool(buf.types[0, 0, 8:].any())
    assert torch.equal(buf.logprob[0, 0, :8], torch.ones(8))
    assert not bool(buf.logprob[0, 0, 8:].any())


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
        "actor_outcome": torch.zeros(B, MAX_ACTORS),
        "targets": torch.zeros(B, MAX_TARGETS, TARGET_FEAT),
        "target_mask": torch.zeros(B, MAX_TARGETS),
        "intent_mask": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, N_INTENT),
        "dir_mask": torch.ones(B, MAX_ACTORS, INTENT_SLOTS, N_DIR),
        "target_select_mask": torch.zeros(B, N_INTENT, MAX_TARGETS),
        "amount_mask": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, N_INTENT, N_AMOUNT),
        "construction_mask": torch.zeros(
            B, MAX_ROOMS, N_CONSTRUCTION_TYPE, CONSTRUCTION_MASK_BYTES,
            dtype=torch.uint8,
        ),
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


def test_discounted_returns_tn_exact_recurrence_and_boundaries():
    from samples.rl.agent.gae import discounted_returns_tn

    gamma = 0.995
    rewards = torch.tensor([[1.0], [2.0], [3.0], [100.0], [200.0]])
    dones = torch.tensor([[0.0], [0.0], [1.0], [0.0], [1.0]])
    ret = discounted_returns_tn(
        rewards, dones, gamma=gamma, next_value=torch.zeros(1),
        truncations=torch.zeros_like(dones),
    )
    expected = torch.tensor([
        1 + gamma * 2 + gamma**2 * 3,
        2 + gamma * 3,
        3,
        100 + gamma * 200,
        200,
    ])
    assert torch.allclose(ret[:, 0], expected)

    # A nonterminal finite chunk uses its explicitly supplied endpoint value.
    bootstrapped = discounted_returns_tn(
        torch.tensor([[4.0], [5.0]]), torch.zeros(2, 1), gamma=gamma,
        next_value=torch.tensor([7.0]), truncations=torch.zeros(2, 1),
    )
    assert torch.allclose(
        bootstrapped[:, 0],
        torch.tensor([4 + gamma * 5 + gamma**2 * 7, 5 + gamma * 7]),
    )


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
            actor_latent=torch.full((N, int(SCHEMA["model"]["dModel"])), 0.25),
            value=torch.ones(N),
            reward=torch.ones(N) * 0.5,
            done=torch.zeros(N),
            trunc=torch.zeros(N),
        )
    assert len(buf) == 3
    flat = buf.as_flat_obs()
    gathered = flat.gather_minibatch(torch.arange(3 * N))
    assert gathered["patches"].shape[:2] == (3 * N, 1)
    assert float(buf.tn("reward").sum()) == 3.0
    assert torch.all(buf.tn("actor_latent") == 0.25)
    buf.set_term_values(0, torch.tensor([1.0, 2.0]), torch.tensor([True, False]))
    assert float(buf.term_value[0, 0]) == 1.0
    assert bool(buf.has_term[0, 0]) and not bool(buf.has_term[0, 1])
    old_capacity = buf.t_max
    buf.ensure_capacity(old_capacity + 3)
    assert buf.t_max >= old_capacity + 3
    assert buf.patch_pages.count == 3 * N
    assert buf.obs["target_select_mask"].shape[-2:] == (N_INTENT, MAX_TARGETS)


def _rollout_host_obs(n: int) -> dict[str, torch.Tensor]:
    module = __import__("samples.rl.agent.rollout_buffer", fromlist=["_OBS_SPEC"])
    return {
        key: torch.zeros((n, *shape), dtype=dtype)
        for key, (shape, dtype) in module._OBS_SPEC.items()
    }


def _write_empty_rollout_step(buf, host) -> None:
    action = torch.zeros(buf.n, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long)
    buf.write_step(
        host_obs=host,
        types=action,
        dirs=action,
        targets=action,
        amounts=action,
        logprob=torch.zeros(buf.n, MAX_ACTORS),
        value=torch.zeros(buf.n),
        reward=torch.zeros(buf.n),
        done=torch.zeros(buf.n),
        trunc=torch.zeros(buf.n),
    )


def test_sparse_rollout_pages_preserve_expansion_reorder_and_gather():
    """Each transition reconstructs its own live slots, including new rooms."""
    from samples.rl.agent.rollout_buffer import HostRolloutBuffer

    buf = HostRolloutBuffer(3, 2)
    first = _rollout_host_obs(2)
    first["room_mask"][0, 0] = 1
    first["room_mask"][1, 2] = 1
    first["patches"][0, 0].fill_(11)
    first["patches"][1, 2].fill_(22)
    _write_empty_rollout_step(buf, first)

    expanded = _rollout_host_obs(2)
    expanded["room_mask"][0, :2] = 1
    expanded["patches"][0, 0].fill_(31)
    expanded["patches"][0, 1].fill_(32)
    # Env 1 changes which room slot is live. Old slot 2 must not leak through.
    expanded["room_mask"][1, 0] = 1
    expanded["patches"][1, 0].fill_(41)
    expanded["patches"][1, 2].fill_(99)
    _write_empty_rollout_step(buf, expanded)

    flat = buf.as_flat_obs()
    # Gather out of temporal order and repeat a transition (last-mb padding case).
    batch = flat.gather_minibatch(torch.tensor([3, 0, 2, 1, 3]))
    patches = batch["patches"]
    assert tuple(patches.shape) == (5, MAX_ROOMS, PATCHES_PER_ROOM, PATCH_FLAT)
    assert int(patches[0, 0, 0, 0]) == 41
    assert torch.count_nonzero(patches[0, 1:]) == 0
    assert int(patches[1, 0, 0, 0]) == 11
    assert int(patches[2, 0, 0, 0]) == 31
    assert int(patches[2, 1, 0, 0]) == 32
    assert int(patches[3, 2, 0, 0]) == 22
    assert torch.equal(patches[0], patches[4])
    assert torch.equal(batch["room_mask"], torch.tensor([
        [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0],
        [0, 0, 1, 0], [1, 0, 0, 0],
    ], dtype=torch.uint8))


def test_sparse_rollout_gather_uses_room_capacity_buckets():
    from samples.rl.agent.rollout_buffer import HostRolloutBuffer

    buf = HostRolloutBuffer(2, 1)
    one_room = _rollout_host_obs(1)
    one_room["room_mask"][0, 0] = 1
    _write_empty_rollout_step(buf, one_room)
    two_rooms = _rollout_host_obs(1)
    two_rooms["room_mask"][0, :2] = 1
    _write_empty_rollout_step(buf, two_rooms)
    flat = buf.as_flat_obs()

    assert flat.gather_minibatch(torch.tensor([0]))["patches"].shape[1] == 1
    assert flat.gather_minibatch(torch.tensor([1]))["patches"].shape[1] == 2
    assert flat.gather_minibatch(torch.tensor([0, 1]))["patches"].shape[1] == 2


def test_sparse_rollout_reset_reuses_pages_without_stale_rooms():
    from samples.rl.agent.rollout_buffer import HostRolloutBuffer

    buf = HostRolloutBuffer(1, 1)
    old = _rollout_host_obs(1)
    old["room_mask"][0, :2] = 1
    old["patches"][0, 0].fill_(7)
    old["patches"][0, 1].fill_(8)
    _write_empty_rollout_step(buf, old)
    allocated = buf.patch_pages.allocated_bytes

    buf.reset()
    assert len(buf) == 0 and buf.patch_pages.count == 0
    new = _rollout_host_obs(1)
    new["room_mask"][0, 3] = 1
    new["patches"][0, 3].fill_(9)
    _write_empty_rollout_step(buf, new)
    gathered = buf.as_flat_obs().gather_minibatch(torch.tensor([0]))["patches"]
    assert torch.count_nonzero(gathered[0, :3]) == 0
    assert int(gathered[0, 3, 0, 0]) == 9
    assert buf.patch_pages.allocated_bytes == allocated


def test_sparse_rollout_memory_accounting_beats_dense_one_room_history():
    from samples.rl.agent.rollout_buffer import HostRolloutBuffer

    steps, envs = 64, 2
    buf = HostRolloutBuffer(steps, envs)
    host = _rollout_host_obs(envs)
    host["room_mask"][:, 0] = 1
    for _ in range(steps):
        _write_empty_rollout_step(buf, host)
    dense = HostRolloutBuffer.dense_storage_bytes(steps, envs)
    assert buf.storage_bytes(allocated=True) < dense * 0.65
    assert buf.patch_pages.used_bytes == steps * envs * PATCHES_PER_ROOM * PATCH_FLAT


def test_spawn_replay_update_is_finite_and_strict():
    from samples.rl.agent.pretrain_joint import _train_spawn_replay

    obs = _dummy_obs_batch(1)
    spawn = INTENT_TYPES.index("spawnCreep")
    none = INTENT_TYPES.index("none")
    obs["actor_mask"][0, 0] = 1
    obs["actors"][0, 0, 0] = 1
    obs["actors"][0, 0, _AF["isSpawn"]] = 1
    obs["actors"][0, 0, _AF["roomEnergyAvailable"]] = 650.0 / MAX_ROOM_ENERGY
    obs["intent_mask"][0, 0, 0, none] = 1
    obs["intent_mask"][0, 0, 0, spawn] = 1
    actor = Actor()
    with torch.no_grad():
        actor.type_head.bias.fill_(-50)
        actor.type_head.bias[spawn] = 50
        out = actor(obs, deterministic=True)
    action = {
        key: getattr(out, key).detach().clone()
        for key in (
            "types", "dirs", "targets", "amounts",
            "body_counts", "body_order",
        )
    }
    eligible = torch.ones(
        1, obs["actors"].shape[1], 6 + 2 * N_BODY_PART, dtype=torch.bool,
    )
    replay = [("ge650:7_15:0_0_0_0_0_0_0_0", 0, obs, action, eligible)]
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    nll, legal, diagnostics = _train_spawn_replay(
        actor, optimizer, replay,
        device=torch.device("cpu"), use_bf16=False, minibatch=1,
    )
    assert nll >= 0 and torch.isfinite(torch.tensor(nll))
    assert legal == 1.0
    assert diagnostics["spawn_type_accuracy"] == 1.0
    assert all(torch.isfinite(parameter).all() for parameter in actor.parameters())


def test_spawn_replay_is_bounded_and_mixed_buckets_pad_locally():
    from samples.rl.agent.pretrain_joint import (
        SPAWN_REPLAY_PER_STRATUM, _append_spawn_replay, _pad_eligible_masks,
        _pad_replay_tensors,
    )

    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, :2] = 1
    obs["actors"][0, 0, _AF["isNonCreep"]] = 1
    obs["actors"][0, 0, _AF["isSpawn"]] = 1
    obs["target_mask"][0, :3] = 1
    spawn = INTENT_TYPES.index("spawnCreep")
    obs["intent_mask"][0, 0, 0, spawn] = 1
    obs["actors"][0, 0, _AF["roomEnergyAvailable"]] = 300.0 / MAX_ROOM_ENERGY
    action = {
        "types": torch.zeros(1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long),
        "dirs": torch.zeros(1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long),
        "targets": torch.zeros(1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long),
        "amounts": torch.zeros(1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long),
        "body_counts": torch.zeros(
            1, MAX_ACTORS, INTENT_SLOTS, N_BODY_PART, dtype=torch.long,
        ),
        "body_order": torch.arange(N_BODY_PART).view(
            1, 1, 1, N_BODY_PART,
        ).expand(1, MAX_ACTORS, INTENT_SLOTS, -1).clone(),
        "construction_types": torch.zeros(
            1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long,
        ),
        "construction_tiles": torch.zeros(
            1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long,
        ),
    }
    action["types"][0, 0, 0] = spawn
    action["body_counts"][0, 0, 0, 0] = 1
    eligible = torch.ones(
        1, MAX_ACTORS, 6 + 2 * N_BODY_PART, dtype=torch.bool,
    )
    replay = []
    for _ in range(SPAWN_REPLAY_PER_STRATUM + 9):
        _append_spawn_replay(replay, obs, action, eligible)
    assert len(replay) == SPAWN_REPLAY_PER_STRATUM
    assert replay[0][2]["actors"].shape[1] == 8
    assert replay[0][2]["targets"].shape[1] == 16
    # The mask is compacted with its row, exactly like obs and action.
    assert replay[0][4].shape == (1, 8, 6 + 2 * N_BODY_PART)

    # A body this ABI cannot express exactly teaches nothing, and a wait row
    # without its intent factor supervises nothing at all: neither is retained.
    body_masked = eligible.clone()
    body_masked[0, 0, 6 : 6 + N_BODY_PART] = False
    bounded = len(replay)
    _append_spawn_replay(replay, obs, action, body_masked)
    assert len(replay) == bounded
    wait_action = {key: value.clone() for key, value in action.items()}
    wait_action["types"][0, 0, 0] = INTENT_TYPES.index("none")
    type_masked = eligible.clone()
    type_masked[0, 0, 0] = False
    _append_spawn_replay(replay, obs, wait_action, type_masked)
    assert len(replay) == bounded
    _append_spawn_replay(replay, obs, wait_action, eligible)
    assert replay[-1][0].startswith("wait")

    larger_obs = _dummy_obs_batch(1)
    larger_obs["actor_mask"][0, :12] = 1
    larger_obs["actors"][0, 0, _AF["isNonCreep"]] = 1
    larger_obs["actors"][0, 0, _AF["isSpawn"]] = 1
    larger_obs["target_mask"][0, :20] = 1
    larger_obs["room_mask"][0, :2] = 1
    larger_action = {key: value.clone() for key, value in action.items()}
    larger_action["body_counts"][0, 0, 0, 0] = 1
    _append_spawn_replay(replay, larger_obs, larger_action, eligible)
    small_row, large_row = replay[0][2], replay[-1][2]
    padded = _pad_replay_tensors(
        [small_row, large_row], actor_cap=16, target_cap=32, room_cap=2,
    )
    padded_eligible = _pad_eligible_masks(
        [replay[0][4], replay[-1][4]], actor_cap=16,
    )
    assert padded_eligible.shape == (2, 16, 6 + 2 * N_BODY_PART)
    assert bool(padded_eligible[0, :8].all())
    assert not bool(padded_eligible[0, 8:].any())
    assert padded["actors"].shape == (2, 16, ACTOR_FEAT)
    assert padded["targets"].shape == (2, 32, TARGET_FEAT)
    assert padded["patches"].shape[1] == 2
    assert not bool(padded["actor_mask"][0, 8:].any())


def test_global_lifecycle_reservoir_and_joint_epoch_are_finite():
    from samples.rl.agent.pretrain_joint import (
        LifecycleSample,
        TEACHER_REPLAY_PER_STRATUM,
        _append_teacher_lifecycle_replay,
        _evaluate_global_lifecycle,
        _train_global_lifecycle_epoch,
    )

    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    obs["intent_mask"][0, 0, 0, INTENT_TYPES.index("none")] = 1
    action = {
        "types": torch.zeros(1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long),
        "dirs": torch.zeros(1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long),
        "targets": torch.zeros(1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long),
        "amounts": torch.zeros(1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long),
        "construction_types": torch.zeros(
            1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long,
        ),
        "construction_tiles": torch.zeros(
            1, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long,
        ),
        "body_counts": torch.zeros(
            1, MAX_ACTORS, INTENT_SLOTS, N_BODY_PART, dtype=torch.long,
        ),
        "body_order": torch.arange(N_BODY_PART).view(
            1, 1, 1, N_BODY_PART,
        ).expand(1, MAX_ACTORS, INTENT_SLOTS, -1).clone(),
    }
    import numpy as np

    rng = np.random.default_rng(7)
    eligible = torch.ones(
        1, MAX_ACTORS, 6 + 2 * N_BODY_PART, dtype=torch.bool,
    )
    reservoir: list[LifecycleSample] = []
    seen: dict[str, int] = {}
    retained: dict[str, list[int]] = {}
    for timestep in range(100):
        _append_teacher_lifecycle_replay(
            reservoir, seen, retained, rng, obs, action, eligible,
            [{"curriculum": "empty"}], timestep=timestep,
        )
    assert len(reservoir) == TEACHER_REPLAY_PER_STRATUM
    assert next(iter(seen.values())) == 100
    assert max(sample.timestep for sample in reservoir) > TEACHER_REPLAY_PER_STRATUM
    assert all(bool(sample.eligible.any()) for sample in reservoir)
    assert reservoir[0].eligible.shape == (
        1, reservoir[0].obs["actors"].shape[1], 6 + 2 * N_BODY_PART,
    )

    replay = [
        LifecycleSample(
            stratum=f"{'empty' if index < 4 else 'seed_full'}:r1:p0_3:none",
            timestep=index,
            env_index=0,
            obs={key: value.clone() for key, value in obs.items()},
            action={key: value.clone() for key, value in action.items()},
            eligible=eligible.clone(),
        )
        for index in range(8)
    ]
    train = [sample for sample in replay if sample.timestep % 4 != 3]
    holdout = [sample for sample in replay if sample.timestep % 4 == 3]

    actor, critic = Actor(), Critic()
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    returns = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    nll, legal, value_loss, updates = _train_global_lifecycle_epoch(
        actor, critic, actor_opt, critic_opt, train, returns,
        device=torch.device("cpu"), use_bf16=False, minibatch=3,
    )
    assert torch.isfinite(torch.tensor([nll, value_loss])).all()
    assert legal == 1.0 and updates == len(train)
    metrics = _evaluate_global_lifecycle(
        actor, critic, holdout, returns,
        device=torch.device("cpu"), minibatch=2,
    )
    assert metrics["legal_frac"] == 1.0
    assert metrics["stage_empty_count"] == 1.0
    assert metrics["stage_seed_full_count"] == 1.0


def test_lifecycle_eligibility_gates_factor_losses_and_tick_retention():
    """A teacher mask is authoritative: unlabelled factors and ticks stay out."""
    import numpy as np

    from samples.rl.agent.pretrain_joint import (
        LifecycleSample,
        _append_teacher_lifecycle_replay,
        _train_global_lifecycle_epoch,
    )

    obs = _dummy_obs_batch(1)
    obs["actor_mask"][0, 0] = 1
    obs["intent_mask"][0, 0, 0, INTENT_TYPES.index("move")] = 1
    reference, reference_critic = Actor(), Critic()
    with torch.no_grad():
        out = reference(obs, deterministic=True)
    action = {
        key: getattr(out, key).detach().clone()
        for key in (
            "types", "dirs", "targets", "amounts",
            "construction_types", "construction_tiles",
            "body_counts", "body_order",
        )
    }
    # The direction factor is genuinely part of this label, so only the teacher
    # mask can keep it out of the loss.
    assert bool(out.factor_active[0, 0, 1])

    factors = 6 + 2 * N_BODY_PART
    actor_count = obs["actors"].shape[1]
    type_only = torch.zeros(1, actor_count, factors, dtype=torch.bool)
    type_only[0, 0, 0] = True
    with_dir = type_only.clone()
    with_dir[0, 0, 1] = True
    returns = torch.zeros(1, 1)

    def direction_gradient(mask: torch.Tensor) -> float:
        actor, critic = Actor(), Critic()
        actor.load_state_dict(reference.state_dict())
        critic.load_state_dict(reference_critic.state_dict())
        _train_global_lifecycle_epoch(
            actor, critic,
            torch.optim.SGD(actor.parameters(), lr=0.0),
            torch.optim.SGD(critic.parameters(), lr=0.0),
            [LifecycleSample(
                stratum="empty:r1:p0_3:none", timestep=0, env_index=0,
                obs={key: value.clone() for key, value in obs.items()},
                action={key: value.clone() for key, value in action.items()},
                eligible=mask.clone(),
            )],
            returns,
            device=torch.device("cpu"), use_bf16=False, minibatch=1,
        )
        grad = actor.dir_head.weight.grad
        return 0.0 if grad is None else float(grad.abs().sum())

    assert direction_gradient(type_only) == 0.0
    assert direction_gradient(with_dir) > 0.0

    reservoir: list[LifecycleSample] = []
    seen: dict[str, int] = {}
    retained: dict[str, list[int]] = {}
    rng = np.random.default_rng(11)
    unlabelled = torch.zeros(1, actor_count, factors, dtype=torch.bool)
    for timestep in range(16):
        _append_teacher_lifecycle_replay(
            reservoir, seen, retained, rng, obs, action, unlabelled,
            [{"curriculum": "empty"}], timestep=timestep,
        )
    assert reservoir == [] and seen == {}

    _append_teacher_lifecycle_replay(
        reservoir, seen, retained, rng, obs, action, with_dir,
        [{"curriculum": "empty"}], timestep=3, env_labels=[5],
    )
    assert len(reservoir) == 1
    assert reservoir[0].env_index == 5
    assert bool(reservoir[0].eligible[0, 0, 1])


def test_rare_intent_lane_balances_actors_and_scores_semantic_factors():
    from samples.rl.agent.pretrain_joint import (
        LifecycleSample,
        _action_factor_values,
        _actor_balanced_factor_nll,
        _evaluate_rare_intent_actors,
        _rare_intent_actor_refs,
        _ti_rare_intent_samples,
        _train_rare_intent_actor_epoch,
    )

    logprob = torch.tensor(
        [[-1.0, -3.0, 0.0], [-5.0, 0.0, 0.0]], requires_grad=True,
    )
    active = torch.tensor([[True, True, False], [True, False, False]])
    balanced_nll, legal, per_actor = _actor_balanced_factor_nll(logprob, active)
    assert legal == 1.0
    assert torch.allclose(per_actor, torch.tensor([2.0, 5.0]))
    assert torch.allclose(balanced_nll, torch.tensor(3.5))
    factor_values = _action_factor_values({
        "types": torch.tensor([[[1]]]),
        "dirs": torch.tensor([[[2]]]),
        "targets": torch.tensor([[[3]]]),
        "amounts": torch.tensor([[[4]]]),
        "construction_types": torch.tensor([[[5]]]),
        "construction_tiles": torch.tensor([[[6]]]),
        "body_counts": torch.arange(7, 15).view(1, 1, 1, N_BODY_PART),
        "body_order": torch.arange(15, 23).view(1, 1, 1, N_BODY_PART),
    })
    assert factor_values[0, 0].tolist() == list(range(1, 23))

    teacher = Actor().eval()
    site_obs = _dummy_obs_batch(1)
    site_obs["actor_mask"][0, 0] = 1
    site_obs["actors"][0, 0, 0] = 1
    site_obs["actors"][0, 0, _AF["isRoom"]] = 1
    site = INTENT_TYPES.index("createConstructionSite")
    site_obs["intent_mask"][0, 0, 0, site] = 1
    tile = 17 * 50 + 23
    alternate_tile = tile + 1
    site_obs["construction_mask"][0, 0, 2, tile // 8] |= 1 << (tile % 8)
    site_obs["construction_mask"][0, 0, 2, alternate_tile // 8] |= (
        1 << (alternate_tile % 8)
    )
    site_obs["construction_mask"][0, 0, 1, tile // 8] |= 1 << (tile % 8)
    site_obs["construction_mask"][0, 0, 1, alternate_tile // 8] |= (
        1 << (alternate_tile % 8)
    )

    claim_obs = _dummy_obs_batch(1)
    claim_obs["actor_mask"][0, 0] = 1
    claim = INTENT_TYPES.index("claimController")
    claim_obs["intent_mask"][0, 0, 0, claim] = 1
    claim_obs["target_mask"][0, 0] = 1
    claim_obs["target_select_mask"][0, claim, 0] = 1

    action_fields = (
        "types", "dirs", "targets", "amounts", "construction_types",
        "construction_tiles", "body_counts", "body_order",
    )
    with torch.no_grad():
        site_out = teacher(site_obs, deterministic=True)
        claim_out = teacher(claim_obs, deterministic=True)
    site_action = {name: getattr(site_out, name).clone() for name in action_fields}
    claim_action = {name: getattr(claim_out, name).clone() for name in action_fields}
    factor_count = int(site_out.factor_logprob.shape[-1])
    site_eligible = torch.zeros(
        1, site_obs["actors"].shape[1], factor_count, dtype=torch.bool,
    )
    claim_eligible = torch.zeros_like(site_eligible)
    site_eligible[0, 0, [0, 4, 5]] = True
    claim_eligible[0, 0, [0, 2]] = True
    replay = [
        LifecycleSample(
            stratum="empty:r3:p12p:construction", timestep=0, env_index=0,
            obs=site_obs, action=site_action, eligible=site_eligible,
        ),
        *[
            LifecycleSample(
                stratum="seed_claimer:r1:p0_3:control", timestep=index,
                env_index=0, obs=claim_obs, action=claim_action,
                eligible=claim_eligible,
            )
            for index in range(1, 4)
        ],
    ]
    refs = _rare_intent_actor_refs(replay)
    assert len(refs["createConstructionSite"]) == 1
    assert len(refs["claimController"]) == 3
    # A row naming the intent while supervising none of the lane's semantic
    # factors carries no rare demonstration, so it must not be indexed.
    type_only = torch.zeros_like(site_eligible)
    type_only[0, 0, 0] = True
    assert _rare_intent_actor_refs([LifecycleSample(
        stratum="empty:r3:p12p:construction", timestep=0, env_index=0,
        obs=site_obs, action=site_action, eligible=type_only,
    )]) == {"createConstructionSite": [], "claimController": []}

    ti_samples, ti_refs = _ti_rare_intent_samples([
        {"timestep": 11, "obs": site_obs, "action": site_action, "eligible": site_eligible},
        {"timestep": 12, "obs": claim_obs, "action": claim_action, "eligible": claim_eligible},
    ])
    assert len(ti_samples) == 2
    assert ti_refs == {
        "createConstructionSite": [(0, 0)],
        "claimController": [(1, 0)],
    }

    metrics = _evaluate_rare_intent_actors(
        teacher, replay, refs, device=torch.device("cpu"), minibatch=2,
    )
    for intent, expected_count in (("createConstructionSite", 1), ("claimController", 3)):
        assert metrics[f"{intent}_count"] == expected_count
        assert metrics[f"{intent}_legal_frac"] == 1.0
        assert metrics[f"{intent}_accuracy"] == 1.0
        assert metrics[f"{intent}_factor_accuracy"] == 1.0
        assert metrics[f"{intent}_type_accuracy"] == 1.0

    mismatched_site_action = {name: value.clone() for name, value in site_action.items()}
    predicted_tile = int(site_action["construction_tiles"][0, 0, 0])
    mismatched_site_action["construction_tiles"][0, 0, 0] = (
        alternate_tile if predicted_tile == tile else tile
    )
    mismatched_replay = [LifecycleSample(
        stratum="empty:r3:p12p:construction", timestep=0, env_index=0,
        obs=site_obs, action=mismatched_site_action, eligible=site_eligible,
    )]
    mismatched_refs = _rare_intent_actor_refs(mismatched_replay)
    mismatched_metrics = _evaluate_rare_intent_actors(
        teacher, mismatched_replay, mismatched_refs,
        device=torch.device("cpu"), minibatch=1,
    )
    assert mismatched_metrics["createConstructionSite_type_accuracy"] == 1.0
    # A construction demonstration is a structure type at a position, so a
    # different legal tile is a different demonstration and must not score.
    assert mismatched_metrics["createConstructionSite_factor_accuracy"] < 1.0
    assert mismatched_metrics["createConstructionSite_accuracy"] == 0.0

    mismatched_type_action = {
        name: value.clone() for name, value in site_action.items()
    }
    predicted_type = int(site_action["construction_types"][0, 0, 0])
    mismatched_type_action["construction_types"][0, 0, 0] = (
        1 if predicted_type != 1 else 2
    )
    type_replay = [LifecycleSample(
        stratum="empty:r3:p12p:construction", timestep=0, env_index=0,
        obs=site_obs, action=mismatched_type_action, eligible=site_eligible,
    )]
    type_metrics = _evaluate_rare_intent_actors(
        teacher, type_replay, _rare_intent_actor_refs(type_replay),
        device=torch.device("cpu"), minibatch=1,
    )
    assert type_metrics["createConstructionSite_accuracy"] == 0.0

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-4)
    nll, legal, trained = _train_rare_intent_actor_epoch(
        teacher, optimizer, replay, refs,
        device=torch.device("cpu"), use_bf16=False, minibatch=2,
        shuffle_generator=torch.Generator().manual_seed(13),
    )
    assert torch.isfinite(torch.tensor(nll))
    assert legal == 1.0
    assert trained == 6  # three actors from each intent after balancing


def test_ti_critic_epoch_consumes_full_balanced_order():
    from samples.rl.agent.pretrain_joint import CriticSample, _train_critic_replay_epoch

    obs = _dummy_obs_batch(1)
    replay = [
        CriticSample(
            stratum="rare" if index == 0 else "common",
            timestep=index,
            env_index=0,
            obs={key: value.clone() for key, value in obs.items()},
        )
        for index in range(4)
    ]
    critic = Critic()

    class CountingAdam(torch.optim.Adam):
        def __init__(self, params):
            super().__init__(params, lr=1e-4)
            self.step_count = 0

        def step(self, closure=None):
            self.step_count += 1
            return super().step(closure)

    optimizer = CountingAdam(critic.parameters())
    loss, trained = _train_critic_replay_epoch(
        critic,
        optimizer,
        replay,
        torch.arange(4, dtype=torch.float32).unsqueeze(1),
        device=torch.device("cpu"),
        use_bf16=False,
        minibatch=2,
        shuffle_generator=torch.Generator().manual_seed(7),
    )
    # Largest stratum has 3 rows, so both strata contribute 3: 6 total rows.
    assert trained == 6
    assert optimizer.step_count == 3
    assert torch.isfinite(torch.tensor(loss))


def test_joint_ckpt_meta_contract():
    """Checkpoint provenance survives a real torch serialization round-trip."""
    from samples.rl.agent.artifacts import artifact_meta, validate_artifact

    actor = Actor()
    critic = Critic()
    captured = artifact_meta(
        "joint_pretrain", actor, critic, source_sha256="startup-source",
    )
    assert captured["source_sha256"] == "startup-source"
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


def test_joint_cli_requires_corpus_and_exposes_no_collection_overrides():
    from samples.rl.agent.pretrain_joint import parse_args

    try:
        parse_args([])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("joint pretraining accepted an implicit fresh collection")
    args = parse_args(["--corpus", "immutable.pt"])
    assert args.corpus == Path("immutable.pt")
    assert args.nextlat_actor_coef == 1.0
    assert args.nextlat_critic_coef == 1.0
    assert args.nextlat_critic_kl_coef == 0.1
    assert args.min_nextlat_relative_gap == 0.01
    assert args.min_nextlat_counterfactual_rows == 128
    for collection_only in (
        "num_envs", "steps", "max_episode", "curriculum", "gamma",
        "holdout_seed_offset", "ti_actor_steps", "ti_bot_dir", "node",
    ):
        assert not hasattr(args, collection_only)


def test_artifact_rejects_pre_hl_gauss_learning_abi():
    from samples.rl.agent.artifacts import artifact_meta, validate_artifact

    actor = Actor()
    critic = Critic()
    meta = artifact_meta("joint_pretrain", actor, critic)
    meta["contracts"] = dict(meta["contracts"])
    meta["contracts"]["learningAbi"] -= 1
    try:
        validate_artifact({"meta": meta}, actor, critic, kinds=("joint_pretrain",))
    except ValueError as error:
        assert "checkpoint contracts" in str(error)
    else:
        raise AssertionError("pre-HL-Gauss learning ABI was accepted")


def test_artifact_source_override_is_explicit_and_narrow():
    from samples.rl.agent.artifacts import artifact_meta, validate_artifact

    actor = Actor()
    critic = Critic()
    meta = artifact_meta("joint_pretrain", actor, critic)
    meta["source_sha256"] = "different-source"
    payload = {"meta": meta}
    try:
        validate_artifact(payload, actor, critic, kinds=("joint_pretrain",))
    except ValueError as error:
        assert "source fingerprint" in str(error)
    else:
        raise AssertionError("source mismatch was accepted without an explicit override")
    restored = validate_artifact(
        payload,
        actor,
        critic,
        kinds=("joint_pretrain",),
        allow_source_mismatch=True,
    )
    assert restored is meta
    meta["kind"] = "ppo"
    try:
        validate_artifact(
            payload,
            actor,
            critic,
            kinds=("ppo",),
            allow_source_mismatch=True,
        )
    except ValueError as error:
        assert "source fingerprint" in str(error)
    else:
        raise AssertionError("source override accepted a PPO checkpoint")
    restored = validate_artifact(
        payload,
        actor,
        critic,
        kinds=("ppo",),
        evaluation_only_source_mismatch=True,
    )
    assert restored is meta


def test_spawn_replay_coverage_uses_tensors_and_eval_seeds_are_disjoint():
    from samples.rl.agent.pretrain_joint import (
        _evaluation_seed_overlap,
        _spawn_replay_coverage,
    )

    def row(stratum: str, budget: int, counts: list[int]):
        actors = torch.zeros(1, 1, ACTOR_FEAT)
        actors[0, 0, ACTOR_FEATURE_INDEX["roomEnergyAvailable"]] = (
            budget / MAX_ROOM_ENERGY
        )
        action = {
            "types": torch.tensor([[[INTENT_TYPES.index("spawnCreep")]]]),
            "body_counts": torch.tensor(counts).view(1, 1, 1, N_BODY_PART),
        }
        eligible = torch.zeros(1, 1, 6 + 2 * N_BODY_PART, dtype=torch.bool)
        eligible[0, 0, 6:] = True
        return stratum, 0, {"actors": actors}, action, eligible

    coverage = _spawn_replay_coverage([
        row("spawn:contract:spawn_flexible_300", 300, [1, 1, 1, 0, 0, 0, 0, 0]),
        row("spawn:contract:spawn_hauler_3000", 3000, [25, 0, 25, 0, 0, 0, 0, 0]),
        row("spawn:contract:spawn_builder_650", 650, [4, 2, 4, 0, 0, 0, 0, 0]),
    ])
    assert coverage["budget_le300"] == 1
    assert coverage["budget_ge650"] == 2
    assert coverage["length_le6"] == 1
    assert coverage["length_7_15"] == 1
    assert coverage["length_ge16"] == 1

    # One bucketing convention decides the replay strata, the teacher coverage
    # totals, and the standalone contract, so its boundaries are load bearing.
    from samples.rl.agent.pretrain_joint import (
        SPAWN_LENGTH_REQUIRED_BUCKETS,
        SPAWN_LENGTH_UNREACHED_BUCKETS,
        _spawn_budget_bucket,
        _spawn_length_bucket,
    )

    assert [_spawn_budget_bucket(value) for value in (0, 300, 301, 549, 550, 649, 650)] == [
        "le300", "le300", "301_549", "301_549", "550_649", "550_649", "ge650",
    ]
    assert [_spawn_length_bucket(value) for value in (1, 6, 7, 15, 16, 50)] == [
        "le6", "le6", "7_15", "7_15", "ge16", "ge16",
    ]
    assert SPAWN_LENGTH_UNREACHED_BUCKETS == ("ge16",)
    assert SPAWN_LENGTH_REQUIRED_BUCKETS == ("le6", "7_15")

    meta = {
        "seed": 3,
        "train_env_map": [{"seed": 3 + index} for index in range(32)],
        "holdout_env_map": [{"seed": 10_003 + index} for index in range(32)],
    }
    assert _evaluation_seed_overlap(meta, offset=1, num_envs=32)
    assert _evaluation_seed_overlap(meta, offset=10_000, num_envs=32)
    assert not _evaluation_seed_overlap(meta, offset=20_000, num_envs=32)


def test_closed_loop_late_window_uses_current_not_peak_activity():
    from types import SimpleNamespace

    from samples.rl.agent.pretrain_joint import _validate_closed_loop

    class ActorStub:
        def eval(self):
            return self

        def __call__(self, _obs, deterministic=False):
            assert deterministic
            scalar = torch.zeros(1, 1, 1, dtype=torch.long)
            return SimpleNamespace(
                types=scalar, dirs=scalar, targets=scalar, amounts=scalar,
                body_counts=torch.zeros(1, 1, 1, N_BODY_PART, dtype=torch.long),
                body_order=torch.arange(N_BODY_PART).view(1, 1, 1, -1),
                construction_types=scalar, construction_tiles=scalar,
            )

    class EnvStub:
        def __init__(self):
            self.n = 1
            self.envs = [SimpleNamespace(seed=7)]
            self.timestep = 0

        def reset(self):
            self.timestep = 0
            return {}

        def step(self, _actions):
            self.timestep += 1
            late = self.timestep == 5
            info = {
                "curriculum": "seed_outpost", "harvestDelta": 0,
                "controlDelta": 0, "transferDelta": 0, "buildDelta": 0,
                "claimDelta": 0, "remoteHarvestDelta": 3 if late else 0,
                "remoteHomeDeliveryDelta": 2 if late else 0,
                "remoteRoomsStaffed": 1 if late else 0,
                "remoteProductiveCreeps": 1 if late else 0,
                # Historical peaks alone must not increment late active ticks.
                "remoteRoomsStaffedPeak": 1, "remoteProductiveCreepsPeak": 1,
                "remoteOwnedRoomsPeak": 0, "neutralOutpostRooms": 1,
                "creeps": 1, "spawnSuccess": 1 if late else 0,
                "intentResults": [], "intentByType": {},
            }
            return {}, torch.zeros(1), torch.zeros(1), [info]

    result = _validate_closed_loop(
        ActorStub(), EnvStub(), steps=5, device=torch.device("cpu"),
    )
    stage = result["closed_loop_by_curriculum"]["seed_outpost"]
    assert stage["late_transitions"] == 1.0
    assert stage["late_remote_harvest"] == 3.0
    assert stage["late_remote_home_delivery"] == 2.0
    assert stage["late_remote_staffed_ticks"] == 1.0
    assert stage["late_remote_productive_ticks"] == 1.0
    assert stage["late_spawn_success"] == 1.0


def _spawn_contract_row(stage: str, raw_parts: list[int]):
    """Build one corpus spawn-contract row the way the TI collector labels it."""
    from samples.rl.agent.pretrain_joint import (
        _body_label_factor_mask, _parts_to_count_order,
    )

    counts, order, order_exact = _parts_to_count_order(raw_parts)
    action = {
        "types": torch.full(
            (1, 1, 1), INTENT_TYPES.index("spawnCreep"), dtype=torch.long,
        ),
        "body_counts": counts.view(1, 1, 1, N_BODY_PART),
        "body_order": order.view(1, 1, 1, N_BODY_PART),
    }
    eligible = torch.zeros(1, 1, 6 + 2 * N_BODY_PART, dtype=torch.bool)
    eligible[0, 0, 0] = True
    eligible[0, 0] |= _body_label_factor_mask(counts, order_exact)
    obs = {"actors": torch.zeros(1, 1, ACTOR_FEAT)}
    return f"spawn:contract:{stage}", 0, obs, action, eligible


def test_joint_qualification_requires_critic_expansion_and_scale():
    from types import SimpleNamespace

    from samples.rl.agent.pretrain_joint import (
        SPAWN_CURRICULA,
        _qualification_failures,
        _teacher_contract_bodies,
        _teacher_spawn_coverage,
    )

    args = SimpleNamespace(
        curriculum="empty,seed_creep,seed_full,seed_claimer",
        min_teacher_delivery=1.0,
        min_teacher_build=1.0,
        min_teacher_claims=1,
        min_teacher_creeps=8,
        min_validation_ev=0.0,
        min_closed_loop_rate=0.1,
        min_closed_loop_creeps=4,
        min_closed_loop_claims=1,
        min_outpost_closed_loop_success_rate=0.5,
        min_nextlat_relative_gap=0.01,
        min_nextlat_counterfactual_rows=128,
        max_aux_lifecycle_nll_ratio=1.1,
        max_spawn_validation_nll=1.5,
        min_spawn_replay_accuracy=0.8,
        max_rare_intent_nll=1.0,
        min_rare_intent_accuracy=0.8,
    )
    validation = {
        "validation_factor_nll": 0.5,
        "validation_legal_frac": 1.0,
        "validation_value_loss": 1.0,
        "validation_value_ev": 0.2,
        "lifecycle_holdout_value_loss": 1.0,
        "lifecycle_holdout_value_ev": 0.2,
        "nextlat_holdout_actor_mse": 0.4,
        "nextlat_holdout_actor_identity_mse": 0.5,
        "nextlat_holdout_actor_counterfactual_action_mse": 0.6,
        "nextlat_holdout_actor_counterfactual_reference_mse": 0.4,
        "nextlat_holdout_critic_mse": 0.4,
        "nextlat_holdout_critic_identity_mse": 0.5,
        "nextlat_holdout_critic_counterfactual_action_mse": 0.6,
        "nextlat_holdout_critic_counterfactual_reference_mse": 0.4,
        "nextlat_holdout_counterfactual_rows": 256.0,
        "lifecycle_after_joint_train_nll": 0.5,
        "lifecycle_final_train_nll": 0.5,
        # Every joint epoch persists fused-lane geometry; the capacity gate
        # rejects an artifact that lacks it.
        "nextlat_train_optimizer_steps": 100.0,
        "nextlat_train_rare_batches": 4.0,
        "nextlat_train_spawn_batches": 10.0,
        "nextlat_train_auxiliary_batches": 14.0,
        "nextlat_train_auxiliary_minibatch": 32.0,
        "nextlat_train_rare_exposures": 100.0,
        "nextlat_train_spawn_exposures": 320.0,
    }
    for stage in args.curriculum.split(","):
        validation[f"lifecycle_holdout_stage_{stage}_ev"] = 0.2
    for stage in SPAWN_CURRICULA:
        validation[f"validation_{stage}_labels"] = 1.0
        validation[f"validation_{stage}_nll"] = 0.5
        validation[f"validation_{stage}_success"] = 1.0
    for split in ("train", "holdout"):
        for intent in ("createConstructionSite", "claimController"):
            validation[f"rare_{split}_{intent}_count"] = 4.0
            validation[f"rare_{split}_{intent}_nll"] = 0.2
            validation[f"rare_{split}_{intent}_accuracy"] = 1.0
            validation[f"rare_{split}_{intent}_legal_frac"] = 1.0
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
                "control": 10.0,
                "claims": 0.0,
                "max_creeps": 8.0,
            },
            "seed_outpost": {
                "transitions": 1000.0, "skill_rate": 0.2,
                "invalid_frac": 0.0, "claims": 0.0,
                "remote_harvest": 40.0, "remote_home_delivery": 40.0,
                "remote_staffed_peak": 1.0, "remote_productive_peak": 1.0,
                "remote_owned_peak": 0.0,
                "late_transitions": 200.0, "late_remote_harvest": 10.0,
                "late_remote_home_delivery": 10.0,
                "late_remote_staffed_ticks": 100.0,
                "late_remote_productive_ticks": 100.0,
            },
            "seed_claimer": {
                "transitions": 1000.0, "skill_rate": 0.2,
                "invalid_frac": 0.0, "claims": 1.0,
                "remote_owned_peak": 1.0,
            },
        },
        "closed_loop_by_env": {
            "0": {
                "curriculum": "empty", "delivery": 25.0, "build": 5.0,
                "claims": 0.0, "max_creeps": 8.0, "invalid_frac": 0.0,
            },
            "4": {
                "curriculum": "empty", "delivery": 25.0, "build": 5.0,
                "claims": 0.0, "max_creeps": 8.0, "invalid_frac": 0.0,
            },
            "8": {
                "curriculum": "seed_outpost", "claims": 0.0,
                "remote_harvest": 20.0, "remote_home_delivery": 20.0,
                "remote_staffed_peak": 1.0, "remote_productive_peak": 1.0,
                "remote_owned_peak": 0.0, "invalid_frac": 0.0,
                "late_transitions": 100.0, "late_remote_harvest": 5.0,
                "late_remote_home_delivery": 5.0,
                "late_remote_staffed_ticks": 50.0,
                "late_remote_productive_ticks": 50.0,
            },
            "13": {
                "curriculum": "seed_outpost", "claims": 0.0,
                "remote_harvest": 20.0, "remote_home_delivery": 20.0,
                "remote_staffed_peak": 1.0, "remote_productive_peak": 1.0,
                "remote_owned_peak": 0.0, "invalid_frac": 0.0,
                "late_transitions": 100.0, "late_remote_harvest": 5.0,
                "late_remote_home_delivery": 5.0,
                "late_remote_staffed_ticks": 50.0,
                "late_remote_productive_ticks": 50.0,
            },
        },
    }
    # Synthetic *corpus* spawn-contract rows. The reference body for a world is
    # whatever the teacher was labelled doing in it, so the gate's expectation is
    # derived from these rows rather than from a table of scripted archetypes.
    # `spawn_hauler_3000` interleaves its part types the way The International
    # does, which leaves its order factors unsupervised, so only its composition
    # is comparable.
    spawn_replay_rows = [
        _spawn_contract_row("spawn_flexible_300", [1, 1, 2, 0]),
        _spawn_contract_row("spawn_miner_450", [1, 1, 1, 2, 0]),
        _spawn_contract_row("spawn_hauler_3000", [1, 2, 1, 0, 1, 2, 1, 0, 1, 0]),
        _spawn_contract_row("spawn_builder_650", [1, 1, 1, 1, 1, 2, 0]),
        _spawn_contract_row("spawn_upgrader_550", [1, 1, 1, 1, 1, 0]),
        _spawn_contract_row("spawn_claimer_650", [1, 1, 1, 1, 1, 2, 0]),
    ]
    teacher_spawn_bodies = _teacher_contract_bodies(spawn_replay_rows)
    assert not teacher_spawn_bodies["spawn_hauler_3000"]["order_supervised"]
    assert teacher_spawn_bodies["spawn_flexible_300"]["order_supervised"]
    for stage in SPAWN_CURRICULA:
        spawn_parts = list(teacher_spawn_bodies[stage]["parts"])
        closed_loop["closed_loop_by_curriculum"][stage] = {
            "spawn_success": 1.0,
            "spawn_body_length": float(len(spawn_parts)),
            "spawn_body_parts": spawn_parts,
            "spawn_body_parts_all": [spawn_parts],
        }
    teacher_by_curriculum = {
        "empty": {
            "transitions": 6000.0,
            "skill": 100.0,
            "delivery": 50.0,
            "build": 5.0,
            "claims": 1.0,
            "max_creeps": 8.0,
            "spawn_labels": 12.0,
            "spawn_budget_le300": 1.0,
            "spawn_budget_301_549": 1.0,
            "spawn_budget_550_649": 1.0,
            "spawn_budget_ge650": 1.0,
            "spawn_length_le6": 1.0,
            "spawn_length_7_15": 1.0,
            # This economy cannot fund a 16-part body; the bucket is recorded as
            # a coverage gap, never required.
            "spawn_length_ge16": 0.0,
        },
        "seed_outpost": {
            "transitions": 6000.0,
            "remote_harvest": 40.0,
            "remote_home_delivery": 40.0,
            "remote_staffed_peak": 1.0,
            "remote_productive_peak": 1.0,
            "remote_owned_peak": 0.0,
            "neutral_outposts": 1.0,
            "late_transitions": 1200.0,
            "late_remote_harvest": 20.0,
            "late_remote_home_delivery": 20.0,
            "late_remote_staffed_ticks": 600.0,
            "late_remote_productive_ticks": 600.0,
            "claims": 0.0,
        },
    }
    common = dict(
        last_nll=0.5,
        last_vloss=1.0,
        corpus_hist={
            "_bc_legal_frac": 1.0,
            "_spawn_replay_legal_frac": 1.0,
            "_spawn_replay_size": 12.0,
            "_spawn_replay_nll": 0.5,
            "_spawn_replay_wait_legal_size": 3.0,
            "_spawn_replay_wait_legal_strata": 3.0,
            "_spawn_replay_wait_legal_type_nll": 0.2,
            "_spawn_replay_wait_legal_type_accuracy": 1.0,
            "_spawn_replay_spawn_type_accuracy": 1.0,
            "_spawn_replay_spawn_body_accuracy": 1.0,
            "_spawn_replay_spawn_min_stratum_body_accuracy": 1.0,
            "_spawn_replay_budget_le300": 1.0,
            "_spawn_replay_budget_301_549": 1.0,
            "_spawn_replay_budget_550_649": 1.0,
            "_spawn_replay_budget_ge650": 1.0,
            "_spawn_replay_length_le6": 1.0,
            "_spawn_replay_length_7_15": 1.0,
            # Unreachable at this economy's RCL: recorded, never required.
            "_spawn_replay_length_ge16": 0.0,
            "_lifecycle_replay_size": 5.0,
            "_lifecycle_holdout_size": 2.0,
            "_lifecycle_replay_nll": 0.5,
            "_lifecycle_replay_legal_frac": 1.0,
            "_lifecycle_holdout_legal_frac": 1.0,
            "_lifecycle_replay_spawn": 1.0,
            "_lifecycle_replay_harvest": 1.0,
            "_lifecycle_replay_logistics": 1.0,
            "_lifecycle_replay_construction": 1.0,
            "_lifecycle_replay_control": 1.0,
        },
        skill=0.2,
        total_delivered=50.0,
        total_built=5.0,
        total_claims=1,
        max_creeps=8,
        teacher_by_curriculum=teacher_by_curriculum,
        teacher_spawn_bodies=teacher_spawn_bodies,
        validation=validation,
        closed_loop=closed_loop,
    )
    assert _qualification_failures(args, **common) == []

    teacher_by_curriculum["seed_outpost"]["remote_home_delivery"] = 0.0
    assert "outpost_teacher_home_delivery" in _qualification_failures(args, **common)
    teacher_by_curriculum["seed_outpost"]["remote_home_delivery"] = 40.0
    teacher_by_curriculum["seed_outpost"]["late_remote_harvest"] = 0.0
    assert "outpost_teacher_late_activity" in _qualification_failures(args, **common)
    teacher_by_curriculum["seed_outpost"]["late_remote_harvest"] = 20.0

    validation["nextlat_holdout_actor_mse"] = 0.5
    assert "nextlat_actor_beats_identity" in _qualification_failures(args, **common)
    validation["nextlat_holdout_actor_mse"] = 0.4999
    assert "nextlat_actor_beats_identity" in _qualification_failures(args, **common)
    validation["nextlat_holdout_actor_mse"] = 0.4
    validation["nextlat_holdout_critic_counterfactual_action_mse"] = 0.4
    assert "nextlat_critic_uses_action" in _qualification_failures(args, **common)
    validation["nextlat_holdout_critic_counterfactual_action_mse"] = 0.6

    validation["lifecycle_final_train_nll"] = 0.551
    assert "auxiliary_lifecycle_retention" in _qualification_failures(args, **common)
    validation["lifecycle_final_train_nll"] = 0.5

    validation["rare_holdout_createConstructionSite_accuracy"] = 0.75
    assert "rare_intent_createConstructionSite" in _qualification_failures(args, **common)
    validation["rare_holdout_createConstructionSite_accuracy"] = 1.0
    validation["rare_train_claimController_nll"] = 1.01
    assert "rare_intent_claimController" in _qualification_failures(args, **common)
    validation["rare_train_claimController_nll"] = 0.2

    args.ti_actor_steps = 20_000
    common["corpus_hist"].update({
        "_ti_actor_replay_rows": 8_192.0,
        "_ti_actor_nll": 0.2,
        "_ti_actor_legal_coverage": 1.0,
    })
    validation["ti_critic_holdout_value_ev"] = 0.2

    closed_loop["closed_loop_by_env"]["13"]["remote_home_delivery"] = 0.0
    assert _qualification_failures(args, **common) == []  # exactly 50% succeeds
    closed_loop["closed_loop_by_env"]["8"]["remote_home_delivery"] = 0.0
    assert "outpost_closed_loop_success_rate" in _qualification_failures(args, **common)
    closed_loop["closed_loop_by_env"]["8"]["remote_home_delivery"] = 20.0
    closed_loop["closed_loop_by_env"]["13"]["remote_home_delivery"] = 20.0
    assert _qualification_failures(args, **common) == []
    closed_loop["closed_loop_by_env"]["0"]["invalid_frac"] = 0.005
    assert _qualification_failures(args, **common) == []
    closed_loop["closed_loop_by_env"]["0"]["invalid_frac"] = 0.0
    validation["ti_critic_holdout_value_ev"] = -0.1
    # TI is an initialization/representation source from a different behavior
    # policy. Its final-policy EV is diagnostic, not a promotion target.
    assert _qualification_failures(args, **common) == []
    validation["ti_critic_holdout_value_ev"] = 0.2

    flexible = closed_loop["closed_loop_by_curriculum"]["spawn_flexible_300"]
    teacher_flexible = list(teacher_spawn_bodies["spawn_flexible_300"]["parts"])
    flexible["spawn_body_parts"] = [3, 3, 3, 7]  # same length, useless bootstrap body
    assert "closed_loop_spawn_scenarios" in _qualification_failures(args, **common)
    # A permutation is a different body wherever the teacher's order was exact.
    flexible["spawn_body_parts"] = list(reversed(teacher_flexible))
    assert "closed_loop_spawn_scenarios" in _qualification_failures(args, **common)
    flexible["spawn_body_parts"] = teacher_flexible
    assert _qualification_failures(args, **common) == []

    # The reference is the corpus row, not a constant: relabel the teacher and the
    # very same policy body stops qualifying.
    relabelled = dict(common)
    relabelled["teacher_spawn_bodies"] = _teacher_contract_bodies([
        _spawn_contract_row("spawn_flexible_300", [1, 1, 1, 0]),
        *spawn_replay_rows[1:],
    ])
    assert "closed_loop_spawn_scenarios" in _qualification_failures(args, **relabelled)

    # An interleaved teacher body supervises composition only, so the policy's
    # grouped order cannot be demanded there.
    hauler = closed_loop["closed_loop_by_curriculum"]["spawn_hauler_3000"]
    teacher_hauler = list(teacher_spawn_bodies["spawn_hauler_3000"]["parts"])
    hauler["spawn_body_parts"] = list(reversed(teacher_hauler))
    assert _qualification_failures(args, **common) == []
    hauler["spawn_body_parts"] = teacher_hauler[:-1]
    assert "closed_loop_spawn_scenarios" in _qualification_failures(args, **common)
    hauler["spawn_body_parts"] = teacher_hauler

    common["corpus_hist"]["_spawn_replay_budget_ge650"] = 0.0
    assert "spawn_replay_semantics" in _qualification_failures(args, **common)
    common["corpus_hist"]["_spawn_replay_budget_ge650"] = 1.0

    empty_closed = closed_loop["closed_loop_by_curriculum"]["empty"]
    empty_closed["control"] = 0.0
    assert "empty_closed_loop_control" in _qualification_failures(args, **common)
    empty_closed["control"] = 10.0
    empty_closed["max_creeps"] = args.min_closed_loop_creeps - 1
    assert "empty_closed_loop_population" in _qualification_failures(args, **common)
    empty_closed["max_creeps"] = 8.0

    teacher_by_curriculum["empty"]["spawn_length_7_15"] = 0.0
    assert "teacher_spawn_body_coverage" in _qualification_failures(args, **common)
    teacher_by_curriculum["empty"]["spawn_length_7_15"] = 1.0
    # The unreached bucket is recorded, not required.
    assert _qualification_failures(args, **common) == []
    coverage = _teacher_spawn_coverage(teacher_by_curriculum)
    assert coverage["teacher_spawn_length_unreached_ge16"] == 0.0
    assert coverage["teacher_spawn_length_le6"] > 0

    validation["lifecycle_holdout_value_ev"] = -0.1
    failures = _qualification_failures(args, **common)
    assert "lifecycle_holdout_critic_ev" in failures

    # Aggregate success from seeded stages must not certify a model that cannot
    # bootstrap its home economy.
    validation["lifecycle_holdout_value_ev"] = 0.2
    validation["lifecycle_holdout_stage_seed_full_ev"] = -0.1
    assert "lifecycle_holdout_stage_ev" in _qualification_failures(args, **common)
    validation["lifecycle_holdout_stage_seed_full_ev"] = 0.2
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
    }
    # Teacher engine legality is no longer gated: `stepExpert` reports no intent
    # summary, so a degraded teacher shows up in its economic metrics instead.
    failures = _qualification_failures(args, **common)
    assert "empty_closed_loop_delivery" in failures
    assert "empty_closed_loop_build" in failures
    assert "empty_closed_loop_control" in failures
    assert "empty_closed_loop_population" in failures

    closed_loop["closed_loop_by_curriculum"]["seed_outpost"][
        "remote_home_delivery"
    ] = 0.0
    failures = _qualification_failures(args, **common)
    assert "outpost_closed_loop_home_delivery" in failures
    closed_loop["closed_loop_by_curriculum"]["seed_outpost"][
        "remote_home_delivery"
    ] = 40.0
    closed_loop["closed_loop_by_curriculum"]["seed_outpost"][
        "late_remote_productive_ticks"
    ] = 0.0
    assert "outpost_closed_loop_late_activity" in _qualification_failures(args, **common)
    closed_loop["closed_loop_by_curriculum"]["seed_outpost"][
        "late_remote_productive_ticks"
    ] = 100.0
    closed_loop["closed_loop_by_curriculum"]["seed_claimer"]["claims"] = 0.0
    assert "seed_claimer_closed_loop_claim" in _qualification_failures(args, **common)


def test_spawn_body_expectations_are_measured_never_frozen():
    """No module may carry a table of expected spawn bodies.

    The International ignores a world's archetype tag and sizes its bootstrap
    body to the budget it sees, so every spawn-body expectation has exactly two
    admissible sources: a corpus row it was labelled into, or a live measurement
    of the teacher in that same world.  A module-level curriculum-to-body table
    would silently reintroduce the scripted planner's archetypes.
    """
    from samples.rl.agent import eval_spawn_contract
    from samples.rl.agent import pretrain_joint
    from samples.rl.agent.pretrain_joint import (
        SPAWN_CURRICULA, _spawn_body_matches, _teacher_contract_bodies,
    )

    for module in (pretrain_joint, eval_spawn_contract):
        assert not hasattr(module, "EXPECTED_BODIES"), module.__name__
        frozen = {
            name: value
            for name, value in vars(module).items()
            if isinstance(value, dict)
            and any(stage in value for stage in SPAWN_CURRICULA)
        }
        assert not frozen, f"{module.__name__} froze spawn bodies: {sorted(frozen)}"

    # The gate's reference tracks the corpus row it reads, which a frozen table
    # by construction cannot do.
    body = [1, 1, 2, 0]
    measured = _teacher_contract_bodies([
        _spawn_contract_row("spawn_flexible_300", body),
    ])["spawn_flexible_300"]
    assert _spawn_body_matches(measured["parts"], measured)
    relabelled = _teacher_contract_bodies([
        _spawn_contract_row("spawn_flexible_300", [*body, 2]),
    ])["spawn_flexible_300"]
    assert not _spawn_body_matches(measured["parts"], relabelled)
    assert _spawn_body_matches(relabelled["parts"], relabelled)


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
        "actor_outcome",
        "target_mask",
        "intent_mask",
        "dir_mask",
        "target_select_mask",
        "amount_mask",
        "construction_mask",
    )
    import re

    encode_source = (_RL / "env" / "encode.mjs").read_text(encoding="utf-8")
    encode_match = re.search(
        r"const tensorParts = \[(.*?)\];", encode_source, flags=re.DOTALL,
    )
    assert encode_match is not None
    encode_names = tuple(re.findall(
        r"\b(?:patchPacked|actors|targets|roomCoords|rm|am|actorRole|actorOutcome|tm|im|dm|tsm|amm|constructionMask)\b",
        encode_match.group(1),
    ))
    assert encode_names == (
        "patchPacked", "actors", "targets", "roomCoords", "rm", "am",
        "actorOutcome", "tm", "im", "dm", "tsm", "amm",
        "constructionMask",
    )

    server_source = (_RL / "env" / "server.mjs").read_text(encoding="utf-8")
    server_match = re.search(r"parts = \[(.*?)\];", server_source, flags=re.DOTALL)
    assert server_match is not None
    server_names = tuple(re.findall(r"raw\.(\w+)", server_match.group(1)))
    assert server_names == (
        "patches", "actors", "targets", "roomCoords", "roomMask", "actorMask",
        "actorOutcome", "targetMask", "intentMask", "dirMask",
        "targetSelectMask", "amountMask", "constructionMask",
    )

    client_source = (_RL / "agent" / "env_client.py").read_text(encoding="utf-8")
    client_match = re.search(
        r"patches = take_u8\(.*?construction_mask = take_u8\(.*?\n",
        client_source,
        flags=re.DOTALL,
    )
    assert client_match is not None
    client_names = tuple(re.findall(
        r"^\s*(patches|actors|targets|room_coords|room_mask|actor_mask|actor_outcome|target_mask|"
        r"intent_mask|dir_mask|target_select|amount_mask|construction_mask)\s*=\s*take_",
        client_match.group(0),
        flags=re.MULTILINE,
    ))
    assert client_names == (
        "patches", "actors", "targets", "room_coords", "room_mask", "actor_mask",
        "actor_outcome", "target_mask", "intent_mask", "dir_mask",
        "target_select", "amount_mask", "construction_mask",
    )
    assert tuple("target_select_mask" if name == "target_select" else name for name in client_names) == order


def test_server_neutrality_avoids_fragile_reservation_username_getter():
    """Long synthetic reservations may outlive their runtime user registry row."""
    server_source = (_RL / "env" / "server.mjs").read_text(encoding="utf-8")
    assert "function controllerHasActiveReservation(Game, controller)" in server_source
    assert "function isNeutralController(Game, controller)" in server_source
    assert "controller['#reservationEndTime']" in server_source
    # A caught presentation lookup may populate optional username telemetry, but
    # neutrality/readiness must never invoke that getter in a boolean condition.
    assert "!room.controller.reservation" not in server_source
    assert "!room.controller.owner" not in server_source


def test_segment_lengths_are_per_transition_not_a_per_env_mean():
    """VAPO's `l` is the length of the segment the transition itself sits in."""
    T, N = 8, 3
    dones = torch.zeros(T, N)
    trunc = torch.zeros(T, N)
    trunc[3, 0] = 1  # env 0: two segments of four
    dones[7, 1] = 1  # env 1: a cut on the last step ends the only segment
    trunc[1, 2] = trunc[5, 2] = 1  # env 2: segments of two, four and two
    lengths = segment_lengths_tn(dones, trunc)
    assert lengths.shape == (T, N)
    assert lengths[:, 0].tolist() == [4.0] * 8
    assert lengths[:, 1].tolist() == [8.0] * 8
    assert lengths[:, 2].tolist() == [2, 2, 4, 4, 4, 4, 2, 2]
    # Without a cut every transition sits in one segment of the full rollout.
    assert bool((segment_lengths_tn(torch.zeros(T, N)) == float(T)).all())
    # A cut on step 0 opens a new segment from step 1.
    head = torch.zeros(4, 1)
    head[0, 0] = 1
    assert segment_lengths_tn(head)[:, 0].tolist() == [1.0, 3.0, 3.0, 3.0]
    # The transition-weighted mean is what the estimator effectively applies; a
    # segment-count mean would report 8/3 for env 2 instead of 3.
    assert torch.allclose(
        lengths.mean(dim=0), torch.tensor([4.0, 8.0, 3.0]),
    ), lengths.mean(dim=0)


def test_length_adaptive_lambda_matches_vapo_equation_five_and_clamps():
    """λ_policy = 1 - 1/(alpha*l), clamped into [0, 1)."""
    lengths = torch.tensor([4096.0, 100.0, 2.0, 1.0, 0.0])
    lam = length_adaptive_lambda(lengths, 0.5)
    assert float(lam[0]) == 1.0 - 1.0 / 2048.0
    assert abs(float(lam[1]) - (1.0 - 1.0 / 50.0)) < 1e-6
    assert float(lam[2]) == 0.0  # alpha*l == 1 → one-step TD
    assert float(lam[3]) == 0.0  # alpha*l < 1 would be negative
    assert float(lam[4]) == 0.0  # a zero-length segment cannot bootstrap
    # The upper clamp stays strictly below one for any length.
    huge = float(length_adaptive_lambda(torch.tensor([1e30]), 0.5)[0])
    assert 0.0 < huge < 1.0, huge
    for bad in (0.0, -1.0, float("nan")):
        try:
            length_adaptive_lambda(lengths, bad)
        except ValueError:
            pass
        else:  # pragma: no cover - contract violation
            raise AssertionError(f"alpha={bad} must be rejected")
    # A NaN length passes a `< 0` guard and clamp propagates it into every
    # advantage, so it has to be rejected outright.
    for bad_lengths in (
        torch.tensor([4.0, float("nan")]),
        torch.tensor([4.0, float("inf")]),
        torch.tensor([4.0, -1.0]),
    ):
        try:
            length_adaptive_lambda(bad_lengths, 0.5)
        except ValueError:
            pass
        else:  # pragma: no cover - contract violation
            raise AssertionError(f"lengths={bad_lengths.tolist()} must be rejected")
    # The credit window the configured schema actually buys at a full rollout.
    # The horizon is the largest one host RAM allows at four live rooms, so this
    # is the real window, not the one a 4096-step rollout would have bought.
    gamma = float(SCHEMA["ppo"]["gamma"])
    alpha = float(SCHEMA["ppo"]["gaeLambdaPolicyAlpha"])
    steps = int(SCHEMA["ppo"]["rolloutSteps"])
    configured = float(length_adaptive_lambda(torch.tensor([float(steps)]), alpha)[0])
    horizon = 1.0 / (1.0 - gamma * configured)
    assert 600.0 < horizon < 800.0, horizon


def test_decoupled_gae_consumes_the_per_transition_length_adaptive_lambda():
    """The exact per-segment λ tensor reaches the policy pass unchanged."""
    T, N = 6, 2
    rewards = torch.ones(T, N)
    values = torch.zeros(T, N)
    dones = torch.zeros(T, N)
    trunc = torch.zeros(T, N)
    trunc[2, 1] = 1  # env 1 restarted mid-rollout: shorter segments, smaller λ
    lengths = segment_lengths_tn(dones, trunc)
    assert lengths[:, 0].tolist() == [6.0] * 6
    assert lengths[:, 1].tolist() == [3.0] * 6
    lam = length_adaptive_lambda(lengths, 0.5)
    assert lam.shape == (T, N)
    lam0, lam1 = float(lam[0, 0]), float(lam[0, 1])
    assert abs(lam0 - (1.0 - 1.0 / 3.0)) < 1e-6
    assert abs(lam1 - (1.0 - 2.0 / 3.0)) < 1e-6
    gamma = 0.99
    adv, ret, info = decoupled_gae(
        rewards, values, dones, gamma=gamma, lambda_policy=lam,
        next_value=torch.zeros(N), truncations=trunc,
        next_values_tn=torch.zeros(T, N),
    )
    assert info["gae_lambda_policy_min"] == float(lam.min())
    assert info["gae_lambda_policy_max"] == float(lam.max())
    # Zero values make every TD error exactly the reward, so the λ=1 target is
    # the discounted reward-to-go of the environment's own segment.
    assert abs(float(ret[0, 0]) - sum(gamma ** k for k in range(6))) < 1e-5
    assert abs(float(ret[0, 1]) - sum(gamma ** k for k in range(3))) < 1e-5
    # The policy pass weights the same TD errors by each segment's own λ.
    assert abs(float(adv[0, 0]) - sum((gamma * lam0) ** k for k in range(6))) < 1e-5
    assert abs(float(adv[0, 1]) - sum((gamma * lam1) ** k for k in range(3))) < 1e-5
    assert float(adv[0, 0]) > float(adv[0, 1]) > 0.0
    # Reported horizons average the per-transition windows, not the mean λ.
    expected = float((1.0 / (1.0 - gamma * lam)).mean())
    assert abs(info["policy_effective_horizon"] - expected) < 1e-4
    assert info["critic_effective_horizon"] == 1.0 / (1.0 - gamma)
    # A λ outside [0, 1) diverges the chain and must be refused.
    for bad in (1.0, 1.5, -0.1, float("nan")):
        try:
            decoupled_gae(
                rewards, values, dones, gamma=gamma, lambda_policy=bad,
                next_value=torch.zeros(N), truncations=trunc,
            )
        except ValueError:
            pass
        else:  # pragma: no cover - contract violation
            raise AssertionError(f"lambda_policy={bad} must be rejected")


def test_actor_level_reduction_weights_every_decision_once():
    """DAPO/VAPO eq. 7 against the eq. 6 team mean it replaces."""
    from samples.rl.agent.ppo import _actor_level_mean, _mean_actor_then_transition

    # A one-actor state beside a three-actor state: four decisions in total.
    values = torch.tensor([[0.5, 0.0, 0.0], [1.0, 2.0, 3.0]], requires_grad=True)
    live = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    actor_level = _actor_level_mean(values, live)
    team_mean = _mean_actor_then_transition(values, live)
    assert torch.allclose(actor_level, torch.tensor((0.5 + 1.0 + 2.0 + 3.0) / 4.0))
    assert torch.allclose(team_mean, torch.tensor((0.5 + 6.0 / 3.0) / 2.0))
    assert not torch.allclose(actor_level, team_mean)
    actor_grad = torch.autograd.grad(actor_level, values, retain_graph=True)[0]
    team_grad = torch.autograd.grad(team_mean, values)[0]
    assert not torch.allclose(actor_grad, team_grad)
    # Uniform per decision, and nothing leaks into the masked slots.
    assert torch.allclose(actor_grad[live > 0], torch.full((4,), 0.25))
    assert float(actor_grad[0, 1]) == 0.0
    # The team mean gave the lone actor 0.5 and each of the three 1/6.
    assert abs(float(team_grad[0, 0]) - 0.5) < 1e-7
    assert abs(float(team_grad[1, 0]) - 1.0 / 6.0) < 1e-7


def test_self_imitation_quantile_selects_top_return_environments():
    """The positive set is the rollout's own upper slice of segment returns."""
    T, N = 4, 5
    rewards = torch.zeros(T, N)
    rewards[:, 0] = 1.0  # segment return 4
    rewards[:, 1] = 0.0  # 0
    rewards[:, 2] = 2.0  # 8
    rewards[:, 3] = -1.0  # -4
    rewards[:, 4] = 0.5  # 2
    dones = torch.zeros(T, N)
    assert torch.allclose(
        segment_returns_per_env(rewards, dones),
        torch.tensor([4.0, 0.0, 8.0, -4.0, 2.0]),
    )
    top = self_imitation_mask(rewards, dones, quantile=0.8)
    assert top.shape == (T, N)
    # Rank cut of ceil((1-q)*N) = 1 environment, which on distinct returns is
    # exactly the set at or above the interpolated 0.8 quantile (4.8 here).
    assert top[0].tolist() == [False, False, True, False, False]
    # The mask is a property of the environment, so it is constant along time.
    assert bool((top == top[0]).all())
    median = self_imitation_mask(rewards, dones, quantile=0.5)
    assert median[0].tolist() == [True, False, True, False, True]
    # Everything qualifies at quantile 0, nothing but the best at quantile 1.
    assert bool(self_imitation_mask(rewards, dones, quantile=0.0).all())
    assert self_imitation_mask(rewards, dones, quantile=1.0)[0].tolist() == [
        False, False, True, False, False,
    ]
    # Restarting three times must not rank an environment above an equally
    # productive one that ran a single segment.
    split = torch.zeros(T, N)
    split[1, 0] = 1
    doubled = rewards.clone()
    doubled[:, 0] *= 2
    assert abs(
        float(segment_returns_per_env(doubled, torch.zeros(T, N), split)[0])
        - float(segment_returns_per_env(rewards, dones)[0])
    ) < 1e-6

    # Ties must not invert the positive set. Reward is `0.1*harvest +
    # 1.0*control`, so early in training most colonies score exactly 0.0, and a
    # `>= quantile` rule would then admit every one of them.
    tied = torch.zeros(T, 24)
    tied[:, 20] = 1.0
    tied[:, 21] = 2.0
    tied[:, 22] = 3.0
    tied[:, 23] = 4.0
    zeros = torch.zeros(T, 24)
    threshold = float(torch.quantile(segment_returns_per_env(tied, zeros), 0.8))
    assert threshold == 0.0  # the naive rule would select all 24
    selected = self_imitation_mask(tied, zeros, quantile=0.8)
    assert int(selected[0].sum()) == 5  # ceil(0.2 * 24)
    assert selected[0, 20:].tolist() == [True] * 4  # every earner is in
    # Self-imitation needs a strictly better slice to imitate. A rollout where
    # every environment scored identically has none, and cloning an arbitrary
    # five of them is cloning the current policy's failures, so the positive set
    # is empty rather than the intended size.
    flat = torch.ones(T, 24)
    assert int(self_imitation_mask(flat, zeros, quantile=0.8)[0].sum()) == 0
    assert int(self_imitation_mask(zeros, zeros, quantile=0.8)[0].sum()) == 0


def test_self_imitation_term_enters_the_policy_loss():
    """Eq. 10 weighting: the NLL is reported and it moves the actor."""
    import copy

    from samples.rl.agent.ppo import PPOTrainer, RolloutBatch

    torch.manual_seed(29)
    batch_size = 4
    obs = _dummy_obs_batch(batch_size)
    obs["actor_mask"][:, :3] = 1
    obs["intent_mask"][:, :3, :, INTENT_TYPES.index("none")] = 1
    obs["intent_mask"][:, :3, :, INTENT_TYPES.index("move")] = 1
    obs["amount_mask"][..., 0] = 1
    agent = Agent()
    with torch.no_grad():
        out = agent.actor(obs)
        critic_logits, critic_latent = agent.critic.value_logits_and_latent(obs)
        values = agent.critic.support.to_expected_scalar(critic_logits)
    positive = torch.tensor([True, False, True, False])

    def build() -> RolloutBatch:
        return RolloutBatch(
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
            actor_latent=out.state_latent,
            critic_latent=critic_latent,
            critic_logits=critic_logits,
            self_imitation=positive,
            batch_size=batch_size,
        )

    def run(coefficient: float) -> tuple[dict, torch.Tensor]:
        torch.manual_seed(31)
        trainer = PPOTrainer(
            copy.deepcopy(agent),
            device="cpu",
            compile_model=False,
            epochs=1,
            minibatch=batch_size,
            self_imitation_coef=coefficient,
        )
        stats = trainer.update(build())
        flat = torch.cat(
            [p.detach().reshape(-1) for p in trainer.actor.parameters()]
        )
        return stats, flat

    off_stats, off_params = run(0.0)
    on_stats, on_params = run(1.0)
    assert off_stats["self_imitation_coef"] == 0.0
    assert on_stats["self_imitation_coef"] == 1.0
    # Two of four transitions contribute, and the reported term is a real NLL.
    assert on_stats["self_imitation_transitions"] == 2.0
    assert on_stats["self_imitation_fraction"] == 0.5
    assert on_stats["self_imitation_nll"] > 0.0
    # The term is reported independently of its weight.
    assert abs(on_stats["self_imitation_nll"] - off_stats["self_imitation_nll"]) < 1e-6
    # A weighted term that changes no parameter is not in the loss.
    assert not torch.allclose(off_params, on_params)
    # An empty positive set contributes nothing rather than a spurious zero mean.
    torch.manual_seed(31)
    empty = PPOTrainer(
        copy.deepcopy(agent), device="cpu", compile_model=False,
        epochs=1, minibatch=batch_size, self_imitation_coef=1.0,
    )
    empty_rollout = build()
    empty_rollout.self_imitation = torch.zeros(batch_size, dtype=torch.bool)
    empty_stats = empty.update(empty_rollout)
    assert empty_stats["self_imitation_nll"] == 0.0
    assert empty_stats["self_imitation_transitions"] == 0.0


def _group_controller(num_envs: int, policy_envs: int, work: str):
    from samples.rl.agent.state_reservoir import (
        LaneMixture,
        ReservoirConfig,
        StartStateController,
        StartStateReservoir,
    )

    import random as _random

    reservoir = StartStateReservoir(
        work, config=ReservoirConfig(), rng=_random.Random(7),
    )
    for index in range(8):
        destination = reservoir.destination(
            lane="policy", event="pre_spawn", phase="mid", outcome="success",
            tick=1000 * (index + 1), env=index,
        )
        destination.write_bytes(b"snapshot")
        reservoir.admit(_snapshot_record(destination, tick=1000 * (index + 1)))
    return StartStateController(
        reservoir,
        mixture=LaneMixture(
            fresh=num_envs - policy_envs, policy=policy_envs, teacher=0,
        ),
        num_envs=num_envs,
        max_episode=20_000,
    )


def _snapshot_record(destination: Path, *, tick: int):
    from samples.rl.agent.state_reservoir import SnapshotRecord

    return SnapshotRecord(
        path=str(destination),
        lane="policy",
        event="pre_spawn",
        phase="mid",
        tick=tick,
        step=tick - 1,
        outcome="success",
        curriculum="seed_outpost",
        bytes=8,
        creeps=12,
        owned_rooms=1,
        remote_staffed=1,
        skill_rate=6.0,
        update=3,
        env=0,
        seed=3,
        events=("pre_spawn",),
        created=0.0,
    )


def test_group_starts_serve_k_environments_one_state():
    """VAPO §4.3 group sampling: k environments diverge from an identical world."""
    from samples.rl.agent.train import GroupedStartProvider

    with tempfile.TemporaryDirectory() as work:
        controller = _group_controller(6, 4, work)
        assert controller.lanes == (
            "fresh", "fresh", "policy", "policy", "policy", "policy",
        )
        provider = GroupedStartProvider(controller, group_size=2)
        assert provider.grouped_envs == 4
        assert provider.members == {0: (2, 3), 1: (4, 5)}
        # A fresh-lane environment is never grouped and never restored.
        assert provider(0) is None
        assert provider(1) is None
        first = provider(2)
        assert first is not None
        assert provider(3) == first
        second = provider(4)
        assert provider(5) == second
        # Two records were drawn to start four grouped environments; the fresh
        # lane bypasses the provider's bookkeeping entirely.
        assert provider.drawn == 2
        assert provider.shared_starts == 2
        # The follower's bookkeeping describes the world it actually resumed.
        assert controller.envs[3].origin_path == controller.envs[2].origin_path
        assert controller.envs[3].origin_tick == controller.envs[2].origin_tick
        assert controller.envs[3].last_capture_step == (
            controller.envs[2].last_capture_step
        )
        # Sharing does not persist across rollouts: the record may be evicted.
        provider.begin_rollout()
        before = provider.drawn
        provider(3)
        assert provider.drawn == before + 1


def test_group_start_size_is_configurable_and_one_disables_grouping():
    from samples.rl.agent.train import GroupedStartProvider

    with tempfile.TemporaryDirectory() as work:
        controller = _group_controller(6, 4, work)
        whole_lane = GroupedStartProvider(controller, group_size=4)
        paths = {whole_lane(index) for index in (2, 3, 4, 5)}
        assert len(paths) == 1 and None not in paths
        assert whole_lane.drawn == 1 and whole_lane.shared_starts == 3

        ungrouped = GroupedStartProvider(controller, group_size=1)
        assert ungrouped.grouped_envs == 0
        assert ungrouped.members == {}
        for index in (2, 3, 4, 5):
            assert ungrouped(index) is not None
        assert ungrouped.shared_starts == 0
        try:
            GroupedStartProvider(controller, group_size=0)
        except ValueError:
            pass
        else:  # pragma: no cover - contract violation
            raise AssertionError("group_size=0 must be rejected")


def test_rollout_page_chunking_scales_with_the_configured_horizon():
    """A 4096-step rollout must not multiply the per-minibatch gather loop."""
    from samples.rl.agent.rollout_buffer import (
        _PAGE_CHUNK_MAX_PAGES,
        _pages_per_chunk,
        HostRolloutBuffer,
    )

    envs = int(SCHEMA["ppo"]["numEnvs"])
    short = _pages_per_chunk(512, envs)
    long = _pages_per_chunk(int(SCHEMA["ppo"]["maxRolloutSteps"]), envs)
    assert long > short
    assert long <= _PAGE_CHUNK_MAX_PAGES
    # Chunk count, not chunk size, is what the gather loop pays for.
    def chunks(steps: int) -> int:
        pages = steps * envs  # one live room, the common early-economy case
        return -(-pages // _pages_per_chunk(steps, envs))
    assert chunks(4096) <= 2 * chunks(512)
    small = HostRolloutBuffer(4, 2)
    assert small.patch_pages.pages_per_chunk == _pages_per_chunk(4, 2)
    # Page ids are int32 in `patch_refs`; the store refuses to outgrow them.
    # `count` is how many pages are already stored, so ids run `0..count-1` and
    # the next id is `count` itself. `_MAX_PAGES` is the largest representable
    # id, so the first append that must be refused is the one at `count ==
    # _MAX_PAGES + 1`. Fabricating `_MAX_PAGES` instead asks for a legal id and
    # lets the store try to materialize 8.4M chunks of 17.9 MB before failing,
    # which OOM-kills the host rather than testing the guard.
    small.patch_pages.count = 2**31
    try:
        small.patch_pages.append(
            torch.zeros(1, *small.patch_pages.chunks[0].shape[1:], dtype=torch.uint8)
            if small.patch_pages.chunks
            else torch.zeros(1, PATCHES_PER_ROOM, PATCH_FLAT, dtype=torch.uint8)
        )
    except RuntimeError as error:
        assert "int32" in str(error)
    else:  # pragma: no cover - contract violation
        raise AssertionError("page ids must stay inside the int32 reference range")


def test_rollout_footprint_projection_tracks_the_horizon():
    from samples.rl.agent.rollout_buffer import HostRolloutBuffer

    envs = int(SCHEMA["ppo"]["numEnvs"])
    one = HostRolloutBuffer.projected_bytes(512, envs, rooms=1)
    eight = HostRolloutBuffer.projected_bytes(4096, envs, rooms=1)
    assert eight == 8 * one
    assert HostRolloutBuffer.projected_bytes(4096, envs, rooms=4) > eight
    # Sparse projection must stay under the dense four-room equivalent.
    assert eight < HostRolloutBuffer.dense_storage_bytes(4096, envs)


def main() -> int:
    tests = [
        test_muon_partition_covers_hidden_matrices_only,
        test_polar_express_beats_newton_schulz_on_small_singular_values,
        test_normuon_step_preserves_update_norm,
        test_normuon_decay_is_cautious,
        test_muon_second_moment_reweights_anisotropic_rows,
        test_muon_batched_group_matches_one_matrix_at_a_time,
        test_ppo_trainer_uses_hybrid_muon_with_rms_matched_rate,
        test_hybrid_muon_optimizer_state_roundtrip,
        test_muon_group_batches_same_shape_matrices,
        test_muon_rejects_partial_gradients,
        test_gae_truncation_bootstraps,
        test_gae_cuts_chain_on_done,
        test_gae_cuts_chain_on_truncation_without_done,
        test_decoupled_gae_pairs_policy_lambda_advantages_with_lambda_one_targets,
        test_type_gated_logprob_none_low_entropy,
        test_hl_gauss_support,
        test_hl_gauss_symlog_high_return_geometry_and_decode,
        test_hl_gauss_rejects_instead_of_clamping_overflow,
        test_reward_normalizer_roundtrip,
        test_critic_scalar_default,
        test_critic_value_head_stays_fp32_under_autocast,
        test_value_contract_has_no_clipping_and_fixed_critic_grad_clip,
        test_schema_reward_values_productive_economy,
        test_intent_enum_parity,
        test_logprob_sample_matches_evaluate,
        test_entity_context_changes_peer_policy_and_critic,
        test_creep_token_contains_exact_body_and_storage_state,
        test_per_actor_logprob_factorization,
        test_structure_cannot_select_remote_target_intent,
        test_immobile_creep_can_only_select_targets_already_in_primitive_range,
        test_creep_cannot_issue_room_construction,
        test_construction_uses_exact_type_and_tile_factors,
        test_construction_sample_and_reevaluate_logprob_match,
        test_spawn_count_chain_is_exactly_affordable_and_reevaluable,
        test_spawn_count_chain_has_full_support_without_forced_cheap_suffix,
        test_resource_amount_bins_are_unique_for_selected_target,
        test_creep_cannot_transfer_to_itself,
        test_tower_attack_requires_creep_target,
        test_factorized_ppo_update_finite,
        test_factorized_ppo_update_accepts_sparse_rollout_observations,
        test_nextlat_ppo_pairs_are_causal_masked_and_train_both_trunks,
        test_nextlat_time_major_pairing_cuts_done_truncation_and_tail,
        test_nextlat_action_encoding_uses_target_features_not_table_identity,
        test_nextlat_none_actions_have_zero_context_and_ignore_idle_state,
        test_nextlat_issued_actions_are_sensitive_and_permutation_invariant,
        test_nextlat_action_context_gradients_are_finite_and_mask_none_exactly,
        test_ratio_diagnostics_equal_weight_team_states,
        test_room_pack_keeps_expansion_capacity,
        test_entity_capacity_buckets_preserve_live_policy_and_value,
        test_rollout_zero_pads_compact_action_prefix,
        test_trunk_only_filter_for_reward_norm,
        test_safe_bc_nll_skips_neginf,
        test_safe_bc_nll_skips_finfo_min,
        test_safe_bc_nll_strict_rejects_masked_teacher_factor,
        test_discounted_returns_tn_exact_recurrence_and_boundaries,
        test_host_rollout_buffer_write_and_flat,
        test_sparse_rollout_pages_preserve_expansion_reorder_and_gather,
        test_sparse_rollout_gather_uses_room_capacity_buckets,
        test_sparse_rollout_reset_reuses_pages_without_stale_rooms,
        test_sparse_rollout_memory_accounting_beats_dense_one_room_history,
        test_spawn_replay_update_is_finite_and_strict,
        test_spawn_replay_is_bounded_and_mixed_buckets_pad_locally,
        test_global_lifecycle_reservoir_and_joint_epoch_are_finite,
        test_lifecycle_eligibility_gates_factor_losses_and_tick_retention,
        test_rare_intent_lane_balances_actors_and_scores_semantic_factors,
        test_ti_critic_epoch_consumes_full_balanced_order,
        test_joint_ckpt_meta_contract,
        test_joint_cli_requires_corpus_and_exposes_no_collection_overrides,
        test_artifact_rejects_pre_hl_gauss_learning_abi,
        test_artifact_source_override_is_explicit_and_narrow,
        test_spawn_replay_coverage_uses_tensors_and_eval_seeds_are_disjoint,
        test_joint_qualification_requires_critic_expansion_and_scale,
        test_spawn_body_expectations_are_measured_never_frozen,
        test_artifact_rejects_partial_actor_state,
        test_pack_field_order_documented,
        test_spawn_order_canonicalizes_zero_aliases_and_rejects_bad_wire_order,
        test_spawn_factor_budget_is_eight_counts_plus_at_most_seven_order_choices,
        test_ti_interleaved_spawn_supervises_counts_but_not_grouped_order,
        test_closed_loop_late_window_uses_current_not_peak_activity,
        test_server_neutrality_avoids_fragile_reservation_username_getter,
        test_segment_lengths_are_per_transition_not_a_per_env_mean,
        test_length_adaptive_lambda_matches_vapo_equation_five_and_clamps,
        test_decoupled_gae_consumes_the_per_transition_length_adaptive_lambda,
        test_actor_level_reduction_weights_every_decision_once,
        test_self_imitation_quantile_selects_top_return_environments,
        test_self_imitation_term_enters_the_policy_loss,
        test_group_starts_serve_k_environments_one_state,
        test_group_start_size_is_configurable_and_one_disables_grouping,
        test_rollout_page_chunking_scales_with_the_configured_horizon,
        test_rollout_footprint_projection_tracks_the_horizon,
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
