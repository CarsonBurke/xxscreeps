"""ViT multi-room encoder + separate Actor / Critic (VAPO-style).

Two full networks (not shared weights):
  · Actor  — AR multi-intent policy
  · Critic — scalar value (MC λ=1 targets)
Each can be wrapped in torch.compile independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .constants import (
    ACTOR_FEAT,
    INTENT_SLOTS,
    INTENT_TYPES,
    MAX_ACTORS,
    MAX_ROOMS,
    MODEL_CFG,
    N_AMOUNT,
    N_DIR,
    N_INTENT,
    PATCH_FLAT,
    PATCHES_PER_ROOM,
    PATCHES_PER_SIDE,
    TARGET_FEAT,
)
from .rope2d import RoPE2D


def _room_indices(room_feat: Tensor, r_use: int) -> Tensor:
    """Decode encode.mjs room slot: index / max(1, maxRooms-1) → integer room id."""
    scale = max(1, MAX_ROOMS - 1)
    return (room_feat * scale).round().long().clamp(0, max(0, r_use - 1))


def _gather_rooms(room_tok: Tensor, room_idx: Tensor) -> Tensor:
    """room_tok [B,R,D], room_idx [B,N] → [B,N,D] per-entity room CLS."""
    d = room_tok.shape[-1]
    return room_tok.gather(1, room_idx.unsqueeze(-1).expand(-1, -1, d))


@dataclass
class ActionOutput:
    types: Tensor  # [B, A, S]
    dirs: Tensor
    targets: Tensor
    amounts: Tensor
    logprob: Tensor  # [B]
    entropy: Tensor  # [B]
    value: Tensor  # [B]  (filled by Agent from critic)


class RMSNorm(nn.Module):
    """QK-Norm style RMSNorm over the last dim (no mean centering)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., dim]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class PatchEncoder(nn.Module):
    """Pre-LN transformer + QK RMSNorm + 2D RoPE + SDPA."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, ff_mult: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(PATCH_FLAT, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.rope = RoPE2D(d_model // n_heads, max_h=PATCHES_PER_SIDE + 1, max_w=PATCHES_PER_SIDE + 1)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                # Pre-LN
                "norm1": nn.LayerNorm(d_model),
                "wq": nn.Linear(d_model, d_model),
                "wk": nn.Linear(d_model, d_model),
                "wv": nn.Linear(d_model, d_model),
                # QK-Norm (per head)
                "q_norm": RMSNorm(self.head_dim),
                "k_norm": RMSNorm(self.head_dim),
                "out": nn.Linear(d_model, d_model),
                "norm2": nn.LayerNorm(d_model),
                "ff": nn.Sequential(
                    nn.Linear(d_model, d_model * ff_mult),
                    nn.GELU(),
                    nn.Linear(d_model * ff_mult, d_model),
                ),
            })
            for _ in range(n_layers)
        ])

    def _attn(self, x: Tensor, block: nn.ModuleDict, key_padding_mask: Tensor | None) -> Tensor:
        B, L, D = x.shape
        h, dh = self.n_heads, self.head_dim
        q = block["wq"](x).view(B, L, h, dh).transpose(1, 2)
        k = block["wk"](x).view(B, L, h, dh).transpose(1, 2)
        v = block["wv"](x).view(B, L, h, dh).transpose(1, 2)

        # QK-Norm before RoPE (common practice)
        q = block["q_norm"](q)
        k = block["k_norm"](k)

        q_cls, q_pat = q[:, :, :1], q[:, :, 1:]
        k_cls, k_pat = k[:, :, :1], k[:, :, 1:]
        if q_pat.shape[2] > 0:
            q_pat, k_pat = self.rope(q_pat, k_pat, PATCHES_PER_SIDE, PATCHES_PER_SIDE)
        q = torch.cat((q_cls, q_pat), dim=2)
        k = torch.cat((k_cls, k_pat), dim=2)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(
                key_padding_mask[:, None, None, :], torch.finfo(q.dtype).min,
            )

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return block["out"](out)

    def forward(self, patches: Tensor, patch_pad: Tensor) -> Tensor:
        B, R, P, F = patches.shape
        x = self.proj(patches.reshape(B * R, P, F))
        cls = self.cls.expand(B * R, -1, -1)
        x = torch.cat((cls, x), dim=1)
        pad = torch.cat(
            (
                torch.zeros(B * R, 1, dtype=torch.bool, device=patches.device),
                patch_pad.reshape(B * R, P),
            ),
            dim=1,
        )
        # Pre-LN residual blocks: x = x + f(LN(x))
        for block in self.blocks:
            x = x + self._attn(block["norm1"](x), block, pad)
            x = x + block["ff"](block["norm2"](x))
        return x[:, 0].view(B, R, -1)


class WorldTrunk(nn.Module):
    """Room ViT + per-entity room CLS (no multi-room mean pool).

    Per-room CLS tokens are kept intact. Actors/targets gather the CLS for
    *their* room. Critic backbone is a learned attention pool over room CLSes
    (not mean/max/min), plus globals.
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or MODEL_CFG
        d = int(cfg["dModel"])
        self.d_model = d
        self.room_enc = PatchEncoder(
            d_model=d,
            n_heads=int(cfg["nHeads"]),
            n_layers=int(cfg["nLayers"]),
            ff_mult=int(cfg["ffMult"]),
            dropout=float(cfg["dropout"]),
        )
        self.actor_proj = nn.Linear(ACTOR_FEAT, d)
        self.target_proj = nn.Linear(TARGET_FEAT, d)
        self.global_proj = nn.Linear(6, d)
        # Learned query scores for attention-pooling room CLSes → critic backbone
        self.room_pool_score = nn.Linear(d, 1)
        # Freeze active-room count after first forward so torch.compile reduce-overhead
        # sees a stable R (CUDA graphs hate .item() + reshape every step).
        self._static_r: int | None = None

    def forward(
        self,
        patches: Tensor,
        room_mask: Tensor,
        actors: Tensor,
        actor_mask: Tensor,
        targets: Tensor,
        target_mask: Tensor,
        globals_: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns actor_tok, target_tok, actor_room [B,A,D], backbone [B,D]."""
        # Rooms packed from the front in encode.mjs. Slice to frozen R_use (usually 1).
        if self._static_r is None:
            r0 = int(room_mask.sum(dim=1).max().item()) if room_mask.numel() else 1
            self._static_r = max(1, min(r0, int(patches.shape[1])))
        r_use = self._static_r
        if r_use < patches.shape[1]:
            patches = patches[:, :r_use]
            room_mask = room_mask[:, :r_use]

        patch_pad = (~room_mask.bool())[:, :, None].expand(-1, -1, PATCHES_PER_ROOM)
        # room_tok[b,r] = per-room CLS — keep all of them, no mean over rooms
        room_tok = self.room_enc(patches, patch_pad) * room_mask.unsqueeze(-1)
        g = self.global_proj(globals_)
        # Globals available in every room context (shared empire state)
        room_ctx = room_tok + g.unsqueeze(1)

        # Actors / targets bind to *their* room CLS (encode.mjs feat index 3 = room slot)
        a_idx = _room_indices(actors[..., 3], r_use)
        t_idx = _room_indices(targets[..., 3], r_use)
        actor_room = _gather_rooms(room_ctx, a_idx)  # [B,A,D]
        target_room = _gather_rooms(room_ctx, t_idx)  # [B,T,D]

        # Keep room CLS separate from table feats so the AR head sees both streams
        # (not double-counted). Targets still add room so bilinear scores are room-aware.
        actor_tok = self.actor_proj(actors) * actor_mask.unsqueeze(-1)
        target_tok = (self.target_proj(targets) + target_room) * target_mask.unsqueeze(-1)

        # Critic needs one world vector: attention over room CLSes (not mean/max/min)
        logits = self.room_pool_score(room_ctx).squeeze(-1)  # [B,R]
        logits = logits.masked_fill(~room_mask.bool(), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        backbone = (weights.unsqueeze(-1) * room_ctx).sum(dim=1)
        return actor_tok, target_tok, actor_room, backbone


class Actor(nn.Module):
    """Autoregressive multi-intent policy (own trunk)."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or MODEL_CFG
        d = int(cfg["dModel"])
        self.d_model = d
        self.trunk = WorldTrunk(cfg)
        self.intent_embed = nn.Embedding(N_INTENT, d)
        self.slot_embed = nn.Embedding(INTENT_SLOTS, d)
        self.step_norm = nn.LayerNorm(d * 4)
        self.step_ff = nn.Sequential(nn.Linear(d * 4, d), nn.GELU(), nn.Linear(d, d))
        self.type_head = nn.Linear(d, N_INTENT)
        self.dir_head = nn.Linear(d, N_DIR)
        self.target_head = nn.Linear(d, d)
        self.amount_head = nn.Linear(d, N_AMOUNT)
        self._init_action_priors(cfg)

    def _init_action_priors(self, cfg: dict) -> None:
        """Bias type logits so early exploration favors move/harvest, not combat/spawn spam.

        Masks still zero-out illegal intents. For spawns (only spawnCreep legal), the
        large negative spawn bias is irrelevant after masking. For creeps, move/harvest
        dominate the prior.
        """
        # Small weights so bias dominates early; orthogonal keeps a bit of structure.
        nn.init.orthogonal_(self.type_head.weight, gain=0.01)
        nn.init.orthogonal_(self.dir_head.weight, gain=0.5)
        nn.init.orthogonal_(self.amount_head.weight, gain=0.01)

        # Default: strong negative on everything except move/harvest (+ mild none).
        prior = cfg.get("intentPriorBias") or {
            "move": 3.0,
            "harvest": 3.5,
            "none": 0.5,
            "transfer": -1.0,
            "withdraw": -1.5,
            "pickup": -2.0,
            "drop": -2.5,
            "upgradeController": -1.0,
            "build": -2.0,
            "repair": -2.0,
            "attack": -4.0,
            "rangedAttack": -4.0,
            "heal": -4.0,
            "rangedHeal": -4.0,
            "claimController": -4.0,
            "reserveController": -4.0,
            "attackController": -4.0,
            "dismantle": -4.0,
            "generateSafeMode": -4.0,
            "spawnCreep": -3.0,
        }
        default_bias = float(cfg.get("intentPriorDefault", -3.0))
        bias = torch.full((N_INTENT,), default_bias)
        for name, val in prior.items():
            if name in INTENT_TYPES:
                bias[INTENT_TYPES.index(name)] = float(val)
        with torch.no_grad():
            self.type_head.bias.copy_(bias)
            # Prefer amount bin 0 ("all / default") when amounts matter
            if self.amount_head.bias is not None:
                self.amount_head.bias.zero_()
                self.amount_head.bias[0] = 1.0

    def _masked_logits(self, logits: Tensor, mask: Tensor) -> Tensor:
        return logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)

    def _sample_or_mode(self, logits: Tensor, deterministic: bool) -> tuple[Tensor, Tensor, Tensor]:
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def forward(
        self,
        batch: dict[str, Tensor],
        deterministic: bool = False,
        action: dict[str, Tensor] | None = None,
    ) -> ActionOutput:
        patches = batch["patches"]
        B = patches.shape[0]
        device = patches.device
        # actor_room = that actor's room CLS (+ globals); not a multi-room mean
        actor_tok, target_tok, actor_room, _backbone = self.trunk(
            patches,
            batch["room_mask"],
            batch["actors"],
            batch["actor_mask"],
            batch["targets"],
            batch["target_mask"],
            batch["globals"],
        )
        actor_mask = batch["actor_mask"]
        intent_mask = batch["intent_mask"]
        dir_mask = batch["dir_mask"]
        target_select = batch["target_select_mask"]
        amount_mask = batch["amount_mask"]
        target_mask = batch["target_mask"]

        types_out, dirs_out, tgts_out, amts_out = [], [], [], []
        logps, ents = [], []
        prev = torch.zeros(B, MAX_ACTORS, self.d_model, device=device)

        for slot in range(INTENT_SLOTS):
            slot_e = self.slot_embed.weight[slot][None, None, :].expand(B, MAX_ACTORS, -1)
            # Per-actor room CLS (not shared mean of all rooms)
            h = self.step_ff(
                self.step_norm(
                    torch.cat((actor_room, actor_tok, prev, slot_e), dim=-1)
                )
            )

            type_logits = self._masked_logits(self.type_head(h), intent_mask[:, :, slot, :])
            if action is not None:
                t = action["types"][:, :, slot]
                dist = torch.distributions.Categorical(logits=type_logits)
                lp_t, ent_t = dist.log_prob(t), dist.entropy()
            else:
                t, lp_t, ent_t = self._sample_or_mode(type_logits, deterministic)
            types_out.append(t)
            prev = prev + self.intent_embed(t)
            h = h + self.intent_embed(t)

            dir_logits = self._masked_logits(self.dir_head(h), dir_mask[:, :, slot, :])
            if (dir_mask[:, :, slot, :].sum(dim=-1, keepdim=True) <= 0).any():
                fix = dir_mask[:, :, slot, :].clone()
                fix[..., 0] = 1
                dir_logits = self._masked_logits(self.dir_head(h), fix)
            if action is not None:
                d = action["dirs"][:, :, slot]
                dist = torch.distributions.Categorical(logits=dir_logits)
                lp_d, ent_d = dist.log_prob(d), dist.entropy()
            else:
                d, lp_d, ent_d = self._sample_or_mode(dir_logits, deterministic)
            dirs_out.append(d)

            q = self.target_head(h)
            scores = torch.einsum("bad,btd->bat", q, target_tok)
            tmask = target_select[:, :, slot, :] * target_mask[:, None, :]
            scores = self._masked_logits(scores, tmask)
            if (tmask.sum(dim=-1, keepdim=True) <= 0).any():
                fix = tmask.clone()
                fix[..., 0] = 1
                scores = self._masked_logits(torch.einsum("bad,btd->bat", q, target_tok), fix)
            if action is not None:
                tg = action["targets"][:, :, slot]
                dist = torch.distributions.Categorical(logits=scores)
                lp_tg, ent_tg = dist.log_prob(tg), dist.entropy()
            else:
                tg, lp_tg, ent_tg = self._sample_or_mode(scores, deterministic)
            tgts_out.append(tg)

            amt_logits = self._masked_logits(self.amount_head(h), amount_mask[:, :, slot, :])
            if (amount_mask[:, :, slot, :].sum(dim=-1, keepdim=True) <= 0).any():
                fix = amount_mask[:, :, slot, :].clone()
                fix[..., 0] = 1
                amt_logits = self._masked_logits(self.amount_head(h), fix)
            if action is not None:
                am = action["amounts"][:, :, slot]
                dist = torch.distributions.Categorical(logits=amt_logits)
                lp_am, ent_am = dist.log_prob(am), dist.entropy()
            else:
                am, lp_am, ent_am = self._sample_or_mode(amt_logits, deterministic)
            amts_out.append(am)

            m = actor_mask
            logps.append(((lp_t + lp_d + lp_tg + lp_am) * m).sum(dim=-1))
            ents.append(((ent_t + ent_d + ent_tg + ent_am) * m).sum(dim=-1))

        return ActionOutput(
            types=torch.stack(types_out, dim=-1),
            dirs=torch.stack(dirs_out, dim=-1),
            targets=torch.stack(tgts_out, dim=-1),
            amounts=torch.stack(amts_out, dim=-1),
            logprob=torch.stack(logps, dim=-1).sum(dim=-1),
            entropy=torch.stack(ents, dim=-1).sum(dim=-1),
            value=torch.zeros(B, device=device),  # filled by Agent via critic
        )


class Critic(nn.Module):
    """Independent value network (own trunk) — MC λ=1 targets."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or MODEL_CFG
        d = int(cfg["dModel"])
        self.trunk = WorldTrunk(cfg)
        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        *_, backbone = self.trunk(
            batch["patches"],
            batch["room_mask"],
            batch["actors"],
            batch["actor_mask"],
            batch["targets"],
            batch["target_mask"],
            batch["globals"],
        )
        # backbone = attention over per-room CLSes (WorldTrunk); no mean-pool
        return self.value_head(backbone).squeeze(-1)


class Agent(nn.Module):
    """Monolithic actor+critic module (single unit for torch.compile).

    forward/act always runs both trunks — one compiled graph for rollout.
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.actor = Actor(cfg)
        self.critic = Critic(cfg)

    def freeze_room_pack(self, room_mask: Tensor) -> int:
        """Freeze active-room count before compile so graphs stay static."""
        r0 = int(room_mask.sum(dim=1).max().item()) if room_mask.numel() else 1
        r0 = max(1, min(r0, int(room_mask.shape[1])))
        self.actor.trunk._static_r = r0
        self.critic.trunk._static_r = r0
        return r0

    def act(
        self,
        batch: dict[str, Tensor],
        deterministic: bool = False,
        action: dict[str, Tensor] | None = None,
    ) -> ActionOutput:
        out = self.actor(batch, deterministic=deterministic, action=action)
        out.value = self.critic(batch)
        return out

    def get_value(self, batch: dict[str, Tensor]) -> Tensor:
        return self.critic(batch)

    def forward(
        self,
        batch: dict[str, Tensor],
        deterministic: bool = False,
        action: dict[str, Tensor] | None = None,
    ) -> ActionOutput:
        """Monolithic entrypoint (compiled for reduce-overhead rollout)."""
        return self.act(batch, deterministic=deterministic, action=action)


# Back-compat alias
ScreepsPolicy = Agent


def maybe_compile(
    module: nn.Module,
    enabled: bool,
    name: str,
    *,
    mode: str = "reduce-overhead",
) -> nn.Module:
    """Monolithic module compile (CUDA-graph-friendly reduce-overhead)."""
    if not enabled or not hasattr(torch, "compile"):
        return module
    try:
        compiled = torch.compile(
            module,
            mode=mode,
            fullgraph=False,
            dynamic=False,
        )
        print(f"[model] torch.compile({name}, mode={mode}, monolithic) ok", flush=True)
        return compiled  # type: ignore[return-value]
    except Exception as err:  # noqa: BLE001
        print(f"[model] torch.compile({name}) failed: {err}", flush=True)
        return module


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def estimate_peak_vram_mb(batch: dict[str, Tensor], actor: nn.Module, critic: nn.Module) -> float:
    """Rough activation+param peak for one mb step (CUDA only)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    actor.train()
    critic.train()
    B = batch["patches"].shape[0]
    device = batch["patches"].device
    # Actor may be Critic-only probe (same module twice)
    loss_terms = []
    try:
        out = actor(batch, action={
            "types": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long, device=device),
            "dirs": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long, device=device),
            "targets": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long, device=device),
            "amounts": torch.zeros(B, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long, device=device),
        })
        loss_terms.append(out.logprob.mean())
    except TypeError:
        # critic-only: module(batch) → value
        v_a = actor(batch)
        loss_terms.append(v_a.mean())
    v = critic(batch)
    loss_terms.append(v.mean())
    loss = sum(loss_terms)
    loss.backward()
    peak = torch.cuda.max_memory_allocated()
    actor.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    del loss, v
    torch.cuda.empty_cache()
    return (peak - base) / (1024 ** 2)


def auto_minibatch_size(
    sample_obs: dict[str, Tensor],
    actor: nn.Module,
    critic: nn.Module,
    *,
    total_vram_gb: float = 32.0,
    reserved_gb: float = 12.0,
    safety: float = 0.85,
    max_mb: int = 2048,
    min_mb: int = 8,
) -> int:
    """
    Pick largest safe minibatch under (total − reserved) × safety,
    also clamping to *currently free* CUDA memory (other processes).
    """
    if not torch.cuda.is_available():
        return min(64, max_mb)

    free_b, total_b = torch.cuda.mem_get_info()
    free_gb = free_b / (1024 ** 3)
    # Full budget: min(configured reserve, nearly all currently free)
    configured = (total_vram_gb - reserved_gb) * safety
    live = free_gb * safety
    budget_gb = max(0.5, min(configured, live))
    budget_bytes = budget_gb * (1024 ** 3)

    already = torch.cuda.memory_allocated()
    free_budget = max(budget_bytes - already * 0.1, 128 * 1024 ** 2)

    device = next(actor.parameters()).device
    probe = {
        k: (v[:1].to(device) if torch.is_tensor(v) and v.shape[0] != 1 else v.to(device) if torch.is_tensor(v) else v)
        for k, v in sample_obs.items() if not k.startswith("_")
    }
    for k, v in list(probe.items()):
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] != 1:
            probe[k] = v[:1]

    try:
        mb1_mb = estimate_peak_vram_mb(probe, actor, critic)
    except Exception as err:  # noqa: BLE001
        print(f"[vram] probe failed ({err}); default minibatch=32", flush=True)
        return 32

    # modest overhead pad for SDPA/bf16 (not the old 2.5× cut)
    mb1_mb = max(mb1_mb * 1.25, 40.0)
    act_budget_mb = free_budget / (1024 ** 2)
    best = max(min_mb, int(act_budget_mb / mb1_mb))
    best = min(best, max_mb)
    # Full VRAM-max fit (round down to multiple of 8)
    best = max(min_mb, (best // 8) * 8)
    print(
        f"[vram] probe_mb1≈{mb1_mb:.1f}MiB cuda_free={free_gb:.1f}GiB "
        f"budget={budget_gb:.1f}GiB → minibatch={best} (full VRAM-max fit) "
        f"(total={total_vram_gb}G reserved={reserved_gb}G safety={safety})",
        flush=True,
    )
    return best
