"""Entity-centric multi-room actor and centralized critic.

Two full networks (not shared weights):
  · Actor  — one factorized, goal-conditioned action per controlled entity
  · Critic — HL-Gauss return distribution decoded to a scalar expectation
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
    ACTOR_FEATURE_INDEX,
    AMOUNT_BINS,
    BODY_PART_COSTS,
    CONSTRUCTION_MASK_BYTES,
    GLOBAL_FEAT,
    INTENT_SPECS,
    INTENT_SLOTS,
    INTENT_TYPES,
    MAX_ACTORS,
    MAX_BODY_PARTS,
    MAX_ROOM_ENERGY,
    MAX_ROOMS,
    MAX_TARGETS,
    MODEL_CFG,
    N_ACTION_OUTCOME,
    N_AMOUNT,
    N_BODY_PART,
    N_CONSTRUCTION_TILE,
    N_CONSTRUCTION_TYPE,
    N_DIR,
    N_INTENT,
    PATCH_FLAT,
    PATCHES_PER_ROOM,
    PATCHES_PER_SIDE,
    ROOM_SIZE,
    TARGET_FEAT,
)
from .rope2d import RoPE2D

# Which argument heads affect the environment for each intent type. The schema is
# shared with Node so dummy heads cannot drift into logπ / entropy.
_DIR_TYPES = torch.zeros(N_INTENT, dtype=torch.bool)
_TGT_TYPES = torch.zeros(N_INTENT, dtype=torch.bool)
_AMT_TYPES = torch.zeros(N_INTENT, dtype=torch.bool)
_BODY_TYPES = torch.zeros(N_INTENT, dtype=torch.bool)
_CONSTRUCTION_TYPE_TYPES = torch.zeros(N_INTENT, dtype=torch.bool)
_CONSTRUCTION_TILE_TYPES = torch.zeros(N_INTENT, dtype=torch.bool)
_LOCAL_TARGET_TYPES = torch.zeros(N_INTENT, dtype=torch.bool)
_TARGET_RANGES = torch.zeros(N_INTENT, dtype=torch.float32)
_TRANSFER_TYPE = INTENT_TYPES.index("transfer")
_ATTACK_TYPE = INTENT_TYPES.index("attack")
_SPAWN_TYPE = INTENT_TYPES.index("spawnCreep")
_NONE_TYPE = INTENT_TYPES.index("none")
_AF = ACTOR_FEATURE_INDEX
if INTENT_SLOTS != 1:
    raise RuntimeError(
        "the current policy supports exactly one executable goal per actor; "
        f"got intentSlots={INTENT_SLOTS}"
    )
for _name in INTENT_TYPES:
    _i = INTENT_TYPES.index(_name)
    _spec = INTENT_SPECS[_name]
    _factors = set(_spec.get("factors", ()))
    if "direction" in _factors:
        _DIR_TYPES[_i] = True
    if "target" in _factors:
        _TGT_TYPES[_i] = True
    if "amount" in _factors:
        _AMT_TYPES[_i] = True
    if "bodyCounts" in _factors or "bodyOrder" in _factors:
        _BODY_TYPES[_i] = True
    if "constructionType" in _factors:
        _CONSTRUCTION_TYPE_TYPES[_i] = True
    if "constructionTile" in _factors:
        _CONSTRUCTION_TILE_TYPES[_i] = True
    if _spec.get("localTarget", False):
        _LOCAL_TARGET_TYPES[_i] = True
for _name, _range in {
    "harvest": 1, "transfer": 1, "withdraw": 1, "pickup": 1,
    "upgradeController": 3, "build": 3, "repair": 3,
    "attack": 1, "rangedAttack": 3, "heal": 1, "rangedHeal": 3,
    "claimController": 1, "reserveController": 1,
    "attackController": 1, "dismantle": 1,
}.items():
    if _name in INTENT_TYPES:
        _TARGET_RANGES[INTENT_TYPES.index(_name)] = float(_range)


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
    body_counts: Tensor
    body_order: Tensor
    construction_types: Tensor
    construction_tiles: Tensor
    logprob: Tensor  # [B]
    entropy: Tensor  # [B]
    actor_logprob: Tensor  # [B, A], one PPO factor per live actor
    actor_entropy: Tensor  # [B, A]
    # type, direction, target, amount, construction type/tile, then eight
    # body-count decisions and eight body-order decisions.
    factor_logprob: Tensor  # [B, A, 6 + 2*N_BODY_PART]
    factor_active: Tensor  # same shape; inactive arguments do not enter BC/PPO
    value: Tensor  # [B]  (filled by Agent from critic)
    state_latent: Tensor  # [B,D] contextual world state for temporal auxiliaries


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

    def forward(self, patches: Tensor, patch_pad: Tensor) -> tuple[Tensor, Tensor]:
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
        return x[:, 0].view(B, R, -1), x[:, 1:].view(B, R, P, -1)


class CoordinateEmbedding(nn.Module):
    """Fourier features for normalized entity coordinates.

    A plain linear projection makes distance and periodic boundary structure hard
    to recover.  Four frequency bands preserve both coarse room layout and exact
    local offsets without introducing an entity-order embedding.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.register_buffer(
            "frequencies",
            2.0 * torch.pi * (2.0 ** torch.arange(4, dtype=torch.float32)),
            persistent=False,
        )
        self.proj = nn.Sequential(
            nn.Linear(16, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, xy: Tensor) -> Tensor:
        phase = xy.unsqueeze(-1) * self.frequencies
        features = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        return self.proj(features)


class EntityBlock(nn.Module):
    """Pre-norm self-attention over rooms, actors, and targets."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        residual = x
        x = self.norm1(x)
        b, length, d_model = x.shape
        q = self.wq(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_mask = torch.zeros(b, 1, 1, length, device=x.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(
            padding_mask[:, None, None, :], torch.finfo(q.dtype).min,
        )
        mixed = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )
        mixed = mixed.transpose(1, 2).contiguous().view(b, length, d_model)
        x = residual + self.out(mixed)
        x = x + self.ff(self.norm2(x))
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class WorldTrunk(nn.Module):
    """Hierarchical spatial/entity transformer.

    Patch attention builds room summaries.  Entity attention then contextualizes
    every actor and target jointly, so a creep can coordinate with peers and the
    centralized critic can condition on the actual economy rather than only a
    lossy room CLS.
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or MODEL_CFG
        d = int(cfg["dModel"])
        self.d_model = d
        self.room_enc = PatchEncoder(
            d_model=d,
            n_heads=int(cfg["nHeads"]),
            n_layers=int(cfg.get("spatialLayers", cfg.get("nLayers", 3))),
            ff_mult=int(cfg["ffMult"]),
            dropout=float(cfg["dropout"]),
        )
        self.actor_proj = nn.Linear(ACTOR_FEAT, d)
        # Exact body composition is categorical, not an ordinal role shortcut.
        # One table entry per (part kind, active count 0..50) preserves absolute
        # throughput and damage state while the raw actor features still expose
        # the same counts to the general projection.
        self.body_count_embed = nn.Embedding(
            2 * N_BODY_PART * (MAX_BODY_PARTS + 1), d,
        )
        self.register_buffer(
            "_total_body_count_offsets",
            torch.arange(N_BODY_PART) * (MAX_BODY_PARTS + 1),
            persistent=False,
        )
        self.register_buffer(
            "_active_body_count_offsets",
            (torch.arange(N_BODY_PART) + N_BODY_PART) * (MAX_BODY_PARTS + 1),
            persistent=False,
        )
        self.target_proj = nn.Linear(TARGET_FEAT, d)
        self.global_proj = nn.Linear(GLOBAL_FEAT, d)
        self.coord_embed = CoordinateEmbedding(d)
        self.room_coord_embed = CoordinateEmbedding(d)
        self.room_embed = nn.Embedding(MAX_ROOMS, d)
        self.actor_kind_embed = nn.Embedding(2, d)
        self.actor_outcome_embed = nn.Embedding(N_ACTION_OUTCOME, d)
        self.target_kind_embed = nn.Embedding(7, d)
        # Index 0 means “not a structure/site”; encoded structure classes use 1..11.
        self.structure_kind_embed = nn.Embedding(12, d)
        # global, room, actor, target
        self.entity_type_embed = nn.Embedding(4, d)
        self.global_token = nn.Parameter(torch.zeros(1, 1, d))
        self.entity_blocks = nn.ModuleList([
            EntityBlock(d, int(cfg["nHeads"]), int(cfg["ffMult"]))
            for _ in range(int(cfg.get("entityLayers", 2)))
        ])
        self.final_norm = nn.LayerNorm(d)

    def forward(
        self,
        patches: Tensor,
        room_mask: Tensor,
        room_coords: Tensor,
        actors: Tensor,
        actor_mask: Tensor,
        actor_outcome: Tensor,
        targets: Tensor,
        target_mask: Tensor,
        globals_: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return actors, targets, actor rooms, global token, and spatial patches."""
        # Host promotion uses finite 1/2/4 room-capacity buckets. Shape—not
        # reset-time visibility—is authoritative, so expansion can grow later.
        r_use = max(1, min(int(room_mask.shape[1]), int(patches.shape[1])))
        patches = patches[:, :r_use]
        room_mask = room_mask[:, :r_use]

        patch_pad = (~room_mask.bool())[:, :, None].expand(-1, -1, PATCHES_PER_ROOM)
        # room_tok[b,r] = per-room CLS — keep all of them, no mean over rooms
        room_tok, room_patch_tok = self.room_enc(patches, patch_pad)
        room_tok = room_tok * room_mask.unsqueeze(-1)
        room_patch_tok = room_patch_tok * room_mask[:, :, None, None]
        g = self.global_proj(globals_)
        room_ids = torch.arange(r_use, device=patches.device)
        room_ctx = (
            room_tok
            + g.unsqueeze(1)
            + self.room_embed(room_ids)[None, :, :]
            + self.room_coord_embed(room_coords[:, :r_use])
            + self.entity_type_embed.weight[1][None, None, :]
        )

        # Actors / targets bind to *their* room CLS (encode.mjs feat index 3 = room slot)
        a_idx = _room_indices(actors[..., 3], r_use)
        t_idx = _room_indices(targets[..., 3], r_use)
        actor_room = _gather_rooms(room_ctx, a_idx)  # [B,A,D]
        target_room = _gather_rooms(room_ctx, t_idx)  # [B,T,D]

        actor_kind = actors[..., 0].round().long().clamp(0, 1)
        total_body_counts = (
            actors[..., _AF["totalMove"] : _AF["totalTough"] + 1]
            * float(MAX_BODY_PARTS)
        ).round().long().clamp(0, MAX_BODY_PARTS)
        active_body_counts = (
            actors[..., _AF["activeMove"] : _AF["activeTough"] + 1]
            * float(MAX_BODY_PARTS)
        ).round().long().clamp(0, MAX_BODY_PARTS)
        total_body_indices = total_body_counts + self._total_body_count_offsets
        active_body_indices = active_body_counts + self._active_body_count_offsets
        body_context = (
            self.body_count_embed(total_body_indices).sum(dim=-2)
            + self.body_count_embed(active_body_indices).sum(dim=-2)
        )
        body_context = body_context * (actor_kind == 0).unsqueeze(-1)
        actor_outcome = actor_outcome.round().long().clamp(0, N_ACTION_OUTCOME - 1)
        target_kind = (targets[..., 0] * 6).round().long().clamp(0, 6)
        structure_class = (targets[..., 5] * 10).round().long().clamp(0, 10) + 1
        has_structure_class = (target_kind == 2) | (target_kind == 5)
        structure_kind = torch.where(
            has_structure_class, structure_class, torch.zeros_like(structure_class),
        )
        actor_tok = (
            self.actor_proj(actors)
            + self.coord_embed(actors[..., 1:3])
            + self.actor_kind_embed(actor_kind)
            + body_context
            + self.actor_outcome_embed(actor_outcome)
            + actor_room
            + self.entity_type_embed.weight[2][None, None, :]
        ) * actor_mask.unsqueeze(-1)
        target_tok = (
            self.target_proj(targets)
            + self.coord_embed(targets[..., 1:3])
            + self.target_kind_embed(target_kind)
            + self.structure_kind_embed(structure_kind)
            + target_room
            + self.entity_type_embed.weight[3][None, None, :]
        ) * target_mask.unsqueeze(-1)

        global_tok = (
            self.global_token.expand(patches.shape[0], -1, -1)
            + g.unsqueeze(1)
            + self.entity_type_embed.weight[0][None, None, :]
        )
        entities = torch.cat((global_tok, room_ctx, actor_tok, target_tok), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(patches.shape[0], 1, dtype=torch.bool, device=patches.device),
                ~room_mask.bool(),
                ~actor_mask.bool(),
                ~target_mask.bool(),
            ),
            dim=1,
        )
        for block in self.entity_blocks:
            entities = block(entities, padding_mask)
        entities = self.final_norm(entities)

        actor_start = 1 + r_use
        actor_count = actors.shape[1]
        target_count = targets.shape[1]
        target_start = actor_start + actor_count
        backbone = entities[:, 0]
        actor_tok = entities[:, actor_start:target_start] * actor_mask.unsqueeze(-1)
        target_tok = entities[:, target_start:target_start + target_count] * target_mask.unsqueeze(-1)
        return actor_tok, target_tok, actor_room, backbone, room_patch_tok


class ActionConditionedDynamics(nn.Module):
    """Predict the next contextual world latent from state and joint action.

    This is the RL analogue of NextLat's ``(h_t, x_{t+1}) -> h_{t+1}``
    dynamics model.  Here the causal transition input is the action actually
    submitted at ``t``; the future observation is used only as a detached
    training target.  Target-table indices are never embedded as identities:
    selected target *features* are gathered from the current observation so
    the representation remains invariant to table packing order.
    """

    def __init__(self, d_model: int):
        super().__init__()
        d = int(d_model)
        self.type_embed = nn.Embedding(N_INTENT, d)
        self.dir_embed = nn.Embedding(N_DIR, d)
        self.amount_embed = nn.Embedding(N_AMOUNT, d)
        self.construction_type_embed = nn.Embedding(N_CONSTRUCTION_TYPE, d)
        self.body_order_embed = nn.Embedding(N_BODY_PART, d)
        self.actor_proj = nn.Linear(ACTOR_FEAT, d)
        self.target_proj = nn.Linear(TARGET_FEAT, d)
        self.body_count_proj = nn.Linear(N_BODY_PART, d, bias=False)
        self.tile_proj = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.action_norm = nn.LayerNorm(d)
        self.action_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.dynamics_norm = nn.LayerNorm(2 * d)
        self.dynamics_mlp = nn.Sequential(
            nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d),
        )
        self.register_buffer("_dir_need", _DIR_TYPES.clone(), persistent=False)
        self.register_buffer("_tgt_need", _TGT_TYPES.clone(), persistent=False)
        self.register_buffer("_amt_need", _AMT_TYPES.clone(), persistent=False)
        self.register_buffer("_body_need", _BODY_TYPES.clone(), persistent=False)
        self.register_buffer(
            "_construction_type_need", _CONSTRUCTION_TYPE_TYPES.clone(), persistent=False,
        )
        self.register_buffer(
            "_construction_tile_need", _CONSTRUCTION_TILE_TYPES.clone(), persistent=False,
        )
        self.register_buffer(
            "_body_order_positions",
            torch.linspace(1.0, 0.125, N_BODY_PART),
            persistent=False,
        )
        self.register_buffer(
            "_identity_body_order",
            torch.arange(N_BODY_PART, dtype=torch.long),
            persistent=False,
        )
        # Match NextLat's deliberately small dynamics initialization.  A random
        # residual of ordinary Kaiming scale would perturb the first PPO update
        # far more than the already pretrained policy/value objectives.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def action_context(
        self,
        state_latent: Tensor,
        batch: dict[str, Tensor],
        action: dict[str, Tensor],
    ) -> Tensor:
        """Encode the issued joint action independently of actor-table order.

        ``none`` is the absence of an environment action, so idle actors must
        not leak their state through the action branch.  Sum/sqrt(count) is a
        permutation-invariant set reduction that preserves useful joint-action
        cardinality while avoiding linear norm growth as population increases.
        Its clamped denominator also makes the all-none context exactly zero.
        """
        actor_mask = batch["actor_mask"].to(state_latent.dtype)
        actors = batch["actors"]
        targets = batch["targets"]
        actor_count = actors.shape[1]
        action = {
            key: (value[:, :actor_count] if value.dim() >= 2 else value)
            for key, value in action.items()
        }
        b = actors.shape[0]
        device = actors.device
        if "construction_types" not in action or "construction_tiles" not in action:
            scalar_default = torch.zeros(
                b, actor_count, INTENT_SLOTS, dtype=torch.long, device=device,
            )
            if "construction_types" not in action:
                action["construction_types"] = scalar_default
            if "construction_tiles" not in action:
                action["construction_tiles"] = scalar_default
        if "body_counts" not in action:
            action["body_counts"] = torch.zeros(
                b, actor_count, INTENT_SLOTS, N_BODY_PART,
                dtype=torch.long, device=device,
            )
        if "body_order" not in action:
            action["body_order"] = self._identity_body_order.view(1, 1, 1, -1).expand(
                b, actor_count, INTENT_SLOTS, -1,
            )
        types = action["types"][:, :, 0].long().clamp(0, N_INTENT - 1)
        dirs = action["dirs"][:, :, 0].long().clamp(0, N_DIR - 1)
        amounts = action["amounts"][:, :, 0].long().clamp(0, N_AMOUNT - 1)
        target_indices = action["targets"][:, :, 0].long().clamp(
            0, max(0, targets.shape[1] - 1),
        )
        selected_targets = targets.gather(
            1, target_indices.unsqueeze(-1).expand(-1, -1, TARGET_FEAT),
        )

        action_tok = self.actor_proj(actors) + self.type_embed(types)
        action_tok = action_tok + self.dir_embed(dirs) * self._dir_need[types].unsqueeze(-1)
        action_tok = action_tok + self.target_proj(selected_targets) * self._tgt_need[
            types
        ].unsqueeze(-1)
        action_tok = action_tok + self.amount_embed(amounts) * self._amt_need[
            types
        ].unsqueeze(-1)

        construction_types = action["construction_types"][:, :, 0].long().clamp(
            0, N_CONSTRUCTION_TYPE - 1,
        )
        action_tok = action_tok + self.construction_type_embed(
            construction_types,
        ) * self._construction_type_need[types].unsqueeze(-1)
        construction_tiles = action["construction_tiles"][:, :, 0].long().clamp(
            0, N_CONSTRUCTION_TILE - 1,
        )
        tile_xy = torch.stack(
            (
                (construction_tiles % ROOM_SIZE).to(state_latent.dtype),
                torch.div(construction_tiles, ROOM_SIZE, rounding_mode="floor").to(
                    state_latent.dtype,
                ),
            ),
            dim=-1,
        ) / float(ROOM_SIZE - 1)
        action_tok = action_tok + self.tile_proj(tile_xy) * self._construction_tile_need[
            types
        ].unsqueeze(-1)

        body_counts = action["body_counts"][:, :, 0].to(state_latent.dtype)
        action_tok = action_tok + self.body_count_proj(
            body_counts / float(MAX_BODY_PARTS),
        ) * self._body_need[types].unsqueeze(-1)
        body_order = action["body_order"][:, :, 0].long().clamp(0, N_BODY_PART - 1)
        order_context = (
            self.body_order_embed(body_order)
            * self._body_order_positions.to(state_latent.dtype)[None, None, :, None]
        ).sum(dim=-2)
        action_tok = action_tok + order_context * self._body_need[types].unsqueeze(-1)

        issued_mask = actor_mask.bool() & types.ne(_NONE_TYPE)
        issued_weight = issued_mask.to(state_latent.dtype)
        action_tok = self.action_mlp(self.action_norm(action_tok))
        action_tok = action_tok * issued_weight.unsqueeze(-1)
        issued_count = issued_weight.sum(dim=1, keepdim=True)
        return action_tok.sum(dim=1) / issued_count.clamp_min(1.0).sqrt()

    def forward(
        self,
        state_latent: Tensor,
        batch: dict[str, Tensor],
        action: dict[str, Tensor],
    ) -> Tensor:
        joint_action = self.action_context(state_latent, batch, action)
        delta = self.dynamics_mlp(
            self.dynamics_norm(torch.cat((state_latent, joint_action), dim=-1)),
        )
        return state_latent + delta


class Actor(nn.Module):
    """One factorized, goal-conditioned action per controlled entity."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or MODEL_CFG
        d = int(cfg["dModel"])
        self.d_model = d
        self.trunk = WorldTrunk(cfg)
        self.latent_dynamics = ActionConditionedDynamics(d)
        self.intent_embed = nn.Embedding(N_INTENT, d)
        self.step_norm = nn.LayerNorm(d * 2)
        self.step_ff = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
        self.type_head = nn.Linear(d, N_INTENT)
        self.dir_head = nn.Linear(d, N_DIR)
        self.target_head = nn.Linear(d, d)
        self.amount_head = nn.Linear(d, N_AMOUNT)
        self.construction_type_head = nn.Linear(d, N_CONSTRUCTION_TYPE)
        self.construction_type_embed = nn.Embedding(N_CONSTRUCTION_TYPE, d)
        self.construction_patch_embed = nn.Embedding(PATCHES_PER_ROOM, d)
        self.construction_tile_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 25),
        )
        # The neural network emits all count and ordering scores in one pass.
        # A cheap fixed eight-type scan below turns these parallel scores into an
        # exactly affordable joint action with an exact reevaluation likelihood.
        self.body_count_head = nn.Linear(d, N_BODY_PART * (MAX_BODY_PARTS + 1))
        self.body_order_head = nn.Linear(d, N_BODY_PART)
        # Avoid .to(device) every forward (extra H2D + kernel); compile-stable buffers.
        self.register_buffer("_dir_need", _DIR_TYPES.clone(), persistent=False)
        self.register_buffer("_tgt_need", _TGT_TYPES.clone(), persistent=False)
        self.register_buffer("_amt_need", _AMT_TYPES.clone(), persistent=False)
        self.register_buffer("_body_need", _BODY_TYPES.clone(), persistent=False)
        self.register_buffer(
            "_construction_type_need", _CONSTRUCTION_TYPE_TYPES.clone(), persistent=False,
        )
        self.register_buffer(
            "_construction_tile_need", _CONSTRUCTION_TILE_TYPES.clone(), persistent=False,
        )
        tile_patch_indices = []
        for tile in range(N_CONSTRUCTION_TILE):
            y, x = divmod(tile, 50)
            patch = (y // 5) * PATCHES_PER_SIDE + (x // 5)
            cell = (y % 5) * 5 + (x % 5)
            tile_patch_indices.append(patch * 25 + cell)
        self.register_buffer(
            "_tile_patch_indices",
            torch.tensor(tile_patch_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("_local_target", _LOCAL_TARGET_TYPES.clone(), persistent=False)
        self.register_buffer("_target_range", _TARGET_RANGES.clone(), persistent=False)
        self.register_buffer(
            "_amount_values", torch.tensor(AMOUNT_BINS, dtype=torch.float32), persistent=False,
        )
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
        nn.init.orthogonal_(self.construction_type_head.weight, gain=0.01)
        nn.init.orthogonal_(self.construction_tile_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.body_count_head.weight, gain=0.01)
        nn.init.orthogonal_(self.body_order_head.weight, gain=0.01)

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
            self.body_count_head.bias.zero_()
            self.body_order_head.bias.zero_()
            self.construction_type_head.bias.zero_()
            self.construction_tile_head[-1].bias.zero_()

    def _masked_logits(self, logits: Tensor, mask: Tensor) -> Tensor:
        # Contiguous masks keep torch.compile reduce-overhead from recompile-storming
        # on slot slices with different strides (E8).
        if not mask.is_contiguous():
            mask = mask.contiguous()
        if not logits.is_contiguous():
            logits = logits.contiguous()
        return logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)

    @staticmethod
    def _open_class0(mask: Tensor) -> Tensor:
        """Ensure ≥1 legal class without in-place clone (compile + launch friendly)."""
        empty = mask.sum(dim=-1) <= 0  # [...]
        m0 = mask[..., 0] + empty.to(dtype=mask.dtype)
        return torch.cat((m0.unsqueeze(-1), mask[..., 1:]), dim=-1)

    def _log_softmax_lp_ent(
        self, logits: Tensor, action: Tensor | None, deterministic: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Fused categorical sample/eval without torch.distributions objects.

        One log_softmax + gather + entropy reduction per head instead of
        Categorical construction (fewer Python/kernel launches on the AR path).
        """
        # float32 for numerical stability of log_softmax under bf16 autocast
        log_p = F.log_softmax(logits.float(), dim=-1)
        if action is None:
            if deterministic:
                action = log_p.argmax(dim=-1)
            else:
                # multinomial on last dim; reshape for arbitrary rank
                shape = log_p.shape
                flat = log_p.exp().reshape(-1, shape[-1])
                # Guard all-masked rows (uniform NaN risk): open class 0
                row_sum = flat.sum(dim=-1, keepdim=True)
                flat = torch.where(row_sum > 0, flat, torch.zeros_like(flat))
                flat[:, 0] = torch.where(row_sum.squeeze(-1) > 0, flat[:, 0], torch.ones_like(flat[:, 0]))
                action = torch.multinomial(flat, 1).reshape(shape[:-1])
        else:
            action = action.long()
        lp = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        # H = −Σ p log p — skip -inf classes (0 * -inf = NaN under mask fill)
        p = log_p.exp()
        ent = -torch.where(log_p.isneginf(), torch.zeros_like(log_p), p * log_p).sum(dim=-1)
        return action, lp, ent

    def _budget_conditioned_counts(
        self,
        count_logits: Tensor,
        budgets: Tensor,
        deterministic: bool,
        count_action: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample/evaluate every affordable nonempty body composition exactly once.

        The eight conditionals consume scores emitted in one neural pass. At
        type ``i`` a count is legal iff it fits the remaining part and energy
        budgets and, when the prefix is empty, leaves a possible nonempty
        continuation. Consequently the chain's support is precisely all count
        vectors with 1..50 parts and cost <= budget—without a separate length
        choice that can force a cheap-part suffix.
        """
        rows = int(count_logits.shape[0])
        costs = count_logits.new_tensor(BODY_PART_COSTS, dtype=torch.long)
        budgets = budgets.long().clamp(min(BODY_PART_COSTS), MAX_ROOM_ENERGY)
        values = torch.arange(
            MAX_BODY_PARTS + 1, device=count_logits.device, dtype=torch.long,
        )
        counts = torch.zeros(rows, N_BODY_PART, dtype=torch.long, device=count_logits.device)
        lp_counts = count_logits.new_zeros((rows, N_BODY_PART), dtype=torch.float32)
        ent_counts = count_logits.new_zeros((rows, N_BODY_PART), dtype=torch.float32)
        remaining_energy = budgets.clone()
        remaining_slots = torch.full_like(budgets, MAX_BODY_PARTS)
        total = torch.zeros_like(budgets)

        for part_type in range(N_BODY_PART):
            cost = costs[part_type]
            mask = (
                (values.unsqueeze(0) <= remaining_slots.unsqueeze(-1))
                & (values.unsqueeze(0) * cost <= remaining_energy.unsqueeze(-1))
            )
            # Zero is legal for an empty prefix only when a later type can still
            # make the body nonempty. TOUGH is last and therefore becomes the
            # sole forced choice only for budgets where every earlier count was 0.
            if part_type + 1 < N_BODY_PART:
                future_min_cost = int(min(BODY_PART_COSTS[part_type + 1 :]))
                zero_has_continuation = (
                    (total > 0)
                    | ((remaining_slots >= 1) & (remaining_energy >= future_min_cost))
                )
            else:
                zero_has_continuation = total > 0
            mask[:, 0] &= zero_has_continuation
            logits = self._masked_logits(count_logits[:, part_type, :], mask)
            raw_given = count_action[:, part_type] if count_action is not None else None
            given = raw_given.long().clamp(0, MAX_BODY_PARTS) if raw_given is not None else None
            chosen, lp, ent = self._log_softmax_lp_ent(logits, given, deterministic)
            if raw_given is not None:
                in_range = (raw_given >= 0) & (raw_given <= MAX_BODY_PARTS)
                lp = torch.where(
                    in_range, lp,
                    torch.full_like(lp, torch.finfo(lp.dtype).min),
                )
            counts[:, part_type] = chosen
            lp_counts[:, part_type] = lp
            ent_counts[:, part_type] = ent
            remaining_energy = (remaining_energy - chosen * cost).clamp_min(0)
            remaining_slots = (remaining_slots - chosen).clamp_min(0)
            total = total + chosen
        return counts, lp_counts, ent_counts

    def _positive_type_order(
        self,
        order_logits: Tensor,
        counts: Tensor,
        deterministic: bool,
        order_action: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Plackett–Luce order over positive types with a canonical zero suffix.

        Zero-count types do not affect the materialized body. Appending them in
        ascending type order removes probability aliases while retaining the
        protocol's full eight-token permutation. ``active`` contains only true
        stochastic decisions (at least two positive candidates); position zero
        is additionally active as a zero-logprob contract sentinel so malformed
        teacher permutations remain strict BC errors even for one-type bodies.
        """
        rows = int(order_logits.shape[0])
        selected = torch.zeros(rows, N_BODY_PART, dtype=torch.bool, device=order_logits.device)
        positive = counts > 0
        order = torch.zeros(rows, N_BODY_PART, dtype=torch.long, device=order_logits.device)
        lp_order = order_logits.new_zeros((rows, N_BODY_PART), dtype=torch.float32)
        ent_order = order_logits.new_zeros((rows, N_BODY_PART), dtype=torch.float32)
        active = torch.zeros(rows, N_BODY_PART, dtype=torch.bool, device=order_logits.device)
        contract_valid = torch.ones(rows, dtype=torch.bool, device=order_logits.device)
        type_ids = torch.arange(N_BODY_PART, device=order_logits.device)

        for pos in range(N_BODY_PART):
            candidates = positive & ~selected
            n_candidates = candidates.sum(dim=-1)
            has_positive = n_candidates > 0
            # Once all positive types are emitted, the smallest unused type is
            # the deterministic canonical suffix token.
            canonical = torch.where(
                ~selected,
                type_ids.unsqueeze(0),
                torch.full_like(type_ids.unsqueeze(0), N_BODY_PART),
            ).amin(dim=-1)
            logits = self._masked_logits(order_logits, self._open_class0(candidates))
            raw_given = order_action[:, pos] if order_action is not None else None
            given = raw_given.long().clamp(0, N_BODY_PART - 1) if raw_given is not None else None
            sampled, lp, ent = self._log_softmax_lp_ent(logits, given, deterministic)
            chosen = torch.where(has_positive, sampled, canonical)
            if raw_given is not None:
                in_range = (raw_given >= 0) & (raw_given < N_BODY_PART)
                expected_suffix = raw_given.long() == canonical
                valid_step = in_range & torch.where(
                    has_positive,
                    candidates.gather(-1, given.unsqueeze(-1)).squeeze(-1),
                    expected_suffix,
                )
                contract_valid &= valid_step
                chosen = raw_given.long().clamp(0, N_BODY_PART - 1)
            order[:, pos] = chosen
            stochastic = has_positive & (n_candidates > 1)
            active[:, pos] = stochastic
            lp_order[:, pos] = torch.where(stochastic, lp, torch.zeros_like(lp))
            ent_order[:, pos] = torch.where(stochastic, ent, torch.zeros_like(ent))
            selected |= F.one_hot(chosen, num_classes=N_BODY_PART).bool()

        # There is no stochastic order factor for a one-type body, but strict BC
        # still needs a place to report a malformed full permutation.
        active[:, 0] = True
        lp_order[:, 0] = torch.where(
            contract_valid,
            lp_order[:, 0],
            torch.full_like(lp_order[:, 0], torch.finfo(lp_order.dtype).min),
        )
        return order, lp_order, ent_order, active

    def forward(
        self,
        batch: dict[str, Tensor],
        deterministic: bool = False,
        action: dict[str, Tensor] | None = None,
    ) -> ActionOutput:
        patches = batch["patches"]
        B = patches.shape[0]
        A = batch["actors"].shape[1]
        device = patches.device
        if action is not None:
            action = {
                key: (value[:, :A] if torch.is_tensor(value) and value.dim() >= 2 else value)
                for key, value in action.items()
            }
        # actor_room = that actor's room CLS (+ globals); not a multi-room mean
        actor_tok, target_tok, actor_room, backbone, room_patch_tok = self.trunk(
            patches,
            batch["room_mask"],
            batch["room_coords"],
            batch["actors"],
            batch["actor_mask"],
            batch.get(
                "actor_outcome",
                torch.zeros_like(batch["actor_mask"], dtype=torch.long),
            ),
            batch["targets"],
            batch["target_mask"],
            batch["globals"],
        )
        actor_mask = batch["actor_mask"]
        intent_mask = batch["intent_mask"]
        dir_mask = batch["dir_mask"]
        target_select = batch["target_select_mask"]
        amount_mask = batch["amount_mask"]
        construction_mask = batch.get(
            "construction_mask",
            torch.zeros(
                B, batch["room_mask"].shape[1], N_CONSTRUCTION_TYPE,
                CONSTRUCTION_MASK_BYTES,
                dtype=torch.uint8, device=device,
            ),
        )
        target_mask = batch["target_mask"]

        dir_need_t = self._dir_need
        tgt_need_t = self._tgt_need
        amt_need_t = self._amt_need
        body_need_t = self._body_need
        construction_type_need_t = self._construction_type_need
        construction_tile_need_t = self._construction_tile_need

        # The wire retains a singleton sequence axis; the policy has one action.
        im = intent_mask[:, :, 0, :].contiguous()
        dm = dir_mask[:, :, 0, :].contiguous()
        if target_select.dim() != 3:
            raise ValueError(
                f"target_select_mask must be [B,nIntent,T], got {tuple(target_select.shape)}"
            )
        tm = target_select.contiguous()
        if amount_mask.dim() != 5:
            raise ValueError(
                "amount_mask must be [B,A,S,nIntent,nAmount], "
                f"got {tuple(amount_mask.shape)}"
            )
        am_m = amount_mask[:, :, 0, :, :].contiguous()
        actor_is_spawn = (
            batch["actors"][..., _AF["isNonCreep"]] >= 0.5
        ) & (batch["actors"][..., _AF["isSpawn"]] > 0.5)
        spawn_energy = (
            batch["actors"][..., _AF["roomEnergyAvailable"]]
            * float(MAX_ROOM_ENERGY)
        )
        spawn_compatible = actor_is_spawn & (spawn_energy + 1e-4 >= min(BODY_PART_COSTS))
        spawn_selector = (
            torch.arange(N_INTENT, device=device) == _SPAWN_TYPE
        )[None, None, :]
        im = im * torch.where(
            spawn_selector,
            spawn_compatible.unsqueeze(-1),
            torch.ones_like(spawn_selector.expand(B, A, -1)),
        ).to(im.dtype)
        r_use = int(batch["room_mask"].shape[1])
        actor_rooms = _room_indices(batch["actors"][..., 3], r_use)
        target_rooms = _room_indices(batch["targets"][..., 3], r_use)
        same_room = actor_rooms.unsqueeze(-1) == target_rooms.unsqueeze(1)
        target_kind = (batch["targets"][..., 0] * 6).round().long()
        same_position = (
            batch["actors"][..., None, 1:3]
            - batch["targets"][:, None, :, 1:3]
        ).abs().amax(dim=-1) < 1e-6
        actor_is_creep = batch["actors"][..., _AF["isNonCreep"]] < 0.5
        actor_is_tower = (
            ~actor_is_creep
        ) & (batch["actors"][..., _AF["isTower"]] > 0.5)
        target_is_creep = target_kind == 4
        self_target = (
            actor_is_creep.unsqueeze(-1)
            & (target_kind == 4).unsqueeze(1)
            & same_room
            & same_position
        )
        # Creeps with active MOVE may pursue visible cross-room macro goals.
        # An immobile creep may still execute a primitive against a target already
        # in range (stationary miners are important), but must never select a goal
        # that approachOr can only satisfy by returning ERR_NO_BODYPART forever.
        mobile_or_local = actor_is_creep.unsqueeze(-1) | same_room
        can_move = batch["actors"][..., _AF["activeMove"]] > 0
        target_distance = (
            batch["actors"][..., None, 1:3]
            - batch["targets"][:, None, :, 1:3]
        ).abs().amax(dim=-1) * float(ROOM_SIZE - 1)
        stationary_in_range = (
            same_room[:, :, None, :]
            & (
                target_distance[:, :, None, :]
                <= self._target_range[None, None, :, None] + 1e-4
            )
        )
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        actor_idx = torch.arange(A, device=device).unsqueeze(0)
        # A compact [intent,target] table is shared by actors, but legality is
        # actor-local for structures: a tower cannot act on another room's target.
        # Close target-requiring types before sampling when this actor's candidate
        # set is empty; opening target class 0 is only a numerical fallback.
        candidates = tm * target_mask[:, None, :]
        intent_target_compat = torch.where(
            self._local_target[None, None, :, None],
            same_room[:, :, None, :],
            mobile_or_local[:, :, None, :],
        )
        intent_target_compat = torch.where(
            (actor_is_creep & ~can_move)[:, :, None, None],
            stationary_in_range,
            intent_target_compat,
        )
        intent_target_compat = intent_target_compat & ~(
            (torch.arange(N_INTENT, device=device) == _TRANSFER_TYPE)[None, None, :, None]
            & self_target[:, :, None, :]
        )
        intent_target_compat = intent_target_compat & ~(
            actor_is_tower[:, :, None, None]
            & (torch.arange(N_INTENT, device=device) == _ATTACK_TYPE)[None, None, :, None]
            & ~target_is_creep[:, None, None, :]
        )
        candidate_exists = (
            candidates[:, None, :, :]
            * intent_target_compat.to(candidates.dtype)
        ).sum(dim=-1) > 0
        type_has_arguments = (
            (~self._tgt_need)[None, None, :] | candidate_exists
        )
        im = im * type_has_arguments.to(im.dtype)
        im = self._open_class0(im)

        # Make the policy explicitly consume the global state latent supervised
        # by the temporal objective.  The actor remains entity-specific through
        # actor_tok and actor_room; backbone supplies coordinated world context.
        h = self.step_ff(
            self.step_norm(torch.cat((actor_room, actor_tok + backbone.unsqueeze(1)), dim=-1)),
        )
        type_logits = self._masked_logits(self.type_head(h), im)
        t_given = action["types"][:, :, 0] if action is not None else None
        t, lp_t, ent_t = self._log_softmax_lp_ent(type_logits, t_given, deterministic)
        h = h + self.intent_embed(t)

        # Type-gated argument heads: only active factors enter logπ / entropy.
        need_dir = dir_need_t[t]
        need_tgt = tgt_need_t[t]
        need_amt = amt_need_t[t]
        need_body = body_need_t[t]
        need_construction_type = construction_type_need_t[t]
        need_construction_tile = construction_tile_need_t[t]

        dmask = self._open_class0(dm)
        dir_logits = self._masked_logits(self.dir_head(h), dmask)
        d_given = action["dirs"][:, :, 0] if action is not None else None
        d, lp_d, ent_d = self._log_softmax_lp_ent(dir_logits, d_given, deterministic)

        q = self.target_head(h)
        scores = torch.einsum("bad,btd->bat", q, target_tok) * (self.d_model ** -0.5)
        tmask = tm[batch_idx, t.long()]
        target_compat = intent_target_compat.gather(
            2,
            t.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, target_mask.shape[1]),
        ).squeeze(2)
        tmask = self._open_class0(
            tmask * target_mask[:, None, :] * target_compat.to(tmask.dtype)
        )
        scores = self._masked_logits(scores, tmask)
        tg_given = action["targets"][:, :, 0] if action is not None else None
        tg, lp_tg, ent_tg = self._log_softmax_lp_ent(scores, tg_given, deterministic)

        # Amount legality is intent-specific and selected-target-conditioned.
        amask = self._open_class0(am_m[batch_idx, actor_idx, t.long()])
        selected_target = target_tok.gather(
            1, tg.unsqueeze(-1).expand(-1, -1, self.d_model),
        )
        selected_features = batch["targets"].gather(
            1, tg.unsqueeze(-1).expand(-1, -1, TARGET_FEAT),
        )
        selected_kind = (selected_features[..., 0] * 6).round().long()
        structure_energy = selected_features[..., 13] * 1_000_000.0
        structure_capacity = selected_features[..., 14] * 1_000_000.0
        creep_energy = selected_features[..., 13] * 2_000.0
        creep_capacity = selected_features[..., 14] * 2_000.0
        target_energy = torch.where(selected_kind == 2, structure_energy, creep_energy)
        target_capacity = torch.where(selected_kind == 2, structure_capacity, creep_capacity)
        actor_energy = batch["actors"][..., _AF["storedEnergy"]] * float(MAX_ROOM_ENERGY)
        actor_capacity = batch["actors"][..., _AF["storeCapacity"]] * float(MAX_ROOM_ENERGY)
        actor_free = batch["actors"][..., _AF["storeFree"]] * actor_capacity
        transfer_limit = torch.minimum(
            actor_energy, (target_capacity - target_energy).clamp_min(0),
        )
        withdraw_limit = torch.minimum(actor_free, target_energy)
        transfer_type = INTENT_TYPES.index("transfer")
        withdraw_type = INTENT_TYPES.index("withdraw")
        drop_type = INTENT_TYPES.index("drop")
        resource_limit = torch.where(
            t == transfer_type,
            transfer_limit,
            torch.where(t == withdraw_type, withdraw_limit, actor_energy),
        )
        is_resource_amount = (t == transfer_type) | (t == withdraw_type) | (t == drop_type)
        values = self._amount_values.to(dtype=resource_limit.dtype)
        unique_resource_mask = (values == 0)[None, None, :] | (
            values[None, None, :] < resource_limit.unsqueeze(-1) - 1e-4
        )
        amask = amask * torch.where(
            is_resource_amount.unsqueeze(-1),
            unique_resource_mask,
            torch.ones_like(unique_resource_mask),
        ).to(amask.dtype)
        amask = self._open_class0(amask)
        amount_h = h + selected_target * need_tgt.unsqueeze(-1).to(h.dtype)
        amt_logits = self._masked_logits(self.amount_head(amount_h), amask)
        am_given = action["amounts"][:, :, 0] if action is not None else None
        am, lp_am, ent_am = self._log_softmax_lp_ent(amt_logits, am_given, deterministic)

        # Construction is one room-strategic action: choose a structure type, then
        # one exact tile in row-major y*50+x order. The bit-packed support comes
        # directly from the engine's construction validator and is conditioned on
        # the acting room, not the shared entity target budget.
        if construction_mask.dim() != 4 or construction_mask.shape[-1] != CONSTRUCTION_MASK_BYTES:
            raise ValueError(
                "construction_mask must be [B,R,nConstructionType,maskBytes], "
                f"got {tuple(construction_mask.shape)}"
            )
        room_actor = (
            batch["actors"][..., _AF["isRoom"]] > 0.5
        ) & actor_mask.bool()
        actor_room_one_hot = F.one_hot(actor_rooms, num_classes=r_use).bool()
        room_actor_one_hot = actor_room_one_hot & room_actor.unsqueeze(-1)
        # Exactly one strategic actor is encoded per owned room. Reduce its fully
        # contextual entity state into four fixed room rows, avoiding a 2,500-way
        # distribution for every creep and structure.
        room_h = torch.einsum(
            "bar,bad->brd", room_actor_one_hot.to(h.dtype), h,
        )
        room_type_mask = self._open_class0(construction_mask.any(dim=-1))
        room_type_logits = self._masked_logits(
            self.construction_type_head(room_h), room_type_mask,
        )
        if action is not None:
            actor_type_given = action.get(
                "construction_types", torch.zeros_like(action["types"]),
            )[:, :, 0].long()
            # These are categorical indices, not continuous features. A Long
            # einsum lowers to CUDA baddbmm, which has no integer kernel.
            room_type_given = torch.where(
                room_actor_one_hot,
                actor_type_given.unsqueeze(-1),
                torch.zeros((), dtype=actor_type_given.dtype, device=device),
            ).sum(dim=1)
        else:
            room_type_given = None
        room_construction_type, room_lp_construction_type, room_ent_construction_type = (
            self._log_softmax_lp_ent(
                room_type_logits, room_type_given, deterministic,
            )
        )
        construction_type = room_construction_type.gather(1, actor_rooms)
        lp_construction_type = room_lp_construction_type.gather(1, actor_rooms)
        ent_construction_type = room_ent_construction_type.gather(1, actor_rooms)
        selected_packed = construction_mask.gather(
            2,
            room_construction_type[:, :, None, None].expand(
                -1, -1, 1, CONSTRUCTION_MASK_BYTES,
            ),
        ).squeeze(2)
        tile_ids = torch.arange(N_CONSTRUCTION_TILE, device=device)
        tile_byte = torch.div(tile_ids, 8, rounding_mode="floor")
        tile_bit = tile_ids.remainder(8)
        construction_tile_mask = (
            selected_packed.index_select(-1, tile_byte)
            .bitwise_right_shift(tile_bit)
            .bitwise_and(1)
        )
        construction_tile_mask = self._open_class0(construction_tile_mask)
        patch_ids = torch.arange(PATCHES_PER_ROOM, device=device)
        patch_context = (
            room_patch_tok
            + self.construction_patch_embed(patch_ids)[None, None, :, :]
            + room_h[:, :, None, :]
            + self.construction_type_embed(room_construction_type)[:, :, None, :]
        )
        # Each contextual 5×5 patch emits its own 25 within-patch logits. This is
        # a spatial decoder over 100 map tokens, not a memorized absolute CLS head.
        room_tile_logits = self.construction_tile_head(patch_context).flatten(2)
        room_tile_logits = room_tile_logits.index_select(-1, self._tile_patch_indices)
        room_tile_logits = self._masked_logits(room_tile_logits, construction_tile_mask)
        if action is not None:
            actor_tile_given = action.get(
                "construction_tiles", torch.zeros_like(action["types"]),
            )[:, :, 0].long()
            room_tile_given = torch.where(
                room_actor_one_hot,
                actor_tile_given.unsqueeze(-1),
                torch.zeros((), dtype=actor_tile_given.dtype, device=device),
            ).sum(dim=1)
        else:
            room_tile_given = None
        room_construction_tile, room_lp_construction_tile, room_ent_construction_tile = (
            self._log_softmax_lp_ent(
                room_tile_logits, room_tile_given, deterministic,
            )
        )
        construction_tile = room_construction_tile.gather(1, actor_rooms)
        lp_construction_tile = room_lp_construction_tile.gather(1, actor_rooms)
        ent_construction_tile = room_ent_construction_tile.gather(1, actor_rooms)

        # Spawn composition is eight part counts followed by a positive-type
        # ordering. There is no independent body length and no 50-token decoder.
        count_logits = self.body_count_head(h).view(
            B, A, N_BODY_PART, MAX_BODY_PARTS + 1,
        )
        order_logits = self.body_order_head(h)
        counts_given = (
            action.get(
                "body_counts",
                torch.zeros(
                    B, A, INTENT_SLOTS, N_BODY_PART,
                    dtype=torch.long, device=device,
                ),
            )[:, :, 0, :]
            if action is not None else None
        )
        order_given = (
            action.get(
                "body_order",
                torch.arange(N_BODY_PART, device=device)[None, None, None, :].expand(
                    B, A, INTENT_SLOTS, -1,
                ),
            )[:, :, 0, :]
            if action is not None else None
        )
        # Fixed-shape scans over every padded actor keep one compileable graph.
        flat_count_logits = count_logits.reshape(-1, N_BODY_PART, MAX_BODY_PARTS + 1)
        budgets = (
            batch["actors"][..., _AF["roomEnergyAvailable"]].reshape(-1)
            * float(MAX_ROOM_ENERGY)
        ).round().long()
        selected_counts = (
            counts_given.reshape(-1, N_BODY_PART) if counts_given is not None else None
        )
        body_counts, lp_counts, ent_counts = self._budget_conditioned_counts(
            flat_count_logits,
            budgets,
            deterministic,
            selected_counts,
        )
        selected_order = (
            order_given.reshape(-1, N_BODY_PART) if order_given is not None else None
        )
        body_order, lp_order, ent_order, order_active = self._positive_type_order(
            order_logits.reshape(-1, N_BODY_PART),
            body_counts,
            deterministic,
            selected_order,
        )
        body_counts = body_counts.view(B, A, N_BODY_PART)
        body_order = body_order.view(B, A, N_BODY_PART)
        lp_counts = lp_counts.view(B, A, N_BODY_PART)
        ent_counts = ent_counts.view(B, A, N_BODY_PART)
        lp_order = lp_order.view(B, A, N_BODY_PART)
        ent_order = ent_order.view(B, A, N_BODY_PART)
        order_active = order_active.view(B, A, N_BODY_PART)
        body_gate = need_body.unsqueeze(-1) & actor_mask.bool().unsqueeze(-1)
        count_active = body_gate.expand(-1, -1, N_BODY_PART)
        order_active = order_active & body_gate

        base_active = torch.stack(
            (
                torch.ones_like(need_dir), need_dir, need_tgt, need_amt,
                need_construction_type, need_construction_tile,
            ),
            dim=-1,
        ) & actor_mask.bool().unsqueeze(-1)
        factor_active = torch.cat((base_active, count_active, order_active), dim=-1)
        factor_logprob = torch.cat(
            (
                torch.stack(
                    (
                        lp_t, lp_d, lp_tg, lp_am,
                        lp_construction_type, lp_construction_tile,
                    ),
                    dim=-1,
                ),
                lp_counts,
                lp_order,
            ),
            dim=-1,
        )
        factor_entropy = torch.cat(
            (
                torch.stack(
                    (
                        ent_t, ent_d, ent_tg, ent_am,
                        ent_construction_type, ent_construction_tile,
                    ),
                    dim=-1,
                ),
                ent_counts,
                ent_order,
            ),
            dim=-1,
        )
        actor_logprob = torch.where(
            factor_active, factor_logprob, torch.zeros_like(factor_logprob),
        ).sum(dim=-1)
        actor_entropy = torch.where(
            factor_active, factor_entropy, torch.zeros_like(factor_entropy),
        ).sum(dim=-1)
        n_live = actor_mask.sum(dim=-1).clamp_min(1.0)
        wire_body_counts = torch.where(
            need_body.unsqueeze(-1), body_counts, torch.zeros_like(body_counts),
        )
        identity_order = torch.arange(N_BODY_PART, device=device).view(1, 1, -1)
        wire_body_order = torch.where(
            need_body.unsqueeze(-1), body_order, identity_order,
        )

        return ActionOutput(
            types=t.unsqueeze(-1),
            dirs=d.unsqueeze(-1),
            targets=tg.unsqueeze(-1),
            amounts=am.unsqueeze(-1),
            body_counts=wire_body_counts.unsqueeze(2),
            body_order=wire_body_order.unsqueeze(2),
            construction_types=construction_type.unsqueeze(-1),
            construction_tiles=construction_tile.unsqueeze(-1),
            logprob=actor_logprob.sum(dim=-1),
            entropy=actor_entropy.sum(dim=-1) / n_live,
            actor_logprob=actor_logprob,
            actor_entropy=actor_entropy,
            factor_logprob=factor_logprob,
            factor_active=factor_active,
            value=torch.zeros(B, device=device),  # filled by Agent via critic
            state_latent=backbone,
        )

    def encode_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode only the contextual global state used by NextLat targets."""
        _, _, _, backbone, _ = self.trunk(
            batch["patches"], batch["room_mask"], batch["room_coords"],
            batch["actors"], batch["actor_mask"],
            batch.get("actor_outcome", torch.zeros_like(batch["actor_mask"], dtype=torch.long)),
            batch["targets"], batch["target_mask"], batch["globals"],
        )
        return backbone

    def predict_next_latent(
        self, state_latent: Tensor, batch: dict[str, Tensor], action: dict[str, Tensor],
    ) -> Tensor:
        return self.latent_dynamics(state_latent, batch, action)


class Critic(nn.Module):
    """Independent entity-aware HL-Gauss critic for full-return targets.

    The head predicts a categorical distribution over a signed-log return
    support. ``forward`` decodes a scalar for bootstrapping and GAE; training
    consumes ``value_logits`` directly and never applies PPO value clipping.
    """

    def __init__(self, cfg: dict | None = None, value_cfg: dict | None = None):
        super().__init__()
        cfg = cfg or MODEL_CFG
        d = int(cfg["dModel"])
        self.trunk = WorldTrunk(cfg)
        self.latent_dynamics = ActionConditionedDynamics(d)
        from .constants import SCHEMA

        vcfg = value_cfg if value_cfg is not None else (SCHEMA.get("value") or {})
        if vcfg.get("loss") != "hlGauss" or vcfg.get("transform") != "symlog":
            raise ValueError("critic requires value.loss=hlGauss and value.transform=symlog")
        from .hl_gauss import HLGaussSupport

        self.use_hl_gauss = True
        self.support = HLGaussSupport.symmetric_symlog(
            max_abs_return=float(vcfg["maxAbsReturn"]),
            interior_bins=int(vcfg["interiorBins"]),
            margin_bins=int(vcfg["marginBins"]),
            sigma_ratio=float(vcfg["sigmaRatio"]),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, self.support.num_bins),
        )
        prior = float(vcfg.get("prior", 0.0))
        last = self.value_head[-1]
        with torch.no_grad():
            last.weight.zero_()
            # Initialize to a projected zero-return prior and keep far-bin logits
            # finite enough to receive early gradients.
            last.bias.copy_(
                self.support.project_to_logprobs(torch.tensor(prior), eps=1e-6)
            )

    def _backbone(self, batch: dict[str, Tensor]) -> Tensor:
        _, _, _, backbone, _ = self.trunk(
            batch["patches"],
            batch["room_mask"],
            batch["room_coords"],
            batch["actors"],
            batch["actor_mask"],
            batch.get(
                "actor_outcome",
                torch.zeros_like(batch["actor_mask"], dtype=torch.long),
            ),
            batch["targets"],
            batch["target_mask"],
            batch["globals"],
        )
        return backbone

    def value_logits(self, batch: dict[str, Tensor]) -> Tensor:
        backbone = self._backbone(batch)
        # Distribution parameters are fp32 statistics even when the much larger
        # trunk runs under bf16 autocast.
        with torch.autocast(device_type=backbone.device.type, enabled=False):
            return self.value_head(backbone.float())

    def value_logits_and_latent(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        backbone = self._backbone(batch)
        with torch.autocast(device_type=backbone.device.type, enabled=False):
            logits = self.value_head(backbone.float())
        return logits, backbone

    def predict_next_latent(
        self, state_latent: Tensor, batch: dict[str, Tensor], action: dict[str, Tensor],
    ) -> Tensor:
        return self.latent_dynamics(state_latent, batch, action)

    def detached_value_logits(self, state_latent: Tensor) -> Tensor:
        """Evaluate the value head without updating it from the NextLat KL."""
        first, second, last = self.value_head[0], self.value_head[2], self.value_head[4]
        x = F.linear(state_latent.float(), first.weight.detach(), first.bias.detach())
        x = F.gelu(x)
        x = F.linear(x, second.weight.detach(), second.bias.detach())
        x = F.gelu(x)
        return F.linear(x, last.weight.detach(), last.bias.detach())

    def forward(
        self,
        batch: dict[str, Tensor],
        return_logits: bool = False,
        return_latent: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # backbone = attention over per-room CLSes (WorldTrunk); no mean-pool
        logits, backbone = self.value_logits_and_latent(batch)
        if return_latent:
            return logits, backbone
        if return_logits:
            return logits
        return self.support.to_expected_scalar(logits)


class Agent(nn.Module):
    """Actor/critic owner; PPO compiles and executes the networks separately."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.actor = Actor(cfg)
        self.critic = Critic(cfg)

    def freeze_room_pack(self, room_mask: Tensor) -> int:
        """Report the current finite room bucket without freezing expansion."""
        r0 = max(1, min(MAX_ROOMS, int(room_mask.shape[1])))
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


def configure_cuda_backends() -> None:
    """TF32 + cudnn benchmark — fewer slow kernels on Ampere+ without accuracy hits for RL."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Prefer matmul precision for float32 accum paths under autocast
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        # Avoid dynamo cache thrash from minor shape churn
        import torch._dynamo as dynamo

        dynamo.config.cache_size_limit = max(64, int(getattr(dynamo.config, "cache_size_limit", 8)))
    except Exception:
        pass


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
        print(
            f"[model] torch.compile({name}, mode={mode}) wrapped; "
            "compilation is lazy on first use",
            flush=True,
        )
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
            "body_counts": torch.zeros(
                B, MAX_ACTORS, INTENT_SLOTS, N_BODY_PART,
                dtype=torch.long, device=device,
            ),
            "construction_types": torch.zeros(
                B, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long, device=device,
            ),
            "construction_tiles": torch.zeros(
                B, MAX_ACTORS, INTENT_SLOTS, dtype=torch.long, device=device,
            ),
            "body_order": torch.arange(N_BODY_PART, device=device).view(
                1, 1, 1, N_BODY_PART,
            ).expand(B, MAX_ACTORS, INTENT_SLOTS, -1),
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
