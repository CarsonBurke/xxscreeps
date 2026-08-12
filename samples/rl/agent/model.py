"""Entity-centric multi-room actor and centralized critic.

Two full networks (not shared weights):
  · Actor  — one factorized, goal-conditioned action per controlled entity
  · Critic — scalar value trained from full-return targets
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
    AMOUNT_BINS,
    GLOBAL_FEAT,
    INTENT_SPECS,
    INTENT_SLOTS,
    INTENT_TYPES,
    MAX_ACTORS,
    MAX_ROOMS,
    MAX_TARGETS,
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

# Which argument heads affect the environment for each intent type. The schema is
# shared with Node so dummy heads cannot drift into logπ / entropy.
_DIR_TYPES = torch.zeros(N_INTENT, dtype=torch.bool)
_TGT_TYPES = torch.zeros(N_INTENT, dtype=torch.bool)
_AMT_TYPES = torch.zeros(N_INTENT, dtype=torch.bool)
_LOCAL_TARGET_TYPES = torch.zeros(N_INTENT, dtype=torch.bool)
_TRANSFER_TYPE = INTENT_TYPES.index("transfer")
_ATTACK_TYPE = INTENT_TYPES.index("attack")
if INTENT_SLOTS != 1:
    raise RuntimeError(
        "schema-v2 supports exactly one executable goal per actor; "
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
    if _spec.get("localTarget", False):
        _LOCAL_TARGET_TYPES[_i] = True


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
    actor_logprob: Tensor  # [B, A], one PPO factor per live actor
    actor_entropy: Tensor  # [B, A]
    factor_logprob: Tensor  # [B, A, 4] = type, direction, target, amount
    factor_active: Tensor  # [B, A, 4] bool; inactive arguments do not enter BC/PPO
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
        self.target_proj = nn.Linear(TARGET_FEAT, d)
        self.global_proj = nn.Linear(GLOBAL_FEAT, d)
        self.coord_embed = CoordinateEmbedding(d)
        self.room_coord_embed = CoordinateEmbedding(d)
        self.room_embed = nn.Embedding(MAX_ROOMS, d)
        self.actor_kind_embed = nn.Embedding(2, d)
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
        # Freeze active-room count before compile so reduce-overhead never sees
        # data-dependent .item() / R changes. Prefer freeze_room_pack() at warmup.
        self._static_r: int | None = None

    def forward(
        self,
        patches: Tensor,
        room_mask: Tensor,
        room_coords: Tensor,
        actors: Tensor,
        actor_mask: Tensor,
        targets: Tensor,
        target_mask: Tensor,
        globals_: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return contextual actors, targets, actor rooms, and global value token."""
        # Rooms packed from the front. R is frozen — never recompute from mask.sum().
        if self._static_r is None:
            # Fallback only if warmup forgot freeze_room_pack (still once).
            r0 = int(room_mask.shape[1])
            self._static_r = max(1, min(r0, int(patches.shape[1])))
        r_use = self._static_r
        # Always slice to fixed R so the graph does not branch on "already R".
        patches = patches[:, :r_use]
        room_mask = room_mask[:, :r_use]

        patch_pad = (~room_mask.bool())[:, :, None].expand(-1, -1, PATCHES_PER_ROOM)
        # room_tok[b,r] = per-room CLS — keep all of them, no mean over rooms
        room_tok = self.room_enc(patches, patch_pad) * room_mask.unsqueeze(-1)
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
        target_start = actor_start + MAX_ACTORS
        backbone = entities[:, 0]
        actor_tok = entities[:, actor_start:target_start] * actor_mask.unsqueeze(-1)
        target_tok = entities[:, target_start:target_start + MAX_TARGETS] * target_mask.unsqueeze(-1)
        return actor_tok, target_tok, actor_room, backbone


class Actor(nn.Module):
    """One factorized, goal-conditioned action per controlled entity."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or MODEL_CFG
        d = int(cfg["dModel"])
        self.d_model = d
        self.trunk = WorldTrunk(cfg)
        self.intent_embed = nn.Embedding(N_INTENT, d)
        self.step_norm = nn.LayerNorm(d * 2)
        self.step_ff = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
        self.type_head = nn.Linear(d, N_INTENT)
        self.dir_head = nn.Linear(d, N_DIR)
        self.target_head = nn.Linear(d, d)
        self.amount_head = nn.Linear(d, N_AMOUNT)
        # Avoid .to(device) every forward (extra H2D + kernel); compile-stable buffers.
        self.register_buffer("_dir_need", _DIR_TYPES.clone(), persistent=False)
        self.register_buffer("_tgt_need", _TGT_TYPES.clone(), persistent=False)
        self.register_buffer("_amt_need", _AMT_TYPES.clone(), persistent=False)
        self.register_buffer("_local_target", _LOCAL_TARGET_TYPES.clone(), persistent=False)
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
            batch["room_coords"],
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

        dir_need_t = self._dir_need
        tgt_need_t = self._tgt_need
        amt_need_t = self._amt_need

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
        r_use = int(self.trunk._static_r or batch["room_mask"].shape[1])
        actor_rooms = _room_indices(batch["actors"][..., 3], r_use)
        target_rooms = _room_indices(batch["targets"][..., 3], r_use)
        same_room = actor_rooms.unsqueeze(-1) == target_rooms.unsqueeze(1)
        target_kind = (batch["targets"][..., 0] * 6).round().long()
        same_position = (
            batch["actors"][..., None, 1:3]
            - batch["targets"][:, None, :, 1:3]
        ).abs().amax(dim=-1) < 1e-6
        actor_is_creep = batch["actors"][..., 0] < 0.5
        actor_is_tower = (~actor_is_creep) & (batch["actors"][..., 5] > 0.5)
        target_is_creep = target_kind == 4
        self_target = (
            actor_is_creep.unsqueeze(-1)
            & (target_kind == 4).unsqueeze(1)
            & same_room
            & same_position
        )
        # Creeps may pursue visible cross-room macro goals. Structures remain local.
        mobile_or_local = actor_is_creep.unsqueeze(-1) | same_room
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        actor_idx = torch.arange(MAX_ACTORS, device=device).unsqueeze(0)
        # A compact [intent,target] table is shared by actors, but legality is
        # actor-local for structures: a tower cannot act on another room's target.
        # Close target-requiring types before sampling when this actor's candidate
        # set is empty; opening target class 0 is only a numerical fallback.
        candidates = tm * target_mask[:, None, :]
        global_candidate = candidates.sum(dim=-1) > 0  # [B,nIntent]
        local_candidate = torch.einsum(
            "bit,bat->bai", candidates, same_room.to(candidates.dtype),
        ) > 0
        is_mobile = actor_is_creep
        candidate_exists = torch.where(
            is_mobile.unsqueeze(-1), global_candidate.unsqueeze(1), local_candidate,
        )
        candidate_exists = torch.where(
            self._local_target[None, None, :], local_candidate, candidate_exists,
        )
        transfer_exists = (
            candidates[:, _TRANSFER_TYPE, :].unsqueeze(1)
            * (~self_target).to(candidates.dtype)
        ).sum(dim=-1) > 0
        transfer_selector = (
            torch.arange(N_INTENT, device=device) == _TRANSFER_TYPE
        )[None, None, :]
        candidate_exists = torch.where(
            transfer_selector, transfer_exists.unsqueeze(-1), candidate_exists,
        )
        tower_attack_exists = (
            candidates[:, _ATTACK_TYPE, :].unsqueeze(1)
            * same_room.to(candidates.dtype)
            * target_is_creep.unsqueeze(1).to(candidates.dtype)
        ).sum(dim=-1) > 0
        attack_selector = (
            torch.arange(N_INTENT, device=device) == _ATTACK_TYPE
        )[None, None, :]
        candidate_exists = torch.where(
            attack_selector & actor_is_tower.unsqueeze(-1),
            tower_attack_exists.unsqueeze(-1),
            candidate_exists,
        )
        type_has_arguments = (
            (~self._tgt_need)[None, None, :] | candidate_exists
        )
        im = im * type_has_arguments.to(im.dtype)

        h = self.step_ff(self.step_norm(torch.cat((actor_room, actor_tok), dim=-1)))
        type_logits = self._masked_logits(self.type_head(h), im)
        t_given = action["types"][:, :, 0] if action is not None else None
        t, lp_t, ent_t = self._log_softmax_lp_ent(type_logits, t_given, deterministic)
        h = h + self.intent_embed(t)

        # Type-gated argument heads: only active factors enter logπ / entropy.
        need_dir = dir_need_t[t]
        need_tgt = tgt_need_t[t]
        need_amt = amt_need_t[t]

        dmask = self._open_class0(dm)
        dir_logits = self._masked_logits(self.dir_head(h), dmask)
        d_given = action["dirs"][:, :, 0] if action is not None else None
        d, lp_d, ent_d = self._log_softmax_lp_ent(dir_logits, d_given, deterministic)

        q = self.target_head(h)
        scores = torch.einsum("bad,btd->bat", q, target_tok) * (self.d_model ** -0.5)
        tmask = tm[batch_idx, t.long()]
        target_compat = torch.where(
            self._local_target[t.long()].unsqueeze(-1), same_room, mobile_or_local,
        )
        target_compat = target_compat & ~(
            (t == _TRANSFER_TYPE).unsqueeze(-1) & self_target
        )
        target_compat = target_compat & ~(
            actor_is_tower.unsqueeze(-1)
            & (t == _ATTACK_TYPE).unsqueeze(-1)
            & ~target_is_creep.unsqueeze(1)
        )
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
        actor_energy = batch["actors"][..., 20] * 2_000.0
        actor_capacity = batch["actors"][..., 21] * 2_000.0
        actor_free = batch["actors"][..., 16] * actor_capacity
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

        factor_active = torch.stack(
            (torch.ones_like(need_dir), need_dir, need_tgt, need_amt), dim=-1,
        ) & actor_mask.bool().unsqueeze(-1)
        factor_logprob = torch.stack((lp_t, lp_d, lp_tg, lp_am), dim=-1)
        factor_entropy = torch.stack((ent_t, ent_d, ent_tg, ent_am), dim=-1)
        actor_logprob = (
            factor_logprob * factor_active.to(factor_logprob.dtype)
        ).sum(dim=-1)
        actor_entropy = (
            factor_entropy * factor_active.to(factor_entropy.dtype)
        ).sum(dim=-1)
        n_live = actor_mask.sum(dim=-1).clamp_min(1.0)

        return ActionOutput(
            types=t.unsqueeze(-1),
            dirs=d.unsqueeze(-1),
            targets=tg.unsqueeze(-1),
            amounts=am.unsqueeze(-1),
            logprob=actor_logprob.sum(dim=-1),
            entropy=actor_entropy.sum(dim=-1) / n_live,
            actor_logprob=actor_logprob,
            actor_entropy=actor_entropy,
            factor_logprob=factor_logprob,
            factor_active=factor_active,
            value=torch.zeros(B, device=device),  # filled by Agent via critic
        )


class Critic(nn.Module):
    """Independent entity-aware value network trained with full-return targets.

    Optional HL-Gauss categorical head (parameter-golf / cleanrl recipe) when
    schema.value.hlGauss is true. forward() always returns a scalar for GAE.
    """

    def __init__(self, cfg: dict | None = None, value_cfg: dict | None = None):
        super().__init__()
        cfg = cfg or MODEL_CFG
        d = int(cfg["dModel"])
        self.trunk = WorldTrunk(cfg)
        from .constants import SCHEMA

        vcfg = value_cfg if value_cfg is not None else (SCHEMA.get("value") or {})
        self.use_hl_gauss = bool(vcfg.get("hlGauss", False))
        if self.use_hl_gauss:
            from .hl_gauss import HLGaussSupport

            bins = int(vcfg.get("numBins", 101))
            self.support = HLGaussSupport(
                num_bins=bins,
                v_min=float(vcfg.get("vMin", -10.0)),
                v_max=float(vcfg.get("vMax", 10.0)),
                sigma_ratio=float(vcfg.get("sigmaRatio", 2.0)),
            )
            self.value_head = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, bins),
            )
            prior = float(vcfg.get("prior", 0.0))
            last = self.value_head[-1]
            with torch.no_grad():
                last.weight.zero_()
                last.bias.copy_(self.support.project_to_logprobs(torch.tensor(prior), eps=1e-6))
        else:
            self.support = None
            self.value_head = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, 1),
            )
            # Near-zero prior: avoid huge initial V under reward-norm (scale poison).
            last = self.value_head[-1]
            with torch.no_grad():
                last.weight.zero_()
                last.bias.zero_()

    def _backbone(self, batch: dict[str, Tensor]) -> Tensor:
        *_, backbone = self.trunk(
            batch["patches"],
            batch["room_mask"],
            batch["room_coords"],
            batch["actors"],
            batch["actor_mask"],
            batch["targets"],
            batch["target_mask"],
            batch["globals"],
        )
        return backbone

    def value_logits(self, batch: dict[str, Tensor]) -> Tensor:
        backbone = self._backbone(batch)
        if self.use_hl_gauss:
            return self.value_head(backbone.float())
        return self.value_head(backbone)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        # backbone = attention over per-room CLSes (WorldTrunk); no mean-pool
        if self.use_hl_gauss:
            logits = self.value_logits(batch)
            return self.support.to_expected_scalar(logits)
        return self.value_head(self._backbone(batch)).squeeze(-1)


class Agent(nn.Module):
    """Actor/critic owner; PPO compiles and executes the networks separately."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.actor = Actor(cfg)
        self.critic = Critic(cfg)

    def freeze_room_pack(self, room_mask: Tensor) -> int:
        """Freeze the supported room capacity, not reset-time visible rooms.

        Visibility can grow after claiming/scouting. Freezing the initial active
        count made every later room permanently invisible to both policy and PPO.
        """
        r0 = max(1, min(MAX_ROOMS, int(room_mask.shape[1])))
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
