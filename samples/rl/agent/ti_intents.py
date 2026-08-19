"""Exact, conservative translation of The International's raw intents.

The expert can submit several engine intents for one creep in a tick, while the
current policy has one macro slot.  This module never guesses: it emits a full
label only when the submitted command's type is exactly one of our macro
actions, which includes a spawn, because the arguments a spawn carries beyond
its body are executor or engine concerns and not action factors here.  A spawn
body is preserved raw: downstream training always uses its exact counts and uses
its order only when the sequence is representable as contiguous type blocks.
Everything else carries an explicit rejection reason and remains useful for
critic/representation training.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .constants import (
    AMOUNT_BINS,
    BODY_PART_TYPES,
    CONSTRUCTION_TYPES,
    INTENT_TYPES,
    MAX_BODY_PARTS,
)


_DIRECT_TARGET_INTENTS = {
    "harvest", "pickup", "upgradeController", "build", "repair", "attack",
    "rangedAttack", "heal", "rangedHeal", "claimController",
    "reserveController", "attackController", "dismantle",
}

# Logistics carry an amount. The executor reads amount bin 0 as "everything
# available" (`legalAmount` returns the full capacity for a non-positive
# request), so an expert command that omits the amount is exactly representable
# rather than a guess. An explicit amount is representable only when it lands on
# a bin value; anything else would silently change how much moves.
_AMOUNT_TARGET_INTENTS = {"transfer", "withdraw"}
_ENERGY_RESOURCE = "energy"
_EVERYTHING_BIN = 0


def _amount_bin(amount: Any) -> int | None:
    """Bin index for an expert amount, or None when it is not representable."""
    if amount is None:
        return _EVERYTHING_BIN
    if isinstance(amount, bool) or not isinstance(amount, int):
        return None
    if amount <= 0:
        return None
    return AMOUNT_BINS.index(amount) if amount in AMOUNT_BINS else None


@dataclass(frozen=True)
class TiLabel:
    actor_index: int | None
    intent: str | None
    target_index: int | None = None
    # The candidate list is rebuilt every tick, so an index is only meaningful
    # for the tick it came from. Retro-labelling an earlier tick needs the
    # object identity to re-resolve against that tick's candidates.
    target_id: str | None = None
    direction: int | None = None
    construction_type: int | None = None
    construction_tile: int | None = None
    amount_index: int | None = None
    body_parts: tuple[int, ...] | None = None
    full_action: bool = False
    rejection: str | None = None


def _arguments(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("intent arguments are not a list")
    return value


def translate_ti_intents(
    payload: dict[str, Any] | None,
    actor_meta: list[dict[str, Any]],
    target_meta: list[dict[str, Any]],
    room_names: list[str],
) -> list[TiLabel]:
    """Translate exact representable labels from one pre-action expert tick."""
    if not isinstance(payload, dict):
        return []
    rooms = payload.get("room") if "room" in payload else payload
    if not isinstance(rooms, dict):
        return [TiLabel(None, None, rejection="malformed_payload")]
    actor_index = {
        str(identifier): index
        for index, row in enumerate(actor_meta)
        for identifier in (row.get("objectId"), row.get("id"))
        if identifier
    }
    target_index = {
        str(row.get("id")): index for index, row in enumerate(target_meta) if row.get("id")
    }
    room_actor = {
        str(row.get("room")): index
        for index, row in enumerate(actor_meta)
        if row.get("kind") == "room"
    }
    labels: list[TiLabel] = []
    for room_name, room_payload in rooms.items():
        if not isinstance(room_payload, dict):
            labels.append(TiLabel(None, None, rejection="malformed_room"))
            continue
        local = room_payload.get("local") or {}
        create_sites = local.get("createConstructionSite") if isinstance(local, dict) else None
        if create_sites:
            rows = create_sites if isinstance(create_sites, list) else []
            # One room actor can express one construction command per policy tick.
            if len(rows) != 1 or not isinstance(rows[0], list) or len(rows[0]) < 3:
                labels.append(TiLabel(room_actor.get(room_name), None, rejection="multi_or_malformed_site"))
            else:
                structure_type, x, y = rows[0][:3]
                ai = room_actor.get(room_name)
                if ai is None:
                    labels.append(TiLabel(None, None, rejection="room_actor_truncated"))
                elif structure_type not in CONSTRUCTION_TYPES:
                    labels.append(TiLabel(ai, None, rejection="unsupported_construction_type"))
                elif not all(isinstance(v, int) and 0 <= v < 50 for v in (x, y)):
                    labels.append(TiLabel(ai, None, rejection="invalid_construction_tile"))
                else:
                    labels.append(TiLabel(
                        ai, "createConstructionSite",
                        construction_type=CONSTRUCTION_TYPES.index(structure_type),
                        construction_tile=y * 50 + x,
                        full_action=True,
                    ))
        objects = room_payload.get("object") or {}
        if not isinstance(objects, dict):
            labels.append(TiLabel(None, None, rejection="malformed_objects"))
            continue
        for object_id, intents in objects.items():
            ai = actor_index.get(str(object_id))
            if ai is None:
                labels.append(TiLabel(None, None, rejection="actor_truncated"))
                continue
            if not isinstance(intents, dict) or len(intents) != 1:
                labels.append(TiLabel(ai, None, rejection="concurrent_or_malformed_intents"))
                continue
            intent, raw_args = next(iter(intents.items()))
            try:
                args = _arguments(raw_args)
            except ValueError:
                labels.append(TiLabel(ai, None, rejection="malformed_arguments"))
                continue
            if intent == "spawn":
                body = args[0] if args else None
                if not isinstance(body, list) or not 1 <= len(body) <= MAX_BODY_PARTS:
                    labels.append(TiLabel(ai, None, rejection="invalid_spawn_body"))
                    continue
                try:
                    tokens = tuple(BODY_PART_TYPES.index(str(part)) for part in body)
                except ValueError:
                    labels.append(TiLabel(ai, None, rejection="unsupported_body_part"))
                    continue
                # The type choice is exact: TI's `spawn` IS our `spawnCreep`
                # macro action.  Its remaining arguments -- creep name, spawn
                # direction, dry run, and the energy-structure order to drain --
                # are executor and engine concerns rather than action factors of
                # this ABI, so withholding the type would claim we do not know
                # what the expert did, which is false.  The body itself is
                # supervised separately through exact counts plus, when the raw
                # sequence is contiguous type blocks, its order.
                labels.append(TiLabel(ai, "spawnCreep", body_parts=tokens, full_action=True))
                continue
            if intent == "move":
                direction = args[0] if args else None
                if isinstance(direction, int) and 1 <= direction <= 8:
                    labels.append(TiLabel(ai, "move", direction=direction - 1, full_action=True))
                else:
                    labels.append(TiLabel(ai, None, rejection="unsupported_move_target"))
                continue
            if intent in _DIRECT_TARGET_INTENTS:
                ti = target_index.get(str(args[0])) if args else None
                if ti is None:
                    labels.append(TiLabel(ai, None, rejection="target_truncated"))
                elif intent not in INTENT_TYPES:
                    labels.append(TiLabel(ai, None, rejection="unsupported_intent"))
                else:
                    labels.append(TiLabel(
                        ai, intent, target_index=ti, target_id=str(args[0]),
                        full_action=True,
                    ))
                continue
            if intent in _AMOUNT_TARGET_INTENTS:
                ti = target_index.get(str(args[0])) if args else None
                resource = args[1] if len(args) > 1 else None
                bin_index = _amount_bin(args[2] if len(args) > 2 else None)
                if ti is None:
                    labels.append(TiLabel(ai, None, rejection="target_truncated"))
                elif resource != _ENERGY_RESOURCE:
                    labels.append(
                        TiLabel(ai, None, rejection=f"unsupported_resource:{resource}"),
                    )
                elif bin_index is None:
                    labels.append(
                        TiLabel(ai, None, rejection=f"unrepresentable_amount:{intent}"),
                    )
                else:
                    labels.append(TiLabel(
                        ai, intent, target_index=ti, target_id=str(args[0]),
                        amount_index=bin_index, full_action=True,
                    ))
                continue
            labels.append(TiLabel(ai, None, rejection=f"unsupported:{intent}"))
    return labels


def summarize_ti_labels(labels: list[TiLabel]) -> dict[str, Any]:
    accepted = Counter(
        label.intent for label in labels if label.intent is not None and label.full_action
    )
    factor_only = Counter(
        label.intent for label in labels if label.intent is not None and not label.full_action
    )
    rejected = Counter(label.rejection for label in labels if label.rejection)
    return {
        "full": dict(sorted(accepted.items())),
        "factor_only": dict(sorted(factor_only.items())),
        "rejected": dict(sorted(rejected.items())),
    }
