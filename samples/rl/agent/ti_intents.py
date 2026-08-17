"""Exact, conservative translation of The International's raw intents.

The expert can submit several engine intents for one creep in a tick, while the
current policy has one macro slot.  This module never guesses: it emits a full
label only when the submitted command is exactly representable, and preserves a
raw spawn body when extra spawn arguments are outside the policy ABI. Downstream
training always uses its exact counts and uses its order only when the sequence
is representable as contiguous type blocks. Everything else carries an explicit rejection
reason and remains useful for critic/representation training.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .constants import (
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


@dataclass(frozen=True)
class TiLabel:
    actor_index: int | None
    intent: str | None
    target_index: int | None = None
    direction: int | None = None
    construction_type: int | None = None
    construction_tile: int | None = None
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
                # TI supplies energy structure order and spawn directions that
                # our ABI does not expose. Preserve the raw body so downstream
                # code can supervise exact counts and only a representable block
                # order; the complete spawn command is deliberately not exact.
                labels.append(TiLabel(ai, "spawnCreep", body_parts=tokens, full_action=False))
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
                    labels.append(TiLabel(ai, intent, target_index=ti, full_action=True))
                continue
            # Amount-bearing logistics require an exact amount-bin and resource
            # semantic proof; add them only with that pre-state contract.
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
