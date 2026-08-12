"""Load shared schema (samples/rl/schema.json)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = _ROOT / "schema.json"

SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _semantic_schema(value):
    """Drop documentation-only keys before canonical contract hashing."""
    if isinstance(value, dict):
        return {
            key: _semantic_schema(item)
            for key, item in sorted(value.items())
            if not key.startswith("_")
        }
    if isinstance(value, list):
        return [_semantic_schema(item) for item in value]
    return value


_SCHEMA_CANONICAL = json.dumps(
    _semantic_schema(SCHEMA), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
).encode("utf-8")
SCHEMA_SHA256 = hashlib.sha256(_SCHEMA_CANONICAL).hexdigest()

ROOM_SIZE = SCHEMA["roomSize"]
PATCH_SIZE = SCHEMA["patchSize"]
PATCHES_PER_SIDE = SCHEMA["patchesPerSide"]
PATCHES_PER_ROOM = SCHEMA["patchesPerRoom"]
MAX_ROOMS = SCHEMA["maxRooms"]
MAX_ACTORS = SCHEMA["maxActors"]
MAX_TARGETS = SCHEMA["maxTargets"]
INTENT_SLOTS = SCHEMA["intentSlots"]
TILE_FEAT = SCHEMA["tileFeat"]
ACTOR_FEAT = SCHEMA["actorFeat"]
TARGET_FEAT = SCHEMA["targetFeat"]
GLOBAL_FEAT = SCHEMA["globalFeat"]
AMOUNT_BINS = SCHEMA["amountBins"]
N_AMOUNT = len(AMOUNT_BINS)
INTENT_TYPES = SCHEMA["intentTypes"]
INTENT_SPECS = SCHEMA["intentSpecs"]
N_INTENT = len(INTENT_TYPES)
N_DIR = 8
PATCH_FLAT = PATCH_SIZE * PATCH_SIZE * TILE_FEAT

MODEL_CFG = SCHEMA["model"]
PPO_CFG = SCHEMA["ppo"]
