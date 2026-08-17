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
MAX_BODY_PARTS = SCHEMA["maxBodyParts"]
MAX_ROOM_ENERGY = SCHEMA["maxRoomEnergy"]
INTENT_SLOTS = SCHEMA["intentSlots"]
TILE_FEAT = SCHEMA["tileFeat"]
ACTOR_FEAT = SCHEMA["actorFeat"]
ACTOR_FEATURES = SCHEMA["actorFeatures"]
if len(ACTOR_FEATURES) != ACTOR_FEAT or len(set(ACTOR_FEATURES)) != ACTOR_FEAT:
    raise RuntimeError("schema actorFeatures must uniquely name every actor feature")
ACTOR_FEATURE_INDEX = {name: index for index, name in enumerate(ACTOR_FEATURES)}
TARGET_FEAT = SCHEMA["targetFeat"]
GLOBAL_FEAT = SCHEMA["globalFeat"]
AMOUNT_BINS = SCHEMA["amountBins"]
N_AMOUNT = len(AMOUNT_BINS)
BODY_PART_TYPES = SCHEMA["bodyPartTypes"]
BODY_PART_COSTS = SCHEMA["bodyPartCosts"]
N_BODY_PART = len(BODY_PART_TYPES)
ACTION_OUTCOMES = SCHEMA["actionOutcomes"]
N_ACTION_OUTCOME = len(ACTION_OUTCOMES)
CONSTRUCTION_TYPES = SCHEMA["constructionTypes"]
N_CONSTRUCTION_TYPE = len(CONSTRUCTION_TYPES)
N_CONSTRUCTION_TILE = ROOM_SIZE * ROOM_SIZE
CONSTRUCTION_MASK_BYTES = (N_CONSTRUCTION_TILE + 7) // 8
INTENT_TYPES = SCHEMA["intentTypes"]
INTENT_SPECS = SCHEMA["intentSpecs"]
N_INTENT = len(INTENT_TYPES)
N_DIR = 8
PATCH_FLAT = PATCH_SIZE * PATCH_SIZE * TILE_FEAT

MODEL_CFG = SCHEMA["model"]
NEXTLAT_CFG = SCHEMA["nextLat"]
VALUE_CFG = SCHEMA["value"]
PPO_CFG = SCHEMA["ppo"]
