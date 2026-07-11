"""Load shared schema (samples/rl/schema.json)."""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = _ROOT / "schema.json"

with SCHEMA_PATH.open(encoding="utf-8") as f:
    SCHEMA = json.load(f)

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
AMOUNT_BINS = SCHEMA["amountBins"]
N_AMOUNT = len(AMOUNT_BINS)
INTENT_TYPES = SCHEMA["intentTypes"]
N_INTENT = len(INTENT_TYPES)
N_DIR = 8
PATCH_FLAT = PATCH_SIZE * PATCH_SIZE * TILE_FEAT

MODEL_CFG = SCHEMA["model"]
PPO_CFG = SCHEMA["ppo"]
