from __future__ import annotations

import unittest

from .constants import BODY_PART_TYPES, CONSTRUCTION_TYPES
from .ti_intents import translate_ti_intents


class TiIntentTranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actors = [
            {"id": "spawn", "kind": "structure", "room": "W7N3"},
            {"id": "creep", "kind": "creep", "room": "W7N3"},
            {"id": "room:W7N3", "kind": "room", "room": "W7N3"},
        ]
        self.targets = [{"id": "source", "kind": "source", "room": "W7N3"}]

    def test_spawn_body_is_exact_factor_only(self) -> None:
        labels = translate_ti_intents({"W7N3": {"local": {}, "object": {
            "spawn": {"spawn": [["work", "carry", "move"], "name", ["spawn"], [1]]},
        }}}, self.actors, self.targets, ["W7N3"])
        self.assertEqual(labels[0].intent, "spawnCreep")
        self.assertEqual(
            labels[0].body_parts,
            tuple(BODY_PART_TYPES.index(part) for part in ("work", "carry", "move")),
        )
        self.assertFalse(labels[0].full_action)

    def test_direct_move_and_target_are_full_labels(self) -> None:
        move = translate_ti_intents({"W7N3": {"local": {}, "object": {
            "creep": {"move": [8]},
        }}}, self.actors, self.targets, ["W7N3"])[0]
        self.assertTrue(move.full_action)
        self.assertEqual((move.intent, move.direction), ("move", 7))
        harvest = translate_ti_intents({"W7N3": {"local": {}, "object": {
            "creep": {"harvest": ["source"]},
        }}}, self.actors, self.targets, ["W7N3"])[0]
        self.assertTrue(harvest.full_action)
        self.assertEqual((harvest.intent, harvest.target_index), ("harvest", 0))

    def test_concurrent_intents_are_rejected(self) -> None:
        label = translate_ti_intents({"W7N3": {"local": {}, "object": {
            "creep": {"move": [1], "harvest": ["source"]},
        }}}, self.actors, self.targets, ["W7N3"])[0]
        self.assertEqual(label.rejection, "concurrent_or_malformed_intents")

    def test_single_construction_site_is_exact(self) -> None:
        label = translate_ti_intents({"W7N3": {
            "local": {"createConstructionSite": [["extension", 4, 7, None]]},
            "object": {},
        }}, self.actors, self.targets, ["W7N3"])[0]
        self.assertTrue(label.full_action)
        self.assertEqual(label.construction_type, CONSTRUCTION_TYPES.index("extension"))
        self.assertEqual(label.construction_tile, 7 * 50 + 4)


if __name__ == "__main__":
    unittest.main()
