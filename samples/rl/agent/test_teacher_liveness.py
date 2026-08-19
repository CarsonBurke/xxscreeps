"""Engine-backed liveness of the configured teacher, on every curriculum.

Collection already fails closed on a silent teacher, but it fails after hours of
work and only on the curricula its environments happened to draw. This is the
cheap version of the same check, and it exists because a teacher configuration
change caused exactly this failure: bounding scout targets threw inside the bot
for creeps whose commune the bot had not recorded yet - which is every creep a
seeded scenario places - and The International aborts its entire tick on an
exception, so the observable symptom was silence rather than an error.

Thresholds match `TiLivenessGate`: a healthy episode's longest silence is about
12 ticks while the first creep spawns, and 30 consecutive silent ticks is the
documented failure line.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from .env_client import ScreepsEnv

_CURRICULA = ("empty", "seed_creep", "seed_full", "seed_claimer", "seed_outpost")
_TICKS = 100
_MAX_SILENT_RUN = 30


class TeacherLivenessEngineTest(unittest.TestCase):
    def test_configured_teacher_acts_on_every_curriculum(self) -> None:
        for curriculum in _CURRICULA:
            with self.subTest(curriculum=curriculum), patch.dict(
                os.environ, {"RL_OBS_FMT": "bin"},
            ):
                env = ScreepsEnv(
                    max_episode=_TICKS + 40, curriculum=curriculum, lean_meta=False,
                    seed=10, expert=True, capture_expert_intents=True,
                )
                worst = 0
                run = 0
                acting = 0
                try:
                    env.reset()
                    for _ in range(_TICKS):
                        _obs, _reward, _done, info = env.step()
                        if info.get("expertIntents"):
                            acting += 1
                            run = 0
                        else:
                            run += 1
                            worst = max(worst, run)
                finally:
                    env.close()

                self.assertLess(
                    worst, _MAX_SILENT_RUN,
                    f"{curriculum}: teacher went {worst} consecutive ticks without "
                    "an engine intent; an exception aborts its whole tick",
                )
                # Startup silence is expected; a teacher that only ever acts a
                # handful of times is stalled even without a long silent run.
                self.assertGreater(acting, _TICKS // 2, curriculum)


if __name__ == "__main__":
    unittest.main()
