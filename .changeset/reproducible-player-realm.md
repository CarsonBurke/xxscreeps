---
'xxscreeps': minor
---

Make a `deterministicCpu` run reproducible end to end. The player realm now gets a virtual wall clock (`Date.now`, `performance`-free WASI `clock_time_get`, and fixed heap statistics) instead of host time, room name sets are sorted before they reach `Game.rooms`, and the sandbox's abort timeout (`runner.cpu.runawayTimeout`) is separated from the billed `tickLimit` so host load cannot decide whether a tick's intents survive. Skipped ticks, dropped ticks, sandbox restarts, and intents recorded against rooms that were not loaded are reported instead of silently changing the replay; a dropped tick in a reproducible run now fails.
