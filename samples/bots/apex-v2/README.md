# Apex v2

A multi-room Screeps empire bot for **xxscreeps** (compatible with vanilla private servers).

This is a significant evolution of [Apex v1](../apex/) with better traffic, economy, spawning, remotes, combat, construction, observability, and CPU robustness. **v1 under `samples/bots/apex/` is unchanged.**

## Metrics / TensorBoard

Same as v1: `metrics.js` writes **RawMemory segment 87** each tick (RCL, energy, control points, CPU, …). Host watcher → TensorBoard; no TB dependency in the bot. See [../PROTOCOL.md](../PROTOCOL.md).

```bash
mise exec node@24 -- node --import xxscreeps/loader samples/bots/apex/bench.mjs 20000 ../apex-v2
python3 samples/bots/metrics-watcher/watch.py \
  --jsonl samples/bots/apex/runs/latest/metrics.jsonl \
  --logdir samples/bots/apex/runs/latest/tb
```

## Install

```bash
npx xxscreeps manage bot add apex-v2 samples/bots/apex-v2 --spawn W5N5
```

Update code later:

```bash
npx xxscreeps manage bot update apex-v2 samples/bots/apex-v2
```

`<dir>` must be the **flat module directory** (this folder). No build step — plain CommonJS.

## Architecture

```text
main.js          kernel: mem hygiene → intel → colonies → creeps → stats
config.js        tunables (v2: hysteresis, FSM, heat, bucket, …)
util.js          helpers, safeGet, path caches, TTL / bucket
traffic.js       CostMatrix cache, road preference, parking, goDo/moveToRoom
body.js          scaled body factories
intel.js         room memory, remote + expansion scoring
remote.js        remote room FSM (scout→reserve→container→mine→haul)
room.js          per-colony loop
spawn.js         priority queue, demand model, name registry, hysteresis
construction.js  RCL-gated planner, site budget, path-heat roads
logistics.js     link-first network + terminal balance + hauler need
defense.js       towers (rampart-aware), safe mode
combat.js        defenders (recycle), rally-then-march squads, healers
roles.js         all creep brains
stats.js         Memory.stats + console summary
```

## v2 vs v1 (changelog)

### Traffic / movement
- **CostMatrix caching** per room, rebuilt every `costMatrixRefresh` ticks; roads preferred.
- **Hauler/miner parking**: miners sit on container seats; haulers park off the seat.
- **`reusePath` tuned by role** via `config.pathReuseByRole`.

### Economy
- **Link-first logistics** when links exist (source → controller/storage).
- **Hauler assignment by source need** (container fullness / drops).
- **Dead-creep replacement** from `ticksToLive` + spawn-time ETA (not a fixed 80 only).
- **Bootstrap cap** (`maxBootstrapWorkers`); clean transition at `specializeCapacity`.
- **Drop mining only until container/link**; no perpetual drop waste.

### Spawning
- **Single priority queue** with **energy-wait hysteresis** (`spawnEnergyHysteresis`) — avoids tiny bodies when a full body is almost affordable.
- **Per-source demand model**: hauler count/size from path length × throughput (+ container fullness).
- **Name registry** (`Memory.apex.nameSeq`) and **no duplicate source claims**.

### Multi-room
- **Remote FSM**: `scout → reserve → container → mine → haul`; **abandon** on sustained threat, re-scout after cooldown.
- **Room-to-room path length cache** (`Memory.apex.pathCache`).
- **Expansion scoring**: prefers 2-source rooms; skips SK/highway.

### Combat
- **Tower** heal/attack with **rampart-aware** focus (prefer exposed hostiles).
- **Defenders only under threat**; **recycle** at spawn after `defenderRecycleSafeTicks` safe.
- **Attack squads rally** (flag `rally_*` or home spawn) **before march**; healers stick to **lowest-HP** squad ally.

### Construction
- **Fewer road sites**: path **heat** map + small critical-path seed; `maxRoadSites`.
- **Finish existing sites** of a type before spamming more (`finishSitesBeforeMore`).
- **RCL-gated planner** with `siteBudget` / `sitesPerPass`.

### Observability
- **`Memory.stats`**: `energyHarvested`, `upgradePoints`, `buildEnergy`, `spawnEnergy` (role event hooks).
- **Periodic console summary** every `consoleSummaryInterval` ticks.

### Robustness
- **`safeGet` / `Game.getObjectById` null guards** throughout roles.
- **CPU bucket awareness**: skip new remotes/scouts/expansion when `bucket < lowBucketThreshold`.
- Misc v1 fixes: clearer miner rebind, unique names, cache pruning, remote phase gating.

## Colony lifecycle

| Phase | Behavior |
|--------|----------|
| **Bootstrap** | Capped multi-role workers until capacity ≥ `specializeCapacity` |
| **Establish** | Static miner per source, demand-sized haulers, containers |
| **Grow** | Upgraders, builders, repairers, towers, storage/links |
| **Remote FSM** | Scout → reserve → container → mine → haul; abandon on threat |
| **Expand** | Claimer on `claim_*` or auto (2-source preference); pioneers |
| **War** | Towers + defenders; `attack_*` squads rally then march |

## Flag orders

| Flag name prefix | Effect |
|------------------|--------|
| `attack_…` | Siege squad → that room (rallies first) |
| `claim_…` | Send claimer |
| `remote_…` | Force remote mining |
| `ignore_…` | Never auto-remote / expand |
| `rally_…` | Squad rally point |

## Tuning

Edit `config.js` before uploading. Important v2 knobs:

| Knob | Meaning |
|------|---------|
| `spawnEnergyHysteresis` | Wait for ~full body before spawn |
| `costMatrixRefresh` | CostMatrix rebuild interval |
| `pathReuseByRole` | Path reuse ticks by role |
| `roadHeatThreshold` | Visits before road site |
| `siteBudget` / `sitesPerPass` | Construction caps |
| `remoteAbandonThreatTicks` | Sustained threat → abandon |
| `lowBucketThreshold` | Skip remotes/scouts when low |
| `defenderRecycleSafeTicks` | Recycle idle defenders |
| `maxBootstrapWorkers` | Cap early workers |
| `haulerCarryPerPathTile` | Economy throughput |

## Roles

Same roles as v1: `bootstrap`, `miner`/`remoteMiner`, `hauler`/`remoteHauler`, `upgrader`, `builder`/pioneer, `repairer`, `reserver`, `claimer`, `scout`, `mineralMiner`, `defender`, `attacker`/`ranged`/`healer`/`dismantler`.

## What Apex does *not* rely on

- Market order book — only `terminal.send` balancing  
- Power creeps  
- Intershard resources / pixels  

This is a **sample empire**, not a season-winning bot. It stresses xxscreeps APIs and demonstrates multi-room ops with stronger logistics and CPU hygiene than v1.
