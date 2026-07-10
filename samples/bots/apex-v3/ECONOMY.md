# Apex v3 Economy Projections

Projection-only module set for multi-room Screeps (xxscreeps). No role AI — math + optional `Game.map` distances, then a Memory snapshot for spawn/combat.

| File | Role |
|------|------|
| [`projections.js`](./projections.js) | Pure formulas (source, harvester, hauler, remote package, afford, staffing) |
| [`economy.js`](./economy.js) | Public API, live-room helpers, `tick()` → `Memory.empire.economy` |

## Design (v3 roles)

| Role | Job |
|------|-----|
| **harvester** | Static at source (drop / container). 5 WORK mines 10 e/t. |
| **hauler** | Source ↔ storage/spawn pipeline. |
| **filler** | Spawn + extensions (1+ when extensions exist). |
| **upgrader** | Controller energy from surplus. |
| **builder** | Construction / walls from containers & storage. |
| **remote\*** | Same mining model + reserver CLAIM upkeep. |

## Formulas

### 1. Source throughput

```
ePerTick = energyCapacity / ENERGY_REGEN_TIME
```

| Source | Capacity | Regen | e/t |
|--------|----------|-------|-----|
| Normal owned / neutral | 3000 | 300 | **10** |
| Source Keeper (SK) | ~4000 | 300 | **≈13.33** (higher, riskier — combat upkeep not free) |

`estimateSource(sourceOrOpts)` accepts a path length number, a Source-like object, or `{ pathLen, sk, energyCapacity }`.

### 2. Static harvester body

```
workParts  = 5                    # ceil(10 e/t / 2 harvest power)
carryParts = 1                    # seed container / drop
moveParts  = f(pathLen, roads)    # enough to reach seat once, then park
cost       = Σ BODYPART_COST[part]
spawnTime  = parts * CREEP_SPAWN_TIME   # 3 ticks/part
upkeepEt   = cost / CREEP_LIFE_TIME     # 1500 ticks
spawnBusyFrac = spawnTime / lifetime
```

Typical local roads body: `5W 1C 3M` (or `5W 1C 1M` if seat is adjacent).  
Remote off-road: full MOVE so the miner is not stranded.

### 3. Hauler demand (route)

Steady pipeline home ↔ source:

```
roundTrip       = 2 * pathLen
energyInFlight  = roundTrip * ePerTick
carryPartsNeed  = ceil(energyInFlight / 50)    # 50 energy per CARRY
```

Body packing:

- **Roads:** ~2 CARRY : 1 MOVE, max ~32 CARRY / creep (`MAX_CREEP_SIZE` 50).
- **Off-road:** 1 CARRY : 1 MOVE.

If `carryPartsNeed` exceeds one creep’s max, split into **N haulers**:

```
haulerCount    = ceil(carryPartsNeed / maxCarryPerCreep)
carryPerHauler = ceil(carryPartsNeed / haulerCount)
```

```
upkeepEt       = totalBodyCost / 1500
spawnBusyFrac  = totalSpawnTime / 1500
deliveryEt     ≈ min(ePerTick, capacity / roundTrip)
```

### 4. Remote room package

For one remote with sources `[{ id, pathLen }]`:

| Component | Model |
|-----------|--------|
| **Reserver** | `CLAIM×n + MOVE×n`, lifetime **600**. `upkeepEt = cost / 600`. Default 1 CLAIM. |
| **Harvester** | 1 per source (static body, often off-road MOVE). |
| **Haulers** | `estimateHauler(pathLen, ePerTick)` per source. |
| **Delivered** | `Σ ePerTick * (1 - travelWaste)` (default waste 5%). |
| **Net** | `deliveredEt - reserverUpkeep - harvestUpkeep - haulUpkeep`. |
| **Spawn busy %** | `Σ (spawnTime_i / life_i) * 100` for **one** spawn. |

Path length when unknown:

```
pathLen ≈ Game.map.findRoute length * 50 + 20
       or linearRoomDistance * 50 + 20
```

### 5. How many remotes can we afford?

`affordRemotes({ colonyIncomeEt, spawnCount, rcl, freeCpu?, candidatePackages })`

Greedy best-net packages until a bottleneck:

| Reason | Condition |
|--------|-----------|
| `rcl` | RCL &lt; 3 |
| `spawn` | package `spawnBusyFrac` &gt; remaining spawn headroom |
| `energy` | not enough local surplus to float package upkeep |
| `cpu` | optional free CPU exhausted (~1 ms/remote rough) |
| `hauler` | absurd creep count on path (skip candidate) |
| `cap` | soft max remotes (default 6) |

### 6. Filler / upgrade / build staffing

| Role | Rule |
|------|------|
| **filler** | 1 when extensions exist (2 if ext ≥ 30). Carry scales with extension count. |
| **upgrader** | ~60% of post-upkeep surplus as upgrade budget; always ≥1 before RCL8. RCL8: 1–2 for 15 WORK cap. |
| **builder** | From construction site count; capped if surplus thin. |
| **militaryBudgetEt** | `staffedSurplus * 0.35` — energy/tick that can go to war without starving economy. |

## API

```js
const economy = require('economy');

// Point estimates
economy.estimateSource(sourceOrPathLen);
economy.estimateHauler(pathLen, ePerTick);
economy.estimateRemotePackage(homeRoomName, remoteRoomName, sources);
// sources: [{ id, pathLen, sk? }]

// Full colony projection
const p = economy.projectColony(room, remotesIntel);
// {
//   incomeEt, upkeepEt, netEt, maxRemotes, packages[],
//   spawnBusyPct, militaryBudgetEt,
//   recommendations: [{ role, count, reason }],
//   afford, local, staffing, ...
// }

// Empire write (spawn/combat read this)
economy.tick(colonies);
// Memory.empire.economy = {
//   tick, incomeEt, maxRemotes, remotes: [...],
//   spawnBusyPct, militaryBudgetEt,
//   colonies: { [roomName]: {...} }
// }

economy.last();                      // last snapshot
economy.militaryBudgetFor(roomName);
economy.SPAWN_PRIORITY;              // shared priority list
economy.CONST;                       // tunables
```

### `tick(colonies)` input shapes

```js
economy.tick();                          // all owned rooms, remotes from Memory
economy.tick([ room1, room2 ]);
economy.tick([ { room, remotesIntel } ]);
economy.tick(rooms, { remotesIntel });   // default remotes for all
```

`remotesIntel`: `string[]`, `{ roomName, sources }[]`, or `{ [roomName]: { sources } }`.

## Memory contract

```js
Memory.empire.economy = {
  tick: number,
  incomeEt: number,           // empire gross delivered e/t
  maxRemotes: number,         // sum of affordable remotes
  remotes: [{
    home, remote, deliveredEt, upkeepEt, netEt,
    spawnBusyPct, sources, haulers
  }],
  spawnBusyPct: number,       // busiest colony (one-spawn %)
  militaryBudgetEt: number,   // surplus * factor (empire sum)
  colonies: {
    [roomName]: {
      incomeEt, upkeepEt, netEt, maxRemotes,
      spawnBusyPct, militaryBudgetEt,
      recommendations, affordReason
    }
  }
};
```

## Spawn priority export

```js
economy.SPAWN_PRIORITY
// bootstrap, defender, harvester, filler, hauler,
// remoteHarvester, remoteHauler, reserver,
// upgrader, builder, repairer, scout, claimer,
// attacker, ranged, healer, dismantler
```

## Integration notes

- Call `economy.tick(colonies)` once per tick after intel / remote selection, before spawn planning.
- Spawn queue reads `recommendations` + `SPAWN_PRIORITY`; combat spends ≤ `militaryBudgetEt`.
- No TensorBoard / metrics dependency.
- Flat CommonJS only (`require('economy')`, `require('projections')`).

## Worked example

Dual-source home, pathLens 12 and 18, one remote one room away (path ~70) with 2 sources:

| Stream | Gross e/t | Rough upkeep e/t |
|--------|-----------|------------------|
| Local 2×10 | 20 | harvesters + short haulers ≈ 1–2 |
| Remote 2×10 × 0.95 waste | 19 | reserver + 2 harvesters + long haulers ≈ 4–8 |
| **Net** | | often **+25–30 e/t** if spawn can replace the package |

Hauler for pathLen 70 @ 10 e/t:

```
energyInFlight = 2 * 70 * 10 = 1400
carryPartsNeed = ceil(1400 / 50) = 28
→ one roads hauler with 28C + 14M (42 parts), or split if off-road
```
