# Apex v4 (TypeScript)

Phase-aware multi-room empire bot for **xxscreeps** (package path still `apex-v3`).

## Build

```bash
cd samples/bots/apex-v3 && npx tsc -p tsconfig.json
```

Install from **`dist/`** (flat CommonJS):

```bash
npx xxscreeps manage bot add apex-v3 samples/bots/apex-v3/dist --spawn W5N5
```

## RCL phases (v4)

| RCL | Mode | Behavior |
|-----|------|----------|
| 1 | `rcl_push` | Bootstraps fill spawn then **upgrade**; planner places **nothing** |
| 2 | `infra` | Extensions first; upgraders self-feed; no roads yet |
| 3+ | `grow` | Containers, remotes (min RCL3), roads (min RCL3) |
| 4+ | `expand` | Claim/expand gated (`expandMinRcl`) |

```bash
# 20k sim from repo root (Node 24+)
mise exec node@24 -- node --import xxscreeps/loader \
  samples/bots/apex/bench.mjs 20000 samples/bots/apex-v3/dist
```

## Memory (char-cost aware)

Screeps serializes Memory by character count. Apex v3 follows The International:

| Technique | Where |
|-----------|--------|
| **Numeric enums** as object keys | `CreepMem`, `RoomApexMem` (`memoryKeys.ts`) |
| **Short string keys** for metrics | `MetricKey` (`t`,`r`,`c`,`h`,…) in segment 87 |
| **Coords only** in Memory | `{x,y}` via `codec.ts` — never store `RoomPosition` |
| **packId / packCoord** | Optional denser packing for IDs & tiles |
| **Memhack** | `memoryHack.ts` — reuse Memory object across ticks |

Positions example:

```ts
// store
memory.seatCoord = asCoord(creep.pos); // {x, y}
// use
const pos = asPos(memory.seatCoord, room.name);
```

## Architecture

- **Roles** — harvester / hauler / filler / upgrader / builder (strict delegation)
- **empire.ts** — GCL-first expansion, colony strength heuristics
- **planner.ts** — utility-scored construction (not a fixed structure ladder)
- **war.ts / combat.ts** — campaigns + abandon
- **economy.ts / projections.ts** — remote affordability math
- **metrics.ts** — dense segment samples for TensorBoard watcher

See `WAR.md`, `ECONOMY.md`, `../PROTOCOL.md`.
