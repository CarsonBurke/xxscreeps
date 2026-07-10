# Apex v3

Delegated multi-room empire bot for **xxscreeps**.

## Install

```bash
npx xxscreeps manage bot add apex-v3 samples/bots/apex-v3 --spawn W5N5
```

## Role delegation (economy)

| Role | Behavior |
|------|----------|
| **harvester** | Sits on a source; harvests; **drops** energy or fills adjacent **container** (never hauls home) |
| **hauler** | Picks drops/containers at sources; delivers to storage / spawn-side container / spawn network |
| **filler** | Parks on reserved **filler pads** next to spawn; fills spawn/extensions/towers from nearby energy |
| **upgrader** | Parks at controller; only takes energy from **nearby** containers/drops (haulers feed them) |
| **builder** | Builds/repairs from containers/drops/storage — **does not harvest** (except pioneers) |
| **bootstrap** | Early multi-skill only until capacity ≥ `specializeCapacity` |

## Heuristic empire (`empire.js`)

Dynamic scoring every ~10 ticks into `Memory.empire.plan`:

- **Colony strength** — RCL, energy capacity, storage, towers, creeps, threat
- **Expand ASAP** — if free GCL ≥ 1, pick claim targets (prefer 2-source, adjacent, low threat)
- **Support assignment** — strongest nearby colonies send claimers / pioneers / escorts (multi-home)
- **Remote budget** — remotes scale with strength (not a fixed RCL7 gate)
- **Modes** — `bootstrap` | `grow` | `expand` per colony

## Base planner (`planner.js`) — utility scoring, not a fixed ladder

Construction is **not** a hard-coded “containers → roads → storage” list.

Each pass the planner:

1. **Analyzes bottlenecks** — spawn capacity, missing logistics, defense pressure, storage readiness, site backlog  
2. **Enumerates candidates** — any legal job we could place now  
3. **Scores** with soft factors: bottleneck relief, leverage, readiness penalties, saturation (diminishing returns), backlog  
4. **Places** highest scores until a site budget is used  

So extensions can beat roads when spawn energy is the binding constraint; source containers beat storage when miners drop on open ground; storage only scores well once logistics can feed it. Remotes use the same idea. Filler pads: reserved walkable tiles beside spawn.

## Optional modules

| File | Purpose |
|------|---------|
| `war.js` + `combat.js` | Campaigns for remotes/claims, abandon-on-fail |
| `economy.js` | Finer remote upkeep / income projections |

Core bot runs without them.

## Flags

| Prefix | Effect |
|--------|--------|
| `attack_*` | War / siege target |
| `claim_*` | Claim expansion |
| `remote_*` | Force remote mining |
| `ignore_*` | Never remote there |
| `rally_*` | Military rally |
| `defend_*` | Priority defense room |

## Metrics

Segment **87** → TensorBoard (see [../PROTOCOL.md](../PROTOCOL.md)).

```bash
mise exec node@24 -- node --import xxscreeps/loader \
  samples/bots/apex/bench.mjs 5000 ../apex-v3
```

## Docs

- [WAR.md](./WAR.md) — campaigns / abandon (when present)
- [ECONOMY.md](./ECONOMY.md) — projection formulas (when present)
