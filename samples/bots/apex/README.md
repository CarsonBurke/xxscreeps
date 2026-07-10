# Apex

A full multi-room Screeps empire bot for **xxscreeps** (compatible with vanilla private servers).

Apex is intentionally ambitious: static mining, remote harvesting, intelligent spawn priorities, bunker-lite construction, tower defense, flag-driven combat and expansion, links/terminals, and new-colony pioneers.

## Metrics / TensorBoard

Each tick the bot writes empire stats (**RCL**, control points, energy harvested/build/creep costs, CPU, …) to **RawMemory segment 87**. That is host-readable without putting TensorBoard inside the bot.

See [../PROTOCOL.md](../PROTOCOL.md) and [../metrics-watcher/README.md](../metrics-watcher/README.md).

```bash
# Sim + JSONL dump
mise exec node@24 -- node --import xxscreeps/loader samples/bots/apex/bench.mjs 20000
# TensorBoard event files
python3 samples/bots/metrics-watcher/watch.py \
  --jsonl samples/bots/apex/runs/latest/metrics.jsonl \
  --logdir samples/bots/apex/runs/latest/tb
tensorboard --logdir samples/bots/apex/runs/latest/tb
```

## Install

From an xxscreeps workspace (server may be running):

```bash
npx xxscreeps manage bot add apex samples/bots/apex --spawn W5N5
```

Update code later:

```bash
npx xxscreeps manage bot update apex samples/bots/apex
```

Replace every imported bot’s code (including launcher AliceBot/EmmaBot/…):

```bash
npx xxscreeps import --overwrite-code samples/bots/apex
```

`<dir>` must be the **flat module directory** (this folder). No build step — plain CommonJS.

## Architecture

```text
main.js          kernel: mem hygiene → intel → colonies → creeps
config.js        tunables (RCL thresholds, remote caps, squad sizes, …)
util.js          pathing, energy helpers, map math
body.js          scaled body factories (miner/hauler/combat/…)
intel.js         room memory, remote scoring, flag orders
room.js          per-colony loop + remote list
spawn.js         priority spawn queue
construction.js  extensions, towers, storage cluster, containers, roads, ramparts
logistics.js     link network + terminal energy balance
defense.js       towers, safe mode
combat.js        defenders + attack squads
roles.js         all creep brains (economy + military)
```

### Colony lifecycle

| Phase | Behavior |
|--------|----------|
| **Bootstrap** (RCL 1–2 / low energy capacity) | Multi-role `bootstrap` workers until specialization is affordable |
| **Establish** | Static `miner` per source, road-scaled `hauler`, container infrastructure |
| **Grow** | Upgraders (storage-fed), builders, repairers, towers, storage/links |
| **Remote** | Adjacent (and RCL6+ depth-2) rooms scored for sources; `reserver` + `remoteMiner` + `remoteHauler` |
| **Expand** | `claimer` on `claim_*` flags or auto when GCL/storage allow; `pioneer` builders place first spawn |
| **War** | Towers + `defender`; `attack_*` flags spawn attacker/healer/ranged squads |

### Mining math

- Source: 3000 energy / 300 ticks ≈ **10 e/t** → **5 WORK** static miner (configurable).
- Hauler carry ≈ `2 × pathLength` energy so the pipeline stays full on roads (2 CARRY : 1 MOVE).
- Remotes: same miner pattern; hauler count doubles on long paths.

### Construction (bunker-lite)

- Extensions on a checkerboard around the first spawn; roads on the alternate lattice.
- Towers offset from spawn for coverage.
- Storage / terminal / storage-link cluster near spawn.
- Containers at sources + controller; source links from RCL 5.
- Roads along spawn→source and spawn→controller paths.
- Light rampart shell on core structures when energy allows.

### Flag orders

Place flags (name prefix matters; room is the flag’s room):

| Flag name prefix | Effect |
|------------------|--------|
| `attack_…` | Siege squad → that room |
| `claim_…` | Send claimer |
| `remote_…` | Force remote mining |
| `ignore_…` | Never auto-remote / expand |
| `rally_…` | Rally point (stored on empire memory) |

### Combat

- **Towers:** focus healers → high DPS → low hits; heal friendlies; repair when idle.
- **Safe mode:** activates if hostiles threaten the core while towers are starved.
- **Defenders:** spawn when military threat parts ≥ threshold.
- **Attack squads:** default 2 attackers, 2 healers, 1 ranged (see `config.attackSquad`).

### What Apex does *not* rely on

xxscreeps stubs some vanilla systems — Apex avoids them:

- Market order book (`Game.market.createOrder` / `deal`) — only `terminal.send` balancing
- Power creeps as units
- Intershard resources / pixels

## Tuning

Edit `config.js` before uploading. Important knobs:

- `remoteMinRcl`, `maxRemotesPerColony` — remote aggression
- `sourceWorkParts`, `haulerCarryPerPathTile` — economy throughput
- `expandMinRcl`, `expandMinStorage` — when to claim
- `attackSquad` — siege composition
- `wallHitsTarget` — wall grinding by RCL
- `visuals` — room text overlays

## Roles

| Role | Job |
|------|-----|
| `bootstrap` | Early multi-skill worker |
| `miner` / `remoteMiner` | Static harvest → container/link/drop |
| `hauler` / `remoteHauler` | Move energy to spawn network / storage |
| `upgrader` | Controller (link/container/storage fed) |
| `builder` / pioneer | Construction; new-colony spawn |
| `repairer` | Roads/containers + wall grind |
| `reserver` | Keep remotes reserved |
| `claimer` | Claim expansion targets |
| `scout` | Refresh intel on stale rooms |
| `mineralMiner` | Extractor mining → storage/terminal |
| `defender` | Home military |
| `attacker` / `ranged` / `healer` / `dismantler` | Flag attack waves |

## Design notes (why this shape)

1. **Spawn priority as data** — emergency and defense always outrank remote luxury creeps.
2. **Specialization after capacity** — avoids 5 WORK miners the spawn cannot refill.
3. **Remotes are scored, not hardcoded** — threat, SK rooms, highways, and foreign reservations push scores down.
4. **Pioneers separate from claimers** — claim alone does not create a spawn; builders must finish infrastructure.
5. **Flat CommonJS modules** — matches `xxscreeps manage bot` loaders with zero toolchain.

This is a **sample empire**, not a season-winning bot. It is meant to stress xxscreeps APIs, demonstrate multi-room ops, and give private-server operators a living world out of the box.
