# Apex v3 — War & Campaigns

Combat coordination for multi-room ops. Modules: `combat.js` (creep runners + towers), `war.js` (campaign FSM).

Economy roles are **not** defined here; spawn merges `war.spawnRequests(home)` into its queue and `roles.js` wires military runners from `combat` or `war.runners`.

## Integration

```js
// main.js (each tick, after intel)
const war = require('war');
war.tick();
// optional if no defense.js:
// war.tickDefense();

// spawn.js
const military = war.spawnRequests(room); // or room.name
// merge into colony request list by priority

// roles.js
const combat = require('combat');
// defender: combat.runDefender, attacker: combat.runAttacker, ...
// or: const war = require('war'); ... war.runners
```

## Flag orders

| Flag prefix | Effect |
|-------------|--------|
| `attack_ROOM` | Open **attack** campaign → siege squad (rally then march) |
| `claim_ROOM` | Open **claim** campaign → light military screen + support until spawn |
| `defend_ROOM` | Open **defend** campaign → station defenders until flag removed |
| `rally_…` | Squad rally point (`Memory.empire.rally`) |
| `remote_ROOM` | Force remote interest (security campaigns may open on threat) |
| `ignore_ROOM` | Never auto-remote / skip auto campaigns |

Place flags anywhere; the **room name is taken from the flag’s position**, not the name suffix (suffix is conventional).

## Campaign memory

`Memory.empire.campaigns` is an **object keyed by campaign id** (intel uses `Object.values`):

```js
Memory.empire.campaigns[id] = {
  id,                    // e.g. attack_W1N1_3
  type: 'remote'|'claim'|'attack'|'defend',
  room, home,
  started, lastProgress,
  spentEnergy,           // body cost of military assigned (+ tracked at first sight)
  deaths, kills,
  safeTicks,             // consecutive visible ticks with no hostiles/cores
  goal,                  // 'clear_hostiles' | 'claim' | 'hold'
  status: 'active'|'won'|'abandoned'|'cooldown',
  abandonReason?,
  flag?, squad?,
  assigned: { [creepName]: bodyCost },
  cooldownUntil?,
  ended?,
}
```

## Campaign types

| Type | Opens when | Win |
|------|------------|-----|
| **remote** | Remote harvest room has hostiles / invader core (or sustained remote threat) | Room clear for `clearSafeTicks` |
| **claim** | `claim_*` flag, or we own a room with no spawn yet | Our controller **and** spawn present |
| **attack** | `attack_*` flag | Clear, or flag gone + clear |
| **defend** | `defend_*` flag | Flag removed (holds while flag exists) |

## Progress (preferred over pure energy)

Progress resets the stall timer (`lastProgress`):

- Hostile count decreases (kills)
- Room becomes clear (`safeTicks` reaches threshold)
- Reservation ticks increase
- Controller claimed / spawn built (claim campaigns)
- Invader core destroyed

## Abandon rules

Tunables live in `war.js` → `WAR_CFG` (override with `config.war`).  
Aliases: `config.war.maxDeaths` → `deathLimit`, `config.war.noProgressTicks` → `stallTicks`.

| Condition | Default |
|-----------|---------|
| `spentEnergy` > budget **and** no progress for `stallTicks` | budget **50 000** (capped by `empire.plan.byColony[home].militaryBudget` when set) |
| `deaths` > limit while room not clear | **8** deaths |
| No progress for `stallTicks` | **1500** ticks |
| Enemy RCL > home RCL + `maxEnemyRclDelta` | delta **1** |
| Enemy towers ≥ `enemyTowerAbandon` while home RCL < `minHomeRclForTowers` | **2** towers / RCL **6** |

After **abandon** or **won**, campaign enters **cooldown** (`cooldownTicks` **1000**, win cool **200**) so the same room/type is not re-opened immediately.

## Hook points (already wired in apex-v3)

- `room.js` calls `war.tick(colonies)` each tick
- `spawn.js` merges `war.spawnRequests(room)` into the colony queue
- `roles.js` uses `combat.run*` for military roles when `combat.js` is present

## Squads

Default compositions (`WAR_CFG.squads` / `config.war.squads`):

| Type | Attackers | Healers | Ranged | Dismantlers | Defenders |
|------|-----------|---------|--------|-------------|-----------|
| remote | 1 | 1 | 0 | 0 | 0 |
| claim | 1 | 1 | 0 | 0 | 1 |
| attack | 2 | 2 | 1 | 0 | 0 |
| defend | 0 | 0 | 0 | 0 | 2 |

Offensive roles **rally** (see `config.squadRallyBeforeMarch`) at `rally_*` or home spawn, then march to `targetRoom`.

## Creep memory (military)

```js
{
  role, home,
  targetRoom,          // campaign room
  defendRoom?,         // defend campaigns
  campaignId, squad,
  flag?, marching?,
}
```

## Combat roles

| Role | Behavior |
|------|----------|
| `defender` | Clear hostiles at home / station; recycle after safe ticks if unassigned |
| `attacker` | Melee priority targets (healers → threat → structures / cores) |
| `ranged` | Kite melee, mass attack clumps |
| `healer` | Lowest-HP squad ally; self-heal threshold |
| `dismantler` | Walls/ramparts → towers/spawns |

Towers: `combat.runTowers` / `war.tickDefense` — healers first, rampart-aware, then heal friendlies, light repair.

## What war does *not* do

- Market, power creeps, nukes
- Civilian remote mining / claiming (economy / claimer roles)
- Rewriting economy spawn priorities beyond merging `spawnRequests`
