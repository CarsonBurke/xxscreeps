# Apex v3 — War & Campaigns

Combat coordination for multi-room ops. Modules: `combat.ts` (creep runners + towers), `war.ts` (campaign FSM).

Economy roles are **not** defined here; spawn merges `war.spawnRequests(home)` into its queue and `roles.ts` wires military runners from `combat` or `war.runners`.

## Integration

```js
// main → intel.tick() syncs intents, then roomManager → war.tick()
const war = require('./war');
war.tick();

// spawn
const military = war.spawnRequests(room);
// merge into colony request list by priority
```

## Room intents (not flag-name prefixes)

Orders are **`RoomIntent` enums** on room memory (`Memory.rooms[name].a[intent]`).

Player input: place **any-named** flag in the target room; **primary color** selects intent.

| Color  | `RoomIntent` | Effect |
|--------|--------------|--------|
| RED    | `attack`     | Open attack campaign (siege / clear) |
| PURPLE | `defend`     | Hold defenders until intent cleared |
| BLUE   | `claim`      | Claim support until spawn up |
| CYAN   | `remote`     | Force remote interest |
| WHITE  | `ignore`     | Never auto-remote |
| YELLOW | `rally`      | Military rally point (`Memory.empire.rally`) |

```ts
if (getRoomIntent(roomName) === RoomIntent.attack) { /* war opens campaign */ }
```

Flag **name is ignored**. Multiple colors in one room: highest priority wins (attack > defend > claim > remote > ignore > rally).

Removing the flag clears that room’s intent next tick.

## Campaign memory

`Memory.empire.campaigns` is an **object keyed by campaign id**:

```js
Memory.empire.campaigns[id] = {
  id, type, room, home,
  started, lastProgress,
  spentEnergy, deaths, kills, safeTicks,
  goal, status, abandonReason?,
  intent,   // RoomIntent number or null for auto remote security
  squad?, assigned: { [creepName]: bodyCost },
  cooldownUntil?, ended?,
}
```

## Campaign types

| Type | Opens when | Win |
|------|------------|-----|
| **remote** | Remote harvest room has hostiles / invader core | Room clear for `clearSafeTicks` |
| **claim** | `RoomIntent.claim`, or we own a room with no spawn yet | Our controller **and** spawn present |
| **attack** | `RoomIntent.attack` | Clear, or intent lifted + clear |
| **defend** | `RoomIntent.defend` | Intent cleared (holds while defend intent active) |

## Progress (preferred over pure energy)

Progress resets the stall timer (`lastProgress`):

- Hostile count decreases (kills)
- Room becomes clear (`safeTicks` reaches threshold)
- Reservation ticks increase
- Controller claimed / spawn built (claim campaigns)
- Invader core destroyed

## Abandon

See `WAR_CFG` in `war.ts`: energy budget, death limit, stall ticks, enemy RCL/towers too strong. After abandon or win → cooldown before re-open.
