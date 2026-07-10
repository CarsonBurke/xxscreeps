# Intent-handoff stalls at high tick rates — investigation notes

**Status:** diagnosed to a probable root cause, not yet fixed. This doc is a handoff for whoever
picks it up.

## Symptom

With `game.tickSpeed: 0` (uncapped TPS) and the four built-in bots running, the main loop
periodically stalls for exactly `processor.intentAbandonTimeout` (default 5000 ms), then prints:

```
Abandoning intents in rooms [W1N1, W1N9, W9N1, W9N9] for tick 8682
Tick 8682 ran in 5019ms; avg: 51.48ms
```

- Typical ticks are 1–2 ms; the stalled ticks are pinned at the abandon timeout (5001–5021 ms
  observed), i.e. nothing was making progress — the tick only completed because the watchdog fired.
- The abandoned rooms are always (a subset of) the four bot-owned rooms — the only rooms with a
  nonzero intent-user count. Sometimes all four, sometimes one.
- Frequency: roughly 1 in 4,000–8,000 ticks (~0.02%), every 10–20 wall seconds at 400–750 TPS.
  Irregular spacing; no correlation with the 2-minute database save.
- **Zero errors or warnings are logged** around the stalls. No "left behind" resets (the
  `instance.ts:183` path never fired), no sandbox disposals, no console output from bots.
- Impact is large at high TPS: in a ~5-minute run, 18 stalls × 5 s ≈ 90 s, nearly halving
  sustained throughput (369 TPS observed vs ~750 TPS with the timeout tightened to 250 ms).
- Not reproducible in practice at the default 250 ms tickSpeed — the idle window between ticks
  hides the race (see hypothesis below), which is presumably why this has gone unnoticed.

Repro: `.screepsrc.yaml` with `game.tickSpeed: 0`, default world (`xxscreeps import`), the four
built-in simplebots, `xxscreeps start`, then watch stdout for `Abandoning`. Setting
`processor.intentAbandonTimeout: 250` bounds the damage but drops that tick's bot intents — it's a
mitigation, not a fix.

## Tick pipeline (what has to happen for a tick to complete)

All coordination goes through the shard **scratch** keyval store plus pubsub channels. Three
services (worker threads under the launcher): **main**, **processor**, **runner**.

1. **Main** (`packages/xxscreeps/engine/service/main.ts` `tick()`, ~line 98):
   copies `activeUsers` → `runnerUsersSetKey(time)`, publishes `{type: 'process', time}` on the
   processor channel and `{type: 'run', time}` on the runner channel, then waits (indefinitely) on
   the service channel for `tickFinished`, with a watchdog timer that fires
   `abandonIntentsForTick()` after `intentAbandonTimeout`.

2. **Room process queue**: `begetRoomProcessQueue(shard, time)`
   (`engine/processor/model.ts:166`) copies `activeRoomsKey` (zset: room → **score = number of
   intent users in that room**) to `processRoomsSetKey(time)`. Rooms with score 0 (NPC-only,
   terrain-only) are immediately consumable. Bot rooms sit at score 1 until their user's intents
   arrive. `processRoomsPendingKey(time)` is initialized to the total room count.

3. **Runner** (`engine/service/runner.ts`): on `{type: 'run'}`, pops user ids from
   `runnerUsersSetKey(time)`, runs each sandbox, then calls `publishRunnerIntentsForRooms()`
   (`engine/processor/model.ts:106`), which for each of the user's intent rooms:
   - `zAdd(processRoomsSetKey(time), [[-1, room]], {if: 'XX', incr: true})` — decrements the
     room's score;
   - pushes the intent payload onto `rooms/<room>/intents`;
   - if the score reached 0, publishes `{type: 'process', time, roomNames: [room]}` on the
     processor channel. **This publish is the only thing that wakes the processor for that room.**

4. **Processor** (`engine/service/processor.ts`): owns N workers (`processorCount =
   clamp(1, concurrency, ceil(userCount / 2))` — with 4 bot users that is only **2 workers**,
   despite `concurrency` defaulting to cores+1; `main.ts` similarly caps nothing). On a `process`
   message it activates **idle** workers; each activated worker runs `consumeRoomsQueue()`, which
   pops score-0 rooms from `processRoomsSetKey(time)` (`ZPopByScore` in `engine/db/async.ts:59` —
   non-blocking, returns when no member is in range) and feeds them to a worker thread. When the
   generator drains, the worker sets `worker.idle = true` and stops polling.
   `roomDidProcess()` decrements `processRoomsPendingKey`; at 0 it triggers finalize →
   `roomsDidFinalize()` → publishes `tickFinished` → main advances the clock.

5. **Watchdog**: `abandonIntentsForTick()` (`model.ts:296`) zeroes every remaining score in
   `processRoomsSetKey(time)` (`zInterStore` weights 0), deletes the runner queue, and publishes a
   bare `{type: 'process', time}` — which activates workers with `activations = Infinity`. This is
   why the system always recovers after exactly the timeout.

## Primary hypothesis: lost wakeup in the processor's idle/activation handshake

`engine/service/processor.ts`, `case 'process'` (~line 132):

```ts
for (const worker of workers) {
    if (worker.idle) {
        worker.idle = false;
        Async.mustNotReject(async () => {
            for await (const roomName of consumeRoomsQueue(worker, time)) { ... }
            worker.idle = true;          // <-- (B)
        });
        if (--activations <= 0) break;
    }
}
```

There is a classic check-then-sleep race between the queue poll and the idle flag:

- A worker's `consumeRoomsQueue` generator observes the queue empty (its final `ZPopByScore`
  returns null) and unwinds. Until line (B) runs — and generator unwinding crosses several
  microtask boundaries (`Fn.concatAsync`, `lookAhead`) plus a possible `checkAffinity` re-loop —
  the worker is *neither polling the queue nor marked idle*.
- In that window the runner decrements the last bot room to score 0 and publishes
  `{type: 'process', roomNames: ['W1N1']}`. The processor's message handler finds **no idle
  worker** (`if (worker.idle)` is false for both), sets `checkAffinity` on the affinity worker
  (which is also about to exit and never rechecks), decrements `activations`, and does nothing.
- Both workers then set `idle = true`. The room sits at score 0 in the queue, consumable, but
  nothing ever polls again. The tick hangs until the watchdog's bare `process` publish re-activates
  the (now idle) workers.

This fits every observation: exact-timeout stall length, no errors, only intent rooms affected
(score-0 rooms are consumed in the initial burst while workers are certainly awake), vanishing
probability at 250 ms tickSpeed (workers are long idle before the next tick starts), and rate
proportional to tick frequency. With only 2 workers and 4 single-user rooms whose readiness
trickles in one runner-publish at a time, the window is hit every ~10⁴ ticks.

A subtlety for the "all four rooms at once" case: the runner runs all 4 users and may batch its
four `process` publishes into the same delivery burst while both workers are mid-unwind, losing
all four wakeups together.

### Suggested fix directions (pick one, verify under load)

- **Re-check after declaring idle**: after `worker.idle = true`, do one more
  `ZPopByScore`/`zCard` on `processRoomsSetKey(time)`; if non-empty, flip back and loop. Closes
  the window because the publisher's decrement is already visible in scratch by the time the
  wakeup would be lost. (The publish-after-write ordering in `publishRunnerIntentsForRooms` is
  correct; only the consumer-side sleep transition is racy.)
- **Buffer missed activations**: in the `process` handler, when no worker is idle, record a
  pending-activation flag; workers check it before setting `idle = true`.
- Don't try to fix it by making the message handler retry/spin — the queue state in scratch is the
  source of truth; the fix belongs at the sleep transition.

A useful diagnostic while validating: log (or count) transitions where a `process` message with
`roomNames` finds zero idle workers, and correlate with the `Abandoning` lines. That signature
firing right before each stall would confirm the hypothesis; instrumentation should be removed
after.

## Alternative hypotheses (less likely, not fully excluded)

- **Pubsub delivery lag/loss** between runner and processor threads (local socket responder,
  `engine/db/storage/local/`). A ≥5 s delivery delay would look identical. Nothing observed
  supports outright message loss, and the storage host is otherwise responsive during stalls
  (the watchdog's own scratch ops complete instantly), so this is second-tier.
- **`zAdd {if: 'XX'}` decrement racing `begetRoomProcessQueue`'s `copy`** for the *next* tick —
  ordering appears safe (beget only runs after `tickFinished`), but nobody has proven the runner
  can't still be publishing for tick N at that point (see `model.ts:119`'s own NOTE about intents
  published for an abandoned tick being processed later, "which is probably not desired").

## Ruled out

- **Runner sandbox resets / "left behind"** (`engine/runner/instance.ts:183`): never logged.
- **CPU bucket exhaustion**: bots use ~1 ms/tick against `kCPU` refill; bucket stays full; and a
  skipped user still publishes empty intents (`instance.ts`, `publishRunnerIntentsForRooms` with
  `{}`), which would unblock the room anyway.
- **Database save pauses**: stall spacing (10–20 s) doesn't match `saveInterval` (2 min), and
  stalls persisted with `saveInterval: 120`.
- **Sandbox migration** (`runner.migrationTimeout`): there is a single runner worker under the
  launcher; the migration path between runner services never applies.
- **isolated-vm GC/memory limit**: RSS stable ~1.2 GB, no disposals logged.

## Environment / context for reproducing

- Node 24.15.0 (mise; repo requires ≥24 — system node 20 fails), `pnpm install && pnpm build`
  at the workspace root, run everything from the repo root (`/screeps` data dir and
  `.screepsrc.yaml` are gitignored there; unix-socket paths must stay short).
- World: `node packages/xxscreeps/xxscreeps.js import` (default @screeps/launcher `db.json` —
  121 rooms, 4 simplebots with spawns, source keepers).
- Start: `node packages/xxscreeps/xxscreeps.js start`. Tick timing prints on stdout.
- Benchmark method: count `^Tick` lines in the log over a 60 s wall window. Baseline observed:
  369 TPS (5 s abandon timeout) → 754 TPS (250 ms timeout), avg tick 1.55 ms, ~250% CPU on a
  24-core machine, so the ceiling is pipeline latency, not compute.
- The working tree contains three uncommitted fixes required to build/run HEAD at all
  (regressions from `8237674d` and `2ff79b3d`): keyval JSON-reviver `default` case restored
  (`engine/db/storage/local/keyval.ts`), blob `del()` deletion-marker ordering restored
  (`engine/db/storage/local/blob.ts`), and the `/xxscreeps:mods/schema` virtual module added to
  the isolated sandbox (`driver/sandbox/isolated/index.ts`). All three were adversarially
  reviewed; don't rebase them away.
- Current `.screepsrc.yaml` (gitignored) also carries the benchmark settings
  (`tickSpeed: 0`, `intentAbandonTimeout: 250`, `saveInterval: 120`). Restore
  `intentAbandonTimeout` to its default when testing a real fix — the tight timeout masks the
  stall you're trying to reproduce.

## Acceptance criteria for a fix

1. With `tickSpeed: 0` and `intentAbandonTimeout` at its **default 5000 ms**, run ≥10 minutes:
   zero `Abandoning intents` lines after the startup transient (the first couple of ticks after a
   restart legitimately abandon while sandboxes cold-start; that's separate and benign).
2. Sustained TPS should land at or above the ~750 TPS previously achieved only via the
   250 ms-timeout mitigation.
3. Bots remain in `activeUsers` with live creeps (check with a room scan; creep lifetime is 1500
   ticks ≈ 2 s of wall time at these speeds, so momentary zero-creep rooms are normal churn).
