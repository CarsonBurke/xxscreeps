# Tick pipeline performance

## Profile

The default local-storage configuration was profiled with the four imported launcher bots,
`game.tickSpeed: 0`, and `processor.intentAbandonTimeout: 5000`.

The local storage host was the busiest engine thread: it was active for about 40% of sampled wall
time, compared with about 28% for the runner and 22–23% for each processor worker. Its leading named
self-time costs were MessagePort `postMessage`, port iteration, and responder dispatch.

Storage request instrumentation counted 1,487,984 worker-to-host RPCs over 21,090 ticks, or **70.6
RPCs per tick**. Script evaluation was the largest category at 301,148 calls (**14.3 per tick**), and
each call serialized the complete JavaScript source of its `KeyvalScript`.

The transport also contained a lost wakeup. `messagePortToIterable()` replaced its pending wake
promise in the message callback. If another message arrived while the async generator was suspended
at its previous `yield`, the callback resolved the old promise, replaced it, and left the queued
message waiting for a later message to provide another wake.

## Changes

- Reset the MessagePort wake promise only when the consumer takes ownership of the queued messages.
  Messages arriving while the consumer handles an earlier batch now remain observable without a
  subsequent message.
- Give local key-value scripts stable content identifiers. Worker clients send the full script once
  to load it, then send the 16-character identifier for later evaluations instead of retransmitting
  source code about 14 times per game tick.

## Adjacent A/B benchmark

The optimized and unchanged builds were run consecutively against fresh imports on the same busy
host. Background load varied, so the adjacent comparison is more meaningful than comparison with an
older absolute TPS number.

| Window | Optimized | Baseline | Change |
| --- | ---: | ---: | ---: |
| 0–30 s | 604.2 TPS | 556.1 TPS | +8.7% |
| 30–60 s | 785.3 TPS | 650.6 TPS | +20.7% |
| 60–90 s | 704.6 TPS | 661.3 TPS | +6.5% |
| Full run | **704.8 TPS** | **622.7 TPS** | **+13.2%** |

Both runs completed without abandoned intents. Absolute throughput was depressed by unrelated
CPU-bound jobs on the host; the earlier unloaded fixed-stall baseline was about 778 TPS.

## Verification

- Workspace TypeScript build passed under Node 24.15.0.
- Strict lint passed for every changed TypeScript file.
- The MessagePort lost-wakeup regression test, stable script-identifier test, existing local-storage
  tests, and the processor wake-scheduler tests all passed (10 targeted tests total).
- The complete test suite passed: **486 tests, 486 passed, 0 failed** under Node 24 with VM modules
  enabled.
- The final build completed a 600-second soak: 410,247 ticks, zero abandoned intents, and zero
  runtime errors. Host contention held the average to 683.7 TPS; individual minutes ranged from
  567.5 to 808.3 TPS.
