# RL performance contract

The training loop is primarily a simulator/encoding pipeline with a relatively
small transformer behind it. Optimize measured wall-clock bottlenecks without
changing environment semantics between training and evaluation.

## Current bottlenecks

| Rank | Area | Current mitigation |
|---|---|---|
| 1 | Node tick, room load, encode, reward | Post-tick encode and reward share one room load |
| 2 | Spatial paint and observation IPC | Pooled direct patch paint, binary frames, lean metadata |
| 3 | Environment parallelism | Headless vector workers with explicit curricula |
| 4 | Rollout transfer and action heads | Pinned single-barrier D2H, compact entity buckets |
| 5 | PPO minibatch transfer/graphs | Deferred batched critic, host GAE, batched Muon plus fused AdamW |
| 6 | Navigation | Shared pathfinder semantics and cached paths |

Measure simulator tick, encoding, IPC, inference, rollout storage, H2D, actor
update, and critic update separately. Overall steps per second alone cannot
locate a regression.

## Measured throughput, RTX 5090 / 12 physical cores

Measured with zero actions so the numbers isolate transport, not policy quality.

| Configuration | Aggregate |
|---|---:|
| One environment, direct client, young colony | 1,479 ticks/s |
| One environment, restored tick-16,897 world | 1,185 ticks/s |
| `VecScreepsEnv`, 4 to 24 envs, default torch threads | 370–374 ticks/s, flat |
| `VecScreepsEnv`, same, one intra-op thread | 1,051–1,177 ticks/s |
| 24 independent processes, one env each | 1,448 ticks/s |
| 4 independent processes | 3,523 ticks/s |

Three conclusions follow, and each is a rule for configuring a run.

**Host intra-op parallelism must be off.** Every environment worker thread calls
torch to decode and stack its observation, and torch's default intra-op pool
fans each of those calls across all cores. N workers then contend for N times
the hardware, which is why the vector environment did not scale at all between
four and twenty-four environments. `vec_env.configure_host_threads()` pins it to
one thread and is called from every entry point that drives environments.

**More environments cost throughput.** Aggregate falls monotonically past four
concurrent simulators: 3,523 at four, 2,050 at twelve, 1,448 at twenty-four,
1,322 at thirty-two. Each simulator carries its own engine heap, so beyond the
core count the fleet loses to memory and scheduler pressure. Environment count
should be chosen for batch diversity and reservoir lane granularity, not for
throughput.

**Fleet throughput is now bounded by GPU work.** One update of 12,288
transitions at twelve environments and minibatch 384 decomposes as roughly
14.3 s of per-step policy forward (1,024 steps at about 14 ms, launch-bound and
independent of environment count), 10.5 s of environment stepping, and 16.4 s of
PPO update (36,864 sample-passes at about 2,250 per second). Collection is no
longer the limiter, so the next real gains are a larger optimizer minibatch when
the device is free, or CUDA graphs for the per-step forward — not a faster
environment driver.

Observation payload is 201 KB per environment tick: 140 KB of spatial patches
for the two active rooms, the remainder actor, target, and mask planes. A sparse
dynamic-entity encoding would cut it, but that is an observation ABI change.

## Optimizer batch and compilation

Minibatch is transitions per optimizer step, and a transition here is a whole
team state: two room patch planes, up to 100 actor rows, 128 targets, and the
masks, driven through a spatial transformer and a 233-token entity transformer
with conditional heads per actor. Measured activation cost is about 15 MB per
transition, so the schema default of 1536 peaks near 13.6 GB.

Sizing the minibatch down to survive alongside another job is a false economy:
at twelve environments and 512 steps, minibatch 1536 runs an update in 17.5 s
(collect 12.4 s, optimize 5.7 s at about 3,200 sample-passes per second), while
minibatch 384 needs 41–50 s for the same transitions at about 2,250 per second,
because 96 launch-bound optimizer steps replace 12. Small minibatches also make
each PPO step noisier, so this is a result-quality choice, not only throughput.
Runs take the whole device through `mlq` rather than fitting beside other work.

`--compile` is now a win, and it is compiled asymmetrically: CUDA graphs on the
tick path, plain eager on the minibatch path. That split is the result of four
measured failures, all worth keeping written down.

What is checked every run: after warmup `PPOTrainer.graph_stats()` reports
**four unique graphs, zero graph breaks, and zero late shape mints** — actor and
critic, at both room packs, all captured before the first timed rollout.
`_note_shape` records every `(call site, shape)` a compiled graph is asked for
and prints any that appears after warmup, so a recompile stall is attributed
instead of inferred from a slow update.

Measured at twelve environments, 512 steps, minibatch 1536:

| Configuration | collect | optimize | total | peak |
|---|---:|---:|---:|---:|
| eager everywhere | 12.4 s (531 sps) | 5.7 s | 17.5–18.4 s | 13.6 GB |
| CUDA graphs on tick path | 6.6–8.0 s (862–924 sps) | 6.9 s | 13.5–14.0 s | 18.5 GB |
| same, four live rooms | 8.5–8.7 s (707–725 sps) | 9.2 s | 17.6–17.9 s | 24.7 GB |

Collection, which is 60% of an update, runs 1.7x faster. The tick path is where
that comes from: `act` and `value_only` are called once per simulated tick at
B=12, thousands of launch-bound calls per update, which is exactly what
`reduce-overhead` removes.

Four defects had to be fixed, and one design attempt had to be abandoned:

- `mode="reduce-overhead"` returns tensors owned by a CUDA-graph capture pool,
  which the next replay overwrites, and `.float()` on a float32 tensor is not a
  copy. `value_only` and the `rollout_values` accumulators handed those buffers
  to callers that outlive the call, which failed with "accessing tensor output
  of CUDAGraphs that has been overwritten by a subsequent run". `_own()` copies
  them out of the pool.
- Entity capacity bucketing changes the model-bound shape as a colony grows, so
  `dynamic=False` minted a graph and a pool per bucket mid-episode.
  `vec_env.set_entity_compaction(False)`, set automatically under `--compile`,
  freezes actor and target capacity. The cost is that every minibatch attends
  the full 100-actor/128-target sequence; the switch is global, so the eager
  minibatch path cannot compact while the compiled tick path needs fixed shapes.
- Room capacity had a one-room bucket, and the live count reaches `MAX_ROOMS`
  once expansion exposes neighbors. Two buckets remain, `(2, MAX_ROOMS)`, and
  `warmup` captures the tick path at both. Before that, six graphs were minted
  mid-run — 43 s inside a timed optimizer step in the worst case.
- Host rollout storage packs rooms to its own capacity, which can sit below the
  model bucket, so `_compact_entity_prefixes` pads up to the bucket instead of
  handing the model a third room shape.
- Compiling the minibatch path was tried twice and rejected on memory, not
  principle. A training forward and backward at minibatch 1536 with four room
  slots already peaks at 24.7 GB; a `reduce-overhead` pool at that batch size
  needed 28.5 GB during capture, and plain inductor's workspace on top of the
  activations OOMed a 32 GB device in warmup. Compiling it bought about 1 s of a
  13 s update. The honest fix is a smaller observation, not a bigger card:
  sparse dynamic-entity encoding is the roadmap item.

Environment count is also measured, not assumed: at minibatch 1536, twelve
environments cost 2.85 ms per transition against 3.20 ms at twenty-four, because
the box saturates well before twenty-four simulators while the optimizer cost
per transition is unchanged. Twelve is the current production choice.

The Muon step itself is not a throughput factor: hidden matrices are stacked by
shape into three batched kernel chains per network, compiled with
`dynamic=False, fullgraph=True`.

## Observation transport

`RL_OBS_FMT=bin` and `RL_CMD_FMT=bin` are the defaults. Both directions use
strict framed binary protocols; JSON commands remain a debug option.

```text
magic "XRL1" | version u8 | flags u8 | schema_version u16 little-endian
| metadata_length u32 | blob_length u32 | metadata JSON | tensor blob

XRL1 response version: 4

flags: bit0=ok, bit1=done, bit2=has_observation,
       bit3=has_scripted_teacher_actions

blob order:
  u8:  patches
  f32: actors, targets, roomCoords
  u8:  roomMask, actorMask, actorOutcome, targetMask,
       intentMask, dirMask, targetSelectMask, amountMask, constructionMask
  when bit3 is set, a trailing teacher-action payload:
    u8:  types, dirs, targets, amounts, constructionTypes
    u16: constructionTiles (little-endian)
    u8:  bodyCounts[8], bodyOrder[8]
```

Step commands contain five scalar uint8 planes (`types`, `dirs`, `targets`,
`amounts`, `constructionTypes`), one uint16 `constructionTiles` plane, then
`bodyCounts[8]` and `bodyOrder[8]` uint8 planes per actor-slot. At 100 actors
this is 2,300 payload bytes, still deterministic and
allocation-light compared with nested JSON.

`step_scripted` responses use the same 23-byte actor-slot plane layout in a
binary tail. Metadata carries only `teacherActions: {rows, slots, byteLength}`;
the nested `info.actions` object remains available only in JSON debug response
formats. The client requires an exact length, validates every categorical and
body-order invariant, and slices the teacher tail before observation decoding.
There is intentionally no XRL1 v3 compatibility path.

The server sends only active room patches plus `roomsUsed`; Python pads to the
four-room model ABI. Host masks and spatial patches remain uint8 until bulk
device promotion. `RL_LEAN_META=1` omits full actor/target metadata on the hot
path while preserving compact action diagnostics.

Debug formats remain available:

```text
RL_OBS_FMT=json  expanded tensors in JSON
RL_OBS_FMT=b64   per-field base64 tensors
RL_OBS_FMT=pack  one base64 tensor blob
RL_OBS_FMT=bin   framed raw bytes; default
```

Do not change field order independently in Node and Python. Any transport
change requires a real environment round-trip contract.

## Rollout storage

`HostRolloutBuffer` preallocates dense non-spatial state but stores only active
uint8 room pages in reusable 256-page chunks. The default rollout is 512 steps
across 24 environments: 12,288 transitions. It uses a 1,536-transition
minibatch, exactly eight minibatches per epoch, and three PPO epochs. The room
page store grows only with rooms actually represented. The 16 spawn
count/order values are uint8 on host.

The normal schema allows one 256-step extension up to a 1,024-step cap. For
predictable wall-clock jobs, pass equal `--steps` and `--max-rollout-steps`.
Observation host mirrors avoid a full device-to-host copy on every step. The
construction legality field is itself bit-packed: four rooms × seven types ×
313 bytes rather than a dense 70,000-byte mask.

Dense entity-target legality is shared as `[intent,target]`; actor capability
and room compatibility are applied in the model. Construction tiles use their
own room-local packed support and no longer consume the shared target budget.
Actor-local typed candidate lists remain roadmap work if entity overflow or
mask cost dominates.

Model-bound observations use front-prefix capacity buckets: rooms 1/2/4,
actors 8/16/32/64/100, and targets 16/32/64/128. Node and rollout storage retain
the full ABI, while eager actor/critic execution attends only over the smallest
bucket containing every live prefix row. Target indices remain unchanged.
Compact actions are sent directly over XAC1 and zero-padded only when stored in
the fixed rollout ABI. Expansion recomputes capacities every state; reset-time
visibility is never frozen. Sparse patch reconstruction also emits only the
minibatch's required 1/2/4-room bucket rather than building four pages and then
discarding inactive suffix rooms during host-to-device promotion.

Joint pretraining reuses the same sparse paged observation store rather than
cloning and then stacking a dense Python-list corpus. At 512 steps × 10 envs,
the empty allocated chunk is about 315 MiB versus 1,682 MiB for dense four-room
storage, before active room pages grow. Spawn replay is capped at 16 examples
per energy × body-length × body-composition stratum, bounding retained memory and the
replay epoch cost.

Actions and behavior log-probabilities share one reusable pinned GPU-to-host
staging barrier. Those exact host action planes feed both Node and rollout
storage, avoiding duplicate transfers. Reward/done stay on the host. One
CleanRL-style GAE recurrence computes both actor advantages and critic returns
on CPU instead of bouncing a small trajectory through CUDA.

The behavior actor world latent shares that same D2H barrier. The deferred
critic pass returns its scalar value, world latent, and value logits together.
PPO uploads these frozen targets once per update (~21 MiB at B=8,192) and uses
index selection for NextLat; it never reruns a future-state trunk per shuffled
minibatch.

## GPU and compilation

`torch.compile(..., dynamic=False, mode="reduce-overhead")` specializes on
exact batch shapes. The intended graph shapes are:

| Graph | Batch | Mode |
|---|---:|---|
| Actor sample | `num_envs` | eval |
| Critic value/bootstrap | `num_envs` | eval |
| Actor update | `minibatch` | train |
| Critic update | `minibatch` | train |

Terminal-value batches are padded to `num_envs`; the last update minibatch is
padded to `minibatch`. Actor backward completes and releases activations before
the independent critic forward to lower peak VRAM. TF32 and contiguous device
promotion are enabled where appropriate.

The current eager PPO path is the default and trusted path. Compilation is
explicitly opt-in with `--compile`. A compiled update previously
hit a TorchInductor target-index assertion. Compilation must not be re-enabled
for a training claim until it passes the same update and evaluation contracts;
compilation success on inference alone is insufficient.

## Architectural compute

Actor and critic each encode the same room patches through an independent
486,528-parameter spatial transformer and then an independent entity
transformer. This is deliberate representation isolation but duplicated
training compute. Benchmark the following before increasing model width:

1. current fully separate trunks;
2. shared spatial patch tokens with separate entity transformers;
3. shared observation encoder with stop-gradient or controlled gradient routing;
4. actor-only inference with deferred or less frequent value evaluation during collection.

Compare wall-clock SPS, VRAM, actor quality, critic EV, and gradient interference.
Parameter count alone does not decide this tradeoff.

Ordinary critic inference is deferred out of the synchronous simulator loop.
The actor samples each tick; after collection, the unchanged critic evaluates
the stored pre-action states in update-sized batches. Final and time-limit
terminal values are still evaluated at their exact boundaries. Training logs
separate collection and update seconds/rates so future optimization is based on
measured phases rather than aggregate SPS.

## High-value remaining work

- Precompute reusable static terrain embeddings and paint only dynamic state.
- Add asynchronous environment workers and a trajectory queue if slowest-worker
  synchronization is material.
- Introduce typed/paged candidate storage when real 32–64 actor scenarios prove
  dense packing or overflow is a bottleneck.
- Keep DataFrames and Parquet off the tick path; use them only for offline
  trajectories and batched metric export.

Every optimization must preserve post-tick observation alignment, action
legality, navigation parity, deterministic scenario seeding, reward events, and
checkpoint ABI validation.
