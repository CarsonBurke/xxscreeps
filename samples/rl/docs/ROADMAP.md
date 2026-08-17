# RL roadmap and open experiments

This file contains unresolved work. Implemented behavior belongs in
[`ARCHITECTURE.md`](./ARCHITECTURE.md), executable gates in
[`TRAINING.md`](./TRAINING.md), and settled conclusions in
[`DECISIONS.md`](./DECISIONS.md).

## Immediate learning blockers

Before another substantial PPO continuation:

0. **Investment behaviour is unfunded, and the cause is not established.**
   `gamma=0.995` gives an effective horizon of 200 ticks on a 20,000-tick
   problem, and construction and claiming both repay well beyond it. That makes
   discounting a candidate explanation for their suppression, not a finding.

   The obvious counterexample has to be accounted for by any explanation:
   spawning is delayed payoff too, since a body costs about 300 energy and
   repays over a ~1,500-tick life, and it was retained under the same discount
   (`spawnCreep` intents at 0.85x of their early rate, 27-35 creeps sustained).
   Remote harvesting has a round trip of hundreds of ticks and grew. A second
   candidate is specific to construction: placement is never supervised, since
   teacher labels carry an arbitrary legal tile, so a built structure's expected
   payoff may be near zero at any horizon.

   The matched pair measured the consequence. Both runs inherited a working
   ~51-creep colony from behavior cloning; the fresh-start control decayed to
   8 creeps within 60 updates and stayed there for 145 more, while the reservoir
   run held 27-35. Then the control's gradient died: value targets collapsed to
   `[0.68, 0.74]` against the reservoir's `[0.16, 3.53]`, KL fell to 0.0005,
   actor gradient norm to 0.048, entropy to 0.22. Raw mean reward halved while
   *normalized* mean reward stayed flat at 0.004, because the reward
   normalizer's running RMS shrank with the economy: the loss never reported the
   collapse. Claims went to zero in **both** runs under greedy decoding, which
   is consistent with either candidate and distinguishes neither.

   Three co-moving series diagnose this in any run and are already logged:
   `max_creeps`, `overflow_step_fraction` (0.000 means nothing in the batch is
   large enough to overflow the observation), and the `value_target_min/max`
   spread. The reservoir prevents the population collapse by re-seeding mature
   states; it changes neither the discount nor placement supervision.

   Separate the two candidates before changing the objective. Placing a
   teacher-quality base and measuring whether throughput improves within a few
   hundred ticks answers whether good placement pays back inside the current
   horizon. Only if it does not is a discount change the indicated fix, and then
   on the same matched protocol, one change at a time: a longer discount with
   the existing finite-horizon critic target, an investment-aware auxiliary
   value head, or the temporal abstraction in
   [`DECISIONS.md`](./DECISIONS.md).
1. **Observation overflow.** Overflow averaged 28.3% over the final ten
   rollouts and reached 75% in one rollout. Diagnose which actor, room, and
   candidate categories overflow at each economy stage before changing caps.
2. **Economic composition.** Prevent excess generalist/harvester production,
   ensure enough haulers recover dropped energy, and spawn/assign creeps for
   remote work. Record body-composition deficits, source utilization, ground energy, sink
   starvation, replacement timing, and body amortization. The expert has been
   refactored around phase demand and proactive replacement, but it still must
   demonstrate complete empty-room progression under the release gate.
   Dedicated spawn scenarios must not launder an end-to-end failure.
3. **Construction quality.** Keep the now-functional construction action, but
   evaluate placement through downstream economy behavior across held-out maps,
   not a hand-designed scalar placement score or construction volume.
4. **Stage-balanced evaluation.** `seed_creep` and `seed_claimer` regressed in
   the first PPO continuation while `empty` and `seed_full` improved. Evaluate
   every stage under several scenario and action seeds and select checkpoints
   with a declared balanced scorecard.
5. **Teacher late-game remote staffing decays.** The oscillation that stopped
   remote delivery entirely is fixed: a remote returner ranked the home spawn
   above the home bank, the spawn's free capacity flipped as it filled and
   drained, and the destination alternated between two sinks on opposite sides
   of the room, so the executor reversed its route every tick and the carrier
   never left the outpost. Remote cargo now targets the bank, and
   `test_seed_outpost_harvests_and_hauls_without_claiming` passes. What remains
   is decay, not deadlock: at seed 7 the seeded outpost worker survives to tick
   1,497 but stops delivering after tick 1,200, only one remote delivery happens
   after 1,200, and there is no remote harvest or delivery at all after tick
   1,600 (`test_outpost_route_finisher_and_replacement_remain_productive` still
   fails on its aging-finisher assertion). The teacher stops restaffing the
   outpost, which is the same late-activity collapse the learned policies show,
   so the planner replacement in "Data, curriculum, and generalization" owns it.
6. **The International outgrows the observation ABI immediately.** Collecting
   teacher start states from TI rejected 10,466 of 12,000 candidate ticks for
   overflow (`roomOverflow` 10,455, `actorOverflow` 6,667, `targetOverflow`
   2,051) and yielded 22 usable states, 21 of them in the first 2,000 ticks and
   none after `mid`. TI is therefore unusable as a late-phase bridge until
   blocker 1 raises the caps, and the scripted planner supplies that bridge
   instead (464 states spanning all four phases). This also bounds what the TI
   critic and actor-auxiliary streams can represent.
7. **The International crashes natively under concurrency.** Four concurrent
   expert sandboxes killed one environment at tick 5,211 with `free(): invalid
   pointer`; the same curriculum, seed, and horizon ran clean alone, so the
   fault is nondeterministic rather than tick-dependent. Collect TI with at most
   two concurrent environments until the sandbox or native pathfinder is
   diagnosed.

## Start-state coverage

The event-stratified start-state reservoir is implemented; the mechanism is in
[`ARCHITECTURE.md`](./ARCHITECTURE.md#environment-state-snapshots) and the
executable contract is in [`TRAINING.md`](./TRAINING.md#start-states).

**The matched retention run is done, and the reservoir won it.** Two PPO runs
from the same joint artifact, seed, optimizer, code fingerprint and update count
(204 updates, 1,259,520 environment steps each), differing only in start states
- twelve fresh lifecycles for the control, `fresh=6,policy=4,teacher=2` for the
reservoir - scored with `eval_closed_loop` on ten fresh, untouched 20,000-tick
worlds at the unused seed 900:

| Stage | Skill/tick, reservoir | Skill/tick, control |
|---|---:|---:|
| `empty` | 17.12 | 3.99 |
| `seed_creep` | 18.49 | 3.99 |
| `seed_full` | 13.13 | 4.05 |
| `seed_claimer` | 17.39 | 3.99 |
| `seed_outpost` | 16.59 | 4.00 |

The control also produced zero late-window remote harvest and zero late remote
staffed ticks in every stage, against 84-246 and 3-359 for the reservoir, and
zero claims against two. Its training trace is the plateau this mechanism was
built for: episodic return fell from 19,339 at update 40 to 7,931 at update 204
while entropy fell to 0.176, actor gradient norm to 0.05 and KL to 0.0005. The
reservoir run rose from 27,803 to 131,176 with entropy at 0.59 and KL at 0.002.

That is one seed pair on one map family, and neither run passes qualification:
claims remain almost absent and late-window home delivery is still near zero.
The reservoir is established as necessary, not sufficient.

1. **Multi-seed repeat.** Repeat the matched pair on at least three training
   seeds and a held-out map family before treating the size of the gap, rather
   than its direction, as established.
2. **Stratification parameters.** The phase boundaries at 2,000, 8,000, and
   15,000 ticks, the per-stratum capacity of 24, and the teacher-retirement
   thresholds of 32 policy records over 3 distinct events are declared, not
   derived. Measure whether a phase behaves as one distribution, whether 24
   states per stratum give diversity without pinning training to a few worlds,
   and whether retirement fires early enough to stop inheriting the teacher's
   distribution while the bridge is still useful.
3. **Throughput at 24 environments.** Track route-search time, route-cache hit
   rate, and route searches per transition as explicit metrics. Restored
   segments start with cold caches by design, so every segment boundary pushes
   work back into the pathfinder; without these counters a start-state schedule
   can pay for its coverage in wall time invisibly.

## Joint-policy ratio ablation

The current actor defines a valid joint team probability, but PPO clips one
ratio per creep. This is not known to be superior to clipping the full team
ratio.

Compare three objectives without changing data, initialization, optimizer, or
total update count:

1. **Full team joint:** sum all active factor log-ratios across all live actors,
   exponentiate once, and apply one PPO clip.
2. **Per actor:** sum factors within each actor, clip each actor ratio, and
   average live actors; this is the current implementation.
3. **Semantic groups:** group strategic/room decisions, creep task/target
   arguments, movement, and primitive work factors, then clip one ratio per
   group.

Log full and grouped log-ratio variance, clip fraction, approximate KL,
gradient norm, effective sample size, return, raw H+C, invalid actions, and
task collisions by live-creep bucket.

AlphaStar is useful precedent but not a verdict. It samples one enormous
structured command autoregressively, including a recurrent selected-unit
pointer, while splitting V-trace/UPGO corrections into action type, delay, and
other arguments because a full correction truncated traces too aggressively.
Its actors were asynchronous and off-policy, and one command targets a unit
subset rather than issuing a distinct simultaneous command to every unit.
Screeps PPO must therefore settle the grouping empirically.

## Action-conditioned latent ablation

The one-step latent dynamics objective clears its own qualification item:
holdout dynamics MSE beats the identity and the same-state legal counterfactual
baselines for both trunks. No measurement attributes a gameplay gain to it, and
no matched comparison exists, so its representation metrics improving is not
evidence that the policy plays better.

Run a matched NextLat-on/NextLat-off pair through pretraining and PPO with the
same corpus, initialization, seed, optimizer, and update count, and compare on
fresh 20,000-tick worlds under the stage-balanced scorecard. Report dynamics
MSE beside raw H+C so a representation gain that buys no gameplay is visible as
exactly that.

## Autoregressive coordination

Entity attention lets parallel creep heads respond to peer state, but it does
not let a later action condition on already selected same-tick assignments.
Test autoregression where it has the highest coordination value:

- a slow planner assigns persistent task, target, room, and priority to creeps;
- assignments are decoded in a stable but permutation-aware order;
- the fast tactical policy executes assigned tasks in parallel;
- deterministic reservation handles destinations and scarce targets;
- planner frequency is 10–50 ticks rather than one 64-step decode every tick.

Also benchmark fully autoregressive per-tick creep actions. Do not reject it
only on presumed latency; measure coordination gains, inference latency, order
sensitivity, and PPO ratio behavior.

## Temporal memory and options

Give every creep a persistent 12-element BF16 learned memory vector and the
empire/global state its own persistent 12-element BF16 vector. At tick `t`, the
actor reads pre-action memory with the observation, emits an environment action,
and writes the memory consumed at tick `t+1`.

Required semantics:

- bind creep memory to stable creep identity, never packed row index;
- zero-initialize on spawn, delete on death, and clear all memory on reset;
- exclude structure actors from creep memory;
- store pre-action memory and stable-ID mapping with each rollout transition;
- checkpoint live memory for interactive continuation;
- recompute writes during PPO rather than treating them as unmodelled actions;
- train contiguous sequences with burn-in and truncated backpropagation through time;
- report memory magnitude, write magnitude, reset counts, and gradient flow;
- test row reorder, death/respawn, room travel, reset, and checkpoint resume.

Learned memory complements rather than replaces an inspectable typed option:
task, target or room, priority, start tick, expiry, and continue/cancel/replace.
Option state must be part of observation and rollout state, not hidden only in
the executor.

## Strategic controller

Run an empire/room planner at a slower cadence over contextual global and room
tokens. It should choose:

- desired capability/body counts and replacement deadlines;
- per-tick spawn-body priorities and replacement deadlines, without a
  hidden queue or cross-tick energy reservation;
- build locations and structure types;
- source, sink, construction, controller, and defense priorities;
- scout, reserve, claim, and remote-room staffing plans.

Split room decisions, structure actions, creep task assignment, tactical
movement, and primary work/combat into explicit domains. Strategic construction
should not be an arbitrary creep side effect.

## Representation and critic

- Replace fixed global target packing with typed actor-local top-k candidates
  when saturation tests establish useful bucket sizes.
- Preserve deterministic overflow accounting and compile a small set of entity
  buckets rather than freezing capacity from the initial reset.
- Extend the implemented exact total/active body counts and store capacities with source regeneration,
  controller ownership/reservation, route costs, and congestion where current
  rows remain ambiguous.
- Add team, per-room, logistics, construction, and territorial value heads with
  PopArt or another principled target-scale mechanism.
- Evaluate counterfactual or difference-return auxiliaries for credit; keep the
  optimized external objective team-level.

### Dual spatial encoder ablation

The original expert review correctly identified duplicated spatial processing
as a likely inefficiency. Schema v2 fixed the more serious representation bugs:
the actor coordinates entities and the critic sees all entities. It did not
remove the duplicate room encoders.

Compare fully separate actor/critic trunks against shared patch tokens with
separate upper entity layers. Measure throughput and VRAM alongside policy
quality, critic EV, and actor-gradient interference. Sharing is worthwhile only
if it saves meaningful wall time without allowing value regression to damage
the actor representation.

## Data, curriculum, and generalization

- Replace the bootstrap teacher with a planner that robustly demonstrates
  mining, hauling, refilling, construction, upgrading, replacement, scouting,
  claiming, remote staffing, storage, and defense.
- Balance pretraining by stage, body composition, and intent; retain scarce examples rather
  than training and discarding non-IID chunks.
- Add DAgger on learner-visited states after closed-loop BC meets the teacher.
- Generate deterministic seeded room layouts and connected 2–4-room topology
  families with held-out seeds.
- Add visible-only scouting and remembered intel; keep pre-exposed neighboring
  rooms only as an explicitly privileged curriculum.
- Stage GCL2, GCL3, and GCL4 with 1, 4, 12, 24, 48, and 64 live-creep gates.

## Suggested sequence

```text
action validity + overflow diagnostics
  → stage-balanced evaluation and economic scorecard
  → start-state reservoir vs matched fresh-start-only PPO
  → NextLat on/off ablation
  → stronger planner teacher + closed-loop BC/DAgger
  → ratio-group and dual-encoder ablations
  → persistent options and learned memory
  → strategic planner + multi-room scenario families
  → longer release PPO run
```
