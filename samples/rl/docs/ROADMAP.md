# RL roadmap and open experiments

This file contains unresolved work. Implemented behavior belongs in
[`ARCHITECTURE.md`](./ARCHITECTURE.md), executable gates in
[`TRAINING.md`](./TRAINING.md), and settled conclusions in
[`DECISIONS.md`](./DECISIONS.md).

## Consequences of a single real teacher

Cloning only The International removes label sources we authored ourselves and
leaves three open items. None is a regression to be patched; each is a
capability the scripted planner faked.

1. **On-policy correction has no teacher.** DAgger relabelled learner-visited
   states with the scripted planner. A black-box bot cannot be asked what it
   would do in a state it never reached, so the scripted DAgger and outpost
   actor supplement corpora were deleted rather than kept as a second label
   source. The honest replacement uses machinery that now exists: snapshot a
   learner-visited state, restore it into an expert session, step the bot one
   tick, and capture its intents through the same labeller. Cost is one process
   boot plus one tick per correction, so it belongs in a batched offline pass,
   not the PPO loop. Until then, correction after cloning comes only from PPO.

2. **Body-length coverage is bounded by the simulated economy.** The old
   `ge16`-part coverage bucket was reachable only because a hand-written teacher
   could be told to emit any body in a 3,000-energy world. The real teacher
   spawns what its RCL affords: at the RCL1-RCL2 economy this stack simulates, a
   16-part body needs 800+ energy and never appears. Its measured contract
   bodies span 300-850 energy and 5-11 parts. Recovering that bucket needs a
   seeded RCL6+ scenario with extensions, not a relaxed gate; the gate records
   the unreached bucket explicitly so it can be promoted back to fatal.

3. **Placement supervision is now the teacher's, not ours.** Construction tile
   labels come from where the bot actually built. That is a real placement
   distribution rather than an arbitrary legal tile, and it is the first time
   this stack can measure whether placement is learnable at all. Whether the
   policy reproduces the teacher's base plan is an open measurement.

## Immediate learning blockers

Before another substantial PPO continuation:

0. **Investment behaviour is unfunded, and the cause is not established.**
   The credit window was the first candidate and it has now been changed, but
   nothing has been measured under the change yet. `gamma=0.995` with a single
   `lambda=0.95` gave a credit window of `1/(1-gamma*lambda)=18` ticks — the
   discount alone allowed 200 — on a 20,000-tick problem where construction and
   claiming repay over thousands. The objective now runs `gamma=0.9995` with
   VAPO decoupled GAE: a length-adaptive `lambda_policy` worth a 677-tick
   window at a full 2048-transition segment (the largest horizon host RAM allows
   at four live rooms), and a `lambda=1` critic target
   whose window is `1/(1-gamma)=2000`. Whether that funds investment is an open
   measurement, not a finding.

   The obvious counterexample has to be accounted for by any explanation:
   spawning is delayed payoff too, since a body costs about 300 energy and
   repays over a ~1,500-tick life, and it was retained under the same discount
   (`spawnCreep` intents at 0.85x of their early rate, 27-35 creeps sustained).
   Remote harvesting has a round trip of hundreds of ticks and grew. The second
   candidate was specific to construction: placement was never supervised,
   because the hand-written teacher's labels carried an arbitrary legal tile, so
   a built structure's expected payoff could be near zero at any horizon. The
   teacher cutover removed that candidate at the source - construction labels now
   carry the tile The International actually built on - so the next matched run
   tests the credit window against a real placement distribution for the first
   time.

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

   The two candidates are now asymmetric rather than tied: both halves of the
   diagnosis have been applied, the credit window by the objective change and
   placement by the teacher cutover, so the next run cannot attribute an
   improvement to either on its own. Placing a teacher-quality base and
   measuring whether throughput improves within a few hundred ticks still
   isolates the placement half directly. Run it on the same matched protocol,
   one change at a time,
   before adding an investment-aware auxiliary value head or the temporal
   abstraction in [`DECISIONS.md`](./DECISIONS.md).
1. **Observation caps, diagnosed; the remaining question is whether to raise
   them.** The diagnosis asked for here is done. Per-tick cap instrumentation
   over five 20,000-tick teacher episodes says room pressure is not production
   pressure: with the teacher bounded to the observation radius, **no owned or
   mined room was dropped at any tick**, and the 29% of ticks that lose a room
   slot lose a wandering scout (0.4-0.6 hidden creeps per tick). Stake-ranked
   room slots, not name order, are what make that true.

   What actually binds is population: live creeps reach 111-118 against 64 creep
   slots, first exceeding them at tick 2,666-3,638, and candidate targets exceed
   128 from tick 5,859-7,492. So the open item is a cap decision with a measured
   cost, not a diagnosis: raising `maxCreepActors` 64 -> 128 and `maxTargets`
   128 -> 256 roughly doubles actor rows and quadruples the actor-target
   attention cost, against a host rollout footprint already projected by
   `HostRolloutBuffer.projected_bytes`. Decide it from the VRAM probe plus a
   matched run, and keep the honest reporting either way: `droppedOwnedRooms`,
   `droppedCreepOnlyRooms`, and `hiddenCreepActors` distinguish data loss from
   correct budgeting, and `roomOverflow` fires only when a room with stake is
   dropped.

   The cap decision now also gates start-state coverage, which is new
   information: a bridge snapshot is admitted only if the restored state is
   fully representable, so the caps decide how late a teacher state can be
   captured at all. A five-env, 20,000-tick teacher collection kept 182 states
   and **none past tick 10,305**: 94 `early`, 82 `mid`, 6 `late`, and an empty
   `endgame` stratum, so the phase the horizon ends in has no teacher bridge at
   all. Per-tick cap instrumentation over the
   same five episodes puts 88.1% of teacher ticks over at least one cap, with
   medians and maxima of 56-96/133 live creeps (64 slots), 90-205/237 candidate
   targets (128 slots), and 4-5/6 visible rooms (4 slots). Covering the full
   lifecycle therefore needs `maxCreepActors` 136, `maxTargets` 256, and
   `maxRooms` 6, and `maxStructureActors` 36 is unused headroom - spawns plus
   towers account for 2-6 actors at these levels - so it can pay for creep
   slots. Until that is decided, the teacher bridge supplies mid-game states
   only, and the late-window retention the reservoir exists to provide has to
   come from policy snapshots.

   Creep-slot admission is also arbitrary today: `encode.mjs` admits creeps in
   lexicographic name order and stops at the cap, so which of the player's own
   creeps become uncommandable is a name lottery rather than a relevance
   ranking. Stake-ranked room slots already do this correctly; creep slots do
   not. Fix the ordering in the same change as the cap decision, because both
   alter the observation ABI and each one alone forces a re-collection.
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
5. **Late-game remote staffing decays, and the real teacher is unmeasured.**
   The measurements here were taken on the hand-written planner, which is no
   longer a teacher: at seed 7 its seeded outpost worker survived to tick 1,497
   but stopped delivering after tick 1,200, and no remote harvest or delivery
   happened after tick 1,600
   (`test_outpost_route_finisher_and_replacement_remain_productive` still fails
   on its aging-finisher assertion, and that test now describes the evaluation
   baseline only). The learned policies showed the same late-activity collapse,
   so the question survives the cutover: whether The International restaffs a
   remote outpost through the full horizon is an open measurement. The corpus
   already fails closed on it - `_validate_outpost_teacher_readiness` requires
   remote harvest, home delivery, and staffed and productive ticks inside the
   final collection window of the `seed_outpost` split - so a decaying teacher
   blocks collection rather than silently teaching decay.
6. **Teacher coverage ends where the observation stops being faithful.** The
   earlier rejection rate (10,466 of 12,000 candidate ticks, 22 usable states)
   was measured against a stock teacher that outgrew the ABI on almost every
   tick. With the teacher bounded to the observation radius and the wall-clock
   leak fixed, the faithful prefix is 570-2,700 ticks and no production room is
   ever dropped, so the bridge lane is usable through `early` and into `mid` -
   but `late` teacher states remain unreachable until the creep and target caps
   in blocker 1 are decided. The same caps apply while labelling: a creep whose
   actor row is truncated cannot be supervised and the labeller drops it, so
   measure labelled-decision counts and drop reasons per economy phase, not only
   overflow fractions. Recollect bridge states and report the new admission
   rate; the old one describes a teacher that no longer exists.
7. **The native concurrency crash no longer reproduces, and the cause was never
   found.** Four concurrent expert sandboxes once killed an environment at tick
   5,211 with `free(): invalid pointer`. Re-measured after the reproducibility
   work (virtual player clock, canonical room order, serialized finalizers),
   sixteen concurrent expert sandboxes ran 8,000 ticks with no crash, twice,
   sustaining 228-306 env-ticks/s aggregate while growing 18-105 creep colonies.
   Collection now runs at sixteen environments on that evidence. Since no root
   cause was identified, a repeat is possible: a collection that dies in a native
   allocator is this blocker returning, not a new fault, and the width should be
   halved and the crash captured under a debugger rather than retried.

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
4. **Lane-mixed reward normalization.** One running RMS normalizes rewards
   across every lane, and a restored mature colony earns far more per tick than
   an empty room, so the shared scale is set mostly by late states and shrinks
   early-game advantages. That may be harmless or even desirable for retention,
   but it is unmeasured: log per-lane reward mean/RMS and per-lane advantage
   magnitude, and only then decide between the shared normalizer, a per-lane
   normalizer, and leaving advantage scale to the critic.

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

- The bootstrap teacher is now a real bot rather than a planner we maintain, so
  the demonstration question becomes measurement: record which of mining,
  hauling, refilling, construction, upgrading, replacement, scouting, claiming,
  remote staffing, storage, and defense The International actually exhibits per
  curriculum, and treat an absent behaviour as missing coverage rather than a
  planner bug to fix.
- Balance pretraining by stage, body composition, and intent; retain scarce examples rather
  than training and discarding non-IID chunks.
- Add snapshot-restore DAgger once closed-loop BC meets the teacher: restore a
  learner-visited state into an expert session, step the bot one tick, and label
  through the same capture path. See "Consequences of a single real teacher".
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
  → snapshot-restore DAgger on learner-visited states
  → ratio-group and dual-encoder ablations
  → persistent options and learned memory
  → strategic planner + multi-room scenario families
  → longer release PPO run
```
