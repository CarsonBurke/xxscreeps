# RL decisions and evidence

This is the durable record distilled from the original expert corpus, red-team
reviews, implementation waves, and measured runs. It records conclusions that
should survive individual experiments. Open questions belong in
[`ROADMAP.md`](./ROADMAP.md).

## Correctness before optimization

### Observations are post-tick

The original environment returned the state from before an issued action was
executed, so the next policy decision was applied to a different state from the
one it observed. The server now applies actions, advances the simulator, then
encodes the resulting observation and reward. Terminal observations are
preserved for value bootstrap and trajectory chains are cut correctly.

This alignment is load-bearing. Throughput changes must not reintroduce a
pre-tick observation path.

### Legality is part of the action definition

Unused action arguments do not enter likelihood or entropy. Candidate masks,
model compatibility, executor validation, and engine behavior must describe the
same executable action. A legal type with no executable argument is not legal.
Teacher-invalid and engine-invalid actions are hard failures, not labels to
drop silently.

This conclusion came from concrete failures involving targetless intents,
unaffordable body aliases, target-independent resource amounts, self-transfer,
miner and tower candidates, construction cross-products, and cross-room target
semantics.

### Never silently truncate required state

The old 24-actor/48-target model could drop creeps, then spawns and towers, while
freezing the active room count from reset. Schema v2 separates 64 creep rows
from 36 structure actors, balances 128 targets, carries room coordinates, and
reports overflow. Capacity is still a tested contract, not proof that a
scenario fits.

The creep cap is a control limit, not only an observation limit. One action is
issued per admitted actor slot, so a creep with no slot receives no intent and
idles for that tick. A colony larger than `maxCreepActors` is therefore not
merely partially observed, it is partially uncontrolled, and its idle creeps
still consume the energy that built them. That makes the cap a bound on the
largest economy any policy in this ABI can run, which is why cap sizing is a
design decision measured against teacher colony size rather than a tuning knob.

## Representation

### Entity state is primary

The original actor saw each creep in isolation and the critic ignored actor and
target tables. This made task allocation unrepresentable and value prediction
blind to cargo, body, TTL, structures, and target state. Schema v2 keeps spatial
room tokens but jointly contextualizes global, room, actor, and target tokens.

The expert corpus was right that early Screeps economies are dominated by a
small number of typed entities and logistics relationships. Larger image
encoders do not substitute for explicit entity state, categorical embeddings,
absolute capacities, and actor-target compatibility.

### Dual encoders remain an open tradeoff

Independent actor and critic trunks avoid value-loss interference but process
the same room observation twice. Entity awareness fixed the old critic bug; it
did not resolve duplicated spatial compute. Shared lower spatial tokenization
with separate upper layers must be benchmarked rather than accepted or rejected
by parameter count alone.

## Action hierarchy

### One executable action domain beats duplicate generic slots

The old two-slot head could emit conflicting Screeps primary intents even
though the engine executes only one. Schema v2 emits one intent with only its
required arguments. Navigation assistance turns a selected target into one
deterministic movement or work step.

Macros shorten the tactical horizon, but repeated stateless macro selection is
not strategic memory. Persistent tasks and room-level decisions should be
explicit model state.

### Autoregression and PPO grouping are separate choices

A chain-rule autoregressive policy can define one enormous joint action
probability. The exact score-function gradient sums all active conditional
log-probabilities under the team advantage. That does not determine where PPO
or off-policy importance ratios should be clipped.

AlphaStar supports both sides of this distinction: it sampled an enormous
structured command autoregressively, including selected units, while splitting
V-trace and UPGO corrections into semantic groups because a full correction
truncated traces too aggressively. Screeps must compare full-team, per-creep,
and semantic-group objectives empirically.

## Learning pipeline

### One real teacher, supervising only what it demonstrably chose

Behaviour cloning learns from The International running inside the engine, and
from nothing else. The hand-written planner in `env/scripted_baseline.mjs`
remains only as an evaluation baseline (`agent/eval_scripted.py`); it produces
no training label, no qualification expectation, and no corpus row.

A hand-authored teacher can be told to emit any body, any placement, and any
role assignment, which makes its labels a description of its author rather than
of competent play. Cloning it teaches the policy to reproduce our own guesses,
and every gate built on those guesses then certifies agreement with the guess.

The cost is that supervision becomes partial. A real bot issues raw engine
intents conditioned on private memory and plans, so most of what it does is not
one of our macro actions. Measured over 2,000 expert ticks, 83.8% of its
decisions are exactly representable; the remainder are raw movement en route
(retro-labelled to the macro action they serve when that action completes),
multi-intent ticks, and amounts our bins cannot express. Every row therefore
carries an eligibility mask over action factors, and the loss supervises only
eligible factors with `safe_bc_nll(..., strict=True)`. Unlabelled factors stay
free for PPO.

Labels are validated against the same tick's candidate masks before they are
retained. A label the engine accepted but our observation cannot express is
dropped rather than admitted by widening a mask: the teacher's intent is
evidence about the world, not about what this ABI can represent.

Learner-visited relabelling (DAgger) is not available from a black-box bot,
because the bot cannot be asked what it would do in a state it did not reach.
The scripted DAgger and outpost supplement corpora were removed with the
scripted teacher rather than kept as a second label source. Restoring a
learner-visited snapshot into an expert session is the honest way to recover
on-policy correction and is specified in `ROADMAP.md`.

### A silent teacher is a broken teacher, and it must fail closed

The International aborts its whole tick when a memory segment it expects is
absent, and the abort is load-sensitive: two runs of one declared seed produced
660 and 2 labels, with byte-identical bot logs in both. Silence is the only
observable, so `TiLivenessGate` grades every episode on it.

Thresholds are measured, not guessed. In a healthy 3,000-tick episode the
expert acted on 2,974 ticks (99.1%) and its longest silence was 12 ticks, ending
at tick 20 while the first creep was still spawning; every later silence lasted
a single tick. The gate therefore ignores the first 50 ticks, fails an episode
that goes 30 consecutive graded ticks without an engine intent, and fails a run
that acts on under 50% of graded ticks. Stalled runs measured 2.5%-20%.

The persisted corpus carries the aggregate, and loading validates it, so a
corpus collected from a stalled teacher cannot be trained on later.

The expert wire reports no learner-intent counters at all: `intentIssued` and
`intentInvalid` measured exactly zero over 1,500 expert ticks, because the bot's
intents are executed inside the engine rather than through our action path. The
corpus gate that required them to be zero was therefore vacuous and was replaced
by the liveness gate. Those counters remain in the scorecard as readings of the
learner path, which is empty during teacher collection.

Silence has a second cause with the same symptom, and separating them needs the
world, not the bot's log. A healthy `seed_full` world lost its spawn at tick
6,997 to an NPC invasion: 13 creeps then idled with full sources and no way to
rebuild a 15,000-progress spawn, the teacher issued no intent for 31 ticks, and
collection failed at tick 7,200 of 20,000 after 25 minutes. Nothing in the bot's
console showed it. `TiLivenessGate` therefore records the trailing world states
(`spawns`, `creeps`, `storedEnergy`, `rclMax`, `controlProgress`) and prints them
with the failure, and `spawns` is carried in the observation metadata for exactly
this purpose. A stall that reports `spawns=0` is a dead colony, not a broken bot.

### The teacher is configured to the observation, and collection asserts it

A teacher whose colony outgrows the observation labels actions whose subject the
policy cannot see. Stock The International runs remotes to distance 5 and scouts
outward without bound; measured over three 20,000-tick episodes it reached 7-8
visible rooms and 111-122 live creeps against 4 room slots and 64 creep slots,
dropped a staked room on 74-86% of ticks, and left 3.5-5.5 creeps per tick
outside the action space.

`maxRemoteRoomDistance` and `maxScoutRoomDistance` are therefore both set to 1
in the teacher build. Bounding exploration is not cosmetic: with it, two further
20,000-tick episodes dropped **no** owned or mined room at any tick, hidden
creeps fell to 0.4-0.6 per tick, and room-slot pressure fell from 83-86% of
ticks to 29% - all of it wandering scouts rather than production.

`schema.json:teacher` declares both radii, and `validate_teacher_configuration`
reads them back out of the executed bundle (`dist/main.js`), not the source
tree, before any collection or bridge-snapshot run starts. Corpus meta records
`ti_room_radii`. Changing either radius changes what a corpus means and must
bump `teacherAbi`.

What remains unrepresentable is the mature economy, not the map: the creep cap
binds first at tick 2,666-3,638 and the target cap at 5,859-7,492, so the fully
faithful prefix of a bounded-teacher episode is roughly 570-2,700 ticks. This is
why teacher states are a bridge lane with a hard admission gate rather than the
training distribution, and why raising `maxCreepActors`/`maxTargets` is a
measured experiment in `ROADMAP.md` rather than an assumed fix.

### A reproducible teacher needs a virtual clock and canonical room order

Two runs of one declared seed diverged at tick 45 and ended 33 versus 66 creeps.
The engine and the scripted baseline were byte-identical over 1,500 ticks, so the
divergence was specific to running a real bot, and three host leaks were found:

- **Wall clock.** The player realm read host time. The International stores
  `Date.now()` deltas in `Memory.stats` and branches on them, so a replay billed
  whatever the host was doing. `deterministicClock` now installs a virtual clock
  that advances one simulated second per tick, covering `Date.now()`, the `Date`
  constructor, and `performance.now()` through a proxy that leaves every other
  `Date` overload intact.
- **Room order.** Visible and intent room sets arrived in storage order, which
  is the order concurrent room finalizers happened to write them. That order
  becomes `Game.rooms` key order, and bots iterate their rooms. Both sets are
  now sorted before they reach the sandbox.
- **Finalizer order.** The simulated shard finalized rooms concurrently, so id
  allocation and shared-state writes interleaved by host scheduling. Finalizers
  now run in canonical room order.

With all three fixed, a 400-tick teacher twin is bit-identical and 20,000-tick
growth changed materially (creeps at tick 2,000 fell from 67 to ~35), but the
teacher is still not fully reproducible: bisecting world digests against captured
intents puts the first disagreement at a tick where the observed world and the
engine telemetry are identical and only the bot's intents differ. The residue is
inside the player realm, and the bundle imports no entropy (no `crypto`,
`WeakRef`, `FinalizationRegistry`, or randomness import in its WASM), so it is
not attributable to a host input we control.

The consequence is a contract, not a caveat: **teacher trajectories are samples,
not replays.** Corpora record `ti_runtime_source_sha256` and their measured
liveness rather than promising reproduction, and matched teacher comparisons need
repeated seeds. What the reservoir depends on is the learner path, which *is*
reproducible: a teacher-sourced snapshot reopened in a learner session with no
expert code replays bit-identically, asserted by
`test_teacher_state_replays_identically_in_a_learner_session`.

### Optimize H+C; gate the rest independently

Only harvested energy and controller progress enter the scalar reward. Delivery,
construction, claims, spawning, storage, and waste remain diagnostics and
qualification gates. Gross transfer is reversible and was demonstrably
farmable; construction volume does not establish placement quality; fixed claim
bonuses can dominate the economy. Keeping these outside reward prevents proxy
success from replacing the product score.

Economic event accounting must still be conservation-aware. Protected sinks
cannot be withdrawn, and scorecards use actual post-tick effects rather than
issued intents.

### Long-horizon problems need temporal abstraction

Increasing rollout length does not give a feed-forward transformer temporal
context. Economic investments span hundreds or thousands of ticks. Gamma
0.995 helps, but task options, persistent learned memory, strategic cadence,
appropriate sequence training, and stage-aware value diagnostics address the
root problem more directly.

### Rollout length is not temporal coverage

Environments that all start at tick zero and advance in lockstep give each
update a batch drawn from one narrow band of a 20,000-tick timeline. A phase is
entered once, trained on for as many updates as its width allows, and then left
behind for good, while any environment that collapses rewinds to the opening.
Behavior cloning can demonstrate remote harvesting, hauling, claiming,
replacement, and recovery, and PPO will still unlearn them, because the states
where those behaviors matter stop appearing. The distribution of start states
is a first-class training decision, distinct from rollout length, discount, and
the temporal-abstraction work above.

Start states must include pre-decision states. Capturing only after the teacher
chose a body, placed a structure, or claimed a room trains execution of a
decision the learner never made and hides the strategy itself. Pre-spawn and
pre-claim states leave the decision open, which is the point of collecting
them.

Untouched full-lifecycle environments always remain in the mixture. A policy
trained only from restored states learns to operate an inherited colony rather
than build one, and loses the only worlds whose late states follow from its own
earlier decisions instead of the teacher's.

Teacher bridge states are scaffolding. They exist because the learner cannot
yet reach some phases, they are collected once into an immutable set, and they
retire per phase as soon as the policy supplies its own examples for that
phase. A permanent teacher lane would pin training to the teacher's state
distribution.

Evaluation never uses snapshots. Qualification stays on fresh, untouched
20,000-tick worlds, because a policy scored from restored states is never
required to reach them.

### One optimizer, spectral on the trunk, adaptive on everything else

Pretraining and PPO use the same hybrid: Muon on the hidden attention and
feed-forward matrices, fused AdamW on embeddings, normalization, biases and all
heads. Two optimizer implementations in one repository would make a pretrained
trunk and a PPO-updated trunk answer to different update geometry, so
`samples/rl/agent/muon.py` is the only one.

The Muon itself is the tuned implementation from `modded-nanogpt`, reduced to
one device and float32 parameters: Polar Express orthogonalization, NorMuon
second-moment reweighting with the update norm renormalized, cautious decay
fused into the update, and same-shape matrices stepped as one batch. Measured
on condition-1e3 matrices, five Polar Express rounds bring the mean singular
value of the update to 1.00 and the smallest to 0.21; five Newton-Schulz rounds
with the classic coefficients reach 0.86 and 0.09. Dropped on purpose:
distributed banks, bfloat16 mantissa tracking, which is meaningless for float32
parameters, and every schedule, including learning-rate decay.

PPO's Muon rate is RMS-matched to the AdamW step it replaced rather than
inherited from pretraining, so switching optimizers is a geometry change that
can be attributed, not a step-size change wearing a new name.

### Deterministic evaluation understates the policy it scores

`eval_closed_loop` and the showcase recorder both decode argmax, while training
rollouts sample. Any behaviour that survives PPO only as a low-probability
action is therefore invisible to them, and reporting it as absent is a
measurement error, not a finding.

Measured on the joint-pretrain artifact over 600 ticks of `empty`, same seed:

| decode | build energy | createConstructionSite | build intents | harvest | delivery |
|---|---:|---:|---:|---:|---:|
| deterministic | 0 | 0 | 0 | 2,712 | 1,443 |
| sampled | 320 | 2 | 16 | 1,172 | 394 |

Construction sits below the argmax threshold even before PPO. The training
tensorboard shows what actually happens to it: `actions/intent_build_issued`
starts near one intent per environment tick (6,220 per update averaged over the
first ten updates of the reservoir run) and decays to 182 by update 205, a 34x
suppression, with the fresh-start control decaying 80x. So PPO suppresses
construction rather than erasing it, and a deterministic score cannot see the
difference.

Two consequences. Reported behaviour counts must name their decode. And the
sampled column is not free: sampled play built, but harvested 2.3x less over the
same window, so decode choice trades economy against behavioural coverage.

### Strict artifacts are part of correctness

Checkpoints carry semantic schema, ABI identifiers, complete model state,
optimizer state, normalization statistics, counters, and RNG state. Partial
pretraining cannot enter PPO. A PPO continuation restarts environments and
reset-local traces, so it preserves optimization state rather than the exact
live trajectory.

## Deliberate exclusions

- The International remains evaluation-only because it does not emit labels in
  the current factorized action contract. Inferring those labels from engine
  intents is a separate inverse-dynamics project.
- A JEPA/world-model or latent THINK/EMIT system is not justified before the
  observable economy, action legality, imitation, memory, and evaluation stack
  works. It may be reconsidered only against a concrete partially observed
  planning bottleneck.
- The critic uses HL-Gauss categorical return prediction on a
  symlog support. Its ±1e9 anchors and approximately ±1.594e9 hard bounds leave
  ample headroom for much stronger economies; overflow fails loudly. This does
  not replace held-out explained-variance and target-range monitoring.
- A start-state snapshot excludes two stores that are inert in this harness and
  must stay that way. User `Memory` is not captured, so a restore always boots a
  learner session and never resumes bot code with a stale world model; and the
  persisted global control level is reseeded on every boot, which is only safe
  because 20,000 ticks cannot fund a third room. Restoring a teacher state *as* a
  teacher, or raising the room capacity, would require capturing both.
- NPC invasion waves are disabled in the RL environment (`game.invaders: false`,
  re-enabled per process with `RL_INVADERS=1`). An invasion is scheduled by a
  room's cumulative harvest against `INVADERS_ENERGY_GOAL`, so every productive
  20,000-tick episode meets one in its second half, and this ABI supervises no
  defense: the teacher does not build a tower at RCL3, and the measured outcome
  was spawn loss and terminal colony decay in one of sixteen worlds. Keeping the
  wave would inject a large unattributable negative tail into the value function
  and fill the late-game corpus with a collapsing colony - the opposite of the
  remoting and claiming behaviour the corpus exists to preserve. Defense enters
  scope with tower actions, defender bodies, and a defense-aware teacher, and
  invasions return with it.

## Current experimental evidence

- The schema-v2 synthetic suite passed 31 learning and architecture contracts
  at the time of the first PPO continuation.
- The first eager PPO run completed 55 updates and 112,640 environment steps
  with complete optimizer, reward-normalizer, and RNG state.
- On one fixed empty-room seed with sampled decoding, the historical reward-ABI return increased
  from 6,372.3 after joint pretraining to 8,315.0 after PPO, a 30.5% gain.
- Deterministic return did not improve, and the evaluation did not establish
  multi-seed, multi-stage, construction-quality, or economy-funded expansion
  competence.
- The same run exposed extensive invalid transfer attempts and observation
  overflow. Those are blockers, not incidental metrics.
- Joint pretraining `samples/rl/runs/joint_pretrain_nextlat_fused_bootstrap16.pt`
  (authority `lifecycle_primary_fused_bc_value_nextlat_dagger_rare_spawn_v2`, no
  actor supplement) reached closed-loop control rate 1.00, nonzero remote
  harvest in every curriculum, and lifecycle train NLL 0.060.
- The later run with the outpost actor supplement
  (`samples/rl/runs/joint_pretrain_nextlat_outpost16.pt`, `--actor-supplement`,
  authority `lifecycle_primary_fused_bc_value_nextlat_correction_rare_spawn_v3`,
  same corpus `dea70f38…`, same seed 3, same 16 global epochs) regressed to
  control rate 0.59, produced zero remote harvest in `empty`, `seed_creep`, and
  `seed_full`, and worsened lifecycle NLL to 0.139.
- Those two runs are the measured case against the correction lanes. Both were
  relabelled by the hand-written planner, and the supplement made every
  gameplay metric worse. The lanes have since been deleted with the scripted
  teacher, so the entries above are the final record of them rather than a
  configuration choice. Start-state reservoir runs initialize from a joint
  artifact collected without them; in this workspace that artifact is
  `joint_pretrain_nextlat_fused_bootstrap16.pt`.
- The matched start-state pair is the first controlled PPO comparison in this
  stack. Both runs resumed `joint_pretrain_nextlat_fused_bootstrap16.pt` at seed
  3 with an identical source fingerprint, the same hybrid Muon optimizer, twelve
  environments, minibatch 1536, `--compile`, and were stopped at the same
  checkpoint: update 204, global step 1,259,520. Start states were the only
  difference. Scored deterministically on ten fresh untouched 20,000-tick worlds
  at seed 900, the reservoir run reached 13.1-18.5 harvest-plus-control per tick
  across the five curricula against the control's 3.99-4.05, with the control at
  zero late-window remote harvest, zero late remote staffed ticks and zero
  claims in every stage. The control's own training return fell from 19,339 to
  7,931 while its entropy, actor gradient norm and KL all collapsed toward zero:
  the lockstep plateau, reproduced deliberately.
- Cost, so the tradeoff is on record: restored mature colonies simulate far more
  slowly than fresh ones. Collection ran at 141-227 environment steps per second
  on restored late-game states against 700-930 on fresh worlds, so the reservoir
  run needed roughly twice the wall time for the same 204 updates.

These measurements are experiment records, not release qualification.
