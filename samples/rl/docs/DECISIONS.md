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

### Imitation first, then closed-loop correction

Sparse long-horizon economy behavior is unlikely to emerge efficiently from a
random policy. The actor and critic pretrain on the same scripted stream, then
the learned actor must demonstrate the behavior in closed loop before PPO.
Training-set NLL or teacher return cannot qualify a learned policy.

The bootstrap teacher is not an oracle. It must cover every required body composition and stage it
is used to certify, and learner-visited states should be relabelled through
DAgger once basic imitation works.

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
- The optional actor-supplement lane is therefore not enabled for PPO
  baselining. Start-state reservoir runs initialize from a joint artifact
  collected without `--actor-supplement`; in this workspace that artifact is
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
