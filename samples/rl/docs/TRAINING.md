# RL training and evaluation

This is the executable schema-v4 training contract. Model structure lives in
[`ARCHITECTURE.md`](./ARCHITECTURE.md); unresolved experiments live in
[`ROADMAP.md`](./ROADMAP.md).

All local training and GPU evaluation must go through `mlq`. Do not promote a
checkpoint from training loss or shaped return alone.

## Objectives

Training intentionally uses only harvest and controller progress:

```text
r_train = 0.10 × harvested_energy
        + 1.00 × controller_progress

score_eval = harvested_energy + controller_progress
```

Delivery, construction, claims, spawning, and storage flows remain post-tick
diagnostics and qualification gates. They do not enter the scalar objective:
gross delivery is reversible, construction progress can reward poor placement,
and a fixed claim bonus can dominate economic quality. Protected sinks remain
non-withdrawable because reversible logistics would still corrupt the teacher
and scorecard. Report reward plus raw H+C, delivery, construction, claims,
waste, invalid actions, and overflow.

## Readiness gates

Before collecting a serious pretraining or PPO run:

- all synthetic learning, action, schema, artifact, and architecture contracts pass;
- the fixed-seed 20,000-tick scripted economy expands from empty with zero engine-invalid teacher intents;
- cross-room navigation and reserve/claim execute in the real engine and emit the claim diagnostic;
- rewarded delivery cannot be recycled through a legal withdrawal path;
- the current semantic schema and all ABI identifiers match the artifact exactly;
- required population, room, and candidate buckets have no silent overflow;
- evaluation records curriculum, scenario seed, action seed, decoding mode, and raw metrics.

Run the engine contracts after changing the encoder, executor, reward,
simulator, spawn-body contract, or teacher:

```bash
export RL_NODE="$(mise exec node@24 -- which node)"

python3 -m samples.rl.agent.test_latent_unit
python3 -m samples.rl.agent.eval_scripted \
  --ticks 20000 --max-episode 20000 --seed 3
python3 -m samples.rl.agent.eval_expansion --ticks 500
python3 -m samples.rl.agent.eval_reward_contract
```

Aggregate reward is insufficient. The scripted gate must establish zero
rejected labels and actual post-tick resource effects, not merely successful
navigation toward a target.

## Same-stream pretraining

The International supplies the strongest value/representation trajectories and
an exact conservative actor-label subset captured from raw engine intents.
In-range target commands, construction, and spawn compositions are used only
when they map exactly. Spawn ordering is included only for bodies already
grouped into contiguous part-type blocks; interleaved bodies supervise their
exact counts but not a different order. Immediate moves, concurrent commands,
and unsupported commands are rejected, never relabeled as `none`. The scripted
planner still supplies complete one-slot macro labels and the end-to-end
expansion qualification stream.

```bash
mlq submit --name screeps-ti-critic --max-parallel-runs 1 --cwd "$PWD" -- \
  python3 -m samples.rl.agent.pretrain_ti \
    --steps 6000 --validation-steps 6000 --min-validation-ev 0.01 \
    --max-validation-mae-ratio 0.9 \
    --device cuda --seed 3 \
    --save samples/rl/runs/ti_critic_seed3.pt
```

The TI critic is an optional experiment, not a prerequisite. It qualifies only
on a fresh full-length TI trajectory where explained variance is positive and
absolute error beats the constant-return baseline by at least 10%. The long
horizon makes endpoint bootstrapping negligible at `gamma=0.995`; a low
training loss or a biased, nearly constant prediction remains insufficient. Do
not pass an unqualified artifact into joint pretraining.

Behavior cloning and critic adaptation consume a reusable immutable corpus:

```text
scripted state/action/reward         independent TI reward streams
        |                    |                    |
        v                    v                    v
per-factor masked CE     discounted reward-to-go (zero endpoint)
        |                    |                    |
        +--------------------+--------------------+
                             v
                   complete joint-pretrain artifact
```

The standalone collector first completes the whole scripted and TI lifecycles.
A bounded
per-stage/state/action reservoir preserves examples across early, middle, and
late economy phases; every optimization epoch globally shuffles the scripted
corpus for actor BC and final critic fitting. Returns are full finite-horizon reward-to-go,
not bootstrapped from the critic being trained. A separate full-lifecycle fleet
under independent seeds is never optimized and supplies the promotion EV. TI
critic rows have their own independent full-return train and holdout splits and
are used once to initialize value representation before actor-aligned scripted
fitting. Repeating their behavior-policy-dependent returns would create a
contradictory value target. Unlabeled TI rows never enter actor BC. Exactly
representable TI action factors remain a separate actor-only auxiliary lane.
Corpus storage schema v3 also retains a separately bounded temporal replay
directly from
same-tick `(observation, complete joint action, next observation)` triples for
both scripted train and independently seeded holdout fleets. It never joins
reservoir rows after collection. Episode-ending rows are excluded before the
vector environment's reset observation can become a false target, with terminal
and truncation exclusions recorded in the content hash and manifest provenance.
Temporal rows store only the nine planes consumed by the two state encoders;
policy masks and construction masks are not duplicated into either frame.
The production curriculum includes `seed_outpost` explicitly; `empty` alone
does not guarantee neutral-outpost remote-harvest and home-return trajectories.
Qualification therefore requires the teacher corpus to contain remote harvest,
home delivery, productive staffing, and no claim or remote ownership for that
stage. The outpost closed-loop success rate uses the same definition on unseen
`seed_outpost` environments; it is not an expansion or claim-success metric.
Both teacher and learned closed-loop gates additionally require harvest, home
delivery, and current staffed/productive ticks in the final collection window;
historical cumulative totals or peaks cannot hide a colony that stopped replacing
its remote worker. Aging route finishers remain responsible for delivering their
visible cargo while a separately chosen viable successor takes over staffing.
Each global pretraining epoch trains the actor and critic action-conditioned
latent dynamics on temporal train rows only. Future encodings and the critic
value-head probe are detached. Qualification reports learned, identity, and a
same-state legal counterfactual holdout MSE: one actually issued command is
replaced by canonical `none`, preserving the actor and target tables. Both
trunks must beat identity by at least 1%, and removing the command must worsen
prediction by at least 1% over at least 128 holdout rows. Pretraining
coefficients are explicit and
separate from PPO (`actor=1.0`, `critic=1.0`, detached critic probe KL `=0.1` by
default); they are checkpointed and resume-checked rather than silently reusing
the PPO auxiliary weights.
Illegal or non-finite
teacher labels fail collection rather than being silently discarded. PPO later imports
the actor exactly and the critic trunk only because normalized PPO values use a
different scale from raw pretraining returns.

Real teacher spawn transitions also enter an energy × body-length × composition
balanced actor replay lane. Intent, the eight count decisions, and positive-type
ordering are equally weighted semantic losses, so a many-part body does not
dominate the spawn/wait decision. Dedicated engine-backed scenarios cover long and
expensive bodies; they augment but cannot replace the empty-world expansion
gate. The corpus is content-addressed and optimizer-independent. Training
checkpoints record its path-independent SHA-256 and a dedicated shuffle RNG;
resume refuses a different corpus ID.

Pretraining and PPO share one optimizer, `samples/rl/agent/muon.py`. Muon
updates only the hidden attention and feed-forward matrices; AdamW (fused on
CUDA, zero weight decay) keeps embeddings, input projections, normalization and
bias parameters, every action head, and the entire HL-Gauss head. Momentum
warms from `0.85` to `0.95` over 300 optimizer steps and a `0.025` cautious
weight decay applies only where the Muon update already shrinks a coordinate.
There is no learning-rate schedule anywhere in this stack.

The Muon implementation is ported from `modded-nanogpt`, reduced to one device
and float32 parameters:

- **Polar Express** orthogonalization replaces fixed-coefficient Newton-Schulz.
  Same five matmul rounds; measured on a batch of condition-1e3 matrices, the
  mean singular value of the update is `1.00` against Newton-Schulz's `0.86`,
  and the smallest direction is lifted to `0.21` against `0.09`.
  `test_polar_express_beats_newton_schulz_on_small_singular_values` pins both.
- **NorMuon** low-rank second-moment reweighting, renormalized so the update's
  Frobenius norm is unchanged. Row scaling changes, step size does not, so
  `muon_lr` keeps its meaning.
- Cautious decay is fused into the update kernel, and equal-shape matrices are
  stacked: the 30 hidden matrices per network are three batched kernel chains,
  compiled with `dynamic=False, fullgraph=True` on CUDA.
- Deliberately dropped: distributed banks and comms, bfloat16 mantissa
  tracking (these parameters are float32), and every schedule.

Pretraining keeps `muon_lr=0.01`, the rate its artifacts were trained with. PPO
defaults to `1.2e-3` (critic `2.4e-3`), which is RMS-matched to the AdamW step
it replaces: an orthogonalized update to an `RxC` matrix has Frobenius norm
`sqrt(min(R, C))` before the `sqrt(max(1, R/C))` aspect-ratio adjustment, so the
mean coordinate of this model's matrices moves by `0.06-0.09 x muon_lr`, or
about `1e-4` - PPO's AdamW actor rate. Adopting Muon in PPO therefore changes
update geometry, not step size. `--muon-lr` overrides it.

Collect once, then reuse that exact corpus for every optimizer run:

```bash
mlq submit --name screeps-pretrain-corpus --priority 10 \
  --max-parallel-runs 1 --cwd "$PWD" -- \
  python3 -m samples.rl.agent.pretrain_corpus \
    --num-envs 32 --steps 20000 --max-episode 20000 \
    --curriculum empty,seed_creep,seed_full,seed_claimer,seed_outpost \
    --ti-actor-steps 20000 --ti-replay-capacity 8192 \
    --output samples/rl/runs/pretrain-corpora
```

Wait for successful collection and inspect the printed immutable content path.
Then submit training as a second explicit stage; do not guess or automatically
select a directory when multiple corpora exist.

If only the six one-step spawn scenarios change without an observation/action
ABI or long-lifecycle teacher change, derive a new immutable corpus rather than
recollecting 640,000 scripted transitions and both TI streams:

```bash
python3 -m samples.rl.agent.refresh_spawn_contract_corpus \
  --base "$CORPUS" --output samples/rl/runs/pretrain-corpora
```

```bash
CORPUS=samples/rl/runs/pretrain-corpora/<corpus-sha256>

mlq submit --name screeps-joint-pretrain --max-parallel-runs 1 --cwd "$PWD" -- \
  python3 -m samples.rl.agent.pretrain_joint \
    --corpus "$CORPUS" \
    --global-epochs 16 --device cuda --seed 3 \
    --validation-steps 256 --closed-loop-steps 20000 \
    --save samples/rl/runs/joint_pretrain_v4.pt
```

### Pretraining qualification

A release-qualified artifact requires:

1. full teacher factor legality and zero engine-invalid issued intents;
2. finite actor and critic losses with per-factor coverage diagnostics;
3. a separate actor-only rare-intent lane that balances
   `createConstructionSite` and `claimController` actors and averages selected
   semantic factors within each actor. Construction supervises intent and
   structure type, not the teacher's arbitrary legal tile; claim supervises
   intent and controller target. Both train and holdout require NLL ≤1.0 and
   deterministic semantic-factor accuracy ≥0.8 for each intent;
4. exact spawn-body supervision for economy, logistics, work, and claim
   compositions across ≤300, 301–549, 550–649, and ≥650 energy contexts and
   short (≤6), medium (7–15), and long (≥16) bodies;
5. affordable ordered body labels with zero invalid spawn executions;
6. per-scenario held-out spawn semantic NLL ≤1.5 plus a successful
   deterministic engine spawn of the exact intended ordered body;
7. positive aggregate and per-curriculum critic explained variance on the
   never-optimized scripted full-return holdout. TI value rows are a one-time
   representation initialization from a different behavior policy; their EV
   after actor-aligned scripted fitting is diagnostic, not a contradictory
   release gate. The short fresh-reset critic metric is also diagnostic;
8. an `empty` teacher trajectory that independently delivers, constructs,
   reaches the declared population gate, and funds a room claim;
9. deterministic learned-policy reproduction on a third, never-collected seed
   family. At least 50% of empty environments must independently deliver,
   construct, reach the population gate, and claim; aggregate success from one
   favorable world cannot certify the fleet;
10. separate per-curriculum metrics so seeded stages cannot assemble a false
   aggregate pass;
11. complete optimizer, RNG, configuration, semantic-schema, and model-state
   provenance;
12. independent temporal-holdout dynamics MSE materially below the identity
   baseline for both actor and critic, with a materially worse same-state legal
   counterfactual baseline and minimum coverage;
13. an atomic checkpoint marked `partial=false, qualified=true`.

Partial checkpoints are resumable pretraining state but cannot start PPO.
The focused spawn contract exercises six distinct body archetypes, four energy bands, and
three body-length bands through the real engine:

```bash
mlq submit --name screeps-spawn-pretrain-contract --max-parallel-runs 1 --cwd "$PWD" -- \
  python3 -m samples.rl.agent.eval_spawn_contract
```
Pretraining resume reloads the same validated corpus and restores model,
optimizer, global RNG, general shuffle RNG, and isolated rare-intent shuffle
RNG state, plus the dedicated DAgger and NextLat shuffle streams. The isolated
streams ensure auxiliary oversampling cannot perturb lifecycle or temporal
minibatch order. Configuration, temporal objective, row counts, or corpus
identity drift is rejected, while the requested target epoch may be increased
beyond the completed epoch. Fresh
teacher-forced and full closed-loop
qualification still run after optimization; these policy-dependent checks are
never cached in the corpus.

## PPO

The current PPO implementation sums active action-argument log-probabilities
within each creep, clips a likelihood ratio per live creep, averages creeps
within a transition, then averages transitions. All creeps use the shared team
advantage. This keeps team states equally weighted across population sizes, but
it is an engineering choice rather than the only mathematically correct team
objective. Full-team and semantic-group ratios are required ablations in
[`ROADMAP.md`](./ROADMAP.md#joint-policy-ratio-ablation).

Other implemented contracts:

- PPO `gamma=0.995`;
- one CleanRL-style GAE with `lambda=0.95`: the actor uses its advantage and
  the critic target is exactly `advantage + behavior_value`;
- critic pretraining uses finite-horizon discounted reward-to-go at the same
  gamma; supervised actor BC has no temporal return estimator, while both
  trunks receive action-conditioned one-step latent prediction;
- advantages normalized once over the valid rollout;
- type-gated likelihood and entropy;
- time-limit value bootstrap with trajectory-chain cuts, including a
  start-state segment boundary, which truncates without ending the episode;
- diagnostic policy KL without early stopping; every rollout receives all three
  configured PPO epochs;
- action-conditioned NextLat supervision on both independent world-state trunks;
- no unconditional random-policy critic-only warmup.

### Start states

Without a reservoir every environment marches forward from tick zero in
lockstep, so one update's batch is a narrow band of the timeline, a phase is
visited once and never revisited after the policy changes, and an environment
that collapses returns to tick zero. The start-state reservoir in
`samples/rl/agent/state_reservoir.py` changes which states PPO trains on; it
changes neither the objective, the horizon, nor the score.

Run 24 environments as `--start-mix fresh=12,policy=8,teacher=4`:

- 12 untouched full-lifecycle environments run the declared horizon and are
  never restored into. At least one is mandatory; a mixture without one is
  rejected.
- 8 policy-lane environments restart from states the current policy reached.
- at most 4 teacher-bridge environments restart from the immutable set
  collected from the teacher. That count is a ceiling, not a quota: when a
  phase's teacher strata are retired or empty the slot draws from the policy
  lane instead of reverting to tick zero.

Without `--start-mix` the run puts half the fleet in the fresh lane and a third
of the remainder in the teacher lane, which is this split at 24 environments —
but only when `--teacher-start-states` is also passed. Without a teacher
directory the teacher lane is empty and those environments join the policy
lane.

A stratum is `(lane, event, phase, outcome)`. Events are the environment tags
in [`ARCHITECTURE.md`](./ARCHITECTURE.md#environment-state-snapshots) plus a
`periodic` background tag; a tick that fires several is filed under the rarest
and most decision-bearing of them. Phases split the horizon at 2,000, 8,000,
and 15,000 ticks into `early`, `mid`, `late`, and `endgame`. Outcome is
`failure` when an environment is down to one creep or its H+C rate falls below
4.0 per tick, and `success` otherwise, so a collapsing colony stays in the
population instead of being quietly filtered out of training.

A policy-lane stratum holds at most `--per-stratum` snapshots, 24 by default,
and evicts oldest-first within that stratum only, so a full common stratum can
never evict a rare one. Imported teacher strata are exempt: the bridge set is
kept whole. Captures within an environment are at least 64 steps apart, and a
full stratum is refreshed with probability 0.25 rather than every time it is
eligible. Sampling is uniform over non-empty strata, then uniform within the
chosen stratum.

A restored environment runs `--segment-ticks` ticks, 2,048 by default, and then
draws a new start state. Segment truncation reuses the time-limit path above:
value is bootstrapped and the trajectory chain is cut, so a boundary produces
neither a false terminal nor a spliced advantage. Fresh-lane environments are
never truncated by the reservoir.

Only fresh-lane environments contribute to `charts/episodic_return` and
`charts/episodic_length`. A restored world's segment end, and its horizon end,
are reported as `segment_return_mean` and `segment_length_mean`: booking a
fragment of a 20,000-tick lifecycle as a completed episode would make the
headline curve fall exactly when the reservoir starts supplying late states.

A resumed run inherits the persisted reservoir configuration from the index in
`--reservoir`; `--segment-ticks` and `--per-stratum` override it when passed
explicitly. `--start-mix`, the lane assignment, and the reservoir directory
cannot change across a resume, and a checkpoint that carries start-state
bookkeeping cannot be resumed without `--reservoir`, or the population it was
trained on would be silently discarded.

Teacher bridging is temporary and retires per phase. Once the policy lane holds
at least 32 records spanning at least 3 distinct events in a phase, that
phase's teacher strata stop being sampled.

Collect the bridge sets once, before PPO. The scripted planner supplies the
late-economy bridge, because it stays inside the observation ABI for the whole
horizon:

```bash
python3 -m samples.rl.agent.teacher_snapshots \
  --teacher scripted --num-envs 4 --steps 20000 \
  --curriculum empty,seed_outpost,seed_claimer,seed_full \
  --seed 301 --per-stratum 8 --min-gap 64 \
  --output samples/rl/runs/teacher-start-states
```

Measured: 464 records, none dropped by restore verification, 1.92 MB, phases
`early=85 mid=135 late=148 endgame=96`, eight event tags including
`replacement_due=103` and `remote_loaded_home=61`, and zero overflow rejections.

The International is a second, optional set and covers only the opening of the
economy. Collect it with at most two concurrent environments:

```bash
python3 -m samples.rl.agent.teacher_snapshots \
  --teacher ti --num-envs 2 --steps 6000 \
  --curriculum empty,seed_outpost --seed 201 \
  --per-stratum 8 --min-gap 64 \
  --output samples/rl/runs/teacher-start-states
```

Measured: 22 records, phases `early=21 mid=1`, with 10,466 of 12,000 candidate
ticks rejected because the world had already outgrown the observation capacity.
Both limits are recorded in [`ROADMAP.md`](./ROADMAP.md#immediate-learning-blockers).

The collector prints one content-addressed directory holding `manifest.json`
and the snapshots it indexes. Pass each directory with its own
`--teacher-start-states`; the flag is repeatable because different teachers
cover different phases. Records are imported by reference and are never evicted
or deleted by a training run.

Report per update: snapshot capture seconds, snapshots captured, total snapshot
bytes, reservoir occupancy per event and per phase, lane composition,
cumulative start-state draws by lane and phase, segment boundaries, and the
fraction of training transitions in each economy phase
(`start_capture_seconds`, `snapshots_captured`, `reservoir_bytes`,
`reservoir_event_*`, `reservoir_phase_*`, `reservoir_<lane>_records`,
`start_mix_*`, `start_origin_*`, `reservoir_sampled_*`, `segment_boundaries`,
`train_phase_fraction_*`, and the vector environment's `restart_counts` split
of resets against restores). A run whose occupancy or transition fractions stay
concentrated in `early` has not fixed the coverage problem the reservoir exists
to fix, whatever its return does.

Evaluation and qualification draw no start state from the reservoir; both run
fresh, untouched 20,000-tick worlds. A policy that performs only from restored
starts therefore fails qualification.

### Comparing two PPO runs

A reservoir run is only credible against a control that differs in nothing but
its start states. Run the control through the same code path with every
environment in the fresh lane, so capture overhead, metrics, and the trajectory
contract are identical and only restores are absent:

```bash
# A: reservoir
--reservoir samples/rl/runs/reservoirs/A --start-mix fresh=12,policy=8,teacher=4 \
  --teacher-start-states <scripted-set> --teacher-start-states <ti-set>
# B: control, same seed, updates, minibatch, and initialization
--reservoir samples/rl/runs/reservoirs/B --start-mix fresh=24,policy=0,teacher=0
```

Score both on fresh worlds with a seed family that neither training nor teacher
collection used:

```bash
python3 -m samples.rl.agent.eval_closed_loop \
  --checkpoint samples/rl/runs/policy_reservoirA.pt \
  --ticks 20000 --num-envs 10 --seed 900 \
  --output samples/rl/runs/eval_A.json
```

The decisive columns are per curriculum and late-window: `late_remote_harvest`,
`late_remote_home_delivery`, `late_remote_staffed_ticks`, `claims`, and
`control_rate`. Training return is not the metric, and neither is performance
from restored starts.

Start release PPO only from a release-qualified artifact. For an explicitly
experimental continuation, `--allow-unqualified-joint` permits a complete,
ABI-compatible joint checkpoint to initialize PPO without rewriting its
qualification result. PPO checkpoints retain the original failed gates and
both source fingerprints. This override is not a promotion.

```bash
mlq submit --name screeps-ppo --max-parallel-runs 1 --cwd "$PWD" -- \
  uv run --project samples/rl python -m samples.rl.agent.train \
    --num-envs 12 --steps 512 --max-rollout-steps 512 --minibatch 1536 \
    --max-episode 20000 --device cuda --seed 3 --compile \
    --curriculum empty,seed_creep,seed_full,seed_claimer,seed_outpost \
    --start-mix fresh=6,policy=4,teacher=2 \
    --reservoir samples/rl/runs/reservoirs/<run> \
    --teacher-start-states samples/rl/runs/teacher-start-states/<sha256> \
    --segment-ticks 2048 \
    --resume samples/rl/runs/joint_pretrain_v4.pt \
    --allow-unqualified-joint \
    --save samples/rl/runs/policy_v4.pt
```

Twelve environments and `--compile` are both measured choices, recorded in
[`PERFORMANCE.md`](./PERFORMANCE.md#optimizer-batch-and-compilation): the box
saturates below twenty-four simulators, and CUDA-graphing the per-tick forward
runs collection 1.7x faster while the minibatch path stays eager because its
capture pool does not fit. Every run prints its compiled-graph inventory and
late shape mints; a run reporting a late mint has a shape the warmup missed, and
its update timings are not comparable. A short wall-clock run is still not
evidence of convergence.

A true PPO continuation restores actor, critic, both optimizers, aggregate
reward normalization, counters, and CPU/CUDA/NumPy RNG state. Environments and
their reset-local discounted-return traces restart, so continuation is exact
for weights and optimizer state, not for live trajectories.

## Evaluation

Every candidate checkpoint needs deterministic and sampled evaluation across:

- every curriculum stage;
- held-out room layouts and topology seeds;
- several action RNG seeds;
- population buckets such as 1, 4, 12, 24, 48, and 64;
- ordinary visible-only expansion rather than only a pre-exposed neighbor.

At minimum report raw H+C, training reward, delivery, construction, controller
progress, claims, live population, role mix, dropped energy, invalid actions,
target contention, task churn, overflow, and critic explained variance by
stage. Checkpoint selection must use a declared stage-balanced scorecard rather
than the last completed environment or one favorable sampled seed.

## Immediate stop conditions

- Any teacher action is masked, non-finite, or rejected by the engine.
- A partial, incompatible, or unqualified artifact enters a release run.
- Actor or critic loss becomes non-finite, or KL/value quality collapses in an
  actor-count or curriculum bucket.
- Training reward rises while the broader economic scorecard regresses.
- A reversible resource loop earns reward.
- Required actors, rooms, or candidates overflow the observation.
- More rollout is proposed as a substitute for fixing action semantics,
  scenario diversity, persistent tasks, or long-horizon credit.
