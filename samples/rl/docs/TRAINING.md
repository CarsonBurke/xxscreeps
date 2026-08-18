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
- the fixed-seed 20,000-tick teacher economy expands from empty, and its
  liveness record shows the expert acting on at least 50% of graded ticks with
  no silence longer than 30 consecutive ticks;
- cross-room navigation and reserve/claim execute in the real engine and emit the claim diagnostic;
- rewarded delivery cannot be recycled through a legal withdrawal path;
- the current semantic schema and all ABI identifiers match the artifact exactly;
- required population, room, and candidate buckets have no silent overflow;
- evaluation records curriculum, scenario seed, action seed, decoding mode, and raw metrics.

Run the engine contracts after changing the encoder, executor, reward,
simulator, spawn-body contract, or teacher:

```bash
export RL_NODE="$(mise exec node@24 -- which node)"
export CUDA_VISIBLE_DEVICES=

python3 -m samples.rl.agent.test_latent_unit
python3 -m samples.rl.agent.eval_scripted \
  --ticks 20000 --max-episode 20000 --seed 3
python3 -m samples.rl.agent.eval_expansion --ticks 500
python3 -m samples.rl.agent.eval_reward_contract
```

These contracts are CPU-only by construction, and `CUDA_VISIBLE_DEVICES=` is
part of the command rather than an optimization: a torch process that
initializes CUDA reserves an enormous virtual address space and tens of
gigabytes of page tables on this host, so running several of them concurrently
has OOM-killed the machine. Run one contract process at a time, in the
foreground; anything that genuinely needs a GPU goes through `mlq`.

Cap that with a *soft* limit only (`ulimit -Sv 12000000`, not `ulimit -v`). The
limit exists to bound torch's host reservations; the simulator has the opposite
profile. V8 reserves heap cages, every WebAssembly instance reserves gigabytes
of address space it never makes resident, and the in-engine teacher loads WASM,
so a hard cap kills the environment server with
`WebAssembly.Instance(): Out of memory` while resident use stays small. Spawned
environments raise their own soft limit back to the hard limit for that reason.

Aggregate reward is insufficient. The teacher gate must establish a live expert,
labels validated against the tick that produced them, and actual post-tick
resource effects, not merely successful navigation toward a target.
`eval_scripted` is the hand-written baseline the learned policy is compared
against; it is not a teacher and its labels never enter training.

## The teacher is configured for the observation ABI

The International plays a larger game than the observation can hold. With stock
radii - `maxRemoteRoomDistance = 5` and unbounded scouting - it runs a 7-to-8
room, 111-to-122 creep colony on this map: measured over three 20,000-tick
lifecycles a room the bot held stake in was dropped on 74-86% of ticks (first at
tick 571-1,389), 3.5-5.5 creeps per tick sat outside the action space, live
creeps exceeded `maxCreepActors` from tick 2,201-10,577, and targets exceeded
`maxTargets` from tick 5,752-10,401. Those ticks label creeps in rooms the
observation does not contain, so the corpus would supervise actions whose subject
is invisible.

The teacher is therefore built with `maxRemoteRoomDistance = 1` **and**
`maxScoutRoomDistance = 1` (`src/constants/general.ts` in the bot checkout;
`npm run build` reproduces the shipped bundle byte-for-byte, so the change is
auditable). Two 20,000-tick episodes under those radii dropped no owned or mined
room at any tick, cut hidden creeps to 0.4-0.6 per tick, and reduced room-slot
pressure from 83-86% of ticks to 29%, all scouts. The creep cap then binds first,
at tick 2,666-3,638, which is where teacher supervision and bridge snapshots stop
being faithful.

`schema.json` declares both required values under `teacher`, and collection reads
the constants back out of the executed bundle: `CorpusConfig.validate` and the
snapshot collector's config both fail before a run starts when the loaded teacher
disagrees. Corpus meta records `ti_room_radii` beside the bundle hash, so an
artifact states which teacher configuration produced it.

Teacher runs are not reproducible, and no gate may assume they are. A virtual
player clock and canonical room ordering removed the host leaks (see
[`DECISIONS.md`](./DECISIONS.md#a-reproducible-teacher-needs-a-virtual-clock-and-canonical-room-order)),
which made short twins bit-identical, but longer twins still diverge inside the
player realm. Compare teacher-derived numbers across repeated seeds, never as
single-run replays. The learner path is reproducible, including restored
snapshots, and that is what the reservoir depends on.

## Same-stream pretraining

The International supplies every label and every value trajectory. Its raw
engine intents are captured at the runner boundary and translated into macro
action factors; in-range target commands, construction tiles, and spawn
compositions are supervised only where they map exactly. Spawn type and body
counts are exact; ordering is supervised only for bodies already grouped into
contiguous part-type blocks, and interleaved bodies supervise counts but not a
different order. Immediate moves, concurrent commands, and unsupported commands
are rejected, never relabeled as `none`. Supervision is therefore partial by
construction: a per-tick eligibility mask records which factors the teacher
actually decided, and the loss reads no other factor.

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
absolute error beats the constant-return baseline by at least 10%. The declared
20,000-tick horizon still makes endpoint bootstrapping negligible at
`gamma=0.9995` (`0.9995^20000 = 4.5e-5`); a low
training loss or a biased, nearly constant prediction remains insufficient. Do
not pass an unqualified artifact into joint pretraining.

Behavior cloning and critic adaptation consume a reusable immutable corpus:

```text
teacher state/action/eligibility        independent TI reward streams
        |                    |                    |
        v                    v                    v
per-factor masked CE     discounted reward-to-go (zero endpoint)
        |                    |                    |
        +--------------------+--------------------+
                             v
                   complete joint-pretrain artifact
```

The standalone collector first completes the whole teacher lifecycle.
A bounded
per-stage/state/action reservoir preserves examples across early, middle, and
late economy phases; every optimization epoch globally shuffles the corpus for
actor BC and final critic fitting. Returns are full finite-horizon reward-to-go,
not bootstrapped from the critic being trained. A separate full-lifecycle fleet
under independent seeds is never optimized and supplies the promotion EV. TI
critic rows have their own independent full-return train and holdout splits and
are used once to initialize value representation before actor-aligned factor
fitting. Repeating their behavior-policy-dependent returns would create a
contradictory value target. Ticks on which the teacher issued nothing carry an
all-false eligibility mask and contribute only a critic target and a temporal
row.
Corpus storage schema v3 also retains a separately bounded temporal replay
directly from
same-tick `(observation, complete joint action, next observation)` triples for
both the teacher train fleet and an independently seeded holdout fleet. It never
joins
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
recollecting every teacher lifecycle and both TI streams:

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

1. full teacher factor legality: every retained label validated against the
   candidate masks of the tick that produced it. There is no engine-invalid
   teacher-intent gate, because `step_expert` returns no intent summary — the
   bot issues its intents inside the engine and the wire never sees an
   issued/invalid count. Teacher wakefulness is gated at collection by the
   liveness record instead;
2. finite actor and critic losses with per-factor coverage diagnostics;
3. a separate actor-only rare-intent lane that balances
   `createConstructionSite` and `claimController` actors and averages selected
   semantic factors within each actor. Construction supervises intent, structure
   type, and the exact tile the teacher built on — a demonstration is a structure
   at a position, and construction is rare enough (a few hundred rows against
   tens of thousands) that the corpus mean cannot carry placement; claim
   supervises intent and controller target. Both train and holdout require NLL
   ≤1.0 and deterministic semantic-factor accuracy ≥0.8 for each intent;
4. exact spawn-body supervision across the ≤300, 301–549, 550–649, and ≥650
   energy-budget buckets, and across the body-length buckets the teacher can
   actually reach. Measured over the six contract worlds: budgets 1/1/1/3, and
   lengths ≤6 = 3, 7–15 = 3, ≥16 = 0. The ≥16 bucket is recorded as a named
   unreached gap (`teacher_spawn_length_unreached_ge16`,
   `_spawn_replay_length_unreached_ge16`) rather than required: the engine
   charges at least 50 energy per part, so a 16-part body needs 800+ energy in
   one spawn, which these RCL1–RCL2 curricula never fund. A seeded higher-RCL
   curriculum promotes it back to fatal by moving one tuple entry. Any reachable
   bucket that comes back empty still fails;
5. affordable ordered body labels with zero invalid spawn executions;
6. per-scenario held-out spawn semantic NLL ≤1.5 plus a successful
   deterministic engine spawn of the exact intended ordered body;
7. positive aggregate and per-curriculum critic explained variance on the
   never-optimized full-return teacher holdout. Value rows and action labels now
   come from the same behavior policy, so this holdout is a release gate rather
   than a cross-policy diagnostic. The short fresh-reset critic metric remains
   diagnostic;
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
The focused spawn contract measures what the teacher decides rather than
comparing it against a frozen archetype: in each of the six worlds it snapshots
the pre-decision state, takes the teacher's body from the tick it labelled, then
restores that exact world into a learner session and requires the engine to
spawn the measured body. At the labelled tick every world is one idle spawn
holding the energy its name advertises — TI removes the seeded context creeps
and the builder world's construction site within two ticks and decides at tick
seven — so the gate asserts spawn legality and funded budget, and reports the
context counts as telemetry:

```bash
mlq submit --name screeps-spawn-pretrain-contract --max-parallel-runs 1 --cwd "$PWD" -- \
  python3 -m samples.rl.agent.eval_spawn_contract
```
Pretraining resume reloads the same validated corpus and restores model,
optimizer, global RNG, general shuffle RNG, and the isolated rare-intent, spawn
and NextLat shuffle streams. The isolated streams ensure auxiliary oversampling
cannot perturb lifecycle or temporal minibatch order. Configuration, temporal
objective, row counts, or corpus identity drift is rejected, while the requested
target epoch may be increased beyond the completed epoch. Fresh
teacher-forced and full closed-loop
qualification still run after optimization; these policy-dependent checks are
never cached in the corpus.

## PPO

The current PPO implementation sums active action-argument log-probabilities
within each creep and clips a likelihood ratio per live creep. The surrogate
is then reduced token-level in the DAPO/VAPO sense (eq. 7): every live actor
decision in the minibatch is summed and divided by the total number of live
actors, so a 40-creep tick carries ten times the weight of a 4-creep tick. It
previously averaged creeps within a transition and then averaged transitions,
which weighted both ticks equally and left the states carrying most of the
colony's decisions least able to correct them. All creeps still use the shared
team advantage, which remains an engineering choice rather than the only
mathematically correct team objective; full-team and semantic-group ratios are
required ablations in
[`ROADMAP.md`](./ROADMAP.md#joint-policy-ratio-ablation). The ratio
diagnostics — approximate KL and clip fraction — stay per team state so their
scale remains comparable across runs.

Other implemented contracts:

- PPO `gamma=0.9995`;
- decoupled GAE (VAPO §4.1): the critic target is the `lambda=1` return, which
  under the truncation bootstrap above is the discounted Monte-Carlo return over
  the segment, while the actor uses advantages at a length-adaptive
  `lambda_policy = 1 - 1/(alpha*l)` (VAPO eq. 5) with
  `gaeLambdaPolicyAlpha=0.5` and `l` the mean transitions per uncut segment
  that environment collected. At a full 2048-transition segment - the largest
  horizon host RAM allows at four live rooms - this is
  `lambda_policy=0.99902`, a credit window of `1/(1-gamma*lambda)=677` ticks
  against 18 under the previous single `lambda=0.95`; the critic target's window
  is `1/(1-gamma)=2000`;
- a positive-example NLL term (VAPO eq. 9-10) at `selfImitationCoef=0.1` over
  the transitions of the environments whose per-segment return sits at or above
  the `selfImitationQuantile=0.8` quantile of that rollout's segment returns.
  The threshold is relative and recomputed per rollout, because a dense reward
  has no verifier to define a correct sample;
- `groupStartsPerState=2` environments begin each rollout from the same
  reservoir start state (VAPO §4.3 group sampling), grouped within a lane so a
  fresh-lane environment still runs an untouched full lifecycle;
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
of the remainder in the teacher lane, which reproduces this split at 24
environments - but the teacher lane is populated only when
`--teacher-start-states` is also passed. Without a teacher directory those
environments join the policy lane.

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

Collect the bridge sets once, before PPO. A start state is a world, not a label:
the policy chooses every action from it, so a bridge set may be generated by any
driver that reaches interesting states. The hand-written planner supplies the
late-economy bridge, because it is the only driver that stays inside the
observation ABI for the whole horizon:

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

The International, the label teacher, is a second and optional set. It plays
well enough to outgrow the observation capacity, so it covers only the opening
of the economy. Collect it with at most two concurrent environments:

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
`--teacher-start-states`; the flag is repeatable because different drivers
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
    --num-envs 12 --steps 2048 --max-rollout-steps 2048 --minibatch 1024 \
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

Every number in that command is measured, and two of them are hard ceilings.

`--minibatch 1024` is the largest update that fits. Measured at observation-ABI
capacity (4 live rooms, 64 live actors, 128 candidates) on a 31.4 GiB RTX 5090,
a complete update peaks at 2.2 GiB at minibatch 128 and scales linearly to
16.5 GiB allocated / 17.7 GiB reserved at 1024; 1536 and 2048 both OOM. The
earlier documented `--minibatch 1536` was never run against full actor rows and
is not viable.

`--steps 2048` is bounded by host RAM, not VRAM. `HostRolloutBuffer` needs
15.8 GiB for 24 environments x 2048 steps once environments hold 4 live rooms
(6.2 GiB at one room), and 31.5 GiB at 4096 steps - more than this box has free.
A 4096-step horizon therefore requires either 16 environments or worlds that
stay under two live rooms, which restored mature starts do not. The trainer
prints the projection before allocating; a run that has to shrink mid-flight has
already failed.

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
