# Reward and same-stream joint pretraining (authoritative)

## Objective

Training uses separate conservation-aware progress channels:

```text
r_train = 0.10 * harvested_energy
        + 0.25 * productive_delivery
        + 0.50 * construction_progress
        + 1.00 * controller_progress
        + 500  * newly_claimed_room

score_eval = harvested_energy + controller_progress
```

`productive_delivery` counts delivery into owned spawn, extension, and tower
sinks. Those rewarded sinks are not valid withdraw sources; withdraw is limited
to storage, containers, and links. This prevents transfer/withdraw recycling from
minting return. There is no direct spawn reward, so body churn is not rewarded.
Construction and claim channels pay for otherwise very delayed investments. The
raw harvest-plus-control score remains the stable external comparison metric.

The objective is still incomplete for a mature empire. Training and qualification
now report each curriculum separately; before a major run, add held-out scenario
families, waste accounting, and preferably vector value heads with PopArt-style
scale adaptation. Reward shaping does not replace held-out economic scorecards.

## Same-stream joint pretraining

Behavior cloning and critic fitting use exactly the same scripted transitions:

```text
scripted state/action/reward stream
        |                    |
        v                    v
per-factor masked CE     raw Monte-Carlo value target
        |                    |
        +---------+----------+
                  v
        complete joint_pretrain artifact
```

The actor loss averages active conditional factors: type plus whichever of
direction, target, and amount the selected intent requires. Illegal or non-finite
eligible labels fail the strict teacher contract. They are not silently dropped
at transition or actor granularity. Per-factor NLL/count diagnostics are saved.

The critic fits returns from the same occupancy distribution. PPO later imports
the critic trunk only because PPO normalizes the reward/value scale; raw-value
head weights and optimizer moments are intentionally not transferred.

## Artifact qualification

A training-set loss is not sufficient. Final qualification also requires:

1. a fresh reset teacher-forced stream with full factor legality;
2. positive critic explained variance on a fresh reset validation stream;
3. deterministic empty-curriculum evaluation in which the learned policy delivers,
   builds, reaches at least 24 live creeps, and funds its own room claim;
4. the same end-to-end empty-curriculum coverage from the teacher corpus; seeded
   curricula are tracked separately and cannot supply missing qualification outcomes;
5. finite actor/critic losses and a nonzero scripted skill stream;
6. complete optimizer, RNG, configuration, semantic-schema, and model-state ABI
   metadata;
7. an atomic save marked `partial=false, qualified=true`.

`train.py` refuses partial or unqualified pretraining artifacts. `watch.py` and
PPO continuations reject architecture drift instead of using `strict=False`.

## Curriculum limits

The scripted curriculum covers empty, seeded-creep, seeded-economy, and seeded
claimer states. It is a bootstrap teacher, not proof of generalization. The
current map family has one seed room and one pre-exposed neighbor, so a qualified
artifact would still need held-out topology/seed/population evaluation, ordinary
visible-only scouting, DAgger on learner-visited states, and multi-room strategic
task persistence before PPO is trusted at dozens-of-creeps scale.

The International remains evaluation-only because it does not provide labels in
this factorized macro-action contract. The obsolete split BC, critic-pretraining,
and merge entry points have been removed.
