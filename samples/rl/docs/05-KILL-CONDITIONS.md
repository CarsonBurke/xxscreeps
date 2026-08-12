# Kill conditions and training gates

The executable commands and artifact rules live in
[`07-TRAINING-PLAYBOOK.md`](./07-TRAINING-PLAYBOOK.md). This file states the
conditions that stop a run.

## Never train seriously until

| Condition | Required evidence |
|-----------|-------------------|
| Observation/action alignment | Post-tick observation and one executable goal per actor |
| Label legality | Every active teacher factor is finite and legal; zero silent row drops |
| Economy contract | Specialist roles harvest/haul/build/refill without rejected intents |
| Reward integrity | Productive-delivery recycling test passes; raw H+C remains reported |
| Expansion contract | Cross-room navigation and claim reward pass in the real engine |
| Artifact integrity | Exact semantic schema, ABI, complete state, optimizer/RNG provenance |
| PPO scaling | Per-actor ratio/KL/clip, rollout-level advantage normalization, finite update |
| Capacity | Required population/room/candidate buckets report no silent overflow |
| Closed loop | Learned deterministic actor meets the declared held-out gate |

## Immediate stop conditions

- Any teacher action masked by the model or rejected by the executor.
- A partial/unqualified artifact reaching PPO or an inexact checkpoint load.
- Non-finite actor/critic losses, KL runaway, or value collapse concentrated in an
  actor-count or economy-stage bucket.
- Training return rises while raw H+C, productive delivery, construction, or room
  progress does not.
- Reward can be farmed by a reversible resource loop.
- Actor/target/room overflow hides entities required by the evaluation scenario.
- More rollout is proposed as a substitute for fixing action semantics, scenario
  diversity, task persistence, or long-horizon credit.

## HL-Gauss

Keep the scalar MSE critic until scale-aligned held-out explained variance is
measurably inadequate. Enabling HL-Gauss changes the critic ABI and requires a
new artifact; it is not a generic cure for an entity-blind critic or bad reward.
