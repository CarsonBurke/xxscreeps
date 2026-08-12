# Training playbook

The workspace contains experimental schema-v2 joint-pretrain and PPO artifacts.
The joint artifact's `meta.qualified=true` reflects deliberately relaxed
short-run thresholds; neither artifact meets the release gates in this
playbook. Old schema-v1 checkpoints are incompatible and must not be resumed.

All training and evaluation workloads go through `mlq`; do not bypass the shared
CPU/GPU queue. A major pretrain or PPO run also requires explicit user approval.

## Readiness gates

- 31/31 synthetic learning, schema, artifact, action, and architecture contracts.
- Scripted 6,000-tick economy gate passes with zero rejected teacher intents.
- Seeded cross-room reserve→claim gate verifies own-reservation allegiance, positive
  `claimDelta`/reward, and a second scripted-teacher claim pass.
- Delivery anti-recycling gate proves a rewarded sink cannot be withdrawn and
  re-deposited for return.
- Action, observation, reward, teacher, model, and learning ABIs in `schema.json`
  match the artifact metadata exactly.

Historical wave documents and `JOB0-REPORT.md` describe earlier schemas. Their
scores are useful context, not current acceptance evidence.

## Gate 0 — engine contracts

```bash
export RL_NODE="$(mise exec node@24 -- which node)"

python3 -m samples.rl.agent.eval_scripted \
  --ticks 6000 --max-episode 6000
python3 -m samples.rl.agent.eval_expansion --ticks 500
python3 -m samples.rl.agent.eval_reward_contract
```

Re-run all three after changes to the encoder, action executor, reward, simulator,
or teacher. The scripted gate must report zero invalid/rejected labels and actual
post-tick energy events for container/storage deposits, withdrawals, and tower
fueling; successful navigation toward those targets is not evidence of execution.
Aggregate reward alone is insufficient.

## Job 1 — same-stream joint pretraining

One environment fleet supplies both behavior-cloning labels and critic returns:

```bash
python3 -m samples.rl.agent.pretrain_joint \
  --num-envs 4 --steps 6000 --chunk 512 --device cpu \
  --max-episode 6000 \
  --curriculum empty,seed_creep,seed_full,seed_claimer \
  --validation-steps 256 --closed-loop-steps 6000 \
  --save samples/rl/runs/joint_pretrain_v2.pt
```

Promotion requirements are executable, not documentary:

- every eligible type/direction/target/amount teacher factor is legal;
- actor and critic losses are finite;
- the `empty` teacher worker independently demonstrates productive delivery,
  construction, at least 24 live creeps, and economy-funded expansion; aggregate
  success assembled from seeded stages cannot qualify an artifact;
- a fresh teacher-forced validation stream is legal and its critic EV exceeds
  `--min-validation-ev`;
- the deterministic learned actor repeats those empty-stage outcomes, meets
  `--min-closed-loop-rate`, and has at most one percent invalid results;
- the saved metadata has `kind=joint_pretrain`, `partial=false`, and
  `qualified=true` with exact semantic-schema and state signatures.

Intermediate snapshots contain complete continuation state but remain
`partial=true, qualified=false`. Resume deterministically replays the scripted
prefix to restore environment phase before restoring RNG state. Changing the
continuation configuration is rejected.

## Job 2 — PPO

Only a qualified joint-pretrain artifact may start a new PPO run:

```bash
mlq submit --name screeps-ppo --max-parallel-runs 1 \
  --cwd /home/marvin/Documents/repositories/xxscreeps \
  --env RL_NODE="$(mise exec node@24 -- which node)" \
  -- python3 -m samples.rl.agent.train \
      --num-envs 16 --steps 512 --updates 5000 --device cuda \
      --curriculum empty,seed_creep,seed_full,seed_claimer \
      --resume samples/rl/runs/joint_pretrain_v2.pt \
      --save samples/rl/runs/policy_v2.pt
```

Joint-pretrain import loads the actor exactly and only the critic trunk; it does
not import behavior-cloning optimizer moments into PPO. A true PPO continuation
loads both models, both optimizers, counters, and reward-normalizer moments
strictly. Reset-local discounted-return traces are cleared because environments
restart. Reward-normalization mode drift is rejected.

## Stop conditions

- Any masked/non-finite teacher factor: fix the teacher/action contract.
- Failed engine gate: fix the environment before collecting data.
- Closed-loop imitation below the declared threshold: do not promote the artifact.
- PPO KL or value loss diverges by actor-count/economy-stage bucket: stop and
  diagnose rather than spending more rollout.
- Actor, target, or room overflow in a required evaluation bucket: redesign the
  candidate/packing contract rather than silently increasing a dense tensor.

## Current scope boundary

The current simulator exercises one seed room plus one privileged-information
neighbor at GCL2. Before claiming multi-room production competence, add seeded
scenario families, visible-only scouting/intel, staged GCL3/GCL4, and 32–48 live
actor gates. A slower strategic option layer remains the planned architecture for
stable room priorities, spawn/build queues, task persistence, and longer credit.
