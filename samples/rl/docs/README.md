# RL documentation index

The schema-v2 source of truth is intentionally small:

- [`17-SCALABLE-ENTITY-MAPPO.md`](./17-SCALABLE-ENTITY-MAPPO.md) — architecture,
  demonstrated fixes, scope boundary, and strategic-layer roadmap.
- [`07-TRAINING-PLAYBOOK.md`](./07-TRAINING-PLAYBOOK.md) — executable gates,
  pretraining, PPO continuation, and stop conditions.
- [`11-REWARD-AND-PRETRAIN.md`](./11-REWARD-AND-PRETRAIN.md) — objective,
  same-stream pretraining, and qualification contract.
- [`05-KILL-CONDITIONS.md`](./05-KILL-CONDITIONS.md) — reasons a run must stop.
- [`13-PERFORMANCE.md`](./13-PERFORMANCE.md) — current transport, storage, compile,
  and measurement contracts.

Documents numbered 00–04, 06, 08–10, 12, and 14–16, plus `JOB0-REPORT.md`
and `COMMIT-PREP.md`, are historical design/review records from earlier action,
reward, and artifact ABIs. They are retained for provenance only. Commands,
scores, “current” labels, and standing orders inside them are superseded by the
five documents above and must not be used to launch schema-v2 work.
