# Metrics watcher (TensorBoard)

The Apex bots write empire metrics to **RawMemory segment 87** (see
[`../PROTOCOL.md`](../PROTOCOL.md)). This package is a **host-side** tool that
turns those samples into TensorBoard event files.

The bot never imports TensorBoard / torch.

## Quick path (bench)

Bench **generates TensorBoard on the fly** (default on):

```bash
# Run sim — metrics.jsonl + tb/ event files (watch.py --follow)
mise exec node@24 -- node --import xxscreeps/loader \
  samples/bots/apex/bench.mjs 5000 samples/bots/apex-v3/dist

tensorboard --logdir samples/bots/apex/runs/latest/tb
```

Disable TB:

```bash
mise exec node@24 -- node --import xxscreeps/loader \
  samples/bots/apex/bench.mjs 5000 samples/bots/apex-v3/dist --no-tb
# or: APEX_BENCH_TB=0 …
```

Manual convert / follow (still supported):

```bash
python3 samples/bots/metrics-watcher/watch.py \
  --jsonl samples/bots/apex/runs/latest/metrics.jsonl \
  --logdir samples/bots/apex/runs/latest/tb \
  --follow
```

## Scalars recorded (lean)

| Tag | Meaning |
|-----|---------|
| `empire/rcl_max` | Highest owned RCL |
| `empire/colonies` | Owned rooms |
| `empire/creeps` | Live creep count |
| `empire/sites` | Construction backlog |
| `empire/stored_energy` | Sum of storage energy |
| `energy/*_total` | Lifetime harvest / build / upgrade / spawn cost |
| **`energy/harvested_per_tick`** | **Harvest e/t** (primary income meter) |
| `energy/build_per_tick` | Build spend e/t |
| `energy/upgrade_per_tick` | Upgrade spend e/t |
| **`controller/control_points_per_tick`** | **Control points / tick** |
| `controller/*_total`, `progress` | Lifetime CP, controller progress |
| `gcl/level` | GCL |
| `cpu/used`, `cpu/bucket` | End-of-tick CPU |

Step = **game tick**. Bot writes segment every 5 ticks with windowed rates (`hr`/`cr`/`ur`/`br`); watcher falls back to Δtotals if rates missing.

Target income with full local + remotes: **≥ 40 e/t** (`harvested_per_tick`).

## Dependencies

Prefers `torch.utils.tensorboard.SummaryWriter`. Falls back to CSV
(`scalars.csv`) if torch/tensorboardX are missing.

```bash
# optional; torch is already enough on this machine
pip install torch tensorboard
```

## Live server (outline)

1. Bot keeps writing segment 87 (and optionally `setPublicSegments([87])`).
2. Ops process polls `user/<id>/segment87` from the shard keyval (same as bench)
   or tails a JSONL exporter.
3. Feed lines into `watch.py --jsonl … --follow`.
