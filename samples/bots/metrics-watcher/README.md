# Metrics watcher (TensorBoard)

The Apex bots write empire metrics to **RawMemory segment 87** (see
[`../PROTOCOL.md`](../PROTOCOL.md)). This package is a **host-side** tool that
turns those samples into TensorBoard event files.

The bot never imports TensorBoard / torch.

## Quick path (bench)

```bash
# 1) Run a sim (Node 24+)
mise exec node@24 -- node --import xxscreeps/loader \
  samples/bots/apex/bench.mjs 5000

# 2) Convert JSONL → TensorBoard
python3 samples/bots/metrics-watcher/watch.py \
  --jsonl samples/bots/apex/runs/latest/metrics.jsonl \
  --logdir samples/bots/apex/runs/latest/tb

# 3) View
tensorboard --logdir samples/bots/apex/runs/latest/tb
```

Follow a live dump:

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
| `energy/*_total` / `*_rate` | Harvest / build / upgrade / spawn body cost |
| `controller/*` | Control points, progress |
| `gcl/level` | GCL |
| `cpu/used`, `cpu/bucket` | End-of-tick CPU |

Step = **game tick**. Bot writes segment every 5 ticks; counters still accumulate every tick.

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
