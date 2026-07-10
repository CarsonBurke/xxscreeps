# Apex metrics segment protocol

Bots write a small JSON sample to **RawMemory segment 87**. A host watcher
turns it into TensorBoard. TensorBoard is **not** a bot dependency.

## Segment

| | |
|--|--|
| ID | `87` |
| Write cadence | every 5 ticks (`WRITE_EVERY`) |
| Shape | single latest sample (no history ring) |

## Sample fields (lean)

| Field | Meaning |
|--------|---------|
| `t` | game tick |
| `rclMax` | highest owned RCL |
| `colonies` | owned rooms |
| `creeps` | live creep count |
| `controlPoints` | lifetime upgrade progress applied |
| `harvested` / `build` / `upgrade` | lifetime energy |
| `creepEnergy` | lifetime body cost of spawned creeps |
| `storedEnergy` | sum of storage energy |
| `sites` | construction sites visible |
| `progress` | sum of controller progress bars |
| `gcl` | GCL level |
| `cpu` | `Game.cpu.getUsed()` at sample time |
| `bucket` | CPU bucket |

## TensorBoard tags

| Tag | Source |
|-----|--------|
| `empire/rcl_max` | `rclMax` |
| `empire/colonies` | `colonies` |
| `empire/creeps` | `creeps` |
| `empire/sites` | `sites` |
| `empire/stored_energy` | `storedEnergy` |
| `energy/*_total` / `*_rate` | harvested, build, upgrade, creepEnergy |
| `controller/control_points_*` | controlPoints |
| `controller/progress` | progress |
| `gcl/level` | gcl |
| `cpu/used` | cpu |
| `cpu/bucket` | bucket |

Step = game tick. Rates are Δ / Δt in the watcher.

## Payload on the wire

```json
{ "v": 1, "seq": 12, "written": 12345, "sample": { "t": 12345, "rclMax": 2, "creeps": 6, "cpu": 1.2, "…": "…" } }
```

Legacy payloads with a `ring` array are still accepted (watcher uses the last entry).

## Flow

```
bot metrics.tick() → segment 87
       ↓
bench / poller → metrics.jsonl
       ↓
watch.py → TensorBoard event files
```
