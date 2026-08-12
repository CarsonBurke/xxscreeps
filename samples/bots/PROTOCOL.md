# Apex metrics segment protocol

Bots write a **dense** JSON sample to **RawMemory segment 87**.
Screeps bills by character count — keys are short enums (International-style).

A host watcher expands keys for TensorBoard. TensorBoard is **not** a bot dependency.

## Segment

| | |
|--|--|
| ID | `87` |
| Write cadence | every 5 ticks |
| Shape | single sample under key `d` |

## Envelope

| Key | Meaning |
|-----|---------|
| `v` | protocol version (1) |
| `s` | seq (monotonic) |
| `w` | Game.time written |
| `d` | sample object |

## Sample fields (`d`)

| Key | Long name | Meaning |
|-----|-----------|---------|
| `t` | tick | game time |
| `r` | rclMax | highest owned RCL |
| `n` | colonies | owned rooms |
| `c` | creeps | live creep count |
| `p` | controlPoints | lifetime upgrade progress |
| `h` | harvested | lifetime harvest energy |
| `b` | build | lifetime build energy |
| `u` | upgrade | lifetime upgrade energy spend |
| `e` | creepEnergy | lifetime body cost of spawns |
| `se` | storedEnergy | sum storage energy |
| `si` | sites | construction sites |
| `pr` | progress | sum controller progress |
| `g` | gcl | GCL level |
| `k` | cpu | Game.cpu.getUsed() |
| `q` | bucket | CPU bucket |
| `hr` | harvestRate | **energy harvested / tick** (window ≈ write cadence) |
| `cr` | controlRate | **control points / tick** |
| `ur` | upgradeRate | upgrade energy spent / tick |
| `br` | buildRate | build energy spent / tick |
| `mh` | maxHarvestEt | **spawn-time-bound max harvest e/t** (greedy packages) |
| `mp` | maxHarvestPhysics | sum of candidate source e/t (no spawn limit) |

Rates are over the last write window (default 5 ticks), not lifetime averages.
`mh` is the real bar: max sustainable e/t given path lengths + hauler/miner duty ≤ #spawns.

Example:

```json
{"v":1,"s":12,"w":12345,"d":{"t":12345,"r":2,"n":1,"c":6,"p":227,"h":35856,"b":21969,"u":227,"e":20000,"se":0,"si":3,"pr":150,"g":1,"k":1.2,"q":10000,"hr":18.4,"cr":12.0,"ur":12.0,"br":1.2}}
```

## Positions in Memory

Store **coords** `{x,y}` (or packed `x+y*50`), never `RoomPosition` objects.
See `codec.ts` (`asCoord` / `packCoord` / `packId`).

## Creep / room keys

Numeric enums (`CreepMem`, `RoomApexMem`) — see `memoryKeys.ts`.

## Flow

```
bot metrics.tick() → segment 87 (dense)
       ↓
bench / poller → metrics.jsonl
       ↓
watch.py (expands keys) → TensorBoard
```
