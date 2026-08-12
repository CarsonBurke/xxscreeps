# Sample bots

Player scripts you can load into an xxscreeps world. Directories are **flat module maps** (`main.js` + `require` siblings) — point `manage bot` at the folder itself, not a monorepo root.

## Apex (full empire)

| Version | Path | Notes |
|---------|------|--------|
| v1 | [`apex/`](./apex/) | Original multi-room sample |
| v2 | [`apex-v2/`](./apex-v2/) | Traffic / FSM remotes / stats |
| **v3** | [`apex-v3/`](./apex-v3/) | **TypeScript** · enum memory keys · coords · delegated roles · utility planner · war/economy |

```bash
cd samples/bots/apex-v3 && npx tsc -p tsconfig.json
npx xxscreeps manage bot add apex-v3 samples/bots/apex-v3/dist --spawn W5N5
```

See [apex-v3/README.md](./apex-v3/README.md). Memory keys follow International-style short/enum packing (char-count cost).

## Metrics → TensorBoard

Bots write empire metrics (**RCL**, control points, energy totals, CPU, …) to
**RawMemory segment 87**. A host-side watcher records TensorBoard runs — TensorBoard
is **not** a bot dependency.

| Piece | Path |
|--------|------|
| Protocol | [PROTOCOL.md](./PROTOCOL.md) |
| Bot writer | `apex/metrics.js`, `apex-v2/metrics.js` |
| Watcher | [metrics-watcher/](./metrics-watcher/) |
| Bench (dumps JSONL) | `apex/bench.mjs` |

```bash
mise exec node@24 -- node --import xxscreeps/loader samples/bots/apex/bench.mjs 5000
python3 samples/bots/metrics-watcher/watch.py \
  --jsonl samples/bots/apex/runs/latest/metrics.jsonl \
  --logdir samples/bots/apex/runs/latest/tb
tensorboard --logdir samples/bots/apex/runs/latest/tb
```

## Notes

- Use **modern APIs**: `spawnCreep`, `store`, `Game.map.getRoomTerrain` (Apex already does).
- Open-source bots from npm also work, e.g. `screeps-bot-tooangel` (see root README).
- `import --overwrite-code <dir>` replaces **all** imported launcher bot code with modules from `<dir>`.
