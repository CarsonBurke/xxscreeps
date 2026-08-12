# Performance program

## Bottleneck ranking (wall-clock training SPS)

| Rank | Bottleneck | Why | Status |
|------|------------|-----|--------|
| 1 | **Node post-tick** | tick + loadRoom + encode + reward | **fused encode+reward** (one load) |
| 2 | **Obs IPC + encode paint** | tile paint + meta JSON | **bin** + **lean meta** + uint8 patches |
| 3 | **Parallel envs** | wall clock / N | scale `--num-envs` headless |
| 4 | **Rollout D2H / AR kernels** | was full-obs D2H + Categorical | dual host obs + fused log_softmax |
| 5 | **PPO stack / mb H2D** | T×N patches | host buffers + stream mb + **static last-mb pad** |
| 6 | **PathFinder** | approachOr / moveTo | one pathfinder contract for train/watch; cached paths |

## Shipped

| Change | Where |
|--------|--------|
| **`RL_OBS_FMT=bin` (default)** | length-prefixed raw frames on stdout (`XRL1` header + meta JSON + tensor blob) |
| **Fused encode+reward** | one `loadRoom`/`runForUser` after tick (`encodeAndRewardAfterTick`) |
| **`RL_LEAN_META=1` (default)** | drop actorMeta/targetMeta/intentResults from wire; server keeps `session.meta` |
| **Single-buffer bin write** | no multi-`Buffer.concat` on hot path |
| **One navigation default** | `pathfinder` for train/watch parity; `RL_NAV=cheap` is an explicit benchmark mode |
| **Static last-mb pad** | fixed B for `torch.compile` reduce-overhead (CUDA) |
| Active-room patch wire | Node sends `roomsUsed`; Python pads to the fixed four-slot model/rollout ABI |
| Host masks as **uint8** | promote to float32 on bulk H2D only |
| **Dual obs** | `VecScreepsEnv.host_obs` — rollout stores host clone, **no per-step D2H** |
| **Batched terminal V** | one `value_only` on stacked terminal_obs vs N×B=1 dual-act |
| **Critic-only bootstrap** | `PPOTrainer.value_only` skips actor trunk |
| **Fused categorical** | `log_softmax` + multinomial / gather (no `torch.distributions`) |
| Reward/done host clone | no alias overwrite on CPU device |
| `metrics_log.py` + train wire | append-only JSONL on train; optional Parquet offline; flush every 5 updates |
| Lean action diagnostics | compact issued/invalid totals and per-intent counts without full metadata |
| TB flush every 5 updates | not every update |
| `empty_cache` every 20 updates | not every update |

## Protocol

```text
RL_OBS_FMT=json   # legacy, debug only
RL_OBS_FMT=b64    # per-field base64 LE float32 / uint8
RL_OBS_FMT=pack   # one base64 blob in JSONL
RL_OBS_FMT=bin    # default — raw frames (no base64)
```

### Binary frame (`bin`)

```text
magic "XRL1" | ver u8 | flags u8 | schema_ver u16 LE
| meta_len u32 LE | blob_len u32 LE | meta JSON | tensor blob

flags: bit0=ok  bit1=done  bit2=has_obs

blob field order (same as pack):
  u8:  patches
  f32: actors, targets, roomCoords
  u8:  roomMask, actorMask, targetMask,
       intentMask, dirMask, targetSelectMask, amountMask
```

Commands (Python→Node) stay JSONL lines. Responses are binary frames when `bin`.

Wire size vs b64: **~26% fewer bytes** + no base64/JSON-of-tensors CPU.

## GPU (kernels / compile / VRAM)

| Issue | Fix | Status |
|-------|-----|--------|
| Per-slot mask `.clone()` + stride churn | `_open_class0` out-of-place; permute-once contiguous | done |
| `_DIR_TYPES.to(device)` every forward | `register_buffer` on Actor | done |
| 3 CUDA graphs (agent+actor+critic) | **actor + critic only**; act = actor then critic | done |
| Update recompile (warmup B=N_env ≠ mb) | warmup capture at **update minibatch** B | done |
| Variable terminal V B | always `value_only` at **B=N** (pad + mask) | done |
| Last mb shape change | static pad last mb to `mb` | done |
| Dual trunk activations peak | sequential actor backward → del → critic | done |
| Host float32 masks T×N | keep **uint8** on host; promote on H2D only | done |
| Host list+stack rollout | **`HostRolloutBuffer`** starts at base horizon and grows once on competent rollouts | done |
| H2D non-contiguous | `contiguous()` once in promote / mb stream | done |
| Slow matmul | TF32 + cudnn.benchmark + matmul precision high | done |
| Dual full ViT every act | still 2 trunks (VAPO); value_only for bootstrap only | accepted |
| Encode pad-to-4 on host | still pads for stack; trunk slices frozen R | accepted |

### Compile contract (do not break)

`torch.compile(..., dynamic=False, mode=reduce-overhead)` specializes on **exact shapes**. Allowed graphs:

| Graph | B | Mode | Notes |
|-------|---|------|-------|
| act (actor sample) | `num_envs` | eval | `action=None` |
| value_only / act critic | `num_envs` | eval | **never** variable `n_term` |
| update actor | `minibatch` | train | `action=dict`; last mb padded |
| update critic | `minibatch` | train | same B |

Anything that changes B (variable terminal stack, unpadded last mb, R flipping) forces **recompile** and can thrash CUDA graphs. `HostRolloutBuffer` does not change B — it only removes host alloc thrash.

## Data pipelines

| Layer | Tool | Rule |
|-------|------|------|
| Act / step | numpy + torch | **no polars** |
| Update metrics | polars parquet | batch flush off hot path |
| Offline traj | polars / parquet | optional future store |
| TB | SummaryWriter | charts only; not full tensors |

```bash
pip install -e "samples/rl[metrics]"   # polars + pyarrow
```

## Do not

- Put polars / DataFrame on the act or step hot path
- JEPA / latent think for throughput
- Dual full ViT growth before IPC is saturated
- Per-env B=1 value bootstraps in a loop
- Full-obs `.cpu()` every rollout step when host mirror exists

## Joint pretrain (same expert)

BC and critic train on **one** env fleet (`pretrain_joint`) — do not launch two pretrain jobs. Env-bound; scale with `--num-envs`, not dual processes.

### Scripted floor

Re-establish the 6,000-tick economy floor after every teacher/action ABI change.
Historical 2026-07-17 results in `JOB0-REPORT.md` belong to the old four-worker
teacher and are not evidence for the current specialist/construction/claim stack.

### Curriculum default

Joint pretraining distributes `empty,seed_creep,seed_full,seed_claimer` across
workers by default. Stage-separated metrics are required so seeded states cannot
hide failure to bootstrap from empty.

### Joint pretrain recipe (LEGACY Wave-9 BC replaced)

```bash
python3 -m samples.rl.agent.pretrain_joint \
  --num-envs 4 --steps 6000 --chunk 512 --device cpu \
  --max-episode 6000 \
  --validation-steps 256 --closed-loop-steps 6000 \
  --curriculum empty,seed_creep,seed_full,seed_claimer \
  --save samples/rl/runs/joint_pretrain_v2.pt
```

Do not queue until readiness gates + user auth. See `07-TRAINING-PLAYBOOK.md`.

## Expected SPS impact (order of magnitude)

| Fix | SPS | Notes |
|-----|-----|-------|
| b64/pack vs json | **3–10×** env IPC | already dominant for multi-env |
| bin vs pack/b64 | **+15–40%** IPC CPU | drop base64 + huge JSON.stringify |
| fuse encode+reward | **~1.2–1.5×** Node step | kill second loadRoom/runForUser |
| lean meta | **~1.1–1.3×** IPC CPU | smaller JSON.stringify every frame |
| cached PathFinder | workload-dependent | parity preserved while avoiding a full route every tick |
| dual host obs | **1.1–1.4×** collect | kills large D2H; env still dominates |
| scale num-envs | **~linear until cores** | primary remaining wall-clock lever |
| static last-mb pad | GPU compile | fewer recompiles / stable reduce-overhead |
| batch term V + value_only | small–medium | episode boundaries + bootstrap |
| fused categorical | small GPU | more if compile fuses better |

## Remaining high-value (not yet)

| Optim | Effort | Note |
|-------|--------|------|
| Encode buffer pool / cheaper paint | M–L | largest remaining Node CPU inside encode |
| Binary action commands | M | cmds still JSONL |
| Sparse/paged spatial storage | M–L | retain expansion correctness without dense four-room history |
| Dual-trunk act share/defer V | M–L | VAPO quality tradeoff |
