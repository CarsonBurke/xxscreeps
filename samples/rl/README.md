# Screeps RL (ViT + 2D-RoPE + PPO)

PyTorch reinforcement-learning agent for **xxscreeps**: a dual-network (actor / critic) policy that issues masked multi-intent actions in a Screeps-like room, trained with PPO and VAPO-style decoupled GAE.

Env side is Node (JSONL server over xxscreeps `simulate`). Train / watch side is Python 3.10+.

---

## Parameter count

| Network | Parameters | Role |
|---------|------------|------|
| **Actor** | **~798,695** | Autoregressive multi-intent policy |
| **Critic** | **~724,418** | Independent value network (no shared trunk) |
| **Total** | **~1.52M** | VAPO-style dual models |

Sizes are for the default config in [`schema.json`](./schema.json) (`dModel=128`, `nLayers=3`, `nHeads=4`).

---

## Model architecture

### Observation

| Field | Shape (per env) | Notes |
|-------|-----------------|-------|
| `patches` | `[R, 100, 700]` | `R≤4` rooms; each room is **10×10** patches of **5×5** tiles; tile feat dim **28** → `5·5·28=700` |
| `room_mask` | `[R]` | Active rooms (packed from the front) |
| `actors` | `[24, 24]` | My creeps + my spawns/towers (features: kind, pos, body fractions, energy, …) |
| `actor_mask` | `[24]` | Live actor slots |
| `targets` | `[48, 16]` | Sources, structures, creeps, resources, construction sites, … |
| `target_mask` | `[48]` | Live target slots |
| Masks | intent / dir / target-select / amount | Legal actions per actor × slot |
| `globals` | `[6]` | RCL, stored energy, control progress, creep count, GCL, bucket |

Encoding: [`env/encode.mjs`](./env/encode.mjs).

### Encoder (per network)

1. **Patch ViT** — linear project each patch → prepend **CLS** → **3× Pre-LN** transformer blocks  
   - **QK RMSNorm**, **2D RoPE** on patch tokens, **SDPA** attention  
2. **Per-room CLS** kept intact (no mean/max over patches beyond CLS)  
3. **Room packing** — only the active prefix of rooms is encoded (`R_use`, usually 1 for single-room BM)  
4. **Actors / targets** gather **their room’s CLS** (+ globals) by room index in the feature table  
5. **Critic backbone** — learned **attention pool** over room CLSes (not mean/max/min)

Implementation: [`agent/model.py`](./agent/model.py), RoPE: [`agent/rope2d.py`](./agent/rope2d.py).

### Policy (actor)

- Dual **intent slots** per actor (e.g. `move` + `transfer` in one tick).  
- **Autoregressive** over slots: type → (embed) → dir / target / amount.  
- Heads are categorical with **action masks** (illegal logits → −∞).  
- **Intent prior bias** at init: strong preference for **`harvest`** and **`move`**, strong negative on combat / claim / spawn spam (masks still allow `spawnCreep` when that is the only legal action on a spawn).

### Critic

- Separate trunk + MLP value head.  
- Trained against **λ = 1** Monte Carlo-style returns (VAPO).  
- Optional **TI expert pretrain** (`pretrain_critic.py`).

---

## Action space

Per **actor** (≤24), for each of **2 slots**, the policy outputs:

| Head | Size | Meaning |
|------|------|---------|
| **type** | 20 | Intent enum (below) |
| **dir** | 8 | N, NE, E, SE, S, SW, W, NW |
| **target** | 48 | Index into the **target table** for this tick |
| **amount** | 10 | Bins `[0, 1, 5, 10, 25, 50, 100, 200, 500, 1000]` — **0 = all / default** |

### Intent types

`none`, `move`, `harvest`, `transfer`, `withdraw`, `pickup`, `drop`, `upgradeController`, `build`, `repair`, `attack`, `rangedAttack`, `heal`, `rangedHeal`, `claimController`, `reserveController`, `attackController`, `dismantle`, `generateSafeMode`, `spawnCreep`

### Masking (high level)

| Actor | When intent is allowed |
|-------|------------------------|
| Creep + MOVE, fatigue 0 | `move` + all 8 dirs |
| Creep + WORK | `harvest`, `build`, `repair`, `upgradeController`, `dismantle` |
| Creep + CARRY, energy &gt; 0 | `transfer`, `drop` + amount bins |
| Creep + CARRY, free capacity | `withdraw`, `pickup` + amount bins |
| Creep + combat parts | `attack` / `rangedAttack` / `heal` / … |
| Spawn | `spawnCreep` |
| Tower | `attack`, `repair`, `heal` on room targets |

**Targets** are not raw (x, y): they are discrete entries in a table of up to **48** nearby / relevant objects (sources typically need **range ≤ 1**, interactive targets often **range ≤ 3**).

Application: [`env/actions.mjs`](./env/actions.mjs).

### Examples

| Goal | Typical slot |
|------|----------------|
| Harvest a source | `type=harvest`, `target=i` (source in table); dir/amount unused |
| Walk one tile | `type=move`, `dir=k` |
| Dump all energy into spawn | `type=transfer`, `target=spawn_j`, `amount=0` (all) |
| Take energy from container | `type=withdraw`, `target=container_j`, amount bin |
| Pick dropped energy | `type=pickup`, `target=resource_j` |
| Upgrade controller | `type=upgradeController`, `target=controller_j` |
| Spawn a creep (spawn actor) | `type=spawnCreep` (only legal type on spawn) |

Two slots can stack, e.g. slot0=`move`, slot1=`transfer` (engine applies both if valid).

---

## Training features

| Feature | Detail |
|---------|--------|
| **Algo** | PPO, separate AdamW for actor / critic (critic LR = 2× actor) |
| **VAPO GAE** | Critic λ=1 (MC); policy λ = `1 − 1/(α L)`, α=0.05, clamped |
| **Asymmetric policy clip** | Low ε=0.2 → ratio ≥ 0.80; high ε=0.28 → ratio ≤ 1.28 |
| **Value clip** | CleanRL-style `clip_vloss` with ε=0.2 |
| **Advantage norm** | Per-minibatch (CleanRL `norm_adv`) |
| **Reward norm** | Running RMS of **discounted returns** (Gymnasium / CleanRL-style); logs keep **raw** reward |
| **Rollouts** | Default 8 parallel envs, 512 steps; optional extend +500 when skill rate &gt; 5 e/t (cap 20k) |
| **skill_rate_et** | `(Σ harvestΔ + controlΔ) / T` over the rollout (sum across envs / ticks) — **not** reward |
| **Compile** | Optional `torch.compile(..., mode="reduce-overhead")` on agent / actor / critic monoliths |
| **Parallel envs** | ThreadPool over independent Node processes; bulk H2D after host decode |
| **Headful train** | `--headful` serves Screeps client on **env 0** only (`:21025`) |
| **TB runs** | Default logdirs are **timestamped** under `runs/tb-ppo/<stamp>` and `runs/tb-critic-pretrain/<stamp>` |

Reward:

```text
r = 1.0 * Δcontrol_points + 0.05 * energy_harvested
```

---

## Layout

```
samples/rl/
  schema.json          # shared dims, intents, model + PPO defaults
  README.md
  pyproject.toml
  env/
    server.mjs         # JSONL env: reset / step / expert / optional headful
    encode.mjs         # observation + masks
    actions.mjs        # apply intents to Game
  agent/
    model.py           # ViT trunk, Actor, Critic, intent prior
    rope2d.py
    gae.py             # decoupled VAPO GAE
    ppo.py             # PPO update + compile warmup
    running_stats.py   # reward RMS normalizer
    env_client.py      # Python ↔ Node JSONL
    vec_env.py         # parallel envs
    train.py           # PPO trainer
    pretrain_critic.py # TI expert critic pretrain
    watch.py           # evaluate / headful watch a checkpoint
    constants.py
```

---

## Requirements

- **Node ≥ 22** (mise `node@24` works): xxscreeps + loader  
- **Python ≥ 3.10**: `torch`, `numpy`, `tensorboard`  
- Optional: [The International](https://github.com/The-International-Screeps-Bot/The-International-Open-Source) build for critic pretrain (`../The-International-Open-Source/dist`)

From **xxscreeps repo root**:

```bash
# Optional expert bot
cd ../The-International-Open-Source && npm i && npm run build && cd -

export RL_NODE="$(mise exec node@24 -- which node)"  # or: which node
```

---

## Critic pretrain (TI expert)

```bash
RL_NODE="$(mise exec node@24 -- which node)" \
python3 -m samples.rl.agent.pretrain_critic \
  --num-envs 8 --steps 2000 --chunk 512 --epochs-per-chunk 4 --device cuda \
  --save samples/rl/runs/critic_pretrained.pt
# TB: samples/rl/runs/tb-critic-pretrain/<timestamp>/
```

---

## PPO train

```bash
# Long run (stop with Ctrl+C / kill)
RL_NODE="$(mise exec node@24 -- which node)" \
python3 -m samples.rl.agent.train \
  --steps 512 --num-envs 8 --minibatch 2048 --updates 1000000000 \
  --device cuda \
  --critic-pretrain samples/rl/runs/critic_pretrained.pt \
  --save samples/rl/runs/policy.pt

# Resume + watch env 0 in the Screeps client (slows training)
python3 -m samples.rl.agent.train \
  --resume samples/rl/runs/policy.pt \
  --headful --tick-ms 100 \
  ...same as above...
```

TensorBoard:

```bash
tensorboard --logdir samples/rl/runs --port 6008
# open the latest tb-ppo/<timestamp> or tb-critic-pretrain/<timestamp> run
```

Useful series: `charts/episodic_return`, `charts/mean_step_reward`, `charts/mean_step_reward_norm`, `charts/skill_rate_et`, `charts/reward_rms_std`, `losses/value_loss`, `losses/approx_kl`, `gae/*`.

---

## Watch a checkpoint

```bash
# Headful (browser)
RL_NODE="$(mise exec node@24 -- which node)" \
python3 -m samples.rl.agent.watch \
  --checkpoint samples/rl/runs/policy.pt \
  --headful --ticks 2000 --tick-ms 150 --deterministic

# Terminal HUD only
python3 -m samples.rl.agent.watch --checkpoint samples/rl/runs/policy.pt --ticks 500
```

Client: **http://127.0.0.1:21025/** — login **Player 1** / **rlwatch** (default) → room **W7N3**.

---

## Env protocol (JSONL)

stdin / stdout lines:

```json
{"cmd":"reset"}
{"cmd":"step","actions":{"types":[[...]],"dirs":[[...]],"targets":[[...]],"amounts":[[...]]}}
{"cmd":"reset_expert","botDir":"/path/to/TI/dist"}
{"cmd":"step_expert"}
{"cmd":"close"}
```

Set `RL_HEADFUL=1` (or `watch.py --headful` / `train.py --headful`) to attach the Screeps client to that sim process.

---

## Defaults (`schema.json`)

| Key | Default |
|-----|---------|
| `model.dModel` | 128 |
| `model.nLayers` | 3 |
| `model.nHeads` | 4 |
| `ppo.gamma` | 0.99 |
| `ppo.clip` / `clipHigh` | 0.2 / 0.28 |
| `ppo.minibatch` | 2048 |
| `ppo.rolloutSteps` | 512 |
| `ppo.numEnvs` | 8 |
| `ppo.normalizeReward` | true |
| `ppo.extendRateThreshold` | 5.0 (skill e/t) |

---

## Notes

- Early episodes have no creeps until `spawnCreep` succeeds on the spawn actor.  
- Intent prior biases exploration; masks still define legality.  
- Value loss can be large if the critic (e.g. TI-pretrained) lives at a different scale than policy returns; reward norm applies only on the **PPO** path.  
- Headful tick delay slows **all** parallel envs (step waits on the slowest).  
- Research harness: inference is out-of-process; not a CPU-safe MMO bot.

---

## License / upstream

Part of the [xxscreeps](https://github.com/laverdet/xxscreeps) tree under `samples/rl/`. Follow the root repository license for redistribution.
