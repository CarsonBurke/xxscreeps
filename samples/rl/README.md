# Screeps RL (ViT + 2D-RoPE + PPO)

PyTorch agent for **[xxscreeps](https://github.com/laverdet/xxscreeps)**: dual actor/critic, masked multi-intent actions, **PPO** + **VAPO**-style decoupled **GAE** (critic λ = 1 Monte Carlo; policy length-adaptive λ).

- **Env**: Node JSONL server over xxscreeps `simulate`
- **Train / watch**: Python 3.10+

---

## Parameter count

| Network | Parameters | Role |
|---------|------------|------|
| **Actor** | **~798,695** | Autoregressive multi-intent policy |
| **Critic** | **~724,418** | Independent value net (no shared trunk) |
| **Total** | **~1.52M** | VAPO-style dual models |

Default config in [`schema.json`](./schema.json): `dModel=128`, `nLayers=3`, `nHeads=4`.

---

## Model architecture

### Observation

| Field | Shape (per env) | Notes |
|-------|-----------------|-------|
| `patches` | `[R, 100, 700]` | `R≤4` rooms; **10×10** patches of **5×5** tiles; 28 tile feats → `700` |
| `room_mask` | `[R]` | Active rooms (packed from the front) |
| `actors` | `[24, 24]` | My creeps + spawns/towers |
| `actor_mask` | `[24]` | Live actor slots |
| `targets` | `[48, 16]` | Sources, structures, creeps, resources, sites, … |
| `target_mask` | `[48]` | Live target slots |
| Masks | intent / dir / target-select / amount | Legal actions per actor × slot |
| `globals` | `[6]` | RCL, stored energy, control progress, creeps, GCL, bucket |

Encoding: [`env/encode.mjs`](./env/encode.mjs).

### Encoder (per network, separate weights)

1. **Patch ViT** — project patches → **CLS** → **3× Pre-LN** blocks (QK **RMSNorm**, **2D RoPE**, **SDPA**)  
2. Keep **per-room CLS** (no mean/max over patches after the transformer)  
3. **Room packing** — only active rooms (usually 1 on the single-room test map)  
4. Actors/targets add **their room’s CLS** (+ globals) via room index in features  
5. Critic backbone: **attention pool** over room CLSes (not mean/max/min)

Code: [`agent/model.py`](./agent/model.py), [`agent/rope2d.py`](./agent/rope2d.py).

### Policy (actor)

- **2 intent slots** per actor (e.g. `move` + `transfer`)  
- **AR** over slots: type → embed → dir / target / amount  
- Categorical heads + **masks**  
- **Init prior**: strong bias to **`harvest`** / **`move`**; strong negative on combat / claim / spawn spam (spawns still work when only `spawnCreep` is legal after masking)

### Critic

- Separate trunk + MLP value head  
- λ = 1 MC targets (VAPO)  
- Optional pretrain vs **[The International](https://github.com/The-International-Screeps-Bot/The-International-Open-Source)** open-source Screeps bot (`pretrain_critic.py`)

---

## Action space

Per **actor** (≤24), per **slot** (2):

| Head | Size | Meaning |
|------|------|---------|
| **type** | 20 | Intent enum |
| **dir** | 8 | N … NW |
| **target** | 48 | Index into this tick’s target table |
| **amount** | 10 | Bins `[0, 1, 5, 10, 25, 50, 100, 200, 500, 1000]` — **0 = all / default** |

### Intent types

`none`, `move`, `harvest`, `transfer`, `withdraw`, `pickup`, `drop`, `upgradeController`, `build`, `repair`, `attack`, `rangedAttack`, `heal`, `rangedHeal`, `claimController`, `reserveController`, `attackController`, `dismantle`, `generateSafeMode`, `spawnCreep`

### Masking (high level)

| Actor | When allowed |
|-------|----------------|
| Creep + MOVE, fatigue 0 | `move` + 8 dirs |
| Creep + WORK | `harvest`, `build`, `repair`, `upgradeController`, `dismantle` |
| Creep + CARRY, energy &gt; 0 | `transfer`, `drop` + amount bins |
| Creep + CARRY, free capacity | `withdraw`, `pickup` + amount bins |
| Creep + combat parts | attack / heal / … |
| Spawn | `spawnCreep` |
| Tower | `attack`, `repair`, `heal` on room targets |

Targets are **not** raw (x, y): discrete table of ≤48 nearby objects (sources typically range ≤1).  
Apply: [`env/actions.mjs`](./env/actions.mjs).

### Examples

| Goal | Slot |
|------|------|
| Harvest source | `type=harvest`, `target=i` |
| Walk | `type=move`, `dir=k` |
| Empty energy to spawn | `type=transfer`, `target=spawn_j`, `amount=0` |
| Withdraw from container | `type=withdraw`, `target=container_j`, amount bin |
| Pickup dropped energy | `type=pickup`, `target=resource_j` |
| Upgrade controller | `type=upgradeController`, `target=controller_j` |
| Spawn creep (spawn actor) | `type=spawnCreep` |

Two slots can stack in one tick if both are valid.

---

## Training features

| Feature | Detail |
|---------|--------|
| **PPO** | Separate AdamW; critic LR = 2× actor |
| **VAPO GAE** | Critic λ=1 (MC); policy λ = `1 − 1/(α L)`, α=0.05 |
| **Asymmetric clip** | Policy ratio ∈ [0.80, 1.28] (ε_low=0.2, ε_high=0.28) |
| **Value clip** | CleanRL `clip_vloss`, ε=0.2 |
| **Adv norm** | Per-minibatch (`norm_adv`) |
| **Reward norm** | Running RMS of discounted returns; logs keep **raw** reward |
| **Rollouts** | 8 envs × 512 steps; extend +500 if skill &gt; 5 e/t (cap 20k) |
| **skill_rate_et** | `(Σ harvestΔ + controlΔ) / T` — game metric, not reward |
| **Compile** | Optional `torch.compile(mode="reduce-overhead")` |
| **Parallel envs** | ThreadPool over Node processes |
| **Headful** | `--headful`: Screeps client on env 0 (`:21025`) |
| **TensorBoard** | Timestamped dirs under `runs/tb-ppo/` and `runs/tb-critic-pretrain/` |

```text
r = 1.0 * Δcontrol_points + 0.05 * energy_harvested
```

---

## Layout

```
samples/rl/
  schema.json
  README.md
  env/          # server.mjs, encode.mjs, actions.mjs
  agent/        # model, ppo, gae, train, pretrain_critic, watch, …
```

---

## Requirements

- Node ≥ 22 (e.g. mise `node@24`)  
- Python ≥ 3.10: `torch`, `numpy`, `tensorboard`  
- Optional expert pretrain: build **[The International](https://github.com/The-International-Screeps-Bot/The-International-Open-Source)** → `../The-International-Open-Source/dist`

```bash
export RL_NODE="$(mise exec node@24 -- which node)"
```

---

## Critic pretrain (The International expert)

```bash
RL_NODE="$(mise exec node@24 -- which node)" \
python3 -m samples.rl.agent.pretrain_critic \
  --num-envs 8 --steps 2000 --chunk 512 --epochs-per-chunk 4 --device cuda \
  --save samples/rl/runs/critic_pretrained.pt
# TensorBoard: samples/rl/runs/tb-critic-pretrain/<timestamp>/
```

---

## PPO train

```bash
RL_NODE="$(mise exec node@24 -- which node)" \
python3 -m samples.rl.agent.train \
  --steps 512 --num-envs 8 --minibatch 2048 --updates 1000000000 \
  --device cuda \
  --critic-pretrain samples/rl/runs/critic_pretrained.pt \
  --save samples/rl/runs/policy.pt

# Resume + live client on env 0 (slows training)
python3 -m samples.rl.agent.train \
  --resume samples/rl/runs/policy.pt --headful --tick-ms 100 \
  ...
```

```bash
tensorboard --logdir samples/rl/runs --port 6008
```

Key series: `charts/episodic_return`, `charts/mean_step_reward`, `charts/mean_step_reward_norm`, `charts/skill_rate_et`, `losses/value_loss`, `losses/approx_kl`, `gae/*`.

---

## Watch a checkpoint

```bash
RL_NODE="$(mise exec node@24 -- which node)" \
python3 -m samples.rl.agent.watch \
  --checkpoint samples/rl/runs/policy.pt \
  --headful --ticks 2000 --tick-ms 150 --deterministic
```

Client: **http://127.0.0.1:21025/** — **Player 1** / **rlwatch** → room **W7N3**.

---

## Env protocol (JSONL)

```json
{"cmd":"reset"}
{"cmd":"step","actions":{"types":[[...]],"dirs":[[...]],"targets":[[...]],"amounts":[[...]]}}
{"cmd":"reset_expert","botDir":"/path/to/The-International-Open-Source/dist"}
{"cmd":"step_expert"}
{"cmd":"close"}
```

---

## Defaults ([`schema.json`](./schema.json))

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

- No creeps until `spawnCreep` succeeds.  
- Value loss can be large if a critic pretrained on **The International** (raw expert returns) is off-scale vs the policy; reward norm is on the **PPO** path only.  
- Headful tick delay slows all parallel envs.  
- Research harness — not a CPU-safe live MMO bot.

---

## License / upstream

Under `samples/rl/` in [xxscreeps](https://github.com/laverdet/xxscreeps). Follow the root repo license.
