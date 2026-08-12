# Expert Corpus (condensed)

Evidence base: `samples/rl/` code + `runs/ppo_train.log` + `parameter-golf/LATENT_RL_PLAN.md`.

---

## E1 — Reward (Expert 1)

**Primary failure is reward geometry, not PPO knobs.**

- Only `Δcontrol` and `0.05·Δharvest` scored. Zero reward for spawn, transfer, build, RCL.
- Fixed body `[W,C,M,M]` caps harvest at **2 e/t/creep**. 10+ e/t economy needs multi-creep + fuel.
- `w_C / w_H = 20` while skill claims H+C → objective ≠ metric.
- **skill_rate bug**: `total_H+C / T` not `/ (T·N)` → ~8× inflated with 8 envs.
- TI critic raw + PPO reward-norm = value_loss 1e5.
- Tier-0 lesson: proxy reward that does not require pipeline compliance is gameable / uninformative.

**Prescription (superseded by user mandate):** equalize `w_H=w_C` **only** — do **not** add spawn/deliver/site shaping (reward-hackable). Bootstrap via BC∥critic pretrain + masks/priors. See `11-REWARD-AND-PRETRAIN.md`.

---

## E2 — Architecture (Expert 2)

**Dual full ViTs mis-spend the 1.5M budget.**

- Joint logprob sum over actors×slots×4 heads + single advantage = credit collapse.
- No cross-actor attention; independent policies on CLS+row.
- Critic ignores actor table content (backbone = room pool only).
- Spawn has no body/role args.

**Prescription:** shared light spatial + entity attention; CTDE critic over all units; type-conditioned pointers; reallocate params 60–70% actor / 30–40% critic; drop second ViT.

---

## E3 — VAPO/PPO (Expert 3)

**`value_loss≈90k` ≈ 0.5·(V_expert − R_norm)². KL 0.9 = 4 epochs on garbage advantages.**

Gaps vs param-golf VAPO:
1. Raw pretrain + reward-norm mismatch (ROOT)
2. No value warmup
3. No KL early-stop
4. L = rollout T not segment length
5. Time-limit treated as terminal (bootstrap 0)

**Prescription:** align scales → value warmup 30–50 → epochs 1–2 + target_kl 0.02 → fix truncation → HL-Gauss optional after.

---

## E4 — Critic (Expert 4)

- MSE unbounded head + scale fracture + λ=1 MC = pathological.
- HL-Gauss on normalized support **[−10, 10]**, 101 bins, σ_ratio=2, zero-init head + prior bias.
- Keep expert for **trunk**, re-init value head or re-pretrain under same RMS.
- Warmup gates: CE ≪ log(101), EV>0.3, |v−ret|<0.5, gnC≪50.

---

## E5 — Action space (Expert 5)

- Factored AR is right long-term; wrong primary exploration space for bootstrap.
- **Macros + sticky pathing** (`harvest/deposit/upgrade` via moveTo) early; primitives residual later.
- Critical mask bugs: unconditional targets, no energy/range gates, spawn always legal, dummy head logprobs.
- Prior `spawnCreep:−3`, `transfer:−1` fights bootstrap.

**Highest ROI without hierarchy:** type-gated logprob + range/kind masks + spawn/deposit priors + delivery reward.

---

## E6 — Observation (Expert 6)

- Actors: only 0–15 used; free capacity not in obs (only in mask calc).
- No POI Δ/range/path costs → navigation is open-loop 8-dir lottery.
- Target table: FIND_STRUCTURES floods 48 slots; walls can crowd out controller.
- ViT expensive for ≤10 early entities; entity-first wins for RCL1–2.

**Must add:** freeCap, empty/full, POI vectors, energyAvailable, prioritized targets, pair features for bilinear head.

---

## E7 — Economy domain (Expert 7)

**MVP is a phase machine:** upgrade 200 → 5 extensions → 5W mine → haul → surplus upgrade → remotes.

| Gate | Solved |
|------|--------|
| Local full mine | `hr ≈ 20` (2 sources) |
| Under 1 spawn | `hr → mh` (~45–50 on W7N3) |
| RCL2 | ideally &lt;1.5k ticks |
| Cap 550 | unlocks 5W body |

**Env hard caps:** fixed body + no `createConstructionSite` ⇒ policy upper bound ≈ multipurpose WCM swarm.

---

## E8 — Infrastructure (Expert 8)

- Env/IPC-bound ~20–45 SPS (JSON patches ~280k floats/step).
- torch.compile recompile: non-contiguous mask slices.
- Resume: opts, global_step, reward_normalizer **not restored**; update loop restarts at 0.
- Train headless; `max-parallel-runs: 1` per GPU.

---

## E9 — Red team current (Expert 9) — REJECT

S0 kill list:
1. **Pre-tick observation** (broken MDP)
2. Joint multi-actor logprob / single reward
3. Sparse mis-shaped reward
4. Critic scale poison
5. Entropy dominates objective
6. Action space cannot express competent bot

Empirical: skill 0.2–0.5 vs expert 10; watch shows zero harvest; V~539 while r=0.

---

## E10 — Red team optimal (Expert 10)

**Optimal = imitation-first, curriculum-gated, lightly hierarchical RL.**

| Approach | EV / 1 GPU-week |
|----------|-----------------|
| BC → curriculum RL | **Highest** |
| Just fix PPO | Hygiene only |
| Full JEPA/latent think | **Negative** |

Steal from param-golf: dual value discipline, clip-higher, length-adaptive λ, reward-variance gates.  
**Do not** port latent THINK/EMIT or JEPA world models to Screeps.
