# Consensus Optimal Design — Screeps ML Bot

Synthesized from 10 experts + code verification. **Latent phase:** implement correctness before GPU training.

## Thesis (Expert 10, refined)

> Screeps single-room economy is a long-horizon multi-agent logistics MDP with a free expert. Optimal is **imitation-first, curriculum-gated, lightly hierarchical online RL** on a **correct MDP** — not “better PPO” alone, not latent JEPA.

## Scope honesty

| Claim | Scope |
|-------|--------|
| **Near-term target** | Single-room W7N3-class: spawn loop, harvest→deliver→upgrade, RCL2, extensions→550, `hr≈18–20` local |
| **Not yet** | Live MMO competence, multi-room empire, combat, market |
| **Env hard cap today** | Fixed body + no site placement → upper bound ≈ multipurpose WCM swarm until action API expands |

## Priority stack

### P0 — Doomed without (MUST before serious training)

| # | Change | Status in this session |
|---|--------|------------------------|
| P0.1 | **Post-tick observations** (broken MDP H1) | **DONE** `server.mjs` |
| P0.2 | **Type-gated logπ / entropy** | **DONE** `model.py` |
| P0.3 | **Intent/energy/range masks + spawn legality** | **DONE** `encode.mjs` |
| P0.4 | **Reward: equal H+C only** (no shaped bonuses) | **DONE** — see `11-REWARD-AND-PRETRAIN.md` |
| P0.5 | **Bootstrap priors** (spawn/transfer/upgrade positive) | **DONE** schema |
| P0.6 | **Critic scale contract** (no raw TI head into normed PPO) | **DONE** trunk-only + HL-Gauss |
| P0.7 | **Value warmup + KL early-stop + epochs=2** | **DONE** train/ppo |
| P0.8 | **skill_rate / (T·N)** | **DONE** train.py |
| P0.9 | **Resume: opts + RMS + global_step** | **DONE** |
| P0.10 | **Policy BC ∥ critic pretrain** (same r=H+C) | **DONE** paths; Job0 floor pass; **BC ckpt pending Wave 11** |

### P1 — Bootstrap learnability

| # | Change | Status |
|---|--------|--------|
| P1.1 | Sticky macros + moveTo (harvest/deposit/upgrade) | **DONE** in `actions.mjs` (`approachOr` + sticky path cache + engine `moveTo`); teacher sets type/target, apply path navigates |
| P1.2 | Discrete spawn body templates | **DONE** amount head → template |
| P1.3 | `createConstructionSite` for extensions | **DONE** RCL≥2 |
| P1.4 | Curriculum starts (seeded mid-economy → empty) | **PARTIAL** — static `empty\|seed_creep\|seed_full` modes exist; default **empty** for spawn learning; progressive mid→empty schedule not required for Wave 11 BC |
| P1.5 | Target priority packing | **DONE** |
| P1.6 | Actor freeCap / phase bits | **DONE** (slots 16–21) |
| P1.7 | Contiguous masks for compile | **DONE** |

### P2 — Architecture reallocation

| # | Change |
|---|--------|
| P2.1 | Shared light spatial + entity attention |
| P2.2 | CTDE critic over actors (not room-only) |
| P2.3 | Drop dual full ViT |

### P3 — Explicitly cut (1 GPU-week)

- JEPA / latent THINK-EMIT / giant separate critic  
- Pure empty-room PPO from random  
- **Shaped rewards** (spawn/deliver/site/extinction) — reward-hackable  
- **TI → AR inverse dynamics for TI-BC** — **DEFERRED** (user 2026-07-17). TI does not emit `types/dirs/targets/amounts`; reconstructing AR labels from intents is a research project. **BC stays scripted-only; TI is critic/obs only.**  

## Learning pipeline (when training is authorized)

```
Phase 0  Post-tick MDP + masks + r = 1·H + 1·C only
Phase 1  PARALLEL: BC actor (scripted) ∥ critic pretrain (same r)
Phase 2  PPO: BC actor + critic trunk; value warmup → joint
Phase 3  Eval gates via diagnostics (RCL/hr), not reward composition
```

Bootstrap = BC + masks + priors — not dense reward. See `11-REWARD-AND-PRETRAIN.md`.  
Wave 11 standing order: run BC on CPU while GPU blocked — `16-WAVE11-STANDING-ORDER.md`.

## Success gates (do not claim optimality without)

| Gate | Pass |
|------|------|
| M0 | within-batch reward std > ε after BC |
| M1 | first creep median ≤ 200 ticks empty room |
| M2 | RCL2 by 2000 ticks |
| M3 | capacity ≥ 550 by 4000 |
| M4 | local hr ≥ 18 sustained 200 ticks by 6000 |
| M5 | no 50-tick zero-creep window |

## Metric definitions

- `skill_rate_et` = mean over envs of `(ΣΔH + ΣΔC) / T`  **per env**
- Primary product metrics: RCL time, t_550, `hr/mh`, death spiral absence
- TB `value_loss`: CE nats if HL-Gauss (~1.5–3.5 healthy); not comparable to old MSE

## Parameter-golf transfers (only)

| Keep | Drop |
|------|------|
| Separate critic | Latent thoughts |
| Clip-higher 0.2/0.28 | JEPA world model |
| Length-adaptive λ (fix L=segment) | 28M critic toys |
| Reward-variance train gate | Tier-0 proxy gaming |
| HL-Gauss for bounded returns | Symlog [0,1] math support |

## Kill conditions (re-open training only if)

1. Agent sees post-tick state  
2. Nonzero pipeline reward variance (spawn/transfer/harvest) under BC or curriculum  
3. Critic and GAE share one reward scale  
4. KL cannot explode past target without abort  
5. Scope = single-room economy milestones, not “MMO bot”  

Until BC checkpoint lands (`bc_scripted.pt`), pure PPO may still stall — but the MDP and loss are no longer self-poisoning.
