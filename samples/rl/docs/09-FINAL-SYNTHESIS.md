# Final synthesis — Optimal Screeps ML bot (latent 10-agent program)

## What “optimal” means here

**Single-room sim closed loop** under xxscreeps RL harness constraints — not live MMO rank.

Success = scripted-baseline-competitive on W7N3: spawn → harvest → deliver → upgrade, then RCL2 / extensions / full mine.

## Expert consensus (after 3 rounds of red team)

| Rank | Insight | Source |
|------|---------|--------|
| 1 | Fix MDP (post-tick obs) before any learning | E9, R2 |
| 2 | Type-gated logπ + hard masks | E5, E9 |
| 3 | Reward = **only equal H+C** (no shaping) | User + anti-hack |
| 4 | Critic scale contract (same r; trunk-only if needed) | E3, E4 |
| 5 | **BC policy ∥ critic pretrain**, then PPO | User + E10 |
| 6 | VAPO hygiene: warmup, KL stop, truncation | E3, param-golf |
| 7 | Steal HL-Gauss/entity-attn only if needed later | R2 demotion |
| 8 | Never port JEPA latent-think to Screeps for 1-GPU-week | E10 |

## Parameter-golf transfer (allowed)

- Separate actor/critic  
- Clip-higher  
- Length-adaptive λ  
- Reward-variance gates  
- HL-Gauss **optional** after scale-aligned MSE  

## Parameter-golf transfer (forbidden)

- Latent THINK/EMIT  
- JEPA world models for full-obs single room  
- Giant training-only critics as identity  
- **TI-BC via intent→AR inverse dynamics** — **DEFERRED** (user 2026-07-17); BC stays scripted-only  

## Code state (this session)

P0 stack implemented + unit tested (CPU). Scripted baseline + BC path + body templates + construction site intent. Docs under `samples/rl/docs/`.

## Honest residual risk

Without a measured scripted floor and BC init, pure PPO may still stall. The stack is no longer **self-poisoning**; it is still **hard**. Next authorized step is Job0/Job1 in `07-TRAINING-PLAYBOOK.md` via mlq.
