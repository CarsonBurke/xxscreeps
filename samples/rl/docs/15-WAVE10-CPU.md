# Wave 10 standing order (CPU while GPU busy)

**Date:** 2026-07-17  
**Constraint:** GPU claimed by other jobs — do **not** start PPO / mlq CUDA work. CPU + bookkeeping only.  
**Canonical W10 file** (the former `15-WAVE10-STANDING-ORDER.md` is a pointer here).  
**BC primary:** moved to **Wave 11** — [`16-WAVE11-STANDING-ORDER.md`](16-WAVE11-STANDING-ORDER.md).

## Three deliverables (this wave)

1. **Commit prep** — stage source/docs only; never `runs/` or `*.pt` (gitignored).
2. **Docs** — keep playbook + Job0 + changelog coherent with dual-source floor + W10 expert eval.
3. **`eval_expert`** — TI skill floor under same `r=H+C` as critic pretrain (not a BC teacher).

## 1) Commit prep

### Track (source of truth)

```text
samples/rl/
  schema.json
  README.md
  .gitignore
  pyproject.toml
  agent/*.py          # including eval_scripted.py, eval_expert.py, pretrain_*, train, model, …
  env/*.mjs
  docs/*.md           # 00–16 + JOB0-REPORT + COMMIT-PREP
```

### Never stage

| Path | Why |
|------|-----|
| `samples/rl/runs/**` | nested `.gitignore` → `runs/` |
| `**/*.pt` | checkpoints |
| `__pycache__/`, `.venv/` | local |
| engine `screeps/` world state | not RL sample |

Verify before commit:

```bash
# from repo root
git check-ignore -v samples/rl/runs/policy.pt samples/rl/runs/bc_scripted.pt
# expect: samples/rl/.gitignore

git status -- samples/rl
# should list source/docs only, no .pt
```

### Suggested Conventional Commit split

| Commit | Scope |
|--------|--------|
| `feat(rl): post-tick MDP, H+C reward, scripted BC path` | env + agent core + schema |
| `docs(rl): expert corpus, playbook, Job0 floor, wave standing orders` | `docs/` + README |
| `feat(rl): eval_scripted + eval_expert floors` | eval drivers only |

Draft subject if squashing one commit:

```text
feat(rl): latent redesign stack + scripted/TI floors (no checkpoints)
```

Body bullets: post-tick obs; r=H+C only; type-gated logπ; scripted multi-cycle teacher; BC∥critic paths; Job0 dual-source floor H1108 C307 skill≈1.42@1k; TI-BC deferred; TI remains critic/obs only.

### Reward schema gate (must still hold)

`schema.json` reward block is **only** `energyHarvested` + `controlPoints` at 1.0 — no spawn/deliver/site shaping. Re-check if any agent re-opens the file.

## 2) Docs status

| Doc | Role after W10 / W11 |
|------|----------------|
| `14-WAVE9-STANDING-ORDER.md` | historical BC auth; artifact did not land |
| `15-WAVE10-CPU.md` | **this file** — commit + expert floor while GPU busy |
| `16-WAVE11-STANDING-ORDER.md` | **SUPERSEDED** (split BC invalid) |
| `16-JOINT-PRETRAIN-STANDING.md` | **current** — same-expert joint Job1 |
| `JOB0-REPORT.md` | scripted floor (dual-source ≈1.42 e/t; frozen) |
| `07-TRAINING-PLAYBOOK.md` | Job0 pass; Job1 = `pretrain_joint` |
| `11-REWARD-AND-PRETRAIN.md` | same-expert joint (authoritative) |

## 3) eval_expert (CPU)

```bash
export RL_NODE="$(mise exec node@24 -- which node)"
cd /home/marvin/Documents/repositories/xxscreeps

python3 -m samples.rl.agent.eval_expert --ticks 1000 --skip-warmup 500 \
  2>&1 | tee samples/rl/runs/eval_expert_w10.log
```

Requires `../The-International-Open-Source/dist` (or `--bot-dir`).

| Metric | Meaning |
|--------|---------|
| `skill_rate` | full-run mean (H+C)/T — **cold start dilutes** |
| `skill_post_warmup(500..)` | matches critic `--skip-warmup-steps 500` train region |
| `skill_last500` | sustained multi-creep / late economy |

**Pass for critic usefulness:** post-warmup skill ≫ 0 and preferably ≥ scripted dual-source floor (~1.42 e/t).  
**Not a fail if G1 >200:** TI RCL0 cold start is expected; fencepost is post-warmup H+C.

## Explicit non-goals (still)

- PPO / GPU mlq  
- TI→AR inverse dynamics / TI-BC  
- Reward shaping revival  
- Architecture rewrites (HL-Gauss, entity attn, bin protocol) unless env-bound and CPU-only

## After GPU free / BC still missing

```text
bc_scripted.pt (Wave 11) + critic_pretrained*.pt
  → Job2 PPO --resume BC --critic-pretrain trunk
```

If BC still missing, run **Wave 11 A** first (`16-WAVE11-STANDING-ORDER.md`).
