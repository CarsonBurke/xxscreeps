# Commit prep (samples/rl) — Wave 10

**Wave:** `15-WAVE10-CPU.md` (approved 2026-07-17)  
**Do not commit until `git status` shows only intended source/docs.**  
Checkpoints and TB under `runs/` are intentionally untracked.

## Staging checklist

```bash
cd /home/marvin/Documents/repositories/xxscreeps

# 1) Ignore fence
git check-ignore -v samples/rl/runs samples/rl/runs/policy.pt \
  samples/rl/agent/__pycache__
# expect hits from samples/rl/.gitignore

# 2) Review
git status --short samples/rl
git diff --stat samples/rl

# 3) Stage (adjust if you split commits)
git add samples/rl/schema.json samples/rl/README.md samples/rl/.gitignore \
  samples/rl/pyproject.toml samples/rl/agent samples/rl/env samples/rl/docs

# 4) Sanity: no binary noise
git diff --cached --name-only | grep -E '\.(pt|pyc)$|runs/' && echo FAIL || echo OK
```

### Ignore fence (verified content)

`samples/rl/.gitignore`:

```text
runs/
__pycache__/
*.pyc
.venv/
*.pt
!schema.json
```

So `runs/**` and all `*.pt` stay out of commits by default.

## Message templates

### Single squash

```
feat(rl): post-tick H+C stack + same-expert joint pretrain path

- Post-tick obs; type-gated logπ; reward = equal energyHarvested+controlPoints only
- Same-expert joint BC∥critic (pretrain_joint); scripted v1; TI eval-only
- eval_scripted (Job0 dual-source H1108 C307 skill≈1.42@1k) + eval_expert
- Design corpus under samples/rl/docs (playbook, joint standing order)
- Checkpoints intentionally gitignored (runs/, *.pt)
```

### Split (preferred)

1. `feat(rl): env+agent stack for post-tick H+C PPO/BC`  
2. `docs(rl): expert corpus, playbook, Job0, wave standing orders`  
3. `feat(rl): eval_scripted and eval_expert CPU floor drivers`

## Pre-push gates (CPU)

```bash
export RL_NODE="$(mise exec node@24 -- which node)"
python3 -m samples.rl.agent.test_latent_unit
python3 -m samples.rl.agent.eval_scripted --ticks 200   # smoke; full Job0 = 1000
# optional if TI dist present:
# python3 -m samples.rl.agent.eval_expert --ticks 200 --skip-warmup 50
```

Last known: unit tests expanded (joint `_train_chunk` shapes + schema contracts); Job0 dual-source floor **H1108 C307 skill≈1.42@1k** (G1–G4 PASS). **Do not queue pretrain** until readiness gates.

## Out of scope for this commit

- `screeps/` world DB / terrain  
- `worktrees/`  
- `packages/xxscreeps` engine changes (separate commits if any)  
- Any `*.pt` or TensorBoard events  
- Running PPO / GPU mlq (Wave 10 non-goal)
