# Wave 9 standing order (GPU-blocked hour)

**Approved:** 2026-07-17 (user OK)  
**Constraint:** no GPU / no heavy mlq until free; do CPU + code path work.  
**Outcome:** authorized correctly; **`bc_scripted.pt` did not land**.  
**Carryover:** W10 = docs/commit/eval_expert (`15-WAVE10-CPU.md`); **BC primary is now Wave 11** — [`16-WAVE11-STANDING-ORDER.md`](16-WAVE11-STANDING-ORDER.md).

## Standing order

While GPU is blocked, do **only** work that unblocks the learning pipeline without competing for CUDA:

1. **Produce `bc_scripted.pt` on CPU** (highest leverage — last playbook gate before PPO).
2. Keep the **scripted teacher floor** honest (Job0 multi-seed / windowed skill).
3. Keep **curriculum default = `empty`**; `seed_*` remains opt-in densification only.
4. Do **not** start PPO, critic re-pretrain on GPU, or architecture rewrites in this wave.
5. Do **not** revive TI→AR BC / inverse dynamics (deferred).

## Preconditions (current)

| Item | Status | Evidence |
|------|--------|----------|
| Post-tick MDP + masks + r=H+C | done | design docs + code |
| Scripted multi-cycle teacher | done | `scripted_baseline.mjs` + approachOr/`moveTo` |
| Job0 G1–G4 on empty | **pass @1k** | dual-source floor; see `JOB0-REPORT.md` |
| Critic checkpoint | present | `runs/critic_pretrained.pt` |
| BC checkpoint | **missing** | no `runs/bc_scripted.pt` |
| TI skip-warmup fencepost | correct | W8 review: `t >= skip_warmup` |

### Job0 floor (W9 measured job0c; later dual-source reconfirm)

```text
# as measured in Wave 9 (job0c):
ticks=1000  skill_rate≈1.395  H=1106  C=289  spawnSuccess=4
first_creep_tick=1  creeps_peak=4
G1 PASS · G2 PASS · G4 PASS · G3 implied (spawns=4, fuel path live)

# later authoritative dual-source (W7N3, 2 sources) — see JOB0-REPORT.md:
skill_rate≈1.42  H=1108  C=307  @1k
```

Sustained floor ≈ multi-cycle harvest after regen; old “BC ≤400 steps” guidance is **stale**.

## Wave 9 execution recipe

### A) BC on CPU (primary) — **superseded by Wave 11**

Historical recipe (did not land). **Run [`16-WAVE11-STANDING-ORDER.md`](16-WAVE11-STANDING-ORDER.md) A** instead.

```bash
export RL_NODE="$(mise exec node@24 -- which node)"
cd /home/marvin/Documents/repositories/xxscreeps

python3 -m samples.rl.agent.pretrain_bc \
  --num-envs 4 --steps 2000 --reset-every 500 \
  --device cpu --curriculum empty \
  --save samples/rl/runs/bc_scripted.pt \
  2>&1 | tee samples/rl/runs/bc_scripted_w9.log
```

- Multi-cycle teacher makes **2000** steps fine.
- `--reset-every 500` densifies spawn labels without killing regen cycles.
- Env/Node-bound; default device is already `cpu`.

Pass if: NLL decreases across epochs; file `samples/rl/runs/bc_scripted.pt` exists with `meta.kind=bc_scripted`.

### B) Optional: re-confirm Job0 windows (CPU, short)

```bash
python3 -m samples.rl.agent.eval_scripted --ticks 1000 \
  2>&1 | tee samples/rl/runs/eval_scripted_job0d.log
```

Want: G1–G4 PASS; `skill_last500` not collapsed to ~0.  
**Status:** job0d landed (skill_last500=0.782). Floor frozen.

### C) Explicit non-goals this wave

- PPO / GPU mlq jobs  
- HL-Gauss / entity attention / binary protocol rewrites  
- TI-BC inverse dynamics  
- Changing reward away from H+C  

## After Wave 9 (when GPU free)

```text
Job1a BC (done in W11) ∥ Job1b critic already on disk
  → Job2 PPO --resume bc_scripted.pt --critic-pretrain critic_pretrained.pt
```

See `07-TRAINING-PLAYBOOK.md`. **As of W11: BC still missing — run Wave 11 A.**

## Wave index (session)

| Wave | Focus | Result |
|------|--------|--------|
| W8 | TI critic `--skip-warmup-steps` fencepost | no off-by-one; `train_this = t >= N` |
| **W9** | **CPU BC + floor honesty** | **authorized; BC artifact did not land** |
| W10 | Commit prep + docs + `eval_expert` | `15-WAVE10-CPU.md` |
| W11 | Ship BC for real + freeze readiness | [`16-WAVE11-STANDING-ORDER.md`](16-WAVE11-STANDING-ORDER.md) |
