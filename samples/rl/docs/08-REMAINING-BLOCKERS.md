# Remaining blockers (after same-expert mandate)

Design P0s for post-tick / H+C / masks largely landed. **Ship-readiness** and **same-expert joint pretrain** are the open gates. Do not treat old “no open P0” checklists as queue permission.

## R1 — Mid-chunk terminal bootstrap (partially fixed)

`collect_rollout` and joint pretrain splice `V(terminal_observation)` into GAE/MC. Watch shape edge cases on first real train.

## R2 — Cold start without joint pretrain — **current target**

Sparse multi-stage economy remains hard from random π under **r=H+C only** (no shaping). **Default: same-expert joint BC∥critic → PPO**, not pure PPO and not split teachers.

- Joint path: `pretrain_joint.py` (scripted v1)  
- Artifact: **`joint_pretrain.pt` not yet produced**  
- Job0 floor: **PASS** (dual-source W7N3, skill≈1.42 e/t)  
- Split BC + TI critic: **rejected** (user mandate)

Scripted teacher **builds existing sites** at RCL≥2 but does **not** place extensions. Fine for RCL1 loop; RCL2 imitation is a later ceiling.

## R3 — Entropy vs joint policy

Mitigated: entropy is **mean over live actors / slots**, `entropyCoef=0.005`. Re-check after multi-creep.

## R4 — Intent-agnostic target table — **DONE**

`targetSelectMask` is `[A,S,nIntent,maxTargets]`; model gathers by sampled type.

## R5 — Env ceiling

Body templates + `createConstructionSite` in AR. Teacher does not demo site placement yet.

## R6 — Sample throughput

Binary obs (`RL_OBS_FMT=bin`) default; still env-bound.

## R7 — Readiness (open)

- Contract tests beyond thin unit suite  
- IPC freeze to bin-only for train  
- Packaging / entrypoints  
- PPO residuals (episode λ, compile strides)  
- **No mlq until G0–G5** in playbook / plan  

## Explicitly not a near-term blocker — TI as joint expert

TI cannot emit AR labels today. Joint expert v1 = **scripted**. TI→AR is multi-week and not required for same-expert mandate. Keep `eval_expert` only.

## Recommended next order

1. Finish readiness: contract tests, joint path harden, docs (no mlq).  
2. When authorized: Job1 `pretrain_joint` → `joint_pretrain.pt`.  
3. When authorized: Job2 PPO `--resume joint_pretrain.pt`.  
4. Only then architecture rewrites / teacher site placement / optional TI-AR.  
