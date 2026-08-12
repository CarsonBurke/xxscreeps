# Standing order — same-expert joint pretrain (schema v1, superseded)

Superseded by [`17-SCALABLE-ENTITY-MAPPO.md`](./17-SCALABLE-ENTITY-MAPPO.md). Schema-v1 checkpoints and recipes are incompatible with the entity-MAPPO architecture.

**Supersedes** Wave 9/10 “ship `bc_scripted.pt` + merge TI/scripted critic” for Job1.

## Mandate

1. BC and critic pretrain **at the same time on the same expert**.  
2. **Do not queue** pretrain until stack readiness (tests + joint path + user auth).

## Do now

1. ~~Harden joint path~~ (`pretrain_joint`, `step_scripted`, legal BC NLL, teacher labels).  
2. ~~Contract tests~~ — `test_latent_unit` 17/17 (incl. joint chunk + finfo.min NLL).  
3. ~~Docs~~ same-expert mandate in 07/11/README.  
4. ~~Quarantine poison `policy.pt`~~ — hard refuse unmarked PPO resume.  
5. Remaining polish (IPC freeze, packaging, headful train lock) optional before first Job1.  
6. **User authorizes** first CPU `pretrain_joint` (then mlq only if GPU).  

## Do not

- `mlq submit` for BC-only, TI-critic-only, or joint until authorized.  
- Split scripted BC + TI critic.  
- TI→AR inverse dynamics on the critical path.  
- Reward shaping.

## Artifact

| Path | Role |
|------|------|
| `runs/joint_pretrain.pt` | Job1 output (actor + critic + meta) |
| `runs/bc_scripted.pt` | **Legacy** split BC |
| `runs/critic_pretrained.pt` | **Legacy** (not same-expert peer) |

## Recipe (CPU, when running Job1 is authorized)

See `07-TRAINING-PLAYBOOK.md` Job 1.
