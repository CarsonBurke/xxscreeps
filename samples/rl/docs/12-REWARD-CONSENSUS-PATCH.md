# Consensus patch: H+C-only reward + BC∥critic

**Authoritative user mandate** overrides earlier E1 dense-shaping prescriptions.

## Reward

```text
r = 1.0 · Δenergy_harvested + 1.0 · Δcontrol_points
```

Everything else (spawn, transfer, build, sites, extinction, PBS) is **out of the scalar reward**. May remain in `info` for TensorBoard / gates.

## Bootstrap without shaping

| Mechanism | Role |
|-----------|------|
| Masks | Illegal actions not sampled |
| Priors | Bias toward spawn/harvest/transfer early |
| **BC (scripted)** | Puts mass on full pipeline including place/build |
| Critic pretrain | Value scale on same r=H+C |
| PPO | Improves H+C under BC init |

## Parallel pretrain

Run **policy BC** and **critic pretrain** concurrently (or back-to-back), then PPO merge. Do not wait for critic to finish before starting BC or vice versa unless single-GPU.

## Rejected from E1 (explicit)

- spawnSuccess bonus  
- transfer-to-spawn premium  
- siteCreated / RCL bonuses in r  
- extinction per-tick penalties  
- PBS crew/pipe potentials in r  

## Code contract

- `schema.json` reward keys: only `energyHarvested`, `controlPoints`  
- `measureReward()` scores only those two  
- `skill_rate_et` with unit weights ≡ mean raw reward  
