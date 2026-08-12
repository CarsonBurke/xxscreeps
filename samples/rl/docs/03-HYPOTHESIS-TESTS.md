# Hypothesis Tests (code-verified, no training)

| ID | Claim | Verdict | Evidence |
|----|-------|---------|----------|
| H1 | step returns pre-tick obs | **WAS true; FIXED** | `stepRl` now: apply → tick → encode (post-tick) |
| H2 | skill rate inflated ×N | **CONFIRMED** | `train.py`: `(total_harvest+total_control)/T` sums all envs |
| H3 | all 4 heads always in logπ | **CONFIRMED** | `model.py:394` `lp_t+lp_d+lp_tg+lp_am` always |
| H4 | critic pretrain raw, no norm | **CONFIRMED** | `pretrain_critic.py` MSE on raw rewards; no RewardNormalizer |
| H5 | fixed spawn body only | **CONFIRMED** | `actions.mjs:178` `[W,C,M,M]` |
| H6 | no createConstructionSite | **CONFIRMED** | no matches under samples/rl |
| H7 | soft target masks | **CONFIRMED** | structures always selectable; WORK enables upgrade w/o energy |
| H8 | resume drops RMS/opts | **CONFIRMED** | ckpt saves opts; load only weights; no normalizer |
| H9 | time-limit = terminal | **CONFIRMED** | `done = step >= MAX_EPISODE`; GAE zeros bootstrap |
| H10 | “lagged obs still Markov” | **PARTIALLY WRONG** | Markov in theory with delayed observation, but agent never conditions on true s_{t+1}; actions applied to wrong state index in practice (see analysis) |

## H1 detailed mechanism

```
step(a_t):
  load world s_t
  applyActions(a_t)     # queue intents against s_t
  obs = encode(s_t)     # PRE-execution
  tick → s_{t+1}
  return obs=s_t, r(s_t→s_{t+1})

step(a_{t+1}):
  load world s_{t+1}
  applyActions(a_{t+1}) # a_{t+1} chosen from s_t !
```

So π(a_{t+1}|s_t) is applied to s_{t+1}. Permanent one-tick observation lag on every transition. For adjacency-critical harvest this is fatal at low sample counts.

## H10 resolution

A POMDP with constant 1-tick delay can still be Markov in the *belief* over (s_t, pending), but:
- the policy never sees fatigue/position after its own move until two steps later relative to intended control;
- watch logs show persistent illegal thrash consistent with acting on stale adjacency.

**Fix:** encode after tick; return post-tick obs. Actions still index into *previous* meta (session.meta), which is correct for “decide on last obs.”

## Skill math

With N envs, per-env mean rate ≈ total/(T·N). Logged skill uses total/T → **N× too large**. “skill=2” with N=8 can mean ~0.25 e/t average.
