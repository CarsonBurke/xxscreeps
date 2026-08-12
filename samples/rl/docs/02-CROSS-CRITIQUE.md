# Round-2 Cross-Critique Matrix

## Expert vs Expert (resolved)

| Conflict | Winner | Resolution |
|----------|--------|------------|
| E1 dense PBS vs E10 BC-first | **User: H+C only + BC∥critic** | No shaped r; bootstrap via BC+masks; critic pretrains in parallel |
| E2 arch rewrite vs E10 keep model | **E10 sequencing** | Steal E2 notes later; don't rewrite before harvest works |
| E3/E4 HL-Gauss now vs E10 later | **E10 / R2-meta** | HL-Gauss optional (`hlGauss: false` default); keep code path |
| E5 macros vs E5's own mask advice | **Masks first** | Macros P2; type-gated logπ + hard masks P0 |
| E7 body/sites ceiling vs bootstrap | **Both** | Ceiling is real; RCL1 loop still learnable without construction |
| E9 pre-tick S0 vs “Markov lag” | **E9** | Confirmed fatal for adjacency control |
| E1 skill ×N | **Confirmed** | Fixed to `/(T·N)` |

## Attacks that landed on our interim consensus

From R2-adversarial (`…2ca56182048`):

1. **Stacking all fixes at once = nonstationary nightmare** — we implement P0 only, stage the rest.
2. **HL-Gauss gates ≠ bot skill** — default off.
3. **TI-BC without env parity is cargo-cult** — **DEFERRED** (user 2026-07-17): no intent→AR inverse dynamics; BC = scripted only; TI = critic only.
4. **Shaping can inflate return while skill is zero** — keep product metrics H+C+RCL as truth.
5. **Do not delete post-tick / masks** — load-bearing. Reward stays pure H+C.

## Attacks that landed on individual experts

| Expert | Valid hit |
|--------|-----------|
| E10 BC-first | No BC substrate; obs timing mismatch; inexpressible TI |
| E2 entity attention | RCL1 has ~0 entities; premature capacity spend |
| E4 HL-Gauss | Units wrong → bins nonsense; MSE scale fix first |
| E5 macros | Second product; hides whether MDP is fixed |
| E1 dense PBS | Overscoped for P0; minimal spawn+deliver enough to start |

## Dependency order (anti-theater)

```
post-tick obs
  → type-gated logπ + hard masks + priors
    → r = H+C equal only
      → PARALLEL: BC policy (scripted) ∥ critic pretrain (same r; TI optional)
        → PPO hygiene (warmup, KL stop, resume)
          → body templates / sites / entity attn only as needed
```

If a proposal still works when you **delete** HL-Gauss and shaped rewards — and **fails** when you delete post-tick, masks, or BC — the dependency order is correct.
