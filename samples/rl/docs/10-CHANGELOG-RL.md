# samples/rl changelog (latent redesign session)

## Breaking

- Observation is **post-tick** (was pre-tick). Old trajectories/BC labels misaligned.
- Intent enum gains `createConstructionSite` (index 20); actor `type_head` out_features 21.
- Reward = **only** equal-weight energy harvested + control points (shaped terms removed as hackable).
- `skill_rate` now per-env mean `/(T·N)`.
- Spawn `amount` head selects **body template**, not energy bin, for structure actors.
- Entropy is mean over live actors (not sum of all heads).

## Added

- `docs/` design corpus (10 experts + rounds 2–3) + wave standing orders (`14`–`16`) + `COMMIT-PREP.md`
- `hl_gauss.py`, optional categorical critic
- `scripted_baseline.mjs`, `step_scripted`, `pretrain_bc.py`, `eval_scripted.py`
- `eval_expert.py` — TI skill floor under r=H+C (post-warmup metric; not BC)
- `test_latent_unit.py` (CPU)
- Body templates in schema
- Truncation-aware GAE + terminal_obs bootstrap
- Value warmup, KL early-stop, trunk-only critic pretrain load
- Checkpoint: reward_normalizer + opts on resume

## Decisions

- **2026-07-17 — Defer TI-BC inverse dynamics.** Policy BC remains scripted-only; TI is critic/obs only. No intent→AR inverse dynamics work until scripted BC + PPO prove the stack. See `04-CONSENSUS-DESIGN.md` P3, `11-REWARD-AND-PRETRAIN.md`.
- **2026-07-17 — Wave 9 (GPU blocked):** CPU BC authorized; artifact still missing → residual. See `14-WAVE9-STANDING-ORDER.md`.
- **2026-07-17 — Wave 10 (GPU busy, user OK):** commit prep + docs + `eval_expert` / Job0d; no PPO. BC remains residual. See `15-WAVE10-CPU.md`, `15-WAVE10-STANDING-ORDER.md`.
- **2026-07-17 — Job0 dual-source floor.** Authoritative scripted floor on W7N3 (2 sources) @1k: **H≈1108 C≈307 skill≈1.42 e/t**. Prior job0c/job0d ~1.40. See `JOB0-REPORT.md`.
- **2026-07-17 — Wave 11 (GPU blocked, user OK):** ship `bc_scripted.pt` on CPU (non-optional); dual-source teacher frozen as confirmed good; freeze readiness after BC. See `16-WAVE11-STANDING-ORDER.md`.

## Fixed

- Double-`player()` on scripted path
- Dummy-head logπ
- Soft harvest/upgrade without energy/range
- Target table wall flood
- Resume amnesia (partial)
- Value clip max() freezing scale repair
