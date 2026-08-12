# Screeps Optimal ML Bot — Continuous Analysis Session

| Field | Value |
|-------|-------|
| Start (UTC) | 2026-07-17T22:32:06Z |
| End target | start + 5h wall clock |
| Mode | **Latent only** — no GPU, no significant training CPU |
| Method | 10 expert agents → cross-critique → hypothesis tests → design → code (no train) |

## Team

| # | Role | ID (short) | Status |
|---|------|------------|--------|
| 1 | Reward / credit | `…941de82919d0` | done |
| 2 | Architecture | `…0d5866466a91` | done |
| 3 | VAPO / PPO | `…0d60ff6eccf1` | done |
| 4 | Critic / HL-Gauss | `…0d79d4adcf74` | done |
| 5 | Action / MARL | `…0d8aea32552a` | done |
| 6 | Observation | `…0d97a4d6af42` | done |
| 7 | Screeps economy | `…0da039dcd730` | done |
| 8 | Infrastructure | `…0db1e7868a59` | done |
| 9 | Red team (current) | `…0dcccdf661d3` | done |
| 10 | Red team (optimal) | `…0ddd3cfb8ad3` | done |

## Deliverables

- `01-EXPERT-CORPUS.md` — condensed expert findings
- `02-CROSS-CRITIQUE.md` — agents attacking each other
- `03-HYPOTHESIS-TESTS.md` — code-verified claims
- `04-CONSENSUS-DESIGN.md` — optimal system
- `05-KILL-CONDITIONS.md` — what must change before training
- Implementation under `samples/rl/` (no training runs)
- Wave standing orders: `14-WAVE9-STANDING-ORDER.md` (CPU BC residual), `15-WAVE10-CPU.md` (commit prep + eval_expert; **user OK**)
