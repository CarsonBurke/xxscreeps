# Job0 report — scripted floor (W7N3, dual source)

## Latest (authoritative): dual-source @1k ticks, empty curriculum

Room: **W7N3 (dual source)**. Multi-cycle scripted teacher (understaffed-first).

```text
skill_rate≈1.42  H=1108  C=307  @1k ticks
```

| Gate | Result |
|------|--------|
| G1 first creep <200 | **PASS** |
| G2 harvestDelta > 0 | **PASS** (H=1108) |
| G3 spawn / deliver path | **PASS** |
| G4 controlDelta > 0 | **PASS** (C=307) |

Multi-cycle teacher: harvest continues after source regen; skill does not collapse to zero after first dry-down.  
Recommended sustained metric: `skill_last500` from `eval_scripted` (fields already printed).  
Floor is **dual-source** — do not compare single-source rooms 1:1 to this number.

### Dual-source teacher (W11 freeze)

- Max 1 dedicated harvester per live source → 2 concurrent harvesters on W7N3.  
- Understaffed-first assignment (claimed asc, then distance) — not nearest-greed spawn bias.  
- Sticky adjacency; idle when both dry until regen is correct.  
- **Confirmed good** (both sources drain with ≥2 fillers; multi-cycle H keeps rising). Do not redesign for W11.

## Prior reconfirms (same stack)

### job0d @1k (windowed metrics)

Log: `runs/eval_scripted_job0d.log`

```text
skill_rate=1.409  H=1104  C=305  spawnSuccess=4
skill_first200=2.530  skill_last500=0.782
first_creep_tick=1  creeps_peak=4
G1–G4 PASS
```

### job0c @1k (first multi-cycle baseline)

Log: `runs/eval_scripted_job0c.log`

```text
skill_rate≈1.395  H=1106  C=289  spawnSuccess=4
first_creep_tick=1  creeps_peak=4
G1–G4 PASS
```

## Earlier runs (superseded)

### job0 (pre multi-cycle / weak upgrade)

```text
skill_rate≈0.83 @1k  H=828  C=4  — G1/G2/G4 PASS but thin control
```

### job0 pre-upgrade-kind-fix

```text
H≈1840 @2k  C=0  — G4 FAIL; upgrade targets filtered out controller
```

## Verdict

**Job0 floor established** on dual-source W7N3 @1k: **H≈1108 C≈307 skill≈1.42 e/t**.  
Full G1–G4 pass with multi-cycle teacher. Round-to-round scatter is small (~1.40–1.42).  
**Frozen for Wave 11** — do not re-eval before BC.  
Next gate: **same-expert joint pretrain** (`joint_pretrain.pt` via `pretrain_joint`) — see [`16-JOINT-PRETRAIN-STANDING.md`](16-JOINT-PRETRAIN-STANDING.md). Split `bc_scripted.pt` is legacy.
