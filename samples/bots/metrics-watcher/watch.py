#!/usr/bin/env python3
"""
Apex metrics JSONL → TensorBoard (host-side only; not a bot dependency).

  python3 samples/bots/metrics-watcher/watch.py \\
      --jsonl samples/bots/apex/runs/<stamp>/metrics.jsonl \\
      --logdir samples/bots/apex/runs/<stamp>/tb

  # Compare all BMs (bench maintains runs/tb/<stamp> → ../<stamp>/tb):
  #   tensorboard --logdir samples/bots/apex/runs/tb

Wire short keys (MetricKey enum values) are NEVER hardcoded here.
They are loaded from apex-v3/dist/memoryKeys.js (MetricKey / SegmentKey).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set


_HERE = Path(__file__).resolve().parent
_MEMORY_KEYS_JS = _HERE.parent / "apex-v3" / "dist" / "memoryKeys.js"


def load_metric_wire_map() -> Dict[str, str]:
    """wire short → long name from MetricKey enum only."""
    if not _MEMORY_KEYS_JS.is_file():
        raise SystemExit(
            f"missing {_MEMORY_KEYS_JS} — build apex-v3 (npx tsc) so MetricKey enum is available"
        )
    # Node: build map from runtime enum (source of truth)
    script = f"""
const m = require({json.dumps(str(_MEMORY_KEYS_JS))});
const MetricKey = m.MetricKey;
const out = {{}};
for (const longName of Object.keys(MetricKey)) {{
  const wire = MetricKey[longName];
  if (typeof wire === 'string') out[wire] = longName;
}}
process.stdout.write(JSON.stringify(out));
"""
    r = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(r.stdout)


def make_writer(logdir: Path):
    logdir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        return ("torch", SummaryWriter(log_dir=str(logdir)))
    except Exception:
        pass
    try:
        from tensorboardX import SummaryWriter  # type: ignore

        return ("tensorboardX", SummaryWriter(log_dir=str(logdir)))
    except Exception:
        pass
    csv_path = logdir / "scalars.csv"
    if not csv_path.exists():
        csv_path.write_text("step,tag,value\n", encoding="utf-8")
    return ("csv", csv_path)


def add_scalar(writer_kind, writer, tag: str, value: Any, step: int) -> None:
    if value is None:
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    if writer_kind in ("torch", "tensorboardX"):
        writer.add_scalar(tag, v, global_step=step)
    else:
        with open(writer, "a", encoding="utf-8") as f:
            f.write(f"{step},{tag},{v}\n")


def rate(sample: Dict[str, Any], prev: Dict[str, Any], key: str) -> Optional[float]:
    if key not in sample or key not in prev:
        return None
    try:
        dt = max(1, int(sample.get("t", 0)) - int(prev.get("t", 0)))
        return (float(sample[key]) - float(prev[key])) / dt
    except (TypeError, ValueError):
        return None


def normalize_sample(sample: Dict[str, Any], wire_to_long: Dict[str, str]) -> Dict[str, Any]:
    """Accept dense (MetricKey wire) or long-key samples; expand via enum map."""
    out: Dict[str, Any] = dict(sample)
    for wire, long in wire_to_long.items():
        if wire in sample and long not in out:
            out[long] = sample[wire]
    if "t" in sample:
        out["t"] = sample["t"]
    return out


def emit_sample(
    writer_kind,
    writer,
    sample: Dict[str, Any],
    prev: Optional[Dict[str, Any]],
    wire_to_long: Dict[str, str],
) -> None:
    """Lean tag set — only what we actually use for bot health."""
    sample = normalize_sample(sample, wire_to_long)
    prev = normalize_sample(prev, wire_to_long) if prev else None
    t = int(sample.get("t", 0))

    # Empire
    add_scalar(writer_kind, writer, "empire/rcl_max", sample.get("rclMax"), t)
    add_scalar(writer_kind, writer, "empire/colonies", sample.get("colonies"), t)
    add_scalar(writer_kind, writer, "empire/creeps", sample.get("creeps"), t)
    add_scalar(writer_kind, writer, "empire/sites", sample.get("sites"), t)
    add_scalar(writer_kind, writer, "empire/stored_energy", sample.get("storedEnergy"), t)

    # Energy (totals + rates)
    add_scalar(writer_kind, writer, "energy/harvested_total", sample.get("harvested"), t)
    add_scalar(writer_kind, writer, "energy/build_total", sample.get("build"), t)
    add_scalar(writer_kind, writer, "energy/upgrade_total", sample.get("upgrade"), t)
    add_scalar(writer_kind, writer, "energy/creep_total", sample.get("creepEnergy"), t)

    # Per-tick rates — prefer bot windowed rates; else derive from totals
    hr = sample.get("harvestRate")
    if hr is None and prev is not None:
        hr = rate(sample, prev, "harvested")
    br = sample.get("buildRate")
    if br is None and prev is not None:
        br = rate(sample, prev, "build")
    ur = sample.get("upgradeRate")
    if ur is None and prev is not None:
        ur = rate(sample, prev, "upgrade")
    cr = sample.get("controlRate")
    if cr is None and prev is not None:
        cr = rate(sample, prev, "controlPoints")

    # Claimed vs remote harvest split (room ownership at harvest event)
    hcr = sample.get("claimedHarvestRate")
    if hcr is None and prev is not None:
        hcr = rate(sample, prev, "harvestedClaimed")
    hor = sample.get("remoteHarvestRate")
    if hor is None and prev is not None:
        hor = rate(sample, prev, "harvestedRemote")

    add_scalar(writer_kind, writer, "energy/harvested_per_tick", hr, t)
    add_scalar(writer_kind, writer, "energy/build_per_tick", br, t)
    add_scalar(writer_kind, writer, "energy/upgrade_per_tick", ur, t)
    add_scalar(writer_kind, writer, "controller/control_points_per_tick", cr, t)

    # claimed/ — energy harvested in owned rooms
    add_scalar(writer_kind, writer, "claimed/harvested_total", sample.get("harvestedClaimed"), t)
    add_scalar(writer_kind, writer, "claimed/harvested_per_tick", hcr, t)
    # remote/ — energy harvested in remotes (non-owned)
    add_scalar(writer_kind, writer, "remote/harvested_total", sample.get("harvestedRemote"), t)
    add_scalar(writer_kind, writer, "remote/harvested_per_tick", hor, t)
    target_remote = 40.0
    if hor is not None:
        add_scalar(writer_kind, writer, "remote/vs_target_40", float(hor) / target_remote, t)

    # Ceiling vs realized (bar is maxHarvestEt, not a fixed number)
    add_scalar(writer_kind, writer, "energy/max_harvest_et", sample.get("maxHarvestEt"), t)
    add_scalar(writer_kind, writer, "energy/max_harvest_physics", sample.get("maxHarvestPhysics"), t)
    mh = sample.get("maxHarvestEt")
    if hr is not None and mh is not None and float(mh) > 0:
        add_scalar(writer_kind, writer, "energy/harvest_vs_ceiling", float(hr) / float(mh), t)
    # Aliases (legacy tag names)
    add_scalar(writer_kind, writer, "energy/harvested_rate", hr, t)
    add_scalar(writer_kind, writer, "controller/control_points_rate", cr, t)

    # Controller / GCL
    add_scalar(writer_kind, writer, "controller/control_points_total", sample.get("controlPoints"), t)
    add_scalar(writer_kind, writer, "controller/progress", sample.get("progress"), t)
    # gcl is continuous: integer level + progress/progressTotal (e.g. 1.37)
    add_scalar(writer_kind, writer, "gcl/level", sample.get("gcl"), t)
    gcl_v = sample.get("gcl")
    if gcl_v is not None:
        try:
            gf = float(gcl_v)
            add_scalar(writer_kind, writer, "gcl/level_floor", int(gf), t)
            add_scalar(writer_kind, writer, "gcl/progress_frac", gf - int(gf), t)
        except (TypeError, ValueError):
            pass

    # CPU
    add_scalar(writer_kind, writer, "cpu/used", sample.get("cpu"), t)
    add_scalar(writer_kind, writer, "cpu/bucket", sample.get("bucket"), t)

    # The International Memory.stats (RoomStatsKeys shorts + empire)
    # eih/eou/eob are already tick-averaged rates inside TI.
    add_scalar(writer_kind, writer, "ti/eih", sample.get("eih"), t)
    add_scalar(writer_kind, writer, "ti/eou", sample.get("eou"), t)
    add_scalar(writer_kind, writer, "ti/eob", sample.get("eob"), t)
    add_scalar(writer_kind, writer, "ti/reih", sample.get("reih"), t)
    add_scalar(writer_kind, writer, "ti/eoro", sample.get("eoro"), t)
    add_scalar(writer_kind, writer, "ti/eorwr", sample.get("eorwr"), t)
    add_scalar(writer_kind, writer, "ti/eosp", sample.get("eosp"), t)
    add_scalar(writer_kind, writer, "ti/es", sample.get("es"), t)
    add_scalar(writer_kind, writer, "ti/su", sample.get("su"), t)
    add_scalar(writer_kind, writer, "ti/cl", sample.get("cl"), t)
    add_scalar(writer_kind, writer, "ti/tick_length_ms", sample.get("tickLength"), t)
    add_scalar(writer_kind, writer, "ti/heap_usage", sample.get("heapUsage"), t)
    add_scalar(writer_kind, writer, "ti/memory_usage", sample.get("memoryUsage"), t)
    rooms = sample.get("rooms")
    if isinstance(rooms, dict):
        for room_name, rs in rooms.items():
            if not isinstance(rs, dict):
                continue
            prefix = f"ti/room/{room_name}"
            for short in ("eih", "eou", "eob", "reih", "es", "su", "cl", "cc", "cpu"):
                add_scalar(writer_kind, writer, f"{prefix}/{short}", rs.get(short), t)


def process_line(
    line: str,
    writer_kind,
    writer,
    seen_seq: Set[int],
    prev_sample: Optional[Dict[str, Any]],
    wire_to_long: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return prev_sample
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return prev_sample

    seq = row.get("seq")
    if seq is not None:
        # seq may be str (ti-123 / harness-123) or int
        if seq in seen_seq:
            return prev_sample
        seen_seq.add(seq)

    # Envelope: sample (already expanded long keys) | dense (wire) | ring[-1]
    sample = row.get("sample") or row.get("d") or row.get("dense")
    if sample is None and "ring" in row:
        ring = row.get("ring") or []
        sample = ring[-1] if ring else None
    if not isinstance(sample, dict):
        return prev_sample

    sample = normalize_sample(sample, wire_to_long)
    emit_sample(writer_kind, writer, sample, prev_sample, wire_to_long)
    return sample


def main() -> int:
    ap = argparse.ArgumentParser(description="Apex metrics JSONL → TensorBoard")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--poll", type=float, default=0.5)
    args = ap.parse_args()

    wire_to_long = load_metric_wire_map()
    writer_kind, writer = make_writer(args.logdir)
    print(f"writer={writer_kind} logdir={args.logdir} metricKeys={len(wire_to_long)}", file=sys.stderr)

    seen_seq: Set[int] = set()
    prev_sample: Optional[Dict[str, Any]] = None
    offset = 0

    def drain() -> None:
        nonlocal offset, prev_sample
        if not args.jsonl.exists():
            return
        with open(args.jsonl, "r", encoding="utf-8") as f:
            f.seek(offset)
            for line in f:
                prev_sample = process_line(
                    line, writer_kind, writer, seen_seq, prev_sample, wire_to_long
                )
            offset = f.tell()
        if writer_kind in ("torch", "tensorboardX"):
            writer.flush()

    if args.follow:
        try:
            while True:
                drain()
                time.sleep(args.poll)
        except KeyboardInterrupt:
            pass
    else:
        if not args.jsonl.exists():
            print(f"missing {args.jsonl}", file=sys.stderr)
            return 1
        drain()

    if writer_kind in ("torch", "tensorboardX"):
        writer.close()
    print(f"done samples≈{len(seen_seq)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
