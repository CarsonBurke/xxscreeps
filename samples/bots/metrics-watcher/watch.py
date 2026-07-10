#!/usr/bin/env python3
"""
Apex metrics JSONL → TensorBoard (host-side only; not a bot dependency).

  python3 samples/bots/metrics-watcher/watch.py \\
      --jsonl samples/bots/apex/runs/latest/metrics.jsonl \\
      --logdir samples/bots/apex/runs/latest/tb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set


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


def emit_sample(writer_kind, writer, sample: Dict[str, Any], prev: Optional[Dict[str, Any]]) -> None:
    """Lean tag set — only what we actually use for bot health."""
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

    # Controller / GCL
    add_scalar(writer_kind, writer, "controller/control_points_total", sample.get("controlPoints"), t)
    add_scalar(writer_kind, writer, "controller/progress", sample.get("progress"), t)
    add_scalar(writer_kind, writer, "gcl/level", sample.get("gcl"), t)

    # CPU
    add_scalar(writer_kind, writer, "cpu/used", sample.get("cpu"), t)
    add_scalar(writer_kind, writer, "cpu/bucket", sample.get("bucket"), t)

    if prev is not None:
        add_scalar(writer_kind, writer, "energy/harvested_rate", rate(sample, prev, "harvested"), t)
        add_scalar(writer_kind, writer, "energy/build_rate", rate(sample, prev, "build"), t)
        add_scalar(writer_kind, writer, "energy/upgrade_rate", rate(sample, prev, "upgrade"), t)
        add_scalar(writer_kind, writer, "controller/control_points_rate", rate(sample, prev, "controlPoints"), t)


def process_line(
    line: str,
    writer_kind,
    writer,
    seen_seq: Set[int],
    prev_sample: Optional[Dict[str, Any]],
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
        if seq in seen_seq:
            return prev_sample
        seen_seq.add(seq)

    sample = row.get("sample")
    if sample is None and "ring" in row:
        ring = row.get("ring") or []
        sample = ring[-1] if ring else None
    if not isinstance(sample, dict):
        return prev_sample

    emit_sample(writer_kind, writer, sample, prev_sample)
    return sample


def main() -> int:
    ap = argparse.ArgumentParser(description="Apex metrics JSONL → TensorBoard")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--poll", type=float, default=0.5)
    args = ap.parse_args()

    writer_kind, writer = make_writer(args.logdir)
    print(f"writer={writer_kind} logdir={args.logdir}", file=sys.stderr)

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
                prev_sample = process_line(line, writer_kind, writer, seen_seq, prev_sample)
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
