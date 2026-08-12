"""Efficient host-side run metrics with polars (or JSONL fallback).

Hot path: append small row dicts to a buffer; flush every N updates to parquet/CSV.
Never log full observations. Never call from the act/step inner loop.

  from samples.rl.agent.metrics_log import MetricsLog
  log = MetricsLog(path="runs/metrics.parquet")
  log.add(step=global_step, skill=0.5, mean_r=0.01, ...)
  log.flush()
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricsLog:
    def __init__(self, path: str | Path, *, flush_every: int = 20):
        self.path = Path(path)
        self.flush_every = int(flush_every)
        self._rows: list[dict[str, Any]] = []
        self._n = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._use_polars = False
        try:
            import polars as pl  # noqa: F401

            self._use_polars = True
        except ImportError:
            pass

    def add(self, **row: Any) -> None:
        # Coerce tensors / numpy scalars so polars gets plain Python types
        clean: dict[str, Any] = {}
        for k, v in row.items():
            if hasattr(v, "item") and callable(v.item):
                try:
                    clean[k] = v.item()
                    continue
                except Exception:
                    pass
            clean[k] = v
        self._rows.append(clean)
        self._n += 1
        if self._n % self.flush_every == 0:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        rows, self._rows = self._rows, []
        # The training default is append-only JSONL: a few thousand scalar rows do
        # not justify rereading and rewriting an ever-growing Parquet file every
        # five updates. Parquet remains available for explicit offline use.
        if self.path.suffix == ".jsonl":
            with self.path.open("a", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
            return
        if self._use_polars:
            import polars as pl

            df = pl.from_dicts(rows)
            if self.path.suffix == ".parquet":
                if self.path.exists():
                    old = pl.read_parquet(self.path)
                    df = pl.concat([old, df], how="diagonal_relaxed")
                df.write_parquet(self.path)
            elif self.path.suffix == ".csv":
                # True append: write header only on first create
                header = not self.path.exists()
                with self.path.open("a", encoding="utf-8") as f:
                    df.write_csv(f, include_header=header)
            else:
                # default parquet even without suffix if polars available
                p = self.path if self.path.suffix else self.path.with_suffix(".parquet")
                if p.exists():
                    old = pl.read_parquet(p)
                    df = pl.concat([old, df], how="diagonal_relaxed")
                df.write_parquet(p)
            return
        # Fallback: JSONL (append-safe, no polars needed)
        out = self.path if self.path.suffix == ".jsonl" else self.path.with_suffix(".jsonl")
        with out.open("a", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")

    def close(self) -> None:
        self.flush()
