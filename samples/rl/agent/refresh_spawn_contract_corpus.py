#!/usr/bin/env python3
"""Derive a corpus with refreshed one-step spawn-composition contracts.

The multi-gigabyte lifecycle and TI tensors remain semantically unchanged. This
tool reuses them in memory, replaces only the six explicitly prefixed contract
rows, and writes a new immutable content-addressed corpus. It is intended for a
spawn-scenario semantic correction that does not change observation/action ABI
or the long teacher trajectories.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .constants import INTENT_TYPES
from .pretrain_corpus import (
    _collection_source_signature,
    _merge_stage_metrics,
    _spawn_row,
    assemble_corpus,
    load_corpus,
    save_corpus,
)
from .pretrain_joint import SPAWN_CURRICULA, _collect_spawn_contract_replay


def refresh_spawn_contracts(
    corpus: dict,
    *,
    node: str | None = None,
) -> dict:
    """Return a validated derived corpus with current contract rows."""
    current_source = _collection_source_signature()
    recorded_source = corpus["meta"].get("collection_source_sha256")
    if recorded_source != current_source:
        raise ValueError(
            "base corpus collection semantics differ from current source; "
            "recollect the full corpus instead of mixing provenance"
        )
    base_id = str(corpus["integrity"]["corpus_sha256"])
    old_rows = list(corpus["data"]["spawn_replay"])
    contract_count = len(SPAWN_CURRICULA)
    old_metrics = corpus["data"]["teacher"].get(
        "spawn_contract_by_curriculum", {}
    )
    if set(old_metrics) != set(SPAWN_CURRICULA):
        raise ValueError("base corpus does not declare the expected spawn contracts")
    if len(old_rows) < contract_count:
        raise ValueError("base corpus spawn replay is shorter than contract prefix")
    spawn_type = INTENT_TYPES.index("spawnCreep")
    for index, row in enumerate(old_rows[:contract_count]):
        actor = int(row["actor_index"])
        if int(row["action"]["types"][0, actor, 0]) != spawn_type:
            raise ValueError(f"base spawn contract prefix row {index} is not spawnCreep")

    contracts: list[tuple] = []
    contract_metrics: dict[str, dict] = {}
    meta = corpus["meta"]
    _collect_spawn_contract_replay(
        contracts,
        contract_metrics,
        node=node if node is not None else meta.get("node"),
        room=str(meta["room"]),
        bot_dir=str(meta["ti_bot_dir"]),
        seed=int(meta["seed"]),
    )
    if len(contracts) != contract_count or set(contract_metrics) != set(SPAWN_CURRICULA):
        raise RuntimeError("current spawn contract collector returned incomplete coverage")

    teacher = dict(corpus["data"]["teacher"])
    teacher["spawn_contract_by_curriculum"] = contract_metrics
    teacher["train_by_curriculum_with_spawn_contracts"] = _merge_stage_metrics(
        teacher["train_by_curriculum"], contract_metrics,
    )
    data = dict(corpus["data"])
    data["teacher"] = teacher
    data["spawn_replay"] = [
        *(_spawn_row(row) for row in contracts),
        *old_rows[contract_count:],
    ]
    derived_meta = dict(meta)
    derived_meta["spawn_contract_refresh"] = {
        "base_corpus_sha256": base_id,
        "contract_count": contract_count,
        "seed": int(meta["seed"]),
        "collection_source_sha256": current_source,
        "policy": "replace_declared_contract_prefix_preserve_all_other_rows",
    }
    return assemble_corpus(derived_meta, data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node", default=None)
    args = parser.parse_args()
    base = load_corpus(args.base)
    derived = refresh_spawn_contracts(base, node=args.node)
    path = save_corpus(derived, args.output)
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
