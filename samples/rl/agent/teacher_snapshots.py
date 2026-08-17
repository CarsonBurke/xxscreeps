#!/usr/bin/env python3
"""Collect immutable, event-stratified teacher start states for the reservoir.

PPO rollouts that always begin at tick 0 only ever train on the first few
thousand ticks of a 20,000-tick episode, so the late-game behavior taught by
behavior cloning decays. This collector runs full teacher lifecycles once and
writes the restorable world states the trainer's teacher lane starts from::

    RL_NODE="$(mise exec node@24 -- which node)" \\
    python3 -m samples.rl.agent.teacher_snapshots \\
        --teacher ti --num-envs 4 --steps 20000 \\
        --curriculum empty,seed_outpost --per-stratum 8 --min-gap 64

The result is a content-addressed directory::

    <output-root>/<sha256>/
      manifest.json
      <curriculum>/<event>/<phase>/teacher/t<tick>_e<env>_<seq>.xsnp

The last stdout line is that directory; paste it into the trainer's
``--teacher-start-states`` flag. The set is immutable: a re-collection with
identical content resolves to the identical directory, and the reservoir
references these files in place instead of owning or evicting them.

Two admission rules keep the set usable:

* stratification identical to the trainer's (``primary_event`` over the
  environment's event tags plus the periodic background tag, ``phase_for_tick``
  over the step index), so coverage is measurable per event and per phase;
* rejection of any state the learner's observation cannot represent. A late TI
  world can exceed the room/actor/target caps, and restoring into a truncated
  observation would train the policy on a world it cannot see.

Evaluation and qualification stay on fresh 20,000-tick worlds; these snapshots
are a training-start mechanism only.
"""
from __future__ import annotations

import argparse
import errno
import filecmp
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_RL_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY = _RL_ROOT.parents[1]
for _search_path in (_REPOSITORY, _RL_ROOT):
    _entry = str(_search_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

try:
    from samples.rl.agent.artifacts import source_signature
    from samples.rl.agent.constants import SCHEMA_SHA256
    from samples.rl.agent.env_client import ScreepsEnv
    from samples.rl.agent.state_reservoir import (
        DEFAULT_PHASE_BOUNDS, OUTCOME_TEACHER, PERIODIC_EVENT,
        RESERVOIR_FORMAT_VERSION, StartStateController, phase_for_tick,
        primary_event,
    )
    from samples.rl.agent.vec_env import configure_host_threads
except ImportError:  # direct `python3 samples/rl/agent/teacher_snapshots.py`
    from agent.artifacts import source_signature
    from agent.constants import SCHEMA_SHA256
    from agent.env_client import ScreepsEnv
    from agent.state_reservoir import (
        DEFAULT_PHASE_BOUNDS, OUTCOME_TEACHER, PERIODIC_EVENT,
        RESERVOIR_FORMAT_VERSION, StartStateController, phase_for_tick,
        primary_event,
    )
    from agent.vec_env import configure_host_threads

TEACHERS = ("ti", "scripted")
MANIFEST_NAME = "manifest.json"
MANIFEST_KIND = "teacher_start_states"
STAGING_PREFIX = ".staging-"

DEFAULT_OUTPUT = _RL_ROOT / "runs" / "teacher-start-states"
DEFAULT_TI_BOT_DIR = (_REPOSITORY.parent / "The-International-Open-Source" / "dist")
DEFAULT_CURRICULUM = "empty,seed_outpost"

# A curriculum name becomes a directory inside the published set, so it must be
# a plain scenario name and never a traversal.
CURRICULUM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Overflow flags mean the encoder truncated the world: more rooms, actors, or
# targets exist than the observation can carry. Such a state is not a valid
# learner start state no matter how interesting the teacher made it. Every boot,
# restore, and step reply reports them in `info.globals`.
OVERFLOW_KEYS = ("roomOverflow", "actorOverflow", "targetOverflow")


class CollectorError(RuntimeError):
    """A collection failure a human must act on (never a silent partial set)."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CollectorConfig:
    teacher: str = "ti"
    num_envs: int = 4
    steps: int = 20_000
    max_episode: int = 20_000
    curricula: tuple[str, ...] = ("empty", "seed_outpost")
    seed: int = 201
    per_stratum: int = 8
    min_gap: int = 64
    output: Path = DEFAULT_OUTPUT
    node: str | None = None
    room: str = "W7N3"
    bot_dir: str | None = None
    progress_every: int = 500

    def __post_init__(self) -> None:
        if self.teacher not in TEACHERS:
            raise CollectorError(f"unknown teacher {self.teacher!r}; use one of {TEACHERS}")
        if self.num_envs < 1:
            raise CollectorError("--num-envs must be at least 1")
        if self.steps < 1:
            raise CollectorError("--steps must be at least 1")
        if self.per_stratum < 1:
            raise CollectorError("--per-stratum must be at least 1")
        if self.min_gap < 1:
            raise CollectorError("--min-gap must be at least 1")
        if not self.curricula:
            raise CollectorError("--curriculum must name at least one curriculum")
        malformed = [name for name in self.curricula if not CURRICULUM_NAME.match(name)]
        if malformed:
            raise CollectorError(
                f"curriculum names {malformed} are not plain scenario names; they "
                "become directories inside the published set"
            )
        if self.steps > self.max_episode:
            raise CollectorError(
                f"--steps {self.steps} exceeds --max-episode {self.max_episode}; "
                "the episode would end before collection finishes"
            )
        if self.num_envs < len(self.curricula):
            # Every named curriculum must actually be driven, otherwise the run
            # would spend hours only to fail the zero-record check at the end.
            raise CollectorError(
                f"--num-envs {self.num_envs} cannot cover {len(self.curricula)} curricula "
                f"{list(self.curricula)}; raise --num-envs or shorten --curriculum"
            )

    @property
    def expert(self) -> bool:
        return self.teacher == "ti"

    @property
    def resolved_bot_dir(self) -> Path:
        return Path(
            self.bot_dir or os.environ.get("RL_EXPERT_BOT") or DEFAULT_TI_BOT_DIR
        ).resolve()

    def env_curriculum(self, index: int) -> str:
        return self.curricula[index % len(self.curricula)]

    def env_seed(self, index: int) -> int:
        return int(self.seed) + int(index)

    def assigned_curricula(self) -> tuple[str, ...]:
        """Curricula actually driven by at least one environment, in order."""
        seen: list[str] = []
        for index in range(self.num_envs):
            name = self.env_curriculum(index)
            if name not in seen:
                seen.append(name)
        return tuple(seen)

    def seeds(self) -> tuple[int, ...]:
        return tuple(self.env_seed(index) for index in range(self.num_envs))


# ---------------------------------------------------------------------------
# admission
# ---------------------------------------------------------------------------
def overflow_fields(globals_payload: Any) -> tuple[str, ...]:
    """Names of the observation caps this globals payload reports overflowed."""
    if not isinstance(globals_payload, Mapping):
        return ()
    return tuple(key for key in OVERFLOW_KEYS if globals_payload.get(key))


@dataclass(frozen=True)
class Candidate:
    """An admitted start state and the relative file it will be written to."""

    env: int
    curriculum: str
    event: str
    phase: str
    step: int
    tick: int
    tags: tuple[str, ...]
    relative_path: str


class StratumBudget:
    """Thread-safe admission: per-stratum cap plus a per-environment tick gap.

    The cap is keyed by `(curriculum, event, phase)` and shared across
    environments so two envs on the same curriculum cannot both overfill one
    stratum; the gap is per environment so envs never throttle each other.
    """

    def __init__(
        self,
        *,
        per_stratum: int,
        min_gap: int,
        phase_bounds: Sequence[tuple[str, int]] = DEFAULT_PHASE_BOUNDS,
    ) -> None:
        if per_stratum < 1:
            raise CollectorError("per_stratum must be at least 1")
        if min_gap < 1:
            raise CollectorError("min_gap must be at least 1")
        self.per_stratum = int(per_stratum)
        self.min_gap = int(min_gap)
        self.phase_bounds = tuple(phase_bounds)
        self.counts: Counter[tuple[str, str, str]] = Counter()
        self.rejected: Counter[str] = Counter()
        self._last_step: dict[int, int] = {}
        self._sequence: Counter[int] = Counter()
        self._lock = threading.Lock()

    def tags_for(self, step: int, info: Mapping[str, Any]) -> tuple[str, ...]:
        """Environment event tags plus the trainer's periodic background tag."""
        tags = tuple(str(tag) for tag in (info.get("events") or ()))
        if step % self.min_gap == 0:
            tags = (*tags, PERIODIC_EVENT)
        return tags

    def offer(
        self, *, env: int, curriculum: str, step: int, tick: int, info: Mapping[str, Any],
    ) -> Candidate | None:
        """Admit one post-tick state, or return None and record why not."""
        tags = self.tags_for(step, info)
        if not tags:
            return None
        with self._lock:
            last = self._last_step.get(env, 0)
            # The opening ticks of a lifecycle are what a fresh environment
            # already provides; a bridge state has to be somewhere the learner
            # is not already starting.
            if step - last < self.min_gap:
                self.rejected["gap"] += 1
                return None
            overflowed = overflow_fields(info.get("globals"))
            if overflowed:
                for name in overflowed:
                    self.rejected[name] += 1
                self.rejected["overflow"] += 1
                return None
            event = primary_event(tags)
            phase = phase_for_tick(step, self.phase_bounds)
            key = (curriculum, event, phase)
            if self.counts[key] >= self.per_stratum:
                self.rejected["full"] += 1
                return None
            self.counts[key] += 1
            self._last_step[env] = step
            self._sequence[env] += 1
            sequence = self._sequence[env]
        return Candidate(
            env=env,
            curriculum=curriculum,
            event=event,
            phase=phase,
            step=step,
            tick=tick,
            tags=tags,
            relative_path=(
                f"{curriculum}/{event}/{phase}/{OUTCOME_TEACHER}/"
                f"t{tick:06d}_e{env:02d}_{sequence:06d}.xsnp"
            ),
        )


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------
def _open_env(config: CollectorConfig, *, curriculum: str, seed: int, expert: bool) -> Any:
    return ScreepsEnv(
        node=config.node,
        room=config.room,
        max_episode=config.max_episode,
        device="cpu",
        expert=expert,
        bot_dir=str(config.resolved_bot_dir) if expert else None,
        curriculum=curriculum,
        lean_meta=True,
        seed=seed,
    )


def _teacher_step(env: Any, config: CollectorConfig) -> tuple[bool, dict[str, Any]]:
    """Advance one teacher tick (TI runs through the expert step command)."""
    if config.expert:
        _obs, _reward, done, info = env.step()
        return bool(done), dict(info or {})
    _obs, _reward, done, info, _actions = env.step_scripted()
    return bool(done), dict(info or {})


def _record_for(
    candidate: Candidate,
    descriptor: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    seed: int,
    skill_rate: float,
) -> dict[str, Any]:
    return {
        "path": candidate.relative_path,
        "event": candidate.event,
        "phase": candidate.phase,
        # The environment guarantees these; a missing field is a broken contract,
        # not something to paper over with a value from a different source.
        "tick": int(descriptor["tick"]),
        "step": int(descriptor["step"]),
        "curriculum": str(descriptor["curriculum"]),
        "bytes": int(descriptor["bytes"]),
        "creeps": int(info.get("creeps") or 0),
        "owned_rooms": int(info.get("ownedRooms") or 0),
        "remote_staffed": int(info.get("remoteRoomsStaffed") or 0),
        "skill_rate": round(float(skill_rate), 6),
        "env": int(candidate.env),
        "seed": int(seed),
        "events": list(candidate.tags),
        "created": time.time(),
    }


def collect_env(
    config: CollectorConfig,
    index: int,
    staging: Path,
    budget: StratumBudget,
    stop: threading.Event,
) -> list[dict[str, Any]]:
    """Run one teacher lifecycle and snapshot every state it is admitted for."""
    curriculum = config.env_curriculum(index)
    seed = config.env_seed(index)
    decay = StartStateController.SKILL_EMA_DECAY
    records: list[dict[str, Any]] = []
    # Bias-corrected exactly like `StartStateController`, which re-seeds a
    # restored environment's EMA from this number and classifies outcomes with
    # it: an uncorrected average would understate every early-phase state.
    skill = 0.0
    skill_weight = 0.0
    completed = 0
    env = _open_env(config, curriculum=curriculum, seed=seed, expert=config.expert)
    try:
        env.reset()
        for _ in range(config.steps):
            if stop.is_set():
                break
            done, info = _teacher_step(env, config)
            completed += 1
            step = int(info.get("step") or completed)
            tick = int(info.get("time") or step)
            skill = decay * skill + (1.0 - decay) * (
                float(info.get("harvestDelta") or 0.0)
                + float(info.get("controlDelta") or 0.0)
            )
            skill_weight = decay * skill_weight + (1.0 - decay)
            candidate = budget.offer(
                env=index, curriculum=curriculum, step=step, tick=tick, info=info,
            )
            if candidate is not None:
                destination = staging / candidate.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                descriptor = env.snapshot(destination, candidate.tags)
                records.append(
                    _record_for(
                        candidate, descriptor, info, seed=seed,
                        skill_rate=skill / skill_weight if skill_weight > 0.0 else 0.0,
                    ),
                )
            if config.progress_every > 0 and completed % config.progress_every == 0:
                print(
                    f"[teacher_snapshots] env{index} {curriculum} "
                    f"{completed}/{config.steps} kept={len(records)}",
                    flush=True,
                )
            if done:
                break
    except Exception as error:  # env process death must fail the whole run
        # Siblings would otherwise burn their whole remaining tick budget.
        stop.set()
        raise CollectorError(
            f"teacher env {index} (curriculum={curriculum}, seed={seed}) failed "
            f"after {completed} ticks: {type(error).__name__}: {error}"
        ) from error
    finally:
        try:
            env.close()
        except Exception:
            pass
    return records


def collect_records(
    config: CollectorConfig, staging: Path, budget: StratumBudget,
) -> list[dict[str, Any]]:
    """Drive every teacher environment concurrently on one thread pool."""
    failures: list[BaseException] = []
    records: list[dict[str, Any]] = []
    stop = threading.Event()
    with ThreadPoolExecutor(
        max_workers=config.num_envs, thread_name_prefix="teacher-env",
    ) as pool:
        futures = [
            pool.submit(collect_env, config, index, staging, budget, stop)
            for index in range(config.num_envs)
        ]
        for future in futures:
            try:
                records.extend(future.result())
            except BaseException as error:  # noqa: BLE001 - re-raised below
                # Includes KeyboardInterrupt: stop the pool before it joins.
                stop.set()
                failures.append(error)
    if failures:
        raise failures[0]
    return records


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def verify_curriculum(
    config: CollectorConfig,
    curriculum: str,
    staging: Path,
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    """Restore every record into one fresh learner session; drop the failures.

    An expert-collected snapshot must reopen in a plain learner session with no
    expert code loaded, so this is the check that proves the teacher lane is
    usable by PPO at all.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[tuple[dict[str, Any], str]] = []
    env = _open_env(config, curriculum=curriculum, seed=config.env_seed(0), expert=False)
    try:
        env.reset()
        for record in records:
            path = staging / str(record["path"])
            try:
                env.restore(path)
            except Exception as error:  # noqa: BLE001 - drop or escalate below
                # A dead Node session looks exactly like an unrestorable
                # snapshot, and would silently truncate the rest of this
                # curriculum. A session that still resets is genuinely alive,
                # and resetting also clears whatever the failed restore left.
                try:
                    env.reset()
                except Exception as fatal:
                    raise CollectorError(
                        f"verification session for {curriculum} died on "
                        f"{record['path']}: {type(error).__name__}: {error}"
                    ) from fatal
                dropped.append((record, f"restore failed: {type(error).__name__}: {error}"))
                continue
            info = dict(getattr(env, "last_info", None) or {})
            if not isinstance(info.get("globals"), Mapping):
                raise CollectorError(
                    "restore reply carries no info.globals; cannot verify that the "
                    "restored state fits the observation"
                )
            overflowed = overflow_fields(info["globals"])
            if overflowed:
                dropped.append((record, f"restored state overflows {','.join(overflowed)}"))
                continue
            if "snapshotTick" not in info:
                raise CollectorError(
                    "restore reply carries no snapshotTick; cannot verify that the "
                    "restored world is the one that was recorded"
                )
            restored_tick = int(info["snapshotTick"])
            if restored_tick != int(record["tick"]):
                dropped.append((
                    record,
                    f"restored tick {restored_tick} does not match recorded {record['tick']}",
                ))
                continue
            kept.append(record)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return kept, dropped


def verify_records(
    config: CollectorConfig, staging: Path, records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    """Verify every retained snapshot, one learner session per curriculum."""
    by_curriculum: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_curriculum.setdefault(str(record["curriculum"]), []).append(record)
    if not by_curriculum:
        return [], []
    kept: list[dict[str, Any]] = []
    dropped: list[tuple[dict[str, Any], str]] = []
    failures: list[BaseException] = []
    with ThreadPoolExecutor(
        max_workers=min(config.num_envs, len(by_curriculum)),
        thread_name_prefix="teacher-verify",
    ) as pool:
        futures = [
            pool.submit(verify_curriculum, config, curriculum, staging, rows)
            for curriculum, rows in sorted(by_curriculum.items())
        ]
        for future in futures:
            try:
                curriculum_kept, curriculum_dropped = future.result()
            except BaseException as error:  # noqa: BLE001 - re-raised below
                failures.append(error)
                continue
            kept.extend(curriculum_kept)
            dropped.extend(curriculum_dropped)
    if failures:
        raise CollectorError(
            f"teacher snapshot verification failed: "
            f"{type(failures[0]).__name__}: {failures[0]}"
        ) from failures[0]
    kept.sort(key=lambda row: row["path"])
    return kept, dropped


# ---------------------------------------------------------------------------
# manifest and content addressing
# ---------------------------------------------------------------------------
def collector_signature() -> str:
    """Hash this collector's own bytes: the admission policy is provenance."""
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def build_manifest(
    config: CollectorConfig,
    records: Sequence[Mapping[str, Any]],
    *,
    collector_sha256: str,
    source_sha256: str,
) -> dict[str, Any]:
    return {
        "format": RESERVOIR_FORMAT_VERSION,
        "kind": MANIFEST_KIND,
        "teacher": config.teacher,
        "collector_source_sha256": collector_sha256,
        "source_sha256": source_sha256,
        "schema_sha256": SCHEMA_SHA256,
        "curricula": list(config.assigned_curricula()),
        "seeds": [int(seed) for seed in config.seeds()],
        "records": [
            dict(record) for record in sorted(records, key=lambda row: row["path"])
        ],
    }


def _addressable_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest minus wall-clock fields, which are not part of identity."""
    payload = {key: value for key, value in manifest.items() if key != "records"}
    payload["records"] = [
        {key: value for key, value in record.items() if key != "created"}
        for record in manifest.get("records") or []
    ]
    return payload


def _digest_entry(digest: "hashlib._Hash", relative: str, source: bytes | Path) -> None:
    """Fold one `(relative path, byte length, bytes)` triple into the digest.

    A `Path` is streamed so a large late-game set never has to be resident.
    """
    name = relative.encode("utf-8")
    digest.update(len(name).to_bytes(4, "little"))
    digest.update(name)
    if isinstance(source, (bytes, bytearray)):
        digest.update(len(source).to_bytes(8, "little"))
        digest.update(source)
        return
    path = Path(source)
    digest.update(path.stat().st_size.to_bytes(8, "little"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)


def content_address(
    manifest: Mapping[str, Any], files: Mapping[str, bytes | Path],
) -> str:
    """SHA-256 over every retained snapshot plus the identity-bearing manifest."""
    digest = hashlib.sha256()
    for relative in sorted(files):
        _digest_entry(digest, relative, files[relative])
    canonical = json.dumps(
        _addressable_manifest(manifest), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(canonical).to_bytes(8, "little"))
    digest.update(canonical)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def discard(directory: Path) -> None:
    """Remove a staging tree; a cleanup failure is reported, never swallowed."""
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        pass
    except OSError as error:
        print(
            f"[teacher_snapshots] WARNING could not remove {directory}: {error}",
            file=sys.stderr, flush=True,
        )


def _unlink_staged(staging: Path, relative: str) -> None:
    """Delete a rejected snapshot and any stratum directories it emptied.

    An empty `curriculum/event/phase/teacher/` directory in the published set
    would advertise a stratum the manifest does not contain.
    """
    path = staging / relative
    path.unlink(missing_ok=True)
    for parent in path.parents:
        if parent == staging or not parent.is_relative_to(staging):
            break
        try:
            parent.rmdir()
        except OSError:
            break


def staged_files(staging: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Locate every recorded snapshot inside the staged set and size-check it."""
    root = staging.resolve()
    files: dict[str, Path] = {}
    for record in manifest.get("records") or []:
        relative = str(record["path"])
        path = (staging / relative).resolve()
        if not path.is_relative_to(root):
            raise CollectorError(
                f"snapshot path {relative!r} escapes the staged set; the published "
                "directory would not be self-contained"
            )
        if not path.is_file():
            raise CollectorError(f"staged snapshot {relative} is missing")
        size = path.stat().st_size
        if size != int(record["bytes"]):
            raise CollectorError(
                f"staged snapshot {relative} is {size} bytes on disk but the "
                f"manifest records {record['bytes']}"
            )
        files[relative] = path
    return files


def _assert_published(destination: Path, manifest: Mapping[str, Any], files: Mapping[str, Path]) -> None:
    """A reused content address must already hold exactly this content.

    The address is only a directory name: an interrupted delete, or a reservoir
    that pruned files it did not own, leaves a directory whose name still looks
    right. Reusing it would publish a manifest pointing at missing snapshots.
    """
    manifest_path = destination / MANIFEST_NAME
    if not manifest_path.is_file():
        raise CollectorError(
            f"{destination} already exists but holds no {MANIFEST_NAME}; "
            "remove it and re-collect"
        )
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _addressable_manifest(existing) != _addressable_manifest(manifest):
        raise CollectorError(
            f"{destination} holds a different manifest than its content address "
            "implies; remove it and re-collect"
        )
    for relative, staged in files.items():
        published = destination / relative
        if not published.is_file() or not filecmp.cmp(str(published), str(staged), shallow=False):
            raise CollectorError(
                f"{destination} is incomplete: {relative} is missing or altered; "
                "remove it and re-collect"
            )


def publish(staging: Path, output: Path, manifest: Mapping[str, Any]) -> Path:
    """Move the staged set to its content address; identical content is reused."""
    files = staged_files(staging, manifest)
    address = content_address(manifest, files)
    destination = Path(output) / address
    write_json_atomic(staging / MANIFEST_NAME, manifest)
    Path(output).mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            os.replace(staging, destination)
            return destination
        except OSError as error:
            # A concurrent collector published the same content first.
            if error.errno != errno.ENOTEMPTY:
                raise
    _assert_published(destination, manifest, files)
    discard(staging)
    return destination


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------
@dataclass
class CollectionResult:
    directory: Path
    manifest: dict[str, Any]
    dropped: list[tuple[dict[str, Any], str]] = field(default_factory=list)
    rejected: Counter[str] = field(default_factory=Counter)

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self.manifest.get("records") or [])

    @property
    def total_bytes(self) -> int:
        return sum(int(record["bytes"]) for record in self.records)

    def histogram(self, key: str) -> Counter[str]:
        return Counter(str(record[key]) for record in self.records)


def collect(config: CollectorConfig) -> CollectionResult:
    """Collect, verify, and publish one immutable teacher start-state set."""
    # The environment writes snapshots itself, from its own working directory,
    # so every path handed to it must already be absolute.
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    staging = output / f"{STAGING_PREFIX}{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    budget = StratumBudget(per_stratum=config.per_stratum, min_gap=config.min_gap)
    published = False
    try:
        records = collect_records(config, staging, budget)
        if not records:
            raise CollectorError(
                "no start states were admitted; lower --min-gap, raise --steps, "
                "or check that the teacher emits event tags"
            )
        kept, dropped = verify_records(config, staging, records)
        for record, reason in dropped:
            _unlink_staged(staging, str(record["path"]))
            print(f"[teacher_snapshots] dropped {record['path']}: {reason}", flush=True)
        if not kept:
            raise CollectorError(
                f"all {len(records)} collected start states failed learner restore; "
                "nothing to publish"
            )
        published_curricula = {str(record["curriculum"]) for record in kept}
        missing = [
            name for name in config.assigned_curricula()
            if name not in published_curricula
        ]
        if missing:
            raise CollectorError(
                f"curricula {missing} produced no usable start states; "
                "raise --steps or relax --min-gap"
            )
        manifest = build_manifest(
            config, kept,
            collector_sha256=collector_signature(),
            source_sha256=source_signature(),
        )
        directory = publish(staging, output, manifest)
        published = True
        return CollectionResult(
            directory=directory,
            manifest=manifest,
            dropped=dropped,
            rejected=Counter(budget.rejected),
        )
    finally:
        if not published:
            discard(staging)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect immutable teacher start states for the PPO reservoir",
    )
    parser.add_argument("--teacher", choices=TEACHERS, default="ti")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20_000, help="teacher ticks per env")
    parser.add_argument("--max-episode", type=int, default=20_000)
    parser.add_argument("--curriculum", type=str, default=DEFAULT_CURRICULUM)
    parser.add_argument("--seed", type=int, default=201, help="base seed; env i uses seed+i")
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument(
        "--min-gap", type=int, default=64, help="minimum ticks between captures per env",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--node", type=str, default=None)
    parser.add_argument("--room", type=str, default="W7N3")
    parser.add_argument("--bot-dir", type=str, default=None, help="TI dist directory")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> CollectorConfig:
    curricula = tuple(
        item.strip() for item in str(args.curriculum).split(",") if item.strip()
    )
    return CollectorConfig(
        teacher=args.teacher,
        num_envs=args.num_envs,
        steps=args.steps,
        max_episode=args.max_episode,
        curricula=curricula,
        seed=args.seed,
        per_stratum=args.per_stratum,
        min_gap=args.min_gap,
        output=Path(args.output),
        node=args.node,
        room=args.room,
        bot_dir=args.bot_dir,
        progress_every=args.progress_every,
    )


def _histogram_line(name: str, counts: Mapping[str, int]) -> str:
    body = " ".join(f"{key}={counts[key]}" for key in sorted(counts))
    return f"[teacher_snapshots] {name}: {body or '-'}"


def main(argv: Sequence[str] | None = None) -> int:
    # Intra-op parallelism makes environment workers contend; see
    # vec_env.configure_host_threads.
    configure_host_threads()
    args = parse_args(argv)
    try:
        config = config_from_args(args)
        print(
            f"[teacher_snapshots] teacher={config.teacher} envs={config.num_envs} "
            f"steps={config.steps} curricula={','.join(config.curricula)} "
            f"seed={config.seed} per_stratum={config.per_stratum} "
            f"min_gap={config.min_gap}",
            flush=True,
        )
        result = collect(config)
    except CollectorError as error:
        print(f"[teacher_snapshots] FAIL {error}", file=sys.stderr, flush=True)
        return 2
    print(
        f"[teacher_snapshots] kept={len(result.records)} "
        f"dropped={len(result.dropped)} bytes={result.total_bytes}",
        flush=True,
    )
    print(_histogram_line("event", result.histogram("event")), flush=True)
    print(_histogram_line("phase", result.histogram("phase")), flush=True)
    print(_histogram_line("curriculum", result.histogram("curriculum")), flush=True)
    print(_histogram_line("rejected", result.rejected), flush=True)
    if len(result.dropped) > len(result.records):
        print(
            f"[teacher_snapshots] WARNING more states failed learner restore "
            f"({len(result.dropped)}) than were kept ({len(result.records)}); "
            "the teacher lane will be thin",
            file=sys.stderr, flush=True,
        )
    # Last line: the immutable directory to paste into --teacher-start-states.
    print(str(result.directory), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
