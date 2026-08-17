"""Event-stratified start-state reservoir for long-horizon PPO.

Rollouts of a few hundred ticks that always begin at tick zero only ever train
on the first slice of a twenty-thousand-tick economy, so behavior that only
appears later — remote staffing, replacement, claiming, recovery — decays even
though behavior cloning demonstrated it. This module keeps a bounded, balanced
population of *start states* so every rollout contains early, middle, late, and
endgame decisions.

Design contract:

- Untouched full-lifecycle environments always exist and are never restored
  into. They are the only source of genuinely late states and the only
  environments whose economy the policy built end to end.
- Policy-lane starts come from recent rollouts, keeping successful and failed
  worlds in separate strata so both are sampled.
- Teacher-lane starts bridge states the learner cannot reach yet. They retire
  per phase as soon as the policy supplies its own examples for that phase.
- Strata are event-defined, and pre-decision events are first class: a state
  captured before a spawn or claim decision leaves the decision to the learner
  instead of inheriting the teacher's choice.
- Sampling is uniform over non-empty strata, then uniform within a stratum, so
  rare late-economy strata are not drowned by common plateau states.
- Evaluation never draws from this reservoir; qualification stays on fresh
  twenty-thousand-tick worlds.
"""
from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

INDEX_NAME = "index.json"
RESERVOIR_FORMAT_VERSION = 1

LANE_FRESH = "fresh"
LANE_POLICY = "policy"
LANE_TEACHER = "teacher"
LANES = (LANE_FRESH, LANE_POLICY, LANE_TEACHER)

OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"
OUTCOME_TEACHER = "teacher"

# Event tags emitted by samples/rl/env/server.mjs, plus the trainer-side
# background tag. `pre_*` tags mark states where a strategic decision is still
# open; `post_*` tags mark the downstream value-learning states.
ENV_EVENTS = (
    "pre_spawn",
    "post_spawn",
    "pre_claim",
    "post_claim",
    "remote_outbound",
    "remote_at_source",
    "remote_loaded_home",
    "replacement_due",
    "rcl_up",
    "recovery",
)
PERIODIC_EVENT = "periodic"
EVENTS = (*ENV_EVENTS, PERIODIC_EVENT)

# Rarer, later, or more decision-bearing tags win when a tick fires several, so
# a state is filed under the stratum it is actually valuable for.
_EVENT_PRIORITY = {
    "post_claim": 0,
    "pre_claim": 1,
    "rcl_up": 2,
    "recovery": 3,
    "remote_loaded_home": 4,
    "remote_at_source": 5,
    "remote_outbound": 6,
    "replacement_due": 7,
    "pre_spawn": 8,
    "post_spawn": 9,
    PERIODIC_EVENT: 10,
}

# Phase boundaries over the declared 20,000-tick horizon. Phases exist so that
# "the policy forgot the late game" is a measurable, coverable deficiency.
DEFAULT_PHASE_BOUNDS: tuple[tuple[str, int], ...] = (
    ("early", 2_000),
    ("mid", 8_000),
    ("late", 15_000),
    ("endgame", 1 << 62),
)


def phase_for_tick(tick: int, bounds: Sequence[tuple[str, int]] = DEFAULT_PHASE_BOUNDS) -> str:
    for name, upper in bounds:
        if tick < upper:
            return name
    return bounds[-1][0]


def primary_event(events: Iterable[str]) -> str:
    """Pick the stratum-defining tag for a tick that fired several."""
    best = PERIODIC_EVENT
    best_rank = _EVENT_PRIORITY[PERIODIC_EVENT]
    for tag in events:
        rank = _EVENT_PRIORITY.get(tag)
        if rank is not None and rank < best_rank:
            best, best_rank = tag, rank
    return best


@dataclass(frozen=True)
class SnapshotRecord:
    """One start state plus the provenance needed to audit and rebalance it."""

    path: str
    lane: str
    event: str
    phase: str
    tick: int
    step: int
    outcome: str
    curriculum: str
    bytes: int
    creeps: int
    owned_rooms: int
    remote_staffed: int
    skill_rate: float
    update: int
    env: int
    seed: int
    events: tuple[str, ...] = ()
    created: float = 0.0

    @property
    def stratum(self) -> tuple[str, str, str, str]:
        return (self.lane, self.event, self.phase, self.outcome)


def stratum_name(stratum: Sequence[str]) -> str:
    return "/".join(stratum)


@dataclass
class ReservoirConfig:
    """Explicit, checkpointed reservoir behavior."""

    per_stratum: int = 24
    refresh_probability: float = 0.25
    min_capture_gap: int = 64
    segment_ticks: int = 2048
    success_skill_rate: float = 4.0
    teacher_retire_records: int = 32
    teacher_retire_events: int = 3
    phase_bounds: tuple[tuple[str, int], ...] = DEFAULT_PHASE_BOUNDS

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phase_bounds"] = [list(item) for item in self.phase_bounds]
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ReservoirConfig:
        data = dict(payload)
        bounds = data.pop("phase_bounds", None)
        config = cls(**data)
        if bounds:
            config.phase_bounds = tuple((str(name), int(upper)) for name, upper in bounds)
        return config


@dataclass
class LaneMixture:
    """Per-update composition of environment start states."""

    fresh: int
    policy: int
    teacher: int

    @property
    def total(self) -> int:
        return self.fresh + self.policy + self.teacher

    @classmethod
    def parse(cls, spec: str, num_envs: int) -> LaneMixture:
        """Parse `fresh=12,policy=8,teacher=4`."""
        counts = {LANE_FRESH: 0, LANE_POLICY: 0, LANE_TEACHER: 0}
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            name, _, value = item.partition("=")
            name = name.strip()
            if name not in counts or not value.strip().isdigit():
                raise ValueError(f"invalid start-state mixture entry {item!r}")
            counts[name] = int(value)
        mixture = cls(**counts)
        if mixture.total != num_envs:
            raise ValueError(
                f"start-state mixture {spec!r} assigns {mixture.total} envs, expected {num_envs}"
            )
        if mixture.fresh < 1:
            raise ValueError("at least one environment must run an untouched full lifecycle")
        return mixture

    def assign(self, num_envs: int) -> tuple[str, ...]:
        """Assign lanes to env indices.

        Fresh environments take the low indices so their curriculum coverage is
        stable across runs, and the snapshot lanes are interleaved over the
        remaining indices so no lane is pinned to one curriculum.
        """
        if self.total != num_envs:
            raise ValueError(f"mixture covers {self.total} envs, expected {num_envs}")
        lanes = [LANE_FRESH] * self.fresh
        rest = [LANE_POLICY] * self.policy + [LANE_TEACHER] * self.teacher
        # Interleave policy/teacher rather than blocking them together.
        interleaved: list[str] = []
        policy_left, teacher_left = self.policy, self.teacher
        while policy_left or teacher_left:
            if policy_left and (
                teacher_left == 0 or policy_left * max(1, self.teacher) >= teacher_left * max(1, self.policy)
            ):
                interleaved.append(LANE_POLICY)
                policy_left -= 1
            else:
                interleaved.append(LANE_TEACHER)
                teacher_left -= 1
        assert sorted(interleaved) == sorted(rest)
        return tuple(lanes + interleaved)


class StartStateReservoir:
    """Bounded, event-stratified population of restorable start states.

    Files live under `root/<lane>/<event>/<phase>/<outcome>/<id>.xsnp`; the index
    is a single JSON document so a PPO resume restores the exact population.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        config: ReservoirConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config or ReservoirConfig()
        self.rng = rng or random.Random(0)
        self.strata: dict[tuple[str, str, str, str], deque[SnapshotRecord]] = {}
        self.stats: Counter[str] = Counter()
        self._sequence = 0
        # Restarts are issued from the vector environment's worker threads, so
        # draws and their counters must be serialized or a seeded run will not
        # reproduce its own start states.
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- population ----------------------------------------------------
    @property
    def size(self) -> int:
        return sum(len(bucket) for bucket in self.strata.values())

    def records(self) -> list[SnapshotRecord]:
        return [record for bucket in self.strata.values() for record in bucket]

    def lane_size(self, lane: str) -> int:
        return sum(len(bucket) for key, bucket in self.strata.items() if key[0] == lane)

    def phase_counts(self, lane: str | None = None) -> Counter[str]:
        counts: Counter[str] = Counter()
        for (record_lane, _event, phase, _outcome), bucket in self.strata.items():
            if lane is not None and record_lane != lane:
                continue
            counts[phase] += len(bucket)
        return counts

    def event_counts(self, lane: str | None = None) -> Counter[str]:
        counts: Counter[str] = Counter()
        for (record_lane, event, _phase, _outcome), bucket in self.strata.items():
            if lane is not None and record_lane != lane:
                continue
            counts[event] += len(bucket)
        return counts

    # ---- admission -----------------------------------------------------
    def destination(
        self, *, lane: str, event: str, phase: str, outcome: str, tick: int, env: int,
    ) -> Path:
        self._sequence += 1
        directory = self.root / lane / event / phase / outcome
        directory.mkdir(parents=True, exist_ok=True)
        name = f"t{int(tick):06d}_e{int(env):02d}_{self._sequence:06d}.xsnp"
        return directory / name

    def should_admit(self, *, lane: str, event: str, phase: str, outcome: str) -> bool:
        """Fill an under-populated stratum first; refresh a full one sometimes.

        Unconditional capture would spend collection time re-recording common
        plateau states, and never capturing into a full stratum would freeze the
        policy lane on the first competent rollout it ever produced.
        """
        bucket = self.strata.get((lane, event, phase, outcome))
        if bucket is None or len(bucket) < self.config.per_stratum:
            return True
        return self.rng.random() < self.config.refresh_probability

    def admit(self, record: SnapshotRecord) -> SnapshotRecord | None:
        """Insert a record, evicting the oldest of its stratum when full.

        Returns the evicted record, whose file has already been removed.
        """
        bucket = self.strata.setdefault(record.stratum, deque())
        bucket.append(record)
        self.stats["admitted"] += 1
        self.stats[f"admitted_{record.lane}"] += 1
        evicted: SnapshotRecord | None = None
        while len(bucket) > self.config.per_stratum:
            evicted = bucket.popleft()
            self._remove_file(evicted)
            self.stats["evicted"] += 1
        return evicted

    def _remove_file(self, record: SnapshotRecord) -> None:
        try:
            os.unlink(record.path)
        except FileNotFoundError:
            pass
        except OSError as error:  # noqa: BLE001
            print(f"[reservoir] could not remove {record.path}: {error}", flush=True)

    # ---- sampling ------------------------------------------------------
    def teacher_retired(self, phase: str) -> bool:
        """Teacher bridging for a phase ends once the policy covers it itself."""
        policy_records = 0
        events: set[str] = set()
        for (lane, event, record_phase, _outcome), bucket in self.strata.items():
            if lane != LANE_POLICY or record_phase != phase or not bucket:
                continue
            policy_records += len(bucket)
            events.add(event)
        return (
            policy_records >= self.config.teacher_retire_records
            and len(events) >= self.config.teacher_retire_events
        )

    def sample(self, lane: str) -> SnapshotRecord | None:
        """Uniform over non-empty strata of `lane`, then uniform within it.

        Frequency-proportional sampling would bury the rare late-economy strata
        this reservoir exists to preserve.
        """
        with self._lock:
            return self._sample_locked(lane)

    def _sample_locked(self, lane: str) -> SnapshotRecord | None:
        if lane == LANE_TEACHER:
            candidates = [
                key for key, bucket in self.strata.items()
                if key[0] == LANE_TEACHER and bucket and not self.teacher_retired(key[2])
            ]
            if not candidates:
                # Retired or empty: a teacher slot becomes a policy slot rather
                # than silently reverting to tick zero.
                return self._sample_locked(LANE_POLICY)
        else:
            candidates = [key for key, bucket in self.strata.items() if key[0] == lane and bucket]
        if not candidates:
            return None
        candidates.sort()
        stratum = candidates[self.rng.randrange(len(candidates))]
        bucket = self.strata[stratum]
        record = bucket[self.rng.randrange(len(bucket))]
        self.stats["sampled"] += 1
        self.stats[f"sampled_{record.lane}"] += 1
        self.stats[f"sampled_phase_{record.phase}"] += 1
        return record

    # ---- persistence ---------------------------------------------------
    def index_path(self) -> Path:
        return self.root / INDEX_NAME

    def save(self) -> Path:
        payload = {
            "format": RESERVOIR_FORMAT_VERSION,
            "config": self.config.to_json(),
            "sequence": self._sequence,
            "stats": dict(self.stats),
            "records": [asdict(record) for record in self.records()],
            "saved": time.time(),
        }
        target = self.index_path()
        temporary = target.with_suffix(".json.tmp")
        # The index is the whole population: a rename that lands before the bytes
        # do would strand every snapshot file it names.
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        return target

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        config: ReservoirConfig | None = None,
        rng: random.Random | None = None,
        prune_orphans: bool = True,
    ) -> StartStateReservoir:
        reservoir = cls(root, config=config, rng=rng)
        index = reservoir.index_path()
        if not index.is_file():
            return reservoir
        payload = json.loads(index.read_text(encoding="utf-8"))
        version = int(payload.get("format", 0))
        if version != RESERVOIR_FORMAT_VERSION:
            raise ValueError(
                f"reservoir index format {version} is not {RESERVOIR_FORMAT_VERSION}"
            )
        if config is None and "config" in payload:
            reservoir.config = ReservoirConfig.from_json(payload["config"])
        reservoir._sequence = int(payload.get("sequence", 0))
        reservoir.stats = Counter({str(k): int(v) for k, v in (payload.get("stats") or {}).items()})
        known: set[str] = set()
        dropped = 0
        for row in payload.get("records") or []:
            row = dict(row)
            row["events"] = tuple(row.get("events") or ())
            record = SnapshotRecord(**row)
            if not Path(record.path).is_file():
                dropped += 1
                continue
            reservoir.strata.setdefault(record.stratum, deque()).append(record)
            known.add(str(Path(record.path).resolve()))
        if dropped:
            print(f"[reservoir] dropped {dropped} index entries without files", flush=True)
        if prune_orphans:
            reservoir.prune_orphans(known)
        return reservoir

    def prune_orphans(self, known: set[str] | None = None) -> int:
        """Delete snapshot files the index does not own.

        A crashed run leaves files behind; without pruning they accumulate
        silently and are never sampled. Only this reservoir's own lane subtrees
        are swept, because an imported teacher bridge set may be published under
        the same root and is never owned by this index.
        """
        if known is None:
            known = {str(Path(record.path).resolve()) for record in self.records()}
        removed = 0
        for lane in LANES:
            lane_root = self.root / lane
            if not lane_root.is_dir():
                continue
            for path in lane_root.rglob("*.xsnp"):
                if str(path.resolve()) in known:
                    continue
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
            for path in lane_root.rglob("*.xsnp.tmp"):
                try:
                    path.unlink()
                except OSError:
                    continue
        if removed:
            print(f"[reservoir] pruned {removed} orphan snapshot files", flush=True)
        self.stats["pruned"] += removed
        return removed

    # ---- metrics -------------------------------------------------------
    def metrics(self) -> dict[str, float]:
        out: dict[str, float] = {
            "reservoir_records": float(self.size),
            "reservoir_strata": float(sum(1 for bucket in self.strata.values() if bucket)),
            "reservoir_bytes": float(sum(record.bytes for record in self.records())),
        }
        for lane in LANES:
            out[f"reservoir_{lane}_records"] = float(self.lane_size(lane))
        for phase, count in self.phase_counts().items():
            out[f"reservoir_phase_{phase}"] = float(count)
        for phase, count in self.phase_counts(LANE_POLICY).items():
            out[f"reservoir_policy_phase_{phase}"] = float(count)
        for event, count in self.event_counts().items():
            out[f"reservoir_event_{event}"] = float(count)
        for phase, _bound in self.config.phase_bounds:
            out[f"reservoir_teacher_retired_{phase}"] = float(self.teacher_retired(phase))
        for key, value in self.stats.items():
            out[f"reservoir_{key}"] = float(value)
        return out


@dataclass
class EnvStartState:
    """Per-environment start-state bookkeeping for one PPO run."""

    lane: str
    segment_ticks: int
    ticks_in_segment: int = 0
    # Captures start one gap into a segment: the first ticks of a world are what
    # the fresh lane already supplies, and a stale marker from a finished
    # episode would otherwise suppress captures for the whole next one.
    last_capture_step: int = 0
    skill_ema: float = 0.0
    skill_ema_weight: float = 0.0
    origin: str = "fresh"
    origin_path: str | None = None
    origin_tick: int = 0
    events_seen: Counter[str] = field(default_factory=Counter)

    @property
    def skill_rate(self) -> float:
        """Bias-corrected recent skill per tick.

        The raw exponential average starts at zero, so without correction every
        newly started segment looks unproductive and the success/failure label
        would measure segment age instead of economy quality.
        """
        if self.skill_ema_weight <= 0.0:
            return 0.0
        return self.skill_ema / self.skill_ema_weight

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["events_seen"] = dict(self.events_seen)
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> EnvStartState:
        data = dict(payload)
        data["events_seen"] = Counter(data.get("events_seen") or {})
        return cls(**data)


class StartStateController:
    """Glue between the reservoir, the vector environment, and PPO bookkeeping.

    Responsibilities: decide which env restarts and from where, decide which
    post-tick states are worth capturing, and label captured states so the
    reservoir stays balanced across events, phases, and outcomes.
    """

    SKILL_EMA_DECAY = 0.995

    def __init__(
        self,
        reservoir: StartStateReservoir,
        *,
        mixture: LaneMixture,
        num_envs: int,
        max_episode: int,
        rng: random.Random | None = None,
    ) -> None:
        self.reservoir = reservoir
        self.mixture = mixture
        self.num_envs = num_envs
        self.max_episode = max_episode
        self.rng = rng or reservoir.rng
        self.lanes = mixture.assign(num_envs)
        self.envs = [
            EnvStartState(
                lane=lane,
                # A fresh-lane environment is never truncated by the reservoir; it
                # runs the declared horizon so late phases actually exist.
                segment_ticks=0 if lane == LANE_FRESH else reservoir.config.segment_ticks,
            )
            for lane in self.lanes
        ]
        self.update = 0
        self.capture_seconds = 0.0
        self.transition_phases: Counter[str] = Counter()

    # ---- restarts ------------------------------------------------------
    def start_path(self, index: int) -> str | None:
        """Called by the vector env when env `index` needs a new start state."""
        state = self.envs[index]
        state.ticks_in_segment = 0
        state.events_seen.clear()
        record = None if state.lane == LANE_FRESH else self.reservoir.sample(state.lane)
        if record is None:
            state.origin, state.origin_path, state.origin_tick = "fresh", None, 0
            state.last_capture_step = 0
            state.skill_ema = 0.0
            state.skill_ema_weight = 0.0
            return None
        state.origin = f"{record.lane}:{record.event}:{record.phase}:{record.outcome}"
        state.origin_path = record.path
        state.origin_tick = record.tick
        # A restored world resumes at the snapshot's absolute step, so both the
        # capture floor and the productivity estimate continue from there.
        state.last_capture_step = int(record.step)
        state.skill_ema = record.skill_rate
        state.skill_ema_weight = 1.0
        return record.path

    def note_step(self, index: int, info: dict[str, Any]) -> None:
        """Update per-env accounting from one post-tick info payload."""
        state = self.envs[index]
        state.ticks_in_segment += 1
        skill = float(info.get("harvestDelta") or 0.0) + float(info.get("controlDelta") or 0.0)
        decay = self.SKILL_EMA_DECAY
        state.skill_ema = decay * state.skill_ema + (1.0 - decay) * skill
        state.skill_ema_weight = decay * state.skill_ema_weight + (1.0 - decay)
        self.transition_phases[phase_for_tick(
            int(info.get("step") or 0), self.reservoir.config.phase_bounds,
        )] += 1
        if info.get("episode_done") or info.get("segment_boundary"):
            state.ticks_in_segment = 0

    def books_full_lifecycle(self, index: int) -> bool:
        """Whether env `index`'s episode end is a real full-lifecycle episode.

        A restored world inherits the snapshot's step, so it can reach the
        declared horizon after a few hundred ticks. Booking that fragment as a
        completed episode would drag the episodic-return curve down precisely
        when the reservoir starts supplying late states.
        """
        return self.lanes[index] == LANE_FRESH

    def restart_due(self, index: int) -> bool:
        state = self.envs[index]
        return state.segment_ticks > 0 and state.ticks_in_segment >= state.segment_ticks

    def due_restarts(self) -> list[int]:
        return [index for index in range(self.num_envs) if self.restart_due(index)]

    # ---- captures ------------------------------------------------------
    def outcome_for(self, index: int, info: dict[str, Any]) -> str:
        state = self.envs[index]
        if int(info.get("creeps") or 0) <= 1:
            return OUTCOME_FAILURE
        threshold = self.reservoir.config.success_skill_rate
        return OUTCOME_SUCCESS if state.skill_rate >= threshold else OUTCOME_FAILURE

    def capture_requests(
        self, infos: Sequence[dict[str, Any]],
    ) -> list[tuple[int, str, tuple[str, ...], dict[str, Any]]]:
        """Select `(env, path, event tags, pending record fields)` to capture."""
        requests: list[tuple[int, str, tuple[str, ...], dict[str, Any]]] = []
        config = self.reservoir.config
        for index, info in enumerate(infos):
            if info.get("episode_done") or info.get("segment_boundary"):
                # The world already moved on to the next start state.
                continue
            state = self.envs[index]
            step = int(info.get("step") or 0)
            tags = tuple(info.get("events") or ())
            if step % max(1, config.min_capture_gap) == 0:
                tags = (*tags, PERIODIC_EVENT)
            if not tags:
                continue
            if step - state.last_capture_step < config.min_capture_gap:
                continue
            event = primary_event(tags)
            phase = phase_for_tick(step, config.phase_bounds)
            outcome = self.outcome_for(index, info)
            if not self.reservoir.should_admit(
                lane=LANE_POLICY, event=event, phase=phase, outcome=outcome,
            ):
                continue
            destination = self.reservoir.destination(
                lane=LANE_POLICY, event=event, phase=phase, outcome=outcome,
                tick=int(info.get("time") or step), env=index,
            )
            # The gap and the per-env event histogram are charged in
            # `commit_captures`: a capture that failed to write must not consume
            # this environment's next capture window.
            requests.append((index, str(destination), tags, {
                "lane": LANE_POLICY,
                "event": event,
                "phase": phase,
                "outcome": outcome,
                "skill_rate": float(state.skill_rate),
                "creeps": int(info.get("creeps") or 0),
                "owned_rooms": int(info.get("ownedRooms") or 0),
                "remote_staffed": int(info.get("remoteRoomsStaffed") or 0),
                "update": int(self.update),
                "env": int(index),
                "events": tags,
            }))
        return requests

    def commit_captures(
        self,
        descriptors: Sequence[dict[str, Any]],
        pending: dict[int, dict[str, Any]],
        *,
        seeds: Sequence[int],
    ) -> int:
        """Turn environment snapshot descriptors into reservoir records."""
        admitted = 0
        for descriptor in descriptors:
            index = int(descriptor["env"])
            fields = pending.get(index)
            if fields is None:
                continue
            state = self.envs[index]
            state.last_capture_step = int(descriptor["step"])
            state.events_seen[str(fields["event"])] += 1
            record = SnapshotRecord(
                path=str(descriptor["path"]),
                lane=str(fields["lane"]),
                event=str(fields["event"]),
                phase=str(fields["phase"]),
                tick=int(descriptor["tick"]),
                step=int(descriptor["step"]),
                outcome=str(fields["outcome"]),
                curriculum=str(descriptor.get("curriculum") or "unknown"),
                bytes=int(descriptor.get("bytes") or 0),
                creeps=int(fields["creeps"]),
                owned_rooms=int(fields["owned_rooms"]),
                remote_staffed=int(fields["remote_staffed"]),
                skill_rate=float(fields["skill_rate"]),
                update=int(fields["update"]),
                env=index,
                seed=int(seeds[index]) if index < len(seeds) else 0,
                events=tuple(fields["events"]),
                created=time.time(),
            )
            self.reservoir.admit(record)
            admitted += 1
        return admitted

    # ---- reporting -----------------------------------------------------
    def metrics(self) -> dict[str, float]:
        out = self.reservoir.metrics()
        out["start_mix_fresh"] = float(self.mixture.fresh)
        out["start_mix_policy"] = float(self.mixture.policy)
        out["start_mix_teacher"] = float(self.mixture.teacher)
        out["start_capture_seconds"] = float(self.capture_seconds)
        total = float(sum(self.transition_phases.values())) or 1.0
        for phase, _bound in self.reservoir.config.phase_bounds:
            out[f"train_phase_fraction_{phase}"] = self.transition_phases[phase] / total
        starts: Counter[str] = Counter(state.origin.split(":")[0] for state in self.envs)
        for origin, count in starts.items():
            out[f"start_origin_{origin}"] = float(count)
        ticks = [float(state.origin_tick) for state in self.envs]
        out["start_tick_mean"] = sum(ticks) / max(1, len(ticks))
        out["start_tick_max"] = max(ticks) if ticks else 0.0
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "lanes": list(self.lanes),
            "update": int(self.update),
            "envs": [state.to_json() for state in self.envs],
            "transition_phases": dict(self.transition_phases),
            "mixture": {
                "fresh": self.mixture.fresh,
                "policy": self.mixture.policy,
                "teacher": self.mixture.teacher,
            },
            "reservoir_root": str(self.reservoir.root),
            "reservoir_config": self.reservoir.config.to_json(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        mixture = payload.get("mixture") or {}
        if mixture and LaneMixture(**mixture) != self.mixture:
            raise ValueError(
                "start-state mixture differs from the checkpoint; resume with the original mixture"
            )
        lanes = tuple(payload.get("lanes") or ())
        if lanes and lanes != self.lanes:
            raise ValueError("start-state lane assignment differs from the checkpoint")
        previous_root = payload.get("reservoir_root")
        if previous_root and Path(previous_root).resolve() != self.reservoir.root.resolve():
            raise ValueError(
                f"checkpoint reservoir root {previous_root} differs from "
                f"{self.reservoir.root}; resume against the population it trained on"
            )
        rows = payload.get("envs") or []
        if len(rows) != self.num_envs:
            raise ValueError(
                f"checkpoint tracks {len(rows)} envs, run has {self.num_envs}"
            )
        # A PPO continuation restarts its environments, so per-env segment
        # counters from the previous process describe worlds that no longer
        # exist. Only the run-level counters and the reservoir population carry
        # over; the live segment state is whatever the fresh fleet just started.
        self.update = int(payload.get("update", 0))
        self.transition_phases = Counter(
            {str(k): int(v) for k, v in (payload.get("transition_phases") or {}).items()}
        )


def import_teacher_snapshots(
    reservoir: StartStateReservoir, manifest_path: str | Path,
) -> int:
    """Load a collected teacher bridge set into the teacher lane.

    Teacher snapshots are collected once, immutably, by
    `samples.rl.agent.teacher_snapshots`. They are referenced in place: the
    reservoir must never evict or delete a file it does not own.
    """
    manifest = Path(manifest_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if int(payload.get("format", 0)) != RESERVOIR_FORMAT_VERSION:
        raise ValueError("teacher snapshot manifest format is not supported")
    imported = 0
    for row in payload.get("records") or []:
        path = Path(row["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        if not path.is_file():
            raise FileNotFoundError(f"teacher snapshot {path} is missing")
        record = SnapshotRecord(
            path=str(path),
            lane=LANE_TEACHER,
            event=str(row["event"]),
            # Phase must come from this run's partition, or teacher coverage and
            # policy coverage would be compared across different partitions.
            phase=str(row.get("phase") or phase_for_tick(
                int(row["step"]), reservoir.config.phase_bounds,
            )),
            tick=int(row["tick"]),
            step=int(row["step"]),
            outcome=OUTCOME_TEACHER,
            curriculum=str(row.get("curriculum") or "unknown"),
            bytes=int(row.get("bytes") or 0),
            creeps=int(row.get("creeps") or 0),
            owned_rooms=int(row.get("owned_rooms") or 0),
            remote_staffed=int(row.get("remote_staffed") or 0),
            skill_rate=float(row.get("skill_rate") or 0.0),
            update=-1,
            env=int(row.get("env") or 0),
            seed=int(row.get("seed") or 0),
            events=tuple(row.get("events") or ()),
            created=float(row.get("created") or 0.0),
        )
        bucket = reservoir.strata.setdefault(record.stratum, deque())
        if any(existing.path == record.path for existing in bucket):
            continue
        # Teacher records are external and immutable: keep the whole bridge set
        # rather than evicting into the policy lane's capacity.
        bucket.append(record)
        imported += 1
    reservoir.stats["teacher_imported"] += imported
    return imported


def relative_entropy_balance(counts: Counter[str], support: Sequence[str]) -> float:
    """Normalized entropy of a coverage histogram (1.0 = perfectly balanced)."""
    total = sum(counts.get(name, 0) for name in support)
    if total <= 0 or len(support) <= 1:
        return 0.0
    entropy = 0.0
    for name in support:
        share = counts.get(name, 0) / total
        if share > 0:
            entropy -= share * math.log(share)
    return entropy / math.log(len(support))
