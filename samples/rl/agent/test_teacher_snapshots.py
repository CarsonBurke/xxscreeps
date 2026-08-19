"""Unit tests for the teacher start-state collector (no engine processes).

Every test drives `teacher_snapshots` against a fake environment so admission,
verification, manifest shape, and content addressing are checked without a Node
simulator. Engine fidelity is covered by the real smoke run, not here.
"""
from __future__ import annotations

import inspect
import json
import threading
import time
from pathlib import Path

import pytest

from samples.rl.agent import teacher_snapshots as ts
from samples.rl.agent.constants import SCHEMA
from samples.rl.agent.env_client import ScreepsEnv as RealScreepsEnv
from samples.rl.agent.state_reservoir import (
    LANE_TEACHER, OUTCOME_TEACHER, ReservoirConfig, StartStateReservoir,
    import_teacher_snapshots,
)

MANIFEST_KEYS = {
    "format", "kind", "teacher", "collector_source_sha256", "source_sha256",
    "schema_sha256", "curricula", "seeds", "records",
}
RECORD_KEYS = {
    "path", "event", "phase", "tick", "step", "curriculum", "bytes", "creeps",
    "owned_rooms", "remote_staffed", "skill_rate", "env", "seed", "events",
    "created",
}


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
def _tick_from_name(path: Path) -> int:
    return int(path.name.split("_")[0][1:])


def _default_restore_info(env: "FakeEnv", path: Path) -> dict:
    return {
        "restored": True,
        "snapshotTick": _tick_from_name(path),
        "step": 0,
        "globals": {},
    }


class FakeEnv:
    """Stand-in for `ScreepsEnv` driven by a per-tick info builder."""

    def __init__(
        self, *, info_for, payload_for, restore_info, curriculum, seed, expert, **kwargs,
    ):
        self.curriculum = curriculum
        self.seed = int(seed)
        self.expert = bool(expert)
        self.kwargs = kwargs
        self._info_for = info_for
        self._payload_for = payload_for
        self._restore_info = restore_info
        self.step_index = 0
        self.last_info: dict = {}
        self.restored: list[str] = []
        self.snapshots: list[str] = []
        self.closed = False
        # Set by a scenario to simulate a Node session that has actually died,
        # as opposed to one snapshot the learner cannot open.
        self.dead = False

    def reset(self) -> dict:
        if self.dead:
            raise RuntimeError("env server died: exit 1")
        self.step_index = 0
        self.last_info = {"step": 0, "time": 1_000, "globals": {}}
        return {}

    def _advance(self) -> dict:
        self.step_index += 1
        info = dict(self._info_for(self, self.step_index))
        info.setdefault("step", self.step_index)
        info.setdefault("time", self.step_index + 1_000)
        info.setdefault("globals", {})
        info.setdefault("harvestDelta", 2.0)
        info.setdefault("controlDelta", 1.0)
        info.setdefault("creeps", 5)
        info.setdefault("ownedRooms", 1)
        info.setdefault("remoteRoomsStaffed", 1)
        self.last_info = info
        return info

    def step(self):
        assert self.expert, "step() is the expert path"
        info = self._advance()
        return {}, 0.0, False, info

    def step_scripted(self):
        assert not self.expert, "step_scripted() is the learner path"
        info = self._advance()
        return {}, 0.0, False, info, {}

    def snapshot(self, path, events=()):
        payload = self._payload_for(self, self.step_index)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        self.snapshots.append(str(destination))
        return {
            "path": str(destination),
            "bytes": len(payload),
            "tick": int(self.last_info["time"]),
            "step": int(self.last_info["step"]),
            "rooms": ["W7N3"],
            "curriculum": self.curriculum,
            "expert": self.expert,
            "events": list(events),
        }

    def restore(self, path):
        self.restored.append(str(path))
        self.last_info = dict(self._restore_info(self, Path(path)))
        return {}

    def close(self) -> None:
        self.closed = True


def install_env(
    monkeypatch, *, info_for, payload_for=None, restore_info=_default_restore_info,
) -> list[FakeEnv]:
    """Replace `ScreepsEnv` and record every fake environment constructed."""
    created: list[FakeEnv] = []
    payload_for = payload_for or (
        lambda env, step: f"snapshot {env.curriculum} {env.seed} {step}".encode()
    )

    def factory(**kwargs):
        # The collector must keep calling the real constructor's keywords; a
        # rename in env_client would otherwise leave this suite green.
        inspect.signature(RealScreepsEnv.__init__).bind_partial(None, **kwargs)
        env = FakeEnv(
            info_for=info_for,
            payload_for=payload_for,
            restore_info=restore_info,
            **kwargs,
        )
        created.append(env)
        return env

    monkeypatch.setattr(ts, "ScreepsEnv", factory)
    monkeypatch.setattr(ts, "source_signature", lambda: "source-sha")
    return created


def plan_info(env: FakeEnv, step: int) -> dict:
    """Shared tick plan: an event, a gap violation, and an overflowed tick."""
    if step == 30:
        return {"events": ["pre_spawn"]}
    if step == 35:
        return {"events": ["rcl_up"]}
    if step == 40:
        return {"globals": {"actorOverflow": 1}}
    return {}


def configured_teacher_dir(tmp_path: Path) -> Path:
    """A teacher bundle whose room radii match the observation ABI.

    Collection refuses a teacher that outgrows the room slots, so a TI config
    needs readable bundle constants rather than a bare path.
    """
    directory = tmp_path / "ti-dist"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "main.js").write_text("".join(
        f"const {name} = {int(value)};\n"
        for name, value in SCHEMA["teacher"].items()
        if not name.startswith("_")
    ))
    return directory


def make_config(tmp_path: Path, **overrides) -> ts.CollectorConfig:
    settings = dict(
        teacher="scripted",
        num_envs=2,
        steps=120,
        max_episode=20_000,
        curricula=("alpha", "beta"),
        seed=201,
        per_stratum=2,
        min_gap=10,
        output=tmp_path / "out",
        progress_every=0,
    )
    settings.update(overrides)
    return ts.CollectorConfig(**settings)


# ---------------------------------------------------------------------------
# admission
# ---------------------------------------------------------------------------
def test_periodic_tag_and_min_gap():
    budget = ts.StratumBudget(per_stratum=8, min_gap=10)
    admitted = [
        step for step in range(1, 31)
        if budget.offer(
            env=0, curriculum="alpha", step=step, tick=step + 1_000, info={},
        )
    ]
    # Untagged ticks are not candidates at all, so they are never "rejections".
    assert admitted == [10, 20, 30]
    assert budget.rejected["gap"] == 0

    # An event five ticks after a capture is inside the per-env gap.
    assert budget.offer(
        env=0, curriculum="alpha", step=35, tick=1_035, info={"events": ["rcl_up"]},
    ) is None
    assert budget.rejected["gap"] == 1
    # A different environment has its own gap.
    assert budget.offer(
        env=1, curriculum="alpha", step=35, tick=1_035, info={"events": ["rcl_up"]},
    ) is not None


def test_per_stratum_cap_is_shared_across_envs():
    budget = ts.StratumBudget(per_stratum=2, min_gap=1)
    info = {"events": ["pre_spawn"]}
    admitted = [
        budget.offer(env=index % 2, curriculum="alpha", step=step, tick=step, info=info)
        for index, step in enumerate((11, 13, 15, 17))
    ]
    assert [bool(item) for item in admitted] == [True, True, False, False]
    assert budget.rejected["full"] == 2
    assert budget.counts[("alpha", "pre_spawn", "early")] == 2
    # The cap is per curriculum, so a second world still fills its own stratum.
    assert budget.offer(env=0, curriculum="beta", step=19, tick=19, info=info) is not None


def test_overflowed_states_are_rejected_and_counted():
    budget = ts.StratumBudget(per_stratum=4, min_gap=10)
    overflowing = {"events": ["pre_spawn"], "globals": {"roomOverflow": 1, "targetOverflow": 1}}
    assert budget.offer(
        env=0, curriculum="alpha", step=20, tick=1_020, info=overflowing,
    ) is None
    assert budget.rejected["overflow"] == 1
    assert budget.rejected["roomOverflow"] == 1
    assert budget.rejected["targetOverflow"] == 1
    assert budget.rejected["actorOverflow"] == 0
    # A rejected candidate consumes neither the stratum nor the capture gap.
    assert not budget.counts
    admitted = budget.offer(
        env=0, curriculum="alpha", step=21, tick=1_021, info={"events": ["pre_spawn"]},
    )
    assert admitted is not None
    assert admitted.relative_path.startswith("alpha/pre_spawn/early/teacher/")


def test_candidate_path_uses_reservoir_stratification():
    budget = ts.StratumBudget(per_stratum=1, min_gap=1)
    candidate = budget.offer(
        env=3,
        curriculum="seed_outpost",
        step=4_210,
        tick=4_211,
        info={"events": ["pre_spawn", "remote_at_source"]},
    )
    assert candidate is not None
    # The rarer `remote_at_source` outranks `pre_spawn` in the reservoir's event
    # priority; step 4210 falls in the mid phase.
    assert candidate.event == "remote_at_source"
    assert candidate.phase == "mid"
    assert candidate.relative_path == (
        "seed_outpost/remote_at_source/mid/teacher/t004211_e03_000001.xsnp"
    )


def test_concurrent_offers_respect_the_shared_cap():
    budget = ts.StratumBudget(per_stratum=5, min_gap=4)
    admitted: list[ts.Candidate] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def worker(env_index: int) -> None:
        start.wait()
        for step in range(1, 200):
            candidate = budget.offer(
                env=env_index, curriculum="alpha", step=step, tick=step,
                info={"events": ["pre_spawn"]},
            )
            if candidate is not None:
                with lock:
                    admitted.append(candidate)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # The cap is one atomic decision with the counter, so it can never be
    # exceeded, and every admitted candidate owns a unique file name.
    assert len(admitted) == sum(budget.counts.values())
    assert max(budget.counts.values()) == 5
    assert len({candidate.relative_path for candidate in admitted}) == len(admitted)
    for env_index in range(8):
        steps = sorted(
            candidate.step for candidate in admitted if candidate.env == env_index
        )
        assert all(later - earlier >= 4 for earlier, later in zip(steps, steps[1:]))


# ---------------------------------------------------------------------------
# collection end to end
# ---------------------------------------------------------------------------
def test_ti_collects_as_expert_and_verifies_as_learner(tmp_path, monkeypatch):
    envs = install_env(monkeypatch, info_for=plan_info)
    config = make_config(
        tmp_path, teacher="ti", curricula=("alpha",),
        bot_dir=str(configured_teacher_dir(tmp_path)),
    )
    result = ts.collect(config)

    collectors, verifiers = envs[:2], envs[2:]
    # TI collects through the expert step command; the snapshots must then reopen
    # in a plain learner session with no expert code loaded.
    assert [env.expert for env in collectors] == [True, True]
    assert all(
        env.kwargs["bot_dir"] == str(config.resolved_bot_dir) for env in collectors
    )
    assert [env.expert for env in verifiers] == [False]
    assert verifiers[0].kwargs["bot_dir"] is None
    assert len(verifiers[0].restored) == len(result.records)
    assert result.manifest["teacher"] == "ti"
    # One curriculum on two envs: the stratum cap is shared, not doubled.
    assert len(result.records) == 4
    assert result.histogram("event") == {"periodic": 2, "pre_spawn": 2}


def test_ti_collection_refuses_a_teacher_wider_than_the_room_slots(tmp_path, monkeypatch):
    """A teacher whose colony outgrows the observation cannot label states."""
    install_env(monkeypatch, info_for=plan_info)
    directory = configured_teacher_dir(tmp_path)
    (directory / "main.js").write_text("".join(
        f"const {name} = {int(value) + 4};\n"
        for name, value in SCHEMA["teacher"].items()
        if name == "maxScoutRoomDistance"
    ) + "".join(
        f"const {name} = {int(value)};\n"
        for name, value in SCHEMA["teacher"].items()
        if not name.startswith("_") and name != "maxScoutRoomDistance"
    ))
    with pytest.raises(ts.CollectorError, match="room slots"):
        make_config(
            tmp_path, teacher="ti", curricula=("alpha",), bot_dir=str(directory),
        )


def test_collect_writes_expected_manifest(tmp_path, monkeypatch):
    envs = install_env(monkeypatch, info_for=plan_info)
    result = ts.collect(make_config(tmp_path))

    # Two collection envs plus one verification session per curriculum.
    assert len(envs) == 4
    assert all(env.closed for env in envs)
    assert [env.expert for env in envs[:2]] == [False, False]
    assert [env.curriculum for env in envs[:2]] == ["alpha", "beta"]
    assert [env.seed for env in envs[:2]] == [201, 202]

    manifest = result.manifest
    assert set(manifest) == MANIFEST_KEYS
    assert manifest["format"] == 1
    assert manifest["kind"] == "teacher_start_states"
    assert manifest["teacher"] == "scripted"
    assert manifest["curricula"] == ["alpha", "beta"]
    assert manifest["seeds"] == [201, 202]
    assert manifest["source_sha256"] == "source-sha"
    assert manifest["collector_source_sha256"] == ts.collector_signature()

    records = manifest["records"]
    # Per env: periodic at steps 10 and 20 fill the cap, pre_spawn at 30 opens
    # its own stratum, tick 35 is inside the gap, tick 40 overflows, and every
    # later periodic tick finds a full stratum.
    assert len(records) == 6
    assert result.histogram("event") == {"periodic": 4, "pre_spawn": 2}
    assert result.histogram("phase") == {"early": 6}
    assert result.histogram("curriculum") == {"alpha": 3, "beta": 3}
    assert result.rejected["overflow"] == 2
    assert result.rejected["actorOverflow"] == 2
    assert result.rejected["gap"] == 2
    assert result.rejected["full"] == 16
    assert not result.dropped

    for record in records:
        assert set(record) == RECORD_KEYS
        assert not Path(record["path"]).is_absolute()
        payload = (result.directory / record["path"]).read_bytes()
        assert len(payload) == record["bytes"]
        assert record["tick"] == record["step"] + 1_000
        assert record["curriculum"] in ("alpha", "beta")
        assert record["seed"] in (201, 202)
        assert record["creeps"] == 5
        assert record["owned_rooms"] == 1
        assert record["remote_staffed"] == 1
        # A constant 2.0 harvest + 1.0 control economy: the reservoir consumes
        # this as an already bias-corrected per-tick rate, so it must be 3.0
        # even a handful of ticks into the EMA.
        assert record["skill_rate"] == pytest.approx(3.0)
    assert [record["path"] for record in records] == sorted(
        record["path"] for record in records
    )
    assert result.total_bytes == sum(record["bytes"] for record in records)

    on_disk = json.loads((result.directory / ts.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert result.directory.name == ts.content_address(
        manifest,
        {
            record["path"]: (result.directory / record["path"]).read_bytes()
            for record in records
        },
    )
    assert not [item for item in result.directory.parent.iterdir() if item != result.directory]


def test_collect_verifies_restores_and_drops_failures(tmp_path, monkeypatch):
    def restore_info(env, path):
        if path.name.endswith("000001.xsnp"):
            raise RuntimeError("snapshot is not restorable")
        info = _default_restore_info(env, path)
        if path.name.endswith("000002.xsnp"):
            info["globals"] = {"targetOverflow": 1}
        return info

    install_env(monkeypatch, info_for=plan_info, restore_info=restore_info)
    result = ts.collect(make_config(tmp_path))

    assert len(result.records) == 2
    assert len(result.dropped) == 4
    reasons = " ".join(reason for _record, reason in result.dropped)
    assert "restore failed" in reasons
    assert "overflows targetOverflow" in reasons
    for record, _reason in result.dropped:
        assert not (result.directory / record["path"]).exists()
    for record in result.records:
        assert (result.directory / record["path"]).is_file()
    # A dropped record must not leave its stratum directory behind: the
    # published tree would advertise strata the manifest does not contain.
    assert not [
        directory for directory in result.directory.rglob("*")
        if directory.is_dir() and not any(directory.iterdir())
    ]


def test_dead_verification_session_fails_the_run(tmp_path, monkeypatch):
    def restore_info(env, path):
        if env.restored[:-1]:
            # A dead Node session raises exactly like an unrestorable snapshot;
            # only the session probe tells them apart.
            env.dead = True
            raise RuntimeError("env server died: exit 1")
        return _default_restore_info(env, path)

    install_env(monkeypatch, info_for=plan_info, restore_info=restore_info)
    with pytest.raises(ts.CollectorError, match="verification session for .* died"):
        ts.collect(make_config(tmp_path))
    assert not list((tmp_path / "out").iterdir())


def test_restore_without_snapshot_tick_fails_the_run(tmp_path, monkeypatch):
    install_env(
        monkeypatch,
        info_for=plan_info,
        restore_info=lambda env, path: {"restored": True, "globals": {}},
    )
    with pytest.raises(ts.CollectorError, match="no snapshotTick"):
        ts.collect(make_config(tmp_path))
    assert not list((tmp_path / "out").iterdir())


def test_snapshot_bytes_must_match_the_file(tmp_path, monkeypatch):
    class LyingEnv(FakeEnv):
        def snapshot(self, path, events=()):
            descriptor = super().snapshot(path, events)
            descriptor["bytes"] += 1
            return descriptor

    install_env(monkeypatch, info_for=plan_info)
    monkeypatch.setattr(
        ts, "ScreepsEnv",
        lambda **kwargs: LyingEnv(
            info_for=plan_info,
            payload_for=lambda env, step: b"payload",
            restore_info=_default_restore_info,
            **kwargs,
        ),
    )
    with pytest.raises(ts.CollectorError, match="bytes on disk but the manifest records"):
        ts.collect(make_config(tmp_path))
    assert not list((tmp_path / "out").iterdir())



def test_collect_rejects_tick_mismatch(tmp_path, monkeypatch):
    def restore_info(env, path):
        info = _default_restore_info(env, path)
        info["snapshotTick"] += 1
        return info

    install_env(monkeypatch, info_for=plan_info, restore_info=restore_info)
    with pytest.raises(ts.CollectorError, match="failed learner restore"):
        ts.collect(make_config(tmp_path))


# ---------------------------------------------------------------------------
# content addressing
# ---------------------------------------------------------------------------
def test_content_address_ignores_created_and_tracks_bytes():
    manifest = {
        "format": 1,
        "kind": "teacher_start_states",
        "teacher": "ti",
        "collector_source_sha256": "a",
        "source_sha256": "b",
        "schema_sha256": "c",
        "curricula": ["alpha"],
        "seeds": [201],
        "records": [{"path": "alpha/x.xsnp", "tick": 7, "created": 1.0}],
    }
    files = {"alpha/x.xsnp": b"world"}
    baseline = ts.content_address(manifest, files)

    later = json.loads(json.dumps(manifest))
    later["records"][0]["created"] = 12345.678
    assert ts.content_address(later, files) == baseline

    assert ts.content_address(manifest, {"alpha/x.xsnp": b"world!"}) != baseline
    assert ts.content_address(manifest, {"alpha/y.xsnp": b"world"}) != baseline

    renamed = json.loads(json.dumps(manifest))
    renamed["teacher"] = "scripted"
    assert ts.content_address(renamed, files) != baseline


def test_content_address_streams_paths_identically(tmp_path):
    manifest = {"format": 1, "records": [{"path": "alpha/x.xsnp", "created": 1.0}]}
    payload = b"a large late-game world" * 1_000
    staged = tmp_path / "alpha" / "x.xsnp"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(payload)
    assert ts.content_address(manifest, {"alpha/x.xsnp": staged}) == ts.content_address(
        manifest, {"alpha/x.xsnp": payload},
    )


def test_identical_collections_publish_to_one_directory(tmp_path, monkeypatch):
    install_env(monkeypatch, info_for=plan_info)
    first = ts.collect(make_config(tmp_path))
    install_env(monkeypatch, info_for=plan_info)
    second = ts.collect(make_config(tmp_path))
    assert first.directory == second.directory
    assert [item.name for item in first.directory.parent.iterdir()] == [
        first.directory.name,
    ]

    install_env(
        monkeypatch,
        info_for=plan_info,
        payload_for=lambda env, step: f"different {env.curriculum} {step}".encode(),
    )
    third = ts.collect(make_config(tmp_path))
    assert third.directory != first.directory
    assert third.directory.parent == first.directory.parent
    assert len(list(first.directory.parent.iterdir())) == 2


def test_incomplete_content_address_is_never_reused(tmp_path, monkeypatch):
    install_env(monkeypatch, info_for=plan_info)
    first = ts.collect(make_config(tmp_path))
    victim = first.directory / first.records[0]["path"]
    victim.unlink()

    # The directory name still looks right, so a re-collection of identical
    # content must refuse it instead of deleting the good staged copy and
    # publishing a manifest that points at a missing snapshot.
    install_env(monkeypatch, info_for=plan_info)
    with pytest.raises(ts.CollectorError, match="is incomplete"):
        ts.collect(make_config(tmp_path))
    assert [item.name for item in first.directory.parent.iterdir()] == [
        first.directory.name,
    ]

    (first.directory / ts.MANIFEST_NAME).unlink()
    install_env(monkeypatch, info_for=plan_info)
    with pytest.raises(ts.CollectorError, match="holds no manifest.json"):
        ts.collect(make_config(tmp_path))


# ---------------------------------------------------------------------------
# reservoir import
# ---------------------------------------------------------------------------
def test_reservoir_imports_collected_manifest(tmp_path, monkeypatch):
    install_env(monkeypatch, info_for=plan_info)
    result = ts.collect(make_config(tmp_path))

    reservoir = StartStateReservoir(
        tmp_path / "reservoir", config=ReservoirConfig(per_stratum=1),
    )
    imported = import_teacher_snapshots(reservoir, result.directory / ts.MANIFEST_NAME)
    assert imported == len(result.records)
    assert reservoir.lane_size(LANE_TEACHER) == imported

    records = reservoir.records()
    assert {record.lane for record in records} == {LANE_TEACHER}
    assert {record.outcome for record in records} == {OUTCOME_TEACHER}
    # The teacher lane is external and immutable: the per-stratum cap of the
    # reservoir must not evict any of it.
    assert reservoir.event_counts(LANE_TEACHER) == {"periodic": 4, "pre_spawn": 2}
    assert reservoir.phase_counts(LANE_TEACHER) == {"early": 6}
    for record in records:
        assert Path(record.path).is_file()
        assert Path(record.path).is_absolute()
        assert record.update == -1
        assert record.events
        assert record.curriculum in ("alpha", "beta")

    # Re-importing the same immutable set is a no-op rather than a duplication.
    assert import_teacher_snapshots(reservoir, result.directory / ts.MANIFEST_NAME) == 0
    assert reservoir.lane_size(LANE_TEACHER) == imported


# ---------------------------------------------------------------------------
# loud failures
# ---------------------------------------------------------------------------
def test_no_admitted_states_fails(tmp_path, monkeypatch):
    install_env(monkeypatch, info_for=lambda env, step: {})
    config = make_config(tmp_path, steps=5, min_gap=10)
    with pytest.raises(ts.CollectorError, match="no start states were admitted"):
        ts.collect(config)
    assert not list((tmp_path / "out").iterdir())


def test_curriculum_without_records_fails(tmp_path, monkeypatch):
    def info_for(env: FakeEnv, step: int) -> dict:
        # Every `beta` state exceeds the observation capacity, so that world
        # contributes nothing the learner could ever be restored into.
        if env.curriculum == "beta":
            return {"events": ["pre_spawn"], "globals": {"actorOverflow": 1}}
        return {"events": ["pre_spawn"]}

    install_env(monkeypatch, info_for=info_for)
    config = make_config(tmp_path, steps=25, min_gap=10)
    with pytest.raises(ts.CollectorError, match=r"\['beta'\] produced no usable"):
        ts.collect(config)
    assert not list((tmp_path / "out").iterdir())


def test_dead_environment_fails_the_run(tmp_path, monkeypatch):
    def info_for(env: FakeEnv, step: int) -> dict:
        if env.curriculum == "beta" and step == 25:
            raise BrokenPipeError("env process exited")
        return plan_info(env, step)

    install_env(monkeypatch, info_for=info_for)
    with pytest.raises(ts.CollectorError, match="teacher env 1 .*failed after 24 ticks"):
        ts.collect(make_config(tmp_path))
    assert not list((tmp_path / "out").iterdir())


def test_a_dying_environment_aborts_its_siblings(tmp_path, monkeypatch):
    def info_for(env: FakeEnv, step: int) -> dict:
        if env.curriculum == "beta":
            if step == 3:
                raise BrokenPipeError("env process exited")
            return {}
        time.sleep(0.002)
        return plan_info(env, step)

    envs = install_env(monkeypatch, info_for=info_for)
    with pytest.raises(ts.CollectorError, match="teacher env 1"):
        ts.collect(make_config(tmp_path, steps=1_000))
    # Without the shared stop signal the survivor would burn all 1000 ticks
    # before the run could abort — hours on a real TI lifecycle.
    survivor = next(env for env in envs if env.curriculum == "alpha")
    assert survivor.step_index < 1_000
    assert all(env.closed for env in envs)


def test_configuration_is_validated(tmp_path):
    with pytest.raises(ts.CollectorError, match="cannot cover 2 curricula"):
        make_config(tmp_path, num_envs=1)
    with pytest.raises(ts.CollectorError, match="exceeds --max-episode"):
        make_config(tmp_path, steps=30_000, max_episode=20_000)
    with pytest.raises(ts.CollectorError, match="unknown teacher"):
        make_config(tmp_path, teacher="human")
    with pytest.raises(ts.CollectorError, match="--min-gap"):
        make_config(tmp_path, min_gap=0)
    with pytest.raises(ts.CollectorError, match="not plain scenario names"):
        make_config(tmp_path, curricula=("../escaped", "alpha"))
    with pytest.raises(ts.CollectorError, match="not plain scenario names"):
        make_config(tmp_path, curricula=("alpha", "sub/dir"))


def test_cli_defaults_match_the_documented_contract(tmp_path):
    config = ts.config_from_args(ts.parse_args([]))
    assert config.teacher == "ti"
    assert config.num_envs == 4
    assert config.steps == 20_000
    assert config.max_episode == 20_000
    assert config.curricula == ("empty", "seed_outpost")
    assert config.seed == 201
    assert config.per_stratum == 8
    assert config.min_gap == 64
    assert config.output == ts.DEFAULT_OUTPUT
    assert config.expert is True
    assert config.seeds() == (201, 202, 203, 204)
    assert config.assigned_curricula() == ("empty", "seed_outpost")

    scripted = ts.config_from_args(
        ts.parse_args(["--teacher", "scripted", "--curriculum", "seed_outpost , empty"]),
    )
    assert scripted.expert is False
    assert scripted.curricula == ("seed_outpost", "empty")


def test_main_prints_the_immutable_path_last(tmp_path, monkeypatch, capsys):
    install_env(monkeypatch, info_for=plan_info)
    code = ts.main([
        "--teacher", "scripted", "--num-envs", "2", "--steps", "120",
        "--curriculum", "alpha,beta", "--per-stratum", "2", "--min-gap", "10",
        "--output", str(tmp_path / "out"), "--progress-every", "0",
    ])
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    published = Path(lines[-1])
    assert published.is_dir()
    assert (published / ts.MANIFEST_NAME).is_file()
    assert published.parent == tmp_path / "out"


def test_main_reports_failure_with_nonzero_exit(tmp_path, monkeypatch, capsys):
    install_env(monkeypatch, info_for=lambda env, step: {})
    code = ts.main([
        "--teacher", "scripted", "--num-envs", "1", "--steps", "5",
        "--curriculum", "alpha", "--min-gap", "10",
        "--output", str(tmp_path / "out"), "--progress-every", "0",
    ])
    assert code == 2
    captured = capsys.readouterr()
    assert "no start states were admitted" in captured.err
    assert not list((tmp_path / "out").iterdir())
