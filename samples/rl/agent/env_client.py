"""Framed-binary client for samples/rl/env/server.mjs (JSONL debug optional)."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import (
    ACTOR_FEAT,
    CONSTRUCTION_MASK_BYTES,
    INTENT_SLOTS,
    MAX_ACTORS,
    MAX_BODY_PARTS,
    MAX_ROOMS,
    MAX_TARGETS,
    N_AMOUNT,
    N_BODY_PART,
    N_CONSTRUCTION_TILE,
    N_CONSTRUCTION_TYPE,
    N_DIR,
    N_INTENT,
    PATCH_FLAT,
    PATCHES_PER_ROOM,
    SCHEMA,
    SCHEMA_PATH,
    TARGET_FEAT,
    TILE_FEAT,
)

_ROOT = Path(__file__).resolve().parents[3]

_COMMAND_MAGIC = b"XAC1"
_COMMAND_VERSION = 6
_RESPONSE_VERSION = 4
_COMMAND_HEADER = struct.Struct("<4sBBHIHBB")
_COMMAND_OPCODES = {
    "reset": 1,
    "reset_expert": 2,
    "step": 3,
    "step_scripted": 4,
    "step_expert": 5,
    "schema": 6,
    "close": 7,
    "step_labeled": 8,
    "snapshot": 9,
    "restore": 10,
}


def _teacher_action_payload_bytes(rows: int, slots: int) -> int:
    if not (1 <= rows <= MAX_ACTORS):
        raise RuntimeError(f"binary teacher-action rows={rows} outside [1,{MAX_ACTORS}]")
    if slots != INTENT_SLOTS:
        raise RuntimeError(
            f"binary teacher-action slots={slots} differs from schema={INTENT_SLOTS}"
        )
    cells = rows * slots
    return cells * (5 + 2 + 2 * N_BODY_PART)


def _observation_payload_bytes(shapes: Any) -> int:
    """Validate an XRL shape descriptor and return its exact packed byte size."""
    if not isinstance(shapes, dict):
        raise RuntimeError("binary observation shapes must be an object")

    def elements(name: str) -> int:
        shape = shapes.get(name)
        if not isinstance(shape, list) or not shape:
            raise RuntimeError(f"binary observation shape {name!r} is missing")
        product = 1
        for extent in shape:
            if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
                raise RuntimeError(
                    f"binary observation shape {name!r} has invalid extent {extent!r}"
                )
            product *= extent
        return product

    float_bytes = 4 * sum(elements(name) for name in (
        "actors", "targets", "roomCoords",
    ))
    packed_bytes = sum(elements(name) for name in (
        "patches", "intentMask", "dirMask", "targetSelectMask",
        "amountMask", "constructionMask",
    ))
    # These masks are fixed-capacity planes in the canonical packed order.
    fixed_bytes = MAX_ROOMS + 2 * MAX_ACTORS + MAX_TARGETS
    return float_bytes + packed_bytes + fixed_bytes


def _decode_teacher_actions(
    payload: bytes | bytearray | memoryview,
    *,
    rows: int,
    slots: int,
) -> dict[str, torch.Tensor]:
    """Decode response teacher planes in the same order as XAC1 step actions."""
    expected = _teacher_action_payload_bytes(rows, slots)
    view = memoryview(payload)
    if len(view) != expected:
        raise RuntimeError(
            f"binary teacher-action payload length={len(view)} expected={expected}"
        )
    cells = rows * slots
    offset = 0

    def take_u8(count: int, shape: tuple[int, ...]) -> np.ndarray:
        nonlocal offset
        result = np.frombuffer(view[offset : offset + count], dtype=np.uint8).copy()
        offset += count
        return result.reshape(shape)

    scalar_names = (
        "types", "dirs", "targets", "amounts", "construction_types",
    )
    scalar_limits = (N_INTENT, N_DIR, MAX_TARGETS, N_AMOUNT, N_CONSTRUCTION_TYPE)
    arrays: dict[str, np.ndarray] = {}
    for name, upper in zip(scalar_names, scalar_limits):
        plane = take_u8(cells, (rows, slots))
        if plane.size and int(plane.max()) >= upper:
            raise RuntimeError(f"binary teacher actions.{name} contains value >= {upper}")
        arrays[name] = plane

    tile_bytes = cells * 2
    tiles = np.frombuffer(view[offset : offset + tile_bytes], dtype="<u2").copy()
    offset += tile_bytes
    tiles = tiles.reshape(rows, slots)
    if tiles.size and int(tiles.max()) >= N_CONSTRUCTION_TILE:
        raise RuntimeError(
            "binary teacher actions.construction_tiles contains out-of-range tile"
        )
    arrays["construction_tiles"] = tiles
    body_shape = (rows, slots, N_BODY_PART)
    body_count = rows * slots * N_BODY_PART
    body_counts = take_u8(body_count, body_shape)
    body_order = take_u8(body_count, body_shape)
    if body_counts.size and int(body_counts.max()) > MAX_BODY_PARTS:
        raise RuntimeError(
            f"binary teacher actions.body_counts contains value > {MAX_BODY_PARTS}"
        )
    body_totals = body_counts.sum(axis=-1, dtype=np.uint16)
    if body_totals.size and int(body_totals.max()) > MAX_BODY_PARTS:
        raise RuntimeError(
            f"binary teacher body contains more than {MAX_BODY_PARTS} parts"
        )
    spawn_type = SCHEMA["intentTypes"].index("spawnCreep")
    if np.any((arrays["types"] == spawn_type) & (body_totals < 1)):
        raise RuntimeError("binary teacher spawnCreep action has an empty body")
    try:
        _validate_body_planes(body_counts, body_order, (rows, slots))
    except ValueError as error:
        raise RuntimeError(f"invalid binary teacher body planes: {error}") from error
    arrays["body_counts"] = body_counts
    arrays["body_order"] = body_order
    if offset != expected:
        raise RuntimeError(
            f"binary teacher-action decoder consumed={offset} expected={expected}"
        )
    return {
        name: torch.from_numpy(array).to(dtype=torch.long)
        for name, array in arrays.items()
    }


def _teacher_actions_from_response(
    data: dict[str, Any],
    info: dict[str, Any],
    *,
    binary: bool,
    command: str,
) -> dict[str, torch.Tensor]:
    """Return strictly validated compact teacher planes for either wire format."""
    if binary:
        actions = data.get("teacher_actions")
        if not isinstance(actions, dict):
            raise RuntimeError(f"binary {command} response lacks teacher action planes")
        return actions

    wire = info.get("actions")
    if not isinstance(wire, dict):
        raise RuntimeError(f"JSON {command} response lacks teacher action planes")
    try:
        scalar_wire_names = {
            "types": "types",
            "dirs": "dirs",
            "targets": "targets",
            "amounts": "amounts",
            "construction_types": "constructionTypes",
        }
        arrays = {
            name: _action_u8(wire.get(wire_name), name)
            for name, wire_name in scalar_wire_names.items()
        }
        shape = arrays["types"].shape
        if any(plane.shape != shape for plane in arrays.values()):
            raise ValueError(
                f"teacher action plane shapes differ: {[p.shape for p in arrays.values()]}"
            )
        rows, slots = shape
        if not 1 <= rows <= MAX_ACTORS:
            raise ValueError(f"teacher action rows={rows} outside [1, {MAX_ACTORS}]")
        if slots != INTENT_SLOTS:
            raise ValueError(
                f"teacher action slots={slots} differs from schema={INTENT_SLOTS}"
            )
        construction_tiles = _action_u16(
            wire.get("constructionTiles"), "construction_tiles", N_CONSTRUCTION_TILE,
        )
        if construction_tiles.shape != shape:
            raise ValueError(
                "teacher actions.construction_tiles shape "
                f"{construction_tiles.shape} differs from {shape}"
            )
        body_counts = _body_vector_u8(
            wire.get("bodyCounts"), "body_counts", MAX_BODY_PARTS + 1,
        )
        body_order = _body_vector_u8(
            wire.get("bodyOrder"), "body_order", N_BODY_PART,
        )
        _validate_body_planes(body_counts, body_order, shape)
        body_totals = body_counts.sum(axis=-1, dtype=np.uint16)
        if body_totals.size and int(body_totals.max()) > MAX_BODY_PARTS:
            raise ValueError(f"teacher body contains more than {MAX_BODY_PARTS} parts")
        spawn_type = SCHEMA["intentTypes"].index("spawnCreep")
        if np.any((arrays["types"] == spawn_type) & (body_totals < 1)):
            raise ValueError("teacher spawnCreep action has an empty body")
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid JSON {command} teacher actions: {error}") from error

    arrays["construction_tiles"] = construction_tiles
    arrays["body_counts"] = body_counts
    arrays["body_order"] = body_order
    return {
        name: torch.from_numpy(array).to(dtype=torch.long)
        for name, array in arrays.items()
    }


class ScreepsEnv:
    def __init__(
        self,
        node: str | None = None,
        room: str = "W7N3",
        max_episode: int = 2000,
        device: str | torch.device = "cpu",
        expert: bool = False,
        bot_dir: str | None = None,
        *,
        headful: bool = False,
        headful_password: str = "rlwatch",
        tick_ms: int | None = None,
        no_open: bool = False,
        curriculum: str | None = None,
        lean_meta: bool | None = None,
        command_format: str | None = None,
        capture_expert_intents: bool = False,
        seed: int | None = None,
    ):
        self.device = torch.device(device)
        self.expert = expert
        self.room = room
        self.max_episode = max_episode
        self.node = node
        self.headful = bool(headful)
        self.headful_password = headful_password
        self.tick_ms = tick_ms
        self.no_open = bool(no_open)
        self.curriculum = curriculum or os.environ.get("RL_CURRICULUM", "empty")
        self.lean_meta = (
            os.environ.get("RL_LEAN_META", "1") != "0"
            if lean_meta is None else bool(lean_meta)
        )
        # _ROOT = xxscreeps repo; The International lives as a sibling checkout
        default_ti = (_ROOT.parent / "The-International-Open-Source" / "dist").resolve()
        self.bot_dir = bot_dir or os.environ.get("RL_EXPERT_BOT", str(default_ti))
        # Only require TI dist when running expert mode (scripted baseline does not need it).
        if self.expert and not Path(self.bot_dir).is_dir():
            raise FileNotFoundError(
                f"expert bot dir missing: {self.bot_dir} "
                f"(build TI: cd ../The-International-Open-Source && npm run build)"
            )
        self._lock = threading.Lock()
        self.command_format = command_format or os.environ.get("RL_CMD_FMT", "bin")
        self.capture_expert_intents = bool(capture_expert_intents)
        self.seed = int(os.environ.get("RL_SEED", "0") if seed is None else seed)
        if self.command_format not in ("bin", "json"):
            raise ValueError(
                f"RL_CMD_FMT={self.command_format!r} unsupported; use bin|json"
            )
        self.obs_fmt = os.environ.get("RL_OBS_FMT", "bin")
        self._bin = self.obs_fmt == "bin"
        if self.obs_fmt not in ("bin", "pack", "b64", "json"):
            raise ValueError(f"RL_OBS_FMT={self.obs_fmt!r} unsupported; use bin|pack|b64|json")
        if self.obs_fmt != "bin":
            # Train path is bin-only for SPS/desync safety; other formats are debug.
            print(
                f"[env] WARNING: RL_OBS_FMT={self.obs_fmt!r} (default is bin). "
                "Non-bin formats are debug-only and desync-prone.",
                flush=True,
            )
        self._spawn_server()
        self.last_info: dict[str, Any] = {}

    def _spawn_server(self) -> None:
        env = os.environ.copy()
        env["RL_ROOM"] = self.room
        env["RL_MAX_EPISODE"] = str(self.max_episode)
        env["RL_CURRICULUM"] = self.curriculum
        env["RL_CMD_FMT"] = self.command_format
        # bin = raw frames (default); pack = single base64 blob; b64 = per-field; json = debug
        env.setdefault("RL_OBS_FMT", self.obs_fmt)
        # Train and evaluation share the same navigation dynamics by default.
        env["RL_LEAN_META"] = "1" if self.lean_meta else "0"
        env["RL_EXPERT_INTENTS"] = "1" if self.capture_expert_intents else "0"
        env["RL_SEED"] = str(self.seed)
        env.setdefault("RL_NAV", "pathfinder")
        if self.bot_dir:
            env["RL_EXPERT_BOT"] = self.bot_dir
        # Per-env headful (only one train env should enable this — port 21025)
        if self.headful:
            env["RL_HEADFUL"] = "1"
            env["RL_HEADFUL_PASSWORD"] = self.headful_password
            if self.tick_ms is not None:
                env["RL_TICK_MS"] = str(self.tick_ms)
            elif "RL_TICK_MS" not in env:
                env["RL_TICK_MS"] = "100"
            if self.no_open:
                env["RL_NO_OPEN"] = "1"
        else:
            # Do not inherit a parent RL_HEADFUL / tick delay (would slow every env)
            env.pop("RL_HEADFUL", None)
            env.pop("RL_TICK_MS", None)
            for key in ("RL_HEADFUL_PASSWORD", "RL_HEADFUL_BIND", "RL_NO_OPEN"):
                if key in os.environ:
                    env[key] = os.environ[key]
        node_bin = self.node or os.environ.get("RL_NODE", "node")
        server = Path(__file__).resolve().parents[1] / "env" / "server.mjs"
        cmd = [node_bin, "--import", "xxscreeps/loader", str(server)]
        try:
            ver = subprocess.check_output([node_bin, "-v"], text=True).strip().lstrip("v")
            major = int(ver.split(".")[0])
        except Exception:
            major = 0
        if major < 22:
            cmd = ["mise", "exec", "node@24", "--", "node", "--import", "xxscreeps/loader", str(server)]
        # Keep both pipes byte-oriented: command and observation formats are
        # independent (for example binary commands with JSON debug responses).
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_ROOT),
            env=env,
            text=False,
            bufsize=0,
        )
        assert self.proc.stdin and self.proc.stdout
        self._stdin = self.proc.stdin
        self._stdout = self.proc.stdout
        # Drain stderr so a chatty Node process cannot fill the pipe and deadlock.
        self._stderr_tail: list[str] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="screeps-env-stderr", daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        try:
            err = self.proc.stderr
            if err is None:
                return
            while True:
                line = err.readline()
                if not line:
                    break
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                self._stderr_tail.append(line.rstrip())
                if len(self._stderr_tail) > 80:
                    self._stderr_tail = self._stderr_tail[-40:]
        except Exception:
            pass

    def _read_exact(self, n: int) -> bytes:
        """Read exactly n bytes from binary stdout."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self._stdout.read(n - len(buf))
            if not chunk:
                tail = "\n".join(self._stderr_tail[-20:]) if self._stderr_tail else ""
                raise RuntimeError(f"env server died mid-frame: {tail[:800]}")
            if isinstance(chunk, str):
                chunk = chunk.encode("latin-1")
            buf.extend(chunk)
        return bytes(buf)

    def _read_bin_frame(self) -> dict[str, Any]:
        """Parse XRL1 length-prefixed response (see docs/PERFORMANCE.md).

        Hard-fail on bad magic (no half-resync — leftover bugs desync permanently).
        Server bin mode must keep stdout frame-only (console.log → stderr).
        """
        hdr = self._read_exact(16)
        if hdr[0:4] != b"XRL1":
            tail = "\n".join(self._stderr_tail[-15:]) if self._stderr_tail else ""
            raise RuntimeError(
                f"bad bin magic {hdr[0:4]!r} (stdout pollution?). stderr:\n{tail[:600]}"
            )
        if hdr[4] != _RESPONSE_VERSION:
            raise RuntimeError(f"unsupported binary protocol version {hdr[4]}")
        wire_schema = int.from_bytes(hdr[6:8], "little")
        if wire_schema != int(SCHEMA["version"]):
            raise RuntimeError(
                f"environment schema={wire_schema} does not match client "
                f"schema={SCHEMA['version']}"
            )
        flags = hdr[5]
        if flags & ~0x0F:
            raise RuntimeError(f"unsupported binary response flags {flags:#x}")
        meta_len = int.from_bytes(hdr[8:12], "little")
        blob_len = int.from_bytes(hdr[12:16], "little")
        # Sanity caps (local trust, avoid OOM on corrupt length)
        if meta_len > 16 * 1024 * 1024 or blob_len > 64 * 1024 * 1024:
            raise RuntimeError(f"bin frame size absurd meta={meta_len} blob={blob_len}")
        meta_buf = self._read_exact(meta_len)
        blob = self._read_exact(blob_len) if blob_len else b""
        meta = json.loads(meta_buf.decode("utf-8"))
        ok_flag = bool(flags & 1)
        done_flag = bool(flags & 2)
        obs_flag = bool(flags & 4)
        if not isinstance(meta.get("ok"), bool) or meta["ok"] != ok_flag:
            raise RuntimeError("binary ok flag and metadata disagree")
        if not isinstance(meta.get("done"), bool) or meta["done"] != done_flag:
            raise RuntimeError("binary done flag and metadata disagree")
        if not ok_flag:
            raise RuntimeError(meta.get("error") or f"env error: {meta}")
        # Normalize to the same shape as JSONL responses
        out: dict[str, Any] = {
            "ok": True,
            "reward": float(meta.get("reward", 0)),
            "done": bool(meta.get("done", False)),
            "info": meta.get("info") or {},
        }
        teacher_descriptor = meta.get("teacherActions")
        has_teacher = bool(flags & 8)
        if has_teacher != (teacher_descriptor is not None):
            raise RuntimeError(
                "binary teacher-action flag and metadata descriptor disagree"
            )
        obs_blob: bytes | memoryview = blob
        if has_teacher:
            if not isinstance(teacher_descriptor, dict):
                raise RuntimeError("binary teacher-action descriptor must be an object")
            try:
                rows = int(teacher_descriptor["rows"])
                slots = int(teacher_descriptor["slots"])
                action_bytes = int(teacher_descriptor["byteLength"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("malformed binary teacher-action descriptor") from error
            expected = _teacher_action_payload_bytes(rows, slots)
            if action_bytes != expected or action_bytes > len(blob):
                raise RuntimeError(
                    "binary teacher-action payload length "
                    f"descriptor={action_bytes} expected={expected} blob={len(blob)}"
                )
            split = len(blob) - action_bytes
            view = memoryview(blob)
            obs_blob = view[:split]
            out["teacher_actions"] = _decode_teacher_actions(
                view[split:], rows=rows, slots=slots,
            )
        has_shapes = "shapes" in meta
        if obs_flag != has_shapes:
            raise RuntimeError("binary observation flag and metadata shapes disagree")
        if obs_flag:
            expected_obs_bytes = _observation_payload_bytes(meta["shapes"])
            if len(obs_blob) != expected_obs_bytes:
                raise RuntimeError(
                    "binary observation payload length "
                    f"actual={len(obs_blob)} expected={expected_obs_bytes}"
                )
        elif len(obs_blob):
            raise RuntimeError("binary frame has observation bytes without observation flag")
        if "schema" in meta:
            out["schema"] = meta["schema"]
        if obs_flag:
            out["obs"] = {
                "encoding": "bin",
                "blob": obs_blob,  # raw bytes, not base64
                "shapes": meta.get("shapes"),
                "globals": meta.get("globals") or {},
                "actorMeta": meta.get("actorMeta") or [],
                "targetMeta": meta.get("targetMeta") or [],
                "time": meta.get("time", 0),
                "roomNames": meta.get("roomNames") or [],
                "roomsUsed": meta.get("roomsUsed", 1),
            }
        return out

    def _rpc(self, msg: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            try:
                if self.command_format == "bin":
                    payload = _encode_binary_command(msg)
                else:
                    payload = (json.dumps(_json_command(msg)) + "\n").encode("utf-8")
                self._stdin.write(payload)
                self._stdin.flush()
            except BrokenPipeError as err:
                raise RuntimeError("env server pipe broken") from err
            if self._bin:
                return self._read_bin_frame()
            while True:
                line = self._stdout.readline()
                if not line:
                    tail = "\n".join(self._stderr_tail[-20:]) if self._stderr_tail else ""
                    raise RuntimeError(f"env server died: {tail[:800]}")
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                # Ignore engine noise (e.g. controller upgrade notify) leaked on stdout
                if not isinstance(data, dict) or "ok" not in data:
                    continue
                if not data.get("ok", False):
                    raise RuntimeError(data.get("error") or f"env error: {data}")
                return data

    def hard_reset(self) -> dict[str, torch.Tensor]:
        """Kill Node process and start a fresh expert/rl episode."""
        with self._lock:
            try:
                self.proc.kill()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
            self._close_pipes()
            self._spawn_server()
        return self.reset()

    def reset(self) -> dict[str, torch.Tensor]:
        if self.expert:
            data = self._rpc({"cmd": "reset_expert", "botDir": self.bot_dir})
        else:
            data = self._rpc({"cmd": "reset"})
        self.last_info = data.get("info") or {}
        return self._obs_to_batch(data["obs"])

    def snapshot(
        self, path: str | Path, events: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Persist the exact post-tick world this env just observed.

        The environment writes the file itself, so a large world never crosses
        the command pipe. Returns the snapshot descriptor (tick, step, bytes,
        rooms, event tags) for reservoir bookkeeping.
        """
        data = self._rpc({"cmd": "snapshot", "path": str(path), "events": list(events)})
        descriptor = (data.get("info") or {}).get("snapshot")
        if not isinstance(descriptor, dict):
            raise RuntimeError("env snapshot response lacks a descriptor")
        return descriptor

    def restore(self, path: str | Path) -> dict[str, torch.Tensor]:
        """Start a new episode segment from a captured world state."""
        if self.expert:
            raise RuntimeError("restore rebuilds a learner session; expert must reset")
        data = self._rpc({"cmd": "restore", "path": str(path)})
        self.last_info = data.get("info") or {}
        if "obs" not in data:
            raise RuntimeError("env restore response lacks an observation")
        return self._obs_to_batch(data["obs"])

    def step(self, actions: dict[str, torch.Tensor | np.ndarray | list] | None = None) -> tuple[
        dict[str, torch.Tensor], float, bool, dict[str, Any]
    ]:
        if self.expert:
            data = self._rpc({"cmd": "step_expert"})
        else:
            assert actions is not None
            data = self._rpc({"cmd": "step", "actions": actions})
        self.last_info = data.get("info") or {}
        if "obs" not in data or "reward" not in data:
            raise RuntimeError(f"malformed step response keys={list(data.keys())}")
        obs = self._obs_to_batch(data["obs"])
        return obs, float(data["reward"]), bool(data.get("done", False)), self.last_info

    def step_scripted(self) -> tuple[
        dict[str, torch.Tensor],
        float,
        bool,
        dict[str, Any],
        dict[str, torch.Tensor],
    ]:
        """RCL1 scripted baseline (same action interface). Returns actions for BC."""
        if self.expert:
            raise RuntimeError("step_scripted is for non-expert sessions")
        data = self._rpc({"cmd": "step_scripted"})
        self.last_info = data.get("info") or {}
        if "obs" not in data or "reward" not in data:
            raise RuntimeError(f"malformed step_scripted response keys={list(data.keys())}")
        obs = self._obs_to_batch(data["obs"])
        actions = _teacher_actions_from_response(
            data, self.last_info, binary=self._bin, command="step_scripted",
        )
        return (
            obs,
            float(data["reward"]),
            bool(data.get("done", False)),
            self.last_info,
            actions,
        )

    def step_labeled(
        self,
        actions: dict[str, torch.Tensor | np.ndarray | list],
    ) -> tuple[
        dict[str, torch.Tensor],
        float,
        bool,
        dict[str, Any],
        dict[str, torch.Tensor],
    ]:
        """Apply learner actions and return a scripted label for the pre-state."""
        if self.expert:
            raise RuntimeError("step_labeled is for non-expert sessions")
        data = self._rpc({"cmd": "step_labeled", "actions": actions})
        self.last_info = data.get("info") or {}
        if "obs" not in data or "reward" not in data:
            raise RuntimeError(f"malformed step_labeled response keys={list(data.keys())}")
        teacher = _teacher_actions_from_response(
            data, self.last_info, binary=self._bin, command="step_labeled",
        )
        return (
            self._obs_to_batch(data["obs"]),
            float(data["reward"]),
            bool(data.get("done", False)),
            self.last_info,
            teacher,
        )

    def close(self) -> None:
        try:
            self._rpc({"cmd": "close"})
        except Exception:
            pass
        try:
            self.proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        finally:
            self._close_pipes()

    def _close_pipes(self) -> None:
        """Close process pipe descriptors after exit so repeated envs do not leak."""
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        thread = getattr(self, "_stderr_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)

    def _obs_to_batch(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Single-env batch dim = 1. Supports encoding=json|b64|pack (RL_OBS_FMT).

        Masks stay uint8 on host (4× less RAM / cat bandwidth); VecScreepsEnv
        promotes them to float32 on the single bulk H2D. Float tensors stay f32.
        """
        import base64

        shapes = obs["shapes"]
        enc = obs.get("encoding") or "json"

        # --- single-blob packing (pack=base64, bin=raw bytes) ---
        if enc in ("pack", "bin"):
            raw_blob = obs["blob"]
            if enc == "pack":
                raw = base64.b64decode(raw_blob)
            else:
                # bin: already bytes
                raw = raw_blob if isinstance(raw_blob, (bytes, bytearray, memoryview)) else base64.b64decode(raw_blob)
            view = memoryview(raw)
            off = 0

            def take_f32(n_elem: int) -> np.ndarray:
                nonlocal off
                nbytes = n_elem * 4
                # Single copy into owned C-contiguous array (pipe buffer is transient).
                arr = np.empty(n_elem, dtype=np.float32)
                arr[:] = np.frombuffer(view[off : off + nbytes], dtype=np.float32)
                off += nbytes
                return arr

            def take_u8(n_elem: int) -> np.ndarray:
                nonlocal off
                arr = np.empty(n_elem, dtype=np.uint8)
                arr[:] = np.frombuffer(view[off : off + n_elem], dtype=np.uint8)
                off += n_elem
                return arr

            sp = shapes["patches"]
            sa = shapes["actors"]
            st = shapes["targets"]
            src = shapes["roomCoords"]
            si = shapes["intentMask"]
            sd = shapes["dirMask"]
            sts = shapes["targetSelectMask"]
            sam = shapes["amountMask"]
            scm = shapes["constructionMask"]
            n_patch = int(np.prod(sp))
            n_act = int(np.prod(sa))
            n_tgt = int(np.prod(st))
            patches = take_u8(n_patch).reshape(sp)
            actors = take_f32(n_act).reshape(sa)
            targets = take_f32(n_tgt).reshape(st)
            room_coords = take_f32(int(np.prod(src))).reshape(src)
            # roomMask length = maxRooms (schema); active rooms may be fewer in mask bits
            room_mask = take_u8(MAX_ROOMS)
            actor_mask = take_u8(MAX_ACTORS)
            actor_outcome = take_u8(MAX_ACTORS)
            target_mask = take_u8(MAX_TARGETS)
            intent_mask = take_u8(int(np.prod(si))).reshape(si)
            dir_mask = take_u8(int(np.prod(sd))).reshape(sd)
            target_select = take_u8(int(np.prod(sts))).reshape(sts)
            amount_mask = take_u8(int(np.prod(sam))).reshape(sam)
            construction_mask = take_u8(int(np.prod(scm))).reshape(scm)
            if off != len(raw):
                raise RuntimeError(
                    f"observation blob length mismatch: consumed={off} actual={len(raw)}"
                )
        else:

            def _f32(key: str, shape: list[int]) -> np.ndarray:
                raw = obs[key]
                if enc == "b64" and isinstance(raw, str):
                    buf = base64.b64decode(raw)
                    arr = np.frombuffer(buf, dtype=np.float32).copy()
                else:
                    arr = np.asarray(raw, dtype=np.float32)
                return arr.reshape(shape)

            def _u8(key: str, shape: list[int] | None = None) -> np.ndarray:
                raw = obs[key]
                if enc == "b64" and isinstance(raw, str):
                    buf = base64.b64decode(raw)
                    arr = np.frombuffer(buf, dtype=np.uint8).copy()
                else:
                    arr = np.asarray(raw, dtype=np.uint8)
                if shape is not None:
                    arr = arr.reshape(shape)
                return arr

            patches = _u8("patches", shapes["patches"])
            actors = _f32("actors", shapes["actors"])
            targets = _f32("targets", shapes["targets"])
            room_coords = _f32("roomCoords", shapes["roomCoords"])
            intent_mask = _u8("intentMask", shapes["intentMask"])
            dir_mask = _u8("dirMask", shapes["dirMask"])
            target_select = _u8("targetSelectMask", shapes["targetSelectMask"])
            amount_mask = _u8("amountMask", shapes["amountMask"])
            construction_mask = _u8("constructionMask", shapes["constructionMask"])
            room_mask = _u8("roomMask")
            actor_mask = _u8("actorMask")
            actor_outcome = _u8("actorOutcome", shapes["actorOutcome"])
            target_mask = _u8("targetMask")

        # Pad rooms to MAX_ROOMS for stable stacking across envs / torch.compile
        r_got = patches.shape[0]
        if r_got < MAX_ROOMS:
            pad = np.zeros((MAX_ROOMS - r_got, *patches.shape[1:]), dtype=patches.dtype)
            patches = np.concatenate([patches, pad], axis=0)
        if room_mask.size < MAX_ROOMS:
            rm = np.zeros(MAX_ROOMS, dtype=np.uint8)
            rm[: room_mask.size] = room_mask
            room_mask = rm
        g = obs["globals"]
        globals_ = np.asarray(
            [
                g.get("rclMax", 0) / 8,
                min(g.get("storedEnergy", 0), 10000) / 10000,
                min(g.get("controlProgress", 0), 1e6) / 1e6,
                min(g.get("creeps", 0), 50) / 50,
                (g.get("gcl", 1) - 1) / 10,
                min(g.get("bucket", 10000), 10000) / 10000,
                min(g.get("visibleRooms", 0), 16) / 16,
                min(g.get("actorCount", 0), 128) / 128,
                min(g.get("targetCount", 0), 256) / 256,
                float(bool(g.get("roomOverflow", 0))),
                float(bool(g.get("actorOverflow", 0))),
                float(bool(g.get("targetOverflow", 0))),
            ],
            dtype=np.float32,
        )

        # Always materialize on CPU. VecScreepsEnv stacks then one H2D.
        # from_numpy shares storage when array is contiguous C-order.
        def t_f32(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)

        def t_u8(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(x, dtype=np.uint8)).unsqueeze(0)

        return {
            "patches": t_u8(patches),
            "room_mask": t_u8(room_mask),
            "room_coords": t_f32(room_coords),
            "actors": t_f32(actors),
            "actor_mask": t_u8(actor_mask),
            "actor_outcome": t_u8(actor_outcome),
            "targets": t_f32(targets),
            "target_mask": t_u8(target_mask),
            "intent_mask": t_u8(intent_mask),
            "dir_mask": t_u8(dir_mask),
            "target_select_mask": t_u8(target_select),
            "amount_mask": t_u8(amount_mask),
            "construction_mask": t_u8(construction_mask),
            "globals": t_f32(globals_),
            "_actor_meta": obs.get("actorMeta") or [],
            "_target_meta": obs.get("targetMeta") or [],
            "_time": obs.get("time", 0),
        }


def _as_nested_list(x: Any) -> list:
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.device.type != "cpu":
            x = x.cpu()
        # drop batch dim if present (single-env actions are [1, A, S])
        if x.dim() >= 1 and x.shape[0] == 1:
            x = x[0]
        # Compact: nested Python lists of ints (actions are small; avoid float dumps)
        return x.to(dtype=torch.int64).tolist()
    if isinstance(x, np.ndarray):
        if x.ndim >= 1 and x.shape[0] == 1:
            x = x[0]
        return x.astype(np.int64, copy=False).tolist()
    return x


def _json_command(msg: dict[str, Any]) -> dict[str, Any]:
    """Materialize tensor actions only in deliberate JSON debug mode."""
    if msg.get("cmd") not in {"step", "step_labeled"}:
        return msg
    actions = msg.get("actions")
    if not isinstance(actions, dict):
        raise ValueError("step actions must be a mapping")
    scalar_names = (
        "types", "dirs", "targets", "amounts", "construction_types",
    )
    planes = [_action_u8(actions.get(name), name) for name in scalar_names]
    shape = planes[0].shape
    if any(plane.shape != shape for plane in planes[1:]):
        raise ValueError(f"action plane shapes differ: {[p.shape for p in planes]}")
    body_counts = _body_vector_u8(
        actions.get("body_counts"), "body_counts", MAX_BODY_PARTS + 1,
    )
    body_order = _body_vector_u8(actions.get("body_order"), "body_order", N_BODY_PART)
    _validate_body_planes(body_counts, body_order, shape)
    construction_tiles = _action_u16(
        actions.get("construction_tiles"), "construction_tiles", N_CONSTRUCTION_TILE,
    )
    if construction_tiles.shape != shape:
        raise ValueError(
            f"actions.construction_tiles shape {construction_tiles.shape} differs from {shape}"
        )
    return {
        **msg,
        "actions": {
            "types": planes[0].tolist(),
            "dirs": planes[1].tolist(),
            "targets": planes[2].tolist(),
            "amounts": planes[3].tolist(),
            "constructionTypes": planes[4].tolist(),
            "constructionTiles": construction_tiles.tolist(),
            "bodyCounts": body_counts.tolist(),
            "bodyOrder": body_order.tolist(),
        },
    }


def _action_u8(x: Any, name: str) -> np.ndarray:
    """Return a validated [actors, slots] uint8 view for the command wire."""
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.device.type != "cpu":
            x = x.cpu()
        array = x.numpy()
    else:
        array = np.asarray(x)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"actions.{name} must have shape [actors, slots], got {array.shape}")
    if array.dtype.kind not in "biuf":
        raise ValueError(f"actions.{name} must be numeric, got dtype={array.dtype}")
    if array.size and (not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all()):
        raise ValueError(f"actions.{name} must contain finite integers")
    upper = {
        "types": N_INTENT,
        "dirs": N_DIR,
        "targets": MAX_TARGETS,
        "amounts": N_AMOUNT,
        "construction_types": N_CONSTRUCTION_TYPE,
    }[name]
    if upper > 256:
        raise RuntimeError(f"XAC1 cannot encode actions.{name} cardinality {upper}")
    if array.size and (array.min() < 0 or array.max() >= upper):
        raise ValueError(f"actions.{name} values must be in [0, {upper})")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _action_u16(x: Any, name: str, upper: int) -> np.ndarray:
    """Return a validated [actors, slots] little-endian uint16 action plane."""
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.device.type != "cpu":
            x = x.cpu()
        array = x.numpy()
    else:
        array = np.asarray(x)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"actions.{name} must have shape [actors, slots], got {array.shape}")
    if array.dtype.kind not in "biuf":
        raise ValueError(f"actions.{name} must be numeric, got dtype={array.dtype}")
    if array.size and (
        not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all()
    ):
        raise ValueError(f"actions.{name} must contain finite integers")
    if array.size and (array.min() < 0 or array.max() >= upper):
        raise ValueError(f"actions.{name} values must be in [0, {upper})")
    return np.ascontiguousarray(array, dtype="<u2")


def _body_vector_u8(x: Any, name: str, upper: int) -> np.ndarray:
    """Return a validated [actors, slots, body-part-types] uint8 plane."""
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.device.type != "cpu":
            x = x.cpu()
        array = x.numpy()
    else:
        array = np.asarray(x)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != N_BODY_PART:
        raise ValueError(
            f"actions.{name} must have shape [actors, slots, {N_BODY_PART}], "
            f"got {array.shape}"
        )
    if array.dtype.kind not in "biuf":
        raise ValueError(f"actions.{name} must be numeric, got dtype={array.dtype}")
    if array.size and (
        not np.isfinite(array).all()
        or not np.equal(array, np.floor(array)).all()
    ):
        raise ValueError(f"actions.{name} must contain finite integers")
    if array.size and (array.min() < 0 or array.max() >= upper):
        raise ValueError(f"actions.{name} values must be in [0, {upper})")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _validate_body_planes(
    body_counts: np.ndarray, body_order: np.ndarray, shape: tuple[int, int],
) -> None:
    for name, body_plane in (("body_counts", body_counts), ("body_order", body_order)):
        if body_plane.shape[:2] != shape:
            raise ValueError(
                f"actions.{name} leading shape {body_plane.shape[:2]} differs from {shape}"
            )
    expected_order = np.arange(N_BODY_PART, dtype=np.uint8)
    if body_order.size and not np.all(np.sort(body_order, axis=-1) == expected_order):
        raise ValueError("each actions.body_order row must be a permutation")
    nonzero = body_counts > 0
    ordered_nonzero = np.take_along_axis(nonzero, body_order, axis=-1)
    nonzero_count = nonzero.sum(axis=-1)
    active = np.arange(N_BODY_PART) < nonzero_count[..., None]
    if not np.array_equal(ordered_nonzero, active):
        raise ValueError("actions.body_order must place nonzero-count types first")
    inactive = ~active
    descending = body_order[..., 1:] < body_order[..., :-1]
    if np.any(descending & inactive[..., 1:] & inactive[..., :-1]):
        raise ValueError("actions.body_order zero-count suffix must be ascending")


def _encode_binary_command(msg: dict[str, Any]) -> bytes:
    """Encode one strict XAC1 command frame.

    Header (16 bytes): magic, protocol version, opcode, schema version,
        payload length, actor rows, intent slots. A step payload contains five scalar
    uint8 planes, one uint16 construction-tile plane, then eight-byte body-count
    and body-order planes per actor/slot.
    """
    cmd = msg.get("cmd")
    try:
        opcode = _COMMAND_OPCODES[cmd]
    except (KeyError, TypeError) as err:
        raise ValueError(f"unsupported command {cmd!r}") from err

    rows = 0
    slots = 0
    payload = b""
    if cmd in {"step", "step_labeled"}:
        actions = msg.get("actions")
        if not isinstance(actions, dict):
            raise ValueError("step actions must be a mapping")
        scalar_names = (
            "types", "dirs", "targets", "amounts", "construction_types",
        )
        planes = [
            _action_u8(actions.get(name), name)
            for name in scalar_names
        ]
        shape = planes[0].shape
        if any(plane.shape != shape for plane in planes[1:]):
            raise ValueError(f"action plane shapes differ: {[p.shape for p in planes]}")
        rows, slots = shape
        if rows > 0xFFFF or slots > 0xFF:
            raise ValueError(f"action shape cannot be represented by XAC1: {shape}")
        if rows > MAX_ACTORS:
            raise ValueError(f"action actor count {rows} exceeds maxActors={MAX_ACTORS}")
        if slots != INTENT_SLOTS:
            raise ValueError(f"action slots={slots} does not match schema={INTENT_SLOTS}")
        body_counts = _body_vector_u8(
            actions.get("body_counts"), "body_counts", MAX_BODY_PARTS + 1,
        )
        body_order = _body_vector_u8(
            actions.get("body_order"), "body_order", N_BODY_PART,
        )
        _validate_body_planes(body_counts, body_order, shape)
        construction_tiles = _action_u16(
            actions.get("construction_tiles"), "construction_tiles", N_CONSTRUCTION_TILE,
        )
        if construction_tiles.shape != shape:
            raise ValueError(
                "actions.construction_tiles shape "
                f"{construction_tiles.shape} differs from {shape}"
            )
        payload = b"".join(plane.tobytes(order="C") for plane in planes)
        payload += construction_tiles.tobytes(order="C")
        payload += body_counts.tobytes(order="C")
        payload += body_order.tobytes(order="C")
    elif cmd in {"snapshot", "restore"}:
        request_path = msg.get("path")
        if not isinstance(request_path, str) or not request_path:
            raise ValueError(f"{cmd} requires a non-empty path")
        events = msg.get("events") or []
        if not isinstance(events, (list, tuple)) or any(
            not isinstance(tag, str) for tag in events
        ):
            raise ValueError(f"{cmd} events must be a sequence of tags")
        payload = json.dumps(
            {"path": request_path, "events": list(events)}, separators=(",", ":"),
        ).encode("utf-8")
    elif cmd == "reset_expert":
        bot_dir = msg.get("botDir")
        if bot_dir is not None:
            if not isinstance(bot_dir, str):
                raise ValueError("reset_expert botDir must be a string")
            payload = bot_dir.encode("utf-8")
            if b"\0" in payload:
                raise ValueError("reset_expert botDir must not contain NUL")
    elif set(msg) - {"cmd"}:
        raise ValueError(f"command {cmd!r} has unexpected fields")

    if len(payload) > 0xFFFFFFFF:
        raise ValueError("command payload is too large")
    return _COMMAND_HEADER.pack(
        _COMMAND_MAGIC,
        _COMMAND_VERSION,
        opcode,
        int(SCHEMA["version"]),
        len(payload),
        rows,
        slots,
        0,
    ) + payload


def stack_batches(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = [
        "patches", "room_mask", "room_coords", "actors", "actor_mask",
        "actor_outcome", "targets", "target_mask",
        "intent_mask", "dir_mask", "target_select_mask", "amount_mask",
        "construction_mask", "globals",
    ]
    return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}
