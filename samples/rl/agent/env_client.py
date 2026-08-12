"""JSONL client for samples/rl/env/server.mjs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import (
    ACTOR_FEAT,
    INTENT_SLOTS,
    MAX_ACTORS,
    MAX_ROOMS,
    MAX_TARGETS,
    N_AMOUNT,
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
        # bin = raw frames (default); pack = single base64 blob; b64 = per-field; json = debug
        env.setdefault("RL_OBS_FMT", self.obs_fmt)
        # Train and evaluation share the same navigation dynamics by default.
        env["RL_LEAN_META"] = "1" if self.lean_meta else "0"
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
        # Binary frames need raw pipes; JSONL modes use text line buffering.
        text_mode = not self._bin
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_ROOT),
            env=env,
            text=text_mode,
            bufsize=1 if text_mode else 0,
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
        """Parse XRL1 length-prefixed response (see docs/13-PERFORMANCE.md).

        Hard-fail on bad magic (no half-resync — leftover bugs desync permanently).
        Server bin mode must keep stdout frame-only (console.log → stderr).
        """
        hdr = self._read_exact(16)
        if hdr[0:4] != b"XRL1":
            tail = "\n".join(self._stderr_tail[-15:]) if self._stderr_tail else ""
            raise RuntimeError(
                f"bad bin magic {hdr[0:4]!r} (stdout pollution?). stderr:\n{tail[:600]}"
            )
        if hdr[4] != 1:
            raise RuntimeError(f"unsupported binary protocol version {hdr[4]}")
        wire_schema = int.from_bytes(hdr[6:8], "little")
        if wire_schema != int(SCHEMA["version"]):
            raise RuntimeError(
                f"environment schema={wire_schema} does not match client "
                f"schema={SCHEMA['version']}"
            )
        flags = hdr[5]
        meta_len = int.from_bytes(hdr[8:12], "little")
        blob_len = int.from_bytes(hdr[12:16], "little")
        # Sanity caps (local trust, avoid OOM on corrupt length)
        if meta_len > 16 * 1024 * 1024 or blob_len > 64 * 1024 * 1024:
            raise RuntimeError(f"bin frame size absurd meta={meta_len} blob={blob_len}")
        meta_buf = self._read_exact(meta_len)
        blob = self._read_exact(blob_len) if blob_len else b""
        meta = json.loads(meta_buf.decode("utf-8"))
        if not meta.get("ok", bool(flags & 1)):
            raise RuntimeError(meta.get("error") or f"env error: {meta}")
        # Normalize to the same shape as JSONL responses
        out: dict[str, Any] = {
            "ok": True,
            "reward": float(meta.get("reward", 0)),
            "done": bool(meta.get("done", False)),
            "info": meta.get("info") or {},
        }
        if flags & 4 or meta.get("shapes") or blob:
            out["obs"] = {
                "encoding": "bin",
                "blob": blob,  # raw bytes, not base64
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
                payload = (json.dumps(msg) + "\n")
                if self._bin:
                    self._stdin.write(payload.encode("utf-8"))
                else:
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
                    data = json.loads(line)
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
            self._spawn_server()
        return self.reset()

    def reset(self) -> dict[str, torch.Tensor]:
        if self.expert:
            data = self._rpc({"cmd": "reset_expert", "botDir": self.bot_dir})
        else:
            data = self._rpc({"cmd": "reset"})
        self.last_info = data.get("info") or {}
        return self._obs_to_batch(data["obs"])

    def step(self, actions: dict[str, torch.Tensor | np.ndarray | list] | None = None) -> tuple[
        dict[str, torch.Tensor], float, bool, dict[str, Any]
    ]:
        if self.expert:
            data = self._rpc({"cmd": "step_expert"})
        else:
            assert actions is not None
            payload = {
                "types": _as_nested_list(actions["types"]),
                "dirs": _as_nested_list(actions["dirs"]),
                "targets": _as_nested_list(actions["targets"]),
                "amounts": _as_nested_list(actions["amounts"]),
            }
            data = self._rpc({"cmd": "step", "actions": payload})
        self.last_info = data.get("info") or {}
        if "obs" not in data or "reward" not in data:
            raise RuntimeError(f"malformed step response keys={list(data.keys())}")
        obs = self._obs_to_batch(data["obs"])
        return obs, float(data["reward"]), bool(data.get("done", False)), self.last_info

    def step_scripted(self) -> tuple[
        dict[str, torch.Tensor], float, bool, dict[str, Any], dict[str, list]
    ]:
        """RCL1 scripted baseline (same action interface). Returns actions for BC."""
        if self.expert:
            raise RuntimeError("step_scripted is for non-expert sessions")
        data = self._rpc({"cmd": "step_scripted"})
        self.last_info = data.get("info") or {}
        if "obs" not in data or "reward" not in data:
            raise RuntimeError(f"malformed step_scripted response keys={list(data.keys())}")
        obs = self._obs_to_batch(data["obs"])
        actions = (self.last_info.get("actions") or {})
        return (
            obs,
            float(data["reward"]),
            bool(data.get("done", False)),
            self.last_info,
            actions,
        )

    def close(self) -> None:
        try:
            self._rpc({"cmd": "close"})
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.kill()

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
            target_mask = take_u8(MAX_TARGETS)
            intent_mask = take_u8(int(np.prod(si))).reshape(si)
            dir_mask = take_u8(int(np.prod(sd))).reshape(sd)
            target_select = take_u8(int(np.prod(sts))).reshape(sts)
            amount_mask = take_u8(int(np.prod(sam))).reshape(sam)
            if off != len(raw):
                # Tolerate trailing padding; hard fail only if short
                pass
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
            room_mask = _u8("roomMask")
            actor_mask = _u8("actorMask")
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
            "targets": t_f32(targets),
            "target_mask": t_u8(target_mask),
            "intent_mask": t_u8(intent_mask),
            "dir_mask": t_u8(dir_mask),
            "target_select_mask": t_u8(target_select),
            "amount_mask": t_u8(amount_mask),
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


def stack_batches(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = [
        "patches", "room_mask", "room_coords", "actors", "actor_mask", "targets", "target_mask",
        "intent_mask", "dir_mask", "target_select_mask", "amount_mask", "globals",
    ]
    return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}
