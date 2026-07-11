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
    SCHEMA_PATH,
    TARGET_FEAT,
    TILE_FEAT,
)

_ROOT = Path(__file__).resolve().parents[2].parent  # repo root (xxscreeps)
# samples/rl/agent -> samples/rl -> samples -> repo
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
        # _ROOT = xxscreeps repo; The International lives as a sibling checkout
        default_ti = (_ROOT.parent / "The-International-Open-Source" / "dist").resolve()
        self.bot_dir = bot_dir or os.environ.get("RL_EXPERT_BOT", str(default_ti))
        if not Path(self.bot_dir).is_dir():
            raise FileNotFoundError(
                f"expert bot dir missing: {self.bot_dir} "
                f"(build TI: cd ../The-International-Open-Source && npm run build)"
            )
        self._lock = threading.Lock()
        self._spawn_server()
        self.last_info: dict[str, Any] = {}

    def _spawn_server(self) -> None:
        env = os.environ.copy()
        env["RL_ROOM"] = self.room
        env["RL_MAX_EPISODE"] = str(self.max_episode)
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
            # Do not inherit a parent RL_HEADFUL (would bind every env to :21025)
            env.pop("RL_HEADFUL", None)
            for key in ("RL_HEADFUL_PASSWORD", "RL_HEADFUL_BIND", "RL_TICK_MS", "RL_NO_OPEN"):
                if key in os.environ and key != "RL_TICK_MS":
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
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_ROOT),
            env=env,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdin and self.proc.stdout
        self._stdin = self.proc.stdin
        self._stdout = self.proc.stdout

    def _rpc(self, msg: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            try:
                self._stdin.write(json.dumps(msg) + "\n")
                self._stdin.flush()
            except BrokenPipeError as err:
                raise RuntimeError("env server pipe broken") from err
            while True:
                line = self._stdout.readline()
                if not line:
                    err = ""
                    try:
                        if self.proc.stderr:
                            err = self.proc.stderr.read()
                    except Exception:
                        pass
                    raise RuntimeError(f"env server died: {err[:500]}")
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

    def close(self) -> None:
        try:
            self._rpc({"cmd": "close"})
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.kill()

    def _obs_to_batch(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Single-env batch dim = 1."""
        shapes = obs["shapes"]
        patches = np.asarray(obs["patches"], dtype=np.float32).reshape(shapes["patches"])
        actors = np.asarray(obs["actors"], dtype=np.float32).reshape(shapes["actors"])
        targets = np.asarray(obs["targets"], dtype=np.float32).reshape(shapes["targets"])
        intent_mask = np.asarray(obs["intentMask"], dtype=np.float32).reshape(shapes["intentMask"])
        dir_mask = np.asarray(obs["dirMask"], dtype=np.float32).reshape(shapes["dirMask"])
        target_select = np.asarray(obs["targetSelectMask"], dtype=np.float32).reshape(
            shapes["targetSelectMask"]
        )
        amount_mask = np.asarray(obs["amountMask"], dtype=np.float32).reshape(shapes["amountMask"])
        room_mask = np.asarray(obs["roomMask"], dtype=np.float32)
        actor_mask = np.asarray(obs["actorMask"], dtype=np.float32)
        target_mask = np.asarray(obs["targetMask"], dtype=np.float32)
        g = obs["globals"]
        globals_ = np.asarray(
            [
                g.get("rclMax", 0) / 8,
                min(g.get("storedEnergy", 0), 10000) / 10000,
                min(g.get("controlProgress", 0), 1e6) / 1e6,
                min(g.get("creeps", 0), 50) / 50,
                (g.get("gcl", 1) - 1) / 10,
                min(g.get("bucket", 10000), 10000) / 10000,
            ],
            dtype=np.float32,
        )

        # Always materialize on CPU. VecScreepsEnv stacks in parallel then one H2D.
        # (Per-env .to(cuda) from worker threads serializes and kills parallelism.)
        def t(x: np.ndarray) -> torch.Tensor:
            ten = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
            if self.device.type == "cpu":
                return ten
            return ten  # host; caller moves

        return {
            "patches": t(patches),
            "room_mask": t(room_mask),
            "actors": t(actors),
            "actor_mask": t(actor_mask),
            "targets": t(targets),
            "target_mask": t(target_mask),
            "intent_mask": t(intent_mask),
            "dir_mask": t(dir_mask),
            "target_select_mask": t(target_select),
            "amount_mask": t(amount_mask),
            "globals": t(globals_),
            # metadata (not for model)
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
        return x.tolist()
    if isinstance(x, np.ndarray):
        if x.ndim >= 1 and x.shape[0] == 1:
            x = x[0]
        return x.tolist()
    return x


def stack_batches(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = [
        "patches", "room_mask", "actors", "actor_mask", "targets", "target_mask",
        "intent_mask", "dir_mask", "target_select_mask", "amount_mask", "globals",
    ]
    return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}
