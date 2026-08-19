"""Strict, atomic model-artifact contracts for the current schema."""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .constants import SCHEMA, SCHEMA_SHA256


SOURCE_SIGNATURE_AGENT_NAMES = (
    "actions_util.py", "artifacts.py", "constants.py", "env_client.py",
    "gae.py", "hl_gauss.py", "model.py", "ppo.py", "pretrain_joint.py",
    "muon.py", "pretrain_ti.py", "rollout_buffer.py", "running_stats.py",
    "ti_intents.py", "train.py", "vec_env.py",
)


def directory_signature(directory: str | Path) -> str | None:
    """Hash a generated/external source tree for provenance, not compatibility."""
    root = Path(directory).resolve()
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def source_signature() -> str:
    """Fingerprint executable RL source so dirty-tree artifacts remain attributable."""
    rl_root = Path(__file__).resolve().parents[1]
    repository = rl_root.parents[1]
    env_names = ("actions.mjs", "encode.mjs", "scripted_baseline.mjs", "server.mjs")
    paths = [rl_root / "schema.json"]
    paths.extend(rl_root / "agent" / name for name in SOURCE_SIGNATURE_AGENT_NAMES)
    paths.extend(rl_root / "env" / name for name in env_names)
    # Simulator behavior is part of the learned MDP. Package exports resolve
    # to dist JavaScript, so fingerprint the executed build rather than TS
    # sources that may not have been rebuilt. Exclude source maps and tests.
    engine_root = repository / "packages" / "xxscreeps"
    paths.append(engine_root / "package.json")
    paths.extend(
        path for path in (engine_root / "dist").rglob("*.js")
        if ".test." not in path.name and "test" not in path.parts
    )
    paths = [path for path in paths if path.is_file()]
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(repository).as_posix()):
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def state_signature(module: nn.Module) -> str:
    """Fingerprint the complete parameter/buffer ABI without reading values."""
    rows = [
        f"{name}|{','.join(map(str, tensor.shape))}|{tensor.dtype}"
        for name, tensor in module.state_dict().items()
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def artifact_meta(
    kind: str,
    actor: nn.Module,
    critic: nn.Module,
    *,
    source_sha256: str | None = None,
    **extra,
) -> dict:
    return {
        "kind": kind,
        "schema_version": SCHEMA["version"],
        "schema_sha256": SCHEMA_SHA256,
        # Capture this once when a run starts. Re-reading a shared dirty tree
        # at save time can falsely attribute already-imported Python/Node code
        # to edits made while the process was running.
        "source_sha256": source_sha256 or source_signature(),
        "contracts": dict(SCHEMA["artifact"]),
        "actor_state_signature": state_signature(actor),
        "critic_state_signature": state_signature(critic),
        **extra,
    }


def validate_artifact(
    checkpoint: dict,
    actor: nn.Module,
    critic: nn.Module,
    *,
    kinds: Iterable[str],
    allow_source_mismatch: bool = False,
    evaluation_only_source_mismatch: bool = False,
) -> dict:
    meta = checkpoint.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("checkpoint has no metadata contract")
    allowed = set(kinds)
    if meta.get("kind") not in allowed:
        raise ValueError(f"checkpoint kind={meta.get('kind')!r}; expected one of {sorted(allowed)}")
    if meta.get("schema_version") != SCHEMA["version"]:
        raise ValueError(
            f"checkpoint schema={meta.get('schema_version')!r}; current={SCHEMA['version']}"
        )
    if meta.get("schema_sha256") != SCHEMA_SHA256:
        raise ValueError("checkpoint semantic schema fingerprint differs from current schema")
    if meta.get("source_sha256") != source_signature():
        if not evaluation_only_source_mismatch and (
            not allow_source_mismatch or meta.get("kind") != "joint_pretrain"
        ):
            raise ValueError("checkpoint executable RL source fingerprint differs from current source")
    if meta.get("contracts") != SCHEMA["artifact"]:
        raise ValueError(
            f"checkpoint contracts={meta.get('contracts')!r}; current={SCHEMA['artifact']!r}"
        )
    expected_actor = state_signature(actor)
    expected_critic = state_signature(critic)
    if meta.get("actor_state_signature") != expected_actor:
        raise ValueError("checkpoint actor parameter ABI differs from current model")
    if meta.get("critic_state_signature") != expected_critic:
        raise ValueError("checkpoint critic parameter ABI differs from current model")
    return meta


def load_full_state(module: nn.Module, state: dict, *, name: str) -> None:
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"incomplete or incompatible {name} state: {error}") from error


def load_critic_trunk(critic: nn.Module, state: dict) -> None:
    """Load exactly the critic trunk; reject missing or unexpected trunk tensors."""
    expected = {key for key in critic.state_dict() if key.startswith("trunk.")}
    supplied = {key for key in state if key.startswith("trunk.")}
    missing = expected - supplied
    unexpected = supplied - expected
    if missing or unexpected:
        raise ValueError(
            f"critic trunk ABI mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    filtered = {key: state[key] for key in expected}
    result = critic.load_state_dict(filtered, strict=False)
    expected_head = set(critic.state_dict()) - expected
    if set(result.missing_keys) != expected_head or result.unexpected_keys:
        raise ValueError("critic trunk import produced an unexpected missing-key contract")


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    """Write a recoverable artifact without exposing a partially-written target."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
