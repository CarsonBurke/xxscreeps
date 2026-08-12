"""Strict, atomic schema-v2 model artifact contracts."""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .constants import SCHEMA, SCHEMA_SHA256


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
    **extra,
) -> dict:
    return {
        "kind": kind,
        "schema_version": SCHEMA["version"],
        "schema_sha256": SCHEMA_SHA256,
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
