"""Atomic, versioned checkpoints with complete stochastic-state restore."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = 1


def _rng_state(loader_generator: torch.Generator | None) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": loader_generator.get_state() if loader_generator is not None else None,
    }


def _restore_rng_state(state: dict[str, Any], loader_generator: torch.Generator | None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if loader_generator is not None and state.get("loader_generator") is not None:
        loader_generator.set_state(state["loader_generator"])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    loader_generator: torch.Generator | None,
    epoch: int,
    step: int,
    best_metric: float,
    history_cursor: int,
    config_hash: str,
    git_commit: str,
    dataset_manifest_hash: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically write model, optimizer, metadata, and every RNG stream."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "dataset_manifest_hash": dataset_manifest_hash,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": _rng_state(loader_generator),
        "epoch": int(epoch),
        "step": int(step),
        "best_metric": float(best_metric),
        "history_cursor": int(history_cursor),
        "extra": {} if extra is None else extra,
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    loader_generator: torch.Generator | None = None,
    map_location: str | torch.device = "cpu",
    expected_config_hash: str | None = None,
    expected_dataset_manifest_hash: str | None = None,
    expected_git_commit: str | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Validate metadata, restore training state, and return the checkpoint."""

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {CHECKPOINT_SCHEMA_VERSION}, "
            f"got {checkpoint.get('schema_version')}"
        )
    expected_values = {
        "config_hash": expected_config_hash,
        "dataset_manifest_hash": expected_dataset_manifest_hash,
        "git_commit": expected_git_commit,
    }
    for name, expected in expected_values.items():
        if expected is not None and checkpoint.get(name) != expected:
            raise ValueError(
                f"{name} mismatch: expected {expected!r}, got {checkpoint.get(name)!r}"
            )
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng:
        _restore_rng_state(checkpoint["rng"], loader_generator)
    return checkpoint
