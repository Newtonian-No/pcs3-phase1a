"""Fail-fast finite-value guards and auditable failure artifacts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn


class NumericalFailure(RuntimeError):
    def __init__(
        self,
        component: str,
        *,
        observed_min: float | None,
        observed_max: float | None,
    ) -> None:
        self.component = component
        self.observed_min = observed_min
        self.observed_max = observed_max
        super().__init__(
            f"non-finite value in {component}; "
            f"finite_min={observed_min!r}, finite_max={observed_max!r}"
        )


def assert_finite_tensor(component: str, tensor: Tensor) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{component} must be a torch.Tensor")
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return
    finite_values = tensor.detach()[finite_mask]
    if finite_values.numel():
        observed_min = float(finite_values.real.min().cpu())
        observed_max = float(finite_values.real.max().cpu())
    else:
        observed_min = None
        observed_max = None
    raise NumericalFailure(
        component,
        observed_min=observed_min,
        observed_max=observed_max,
    )


def assert_finite_model(model: nn.Module, *, check_gradients: bool = False) -> None:
    for name, parameter in model.named_parameters():
        assert_finite_tensor(f"parameter.{name}", parameter)
        if check_gradients and parameter.grad is not None:
            assert_finite_tensor(f"gradient.{name}", parameter.grad)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_failure_artifact(
    path: str | Path,
    *,
    run_id: str,
    epoch: int,
    batch: int,
    tensor_name: str,
    error: BaseException,
    last_healthy_checkpoint: str | Path | None,
    diagnostics: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "epoch": int(epoch),
        "batch": int(batch),
        "tensor_name": tensor_name,
        "error_type": type(error).__name__,
        "error": str(error),
        "last_healthy_checkpoint": (
            None if last_healthy_checkpoint is None else str(last_healthy_checkpoint)
        ),
        "diagnostics": {} if diagnostics is None else diagnostics,
    }
    safe_payload = _json_safe(payload)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(safe_payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
