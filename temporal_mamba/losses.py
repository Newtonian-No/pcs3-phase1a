"""Task and pointwise auxiliary losses for temporal ablations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import VARIANTS
from .model import TemporalModelOutput


_AUX_VARIANTS = {"error_aux", "time_shuffle", "time_reverse"}


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    task: Tensor
    auxiliary: Tensor
    aux_weight: float


def auxiliary_weight(
    epoch: int,
    *,
    total_epochs: int,
    target: float,
    warmup_fraction: float,
) -> float:
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if target < 0:
        raise ValueError("target must be non-negative")
    if not 0 <= warmup_fraction <= 1:
        raise ValueError("warmup_fraction must be in [0, 1]")
    if target == 0:
        return 0.0
    warmup_epochs = max(1, math.ceil(total_epochs * warmup_fraction))
    return float(target * min(epoch / warmup_epochs, 1.0))


def compute_task_loss(logits: Tensor, target: Tensor, *, dataset: str) -> Tensor:
    if dataset == "temporal_logic":
        if logits.ndim != 2 or logits.shape[-1] != 1:
            raise ValueError("temporal_logic logits must be B x 1")
        target = target.float().reshape(-1)
        if len(target) != logits.shape[0]:
            raise ValueError("logits and target batch sizes differ")
        return F.binary_cross_entropy_with_logits(logits[:, 0], target)
    if dataset == "uci_har":
        if logits.ndim != 2 or logits.shape[-1] < 2:
            raise ValueError("uci_har logits must be B x C")
        target = target.long().reshape(-1)
        if len(target) != logits.shape[0]:
            raise ValueError("logits and target batch sizes differ")
        return F.cross_entropy(logits, target)
    raise ValueError(f"unknown dataset: {dataset}")


def pointwise_prediction_loss(
    position_error: Tensor,
    velocity_error: Tensor,
    *,
    velocity_weight: float = 0.5,
) -> Tensor:
    if position_error.shape != velocity_error.shape or position_error.ndim != 3:
        raise ValueError("position_error and velocity_error must share B x T x D shape")
    if position_error.shape[1] < 2:
        raise ValueError("prediction loss requires at least two time steps")
    if velocity_weight < 0:
        raise ValueError("velocity_weight must be non-negative")
    position = F.smooth_l1_loss(
        position_error[:, 1:],
        torch.zeros_like(position_error[:, 1:]),
        reduction="mean",
    )
    velocity = F.smooth_l1_loss(
        velocity_error[:, 1:],
        torch.zeros_like(velocity_error[:, 1:]),
        reduction="mean",
    )
    return position + velocity_weight * velocity


def compute_total_loss(
    output: TemporalModelOutput,
    target: Tensor,
    *,
    dataset: str,
    variant: str,
    epoch: int,
    total_epochs: int,
    lambda_aux: float,
    aux_warmup_fraction: float,
) -> LossBreakdown:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    task = compute_task_loss(output.logits, target, dataset=dataset)
    if variant in _AUX_VARIANTS:
        if output.position_error is None or output.velocity_error is None:
            raise ValueError(f"variant {variant} requires aligned prediction errors")
        auxiliary = pointwise_prediction_loss(output.position_error, output.velocity_error)
        weight = auxiliary_weight(
            epoch,
            total_epochs=total_epochs,
            target=lambda_aux,
            warmup_fraction=aux_warmup_fraction,
        )
    else:
        auxiliary = torch.zeros((), device=task.device, dtype=task.dtype)
        weight = 0.0
    return LossBreakdown(
        total=task + weight * auxiliary,
        task=task,
        auxiliary=auxiliary,
        aux_weight=weight,
    )
