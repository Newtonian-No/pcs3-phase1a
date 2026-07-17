"""Task and pointwise auxiliary losses for temporal ablations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import GC_VARIANTS, VARIANTS
from .model import TemporalModelOutput
from .metrics import BINARY_DATASETS


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
    if dataset in BINARY_DATASETS:
        if logits.ndim != 2 or logits.shape[-1] != 1:
            raise ValueError("temporal logic logits must be B x 1")
        target = target.float().reshape(-1)
        if len(target) != logits.shape[0]:
            raise ValueError("logits and target batch sizes differ")
        return F.binary_cross_entropy_with_logits(logits[:, 0], target)
    if dataset in {"uci_har", "generalized_dynamics"}:
        if logits.ndim != 2 or logits.shape[-1] < 2:
            raise ValueError(f"{dataset} logits must be B x C")
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


def generalized_prediction_loss(
    errors: Tensor,
    valid: Tensor,
    *,
    signal_dim: int,
    active_order: int,
) -> Tensor:
    if signal_dim <= 0:
        raise ValueError("signal_dim must be positive")
    if active_order not in (1, 2, 3):
        raise ValueError("active_order must be 1, 2, or 3")
    if errors.ndim != 3 or errors.shape[-1] != 3 * signal_dim:
        raise ValueError("errors must be B x T x (3 * signal_dim)")
    shaped = errors.float().reshape(*errors.shape[:2], 3, signal_dim)
    try:
        mask = torch.broadcast_to(
            valid.to(device=errors.device, dtype=torch.bool), shaped.shape
        )
    except RuntimeError as error:
        raise ValueError(
            "valid must be broadcastable to B x T x 3 x signal_dim"
        ) from error

    weights = (1.0, 0.5, 0.25)
    loss = torch.zeros((), device=errors.device, dtype=torch.float32)
    for order in range(active_order):
        order_loss = F.smooth_l1_loss(
            shaped[:, :, order],
            torch.zeros_like(shaped[:, :, order]),
            reduction="none",
        )
        order_mask = mask[:, :, order]
        masked_sum = torch.where(order_mask, order_loss, 0.0).sum()
        loss = loss + weights[order] * masked_sum / order_mask.sum().clamp_min(1)
    return loss


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
    if variant not in VARIANTS + GC_VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    task = compute_task_loss(output.logits, target, dataset=dataset)
    if variant in GC_VARIANTS:
        if output.coordinate_errors is None or output.coordinate_mask is None:
            raise ValueError(f"variant {variant} requires generalized-coordinate errors")
        if output.gc_order not in (1, 2, 3):
            raise ValueError(f"variant {variant} requires a valid gc_order")
        if output.coordinate_errors.shape[-1] % 3:
            raise ValueError("coordinate error width must be divisible by 3")
        auxiliary = generalized_prediction_loss(
            output.coordinate_errors,
            output.coordinate_mask,
            signal_dim=output.coordinate_errors.shape[-1] // 3,
            active_order=output.gc_order,
        )
        weight = auxiliary_weight(
            epoch,
            total_epochs=total_epochs,
            target=lambda_aux,
            warmup_fraction=aux_warmup_fraction,
        )
    elif variant in _AUX_VARIANTS:
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
