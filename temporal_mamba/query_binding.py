"""Explicit query-to-channel binding and bounded layer conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BoundTemporalInput:
    sequence: Tensor
    condition: Tensor
    prediction_signal: Tensor


def _validate_one_hot(values: Tensor, name: str, *, allow_zero: bool) -> None:
    binary = (values == 0) | (values == 1)
    if not bool(binary.all()):
        raise ValueError(f"{name} must contain only zero and one")
    sums = values.sum(dim=-1)
    valid = (sums == 1) | ((sums == 0) if allow_zero else torch.zeros_like(sums, dtype=torch.bool))
    if not bool(valid.all()):
        qualifier = "zero or one-hot" if allow_zero else "one-hot"
        raise ValueError(f"{name} must be {qualifier}")


class TemporalQueryBinder(nn.Module):
    """Select query-referenced A/B streams before temporal encoding."""

    def __init__(self, event_dim: int, family_dim: int = 6) -> None:
        super().__init__()
        if event_dim <= 0 or family_dim <= 0:
            raise ValueError("event_dim and family_dim must be positive")
        self.event_dim = event_dim
        self.family_dim = family_dim
        self.query_dim = family_dim + 2 * event_dim + 3

    def forward(self, signal: Tensor, query: Tensor) -> BoundTemporalInput:
        if signal.ndim != 3:
            raise ValueError("signal must be B x T x event_dim")
        if query.ndim != 2:
            raise ValueError("query must be B x query_dim")
        if signal.shape[0] != query.shape[0]:
            raise ValueError("signal and query batch dimensions must match")
        if signal.shape[-1] != self.event_dim:
            raise ValueError(f"signal event_dim must be {self.event_dim}")
        if query.shape[-1] != self.query_dim:
            raise ValueError(f"query last dimension must be {self.query_dim}")
        if not bool(torch.isfinite(signal).all()) or not bool(torch.isfinite(query).all()):
            raise ValueError("signal and query must be finite")

        query_float = query.float()
        family = query_float[:, : self.family_dim]
        a_start = self.family_dim
        b_start = a_start + self.event_dim
        param_start = b_start + self.event_dim
        a_one_hot = query_float[:, a_start:b_start]
        b_one_hot = query_float[:, b_start:param_start]
        params = query_float[:, param_start : param_start + 3]
        _validate_one_hot(family, "family", allow_zero=False)
        _validate_one_hot(a_one_hot, "event_a", allow_zero=False)
        _validate_one_hot(b_one_hot, "event_b", allow_zero=True)
        if not bool(((params >= 0) & (params <= 1)).all()):
            raise ValueError("query parameters must be normalized to [0, 1]")

        signal_float = signal.float()
        a = torch.einsum("btd,bd->bt", signal_float, a_one_hot)
        b = torch.einsum("btd,bd->bt", signal_float, b_one_hot)
        time = torch.linspace(
            0.0,
            1.0,
            signal.shape[1],
            device=signal.device,
            dtype=torch.float32,
        ).expand(signal.shape[0], -1)
        sequence = torch.stack(
            (a, b, time, time - params[:, 0:1], time - params[:, 1:2]),
            dim=-1,
        )
        prediction_signal = torch.stack((a, b), dim=-1)
        return BoundTemporalInput(
            sequence=sequence,
            condition=torch.cat((family, params), dim=-1),
            prediction_signal=prediction_signal,
        )


class BoundedQueryFiLM(nn.Module):
    """Produce identity-initialized, bounded per-layer scale and shift."""

    def __init__(
        self,
        condition_dim: int,
        n_layers: int,
        d_model: int,
        scale_limit: float = 0.25,
        shift_limit: float = 0.25,
    ) -> None:
        super().__init__()
        if condition_dim <= 0 or n_layers <= 0 or d_model <= 0:
            raise ValueError("condition_dim, n_layers, and d_model must be positive")
        if scale_limit < 0 or shift_limit < 0:
            raise ValueError("scale_limit and shift_limit must be non-negative")
        self.condition_dim = condition_dim
        self.n_layers = n_layers
        self.d_model = d_model
        self.scale_limit = float(scale_limit)
        self.shift_limit = float(shift_limit)
        self.projection = nn.Linear(condition_dim, 2 * n_layers * d_model)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError(f"condition must be B x {self.condition_dim}")
        if not bool(torch.isfinite(condition).all()):
            raise ValueError("condition must be finite")
        raw = self.projection(condition.float()).view(
            -1,
            self.n_layers,
            2,
            self.d_model,
        )
        scale = 1.0 + self.scale_limit * torch.tanh(raw[:, :, 0])
        shift = self.shift_limit * torch.tanh(raw[:, :, 1])
        return scale, shift

