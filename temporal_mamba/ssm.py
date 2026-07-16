"""Direct recurrent selective state-space layers for temporal sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def inverse_softplus(value: Tensor) -> Tensor:
    """Numerically stable inverse of ``softplus`` for strictly positive values."""

    if torch.any(value <= 0):
        raise ValueError("inverse_softplus input must be positive")
    return value + torch.log(-torch.expm1(-value))


def _validate_scan_shapes(u: Tensor, dt: Tensor, a_log: Tensor, b: Tensor, c: Tensor, d_skip: Tensor) -> None:
    if u.ndim != 3:
        raise ValueError(f"u must be B x T x D, got {tuple(u.shape)}")
    if dt.shape != u.shape:
        raise ValueError(f"dt must match u, got {tuple(dt.shape)} and {tuple(u.shape)}")
    if a_log.ndim != 2 or a_log.shape[0] != u.shape[2]:
        raise ValueError("a_log must be D x N")
    expected_bc = (u.shape[0], u.shape[1], a_log.shape[1])
    if b.shape != expected_bc or c.shape != expected_bc:
        raise ValueError(f"b and c must be B x T x N={expected_bc}")
    if d_skip.shape != (u.shape[2],):
        raise ValueError("d_skip must have shape D")


def direct_selective_scan(
    u: Tensor,
    dt: Tensor,
    a_log: Tensor,
    b: Tensor,
    c: Tensor,
    d_skip: Tensor,
) -> Tensor:
    """Run the explicit float32 selective recurrence from a zero state."""

    _validate_scan_shapes(u, dt, a_log, b, c, d_skip)
    u32, dt32 = u.float(), dt.float()
    a = -torch.exp(a_log.float())
    hidden = torch.zeros(
        u.shape[0],
        u.shape[2],
        a.shape[1],
        device=u.device,
        dtype=torch.float32,
    )
    outputs = []
    for time_index in range(u.shape[1]):
        a_bar = torch.exp(dt32[:, time_index].unsqueeze(-1) * a)
        b_bar = dt32[:, time_index].unsqueeze(-1) * b[:, time_index].float().unsqueeze(1)
        hidden = a_bar * hidden + b_bar * u32[:, time_index].unsqueeze(-1)
        output = (
            (hidden * c[:, time_index].float().unsqueeze(1)).sum(dim=-1)
            + d_skip.float() * u32[:, time_index]
        )
        outputs.append(output)
    return torch.stack(outputs, dim=1)


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.weight = nn.Parameter(torch.ones(dimension, dtype=torch.float32))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        value32 = value.float()
        scale = torch.rsqrt(value32.square().mean(dim=-1, keepdim=True) + self.eps)
        return value32 * scale * self.weight


@dataclass(frozen=True)
class SSMDiagnostics:
    dt_min: Tensor
    dt_max: Tensor
    error_rms: Tensor
    error_max: Tensor
    output_rms: Tensor
    output_max: Tensor
    finite: Tensor


class DirectSelectiveSSM(nn.Module):
    """Input-selective SSM with bounded, zero-initialized error modulation."""

    def __init__(
        self,
        d_inner: int,
        *,
        d_state: int,
        error_dim: int | None,
        dt_min: float,
        dt_max: float,
        alpha_max: float = math.log(4.0),
    ) -> None:
        super().__init__()
        if d_inner <= 0 or d_state <= 0:
            raise ValueError("d_inner and d_state must be positive")
        if dt_min <= 0 or dt_max <= dt_min:
            raise ValueError("dt_min must be positive and smaller than dt_max")
        if alpha_max <= 0:
            raise ValueError("alpha_max must be positive")
        if error_dim is not None and error_dim <= 0:
            raise ValueError("error_dim must be positive when provided")

        self.d_inner = d_inner
        self.d_state = d_state
        self.error_dim = error_dim
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        dt_ceiling = torch.tensor(dt_max, dtype=torch.float32)
        if float(dt_ceiling) > dt_max:
            dt_ceiling = torch.nextafter(dt_ceiling, torch.tensor(float("-inf")))
        self._dt_ceiling = float(dt_ceiling)
        self.alpha_max = float(alpha_max)
        self.dt_rank = math.ceil(d_inner / 16)

        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        bound = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -bound, bound)
        initial_dt = torch.exp(
            torch.empty(d_inner, dtype=torch.float32).uniform_(math.log(dt_min), math.log(dt_max))
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus(initial_dt))

        base = torch.arange(1, d_state + 1, dtype=torch.float32).log()
        self.a_log = nn.Parameter(base.repeat(d_inner, 1))
        self.d_skip = nn.Parameter(torch.ones(d_inner, dtype=torch.float32))

        if error_dim is None:
            self.error_norm = None
            self.error_proj = None
            self.register_parameter("alpha_raw", None)
        else:
            self.error_norm = RMSNorm(error_dim)
            self.error_proj = nn.Linear(error_dim, d_inner, bias=False)
            self.alpha_raw = nn.Parameter(torch.zeros(d_inner, dtype=torch.float32))

    def _project(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        projected = self.x_proj(u.float())
        return torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=-1)

    def _compute_dt_from_rank(self, dt_features: Tensor, error: Tensor | None) -> Tensor:
        base_dt = F.softplus(self.dt_proj(dt_features.float())).clamp(self.dt_min, self._dt_ceiling)
        if error is None:
            return base_dt
        if self.error_norm is None or self.error_proj is None or self.alpha_raw is None:
            raise ValueError("this SSM was constructed without an error pathway")
        if error.shape[:-1] != base_dt.shape[:-1] or error.shape[-1] != self.error_dim:
            raise ValueError(
                f"error must be B x T x {self.error_dim}, got {tuple(error.shape)}"
            )
        normalized = self.error_norm(error)
        projected_error = torch.tanh(self.error_proj(normalized))
        alpha = self.alpha_max * torch.tanh(self.alpha_raw)
        modulation = projected_error * alpha
        return (base_dt * torch.exp(modulation)).clamp(self.dt_min, self._dt_ceiling)

    def compute_dt(self, u: Tensor, error: Tensor | None = None) -> Tensor:
        dt_features, _, _ = self._project(u)
        return self._compute_dt_from_rank(dt_features, error)

    def forward(
        self,
        u: Tensor,
        error: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, SSMDiagnostics]:
        dt_features, b, c = self._project(u)
        dt = self._compute_dt_from_rank(dt_features, error)
        output = direct_selective_scan(u, dt, self.a_log, b, c, self.d_skip)
        if not return_diagnostics:
            return output

        zero = torch.zeros((), device=output.device, dtype=torch.float32)
        if error is None:
            error_rms, error_max = zero, zero
            error_finite = torch.ones((), device=output.device, dtype=torch.bool)
        else:
            error32 = error.float()
            error_rms = error32.square().mean().sqrt()
            error_max = error32.abs().max()
            error_finite = torch.isfinite(error32).all()
        diagnostics = SSMDiagnostics(
            dt_min=dt.min(),
            dt_max=dt.max(),
            error_rms=error_rms,
            error_max=error_max,
            output_rms=output.square().mean().sqrt(),
            output_max=output.abs().max(),
            finite=torch.isfinite(dt).all() & torch.isfinite(output).all() & error_finite,
        )
        return output, diagnostics


class TemporalMambaBlock(nn.Module):
    """Pre-norm causal Mamba block backed by the direct recurrence."""

    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        expand: int,
        error_dim: int | None,
        dt_min: float,
        dt_max: float,
        alpha_max: float,
        dropout: float,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__()
        if d_model <= 0 or expand <= 0 or conv_kernel <= 0:
            raise ValueError("d_model, expand, and conv_kernel must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        d_inner = d_model * expand
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.causal_conv = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=d_inner,
            bias=True,
        )
        self.ssm = DirectSelectiveSSM(
            d_inner,
            d_state=d_state,
            error_dim=error_dim,
            dt_min=dt_min,
            dt_max=dt_max,
            alpha_max=alpha_max,
        )
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        error: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, SSMDiagnostics]:
        residual = x.float()
        value, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        length = value.shape[1]
        value = self.causal_conv(value.transpose(1, 2))[..., :length].transpose(1, 2)
        value = F.silu(value)
        if return_diagnostics:
            value, diagnostics = self.ssm(value, error=error, return_diagnostics=True)
        else:
            value = self.ssm(value, error=error, return_diagnostics=False)
        output = residual + self.dropout(self.out_proj(value * F.silu(gate)))
        if return_diagnostics:
            return output, diagnostics
        return output
