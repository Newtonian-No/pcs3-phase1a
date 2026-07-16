"""Shared-weight one/two-pass Temporal Mamba model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import VARIANTS, ModelConfig
from .ssm import RMSNorm, SSMDiagnostics, TemporalMambaBlock


_ERROR_VARIANTS = {"error_inject", "error_aux", "time_shuffle", "time_reverse"}


@dataclass(frozen=True)
class TemporalModelOutput:
    logits: Tensor
    position_error: Tensor | None
    velocity_error: Tensor | None
    pass_count: int
    uses_error: bool
    diagnostics: dict[str, Tensor]


class NextStepPredictor(nn.Module):
    """Predict the next signal and its first difference from first-pass state."""

    def __init__(self, hidden_dim: int, signal_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0 or signal_dim <= 0:
            raise ValueError("hidden_dim and signal_dim must be positive")
        self.norm = RMSNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, 2 * signal_dim)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        prediction = self.projection(self.norm(hidden))
        return prediction.chunk(2, dim=-1)


def aligned_errors(
    hidden: Tensor,
    signal: Tensor,
    predictor: NextStepPredictor,
) -> tuple[Tensor, Tensor]:
    """Align predictions emitted at ``t-1`` with detached targets at ``t``."""

    if hidden.shape[:2] != signal.shape[:2]:
        raise ValueError("hidden and signal must align on batch and time")
    predicted_signal, predicted_velocity = predictor(hidden.detach())
    if predicted_signal.shape != signal.shape or predicted_velocity.shape != signal.shape:
        raise ValueError("predictor outputs must match signal shape")
    target = signal.detach().float()
    position = torch.zeros_like(target)
    velocity = torch.zeros_like(target)
    position[:, 1:] = target[:, 1:] - predicted_signal[:, :-1]
    delta = target[:, 1:] - target[:, :-1]
    velocity[:, 1:] = delta - predicted_velocity[:, :-1]
    return position, velocity


def _aggregate_diagnostics(diagnostics: list[SSMDiagnostics]) -> dict[str, Tensor]:
    if not diagnostics:
        raise ValueError("at least one SSM diagnostic is required")
    return {
        "dt_min": torch.stack([item.dt_min for item in diagnostics]).min(),
        "dt_max": torch.stack([item.dt_max for item in diagnostics]).max(),
        "error_rms": torch.stack([item.error_rms for item in diagnostics]).max(),
        "error_max": torch.stack([item.error_max for item in diagnostics]).max(),
        "output_rms": torch.stack([item.output_rms for item in diagnostics]).max(),
        "output_max": torch.stack([item.output_max for item in diagnostics]).max(),
        "finite": torch.stack([item.finite for item in diagnostics]).all(),
    }


class TemporalMambaModel(nn.Module):
    """Causal classifier with optional error-conditioned shared second pass."""

    def __init__(
        self,
        *,
        input_dim: int,
        signal_dim: int,
        num_outputs: int,
        model_config: ModelConfig,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or signal_dim <= 0 or num_outputs <= 0:
            raise ValueError("input_dim, signal_dim, and num_outputs must be positive")
        self.input_dim = input_dim
        self.signal_dim = signal_dim
        self.num_outputs = num_outputs
        self.model_config = model_config

        self.input_projection = nn.Linear(input_dim, model_config.d_model)
        self.layers = nn.ModuleList(
            [
                TemporalMambaBlock(
                    d_model=model_config.d_model,
                    d_state=model_config.d_state,
                    expand=model_config.expand,
                    error_dim=2 * signal_dim,
                    dt_min=model_config.dt_min,
                    dt_max=model_config.dt_max,
                    alpha_max=model_config.alpha_max,
                    dropout=model_config.dropout,
                )
                for _ in range(model_config.n_layers)
            ]
        )
        self.output_norm = RMSNorm(model_config.d_model)
        self.predictor = NextStepPredictor(model_config.d_model, signal_dim)
        self.classifier = nn.Linear(model_config.d_model, num_outputs)

    def _encode(
        self,
        features: Tensor,
        error: Tensor | None,
        *,
        return_diagnostics: bool,
    ) -> tuple[Tensor, list[SSMDiagnostics]]:
        hidden = self.input_projection(features.float())
        diagnostics: list[SSMDiagnostics] = []
        for layer in self.layers:
            if return_diagnostics:
                hidden, layer_diagnostics = layer(
                    hidden,
                    error=error,
                    return_diagnostics=True,
                )
                diagnostics.append(layer_diagnostics)
            else:
                hidden = layer(hidden, error=error, return_diagnostics=False)
        return hidden, diagnostics

    def forward(
        self,
        features: Tensor,
        signal: Tensor,
        *,
        variant: str,
        return_diagnostics: bool = False,
    ) -> TemporalModelOutput:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if features.ndim != 3 or signal.ndim != 3:
            raise ValueError("features and signal must be B x T x D")
        if features.shape[:2] != signal.shape[:2]:
            raise ValueError("features and signal must align on batch and time")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"features last dimension must be {self.input_dim}")
        if signal.shape[-1] != self.signal_dim:
            raise ValueError(f"signal last dimension must be {self.signal_dim}")

        first_hidden, first_diagnostics = self._encode(
            features,
            error=None,
            return_diagnostics=return_diagnostics,
        )
        position_error: Tensor | None = None
        velocity_error: Tensor | None = None
        uses_error = variant in _ERROR_VARIANTS
        all_diagnostics = first_diagnostics

        if variant == "vanilla":
            final_hidden = first_hidden
            pass_count = 1
        else:
            if uses_error:
                position_error, velocity_error = aligned_errors(first_hidden, signal, self.predictor)
                error = torch.cat([position_error, velocity_error], dim=-1)
            else:
                error = None
            final_hidden, second_diagnostics = self._encode(
                features,
                error=error,
                return_diagnostics=return_diagnostics,
            )
            all_diagnostics = first_diagnostics + second_diagnostics
            pass_count = 2

        final_hidden = self.output_norm(final_hidden)
        logits = self.classifier(final_hidden[:, -1])
        if return_diagnostics:
            diagnostics = _aggregate_diagnostics(all_diagnostics)
            diagnostics["finite"] = diagnostics["finite"] & torch.isfinite(logits).all()
        else:
            diagnostics = {"finite": torch.isfinite(logits).all()}
        return TemporalModelOutput(
            logits=logits,
            position_error=position_error,
            velocity_error=velocity_error,
            pass_count=pass_count,
            uses_error=uses_error,
            diagnostics=diagnostics,
        )
