"""Shared-weight one/two-pass Temporal Mamba model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import GC_MATRIX_VARIANTS, GC_VARIANTS, INPUT_MODES, VARIANTS, ModelConfig
from .generalized_coordinates import (
    CoordinateBatch,
    GeneralizedCoordinatePredictor,
    aligned_coordinate_errors,
    controlled_error,
    select_active_orders,
)
from .query_binding import BoundedQueryFiLM, TemporalQueryBinder
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
    coordinate_errors: Tensor | None = None
    coordinate_mask: Tensor | None = None
    gc_order: int = 0


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
        input_mode: str = "standard",
        generalized_coordinates: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or signal_dim <= 0 or num_outputs <= 0:
            raise ValueError("input_dim, signal_dim, and num_outputs must be positive")
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")
        self.input_dim = input_dim
        self.signal_dim = signal_dim
        self.num_outputs = num_outputs
        self.model_config = model_config
        self.input_mode = input_mode
        self.generalized_coordinates = generalized_coordinates

        query_bound = input_mode == "query_bound"
        encoder_input_dim = 5 if query_bound else input_dim
        prediction_signal_dim = 2 if query_bound else signal_dim
        self.query_binder = TemporalQueryBinder(signal_dim) if query_bound else None
        self.query_film = (
            BoundedQueryFiLM(9, model_config.n_layers, model_config.d_model)
            if query_bound
            else None
        )
        self.input_projection = nn.Linear(encoder_input_dim, model_config.d_model)
        self.layers = nn.ModuleList(
            [
                TemporalMambaBlock(
                    d_model=model_config.d_model,
                    d_state=model_config.d_state,
                    expand=model_config.expand,
                    error_dim=(
                        3 * signal_dim
                        if generalized_coordinates
                        else 2 * prediction_signal_dim
                    ),
                    dt_min=model_config.dt_min,
                    dt_max=model_config.dt_max,
                    alpha_max=model_config.alpha_max,
                    dropout=model_config.dropout,
                )
                for _ in range(model_config.n_layers)
            ]
        )
        self.output_norm = RMSNorm(model_config.d_model)
        if generalized_coordinates:
            self.predictor = GeneralizedCoordinatePredictor(
                model_config.d_model, signal_dim
            )
        else:
            self.predictor = NextStepPredictor(
                model_config.d_model, prediction_signal_dim
            )
        classifier_dim = 3 * model_config.d_model if query_bound else model_config.d_model
        self.classifier = nn.Linear(classifier_dim, num_outputs)

    def _encode(
        self,
        features: Tensor,
        error: Tensor | None,
        film: tuple[Tensor, Tensor] | None,
        *,
        return_diagnostics: bool,
    ) -> tuple[Tensor, list[SSMDiagnostics]]:
        hidden = self.input_projection(features.float())
        diagnostics: list[SSMDiagnostics] = []
        for index, layer in enumerate(self.layers):
            if film is not None:
                scale, shift = film
                hidden = hidden * scale[:, index : index + 1, :] + shift[:, index : index + 1, :]
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
        query: Tensor | None = None,
        return_diagnostics: bool = False,
        coordinate_targets: Tensor | None = None,
        coordinate_mask: Tensor | None = None,
        error_control_seed: int | None = None,
    ) -> TemporalModelOutput:
        allowed_variants = GC_MATRIX_VARIANTS if self.generalized_coordinates else VARIANTS
        if variant not in allowed_variants:
            raise ValueError(
                f"variant must be one of {allowed_variants}, got {variant!r}"
            )
        if features.ndim != 3 or signal.ndim != 3:
            raise ValueError("features and signal must be B x T x D")
        if features.shape[:2] != signal.shape[:2]:
            raise ValueError("features and signal must align on batch and time")
        if signal.shape[-1] != self.signal_dim:
            raise ValueError(f"signal last dimension must be {self.signal_dim}")
        if self.input_mode == "query_bound":
            if query is None:
                raise ValueError("query_bound input requires query")
            assert self.query_binder is not None and self.query_film is not None
            bound = self.query_binder(signal, query)
            encoded_features = bound.sequence
            prediction_signal = bound.prediction_signal
            film = self.query_film(bound.condition)
        else:
            if features.shape[-1] != self.input_dim:
                raise ValueError(f"features last dimension must be {self.input_dim}")
            encoded_features = features
            prediction_signal = signal
            film = None

        first_hidden, first_diagnostics = self._encode(
            encoded_features,
            error=None,
            film=film,
            return_diagnostics=return_diagnostics,
        )
        position_error: Tensor | None = None
        velocity_error: Tensor | None = None
        coordinate_errors: Tensor | None = None
        valid_coordinate_mask: Tensor | None = None
        gc_order = 0
        uses_error = (
            variant in GC_VARIANTS
            if self.generalized_coordinates
            else variant in _ERROR_VARIANTS
        )
        all_diagnostics = first_diagnostics

        if variant == "vanilla":
            final_hidden = first_hidden
            pass_count = 1
        else:
            if self.generalized_coordinates and uses_error:
                if coordinate_targets is None:
                    raise ValueError(f"variant {variant} requires coordinate_targets")
                if coordinate_mask is None:
                    raise ValueError(f"variant {variant} requires coordinate_mask")
                assert isinstance(self.predictor, GeneralizedCoordinatePredictor)
                errors_by_order, valid_coordinate_mask = aligned_coordinate_errors(
                    first_hidden,
                    CoordinateBatch(
                        targets=coordinate_targets,
                        mask=coordinate_mask,
                    ),
                    self.predictor,
                )
                gc_order = {
                    "gc_k1": 1,
                    "gc_k2": 2,
                    "gc_k3": 3,
                    "gc_k3_shuffled": 3,
                    "gc_k3_noise": 3,
                }[variant]
                coordinate_errors = select_active_orders(errors_by_order, gc_order)
                error = coordinate_errors
                if variant in {"gc_k3_shuffled", "gc_k3_noise"}:
                    if error_control_seed is None:
                        raise ValueError(
                            f"variant {variant} requires error_control_seed"
                        )
                    error = controlled_error(
                        error,
                        variant,
                        valid=valid_coordinate_mask,
                        seed=error_control_seed,
                    )
            elif not self.generalized_coordinates and uses_error:
                assert isinstance(self.predictor, NextStepPredictor)
                position_error, velocity_error = aligned_errors(
                    first_hidden,
                    prediction_signal,
                    self.predictor,
                )
                error = torch.cat([position_error, velocity_error], dim=-1)
            else:
                error = None
            final_hidden, second_diagnostics = self._encode(
                encoded_features,
                error=error,
                film=film,
                return_diagnostics=return_diagnostics,
            )
            all_diagnostics = first_diagnostics + second_diagnostics
            pass_count = 2

        final_hidden = self.output_norm(final_hidden)
        if self.input_mode == "query_bound":
            final = final_hidden[:, -1]
            prefix_max = torch.cummax(final_hidden, dim=1).values[:, -1]
            prefix_sum = torch.cumsum(final_hidden, dim=1)
            denominator = torch.arange(
                1,
                final_hidden.shape[1] + 1,
                device=final_hidden.device,
                dtype=torch.float32,
            )
            prefix_mean = (prefix_sum / denominator[None, :, None])[:, -1]
            readout = torch.cat((final, prefix_max, prefix_mean), dim=-1)
        else:
            readout = final_hidden[:, -1]
        logits = self.classifier(readout)
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
            coordinate_errors=coordinate_errors,
            coordinate_mask=valid_coordinate_mask,
            gc_order=gc_order,
        )
