from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .ssm import RMSNorm


@dataclass(frozen=True)
class CoordinateBatch:
    targets: Tensor
    mask: Tensor


def causal_coordinate_targets(signal: Tensor) -> CoordinateBatch:
    if signal.ndim != 3:
        raise ValueError("signal must be B x T x D")

    x = signal.float()
    dx = torch.zeros_like(x)
    ddx = torch.zeros_like(x)
    dx[:, 1:] = x[:, 1:] - x[:, :-1]
    ddx[:, 2:] = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]

    targets = torch.stack((x, dx, ddx), dim=2)
    mask = torch.ones((*x.shape[:2], 3, 1), device=x.device, dtype=torch.bool)
    mask[:, 0, 1:] = False
    if x.shape[1] > 1:
        mask[:, 1, 2] = False
    return CoordinateBatch(targets=targets, mask=mask)


class GeneralizedCoordinatePredictor(nn.Module):
    """Predict three same-width coordinate orders from a shared hidden state."""

    def __init__(self, hidden_dim: int, signal_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0 or signal_dim <= 0:
            raise ValueError("hidden_dim and signal_dim must be positive")
        self.norm = RMSNorm(hidden_dim)
        self.heads = nn.ModuleList(
            nn.Linear(hidden_dim, signal_dim) for _ in range(3)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        normalized = self.norm(hidden)
        return torch.stack(tuple(head(normalized) for head in self.heads), dim=2)


def aligned_coordinate_errors(
    hidden: Tensor,
    coordinates: CoordinateBatch,
    predictor: GeneralizedCoordinatePredictor,
) -> tuple[Tensor, Tensor]:
    """Align predictions at ``t-1`` with detached coordinate targets at ``t``."""

    if hidden.ndim != 3:
        raise ValueError("hidden must be B x T x H")
    targets = coordinates.targets
    if targets.ndim != 4 or targets.shape[2] != 3:
        raise ValueError("coordinate targets must be B x T x 3 x D")
    if hidden.shape[:2] != targets.shape[:2]:
        raise ValueError("hidden and coordinate targets must align on batch and time")
    if coordinates.mask.shape != (*targets.shape[:3], 1):
        raise ValueError("coordinate mask must be B x T x 3 x 1")

    predictions = predictor(hidden.detach())
    if predictions.shape != targets.shape:
        raise ValueError("predictor output must match coordinate target shape")

    detached_targets = targets.detach().float()
    errors = torch.zeros_like(detached_targets)
    errors[:, 1:] = detached_targets[:, 1:] - predictions[:, :-1]

    valid = coordinates.mask.to(device=errors.device, dtype=torch.bool).clone()
    valid[:, 0] = False
    expanded_valid = valid.expand_as(errors)
    squared_sum = torch.where(expanded_valid, errors.square(), 0.0).sum(
        dim=(0, 1, 3), keepdim=True
    )
    count = expanded_valid.sum(dim=(0, 1, 3), keepdim=True).clamp_min(1)
    rms = torch.sqrt(squared_sum / count).clamp_min(1e-6)
    errors = torch.where(expanded_valid, errors / rms, 0.0)
    return errors.float(), valid


def select_active_orders(errors: Tensor, active_order: int) -> Tensor:
    """Flatten three coordinate orders while zeroing inactive trailing orders."""

    if errors.ndim != 4 or errors.shape[2] != 3:
        raise ValueError("errors must be B x T x 3 x D")
    if active_order not in (1, 2, 3):
        raise ValueError("active_order must be 1, 2, or 3")

    selected = errors.float().clone()
    selected[:, :, active_order:] = 0.0
    return selected.flatten(start_dim=2)


def controlled_error(error: Tensor, mode: str, *, seed: int) -> Tensor:
    """Build a deterministic shuffled or moment-matched noise control."""

    if error.ndim != 3:
        raise ValueError("error must be B x T x F")
    value = error.float()
    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)

    if mode == "gc_k3_shuffled":
        permutation = torch.randperm(
            value.shape[0], device=value.device, generator=generator
        )
        return value[permutation]
    if mode == "gc_k3_noise":
        noise = torch.randn(
            value.shape,
            dtype=torch.float32,
            device=value.device,
            generator=generator,
        )
        noise_mean = noise.mean(dim=(0, 1), keepdim=True)
        noise_std = noise.std(dim=(0, 1), keepdim=True)
        standardized = (noise - noise_mean) / noise_std.clamp_min(1e-6)
        observed_mean = value.mean(dim=(0, 1), keepdim=True)
        observed_std = value.std(dim=(0, 1), keepdim=True)
        return standardized * observed_std + observed_mean
    raise ValueError(f"unsupported control mode: {mode!r}")
