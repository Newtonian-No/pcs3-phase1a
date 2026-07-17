"""Strict, immutable experiment configuration contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar


VARIANTS = (
    "vanilla",
    "two_pass",
    "error_inject",
    "error_aux",
    "time_shuffle",
    "time_reverse",
)
GC_VARIANTS = (
    "gc_k1",
    "gc_k2",
    "gc_k3",
    "gc_k3_shuffled",
    "gc_k3_noise",
)
GC_MATRIX_VARIANTS = ("vanilla", "two_pass") + GC_VARIANTS
GC_SEEDS = (42, 123, 777)
GC_CONFIRM_SEEDS = (42, 123, 777, 2026, 31415)
SUPPORTED_VARIANTS = VARIANTS + GC_VARIANTS
TRAINING_SEEDS = (42, 123, 777)
DATASETS = ("temporal_logic", "temporal_logic_v2", "uci_har", "generalized_dynamics")
INPUT_MODES = ("standard", "raw_concat", "query_bound")


@dataclass(frozen=True)
class DataConfig:
    train_size: int
    val_size: int
    test_size: int
    long_test_size: int
    validation_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_state: int
    n_layers: int
    expand: int
    dt_min: float
    dt_max: float
    alpha_max: float
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_fraction: float
    lambda_aux: float
    aux_warmup_fraction: float
    patience: int


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    data_seed: int
    signal_dim: int
    num_outputs: int
    seq_len: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    variant: str
    seed: int
    input_mode: str = "standard"
    generalized_coordinates: bool = False

    @property
    def pass_count(self) -> int:
        return 1 if self.variant == "vanilla" else 2

    @property
    def uses_error(self) -> bool:
        return self.variant in GC_VARIANTS or self.variant in {
            "error_inject",
            "error_aux",
            "time_shuffle",
            "time_reverse",
        }

    @property
    def uses_aux(self) -> bool:
        return self.variant in GC_VARIANTS or self.variant in {
            "error_aux",
            "time_shuffle",
            "time_reverse",
        }

    @property
    def uses_gc(self) -> bool:
        return self.generalized_coordinates

    @property
    def uses_gc_aux(self) -> bool:
        return self.variant in GC_VARIANTS

    @property
    def gc_order(self) -> int:
        return {
            "gc_k1": 1,
            "gc_k2": 2,
            "gc_k3": 3,
            "gc_k3_shuffled": 3,
            "gc_k3_noise": 3,
        }.get(self.variant, 0)

    @property
    def time_transform(self) -> str:
        return {
            "time_shuffle": "shuffle",
            "time_reverse": "reverse",
        }.get(self.variant, "none")


_T = TypeVar("_T", DataConfig, ModelConfig, TrainingConfig)


def _strict_dataclass(cls: type[_T], value: Any, field_name: str) -> _T:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    expected = {field.name for field in fields(cls)}
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ValueError(f"unknown {field_name} field: {sorted(unknown)[0]}")
    if missing:
        raise ValueError(f"missing {field_name} field: {sorted(missing)[0]}")
    try:
        return cls(**value)
    except TypeError as exc:
        raise ValueError(f"invalid {field_name}: {exc}") from exc


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> None:
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")


def _fraction(name: str, value: Any, *, upper_inclusive: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    valid = 0.0 <= float(value) <= 1.0 if upper_inclusive else 0.0 <= float(value) < 1.0
    if not valid:
        bracket = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise ValueError(f"{name} must be in {bracket}")


def _validate(config: ExperimentConfig) -> None:
    if config.dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {config.dataset!r}")
    if config.variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"variant must be one of {SUPPORTED_VARIANTS}, got {config.variant!r}"
        )
    if config.seed not in GC_CONFIRM_SEEDS:
        raise ValueError(
            f"seed must be one of {GC_CONFIRM_SEEDS}, got {config.seed!r}"
        )
    if config.input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {config.input_mode!r}")
    if not isinstance(config.generalized_coordinates, bool):
        raise ValueError("generalized_coordinates must be a boolean")
    if config.generalized_coordinates:
        if config.dataset not in {"generalized_dynamics", "uci_har"}:
            raise ValueError(
                "generalized_coordinates is only supported for generalized_dynamics or uci_har"
            )
        if config.variant not in GC_MATRIX_VARIANTS:
            raise ValueError(
                f"generalized_coordinates variant must be one of {GC_MATRIX_VARIANTS}, "
                f"got {config.variant!r}"
            )
    elif config.variant in GC_VARIANTS:
        raise ValueError(
            f"variant {config.variant!r} requires generalized_coordinates=true"
        )
    if config.dataset == "temporal_logic_v2" and config.variant not in {
        "vanilla",
        "two_pass",
        "error_inject",
        "error_aux",
    }:
        raise ValueError(f"variant {config.variant!r} is not supported for temporal_logic_v2")
    if config.dataset == "temporal_logic_v2" and config.input_mode not in {"raw_concat", "query_bound"}:
        raise ValueError("input_mode must be raw_concat or query_bound for temporal_logic_v2")

    _positive_int("data_seed", config.data_seed)
    _positive_int("signal_dim", config.signal_dim)
    _positive_int("num_outputs", config.num_outputs)
    _positive_int("seq_len", config.seq_len)
    for name in ("train_size", "val_size", "test_size", "long_test_size"):
        _positive_int(name, getattr(config.data, name), allow_zero=config.dataset == "uci_har")
    _fraction("validation_fraction", config.data.validation_fraction)
    if config.dataset in {"temporal_logic", "temporal_logic_v2"} and config.data.validation_fraction != 0.0:
        raise ValueError("validation_fraction must be 0 for temporal logic datasets")
    if config.dataset == "uci_har" and config.data.validation_fraction <= 0.0:
        raise ValueError("validation_fraction must be positive for uci_har")

    for name in ("d_model", "d_state", "n_layers", "expand"):
        _positive_int(name, getattr(config.model, name))
    if config.model.dt_min <= 0:
        raise ValueError("dt_min must be positive")
    if config.model.dt_max <= config.model.dt_min:
        raise ValueError("dt_min must be smaller than dt_max")
    if config.model.alpha_max <= 0:
        raise ValueError("alpha_max must be positive")
    _fraction("dropout", config.model.dropout)

    _positive_int("epochs", config.training.epochs)
    _positive_int("batch_size", config.training.batch_size)
    if config.training.lr <= 0:
        raise ValueError("lr must be positive")
    if config.training.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    _fraction("warmup_fraction", config.training.warmup_fraction, upper_inclusive=True)
    _fraction("aux_warmup_fraction", config.training.aux_warmup_fraction, upper_inclusive=True)
    if config.training.lambda_aux < 0:
        raise ValueError("lambda_aux must be non-negative")
    _positive_int("patience", config.training.patience)


def load_experiment_config(
    path: str | Path,
    *,
    variant: str,
    seed: int,
) -> ExperimentConfig:
    """Load a JSON config and reject every contract violation eagerly."""

    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid config file {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("config root must be an object")

    expected = {
        "dataset",
        "data_seed",
        "signal_dim",
        "num_outputs",
        "seq_len",
        "data",
        "model",
        "training",
        "input_mode",
        "generalized_coordinates",
    }
    unknown = set(raw) - expected
    missing = expected - {"input_mode", "generalized_coordinates"} - set(raw)
    if unknown:
        raise ValueError(f"unknown config field: {sorted(unknown)[0]}")
    if missing:
        raise ValueError(f"missing config field: {sorted(missing)[0]}")

    data = _strict_dataclass(DataConfig, raw["data"], "data")
    model = _strict_dataclass(ModelConfig, raw["model"], "model")
    training = _strict_dataclass(TrainingConfig, raw["training"], "training")
    config = ExperimentConfig(
        dataset=raw["dataset"],
        data_seed=raw["data_seed"],
        signal_dim=raw["signal_dim"],
        num_outputs=raw["num_outputs"],
        seq_len=raw["seq_len"],
        data=data,
        model=model,
        training=training,
        variant=variant,
        seed=seed,
        input_mode=raw.get("input_mode", "standard"),
        generalized_coordinates=raw.get("generalized_coordinates", False),
    )
    _validate(config)
    return config
