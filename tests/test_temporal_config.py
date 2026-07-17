import json
from pathlib import Path

import pytest

from temporal_mamba.config import (
    GC_CONFIRM_SEEDS,
    GC_MATRIX_VARIANTS,
    GC_SEEDS,
    GC_VARIANTS,
    TRAINING_SEEDS,
    VARIANTS,
    load_experiment_config,
)


def test_ablation_contract_is_exact():
    assert VARIANTS == (
        "vanilla",
        "two_pass",
        "error_inject",
        "error_aux",
        "time_shuffle",
        "time_reverse",
    )
    assert TRAINING_SEEDS == (42, 123, 777)


def _valid_config():
    return {
        "dataset": "temporal_logic",
        "data_seed": 20260716,
        "signal_dim": 8,
        "num_outputs": 1,
        "seq_len": 128,
        "data": {
            "train_size": 120,
            "val_size": 60,
            "test_size": 60,
            "long_test_size": 60,
            "validation_fraction": 0.0,
        },
        "model": {
            "d_model": 64,
            "d_state": 16,
            "n_layers": 4,
            "expand": 2,
            "dt_min": 0.001,
            "dt_max": 0.1,
            "alpha_max": 1.38629436112,
            "dropout": 0.1,
        },
        "training": {
            "epochs": 30,
            "batch_size": 128,
            "lr": 0.001,
            "weight_decay": 0.01,
            "warmup_fraction": 0.05,
            "lambda_aux": 0.1,
            "aux_warmup_fraction": 0.1,
            "patience": 8,
        },
    }


def test_gc_contract_constants_are_preregistered():
    assert GC_VARIANTS == (
        "gc_k1",
        "gc_k2",
        "gc_k3",
        "gc_k3_shuffled",
        "gc_k3_noise",
    )
    assert GC_MATRIX_VARIANTS == ("vanilla", "two_pass") + GC_VARIANTS
    assert GC_SEEDS == (42, 123, 777)
    assert GC_CONFIRM_SEEDS == (42, 123, 777, 2026, 31415)


def test_generalized_dynamics_gc_orders(tmp_path):
    raw = _valid_config()
    raw.update(
        dataset="generalized_dynamics",
        signal_dim=6,
        num_outputs=3,
        generalized_coordinates=True,
    )
    path = tmp_path / "gc.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    expected = {
        "gc_k1": 1,
        "gc_k2": 2,
        "gc_k3": 3,
        "gc_k3_shuffled": 3,
        "gc_k3_noise": 3,
    }
    for variant, order in expected.items():
        config = load_experiment_config(path, variant=variant, seed=42)
        assert config.uses_gc is True
        assert config.uses_gc_aux is True
        assert config.uses_error is True
        assert config.uses_aux is True
        assert config.gc_order == order


def test_gc_baselines_enable_same_modules(tmp_path):
    raw = _valid_config()
    raw.update(
        dataset="uci_har",
        signal_dim=9,
        num_outputs=6,
        generalized_coordinates=True,
    )
    raw["data"]["validation_fraction"] = 0.2
    path = tmp_path / "uci-gc.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_experiment_config(path, variant="vanilla", seed=42).uses_gc is True
    assert load_experiment_config(path, variant="two_pass", seed=42).uses_gc is True


def test_legacy_variants_remain_unchanged():
    assert VARIANTS == (
        "vanilla",
        "two_pass",
        "error_inject",
        "error_aux",
        "time_shuffle",
        "time_reverse",
    )


def test_generalized_coordinates_defaults_off(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(_valid_config()), encoding="utf-8")
    config = load_experiment_config(path, variant="vanilla", seed=42)
    assert config.generalized_coordinates is False
    assert config.uses_gc is False
    assert config.uses_gc_aux is False
    assert config.gc_order == 0


@pytest.mark.parametrize(
    ("dataset", "variant", "generalized_coordinates"),
    [
        ("temporal_logic", "vanilla", True),
        ("temporal_logic_v2", "vanilla", True),
        ("uci_har", "error_aux", True),
        ("generalized_dynamics", "time_reverse", True),
        ("generalized_dynamics", "gc_k1", False),
    ],
)
def test_rejects_invalid_gc_dataset_and_variant_combinations(
    tmp_path, dataset, variant, generalized_coordinates
):
    raw = _valid_config()
    raw.update(
        dataset=dataset,
        generalized_coordinates=generalized_coordinates,
    )
    if dataset == "uci_har":
        raw["data"]["validation_fraction"] = 0.2
    if dataset == "temporal_logic_v2":
        raw["input_mode"] = "query_bound"
    path = tmp_path / "invalid-gc.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="generalized_coordinates|variant"):
        load_experiment_config(path, variant=variant, seed=42)


def test_rejects_non_boolean_generalized_coordinates(tmp_path):
    raw = _valid_config()
    raw["generalized_coordinates"] = "yes"
    path = tmp_path / "invalid-gc-type.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="generalized_coordinates"):
        load_experiment_config(path, variant="vanilla", seed=42)


def test_v2_retains_existing_variant_restriction(tmp_path):
    raw = _valid_config()
    raw.update(dataset="temporal_logic_v2", input_mode="query_bound")
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="variant"):
        load_experiment_config(path, variant="time_shuffle", seed=42)


@pytest.mark.parametrize("seed", GC_CONFIRM_SEEDS)
def test_confirm_seeds_are_accepted_by_gc_config_contract(tmp_path, seed):
    raw = _valid_config()
    raw.update(dataset="generalized_dynamics", generalized_coordinates=True)
    path = tmp_path / "gc.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_experiment_config(path, variant="vanilla", seed=seed).seed == seed


@pytest.mark.parametrize("seed", [2026, 31415])
@pytest.mark.parametrize(
    ("dataset", "input_mode"),
    [("temporal_logic", None), ("temporal_logic_v2", "query_bound")],
)
def test_legacy_configs_reject_gc_only_confirm_seeds(
    tmp_path, seed, dataset, input_mode
):
    raw = _valid_config()
    raw["dataset"] = dataset
    if input_mode is not None:
        raw["input_mode"] = input_mode
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        load_experiment_config(path, variant="vanilla", seed=seed)


def test_variant_properties(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(_valid_config()), encoding="utf-8")
    cfg = load_experiment_config(path, variant="time_reverse", seed=42)
    assert cfg.pass_count == 2
    assert cfg.uses_error and cfg.uses_aux
    assert cfg.time_transform == "reverse"
    assert cfg.model.dt_min < cfg.model.dt_max
    assert cfg.input_mode == "standard"


@pytest.mark.parametrize(
    ("name", "input_mode"),
    [
        ("temporal_logic_v2.json", "query_bound"),
        ("temporal_logic_v2_raw.json", "raw_concat"),
    ],
)
def test_v2_repository_configs_select_explicit_input_mode(name, input_mode):
    path = Path(__file__).parents[1] / "configs" / name
    cfg = load_experiment_config(path, variant="vanilla", seed=42)
    assert cfg.dataset == "temporal_logic_v2"
    assert cfg.input_mode == input_mode
    assert cfg.data.validation_fraction == 0.0


@pytest.mark.parametrize("input_mode", ["standard", "unknown"])
def test_v2_rejects_non_v2_input_mode(tmp_path, input_mode):
    config = _valid_config()
    config.update(dataset="temporal_logic_v2", input_mode=input_mode)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="input_mode"):
        load_experiment_config(path, variant="vanilla", seed=42)


@pytest.mark.parametrize(
    ("variant", "passes", "uses_error", "uses_aux", "transform"),
    [
        ("vanilla", 1, False, False, "none"),
        ("two_pass", 2, False, False, "none"),
        ("error_inject", 2, True, False, "none"),
        ("error_aux", 2, True, True, "none"),
        ("time_shuffle", 2, True, True, "shuffle"),
        ("time_reverse", 2, True, True, "reverse"),
    ],
)
def test_every_variant_property(tmp_path, variant, passes, uses_error, uses_aux, transform):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(_valid_config()), encoding="utf-8")
    cfg = load_experiment_config(path, variant=variant, seed=42)
    assert cfg.pass_count == passes
    assert cfg.uses_error is uses_error
    assert cfg.uses_aux is uses_aux
    assert cfg.time_transform == transform


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda c: c.update(extra=True), "extra"),
        (lambda c: c.update(dataset="unknown"), "dataset"),
        (lambda c: c["model"].update(dt_min=0.2), "dt_min"),
        (lambda c: c["data"].update(validation_fraction=1.0), "validation_fraction"),
        (lambda c: c["training"].update(batch_size=0), "batch_size"),
    ],
)
def test_invalid_config_names_the_field(tmp_path, mutation, match):
    config = _valid_config()
    mutation(config)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_experiment_config(path, variant="vanilla", seed=42)


def test_rejects_unapproved_variant_and_seed(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(_valid_config()), encoding="utf-8")
    with pytest.raises(ValueError, match="variant"):
        load_experiment_config(path, variant="future", seed=42)
    with pytest.raises(ValueError, match="seed"):
        load_experiment_config(path, variant="vanilla", seed=1)
