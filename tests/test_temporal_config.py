import json

import pytest

from temporal_mamba.config import TRAINING_SEEDS, VARIANTS, load_experiment_config


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


def test_variant_properties(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(_valid_config()), encoding="utf-8")
    cfg = load_experiment_config(path, variant="time_reverse", seed=42)
    assert cfg.pass_count == 2
    assert cfg.uses_error and cfg.uses_aux
    assert cfg.time_transform == "reverse"
    assert cfg.model.dt_min < cfg.model.dt_max


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
