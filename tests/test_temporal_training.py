import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from temporal_mamba.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from temporal_mamba.model import TemporalMambaModel, TemporalModelOutput
from temporal_mamba.train import (
    _classification_metrics,
    _guard_output,
    _move_batch,
    build_datasets,
    build_loaders,
    evaluate,
    overfit_tiny_batch,
    run_training,
    train_epoch,
    validation_selection_score,
)


class MemoryDataset(Dataset):
    def __init__(self, count=8, length=6, input_dim=20, signal_dim=4):
        generator = torch.Generator().manual_seed(41)
        self.features = torch.randn(count, length, input_dim, generator=generator)
        self.signal = torch.randn(count, length, signal_dim, generator=generator)
        self.target = (self.features[:, -1, 0] > 0).float()

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return {
            "features": self.features[index],
            "signal": self.signal[index],
            "target": self.target[index],
            "base_target": 1.0 - self.target[index],
            "sample_id": f"memory-{index}",
            "formula_family": "EVENTUALLY" if index % 2 == 0 else "BEFORE",
        }


def tiny_config(variant="vanilla", epochs=2):
    return ExperimentConfig(
        dataset="temporal_logic",
        data_seed=20260716,
        signal_dim=4,
        num_outputs=1,
        seq_len=6,
        data=DataConfig(8, 4, 4, 4, 0.0),
        model=ModelConfig(8, 3, 1, 1, 1e-3, 1e-1, 1.38629436112, 0.0),
        training=TrainingConfig(epochs, 8, 2e-3, 0.0, 0.05, 0.1, 0.5, 3),
        variant=variant,
        seed=42,
    )


def tiny_v2_config(variant="vanilla", epochs=1):
    return ExperimentConfig(
        dataset="temporal_logic_v2",
        data_seed=20260717,
        signal_dim=8,
        num_outputs=1,
        seq_len=16,
        data=DataConfig(12, 12, 12, 12, 0.0),
        model=ModelConfig(8, 3, 1, 1, 1e-3, 1e-1, 1.38629436112, 0.0),
        training=TrainingConfig(epochs, 6, 2e-3, 0.0, 0.05, 0.1, 0.5, 2),
        variant=variant,
        seed=42,
        input_mode="query_bound",
    )


def tiny_gc_config(dataset="generalized_dynamics", variant="gc_k3", epochs=1):
    return ExperimentConfig(
        dataset=dataset,
        data_seed=20260717,
        signal_dim=3 if dataset == "generalized_dynamics" else 9,
        num_outputs=3 if dataset == "generalized_dynamics" else 6,
        seq_len=16 if dataset == "generalized_dynamics" else 128,
        data=DataConfig(6, 3, 3, 3, 0.0 if dataset == "generalized_dynamics" else 0.2),
        model=ModelConfig(8, 3, 1, 1, 1e-3, 1e-1, 1.38629436112, 0.0),
        training=TrainingConfig(epochs, 3, 2e-3, 0.0, 0.05, 0.1, 0.5, 2),
        variant=variant,
        seed=42,
        generalized_coordinates=True,
    )


@pytest.mark.parametrize(
    ("variant", "passes", "uses_error", "uses_aux"),
    [
        ("vanilla", 1, False, False),
        ("two_pass", 2, False, False),
        ("error_inject", 2, True, False),
        ("error_aux", 2, True, True),
        ("time_shuffle", 2, True, True),
        ("time_reverse", 2, True, True),
    ],
)
def test_one_train_epoch_updates_parameters_and_records_contract(
    variant, passes, uses_error, uses_aux
):
    config = tiny_config(variant=variant)
    model = TemporalMambaModel(
        input_dim=20,
        signal_dim=4,
        num_outputs=1,
        model_config=config.model,
    )
    loader = DataLoader(MemoryDataset(), batch_size=8, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = train_epoch(
        model,
        loader,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
        epoch=1,
        global_step=0,
    )
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert metrics["pass_count"] == passes
    assert metrics["uses_error"] is uses_error
    assert (metrics["aux_weight"] > 0) is uses_aux
    assert metrics["global_step"] == 1
    assert metrics["finite"] is True


class EchoModel(torch.nn.Module):
    def forward(self, features, signal, *, variant, query=None, return_diagnostics=False):
        del signal, variant, query, return_diagnostics
        logits = features[:, -1, :1] * 10
        zero = logits.new_zeros(())
        return TemporalModelOutput(
            logits=logits,
            position_error=None,
            velocity_error=None,
            pass_count=1,
            uses_error=False,
            diagnostics={
                "dt_min": zero + 0.001,
                "dt_max": zero + 0.1,
                "error_rms": zero,
                "error_max": zero,
                "output_rms": zero + 1,
                "output_max": zero + 1,
                "finite": torch.tensor(True),
            },
        )


class GCCaptureModel(torch.nn.Module):
    def __init__(self, num_outputs=3):
        super().__init__()
        self.projection = torch.nn.Linear(1, num_outputs)
        self.seeds = []

    def forward(
        self,
        features,
        signal,
        *,
        variant,
        query=None,
        return_diagnostics=False,
        coordinate_targets=None,
        coordinate_mask=None,
        error_control_seed=None,
    ):
        del variant, query, return_diagnostics
        self.seeds.append(error_control_seed)
        logits = self.projection(features[:, -1, :1])
        zero = logits.new_zeros(())
        errors = torch.zeros(
            signal.shape[0], signal.shape[1], 3 * signal.shape[2], device=signal.device
        )
        return TemporalModelOutput(
            logits=logits,
            position_error=None,
            velocity_error=None,
            pass_count=2,
            uses_error=True,
            diagnostics={
                "dt_min": zero + 0.001,
                "dt_max": zero + 0.1,
                "error_rms": zero,
                "error_max": zero,
                "output_rms": zero + 1,
                "output_max": zero + 1,
                "finite": torch.tensor(True),
            },
            coordinate_errors=errors,
            coordinate_mask=coordinate_mask,
            gc_order=3,
        )


class GCMemoryDataset(Dataset):
    def __init__(self, count=4, length=6, signal_dim=3):
        generator = torch.Generator().manual_seed(9)
        self.signal = torch.randn(count, length, signal_dim, generator=generator)
        self.features = torch.cat(
            (self.signal, torch.linspace(0, 1, length).view(1, length, 1).expand(count, -1, -1)),
            dim=-1,
        )
        self.targets = torch.arange(count) % 3

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        signal = self.signal[index]
        coordinates = torch.stack((signal, torch.zeros_like(signal), torch.zeros_like(signal)), dim=1)
        return {
            "features": self.features[index],
            "signal": signal,
            "coordinate_targets": coordinates,
            "coordinate_mask": torch.ones(signal.shape[0], 3, 1, dtype=torch.bool),
            "target": self.targets[index],
            "base_target": self.targets[index],
        }


def test_evaluate_reports_recomputed_frozen_and_family_metrics():
    dataset = MemoryDataset(count=8)
    dataset.features[:, -1, 0] = torch.where(dataset.target > 0, 1.0, -1.0)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    metrics = evaluate(EchoModel(), loader, tiny_config(), device=torch.device("cpu"))
    assert metrics["accuracy"] == 1.0
    assert metrics["frozen_label_metrics"]["accuracy"] == 0.0
    assert set(metrics["per_family"]) == {"EVENTUALLY", "BEFORE"}
    assert metrics["diagnostics"]["dt_min"] >= 1e-3
    assert metrics["diagnostics"]["dt_max"] <= 1e-1


def test_gc_train_and_evaluation_pass_deterministic_control_seeds():
    config = replace(tiny_gc_config(variant="gc_k3_noise"), training=replace(tiny_gc_config().training, batch_size=2))
    loader = DataLoader(GCMemoryDataset(), batch_size=2, shuffle=False)
    model = GCCaptureModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training = train_epoch(
        model,
        loader,
        optimizer,
        None,
        config,
        device=torch.device("cpu"),
        epoch=0,
        global_step=7,
    )
    base = config.seed * 1_000_003
    assert model.seeds == [base + 7, base + 8]
    assert training["diagnostics"]["error_control_seed_formula"] == (
        "config.seed * 1_000_003 + global_step"
    )
    assert training["diagnostics"]["error_control_seed"] == base + 8

    model.seeds.clear()
    evaluated = evaluate(model, loader, config, device=torch.device("cpu"))
    assert model.seeds == [base, base + 1]
    assert evaluated["diagnostics"]["error_control_seed_formula"] == (
        "config.seed * 1_000_003 + batch_index"
    )
    assert evaluated["diagnostics"]["error_control_seed"] == base + 1


def test_classification_metrics_use_configured_class_count():
    metrics = _classification_metrics(
        "generalized_dynamics",
        torch.tensor([0, 1, 2]).numpy(),
        torch.tensor([0, 1, 2]).numpy(),
        num_classes=3,
    )
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_gc_output_guard_rejects_nonfinite_and_bad_shapes():
    zero = torch.tensor(0.0)
    valid = TemporalModelOutput(
        logits=torch.zeros(2, 3),
        position_error=None,
        velocity_error=None,
        pass_count=2,
        uses_error=True,
        diagnostics={"finite": torch.tensor(True)},
        coordinate_errors=torch.zeros(2, 4, 9),
        coordinate_mask=torch.ones(2, 4, 3, 1, dtype=torch.bool),
        gc_order=3,
    )
    _guard_output(valid)
    with pytest.raises(ValueError, match="coordinate_errors"):
        _guard_output(replace(valid, coordinate_errors=torch.zeros(2, 4, 3, 3)))
    bad = valid.coordinate_errors.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(Exception, match="coordinate_errors"):
        _guard_output(replace(valid, coordinate_errors=bad))


def test_gc_models_are_enabled_for_all_matrix_variants():
    config = tiny_gc_config(variant="vanilla")
    counts = []
    for variant in ("vanilla", "two_pass", "gc_k1", "gc_k2", "gc_k3", "gc_k3_shuffled", "gc_k3_noise"):
        variant_config = replace(config, variant=variant)
        model = TemporalMambaModel(
            input_dim=variant_config.signal_dim + 1,
            signal_dim=variant_config.signal_dim,
            num_outputs=variant_config.num_outputs,
            model_config=variant_config.model,
            generalized_coordinates=variant_config.uses_gc,
        )
        assert model.generalized_coordinates is True
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == 1


def test_loader_shuffle_is_seeded():
    dataset = MemoryDataset(count=16)
    config = replace(tiny_config(), data=DataConfig(16, 4, 4, 4, 0.0))
    loaders_a, _ = build_loaders({"train": dataset}, config, num_workers=0)
    loaders_b, _ = build_loaders({"train": dataset}, config, num_workers=0)
    ids_a = next(iter(loaders_a["train"]))["sample_id"]
    ids_b = next(iter(loaders_b["train"]))["sample_id"]
    assert ids_a == ids_b


def test_build_v2_datasets_has_all_required_views(tmp_path):
    datasets = build_datasets(tiny_v2_config(), tmp_path / "data")
    assert set(datasets) == {"train", "val", "test", "long_test", "channel_ood"}


def test_build_generalized_dynamics_datasets_has_all_required_splits(tmp_path):
    datasets = build_datasets(tiny_gc_config(), tmp_path / "dynamics")
    assert set(datasets) == {
        "train",
        "val",
        "test",
        "length_256",
        "length_512",
        "parameter_ood",
        "noise_ood",
    }
    assert {name: len(dataset) for name, dataset in datasets.items()} == {
        "train": 6,
        "val": 3,
        "test": 3,
        "length_256": 3,
        "length_512": 3,
        "parameter_ood": 3,
        "noise_ood": 3,
    }


def test_uci_gc_adds_only_preregistered_evaluation_views(tmp_path, monkeypatch):
    root = tmp_path / "uci"
    root.mkdir()
    (root / "manifest.json").write_text('{"data_seed": 20260717}', encoding="utf-8")
    import numpy as np

    arrays = {
        "signal": np.zeros((1, 128, 9), dtype=np.float32),
        "target": np.zeros(1, dtype=np.int64),
        "subject": np.ones(1, dtype=np.int16),
        "sample_id": np.asarray(["sample-0"]),
    }
    for split in ("train", "val", "test"):
        np.savez(root / f"{split}.npz", **arrays)

    gc_datasets = build_datasets(tiny_gc_config(dataset="uci_har"), root)
    legacy_datasets = build_datasets(
        replace(tiny_gc_config(dataset="uci_har"), generalized_coordinates=False, variant="vanilla"),
        root,
    )

    assert set(gc_datasets) == {"train", "val", "test", "prefix50", "noise_025"}
    assert gc_datasets["prefix50"].split == "test"
    assert gc_datasets["prefix50"].transform == "prefix50"
    assert gc_datasets["noise_025"].split == "test"
    assert gc_datasets["noise_025"].transform == "noise_025"
    assert set(legacy_datasets) == {"train", "val", "test"}


def test_move_batch_preserves_optional_query():
    moved = _move_batch(
        {
            "features": torch.randn(2, 8, 34),
            "signal": torch.randn(2, 8, 8),
            "query": torch.randn(2, 25),
            "target": torch.tensor([0.0, 1.0]),
        },
        torch.device("cpu"),
    )
    assert moved.query is not None and moved.query.shape == (2, 25)
    assert moved.features.shape == (2, 8, 34)
    assert moved.signal.shape == (2, 8, 8)
    assert moved.target.shape == (2,)


def test_move_batch_preserves_optional_coordinate_tensors():
    moved = _move_batch(
        {
            "features": torch.randn(2, 16, 7),
            "signal": torch.randn(2, 16, 6),
            "coordinate_targets": torch.randn(2, 16, 3, 6),
            "coordinate_mask": torch.ones(2, 16, 3, 1, dtype=torch.bool),
            "target": torch.tensor([0, 2]),
        },
        torch.device("cpu"),
    )
    assert moved.coordinate_targets.shape == (2, 16, 3, 6)
    assert moved.coordinate_targets.dtype == torch.float32
    assert moved.coordinate_mask.shape == (2, 16, 3, 1)
    assert moved.coordinate_mask.dtype == torch.bool


def test_move_batch_rejects_nonfinite_coordinate_mask():
    mask = torch.ones(2, 16, 3, 1)
    mask[0, 0, 0, 0] = float("nan")
    with pytest.raises(Exception, match="coordinate_mask"):
        _move_batch(
            {
                "features": torch.randn(2, 16, 7),
                "signal": torch.randn(2, 16, 6),
                "coordinate_targets": torch.randn(2, 16, 3, 6),
                "coordinate_mask": mask,
                "target": torch.tensor([0, 2]),
            },
            torch.device("cpu"),
        )


def test_validation_selection_score_ignores_test_metrics():
    validation = {"accuracy": 0.7}
    assert validation_selection_score(validation) == 0.7


def test_tiny_batch_gate_can_fit_an_easy_batch():
    config = tiny_config(epochs=3)
    model = TemporalMambaModel(
        input_dim=20,
        signal_dim=4,
        num_outputs=1,
        model_config=config.model,
    )
    batch = next(iter(DataLoader(MemoryDataset(count=4), batch_size=4)))
    batch["target"].zero_()
    result = overfit_tiny_batch(
        model,
        batch,
        config,
        device=torch.device("cpu"),
        max_steps=80,
        target_accuracy=1.0,
    )
    assert result["passed"] is True
    assert result["accuracy"] == 1.0


def test_run_training_writes_complete_artifact_schema(tmp_path):
    raw = {
        "dataset": "temporal_logic",
        "data_seed": 20260716,
        "signal_dim": 8,
        "num_outputs": 1,
        "seq_len": 16,
        "data": {
            "train_size": 12,
            "val_size": 12,
            "test_size": 12,
            "long_test_size": 12,
            "validation_fraction": 0.0,
        },
        "model": {
            "d_model": 8,
            "d_state": 3,
            "n_layers": 1,
            "expand": 1,
            "dt_min": 0.001,
            "dt_max": 0.1,
            "alpha_max": 1.38629436112,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 6,
            "lr": 0.002,
            "weight_decay": 0.0,
            "warmup_fraction": 0.05,
            "lambda_aux": 0.1,
            "aux_warmup_fraction": 0.1,
            "patience": 2,
        },
    }
    config_path = tmp_path / "tiny.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    final = run_training(
        config_path=config_path,
        variant="vanilla",
        seed=42,
        data_root=tmp_path / "data",
        artifact_root=tmp_path / "artifacts",
        device="cpu",
    )
    run_dir = tmp_path / "artifacts" / "temporal_logic-vanilla-seed42"
    for name in (
        "config.json",
        "environment.json",
        "dataset_manifest.json",
        "history.jsonl",
        "best.pt",
        "last.pt",
        "final.json",
    ):
        assert (run_dir / name).exists(), name
    assert final["status"] == "complete"
    assert final["run_id"] == "temporal_logic-vanilla-seed42"
    assert final["selection_split"] == "val"


def test_gc_configuration_files_are_exact():
    generalized = json.loads(
        (Path("configs") / "generalized_dynamics_gc.json").read_text(encoding="utf-8")
    )
    uci = json.loads((Path("configs") / "uci_har_gc.json").read_text(encoding="utf-8"))
    assert generalized == {
        "dataset": "generalized_dynamics",
        "generalized_coordinates": True,
        "input_mode": "standard",
        "data_seed": 20260717,
        "signal_dim": 6,
        "num_outputs": 3,
        "seq_len": 128,
        "data": {"train_size": 18000, "val_size": 3000, "test_size": 3000, "long_test_size": 3000, "validation_fraction": 0.0},
        "model": {"d_model": 64, "d_state": 16, "n_layers": 4, "expand": 2, "dt_min": 0.001, "dt_max": 0.1, "alpha_max": 1.38629436112, "dropout": 0.1},
        "training": {"epochs": 40, "batch_size": 128, "lr": 0.001, "weight_decay": 0.01, "warmup_fraction": 0.05, "lambda_aux": 0.1, "aux_warmup_fraction": 0.1, "patience": 10},
    }
    assert uci == {
        "dataset": "uci_har",
        "generalized_coordinates": True,
        "data_seed": 20260716,
        "signal_dim": 9,
        "num_outputs": 6,
        "seq_len": 128,
        "data": {"train_size": 0, "val_size": 0, "test_size": 0, "long_test_size": 0, "validation_fraction": 0.2},
        "model": {"d_model": 96, "d_state": 16, "n_layers": 4, "expand": 2, "dt_min": 0.001, "dt_max": 0.1, "alpha_max": 1.38629436112, "dropout": 0.1},
        "training": {"epochs": 40, "batch_size": 64, "lr": 0.0005, "weight_decay": 0.01, "warmup_fraction": 0.05, "lambda_aux": 0.1, "aux_warmup_fraction": 0.1, "patience": 10},
    }


def test_gc_final_uses_schema_v3_and_preregistered_views(tmp_path):
    raw = {
        "dataset": "generalized_dynamics",
        "generalized_coordinates": True,
        "input_mode": "standard",
        "data_seed": 20260717,
        "signal_dim": 3,
        "num_outputs": 3,
        "seq_len": 16,
        "data": {"train_size": 6, "val_size": 3, "test_size": 3, "long_test_size": 3, "validation_fraction": 0.0},
        "model": {"d_model": 8, "d_state": 3, "n_layers": 1, "expand": 1, "dt_min": 0.001, "dt_max": 0.1, "alpha_max": 1.38629436112, "dropout": 0.0},
        "training": {"epochs": 1, "batch_size": 3, "lr": 0.002, "weight_decay": 0.0, "warmup_fraction": 0.05, "lambda_aux": 0.1, "aux_warmup_fraction": 0.1, "patience": 2},
    }
    config_path = tmp_path / "gc.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    final = run_training(
        config_path=config_path,
        variant="gc_k1",
        seed=42,
        data_root=tmp_path / "data",
        artifact_root=tmp_path / "artifacts",
        device="cpu",
    )
    assert final["schema_version"] == 3
    assert final["gc_order"] == 1
    assert final["parameter_count"] > 0
    assert set(final["metrics"]) == {
        "val", "test", "length_256", "length_512", "parameter_ood", "noise_ood"
    }
    assert set(final["per_order_auxiliary_losses"]) == {"order_0", "order_1", "order_2"}
    assert set(final["per_order_error_rms"]) == {"order_0", "order_1", "order_2"}
    assert set(final["dt_diagnostics"]) == set(final["metrics"])
    assert final["hashes"] == {
        "git_commit": final["git_commit"],
        "config_sha256": final["config_hash"],
        "manifest_sha256": final["dataset_manifest_hash"],
    }
    environment = json.loads(
        (tmp_path / "artifacts" / "generalized_dynamics-gc_k1-seed42" / "environment.json").read_text(encoding="utf-8")
    )
    assert final["environment"] == environment


def test_v2_final_contains_ood_and_frozen_control_metrics(tmp_path):
    raw = {
        "dataset": "temporal_logic_v2",
        "input_mode": "query_bound",
        "data_seed": 20260717,
        "signal_dim": 8,
        "num_outputs": 1,
        "seq_len": 16,
        "data": {
            "train_size": 12,
            "val_size": 12,
            "test_size": 12,
            "long_test_size": 12,
            "validation_fraction": 0.0,
        },
        "model": {
            "d_model": 8,
            "d_state": 3,
            "n_layers": 1,
            "expand": 1,
            "dt_min": 0.001,
            "dt_max": 0.1,
            "alpha_max": 1.38629436112,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 6,
            "lr": 0.002,
            "weight_decay": 0.0,
            "warmup_fraction": 0.05,
            "lambda_aux": 0.1,
            "aux_warmup_fraction": 0.1,
            "patience": 2,
        },
    }
    config_path = tmp_path / "tiny-v2.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    final = run_training(
        config_path=config_path,
        variant="vanilla",
        seed=42,
        data_root=tmp_path / "data-v2",
        artifact_root=tmp_path / "artifacts-v2",
        device="cpu",
    )

    assert final["schema_version"] == 2
    assert set(final["metrics"]) == {
        "val",
        "test",
        "long_test",
        "channel_ood",
        "reverse_frozen",
        "shuffle_frozen",
    }
    assert set(final["metrics"]["channel_ood"]["per_family"]) == {
        "EVENTUALLY",
        "BEFORE",
        "UNTIL",
        "BOUNDED_RESPONSE",
        "COUNT_WITHIN",
        "GAP",
    }
