import json
from dataclasses import replace

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
