import json
import random

import numpy as np
import pytest
import torch

from temporal_mamba.checkpoint import load_checkpoint, save_checkpoint
from temporal_mamba.numerics import (
    NumericalFailure,
    assert_finite_model,
    assert_finite_tensor,
    write_failure_artifact,
)


def _step(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(5, 3)
    targets = torch.randn(5, 2)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def test_checkpoint_restores_rng_and_exact_next_update(tmp_path):
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    loader_generator = torch.Generator().manual_seed(31)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.95**step)
    _step(model, optimizer, scheduler)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        loader_generator=loader_generator,
        epoch=2,
        step=7,
        best_metric=0.8,
        history_cursor=3,
        config_hash="config-a",
        git_commit="abc123",
        dataset_manifest_hash="data-a",
    )
    assert path.exists()
    assert not (tmp_path / "checkpoint.pt.tmp").exists()

    expected_rng = (
        random.random(),
        float(np.random.rand()),
        torch.rand(3),
        torch.rand(3, generator=loader_generator),
    )
    expected_loss = _step(model, optimizer, scheduler)
    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    for parameter in model.parameters():
        parameter.data.uniform_(-100, 100)
    _step(model, optimizer, scheduler)
    random.random()
    np.random.rand()
    torch.rand(9)
    torch.rand(9, generator=loader_generator)

    checkpoint = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        loader_generator=loader_generator,
        expected_config_hash="config-a",
        expected_dataset_manifest_hash="data-a",
    )
    assert checkpoint["epoch"] == 2
    assert checkpoint["step"] == 7
    actual_rng = (
        random.random(),
        float(np.random.rand()),
        torch.rand(3),
        torch.rand(3, generator=loader_generator),
    )
    assert actual_rng[0] == expected_rng[0]
    assert actual_rng[1] == expected_rng[1]
    assert torch.equal(actual_rng[2], expected_rng[2])
    assert torch.equal(actual_rng[3], expected_rng[3])
    assert _step(model, optimizer, scheduler) == expected_loss
    for actual, expected in zip(model.parameters(), expected_parameters):
        assert torch.equal(actual, expected)


def test_checkpoint_rejects_metadata_mismatch(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        loader_generator=None,
        epoch=0,
        step=0,
        best_metric=0.0,
        history_cursor=0,
        config_hash="expected",
        git_commit="abc",
        dataset_manifest_hash="dataset",
    )
    with pytest.raises(ValueError, match="config_hash"):
        load_checkpoint(path, model=model, expected_config_hash="different")


def test_numerical_guards_name_the_failing_component_and_write_artifact(tmp_path):
    bad_dt = torch.tensor([0.01, float("nan")])
    with pytest.raises(NumericalFailure, match="diagnostics.dt") as caught:
        assert_finite_tensor("diagnostics.dt", bad_dt)
    failure_path = tmp_path / "failure.json"
    write_failure_artifact(
        failure_path,
        run_id="temporal_logic-error_aux-seed42",
        epoch=4,
        batch=9,
        tensor_name="diagnostics.dt",
        error=caught.value,
        last_healthy_checkpoint="last.pt",
        diagnostics={"dt_min": 0.001, "dt_max": float("nan")},
    )
    artifact = json.loads(failure_path.read_text(encoding="utf-8"))
    assert artifact["run_id"] == "temporal_logic-error_aux-seed42"
    assert artifact["epoch"] == 4
    assert artifact["batch"] == 9
    assert artifact["tensor_name"] == "diagnostics.dt"
    assert artifact["last_healthy_checkpoint"] == "last.pt"
    assert artifact["error_type"] == "NumericalFailure"
    assert artifact["diagnostics"]["dt_max"] == "NaN"
    assert not (tmp_path / "failure.json.tmp").exists()


def test_model_guard_checks_parameters_and_gradients():
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    assert_finite_model(model, check_gradients=True)
    model.weight.grad[0, 0] = float("inf")
    with pytest.raises(NumericalFailure, match="gradient"):
        assert_finite_model(model, check_gradients=True)
