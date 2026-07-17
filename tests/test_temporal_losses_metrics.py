import numpy as np
import pytest
import torch

from temporal_mamba.losses import (
    auxiliary_weight,
    compute_task_loss,
    compute_total_loss,
    pointwise_prediction_loss,
)
from temporal_mamba.metrics import binary_metrics, multiclass_metrics
from temporal_mamba.model import TemporalModelOutput


def test_pointwise_loss_does_not_cancel_opposite_errors():
    position = torch.tensor([[[0.0], [10.0], [-10.0]]])
    velocity = torch.zeros_like(position)
    loss = pointwise_prediction_loss(position, velocity, velocity_weight=0.5)
    assert loss > 8.0


def test_pointwise_loss_excludes_unaligned_index_zero():
    position = torch.zeros(2, 4, 3)
    velocity = torch.zeros_like(position)
    position[:, 0] = 1e6
    velocity[:, 0] = -1e6
    assert pointwise_prediction_loss(position, velocity) == 0


def test_auxiliary_weight_warms_from_zero():
    assert auxiliary_weight(0, total_epochs=30, target=0.1, warmup_fraction=0.1) == 0.0
    assert auxiliary_weight(1, total_epochs=30, target=0.1, warmup_fraction=0.1) == pytest.approx(1 / 30)
    assert auxiliary_weight(3, total_epochs=30, target=0.1, warmup_fraction=0.1) == pytest.approx(0.1)


def test_task_losses_match_pytorch_definitions():
    binary_logits = torch.tensor([[0.5], [-1.0]])
    binary_target = torch.tensor([1.0, 0.0])
    actual_binary = compute_task_loss(binary_logits, binary_target, dataset="temporal_logic")
    expected_binary = torch.nn.functional.binary_cross_entropy_with_logits(
        binary_logits[:, 0], binary_target
    )
    torch.testing.assert_close(actual_binary, expected_binary)

    class_logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]])
    class_target = torch.tensor([0, 2])
    actual_class = compute_task_loss(class_logits, class_target, dataset="uci_har")
    torch.testing.assert_close(actual_class, torch.nn.functional.cross_entropy(class_logits, class_target))


def test_v2_task_loss_uses_binary_definition():
    logits = torch.tensor([[0.5], [-1.0]])
    target = torch.tensor([1.0, 0.0])
    actual = compute_task_loss(logits, target, dataset="temporal_logic_v2")
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], target)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("variant", ["vanilla", "two_pass", "error_inject"])
def test_non_aux_variants_force_zero_auxiliary_weight(variant):
    output = TemporalModelOutput(
        logits=torch.zeros(2, 1),
        position_error=torch.ones(2, 4, 1),
        velocity_error=torch.ones(2, 4, 1),
        pass_count=1,
        uses_error=True,
        diagnostics={},
    )
    breakdown = compute_total_loss(
        output,
        torch.tensor([0.0, 1.0]),
        dataset="temporal_logic",
        variant=variant,
        epoch=10,
        total_epochs=30,
        lambda_aux=0.1,
        aux_warmup_fraction=0.1,
    )
    assert breakdown.aux_weight == 0.0
    torch.testing.assert_close(breakdown.total, breakdown.task)


def test_binary_metrics_hand_checked_and_missing_class_safe():
    metrics = binary_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    missing = binary_metrics(np.array([0, 0]), np.array([0, 0]))
    assert missing["per_class_recall"] == [1.0, 0.0]
    assert missing["balanced_accuracy"] == pytest.approx(0.5)
    assert missing["f1"] == 0.0


def test_multiclass_metrics_hand_checked_and_missing_class_safe():
    target = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 1, 1, 1, 2, 0])
    metrics = multiclass_metrics(target, predicted, num_classes=4)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["per_class_recall"] == pytest.approx([0.5, 1.0, 0.5, 0.0])
    assert metrics["confusion_matrix"] == [
        [1, 1, 0, 0],
        [0, 2, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0],
    ]
    assert np.isfinite(metrics["macro_f1"])
