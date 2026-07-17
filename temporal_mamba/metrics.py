"""Dependency-free classification metrics with explicit zero-division rules."""

from __future__ import annotations

import numpy as np


BINARY_DATASETS = frozenset({"temporal_logic", "temporal_logic_v2"})


def _integer_vectors(target, predicted) -> tuple[np.ndarray, np.ndarray]:
    target_array = np.asarray(target)
    predicted_array = np.asarray(predicted)
    if target_array.shape != predicted_array.shape:
        raise ValueError("target and predicted must have the same shape")
    if target_array.size == 0:
        raise ValueError("metrics require at least one sample")
    return target_array.astype(np.int64).reshape(-1), predicted_array.astype(np.int64).reshape(-1)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _confusion_matrix(target: np.ndarray, predicted: np.ndarray, num_classes: int) -> np.ndarray:
    if np.any((target < 0) | (target >= num_classes)):
        raise ValueError("target contains an out-of-range class")
    if np.any((predicted < 0) | (predicted >= num_classes)):
        raise ValueError("predicted contains an out-of-range class")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (target, predicted), 1)
    return confusion


def binary_metrics(target, predicted) -> dict[str, object]:
    target_array, predicted_array = _integer_vectors(target, predicted)
    confusion = _confusion_matrix(target_array, predicted_array, num_classes=2)
    recalls = [
        _safe_ratio(int(confusion[index, index]), int(confusion[index].sum()))
        for index in range(2)
    ]
    true_positive = int(confusion[1, 1])
    false_positive = int(confusion[0, 1])
    false_negative = int(confusion[1, 0])
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    return {
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "balanced_accuracy": float(sum(recalls) / 2),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class_recall": recalls,
        "confusion_matrix": confusion.tolist(),
    }


def multiclass_metrics(target, predicted, *, num_classes: int) -> dict[str, object]:
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    target_array, predicted_array = _integer_vectors(target, predicted)
    confusion = _confusion_matrix(target_array, predicted_array, num_classes=num_classes)
    recalls: list[float] = []
    per_class_f1: list[float] = []
    for class_index in range(num_classes):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum() - true_positive)
        false_negative = int(confusion[class_index].sum() - true_positive)
        recalls.append(_safe_ratio(true_positive, true_positive + false_negative))
        per_class_f1.append(
            _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
        )
    return {
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_f1": float(sum(per_class_f1) / num_classes),
        "per_class_f1": per_class_f1,
        "per_class_recall": recalls,
        "confusion_matrix": confusion.tolist(),
    }
