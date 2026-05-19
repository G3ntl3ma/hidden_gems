from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    confusion_matrix: list[list[int]]
    threshold: float


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float


def _as_numpy(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError("Expected 1-D array.")
    return arr


def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")


def labels_from_threshold(y_score: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    score = _as_numpy(y_score).astype(float)
    return (score >= threshold).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationReport:
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    _validate_shapes(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist()
    return ClassificationReport(
        accuracy=float((y_true == y_pred).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=None,
        pr_auc=None,
        confusion_matrix=matrix,
        threshold=0.5,
    )


def binary_classification_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationReport:
    y_true = _as_numpy(y_true).astype(int)
    y_score = _as_numpy(y_score).astype(float)
    _validate_shapes(y_true, y_score)
    y_pred = labels_from_threshold(y_score, threshold=threshold)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist()

    roc_auc: float | None = None
    pr_auc: float | None = None
    if np.unique(y_true).size > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

    return ClassificationReport(
        accuracy=float((y_true == y_pred).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=matrix,
        threshold=float(threshold),
    )


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[ThresholdPoint]:
    y_true = _as_numpy(y_true).astype(int)
    y_score = _as_numpy(y_score).astype(float)
    _validate_shapes(y_true, y_score)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    points: list[ThresholdPoint] = []
    for thr in thresholds:
        y_pred = labels_from_threshold(y_score, threshold=float(thr))
        points.append(
            ThresholdPoint(
                threshold=float(thr),
                precision=float(precision_score(y_true, y_pred, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, zero_division=0)),
                f1=float(f1_score(y_true, y_pred, zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
            )
        )
    return points


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    optimize_for: Literal["f1", "recall"] = "f1",
    min_precision: float = 0.0,
) -> ThresholdPoint:
    points = threshold_sweep(y_true, y_score)
    eligible = [p for p in points if p.precision >= min_precision]
    if not eligible:
        eligible = points

    if optimize_for == "recall":
        key_fn = lambda p: (p.recall, p.f1, p.precision, -abs(p.threshold - 0.5))
    else:
        key_fn = lambda p: (p.f1, p.recall, p.precision, -abs(p.threshold - 0.5))
    return max(eligible, key=key_fn)

