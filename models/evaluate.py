from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationReport:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    return ClassificationReport(accuracy=float((y_true == y_pred).mean()))

