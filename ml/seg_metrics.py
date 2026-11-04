"""Metricas de segmentacion a partir de una matriz de confusion. Sin dependencias de PyTorch."""
from __future__ import annotations

import numpy as np

from dataset import CLASSES, IGNORE_INDEX, NUM_CLASSES


def confusion_update(matrix: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> None:
    """Suma a `matrix` (filas = verdad, columnas = prediccion), omitiendo IGNORE_INDEX."""
    valid = truth != IGNORE_INDEX
    t = truth[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    if t.size and (t.max() >= NUM_CLASSES or t.min() < 0):
        raise ValueError(f"Indice de clase invalido en la verdad: {np.unique(t).tolist()}")
    matrix += np.bincount(NUM_CLASSES * t + p, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    """mIoU y Dice se promedian sobre las clases con soporte o prediccion en estos datos."""
    matrix = np.asarray(matrix, dtype=np.int64)
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.zeros(NUM_CLASSES), where=union > 0)
    denom = support + predicted
    dice = np.divide(2 * tp, denom, out=np.zeros(NUM_CLASSES), where=denom > 0)
    present = union > 0
    return {
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "dice_macro": float(dice[present].mean()) if present.any() else 0.0,
        "pixel_accuracy": float(tp.sum() / matrix.sum()) if matrix.sum() else 0.0,
        "per_class": {
            name: {"iou": float(iou[i]), "dice": float(dice[i]), "support_pixels": int(support[i])}
            for i, name in enumerate(CLASSES)
        },
        "classes_without_data": [CLASSES[i] for i in np.where(~present)[0]],
    }
