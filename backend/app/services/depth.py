"""Profundidad relativa con MiDaS v2.1 small (CONTRATOS, sección 2.2)."""
from __future__ import annotations

import time

import cv2
import numpy as np

from ..observability import INFERENCE_SECONDS

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess_depth(image_bgr: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Igual que el script ONNX oficial de MiDaS: RGB, cúbico, ImageNet. `size` es (alto, ancho)."""
    height, width = size
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
    chw = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0
    return ((chw - _MEAN) / _STD)[np.newaxis, ...].astype(np.float32)


def normalize_depth(raw: np.ndarray) -> np.ndarray:
    """Min-max a [0, 1]. MiDaS da profundidad inversa, así que 1.0 ya es lo más cercano."""
    raw = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(raw.min()), float(raw.max())
    if high - low < 1e-8:
        return np.zeros_like(raw, dtype=np.float32)
    return (raw - low) / (high - low)


class DepthModel:
    def __init__(self, session) -> None:
        self._session = session
        node = session.get_inputs()[0]
        self._input_name = node.name
        shape = node.shape
        if len(shape) != 4 or not isinstance(shape[2], int) or not isinstance(shape[3], int):
            raise ValueError(f"Entrada de profundidad con forma inesperada: {shape}")
        self._input_size = (shape[2], shape[3])

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        """Profundidad float32 (H, W) en [0, 1], sin registrar métricas."""
        tensor = preprocess_depth(image_bgr, self._input_size)
        raw = np.squeeze(self._session.run(None, {self._input_name: tensor})[0])
        if raw.ndim != 2:
            raise ValueError(f"Salida de profundidad con forma inesperada: {raw.shape}")
        height, width = image_bgr.shape[:2]
        resized = cv2.resize(raw.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        return normalize_depth(resized)

    def predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Devuelve (depth float32 (H, W) en [0, 1], milisegundos medidos)."""
        started = time.perf_counter()
        depth = self.infer(image_bgr)
        elapsed = time.perf_counter() - started
        INFERENCE_SECONDS.labels(model="depth").observe(elapsed)
        return depth, elapsed * 1000.0
