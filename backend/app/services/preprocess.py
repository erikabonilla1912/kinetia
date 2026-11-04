"""Preprocesamiento del modelo de segmentación: única implementación (CONTRATOS, sección 2.1).

ml/ debe producir exactamente lo mismo; por eso este módulo solo depende de numpy y OpenCV
y se puede importar sin el resto de la API.
"""
from __future__ import annotations

import cv2
import numpy as np

SEG_INPUT_SIZE = 320
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess_seg(image_bgr: np.ndarray) -> np.ndarray:
    """Imagen BGR uint8 (H, W, 3) -> tensor float32 NCHW de forma [1, 3, 320, 320], en RGB."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("preprocess_seg espera un arreglo uint8 de forma (H, W, 3)")
    resized = cv2.resize(image_bgr, (SEG_INPUT_SIZE, SEG_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    normalized = (chw - IMAGENET_MEAN) / IMAGENET_STD
    return normalized[np.newaxis, ...].astype(np.float32)


def work_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Tamaño de trabajo (ancho, alto): lado mayor <= max_side, sin ampliar nunca."""
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_to_work(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    new_width, new_height = work_size(width, height, max_side)
    if (new_width, new_height) == (width, height):
        return image_bgr
    return cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
