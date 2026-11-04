"""Validación de entradas. Los mensajes son para quien integra el motor."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .constants import NUM_CLASSES
from .types import MoodParams, RenderConfig


def validate_image(image: np.ndarray, name: str = "image_bgr") -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} debe ser uint8 con forma (H, W, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} está vacía")


def validate_mask(mask: np.ndarray, shape: tuple[int, int]) -> None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask debe tener forma (H, W)")
    if mask.shape != shape:
        raise ValueError(f"mask tiene forma {mask.shape}; se esperaba {shape}")
    if mask.dtype != np.uint8:
        raise ValueError(f"mask debe ser uint8; se recibió {mask.dtype}")
    if int(mask.max()) >= NUM_CLASSES:
        raise ValueError(f"mask contiene el valor {int(mask.max())}; el máximo es {NUM_CLASSES - 1}")


def normalize_depth(depth: Optional[np.ndarray], shape: tuple[int, int]) -> Optional[np.ndarray]:
    if depth is None:
        return None
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        raise ValueError("depth debe tener forma (H, W)")
    if depth.shape != shape:
        raise ValueError(f"depth tiene forma {depth.shape}; se esperaba {shape}")
    if not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"depth debe ser float32; se recibió {depth.dtype}")
    if not np.isfinite(depth).all():
        raise ValueError("depth contiene valores no finitos")
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


def validate_config(config: RenderConfig) -> None:
    if not isinstance(config.mood, MoodParams):
        raise ValueError("config.mood debe ser MoodParams")
    if config.fps <= 0:
        raise ValueError("config.fps debe ser mayor que 0")
    if config.duration_s <= 0:
        raise ValueError("config.duration_s debe ser mayor que 0")
    if config.crossfade_s < 0:
        raise ValueError("config.crossfade_s no puede ser negativo")
    if config.brush_mode not in ("add", "only"):
        raise ValueError(f"brush_mode inválido: {config.brush_mode!r}")
    if config.mood.speed <= 0:
        raise ValueError("mood.speed debe ser mayor que 0")
