"""Índices de clase y reglas derivadas. Ver docs/CONTRATOS.md, secciones 1 y 6."""
from __future__ import annotations

import os

FONDO = 0
CIELO = 1
AGUA = 2
VEGETACION = 3
FUEGO_LUZ = 4
PERSONA = 5
ANIMAL = 6
NUM_CLASSES = 7

DEFAULT_NIGHT_LUMA_THRESHOLD = 60.0
DEFAULT_FOREST_MIN_PCT = 25.0

REFERENCE_LONG_SIDE = 1280.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} debe ser numérico; se recibió {raw!r}") from exc


def night_luma_threshold() -> float:
    """Umbral de la regla de noche (0 a 255). Se lee en cada llamada."""
    return _env_float("NIGHT_LUMA_THRESHOLD", DEFAULT_NIGHT_LUMA_THRESHOLD)


def forest_min_pct() -> float:
    """Cobertura mínima de vegetación, en %, para la regla de bosque."""
    return _env_float("FOREST_MIN_PCT", DEFAULT_FOREST_MIN_PCT)
