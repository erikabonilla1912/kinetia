"""Lectura y rasterización de mask_strokes (CONTRATOS, sección 4)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np
from pydantic import ValidationError

from animation_engine.types import BrushStroke

from .. import errors
from ..schemas import MAX_STROKE_POINTS, MAX_STROKES, StrokesIn


def parse_strokes(raw: str | None) -> tuple[BrushStroke, ...]:
    """Convierte el campo JSON en trazos. Aplica rangos y límites; si fallan, INVALID_PARAMS."""
    if raw is None or not raw.strip():
        return ()
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise errors.invalid_params(
            "mask_strokes no es un JSON válido.", [{"field": "mask_strokes", "problem": "JSON inválido."}]
        ) from exc
    try:
        parsed = StrokesIn.model_validate(payload)
    except ValidationError as exc:
        raise errors.invalid_params(
            f"mask_strokes no cumple el formato. Máximo {MAX_STROKES} trazos y {MAX_STROKE_POINTS} puntos en total.",
            [{"field": "mask_strokes", "problem": "Formato, rango o límite incumplido."}],
        ) from exc
    return tuple(
        BrushStroke(
            points=tuple((float(x), float(y)) for x, y in stroke.points),
            radius=stroke.radius,
            direction_deg=stroke.direction_deg,
            strength=stroke.strength,
        )
        for stroke in parsed.strokes
    )


def strokes_to_json(strokes: tuple[BrushStroke, ...]) -> str:
    return json.dumps(
        [
            {
                "points": [list(point) for point in stroke.points],
                "radius": stroke.radius,
                "direction_deg": stroke.direction_deg,
                "strength": stroke.strength,
            }
            for stroke in strokes
        ]
    )


def strokes_from_json(text: str | None) -> tuple[BrushStroke, ...]:
    if not text:
        return ()
    return tuple(
        BrushStroke(
            points=tuple((float(x), float(y)) for x, y in item["points"]),
            radius=float(item["radius"]),
            direction_deg=float(item["direction_deg"]),
            strength=float(item["strength"]),
        )
        for item in json.loads(text)
    )


@dataclass(frozen=True)
class BrushRaster:
    coverage: np.ndarray  # uint8 (H, W), 255 donde hay pincel
    strength: np.ndarray  # float32 (H, W), 0 a 1
    direction_deg: np.ndarray  # float32 (H, W), del último trazo que pasó por el píxel


def rasterize_strokes(strokes: tuple[BrushStroke, ...], width: int, height: int) -> BrushRaster:
    """`radius` es fracción del lado mayor; origen arriba a la izquierda. Los trazos posteriores mandan."""
    coverage = np.zeros((height, width), dtype=np.uint8)
    strength = np.zeros((height, width), dtype=np.float32)
    direction = np.zeros((height, width), dtype=np.float32)
    longest = max(width, height)
    for stroke in strokes:
        radius_px = max(1, round(stroke.radius * longest))
        pixels = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in stroke.points]
        stamp = np.zeros((height, width), dtype=np.uint8)
        for point in pixels:
            cv2.circle(stamp, point, radius_px, 255, thickness=-1, lineType=cv2.LINE_8)
        for start, end in zip(pixels, pixels[1:]):
            cv2.line(stamp, start, end, 255, thickness=radius_px * 2, lineType=cv2.LINE_8)
        painted = stamp > 0
        coverage[painted] = 255
        strength[painted] = stroke.strength
        direction[painted] = stroke.direction_deg
    return BrushRaster(coverage=coverage, strength=strength, direction_deg=direction)
