"""Pincel: flujo dirigido dentro de lo pintado.

Cada píxel avanza en la dirección del trazo. Para que el movimiento no se acumule
se usan dos copias desfasadas medio ciclo, con pesos triangulares, lo que también
hace el efecto periódico exacto.
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from ..assets import Assets
from ..masks import blend, bounding_roi, inner_feather, clip_alpha
from ..timing import FrameTime, cycles
from ..types import BrushStroke

BASE_FLOW_PX_S = 70.0
FEATHER_RADIUS_FRACTION = 0.45


def rasterize_strokes(
    strokes: Sequence[BrushStroke], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (pintado uint8, vx, vy), con v = dirección * fuerza por píxel."""
    h, w = shape
    painted = np.zeros((h, w), np.uint8)
    vx = np.zeros((h, w), np.float32)
    vy = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    long_side = max(h, w)
    for stroke in strokes:
        if not stroke.points:
            continue
        pts = np.array([[round(p[0] * (w - 1)), round(p[1] * (h - 1))] for p in stroke.points], dtype=np.int32)
        radius = max(1, int(round(stroke.radius * long_side)))
        pad = radius + 2
        x0, y0 = max(0, int(pts[:, 0].min()) - pad), max(0, int(pts[:, 1].min()) - pad)
        x1, y1 = min(w, int(pts[:, 0].max()) + pad + 1), min(h, int(pts[:, 1].max()) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        local = np.zeros((y1 - y0, x1 - x0), np.uint8)
        shifted = pts - np.array([x0, y0])
        if len(shifted) > 1:
            cv2.polylines(local, [shifted.reshape(-1, 1, 2)], False, 1, thickness=2 * radius, lineType=cv2.LINE_8)
        for p in shifted:
            cv2.circle(local, (int(p[0]), int(p[1])), radius, 1, thickness=-1, lineType=cv2.LINE_8)
        on = local > 0
        angle = math.radians(stroke.direction_deg)
        strength = float(np.clip(stroke.strength, 0.0, 1.0))
        painted[y0:y1, x0:x1][on] = 1
        vx[y0:y1, x0:x1][on] += math.cos(angle) * strength
        vy[y0:y1, x0:x1][on] += math.sin(angle) * strength
        weight[y0:y1, x0:x1][on] += 1.0
    nonzero = weight > 0
    vx[nonzero] /= weight[nonzero]
    vy[nonzero] /= weight[nonzero]
    return painted, vx, vy


class BrushEffect:
    def __init__(self, strokes: Sequence[BrushStroke], assets: Assets) -> None:
        self.active = False
        self.paint_alpha = None
        painted, vx, vy = rasterize_strokes(strokes, assets.shape)
        sc = max(assets.scale, 0.25)
        long_side = max(assets.shape)
        radii = [s.radius for s in strokes if s.points]
        feather = max(2.0, FEATHER_RADIUS_FRACTION * (min(radii) if radii else 0.0) * long_side)
        alpha = inner_feather(painted, feather)
        self.roi = bounding_roi(alpha)
        if self.roi is None:
            return
        self.active = True
        self.paint_alpha = clip_alpha(alpha)
        y0, y1, x0, x1 = self.roi
        self.alpha_map = self.paint_alpha[y0:y1, x0:x1]
        speed, period = assets.mood.speed, assets.period
        self.cycle_period = period / cycles(0.45, speed, period)
        # Distancia por ciclo: la que recorre un trazo de fuerza 1 a la velocidad del mood.
        travel = BASE_FLOW_PX_S * speed * sc * self.cycle_period
        ksize = max(3, int(9 * sc)) | 1
        self.ux = cv2.GaussianBlur(vx, (ksize, ksize), 0)[y0:y1, x0:x1] * travel
        self.uy = cv2.GaussianBlur(vy, (ksize, ksize), 0)[y0:y1, x0:x1] * travel
        self.x = assets.grid_x[y0:y1, x0:x1]
        self.y = assets.grid_y[y0:y1, x0:x1]

    def _phase_layer(self, source: np.ndarray, phase: float) -> np.ndarray:
        offset = phase - 0.5
        return cv2.remap(
            source, self.x - self.ux * offset, self.y - self.uy * offset, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def apply(self, canvas: np.ndarray, source: np.ndarray, ft: FrameTime) -> None:
        if not self.active:
            return
        phase = (ft.t / self.cycle_period) % 1.0
        w = abs(2.0 * phase - 1.0)
        a = self._phase_layer(source, phase)
        b = self._phase_layer(source, (phase + 0.5) % 1.0)
        # Con phase=0 pesa solo la capa b, que está centrada en la imagen original.
        layer = cv2.addWeighted(a, 1.0 - w, b, w, 0.0)
        y0, y1, x0, x1 = self.roi
        canvas[y0:y1, x0:x1] = blend(canvas[y0:y1, x0:x1], layer, self.alpha_map)
