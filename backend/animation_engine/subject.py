"""Sujeto (persona o animal): totalmente quieto, como capa por encima de las partículas.

No hay parallax: desplazar el sujeto o el fondo hacía ver las patas como de goma y
hacía resbalar los pies. La profundidad la dan las partículas que pasan por detrás
del sujeto y los efectos de cada región.
"""
from __future__ import annotations

import cv2
import numpy as np

from .constants import ANIMAL, PERSONA
from .masks import bounding_roi, clip_alpha, inner_feather


def subject_mask(mask: np.ndarray) -> np.ndarray:
    return ((mask == PERSONA) | (mask == ANIMAL)).astype(np.uint8)


class SubjectLayer:
    def __init__(self, mask: np.ndarray, scale: float) -> None:
        subject = subject_mask(mask)
        self.roi = bounding_roi(subject)
        self.active = self.roi is not None
        if not self.active:
            return
        y0, y1, x0, x1 = self.roi
        alpha = inner_feather(subject[y0:y1, x0:x1], max(1.5, 2.5 * scale))
        self.alpha = clip_alpha(alpha)
        self.inverse = 1.0 - self.alpha

    def composite(self, canvas: np.ndarray, source: np.ndarray) -> None:
        """Pone encima los píxeles del sujeto tal como están en ``source``."""
        if not self.active:
            return
        y0, y1, x0, x1 = self.roi
        canvas[y0:y1, x0:x1] = cv2.blendLinear(canvas[y0:y1, x0:x1], source[y0:y1, x0:x1], self.inverse, self.alpha)
