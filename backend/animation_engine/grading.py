"""Color del mood: saturación HSV, calidez, contraste y viñeta."""
from __future__ import annotations

import cv2
import numpy as np

from .types import MoodParams


class Grader:
    def __init__(self, mood: MoodParams, shape: tuple[int, int]) -> None:
        self.identity_color = mood.warmth == 0 and mood.contrast == 1
        self.identity_sat = mood.saturation == 1
        self.lut = self._build_lut(mood)
        self.saturation = float(mood.saturation)
        self.vignette = self._build_vignette(mood.vignette, shape) if mood.vignette > 0 else None

    @staticmethod
    def _build_lut(mood: MoodParams) -> np.ndarray:
        x = np.arange(256, dtype=np.float64)
        contrasted = 128.0 + (x - 128.0) * mood.contrast
        # Orden BGR: lo cálido sube rojo y baja azul.
        gains = (1.0 - 0.30 * mood.warmth, 1.0 + 0.04 * mood.warmth, 1.0 + 0.26 * mood.warmth)
        channels = [np.clip(contrasted * g + 0.5, 0, 255).astype(np.uint8) for g in gains]
        return np.stack(channels, axis=-1).reshape(256, 1, 3)

    @staticmethod
    def _build_vignette(strength: float, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        y = (np.arange(h, dtype=np.float32) - (h - 1) / 2.0) / max(h / 2.0, 1.0)
        x = (np.arange(w, dtype=np.float32) - (w - 1) / 2.0) / max(w / 2.0, 1.0)
        r2 = (x[None, :] ** 2 + y[:, None] ** 2) / 2.0  # 0 en el centro, 1 en la esquina
        falloff = np.clip((r2 - 0.12) / 0.88, 0.0, 1.0)
        falloff = falloff * falloff * (3.0 - 2.0 * falloff)
        gain = 1.0 - float(np.clip(strength, 0.0, 1.0)) * falloff
        fixed = (gain * 255.0 + 0.5).astype(np.uint8)
        return cv2.merge([fixed, fixed, fixed])

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Devuelve el frame con el color aplicado; puede reutilizar la entrada."""
        out = frame
        if not self.identity_sat:
            # Multiplicar S por k con H y V fijos equivale a c' = V + k (c - V), con V = max(B, G, R).
            k = self.saturation
            b, g, r = cv2.split(out)
            value = cv2.max(cv2.max(b, g), r)
            out = cv2.merge([cv2.addWeighted(c, k, value, 1.0 - k, 0.0) for c in (b, g, r)])
        if not self.identity_color:
            out = cv2.LUT(out, self.lut)
        if self.vignette is not None:
            out = cv2.multiply(out, self.vignette, scale=1.0 / 255.0)
        return out
