"""Ruido espectral y bancos de ondas precalculadas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .masks import Roi


def spectral_noise(rng: np.random.Generator, h: int, w: int, beta: float) -> np.ndarray:
    """Ruido 1/f^beta float32 normalizado de 0 a 1."""
    white = rng.standard_normal((h, w))
    spectrum = np.fft.rfft2(white)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0
    spectrum *= radius ** (-beta / 2.0)
    spectrum[0, 0] = 0.0
    field = np.fft.irfft2(spectrum, s=(h, w))
    lo, hi = np.percentile(field, (1.0, 99.0))
    return np.clip((field - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class Wave:
    kx: float  # rad/px
    ky: float  # rad/px (o por unidad de ``psi`` si el banco lo usa)
    omega: float  # rad/s
    phase: float


class WaveBank:
    """Ondas sin(a(x, y) - w t) con la parte espacial ya calculada.

    Por identidad de suma de ángulos, cada frame solo combina dos mapas fijos,
    lo que evita calcular senos por píxel y hace la onda periódica exacta.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        waves: Sequence[Wave],
        roi: Optional[Roi] = None,
        psi: Optional[np.ndarray] = None,
        factor: int = 1,
        warp: Optional[np.ndarray] = None,
    ) -> None:
        h, w = shape
        self.roi: Roi = roi if roi is not None else (0, h, 0, w)
        self.factor = factor
        y0, y1, x0, x1 = self.roi
        # Se muestrea cada ``factor`` píxeles: los campos son suaves y se reescalan al final.
        xs = np.arange(x0, x1 + factor, factor, dtype=np.float64)
        rows = np.arange(y0, y1 + factor, factor)
        ys = (psi[np.minimum(rows, h - 1)] if psi is not None else rows.astype(np.float64))
        self.omegas = [wv.omega for wv in waves]
        self.sin_maps: list[np.ndarray] = []
        self.cos_maps: list[np.ndarray] = []
        for wv in waves:
            ax = wv.kx * xs + wv.phase
            ay = wv.ky * ys
            if warp is None:
                sx, cx = np.sin(ax), np.cos(ax)
                sy, cy = np.sin(ay), np.cos(ay)
                self.sin_maps.append((np.outer(cy, sx) + np.outer(sy, cx)).astype(np.float32))
                self.cos_maps.append((np.outer(cy, cx) - np.outer(sy, sx)).astype(np.float32))
            else:
                # El warp deforma los frentes de onda: sin él, la suma de senos forma una retícula.
                angle = ay[:, None] + ax[None, :] + warp[y0 // factor : y0 // factor + len(ys), x0 // factor : x0 // factor + len(xs)]
                self.sin_maps.append(np.sin(angle).astype(np.float32))
                self.cos_maps.append(np.cos(angle).astype(np.float32))

    def field(
        self,
        t: float,
        amps: Sequence[float],
        roi: Optional[Roi] = None,
        quadrature: bool = False,
    ) -> np.ndarray:
        """Suma de A_j sin(a_j - w_j t), o de A_j cos(a_j - w_j t) con ``quadrature``.

        El resultado está a resolución 1/factor; ``Region.from_low`` lo lleva al efecto.
        """
        y0, y1, x0, x1 = roi if roi is not None else self.roi
        oy, _, ox, _ = self.roi
        f = self.factor
        ly0, lx0 = (y0 - oy) // f, (x0 - ox) // f
        ly1 = ly0 + (y1 - y0 + f - 1) // f
        lx1 = lx0 + (x1 - x0 + f - 1) // f
        total = None
        for j, amp in enumerate(amps):
            if amp == 0.0:
                continue
            c, s = float(np.cos(self.omegas[j] * t)), float(np.sin(self.omegas[j] * t))
            sin_map = self.sin_maps[j][ly0:ly1, lx0:lx1]
            cos_map = self.cos_maps[j][ly0:ly1, lx0:lx1]
            if quadrature:  # cos(a - wt) = cos a cos wt + sin a sin wt
                wave = cv2.addWeighted(cos_map, amp * c, sin_map, amp * s, 0.0)
            else:  # sin(a - wt) = sin a cos wt - cos a sin wt
                wave = cv2.addWeighted(sin_map, amp * c, cos_map, -amp * s, 0.0)
            total = wave if total is None else cv2.add(total, wave)
        return total
