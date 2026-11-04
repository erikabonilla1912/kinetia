"""Vegetación: vaivén leve con ráfagas y aleteo. Las hojas que caen están en particles.py."""
from __future__ import annotations

import cv2
import numpy as np

from ..assets import Assets
from ..constants import VEGETACION, forest_min_pct
from ..masks import Region, inner_feather, region_plate, scale_pixels
from ..noise import Wave, WaveBank
from ..timing import FrameTime, cycles, omega

FEATHER_PX = 6.0
SWAY_AMPLITUDE_PX = 0.7


def is_forest(mask: np.ndarray) -> bool:
    """Regla del contrato: la vegetación cubre al menos FOREST_MIN_PCT de la imagen."""
    return 100.0 * float(np.mean(mask == VEGETACION)) >= forest_min_pct()


def _build_bank(assets: Assets, roi) -> WaveBank:
    rng = assets.rng("vegetation.waves")
    sc = max(assets.scale, 0.25)
    speed, period = assets.mood.speed, assets.period
    specs = [(420.0, 0.30), (210.0, 0.45), (96.0, 0.75), (44.0, 1.2), (22.0, 1.8)]
    waves = []
    for wavelength, hz in specs:
        k = 2.0 * np.pi / (wavelength * sc)
        angle = float(rng.uniform(-0.9, 0.9))
        c = cycles(hz, speed, period)
        waves.append(
            Wave(
                kx=k * float(np.cos(angle)),
                ky=k * float(np.sin(angle)),
                omega=omega(c, period),
                phase=float(rng.uniform(0, 2 * np.pi)),
            )
        )
    factor = 2 if max(assets.shape) >= 640 else 1
    warp = assets.phase_warp("vegetation.warp", factor, 160.0, 1.1)
    return WaveBank(assets.shape, waves, roi=roi, factor=factor, warp=warp)


class VegetationEffect:
    def __init__(self, image: np.ndarray, mask: np.ndarray, assets: Assets) -> None:
        region = (mask == VEGETACION).astype(np.uint8)
        sc = max(assets.scale, 0.25)
        alpha = inner_feather(region, FEATHER_PX * sc)
        self.region = Region(alpha)
        self.active = self.region.active
        if not self.active:
            return
        self.roi = self.region.roi
        y0, y1, x0, x1 = self.roi
        self.plate = region_plate(image, region, coarse=assets.shared)
        y_local = assets.grid_y[y0:y1, x0:x1]
        self.x = self.region.pick(assets.grid_x[y0:y1, x0:x1])
        self.y = self.region.pick(y_local)
        if assets.shared:
            self.bank = assets.cached("vegetation.bank", lambda: _build_bank(assets, None))
        else:
            self.bank = _build_bank(assets, self.roi)
        # Lo alto de la foto (copas) oscila más que lo bajo (pasto, troncos).
        h = assets.shape[0]
        self.amp = self.region.pick((0.55 + 0.75 * (1.0 - y_local / max(h - 1, 1))) * SWAY_AMPLITUDE_PX * sc)
        self.amps_x = (1.0, 0.6, 0.0, 0.0, 0.0)
        self.amps_y = (0.2, 0.15, 0.0, 0.0, 0.0)
        self.amps_shade = (0.0, 0.2, 0.3, 0.35, 0.3)

    def apply(self, canvas: np.ndarray, ft: FrameTime) -> None:
        if not self.active:
            return
        roi, up = self.roi, self.region.from_low
        dx = up(self.bank.field(ft.t, self.amps_x, roi)) * self.amp
        dy = up(self.bank.field(ft.t, self.amps_y, roi, quadrature=True)) * self.amp
        layer = cv2.remap(
            self.plate, self.x + dx, self.y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        # Las hojas cambian de orientación frente a la luz: brillo que va y viene.
        shimmer = up(self.bank.field(ft.t, self.amps_shade, roi, quadrature=True))
        self.region.composite(canvas, scale_pixels(layer, 1.0 + 0.03 * shimmer))
