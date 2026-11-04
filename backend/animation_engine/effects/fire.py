"""Fuego y luz: parpadeo, lengüetas que suben y un brillo que sigue a cada zona."""
from __future__ import annotations

import cv2
import numpy as np

from ..assets import Assets
from ..constants import FUEGO_LUZ
from ..masks import Region, add_tinted, bounding_roi, inner_feather, region_plate, scale_pixels
from ..noise import Wave, WaveBank
from ..timing import FrameTime, cycles, omega

FEATHER_PX = 4.0
MAX_ZONES = 8
GLOW_SIGMA_MIN_PX = 14.0
GLOW_SIGMA_MAX_PX = 70.0
GLOW_FACTOR = 4  # el brillo se calcula a 1/4 de resolución
GLOW_CUTOFF = 0.006


def glow_reach_px(scale: float) -> float:
    """Distancia máxima a la que llega el brillo fuera de la zona de fuego."""
    return 4.0 * GLOW_SIGMA_MAX_PX * max(scale, 0.25) + 2.0 * GLOW_FACTOR


def _build_bank(assets: Assets, roi) -> WaveBank:
    rng = assets.rng("fire.waves")
    sc = max(assets.scale, 0.25)
    speed, period = assets.mood.speed, assets.period
    waves = []
    for wavelength, hz in ((90.0, 1.3), (46.0, 2.1), (24.0, 3.2), (13.0, 4.3)):
        k = 2.0 * np.pi / (wavelength * sc)
        c = cycles(hz, speed, period)
        # ky < 0 con omega > 0: el patrón viaja hacia arriba.
        waves.append(
            Wave(
                kx=k * float(rng.uniform(-0.45, 0.45)),
                ky=-k,
                omega=omega(c, period),
                phase=float(rng.uniform(0, 2 * np.pi)),
            )
        )
    return WaveBank(assets.shape, waves, roi=roi, factor=2 if max(assets.shape) >= 640 else 1)


class FireEffect:
    def __init__(self, image: np.ndarray, mask: np.ndarray, assets: Assets) -> None:
        region = (mask == FUEGO_LUZ).astype(np.uint8)
        self.active = bool(region.any())
        if not self.active:
            return
        sc = max(assets.scale, 0.25)
        alpha = inner_feather(region, FEATHER_PX * sc)
        self.region = Region(alpha)
        self.roi = self.region.roi
        if self.roi is not None:
            y0, y1, x0, x1 = self.roi
            self.alpha_float = self.region.alpha
            self.x = self.region.pick(assets.grid_x[y0:y1, x0:x1])
            self.y = self.region.pick(assets.grid_y[y0:y1, x0:x1])
            self.plate = region_plate(image, region, coarse=assets.shared)
            if assets.shared:
                self.bank = assets.cached("fire.bank", lambda: _build_bank(assets, None))
            else:
                self.bank = _build_bank(assets, self.roi)
        self.amp = 4.0 * sc
        self.amps_d = (0.8, 1.0, 0.6, 0.3)
        self.amps_gain = (0.35, 0.45, 0.45, 0.3)

        pixels = image[region > 0].astype(np.float32)
        luma = pixels.mean(axis=1)
        hot = pixels[luma >= np.percentile(luma, 60)]
        zone_color = hot.mean(axis=0) if len(hot) else np.array([60.0, 140.0, 255.0])
        # Mezcla con naranja para que el brillo se vea cálido incluso en una lámpara blanca.
        self.glow_color = np.clip(0.5 * zone_color + 0.5 * np.array([70.0, 150.0, 255.0]), 0, 255)
        self._build_glow(region, assets, sc)

    def _build_glow(self, region: np.ndarray, assets: Assets, sc: float) -> None:
        h, w = region.shape
        lw, lh = max(2, w // GLOW_FACTOR), max(2, h // GLOW_FACTOR)
        small = cv2.resize(region * 255, (lw, lh), interpolation=cv2.INTER_AREA)
        count, labels, stats, _ = cv2.connectedComponentsWithStats((small > 0).astype(np.uint8), connectivity=8)
        order = sorted(range(1, count), key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i))[:MAX_ZONES]
        rng = assets.rng("fire.flicker")
        speed, period = assets.mood.speed, assets.period
        self.glow_small: list[np.ndarray] = []
        self.flickers: list[list[tuple[float, float, float]]] = []
        reach = np.zeros((lh, lw), np.float32)
        for comp in order:
            area_full = float(stats[comp, cv2.CC_STAT_AREA]) * GLOW_FACTOR * GLOW_FACTOR
            sigma = float(np.clip(np.sqrt(area_full) * 0.55, GLOW_SIGMA_MIN_PX * sc, GLOW_SIGMA_MAX_PX * sc))
            zone = small.astype(np.float32) / 255.0 * (labels == comp)
            glow = cv2.GaussianBlur(zone, (0, 0), sigma / GLOW_FACTOR)
            peak = float(glow.max())
            if peak > 0:
                glow /= peak
            # Se cierra a cero sin escalón para que no quede un borde visible.
            glow = np.maximum(glow - GLOW_CUTOFF, 0.0) / (1.0 - GLOW_CUTOFF)
            self.glow_small.append(glow)
            reach = np.maximum(reach, glow)
            terms = []
            for base_hz, amp in ((2.3, 0.45), (4.1, 0.30), (7.3, 0.25)):
                c = cycles(base_hz + float(rng.uniform(-0.3, 0.3)), speed, period)
                terms.append((omega(c, period), amp, float(rng.uniform(0, 2 * np.pi))))
            self.flickers.append(terms)
        self.glow_size = (w, h)
        self.glow_roi = bounding_roi(cv2.resize(reach, (w, h), interpolation=cv2.INTER_LINEAR))

    @staticmethod
    def _flicker(terms, t: float) -> float:
        return float(sum(a * np.sin(w * t + p) for w, a, p in terms))

    def apply(self, canvas: np.ndarray, ft: FrameTime) -> None:
        if not self.active:
            return
        if self.roi is not None:
            up = self.region.from_low
            dy = up(self.bank.field(ft.t, self.amps_d, self.roi)) * self.amp * self.alpha_float
            dx = up(self.bank.field(ft.t, self.amps_d[::-1], self.roi, quadrature=True)) * self.amp * 0.6 * self.alpha_float
            layer = cv2.remap(
                self.plate, self.x + dx, self.y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            master = self._flicker(self.flickers[0], ft.t) if self.flickers else 0.0
            tongues = up(self.bank.field(ft.t, self.amps_gain, self.roi))
            layer = scale_pixels(layer, np.maximum(1.08 + 0.20 * master + 0.16 * tongues, 0.5))
            self.region.composite(canvas, layer)

        if self.glow_roi is None:
            return
        total = None
        for glow, terms in zip(self.glow_small, self.flickers):
            weight = 0.62 + 0.30 * self._flicker(terms, ft.t)
            total = glow * weight if total is None else total + glow * weight
        up = cv2.resize(total, self.glow_size, interpolation=cv2.INTER_LINEAR)
        gy0, gy1, gx0, gx1 = self.glow_roi
        up = up[gy0:gy1, gx0:gx1]
        canvas[gy0:gy1, gx0:gx1] = add_tinted(canvas[gy0:gy1, gx0:gx1], up, self.glow_color * 0.42)
