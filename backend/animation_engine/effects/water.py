"""Agua: ondas con perspectiva, reflejo especular de lo que hay encima y destellos."""
from __future__ import annotations

import cv2
import numpy as np

from ..assets import Assets
from ..constants import AGUA
from ..masks import Region, add_tinted, blend, clip_alpha, inner_feather, region_plate, scale_pixels
from ..noise import Wave, WaveBank
from ..timing import FrameTime, cycles, omega

FEATHER_PX = 8.0
AMPLITUDE_PX = 3.4


def _psi(h: int) -> np.ndarray:
    """Coordenada vertical con perspectiva: las ondas se aprietan hacia el horizonte."""
    rho = np.arange(h, dtype=np.float64) / max(h - 1, 1)
    return np.log(0.12 + rho) * h * 0.55


def _build_bank(assets: Assets, roi) -> WaveBank:
    rng = assets.rng("water.waves")
    h, w = assets.shape
    sc = max(assets.scale, 0.25)
    speed, period = assets.mood.speed, assets.period
    specs = [
        (150.0, 0.06, 0.45), (84.0, -0.05, 0.65), (46.0, 0.08, 0.95),
        (26.0, -0.10, 1.35), (15.0, 0.14, 1.75), (9.0, -0.16, 2.2),
    ]
    waves = []
    for wavelength, tilt, hz in specs:
        k = 2.0 * np.pi / (wavelength * sc)
        c = cycles(hz, speed, period)
        waves.append(Wave(kx=k * tilt, ky=k, omega=omega(c, period), phase=float(rng.uniform(0, 2 * np.pi))))
    factor = 2 if max(assets.shape) >= 640 else 1
    warp = assets.phase_warp("water.warp", factor, 120.0, 1.3)
    return WaveBank((h, w), waves, roi=roi, psi=_psi(h), factor=factor, warp=warp)


def _glint_color(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    others = mask != AGUA
    base = np.array([255.0, 255.0, 255.0])
    if others.any():
        luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        top = image[(luma >= np.percentile(luma, 95)) & others]
        if len(top):
            base = top.mean(axis=0)
    return np.clip(0.5 * base + 0.5 * 255.0, 0, 255).astype(np.float32)


class WaterEffect:
    def __init__(self, image: np.ndarray, mask: np.ndarray, assets: Assets) -> None:
        region = (mask == AGUA).astype(np.uint8)
        sc = max(assets.scale, 0.25)
        alpha = inner_feather(region, FEATHER_PX * sc)
        self.region = Region(alpha)
        self.active = self.region.active
        if not self.active:
            return
        self.roi = self.region.roi
        y0, y1, x0, x1 = self.roi
        self.plate = region_plate(image, region, coarse=assets.shared)
        x_local = assets.grid_x[y0:y1, x0:x1]
        y_local = assets.grid_y[y0:y1, x0:x1]
        self.x = self.region.pick(x_local)
        self.y = self.region.pick(y_local)

        if assets.shared:
            self.bank = assets.cached("water.bank", lambda: _build_bank(assets, None))
        else:
            self.bank = _build_bank(assets, self.roi)

        # Amplitud: casi nula en la orilla lejana y mayor cerca de la cámara.
        rows = max(y1 - y0 - 1, 1)
        rho = (np.arange(y1 - y0, dtype=np.float32) / rows)[:, None]
        amp = (0.16 + 0.84 * rho**1.3) * AMPLITUDE_PX * sc
        self.amp_local = amp
        self.amp = self.region.pick(np.tile(amp, (1, x1 - x0)))
        self.amps_y = (1.0, 0.8, 0.55, 0.32, 0.18, 0.1)
        self.amps_x = (0.35, 0.3, 0.25, 0.16, 0.0, 0.0)
        self.amps_slope = (0.20, 0.28, 0.34, 0.34, 0.30, 0.26)

        # Línea de agua por columna, para reflejar en espejo lo que hay encima.
        h = assets.shape[0]
        has = region[y0:y1, x0:x1] > 0
        first = np.where(has.any(axis=0), has.argmax(axis=0) + y0, y0).astype(np.float32)
        k = max(3, int(40 * sc)) | 1
        line = cv2.GaussianBlur(first.reshape(1, -1), (k, 1), 0).ravel()
        mirror_y = np.clip(2.0 * line[None, :] - y_local - 1.0, 0.0, h - 1.0)
        # El reflejo es blando: basta con media resolución.
        self.roi_size = (x1 - x0, y1 - y0)
        self.mirror_src = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)
        self.mirror_x = x_local[::2, ::2] * 0.5
        self.mirror_y = mirror_y[::2, ::2] * 0.5
        # Fresnel: el reflejo pesa más hacia el horizonte.
        self.reflect = self.region.pick(clip_alpha(np.tile(0.03 + 0.20 * (1.0 - rho) ** 1.4, (1, x1 - x0))))
        self.has_above = y0 > 0

        # Color de los destellos, a partir de lo más brillante de la escena.
        self.glint_color = assets.cached("water.glint", lambda: _glint_color(image, mask))
        self.glint_scale = self.region.pick(np.tile(0.35 + 0.65 * rho, (1, x1 - x0)))

    def apply(self, canvas: np.ndarray, ft: FrameTime) -> None:
        if not self.active:
            return
        roi, up = self.roi, self.region.from_low
        dy_low = self.bank.field(ft.t, self.amps_y, roi)
        dx_low = self.bank.field(ft.t, self.amps_x, roi, quadrature=True)
        dy = up(dy_low) * self.amp
        dx = up(dx_low) * self.amp
        slope = up(self.bank.field(ft.t, self.amps_slope, roi, quadrature=True))

        layer = cv2.remap(
            self.plate, self.x + dx, self.y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        if self.has_above:
            mh, mw = self.mirror_x.shape
            step = 2 // self.bank.factor
            amp_half = self.amp_local[::2]
            mirrored = cv2.remap(
                self.mirror_src,
                self.mirror_x + dx_low[::step, ::step][:mh, :mw] * amp_half,
                self.mirror_y + 1.5 * dy_low[::step, ::step][:mh, :mw] * amp_half,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            layer = blend(layer, up(mirrored), self.reflect)

        # Sombreado por pendiente: da relieve aunque el agua de la foto sea lisa.
        layer = scale_pixels(layer, 1.0 + 0.11 * slope)

        # Destellos: solo las crestas más orientadas a la luz.
        glint = np.maximum(slope - 1.02, 0.0)
        glint = glint * glint * 2.2 * self.glint_scale
        layer = add_tinted(layer, glint, self.glint_color)

        self.region.composite(canvas, layer)
