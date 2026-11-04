"""Cielo: nubes por desplazamiento de textura; estrellas que titilan de noche."""
from __future__ import annotations

import cv2
import numpy as np

from ..assets import Assets
from ..constants import CIELO, night_luma_threshold
from ..masks import Region, blend, clip_alpha, inner_feather, region_plate
from ..noise import spectral_noise
from ..timing import FrameTime, cycles, omega

FEATHER_PX = 10.0
BASE_DRIFT_PX_S = 26.0
STAR_GROUPS = 6
STARS_PER_MEGAPIXEL = 520.0


def is_night(image: np.ndarray, mask: np.ndarray) -> bool:
    """Regla del contrato: luminancia media (Y) de los píxeles de cielo bajo el umbral."""
    sky = mask == CIELO
    if not sky.any():
        return False
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[..., 0]
    return float(luma[sky].mean()) < night_luma_threshold()


def _build_wisps(assets: Assets) -> np.ndarray:
    h, w = assets.shape
    hh, wh = max(8, h // 2), max(8, w // 2)
    noise = spectral_noise(assets.rng("sky.wisps"), hh, wh, 2.9)
    # Se estira en horizontal: las nubes se ven alargadas.
    noise = cv2.resize(noise[:, : max(4, wh // 3)], (wh, hh), interpolation=cv2.INTER_CUBIC)
    wisp = np.clip((noise - 0.45) / 0.55, 0.0, 1.0)
    wisp = wisp * wisp * (3.0 - 2.0 * wisp)
    return cv2.GaussianBlur(wisp, (0, 0), 1.5).astype(np.float32)


def _build_stars(assets: Assets) -> list[np.ndarray]:
    h, w = assets.shape
    rng = assets.rng("sky.stars")
    count = int(STARS_PER_MEGAPIXEL * h * w / 1e6)
    if count == 0:
        return [np.zeros((h, w), np.uint8) for _ in range(STAR_GROUPS)]
    ys = rng.integers(0, h, count)
    xs = rng.integers(0, w, count)
    groups = rng.integers(0, STAR_GROUPS, count)
    # Muchas estrellas tenues y pocas brillantes, como en una foto de cielo real.
    brightness = np.clip(rng.pareto(1.6, count) * 70.0 + 70.0, 0.0, 255.0).astype(np.float32)
    big = rng.random(count) < 0.08
    sigma_small = max(0.55, 0.6 * assets.scale)
    sigma_big = max(0.9, 1.15 * assets.scale)
    out: list[np.ndarray] = []
    for g in range(STAR_GROUPS):
        small_map = np.zeros((h, w), np.float32)
        big_map = np.zeros((h, w), np.float32)
        sel = groups == g
        np.add.at(small_map, (ys[sel & ~big], xs[sel & ~big]), brightness[sel & ~big])
        np.add.at(big_map, (ys[sel & big], xs[sel & big]), brightness[sel & big])
        small_map = cv2.GaussianBlur(small_map, (0, 0), sigma_small) * (2.0 * np.pi * sigma_small**2)
        big_map = cv2.GaussianBlur(big_map, (0, 0), sigma_big) * (2.0 * np.pi * sigma_big**2)
        out.append(np.clip(small_map + big_map, 0, 255).astype(np.uint8))
    return out


def _sky_stats(image: np.ndarray, region: np.ndarray) -> tuple[bool, float, np.ndarray]:
    """Devuelve (es de noche, opacidad de los velos, color de los velos)."""
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[..., 0]
    sky_luma = luma[region > 0]
    sky_pixels = image[region > 0]
    flatness = float(np.clip(1.3 - float(sky_luma.std()) / 22.0, 0.0, 1.0))
    night = float(sky_luma.mean()) < night_luma_threshold()
    bright = sky_pixels[sky_luma >= np.percentile(sky_luma, 85)]
    tint = bright.mean(axis=0) if len(bright) else np.array([235.0, 235.0, 235.0])
    # En un cielo liso los velos de nube dan el movimiento; con nubes reales apenas se notan.
    if night:
        return True, 0.10 + 0.10 * flatness, tint * 1.3 + 12.0
    return False, 0.05 + 0.40 * flatness, 0.35 * tint + 0.65 * 245.0


class SkyEffect:
    def __init__(self, image: np.ndarray, mask: np.ndarray, assets: Assets) -> None:
        region = (mask == CIELO).astype(np.uint8)
        alpha = inner_feather(region, FEATHER_PX * max(assets.scale, 0.25))
        self.region = Region(alpha)
        self.active = self.region.active
        if not self.active:
            return
        y0, y1, x0, x1 = self.region.roi
        self.plate = region_plate(image, region, coarse=assets.shared)
        rng = assets.rng("sky.drift")
        sign = 1.0 if rng.random() < 0.5 else -1.0
        tilt = float(rng.uniform(-0.09, 0.09))
        self.ux, self.uy = sign * float(np.cos(tilt)), float(np.sin(tilt))
        self.drift = BASE_DRIFT_PX_S * assets.mood.speed * assets.scale

        x_local = assets.grid_x[y0:y1, x0:x1]
        y_local = assets.grid_y[y0:y1, x0:x1]
        # Más arriba las nubes parecen más cercanas y se mueven más rápido.
        h = assets.shape[0]
        profile = 1.45 - 0.9 * np.clip(y_local / max(h * 0.6, 1.0), 0.0, 1.0)
        self.x = self.region.pick(x_local)
        self.y = self.region.pick(y_local)
        self.profile = self.region.pick(profile)

        self.wisps = assets.cached("sky.wisps", lambda: _build_wisps(assets))
        hh, wh = self.wisps.shape
        self.roi_size = (x1 - x0, y1 - y0)
        self.x_half = x_local[::2, ::2]
        self.y_half = y_local[::2, ::2]
        self.profile_half = profile[::2, ::2]
        self.wisp_fx = wh / assets.shape[1]
        self.wisp_fy = hh / assets.shape[0]

        # En video se fijan con el primer frame que tenga cielo, para que no parpadeen.
        self.night, self.wisp_gain, tint = assets.cached("sky.stats", lambda: _sky_stats(image, region))
        self.tint_layer = np.full(self.x.shape + (3,), np.clip(tint, 0, 255), np.uint8)

        self.star_maps = None
        if self.night:
            stars = assets.cached("sky.stars", lambda: _build_stars(assets))
            self.star_maps = [s[y0:y1, x0:x1] for s in stars]
            # Las estrellas no se ven sobre zonas iluminadas del cielo (luna, nubes).
            luma = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[..., 0]
            blurred = cv2.GaussianBlur(luma, (0, 0), 3.0)[y0:y1, x0:x1].astype(np.float32)
            self.star_gain = self.region.pick(clip_alpha(1.0 - blurred / 105.0))
            base = [0.55, 0.8, 1.2, 1.7, 0.4, 1.0]
            self.star_omega = [omega(cycles(b, assets.mood.speed, assets.period), assets.period) for b in base]
            self.star_phase = assets.rng("sky.twinkle").uniform(0, 2 * np.pi, STAR_GROUPS)

    def _layer(self, t: float) -> np.ndarray:
        offset = t * self.drift
        shift = offset * self.profile
        map_x = self.x - self.ux * shift
        map_y = self.y - self.uy * shift
        warped = cv2.remap(self.plate, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # Los velos van algo más rápido que la foto: dan sensación de capas.
        wx = (self.x_half - 1.5 * self.ux * offset * self.profile_half) * self.wisp_fx
        wy = (self.y_half - 1.5 * self.uy * offset * self.profile_half) * self.wisp_fy
        wisp = cv2.remap(self.wisps, wx, wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        wisp = self.region.from_low(wisp)
        return blend(warped, self.tint_layer, clip_alpha(wisp * self.wisp_gain))

    def _stars(self, t: float) -> np.ndarray:
        total = None
        for g, smap in enumerate(self.star_maps):
            amp = 0.55 + 0.45 * float(np.sin(self.star_omega[g] * t + self.star_phase[g]))
            layer = smap.astype(np.float32) * amp
            total = layer if total is None else total + layer
        return self.region.pick(total)

    def apply(self, canvas: np.ndarray, ft: FrameTime) -> None:
        if not self.active:
            return
        layers = [(self._layer(t), w) for t, w in ft.sky_layers if w > 0.0]
        if len(layers) == 1:
            sky = layers[0][0]
        else:
            sky = cv2.addWeighted(layers[0][0], layers[0][1], layers[1][0], layers[1][1], 0.0)
        if self.star_maps is not None:
            stars = cv2.convertScaleAbs(self._stars(ft.t) * self.star_gain)
            sky = cv2.add(sky, cv2.merge([stars, stars, stars]))
        self.region.composite(canvas, sky)
