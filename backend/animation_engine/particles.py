"""Partículas del mood y hojas que caen.

Cada partícula tiene un ciclo de vida que se repite un número entero de veces en
el periodo del loop, de modo que su trayectoria es exactamente periódica.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .assets import Assets
from .constants import FUEGO_LUZ, VEGETACION
from .timing import FrameTime

_SUBPIXEL = 4
_SIZE_BUCKETS = 5


@dataclass(frozen=True)
class _Style:
    size_px: tuple[float, float]
    velocity_x: tuple[float, float]
    velocity_y: tuple[float, float]
    life_s: tuple[float, float]
    wobble_px: float
    brightness: float
    colors: tuple[tuple[float, float, float], ...]  # BGR
    twinkle: float  # 0 = constante, 1 = se apaga por completo
    twinkle_cycles: tuple[int, int]
    halo: float
    star: bool


_STYLES: dict[str, _Style] = {
    "polen": _Style(
        size_px=(1.6, 3.2), velocity_x=(-9.0, 9.0), velocity_y=(-12.0, -2.0), life_s=(2.0, 4.0),
        wobble_px=7.0, brightness=0.70, colors=((165, 232, 255), (190, 245, 250), (130, 215, 245)),
        twinkle=0.25, twinkle_cycles=(1, 2), halo=0.35, star=False,
    ),
    "brasas": _Style(
        size_px=(1.8, 4.2), velocity_x=(-22.0, 22.0), velocity_y=(-125.0, -50.0), life_s=(1.2, 2.6),
        wobble_px=14.0, brightness=1.0, colors=((40, 135, 255), (25, 90, 255), (60, 170, 255)),
        twinkle=0.45, twinkle_cycles=(3, 7), halo=0.6, star=False,
    ),
    "chispas": _Style(
        size_px=(1.8, 3.8), velocity_x=(-12.0, 12.0), velocity_y=(-22.0, 8.0), life_s=(1.2, 2.6),
        wobble_px=9.0, brightness=1.0,
        colors=((255, 235, 190), (255, 190, 245), (210, 255, 255), (255, 255, 255)),
        twinkle=1.0, twinkle_cycles=(2, 5), halo=0.5, star=True,
    ),
    "polvo": _Style(
        size_px=(1.8, 4.4), velocity_x=(-7.0, 7.0), velocity_y=(-3.0, 8.0), life_s=(2.5, 4.5),
        wobble_px=5.0, brightness=0.42, colors=((175, 210, 240), (150, 190, 225)),
        twinkle=0.2, twinkle_cycles=(1, 1), halo=0.25, star=False,
    ),
}


def _build_sprites(sizes: np.ndarray, halo: float, star: bool) -> list[list[list[np.ndarray]]]:
    """Sprites float32 de 0 a 1 por tamaño y por posición subpíxel."""
    table: list[list[list[np.ndarray]]] = []
    for sigma in sizes:
        radius = int(np.ceil(sigma * 4.5)) + 1
        coords = np.arange(-radius, radius + 1, dtype=np.float32)
        rows: list[list[np.ndarray]] = []
        for sy in range(_SUBPIXEL):
            row: list[np.ndarray] = []
            for sx in range(_SUBPIXEL):
                dx = (coords - sx / _SUBPIXEL)[None, :]
                dy = (coords - sy / _SUBPIXEL)[:, None]
                r2 = dx**2 + dy**2
                sprite = np.exp(-r2 / (2.0 * (sigma * 0.55) ** 2)) + halo * np.exp(-r2 / (2.0 * (sigma * 1.6) ** 2))
                if star:
                    arm = sigma * 2.2
                    sprite = sprite + 0.8 * (
                        np.exp(-(dy**2) / 0.32) * np.exp(-(dx**2) / (2 * arm * arm))
                        + np.exp(-(dx**2) / 0.32) * np.exp(-(dy**2) / (2 * arm * arm))
                    )
                row.append(np.clip(sprite, 0.0, 1.0).astype(np.float32))
            rows.append(row)
        table.append(rows)
    return table


def _life_cycles(
    rng: np.random.Generator, count: int, life: tuple[float, float], speed: float, period: float
) -> np.ndarray:
    """Vidas por periodo. Son enteras, para que el ciclo cierre con el loop."""
    desired = rng.uniform(life[0], life[1], count) / max(speed, 1e-3)
    return np.maximum(1, np.round(period / desired)).astype(np.int64)


class MoodParticles:
    def __init__(self, image: np.ndarray, mask: np.ndarray, assets: Assets) -> None:
        mood = assets.mood
        h, w = assets.shape
        self.shape = assets.shape
        self.count = 0
        if mood.particle_kind == "none" or mood.particle_density <= 0:
            return
        style = _STYLES[mood.particle_kind]
        self.count = int(round(mood.particle_density * h * w / 1e6))
        if self.count == 0:
            return
        n = self.count
        rng = assets.rng(f"particles.{mood.particle_kind}")
        sc = max(assets.scale, 0.25)
        self.period = assets.period

        self.lives = _life_cycles(rng, n, style.life_s, mood.speed, assets.period)
        self.offset = rng.random(n)
        life_s = assets.period / self.lives
        self.travel_x = rng.uniform(*style.velocity_x, n) * mood.speed * sc * life_s
        self.travel_y = rng.uniform(*style.velocity_y, n) * mood.speed * sc * life_s

        self.x0 = rng.uniform(0, w, n)
        self.y0 = rng.uniform(0, h, n)
        self.centered = True
        if mood.particle_kind == "brasas":
            # Las brasas salen del fuego si lo hay; si no, de la parte baja.
            fire_y, fire_x = np.nonzero(mask == FUEGO_LUZ)
            from_fire = rng.random(n) < 0.6
            if len(fire_y):
                idx = rng.integers(0, len(fire_y), n)
                self.x0 = np.where(from_fire, fire_x[idx], self.x0)
                self.y0 = np.where(from_fire, fire_y[idx], self.y0)
            else:
                self.y0 = np.where(from_fire, rng.uniform(0.55 * h, 1.05 * h, n), self.y0)
            self.centered = False  # nacen en y0 y suben desde ahí

        self.wobble = style.wobble_px * sc * rng.uniform(0.4, 1.0, n)
        self.wobble_cycles = rng.integers(1, 3, n)
        self.wobble_phase = rng.uniform(0, 2 * np.pi, n)
        colors = np.array(style.colors, np.float32)
        self.colors = colors[rng.integers(0, len(colors), n)]
        # Con composición alfa, atenuar el color daría motas oscuras: se atenúa la opacidad.
        self.opacity = style.brightness
        self.twinkle = style.twinkle
        self.twinkle_cycles = rng.integers(style.twinkle_cycles[0], style.twinkle_cycles[1] + 1, n)
        self.twinkle_phase = rng.uniform(0, 2 * np.pi, n)

        sizes = np.linspace(style.size_px[0], style.size_px[1], _SIZE_BUCKETS) * max(sc, 0.5)
        self.sprites = _build_sprites(sizes, style.halo, style.star)
        # Sesgo a tamaños pequeños: pocas partículas grandes parecen más cercanas.
        self.bucket = np.minimum((rng.random(n) ** 1.6 * _SIZE_BUCKETS).astype(np.int64), _SIZE_BUCKETS - 1)
        self.front = rng.random(n) < 0.45

    def render(self, canvas: np.ndarray, ft: FrameTime, front: bool) -> None:
        if self.count == 0:
            return
        age = (self.lives * (ft.t / self.period) + self.offset) % 1.0
        wob = self.wobble * np.sin(2.0 * np.pi * self.wobble_cycles * age + self.wobble_phase)
        origin = 0.5 if self.centered else 0.0
        x = self.x0 + self.travel_x * (age - origin) + wob
        y = self.y0 + self.travel_y * (age - origin)
        env = np.sin(np.pi * age) ** 0.8
        tw = 0.5 + 0.5 * np.sin(2.0 * np.pi * self.twinkle_cycles * age + self.twinkle_phase)
        intensity = env * (1.0 - self.twinkle + self.twinkle * tw)

        h, w = self.shape
        xi = np.floor(x).astype(np.int64)
        yi = np.floor(y).astype(np.int64)
        fx = np.minimum(((x - xi) * _SUBPIXEL).astype(np.int64), _SUBPIXEL - 1)
        fy = np.minimum(((y - yi) * _SUBPIXEL).astype(np.int64), _SUBPIXEL - 1)
        for i in np.nonzero((self.front == front) & (intensity > 0.01))[0]:
            sprite = self.sprites[self.bucket[i]][fy[i]][fx[i]]
            radius = sprite.shape[0] // 2
            ax0, ay0 = int(xi[i]) - radius, int(yi[i]) - radius
            ax1, ay1 = ax0 + sprite.shape[1], ay0 + sprite.shape[0]
            cx0, cy0, cx1, cy1 = max(ax0, 0), max(ay0, 0), min(ax1, w), min(ay1, h)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            part = sprite[cy0 - ay0 : cy1 - ay0, cx0 - ax0 : cx1 - ax0]
            # Composición con alfa: una brasa sigue naranja sobre un cielo claro.
            alpha = part * (float(intensity[i]) * self.opacity)
            roi = canvas[cy0:cy1, cx0:cx1]
            solid = np.empty_like(roi)
            solid[:] = self.colors[i]
            canvas[cy0:cy1, cx0:cx1] = cv2.blendLinear(roi, solid, 1.0 - alpha, alpha)


class FallingLeaves:
    """Hojas de la regla de bosque, con el color de la vegetación de la propia foto."""

    def __init__(self, image: np.ndarray, mask: np.ndarray, assets: Assets) -> None:
        mood = assets.mood
        h, w = assets.shape
        self.shape = assets.shape
        self.count = 0
        veg_y, veg_x = np.nonzero(mask == VEGETACION)
        if len(veg_y) == 0:
            return
        rng = assets.rng("particles.leaves")
        sc = max(assets.scale, 0.25)
        frac = len(veg_y) / float(h * w)
        n = int(np.clip(round((16 + 50 * (frac - 0.25)) * h * w / 921600.0), 5, 70))
        self.count = n
        self.period = assets.period

        self.lives = _life_cycles(rng, n, (3.0, 5.0), mood.speed, assets.period)
        self.offset = rng.random(n)
        life_s = assets.period / self.lives
        self.travel_y = rng.uniform(55.0, 105.0, n) * mood.speed * sc * life_s
        self.travel_x = rng.uniform(-28.0, 28.0, n) * mood.speed * sc * life_s
        self.sway = rng.uniform(10.0, 30.0, n) * sc
        self.sway_cycles = rng.integers(1, 3, n)
        self.sway_phase = rng.uniform(0, 2 * np.pi, n)
        self.spin_cycles = rng.integers(1, 4, n) * rng.choice([-1, 1], n)
        self.spin_phase = rng.uniform(0, 2 * np.pi, n)
        self.flip_cycles = rng.integers(1, 4, n)
        self.flip_phase = rng.uniform(0, 2 * np.pi, n)
        self.size = rng.uniform(4.5, 10.5, n) * sc

        # Nacen en la vegetación, con preferencia por lo alto.
        idx = rng.integers(0, len(veg_y), n * 2).reshape(n, 2)
        best = np.take_along_axis(idx, veg_y[idx].argmin(axis=1)[:, None], axis=1).ravel()
        self.x0 = veg_x[best].astype(np.float64)
        self.y0 = veg_y[best].astype(np.float64)

        sample = rng.integers(0, len(veg_y), n)
        picks = image[veg_y[sample], veg_x[sample]].astype(np.float32)
        # Algunas hojas viran a otoño, para destacar contra el follaje.
        autumn = np.array([38.0, 130.0, 205.0], np.float32)
        mix = (rng.random(n) < 0.55)[:, None] * rng.uniform(0.55, 0.9, n)[:, None]
        self.colors = np.clip((picks * 1.25 + 18.0) * (1.0 - mix) + autumn * mix, 0, 255)
        self.front = rng.random(n) < 0.35

        # Silueta de hoja: elipse con puntas.
        theta = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
        self.shape_x = np.cos(theta)
        self.shape_y = np.sin(theta) * (0.42 + 0.14 * np.abs(np.cos(theta)))

    def render(self, canvas: np.ndarray, ft: FrameTime, front: bool) -> None:
        if self.count == 0:
            return
        age = (self.lives * (ft.t / self.period) + self.offset) % 1.0
        a = 2.0 * np.pi * age
        x = self.x0 + self.travel_x * age + self.sway * np.sin(self.sway_cycles * a + self.sway_phase)
        y = self.y0 + self.travel_y * age
        spin = self.spin_cycles * a + self.spin_phase
        flip = 0.3 + 0.7 * np.abs(np.cos(self.flip_cycles * a + self.flip_phase))
        env = np.clip(np.minimum(age / 0.1, (1.0 - age) / 0.2), 0.0, 1.0)

        h, w = self.shape
        for i in np.nonzero((self.front == front) & (env > 0.01))[0]:
            size = float(self.size[i])
            pad = int(size) + 3
            cx, cy = float(x[i]), float(y[i])
            x0, y0 = int(cx) - pad, int(cy) - pad
            x1, y1 = int(cx) + pad + 1, int(cy) + pad + 1
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue
            cs, sn = float(np.cos(spin[i])), float(np.sin(spin[i]))
            px = self.shape_x * size
            py = self.shape_y * size * float(flip[i])
            pts = np.stack([cx - x0 + px * cs - py * sn, cy - y0 + px * sn + py * cs], axis=1)
            pts16 = np.round(pts * 16.0).astype(np.int32)
            roi = canvas[y0:y1, x0:x1]
            drawn = roi.copy()
            color = tuple(float(c) * (0.8 + 0.2 * float(flip[i])) for c in self.colors[i])
            cv2.fillConvexPoly(drawn, pts16, color, lineType=cv2.LINE_AA, shift=4)
            canvas[y0:y1, x0:x1] = cv2.addWeighted(drawn, float(env[i]), roi, 1.0 - float(env[i]), 0.0)


class ParticleLayer:
    """Agrupa partículas del mood y hojas. ``particle_kind='none'`` apaga también las hojas."""

    def __init__(self, image: np.ndarray, mask: np.ndarray, assets: Assets, forest: bool) -> None:
        self.mood_particles: Optional[MoodParticles] = None
        self.leaves: Optional[FallingLeaves] = None
        if assets.mood.particle_kind == "none":
            return
        self.mood_particles = MoodParticles(image, mask, assets)
        if forest:
            self.leaves = FallingLeaves(image, mask, assets)

    def render(self, canvas: np.ndarray, ft: FrameTime, front: bool) -> None:
        if self.leaves is not None:
            self.leaves.render(canvas, ft, front)
        if self.mood_particles is not None:
            self.mood_particles.render(canvas, ft, front)
