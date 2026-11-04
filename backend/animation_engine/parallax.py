"""Parallax 2.5D: el sujeto se separa del fondo y la cámara respira.

El fondo se rellena una sola vez detrás del sujeto y se reutiliza en todos los
frames, de modo que al moverse el sujeto no aparecen huecos ni halos.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .assets import Assets
from .constants import ANIMAL, PERSONA
from .masks import blend, bounding_roi, inner_feather, inpaint_hole, push_pull_fill, clip_alpha
from .timing import FrameTime, cycles, omega

SUBJECT_MIN_UNIT = 0.45  # separación mínima del sujeto, aunque depth no la marque
BREATH_ZOOM_FACTOR = 0.55
BASE_CAMERA_HZ = 1.0 / 3.5
CONTACT_START = 0.6  # fracción de la altura del sujeto donde empieza el anclaje al suelo


class _Geometry:
    """Encuadre y desplazamientos del fondo. Dependen de depth, que no cambia durante el render."""

    def __init__(self, depth: np.ndarray, subject: np.ndarray, hole: np.ndarray, assets: Assets, sc: float) -> None:
        h, w = assets.shape
        mood = assets.mood
        self.height = h
        depth_bg = push_pull_fill(depth, 1 - hole) if subject.any() else depth
        depth_bg = cv2.GaussianBlur(depth_bg, (0, 0), max(1.0, 6.0 * sc))
        self.depth_near = cv2.GaussianBlur(depth, (0, 0), max(1.0, 5.0 * sc))
        self.reference = float(np.median(depth_bg))
        self.norm = max(self.reference, 1.0 - self.reference, 1e-3)
        self.unit_bg = (depth_bg - self.reference) / self.norm

        self.amp_px = float(mood.parallax_amplitude) * w
        unit_max = float(np.abs(self.unit_bg).max())
        if assets.shared:
            unit_max = max(unit_max, 1.0)  # un sujeto puede aparecer más adelante en el video
        elif subject.any():
            unit_max = max(unit_max, float(self.subject_unit(depth, subject).max()))

        # Encuadre fijo un poco más cerrado para que nunca se vean los bordes.
        cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
        self.zoom_breath = float(mood.parallax_amplitude) * BREATH_ZOOM_FACTOR
        margin_x = self.amp_px * unit_max + self.zoom_breath * unit_max * (w / 2.0)
        margin_y = 0.5 * self.amp_px * unit_max + self.zoom_breath * unit_max * (h / 2.0)
        self.spread = int(np.ceil(max(margin_x, margin_y) * 2.0)) + 4
        self.zoom = max(1.0 + 2.0 * margin_x / max(w, 1), 1.0 + 2.0 * margin_y / max(h, 1))
        self.centered_x = assets.grid_x - cx0
        self.centered_y = assets.grid_y - cy0
        self.base_x = cx0 + self.centered_x / self.zoom
        self.base_y = cy0 + self.centered_y / self.zoom

        # El campo del fondo es suave: se combina a 1/4 de resolución y se reescala.
        self.size = (w, h)
        low = (max(2, w // 4), max(2, h // 4))
        self.pan_x_bg = cv2.resize(self.amp_px * self.unit_bg, low, interpolation=cv2.INTER_AREA)
        self.pan_y_bg = self.pan_x_bg * 0.5
        self.zoom_x_bg = cv2.resize(self.centered_x * self.zoom_breath * self.unit_bg, low, interpolation=cv2.INTER_AREA)
        self.zoom_y_bg = cv2.resize(self.centered_y * self.zoom_breath * self.unit_bg, low, interpolation=cv2.INTER_AREA)

    def subject_unit(self, depth: np.ndarray, subject: np.ndarray) -> np.ndarray:
        d_mean = float(depth[subject > 0].mean())
        near = self.depth_near * 0.3 + d_mean * 0.7
        unit = np.clip((near - self.reference) / self.norm, SUBJECT_MIN_UNIT, 1.0)
        # Donde el sujeto toca el suelo se mueve con el fondo: así los pies no patinan.
        ys = np.nonzero(subject.any(axis=1))[0]
        top, bottom = int(ys[0]), int(ys[-1])
        rows = np.arange(self.height, dtype=np.float32)
        t = np.clip(((rows - top) / max(bottom - top, 1) - CONTACT_START) / (1.0 - CONTACT_START), 0.0, 1.0)
        contact = (t * t * (3.0 - 2.0 * t))[:, None]
        return unit * (1.0 - contact) + self.unit_bg * contact


class ParallaxRig:
    def __init__(self, image: np.ndarray, mask: np.ndarray, depth: np.ndarray, assets: Assets, fast: bool) -> None:
        h, w = assets.shape
        sc = max(assets.scale, 0.25)
        mood = assets.mood
        subject = ((mask == PERSONA) | (mask == ANIMAL)).astype(np.uint8)
        self.has_subject = bool(subject.any())

        radius = max(2, int(round(8 * sc)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        hole = cv2.dilate(subject, kernel)
        # Fondo sin sujeto. Se calcula aquí una sola vez y no se vuelve a tocar.
        self.bg_plate = inpaint_hole(image, hole, fast)

        # En video, el encuadre debe ser el mismo en todos los frames o la imagen saltaría.
        geo = assets.cached("parallax.geometry", lambda: _Geometry(depth, subject, hole, assets, sc))
        self.size, self.zoom = geo.size, geo.zoom
        self.base_x, self.base_y = geo.base_x, geo.base_y
        self.pan_x_bg, self.pan_y_bg = geo.pan_x_bg, geo.pan_y_bg
        self.zoom_x_bg, self.zoom_y_bg = geo.zoom_x_bg, geo.zoom_y_bg

        self.fg_roi = None
        if self.has_subject:
            unit_fg = geo.subject_unit(depth, subject)
            # Bordes del sujeto: alfa solo hacia adentro y color limpio hacia afuera.
            alpha_fg = inner_feather(subject, max(1.5, 2.5 * sc))
            alpha_fg = cv2.GaussianBlur(alpha_fg, (0, 0), max(0.6, 0.9 * sc))
            alpha_fg[subject == 0] = 0.0
            self.fg_alpha = alpha_fg
            core = cv2.erode(subject, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
            self.fg_plate = push_pull_fill(image, core if core.any() else subject, coarse=fast)
            self.fg_roi = _expand(bounding_roi(subject), geo.spread, h, w)
            y0, y1, x0, x1 = self.fg_roi
            unit = unit_fg[y0:y1, x0:x1]
            self.pan_x_fg = geo.amp_px * unit
            self.pan_y_fg = 0.5 * geo.amp_px * unit
            self.zoom_x_fg = geo.centered_x[y0:y1, x0:x1] * geo.zoom_breath * unit
            self.zoom_y_fg = geo.centered_y[y0:y1, x0:x1] * geo.zoom_breath * unit

        rng = assets.rng("parallax.camera")
        c = cycles(BASE_CAMERA_HZ, mood.speed, assets.period)
        self.w_cam = omega(c, assets.period)
        self.phase_breath = float(rng.uniform(0, 2 * np.pi))
        self.direction = 1.0 if rng.random() < 0.5 else -1.0

    def _camera(self, t: float) -> tuple[float, float, float]:
        a = self.w_cam * t
        pan_x = self.direction * float(np.sin(a))
        pan_y = float(np.sin(2.0 * a + self.phase_breath * 0.5))
        breath = float(np.sin(a + self.phase_breath))
        return pan_x, pan_y, breath

    def warp_background(self, canvas: np.ndarray, ft: FrameTime) -> np.ndarray:
        px, py, b = self._camera(ft.t)
        dx = cv2.addWeighted(self.pan_x_bg, -px, self.zoom_x_bg, -b, 0.0)
        dy = cv2.addWeighted(self.pan_y_bg, -py, self.zoom_y_bg, -b, 0.0)
        map_x = cv2.add(self.base_x, cv2.resize(dx, self.size, interpolation=cv2.INTER_LINEAR))
        map_y = cv2.add(self.base_y, cv2.resize(dy, self.size, interpolation=cv2.INTER_LINEAR))
        return cv2.remap(canvas, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def composite_subject(self, canvas: np.ndarray, ft: FrameTime, source: Optional[np.ndarray] = None) -> None:
        if not self.has_subject:
            return
        src = self.fg_plate if source is None else source
        px, py, b = self._camera(ft.t)
        y0, y1, x0, x1 = self.fg_roi
        map_x = cv2.add(self.base_x[y0:y1, x0:x1], cv2.addWeighted(self.pan_x_fg, -px, self.zoom_x_fg, -b, 0.0))
        map_y = cv2.add(self.base_y[y0:y1, x0:x1], cv2.addWeighted(self.pan_y_fg, -py, self.zoom_y_fg, -b, 0.0))
        color = cv2.remap(src, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.remap(self.fg_alpha, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        canvas[y0:y1, x0:x1] = blend(canvas[y0:y1, x0:x1], color, clip_alpha(alpha))


def _expand(roi, spread: int, h: int, w: int):
    y0, y1, x0, x1 = roi
    return max(0, y0 - spread), min(h, y1 + spread), max(0, x0 - spread), min(w, x1 + spread)
