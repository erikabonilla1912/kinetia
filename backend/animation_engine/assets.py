"""Recursos aleatorios de un render: dependen del seed, no de la máscara ni del tiempo.

En video se reutilizan entre frames, para que el ruido y las ondas sean continuos.
"""
from __future__ import annotations

import zlib
from typing import Callable

import cv2
import numpy as np

from .constants import REFERENCE_LONG_SIDE
from .types import RenderConfig


class Assets:
    def __init__(self, shape: tuple[int, int], config: RenderConfig, period: float, shared: bool) -> None:
        self.shape = shape
        self.config = config
        self.mood = config.mood
        self.period = period
        # shared=True (video): los recursos se calculan para todo el frame y se guardan.
        self.shared = shared
        self.scale = max(shape) / REFERENCE_LONG_SIDE
        self.grid_y, self.grid_x = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
        self._cache: dict[str, object] = {}

    def rng(self, tag: str) -> np.random.Generator:
        """Generador independiente por uso. No usa estado global ni ``hash()``."""
        seed = int(self.config.seed) & ((1 << 64) - 1)
        return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(tag.encode())]))

    def phase_warp(self, tag: str, factor: int, wavelength_px: float, radians: float) -> np.ndarray:
        """Ruido suave para deformar la fase de un banco de ondas (a resolución 1/factor)."""
        h, w = self.shape[0] // factor + 2, self.shape[1] // factor + 2
        coarse = max(2, int(wavelength_px * max(self.scale, 0.25) / factor))
        small_h, small_w = h // coarse + 3, w // coarse + 3
        noise = self.rng(tag).standard_normal((small_h, small_w)).astype(np.float32)
        field = cv2.resize(noise, (w + coarse, h + coarse), interpolation=cv2.INTER_CUBIC)[:h, :w]
        return field * radians

    def cached(self, key: str, build: Callable[[], object]):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]
