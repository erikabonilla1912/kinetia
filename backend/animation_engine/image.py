"""Animación de una imagen: genera un loop cerrado de frames."""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from .assets import Assets
from .effects.brush import BrushEffect
from .effects.vegetation import is_forest
from .grading import Grader
from .particles import ParticleLayer
from .scene import Scene
from .threads import render_threads
from .timing import ImageTimeline
from .types import ProgressFn, RenderConfig
from .validation import normalize_depth, validate_config, validate_image, validate_mask


def animate_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    depth: Optional[np.ndarray],
    config: RenderConfig,
    progress: Optional[ProgressFn] = None,
) -> Iterator[np.ndarray]:
    """Genera round(duration_s * fps) frames BGR. Es un loop: el último frame
    empalma con el primero mediante crossfade.

    Las entradas se validan al llamar, no en el primer ``next``.
    """
    validate_image(image_bgr)
    shape = image_bgr.shape[:2]
    validate_mask(mask, shape)
    normalize_depth(depth, shape)  # se valida por contrato; el motor ya no lo usa
    validate_config(config)
    n_frames = int(round(config.duration_s * config.fps))
    if n_frames < 1:
        raise ValueError("duration_s * fps debe dar al menos 1 frame")
    return _generate(image_bgr.copy(), mask.copy(), config, n_frames, progress)


def _generate(image, mask, config, n_frames, progress) -> Iterator[np.ndarray]:
    timeline = ImageTimeline(n_frames, config.fps, config.crossfade_s)
    shape = image.shape[:2]
    with render_threads():
        assets = Assets(shape, config, timeline.period, shared=False)
        grader = Grader(config.mood, shape)
        brush = BrushEffect(config.strokes, assets)
        particles = ParticleLayer(image, mask, assets, forest=is_forest(mask))
        scene = Scene(image, mask, assets, particles, grader, brush)
    for i in range(n_frames):
        with render_threads():
            frame = scene.render(timeline.frame(i))
        if progress is not None:
            progress((i + 1) / n_frames)
        yield frame
