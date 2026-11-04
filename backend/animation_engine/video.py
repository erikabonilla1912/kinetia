"""Animación de video: inferencia cada N frames y Farneback entre medias."""
from __future__ import annotations

from typing import Iterable, Iterator, Optional

import cv2
import numpy as np

from .assets import Assets
from .effects.brush import BrushEffect
from .effects.vegetation import is_forest
from .grading import Grader
from .particles import ParticleLayer
from .scene import Scene
from .threads import render_threads
from .timing import VideoTimeline
from .types import ProgressFn, RenderConfig, SegmentFn
from .validation import normalize_depth, validate_config, validate_image, validate_mask

FLOW_MAX_SIDE = 480


def _flow_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    long_side = max(h, w)
    if long_side <= FLOW_MAX_SIDE:
        return gray
    factor = FLOW_MAX_SIDE / long_side
    size = (max(1, int(round(w * factor))), max(1, int(round(h * factor))))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def propagate_mask(prev_gray: np.ndarray, gray: np.ndarray, prev_mask: np.ndarray) -> np.ndarray:
    """Lleva ``prev_mask`` al frame actual con flujo óptico Farneback.

    Calcula el flujo hacia atrás (actual a anterior): cada píxel del frame actual
    busca su clase en el anterior, así no quedan huecos en la máscara.
    """
    h, w = prev_mask.shape
    # Ventana y pirámide para movimientos de varios píxeles por frame a resolución reducida.
    flow = cv2.calcOpticalFlowFarneback(
        gray, prev_gray, None, pyr_scale=0.5, levels=3, winsize=21, iterations=3,
        poly_n=5, poly_sigma=1.1, flags=0,
    )
    fh, fw = flow.shape[:2]
    sx, sy = w / fw, h / fh
    if (fh, fw) != (h, w):
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    map_x = xs + flow[..., 0] * sx
    map_y = ys + flow[..., 1] * sy
    return cv2.remap(prev_mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


def animate_video(
    frames: Iterable[np.ndarray],
    segment: SegmentFn,
    infer_every_n: int,
    config: RenderConfig,
    depth: Optional[np.ndarray] = None,
    progress: Optional[ProgressFn] = None,
) -> Iterator[np.ndarray]:
    """Llama a segment() cada infer_every_n frames y, entre medias, propaga la
    máscara con flujo óptico Farneback. No acumula todos los frames en memoria.

    ``depth`` se valida, pero no se usa: el sujeto y el fondo quedan quietos. Los efectos periódicos
    usan ``config.duration_s`` como ciclo; el video no se cierra en loop.
    """
    if infer_every_n < 1:
        raise ValueError("infer_every_n debe ser al menos 1")
    validate_config(config)
    return _generate(frames, segment, infer_every_n, config, depth, progress)


def _generate(frames, segment, infer_every_n, config, depth, progress) -> Iterator[np.ndarray]:
    total = len(frames) if hasattr(frames, "__len__") else None
    timeline = VideoTimeline(config.fps, config.duration_s)
    shape = None
    assets = grader = brush = particles = None
    prev_gray = None
    mask = None
    for i, frame in enumerate(frames):
        validate_image(frame, "frame")
        if shape is None:
            shape = frame.shape[:2]
            normalize_depth(depth, shape)  # se valida por contrato; el motor ya no lo usa
        elif frame.shape[:2] != shape:
            raise ValueError(f"El frame {i} tiene forma {frame.shape[:2]}; se esperaba {shape}")

        gray = _flow_gray(frame)
        if i % infer_every_n == 0:
            mask = segment(frame)
            validate_mask(mask, shape)
        else:
            mask = propagate_mask(prev_gray, gray, mask)
        prev_gray = gray

        with render_threads():
            if assets is None:
                assets = Assets(shape, config, timeline.period, shared=True)
                grader = Grader(config.mood, shape)
                brush = BrushEffect(config.strokes, assets)
                # Bosque y partículas se fijan con el primer frame, para que no parpadeen.
                particles = ParticleLayer(frame, mask, assets, forest=is_forest(mask))
            scene = Scene(frame, mask, assets, particles, grader, brush)
            out = scene.render(timeline.frame(i))
        if progress is not None:
            if total:
                progress(min((i + 1) / total, 1.0))
            else:
                # Sin longitud conocida: avance monótono que no llega a 1 hasta terminar.
                progress(1.0 - 1.0 / (1.0 + (i + 1) / max(config.fps * 4.0, 1.0)))
        yield out
    if progress is not None and shape is not None:
        progress(1.0)
