"""Motor de animación de KINETIA. Solo usa NumPy y OpenCV. Ver docs/CONTRATOS.md, sección 5."""
from __future__ import annotations

from .image import animate_image
from .moods import get_mood
from .types import BrushStroke, MoodName, MoodParams, ParticleKind, ProgressFn, RenderConfig, SegmentFn
from .video import animate_video

__all__ = [
    "BrushStroke",
    "MoodName",
    "MoodParams",
    "ParticleKind",
    "ProgressFn",
    "RenderConfig",
    "SegmentFn",
    "animate_image",
    "animate_video",
    "get_mood",
]
