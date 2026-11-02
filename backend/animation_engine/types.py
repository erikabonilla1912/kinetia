"""Tipos públicos del motor de animación. Ver docs/CONTRATOS.md, sección 5."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

MoodName = Literal["calma", "epico", "magico", "nostalgico"]
ParticleKind = Literal["none", "polen", "brasas", "chispas", "polvo"]


@dataclass(frozen=True)
class MoodParams:
    name: MoodName
    speed: float  # multiplica la velocidad de todos los efectos
    saturation: float  # multiplica la saturación HSV
    warmth: float  # -1 (frío) a 1 (cálido)
    contrast: float  # multiplica el contraste
    particle_kind: ParticleKind
    particle_density: float  # partículas por megapíxel
    parallax_amplitude: float  # desplazamiento máximo, fracción del ancho
    vignette: float  # 0 a 1


@dataclass(frozen=True)
class BrushStroke:
    points: tuple[tuple[float, float], ...]  # normalizados de 0 a 1
    radius: float
    direction_deg: float
    strength: float


@dataclass(frozen=True)
class RenderConfig:
    mood: MoodParams
    duration_s: float = 4.0
    fps: int = 24
    seed: int = 0
    crossfade_s: float = 0.75
    strokes: tuple[BrushStroke, ...] = ()
    brush_mode: Literal["add", "only"] = "add"


ProgressFn = Callable[[float], None]  # recibe valores de 0 a 1
SegmentFn = Callable[[np.ndarray], np.ndarray]  # frame BGR -> mask (H, W)
