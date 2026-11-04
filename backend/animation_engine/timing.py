"""Reloj de la animación.

Todos los efectos periódicos dan un número entero de ciclos en el periodo del
loop, de modo que el frame N coincidiría con el frame 0. Lo único que no es
periódico es el desplazamiento de textura del cielo: se resuelve con
``sky_layers``, una lista de instantes con peso que implementa el crossfade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameTime:
    index: int
    t: float  # segundos
    period: float  # segundos que dura un ciclo completo del loop
    sky_layers: tuple[tuple[float, float], ...]  # (instante, peso)


def cycles(base_hz: float, speed: float, period: float) -> int:
    """Ciclos enteros (mínimo 1) más cercanos a ``base_hz * speed`` en el periodo."""
    return max(1, int(round(base_hz * speed * period)))


def omega(n_cycles: int, period: float) -> float:
    return 2.0 * math.pi * n_cycles / period


class ImageTimeline:
    """Loop cerrado de N frames con crossfade de ``crossfade_s`` al final."""

    def __init__(self, n_frames: int, fps: int, crossfade_s: float) -> None:
        self.n_frames = n_frames
        self.fps = fps
        self.period = n_frames / fps
        self.crossfade_frames = min(int(round(crossfade_s * fps)), n_frames // 2)

    def frame(self, index: int) -> FrameTime:
        t = index / self.fps
        start = self.n_frames - self.crossfade_frames
        if self.crossfade_frames > 0 and index >= start:
            w = (index - start + 1) / (self.crossfade_frames + 1)
            layers = ((t, 1.0 - w), (t - self.period, w))
        else:
            layers = ((t, 1.0),)
        return FrameTime(index, t, self.period, layers)


class VideoTimeline:
    """Tiempo abierto para video. El cielo usa dos fases para no acumular
    desplazamiento: dos copias desfasadas medio periodo, con pesos triangulares."""

    def __init__(self, fps: int, period: float) -> None:
        self.fps = fps
        self.period = period

    def frame(self, index: int) -> FrameTime:
        t = index / self.fps
        phase = (t / self.period) % 1.0
        w = abs(2.0 * phase - 1.0)
        layers = (
            ((phase - 0.5) * self.period, 1.0 - w),
            ((((phase + 0.5) % 1.0) - 0.5) * self.period, w),
        )
        return FrameTime(index, t, self.period, layers)
