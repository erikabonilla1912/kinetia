"""Límite de hilos de OpenCV mientras se renderiza un frame.

Los efectos son operaciones cortas: con todos los núcleos, los hilos pasan más tiempo
esperándose que trabajando y el tiempo de CPU se duplica sin acortar el render.
El límite se aplica solo durante el cálculo y se restaura antes de entregar cada frame.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import cv2

RENDER_THREADS = 4


@contextmanager
def render_threads() -> Iterator[None]:
    previous = cv2.getNumThreads()
    target = min(previous, RENDER_THREADS)
    if target != previous:
        cv2.setNumThreads(target)
    try:
        yield
    finally:
        if target != previous:
            cv2.setNumThreads(previous)
