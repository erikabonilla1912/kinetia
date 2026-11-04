"""Escenas sintéticas para probar el motor."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from animation_engine.constants import AGUA, CIELO, FONDO, FUEGO_LUZ, PERSONA, VEGETACION  # noqa: E402
from animation_engine.types import MoodParams, RenderConfig  # noqa: E402


def neutral_mood(**overrides) -> MoodParams:
    """Mood sin color, sin partículas y sin parallax: deja ver cada efecto aislado."""
    values = dict(
        name="calma", speed=1.0, saturation=1.0, warmth=0.0, contrast=1.0,
        particle_kind="none", particle_density=0.0, parallax_amplitude=0.0, vignette=0.0,
    )
    values.update(overrides)
    return MoodParams(**values)


def neutral_config(**overrides) -> RenderConfig:
    values = dict(mood=neutral_mood(), duration_s=1.0, fps=12, seed=7)
    values.update(overrides)
    return RenderConfig(**values)


def textured_image(h: int = 180, w: int = 320, seed: int = 3) -> np.ndarray:
    """Imagen con textura en todas partes, para que cualquier desplazamiento se note."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (h // 4 + 1, w // 4 + 1, 3), dtype=np.uint8)
    img = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(img, 0.7, np.full_like(img, 90), 0.3, 0)


def full_mask(h: int = 180, w: int = 320) -> np.ndarray:
    """Máscara con cielo, agua, vegetación, fuego, persona y fondo."""
    mask = np.full((h, w), FONDO, np.uint8)
    mask[: h // 3, :] = CIELO
    mask[2 * h // 3 :, :] = AGUA
    mask[h // 3 : 2 * h // 3, : w // 3] = VEGETACION
    mask[h // 3 : h // 2, 2 * w // 3 : w - 10] = FUEGO_LUZ
    cv2.circle(mask, (w // 2, h // 2), h // 6, int(PERSONA), -1)
    return mask


def ramp_depth(h: int = 180, w: int = 320) -> np.ndarray:
    return np.tile(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], (1, w))


@pytest.fixture
def image() -> np.ndarray:
    return textured_image()


@pytest.fixture
def mask() -> np.ndarray:
    return full_mask()


@pytest.fixture
def depth() -> np.ndarray:
    return ramp_depth()
