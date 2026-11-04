"""Estilo cinemagraph: lo que no tiene efecto queda quieto y nada se ve como goma."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from animation_engine import BrushStroke, RenderConfig, animate_image, get_mood
from animation_engine.constants import AGUA, ANIMAL, CIELO, FONDO, PERSONA, VEGETACION

from .conftest import neutral_config, neutral_mood, ramp_depth, textured_image

MOODS = ("calma", "epico", "magico", "nostalgico")
SUBPIXEL_TOLERANCE = 0.6


def render(image, mask, depth, config) -> list[np.ndarray]:
    return list(animate_image(image, mask, depth, config))


def scene_without_subject(h=180, w=320) -> np.ndarray:
    mask = np.full((h, w), FONDO, np.uint8)
    mask[: h // 3] = CIELO
    mask[2 * h // 3 :] = AGUA
    mask[h // 3 : 2 * h // 3, : w // 4] = VEGETACION
    return mask


def patch_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Traslación de ``a`` a ``b`` por correlación de fase."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb)
    return dx, dy


@pytest.mark.parametrize("mood", MOODS)
def test_background_is_static_without_subject(mood):
    """Sin persona ni animal: MAE temporal 0 en fondo, aunque haya depth y parallax en el mood."""
    image, mask, depth = textured_image(), scene_without_subject(), ramp_depth()
    preset = get_mood(mood)
    no_particles = RenderConfig(
        mood=type(preset)(**{**preset.__dict__, "particle_kind": "none", "particle_density": 0.0}),
        duration_s=1.0, fps=12, seed=3,
    )
    frames = render(image, mask, depth, no_particles)
    still = mask == FONDO
    assert still.any()
    for frame in frames[1:]:
        assert float(np.mean(cv2.absdiff(frame[still], frames[0][still]))) == 0.0


def test_no_subject_means_no_camera_motion():
    """Sin sujeto, depth no cambia nada: la salida es idéntica con y sin depth."""
    image, mask = textured_image(), scene_without_subject()
    config = neutral_config(mood=neutral_mood(parallax_amplitude=0.02))
    with_depth = render(image, mask, ramp_depth(), config)
    without = render(image, mask, None, config)
    assert [f.tobytes() for f in with_depth] == [f.tobytes() for f in without]


def _standing_scene(cls, h=360, w=640):
    """Sujeto de pie sobre una franja de suelo, con cielo y agua detrás."""
    image = textured_image(h, w)
    mask = np.full((h, w), FONDO, np.uint8)
    mask[: h // 3] = CIELO
    mask[h // 3 : 2 * h // 3, : w // 4] = AGUA
    cv2.ellipse(mask, (w // 2, h // 2), (70, 110), 0, 0, 360, int(cls), -1)
    return image, mask, ramp_depth(h, w)


def _no_particles(name: str) -> RenderConfig:
    preset = get_mood(name)
    mood = type(preset)(**{**preset.__dict__, "particle_kind": "none", "particle_density": 0.0})
    return RenderConfig(mood=mood, duration_s=1.0, fps=12, seed=3)


def test_presets_have_no_parallax():
    for name in MOODS:
        assert get_mood(name).parallax_amplitude == 0.0


@pytest.mark.parametrize("cls", [PERSONA, ANIMAL])
@pytest.mark.parametrize("mood", MOODS)
def test_subject_is_rigid_and_still(cls, mood):
    """MAE temporal 0 dentro del sujeto erosionado, sin partículas."""
    image, mask, depth = _standing_scene(cls)
    frames = render(image, mask, depth, _no_particles(mood))
    inner = cv2.erode((mask == cls).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    assert inner.any()
    for frame in frames[1:]:
        assert float(np.mean(cv2.absdiff(frame[inner], frames[0][inner]))) == 0.0


@pytest.mark.parametrize("cls", [PERSONA, ANIMAL])
def test_subject_is_still_even_with_a_large_amplitude(cls):
    """Un parallax_amplitude grande, sin importar de dónde venga, no mueve nada."""
    image, mask, depth = _standing_scene(cls)
    config = neutral_config(mood=neutral_mood(parallax_amplitude=0.05))
    frames = render(image, mask, depth, config)
    still = (mask == cls) | (mask == FONDO)
    assert all(np.array_equal(f[still], image[still]) for f in frames)


@pytest.mark.parametrize("cls", [PERSONA, ANIMAL])
def test_ground_beside_feet_is_static(cls):
    """La franja de suelo junto a los pies no cambia: los pies no pueden resbalar."""
    image, mask, depth = _standing_scene(cls)
    ys = np.nonzero((mask == cls).any(axis=1))[0]
    feet = int(ys[-1])
    near = cv2.dilate((mask == cls).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    band = near & (mask == FONDO)
    band[: feet - 20] = False
    assert band.any()
    frames = render(image, mask, depth, _no_particles("epico"))
    for frame in frames[1:]:
        assert float(np.mean(cv2.absdiff(frame[band], frames[0][band]))) == 0.0


@pytest.mark.parametrize("cls", [PERSONA, ANIMAL])
def test_subject_scene_is_deterministic(cls):
    image, mask, depth = _standing_scene(cls)
    config = RenderConfig(mood=get_mood("epico"), duration_s=0.5, fps=12, seed=9)
    a = render(image, mask, depth, config)
    b = render(image, mask, depth, config)
    assert [f.tobytes() for f in a] == [f.tobytes() for f in b]
    other = RenderConfig(mood=get_mood("epico"), duration_s=0.5, fps=12, seed=10)
    assert render(image, mask, depth, other)[3].tobytes() != a[3].tobytes()


def test_depth_does_not_change_the_output():
    image, mask, depth = _standing_scene(ANIMAL)
    config = RenderConfig(mood=get_mood("magico"), duration_s=0.5, fps=12, seed=9)
    a = render(image, mask, depth, config)
    b = render(image, mask, None, config)
    assert [f.tobytes() for f in a] == [f.tobytes() for f in b]


def test_particles_pass_behind_the_subject():
    """Hay partículas detrás del sujeto: lo tocan menos que al fondo, que da la profundidad."""
    h, w = 360, 640
    image = textured_image(h, w)
    mask = np.full((h, w), FONDO, np.uint8)
    mask[60:300, 200:440] = PERSONA
    mood = neutral_mood(particle_kind="polvo", particle_density=600.0)
    frames = render(image, mask, None, neutral_config(mood=mood))
    touched = np.any(frames[4] != image, axis=2)
    inner = cv2.erode((mask == PERSONA).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    outside = mask == FONDO
    assert 0.0 < touched[inner].mean() < 0.8 * touched[outside].mean()


def test_brush_add_still_animates_painted_subject():
    image, mask, depth = _standing_scene(PERSONA)
    stroke = BrushStroke(points=((0.5, 0.5),), radius=0.04, direction_deg=0.0, strength=1.0)
    frames = render(image, mask, depth, neutral_config(strokes=(stroke,), duration_s=2.0))
    assert any(np.any(f[170:190, 310:330] != image[170:190, 310:330]) for f in frames)


def _max_local_shift(cls, mood) -> float:
    image = textured_image(720, 1280)
    mask = np.full((720, 1280), cls, np.uint8)
    frames = render(image, mask, None, neutral_config(mood=mood, duration_s=2.0, fps=12, seed=3))
    worst = 0.0
    for frame in frames:
        for y in range(40, 600, 180):
            for x in range(40, 1100, 300):
                p = (slice(y, y + 96), slice(x, x + 96))
                dx, dy = patch_shift(frames[0][p], frame[p])
                worst = max(worst, abs(dx), abs(dy))
    return worst


@pytest.mark.parametrize("cls,limit", [(VEGETACION, 2.0), (AGUA, 3.5)])
def test_local_effects_are_subtle_at_full_size(cls, limit):
    """Máximo desplazamiento local entre dos frames cualesquiera, a 1280x720, en píxeles."""
    assert _max_local_shift(cls, neutral_mood(speed=1.0)) <= limit


@pytest.mark.parametrize("cls", [VEGETACION, AGUA])
def test_epico_is_faster_but_not_more_deformed(cls):
    slow = _max_local_shift(cls, neutral_mood(speed=get_mood("calma").speed))
    fast = _max_local_shift(cls, neutral_mood(speed=get_mood("epico").speed))
    assert fast <= slow * 1.3 + SUBPIXEL_TOLERANCE
