"""Pruebas de animate_video: propagación por flujo, generador y validación."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from animation_engine import RenderConfig, animate_video, get_mood
from animation_engine.constants import CIELO, FONDO, PERSONA
from animation_engine.video import propagate_mask

from .conftest import neutral_config, ramp_depth, textured_image

H, W = 180, 320
BOX = 50
START_X, START_Y = 40, 60
STEP_X, STEP_Y = 3, 1


def moving_scene(index: int) -> tuple[np.ndarray, np.ndarray]:
    """Fondo con textura y un cuadro con otra textura que se desplaza. Devuelve (frame, verdad)."""
    frame = textured_image(H, W, seed=3)
    patch = textured_image(BOX, BOX, seed=9)
    patch = cv2.addWeighted(patch, 0.8, np.full_like(patch, 255), 0.2, 0)
    x, y = START_X + STEP_X * index, START_Y + STEP_Y * index
    frame[y : y + BOX, x : x + BOX] = patch
    truth = np.zeros((H, W), np.uint8)
    truth[y : y + BOX, x : x + BOX] = PERSONA
    return frame, truth


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / max(union, 1))


def test_propagate_mask_follows_moving_object():
    """Un solo paso de flujo contra la verdad conocida."""
    f0, m0 = moving_scene(0)
    f1, m1 = moving_scene(1)
    g0, g1 = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (f0, f1))
    propagated = propagate_mask(g0, g1, m0)
    assert iou(propagated == PERSONA, m1 == PERSONA) > iou(m0 == PERSONA, m1 == PERSONA)
    assert iou(propagated == PERSONA, m1 == PERSONA) > 0.88


def test_propagation_chain_tracks_synthetic_mask(monkeypatch):
    """Entre inferencias, la máscara sigue al objeto. Se mide contra la máscara verdadera."""
    import animation_engine.video as video

    seen: list[np.ndarray] = []
    original_scene = video.Scene

    class SpyScene(original_scene):
        def __init__(self, image, mask, *args, **kwargs):
            seen.append(mask.copy())
            super().__init__(image, mask, *args, **kwargs)

    monkeypatch.setattr(video, "Scene", SpyScene)

    n = 12
    frames = [moving_scene(i)[0] for i in range(n)]
    truths = [moving_scene(i)[1] for i in range(n)]
    calls: list[int] = []

    def segment(frame: np.ndarray) -> np.ndarray:
        calls.append(len(calls))
        return truths[(len(calls) - 1) * 5].copy()

    config = neutral_config(fps=12, duration_s=1.0)
    out = list(animate_video(frames, segment, 5, config))
    assert len(out) == n
    assert len(calls) == 3  # frames 0, 5 y 10

    for i in range(n):
        score = iou(seen[i] == PERSONA, truths[i] == PERSONA)
        steps = i % 5
        stale = iou(truths[i - steps] == PERSONA, truths[i] == PERSONA)
        if steps == 0:
            assert score == 1.0
        else:
            assert score > 0.75, f"frame {i}: IoU={score:.3f}"
            assert score > stale, f"frame {i}: propagada={score:.3f}, sin propagar={stale:.3f}"


@pytest.mark.parametrize("n,every,expected", [(10, 1, 10), (10, 5, 2), (11, 5, 3), (3, 5, 1)])
def test_segment_call_count(n, every, expected):
    frames = [textured_image(60, 80) for _ in range(n)]
    calls = []

    def segment(frame):
        calls.append(1)
        return np.zeros((60, 80), np.uint8)

    out = list(animate_video(frames, segment, every, neutral_config()))
    assert len(out) == n and len(calls) == expected


def test_frame_shape_and_dtype_and_determinism():
    frames = [moving_scene(i)[0] for i in range(6)]

    def segment(frame):
        mask = np.full((H, W), FONDO, np.uint8)
        mask[:50] = CIELO
        return mask

    config = RenderConfig(mood=get_mood("magico"), fps=12, duration_s=1.0, seed=4)
    a = list(animate_video(frames, segment, 2, config, depth=ramp_depth()))
    b = list(animate_video(frames, segment, 2, config, depth=ramp_depth()))
    assert all(f.shape == (H, W, 3) and f.dtype == np.uint8 for f in a)
    assert [f.tobytes() for f in a] == [f.tobytes() for f in b]
    other = RenderConfig(mood=get_mood("magico"), fps=12, duration_s=1.0, seed=5)
    c = list(animate_video(frames, segment, 2, other, depth=ramp_depth()))
    assert any(x.tobytes() != y.tobytes() for x, y in zip(a, c))


def test_generator_does_not_accumulate_or_consume_ahead():
    """El generador consume solo un frame de entrada por cada frame de salida."""
    consumed = 0

    def source():
        nonlocal consumed
        for i in range(30):
            consumed += 1
            yield moving_scene(i % 5)[0]

    gen = animate_video(source(), lambda f: np.zeros((H, W), np.uint8), 5, neutral_config())
    assert consumed == 0
    next(gen)
    assert consumed == 1
    next(gen)
    next(gen)
    assert consumed == 3
    gen.close()


def test_bad_segment_output_raises():
    frames = [textured_image(60, 80)]
    with pytest.raises(ValueError):
        list(animate_video(frames, lambda f: np.zeros((10, 10), np.uint8), 1, neutral_config()))


def test_frame_size_change_raises():
    frames = [textured_image(60, 80), textured_image(70, 80)]
    with pytest.raises(ValueError):
        list(animate_video(frames, lambda f: np.zeros(f.shape[:2], np.uint8), 1, neutral_config()))


def test_invalid_infer_every_n_raises():
    with pytest.raises(ValueError):
        animate_video([], lambda f: f, 0, neutral_config())


def test_empty_video_gives_no_frames():
    assert list(animate_video([], lambda f: f, 5, neutral_config())) == []


def test_wrong_depth_raises():
    frames = [textured_image(60, 80)]
    with pytest.raises(ValueError):
        list(
            animate_video(
                frames, lambda f: np.zeros((60, 80), np.uint8), 1, neutral_config(), depth=np.zeros((5, 5), np.float32)
            )
        )


def test_progress_with_unknown_length_is_monotonic():
    seen: list[float] = []

    def source():
        for i in range(6):
            yield moving_scene(i)[0]

    list(animate_video(source(), lambda f: np.zeros((H, W), np.uint8), 5, neutral_config(), progress=seen.append))
    assert all(a <= b for a, b in zip(seen, seen[1:]))
    assert seen[-1] == 1.0
