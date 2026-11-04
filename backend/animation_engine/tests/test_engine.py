"""Pruebas del contrato: forma, determinismo, loop, regiones, pincel, moods y pureza."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from animation_engine import BrushStroke, RenderConfig, animate_image, get_mood
from animation_engine.constants import AGUA, CIELO, FONDO, FUEGO_LUZ, PERSONA, VEGETACION
from animation_engine.effects.brush import rasterize_strokes
from animation_engine.effects.fire import glow_reach_px
from animation_engine.effects.sky import is_night
from animation_engine.effects.vegetation import is_forest

from .conftest import full_mask, neutral_config, neutral_mood, ramp_depth, textured_image

ENGINE_DIR = Path(__file__).resolve().parents[1]
MOODS = ("calma", "epico", "magico", "nostalgico")


def render(image, mask, depth, config, progress=None) -> list[np.ndarray]:
    return list(animate_image(image, mask, depth, config, progress))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(a, b)))


# ---------- Forma, conteo y validación ----------


@pytest.mark.parametrize("duration,fps,expected", [(1.0, 12, 12), (2.0, 24, 48), (2.5, 10, 25), (0.5, 7, 4)])
def test_exact_frame_count(image, mask, duration, fps, expected):
    config = neutral_config(duration_s=duration, fps=fps)
    assert len(render(image, mask, None, config)) == expected == round(duration * fps)


@pytest.mark.parametrize("mood", MOODS)
def test_frame_shape_and_dtype(image, mask, depth, mood):
    config = RenderConfig(mood=get_mood(mood), duration_s=0.5, fps=12, seed=1)
    for frame in animate_image(image, mask, depth, config):
        assert frame.shape == image.shape
        assert frame.dtype == np.uint8


@pytest.mark.parametrize("bad_shape", [(180, 321), (179, 320), (320, 180), (90, 160)])
def test_wrong_mask_size_raises_on_call(image, bad_shape):
    with pytest.raises(ValueError):
        animate_image(image, np.zeros(bad_shape, np.uint8), None, neutral_config())


def test_wrong_depth_size_raises(image, mask):
    with pytest.raises(ValueError):
        animate_image(image, mask, np.zeros((10, 10), np.float32), neutral_config())


def test_mask_with_invalid_class_raises(image, mask):
    bad = mask.copy()
    bad[0, 0] = 255  # IGNORE_INDEX no existe en inferencia
    with pytest.raises(ValueError):
        animate_image(image, bad, None, neutral_config())


def test_mask_wrong_dtype_raises(image, mask):
    with pytest.raises(ValueError):
        animate_image(image, mask.astype(np.int32), None, neutral_config())


def test_image_wrong_format_raises(mask):
    with pytest.raises(ValueError):
        animate_image(np.zeros((180, 320), np.uint8), mask, None, neutral_config())
    with pytest.raises(ValueError):
        animate_image(np.zeros((180, 320, 3), np.float32), mask, None, neutral_config())


@pytest.mark.parametrize(
    "overrides", [{"duration_s": 0.0}, {"fps": 0}, {"crossfade_s": -0.1}, {"brush_mode": "otro"}]
)
def test_invalid_config_raises(image, mask, overrides):
    with pytest.raises(ValueError):
        animate_image(image, mask, None, neutral_config(**overrides))


def test_unknown_mood_raises():
    with pytest.raises(ValueError):
        get_mood("triste")


def test_progress_is_monotonic_and_ends_at_one(image, mask):
    seen: list[float] = []
    render(image, mask, None, neutral_config(), seen.append)
    assert len(seen) == 12
    assert all(a < b for a, b in zip(seen, seen[1:]))
    assert 0.0 < seen[0] and seen[-1] == 1.0


def test_input_is_not_modified(image, mask, depth):
    img, msk, dep = image.copy(), mask.copy(), depth.copy()
    render(image, mask, depth, RenderConfig(mood=get_mood("epico"), duration_s=0.5, fps=12))
    assert np.array_equal(img, image) and np.array_equal(msk, mask) and np.array_equal(dep, depth)


def test_frames_are_independent_arrays(image):
    """El consumidor puede modificar un frame sin alterar los siguientes."""
    mask = np.zeros(image.shape[:2], np.uint8)
    gen = animate_image(image, mask, None, neutral_config())
    first = next(gen)
    first[:] = 0
    assert np.array_equal(next(gen), image)


# ---------- Determinismo ----------


@pytest.mark.parametrize("mood", MOODS)
def test_same_seed_is_byte_identical(image, mask, depth, mood):
    config = RenderConfig(mood=get_mood(mood), duration_s=0.75, fps=12, seed=123)
    a = render(image, mask, depth, config)
    b = render(image, mask, depth, config)
    assert [f.tobytes() for f in a] == [f.tobytes() for f in b]


def test_different_seeds_differ(image, mask, depth):
    a = render(image, mask, depth, RenderConfig(mood=get_mood("magico"), duration_s=0.5, fps=12, seed=1))
    b = render(image, mask, depth, RenderConfig(mood=get_mood("magico"), duration_s=0.5, fps=12, seed=2))
    assert mae(a[3], b[3]) > 0.1


def test_different_seeds_differ_without_particles(image, mask):
    a = render(image, mask, None, neutral_config(seed=1))
    b = render(image, mask, None, neutral_config(seed=2))
    assert mae(a[3], b[3]) > 0.1


def test_output_does_not_depend_on_thread_count(image, mask, depth):
    config = RenderConfig(mood=get_mood("magico"), duration_s=0.5, fps=12, seed=3)
    previous = cv2.getNumThreads()
    try:
        cv2.setNumThreads(1)
        a = render(image, mask, depth, config)
        cv2.setNumThreads(8)
        b = render(image, mask, depth, config)
        assert cv2.getNumThreads() == 8, "el motor debe restaurar el número de hilos"
    finally:
        cv2.setNumThreads(previous)
    assert [f.tobytes() for f in a] == [f.tobytes() for f in b]


def test_sparse_regions_match_dense_rendering():
    """La ruta de lista de píxeles da el mismo resultado que la de rectángulo completo."""
    from animation_engine import masks as masks_module

    img = textured_image()
    mask = np.zeros(img.shape[:2], np.uint8)
    for i, cls in enumerate((CIELO, AGUA, VEGETACION, FUEGO_LUZ)):
        mask[10 + 30 * i : 30 + 30 * i, 10:310:1] = cls
        mask[10 + 30 * i : 30 + 30 * i, 40:60] = FONDO
    mask[150:175, 280:315] = CIELO  # obliga a que el rectángulo del cielo sea poco denso
    sparse = render(img, mask, None, neutral_config())
    original = masks_module.Region._DENSE_FILL
    masks_module.Region._DENSE_FILL = 0.0
    try:
        dense = render(img, mask, None, neutral_config())
    finally:
        masks_module.Region._DENSE_FILL = original
    assert all(mae(a, b) < 0.02 for a, b in zip(sparse, dense))
    assert all(int(cv2.absdiff(a, b).max()) <= 2 for a, b in zip(sparse, dense))


def test_negative_seed_is_valid(image, mask):
    assert len(render(image, mask, None, neutral_config(seed=-5))) == 12


# ---------- Máscaras y depth degenerados ----------


def test_mask_without_animable_classes(image):
    """Solo fondo: sin color ni partículas, todos los frames son la imagen original."""
    mask = np.full(image.shape[:2], FONDO, np.uint8)
    frames = render(image, mask, None, neutral_config())
    assert len(frames) == 12
    assert all(np.array_equal(f, image) for f in frames)


@pytest.mark.parametrize("mood", MOODS)
def test_mask_without_animable_classes_with_mood(image, mood):
    mask = np.full(image.shape[:2], FONDO, np.uint8)
    frames = render(image, mask, None, RenderConfig(mood=get_mood(mood), duration_s=0.5, fps=12))
    assert len(frames) == 6


@pytest.mark.parametrize("cls", range(7))
def test_single_class_full_image(image, depth, cls):
    mask = np.full(image.shape[:2], cls, np.uint8)
    config = RenderConfig(mood=get_mood("magico"), duration_s=0.25, fps=12)
    assert len(render(image, mask, depth, config)) == 3


def test_depth_none_skips_parallax(image, mask):
    """Sin depth, una máscara solo de persona queda quieta."""
    subject = np.full(image.shape[:2], FONDO, np.uint8)
    subject[40:140, 100:220] = PERSONA
    config = neutral_config(mood=neutral_mood(parallax_amplitude=0.03))
    frames = render(image, subject, None, config)
    assert all(np.array_equal(f, image) for f in frames)
    assert len(render(image, mask, None, RenderConfig(mood=get_mood("epico"), duration_s=0.5, fps=12))) == 6


def test_small_image_works():
    img = textured_image(48, 64)
    config = RenderConfig(mood=get_mood("epico"), duration_s=0.25, fps=12)
    assert len(render(img, full_mask(48, 64), ramp_depth(48, 64), config)) == 3


# ---------- Loop ----------


def _loop_metrics(frames: list[np.ndarray]) -> tuple[float, float]:
    """(MAE entre el último y el primero, mediana de MAE entre frames contiguos)."""
    steps = [mae(a, b) for a, b in zip(frames, frames[1:])]
    return mae(frames[-1], frames[0]), float(np.median(steps))


@pytest.mark.parametrize("mood", MOODS)
def test_first_and_last_frame_close(image, mask, depth, mood):
    """El salto del empalme no supera 1.5 veces el cambio típico entre frames contiguos."""
    config = RenderConfig(mood=get_mood(mood), duration_s=2.0, fps=12, seed=5, crossfade_s=0.75)
    seam, typical = _loop_metrics(render(image, mask, depth, config))
    assert seam <= 1.5 * typical, f"empalme={seam:.3f}, típico={typical:.3f}"


def test_crossfade_removes_sky_jump(image):
    """Solo cielo: sin crossfade hay salto; con crossfade, el empalme es como un paso normal."""
    mask = np.full(image.shape[:2], CIELO, np.uint8)
    hard = render(image, mask, None, neutral_config(duration_s=2.0, crossfade_s=0.0))
    soft = render(image, mask, None, neutral_config(duration_s=2.0, crossfade_s=0.75))
    seam_hard, typical_hard = _loop_metrics(hard)
    seam_soft, typical_soft = _loop_metrics(soft)
    assert seam_hard > 3.0 * typical_hard
    assert seam_soft <= 1.5 * typical_soft
    assert seam_soft < seam_hard / 2.0


@pytest.mark.parametrize("cls", [AGUA, VEGETACION, FUEGO_LUZ])
def test_periodic_effects_close_without_crossfade(image, cls):
    """Agua, vegetación y fuego son periódicos: empalman aun sin crossfade."""
    mask = np.full(image.shape[:2], FONDO, np.uint8)
    mask[20:160, 40:280] = cls
    frames = render(image, mask, None, neutral_config(duration_s=2.0, crossfade_s=0.0))
    seam, typical = _loop_metrics(frames)
    assert seam <= 1.5 * typical


def test_crossfade_longer_than_duration_is_valid(image, mask):
    config = neutral_config(duration_s=1.0, crossfade_s=5.0)
    assert len(render(image, mask, None, config)) == 12


# ---------- Cada efecto toca solo su región ----------


@pytest.mark.parametrize("cls", [CIELO, AGUA, VEGETACION])
def test_pixels_outside_region_do_not_change(image, cls):
    # Se comprueba con una máscara donde solo existe esta clase animable.
    only = np.where(full_mask() == cls, cls, FONDO).astype(np.uint8)
    frames = render(image, only, None, neutral_config())
    outside = only != cls
    moved = False
    for frame in frames:
        assert np.array_equal(frame[outside], image[outside])
        moved = moved or bool(np.any(frame[only == cls] != image[only == cls]))
    assert moved, "el efecto debe cambiar algo dentro de su región"


def test_fire_only_affects_zone_and_glow_reach(image):
    mask = np.full(image.shape[:2], FONDO, np.uint8)
    mask[80:100, 10:30] = FUEGO_LUZ
    frames = render(image, mask, None, neutral_config())
    reach = int(np.ceil(glow_reach_px(max(image.shape[:2]) / 1280.0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * reach + 1, 2 * reach + 1))
    near = cv2.dilate((mask == FUEGO_LUZ).astype(np.uint8), kernel) > 0
    assert (~near).any()
    for frame in frames:
        assert np.array_equal(frame[~near], image[~near])


def test_fire_glow_extends_beyond_zone_and_flickers(image):
    mask = np.full(image.shape[:2], FONDO, np.uint8)
    mask[80:100, 150:170] = FUEGO_LUZ
    frames = render(image, mask, None, neutral_config(duration_s=2.0))
    ring = (cv2.dilate((mask > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & (mask == 0)
    gains = [float((f.astype(np.float32) - image.astype(np.float32))[ring].mean()) for f in frames]
    assert min(gains) > 0.5, "el brillo debe aclarar el entorno de la zona"
    assert max(gains) - min(gains) > 0.5, "el brillo debe parpadear"


def test_sky_edge_band_stays_close_to_original(image):
    """Cielo sin deformar bordes: el primer píxel de la región coincide con el original."""
    mask = np.full(image.shape[:2], FONDO, np.uint8)
    mask[:90, :] = CIELO
    frames = render(image, mask, None, neutral_config(duration_s=2.0))
    for frame in frames:
        assert np.array_equal(frame[89, :], image[89, :])
        assert np.array_equal(frame[90:], image[90:])


def test_sky_does_not_pull_neighbor_colors():
    """Un bloque rojo vecino no debe entrar en el cielo azul al desplazar la textura."""
    img = np.zeros((180, 320, 3), np.uint8)
    img[:] = (200, 120, 60)
    img[60:, :] = (0, 0, 255)
    img[:60, 200:260] = (0, 0, 255)
    mask = np.full((180, 320), FONDO, np.uint8)
    mask[:60, :] = CIELO
    mask[:60, 200:260] = FONDO
    for frame in render(img, mask, None, neutral_config(duration_s=2.0)):
        sky = frame[mask == CIELO].astype(np.int32)
        assert (sky[:, 2] - sky[:, 0]).max() < 0, "ningún píxel de cielo debe volverse rojo"


# ---------- Reglas de noche y bosque ----------


def test_night_rule_threshold(monkeypatch):
    mask = np.full((60, 80), CIELO, np.uint8)
    dark = np.full((60, 80, 3), 40, np.uint8)
    light = np.full((60, 80, 3), 120, np.uint8)
    assert is_night(dark, mask)
    assert not is_night(light, mask)
    monkeypatch.setenv("NIGHT_LUMA_THRESHOLD", "30")
    assert not is_night(dark, mask)


def test_night_rule_needs_sky():
    assert not is_night(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.uint8))


def test_stars_appear_and_twinkle_only_at_night():
    mask = np.full((180, 320), CIELO, np.uint8)
    dark = np.full((180, 320, 3), 20, np.uint8)
    frames = render(dark, mask, None, neutral_config(duration_s=2.0))
    stack = np.stack([f[..., 0] for f in frames]).astype(np.float32)
    assert stack.max() > 100, "de noche debe haber estrellas brillantes"
    bright = stack.mean(axis=0) > stack.mean() + 40
    assert bright.sum() > 0
    assert stack[:, bright].std(axis=0).mean() > 5.0, "las estrellas deben titilar"

    day = np.full((180, 320, 3), 180, np.uint8)
    day_frames = render(day, mask, None, neutral_config(duration_s=2.0, seed=7))
    day_diff = cv2.absdiff(day_frames[5], day).max()
    assert day_diff < 80


def test_forest_rule_threshold(monkeypatch):
    mask = np.zeros((100, 100), np.uint8)
    mask[:25, :] = VEGETACION
    assert is_forest(mask)
    mask[24, :] = FONDO
    assert not is_forest(mask)
    monkeypatch.setenv("FOREST_MIN_PCT", "10")
    assert is_forest(mask)


def test_leaves_fall_only_in_forest(image):
    """Con partículas de densidad cero, solo las hojas tocan los píxeles de fondo."""
    mood = neutral_mood(particle_kind="polvo", particle_density=0.0)
    forest = np.full(image.shape[:2], FONDO, np.uint8)
    forest[:50, :] = VEGETACION
    small = np.full(image.shape[:2], FONDO, np.uint8)
    small[:20, :] = VEGETACION
    config = neutral_config(mood=mood, duration_s=4.0)
    leaves = render(image, forest, None, config)
    assert any(np.any(f[50:] != image[50:]) for f in leaves), "hay hojas sobre el fondo inferior"
    none = render(image, small, None, config)
    assert all(np.array_equal(f[20:], image[20:]) for f in none)


# ---------- Pincel ----------


def _stroke(direction=0.0, strength=1.0) -> BrushStroke:
    return BrushStroke(points=((0.3, 0.5), (0.7, 0.5)), radius=0.06, direction_deg=direction, strength=strength)


def test_rasterize_direction_convention():
    painted, vx, vy = rasterize_strokes([_stroke(90.0, 0.5)], (100, 200))
    on = painted > 0
    assert on.any() and not painted[0, 0]
    assert np.allclose(vx[on], 0.0, atol=1e-5)
    assert np.allclose(vy[on], 0.5)  # 90 grados = hacia abajo


def test_brush_only_affects_painted_area(image, mask, depth):
    config = neutral_config(strokes=(_stroke(),), brush_mode="only", duration_s=2.0)
    frames = render(image, mask, depth, config)
    painted, _, _ = rasterize_strokes(config.strokes, image.shape[:2])
    outside = painted == 0
    changed = False
    for frame in frames:
        assert np.array_equal(frame[outside], image[outside])
        changed = changed or bool(np.any(frame[~outside] != image[~outside]))
    assert changed, "lo pintado debe animarse"


@pytest.mark.parametrize("mood", MOODS)
def test_brush_only_outside_is_static_with_mood(image, mask, depth, mood):
    """Con mood real, lo no pintado lleva el color del mood pero no cambia en el tiempo."""
    config = RenderConfig(mood=get_mood(mood), strokes=(_stroke(),), brush_mode="only", duration_s=1.0, fps=12)
    frames = render(image, mask, depth, config)
    painted, _, _ = rasterize_strokes(config.strokes, image.shape[:2])
    outside = painted == 0
    for frame in frames[1:]:
        assert np.array_equal(frame[outside], frames[0][outside])


def test_brush_only_without_strokes_is_static(image, mask):
    frames = render(image, mask, None, neutral_config(brush_mode="only"))
    assert all(np.array_equal(f, image) for f in frames)


def test_brush_add_keeps_class_effects(image):
    mask = np.full(image.shape[:2], FONDO, np.uint8)
    mask[:60, :] = CIELO
    stroke = BrushStroke(points=((0.5, 0.8),), radius=0.05, direction_deg=45.0, strength=1.0)
    config = neutral_config(strokes=(stroke,), brush_mode="add")
    frames = render(image, mask, None, config)
    assert any(np.any(f[:50] != image[:50]) for f in frames), "el cielo sigue animado"
    painted, _, _ = rasterize_strokes((stroke,), image.shape[:2])
    assert any(np.any(f[painted > 0] != image[painted > 0]) for f in frames), "lo pintado se anima"
    untouched = (mask == FONDO) & (painted == 0)
    assert all(np.array_equal(f[untouched], image[untouched]) for f in frames)


def test_brush_direction_moves_content():
    """Un trazo hacia la derecha desplaza un patrón hacia la derecha."""
    h, w = 120, 240
    img = np.tile(((np.sin(np.arange(w) * 2 * np.pi / 40) + 1) * 110 + 20).astype(np.uint8)[None, :, None], (h, 1, 3))
    mask = np.zeros((h, w), np.uint8)
    stroke = BrushStroke(points=((0.1, 0.5), (0.9, 0.5)), radius=0.15, direction_deg=0.0, strength=0.5)
    config = neutral_config(strokes=(stroke,), brush_mode="only", duration_s=4.0, fps=24, mood=neutral_mood(speed=0.5))
    frames = render(img, mask, None, config)
    row = h // 2

    def shift_between(a, b):
        fa = np.fft.fft(a[row, 40:200, 0].astype(np.float64) - a[row, 40:200, 0].mean())
        fb = np.fft.fft(b[row, 40:200, 0].astype(np.float64) - b[row, 40:200, 0].mean())
        return np.angle(np.sum(fb * np.conj(fa)))

    # Pequeños pasos entre frames: la fase del patrón avanza hacia +x (fase negativa en FFT).
    steps = [shift_between(frames[i], frames[i + 1]) for i in range(2, 10)]
    assert np.median(steps) < 0


def test_brush_is_periodic(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    config = neutral_config(strokes=(_stroke(),), brush_mode="only", duration_s=2.0, crossfade_s=0.0)
    seam, typical = _loop_metrics(render(image, mask, None, config))
    assert seam <= 1.5 * typical


# ---------- Moods ----------


def _moods_render(image, mask, depth):
    return {
        name: render(image, mask, depth, RenderConfig(mood=get_mood(name), duration_s=2.0, fps=12, seed=11))
        for name in MOODS
    }


def test_moods_differ_measurably(image, mask, depth):
    """Paleta, velocidad y partículas: los cuatro moods dan valores medibles distintos."""
    results = _moods_render(image, mask, depth)
    for i, a in enumerate(MOODS):
        for b in MOODS[i + 1 :]:
            assert mae(results[a][6], results[b][6]) > 3.0, f"{a} y {b} se parecen demasiado"

    def saturation(frames):
        return float(cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)[..., 1].mean())

    def warmth(frames):
        return float(frames[0][..., 2].mean() - frames[0][..., 0].mean())

    def motion(frames):
        return float(np.mean([mae(x, y) for x, y in zip(frames, frames[1:])]))

    assert saturation(results["magico"]) > saturation(results["calma"]) > saturation(results["nostalgico"])
    assert warmth(results["nostalgico"]) > warmth(results["calma"]) > warmth(results["epico"])
    assert motion(results["epico"]) > motion(results["calma"]) * 1.2


def test_mood_speed_scales_motion(image):
    mask = np.full(image.shape[:2], AGUA, np.uint8)
    slow = render(image, mask, None, neutral_config(mood=neutral_mood(speed=0.6), duration_s=2.0))
    fast = render(image, mask, None, neutral_config(mood=neutral_mood(speed=1.6), duration_s=2.0))

    def motion(frames):
        return np.mean([mae(x, y) for x, y in zip(frames, frames[1:])])

    assert motion(fast) > motion(slow) * 1.2


@pytest.mark.parametrize("kind", ["polen", "brasas", "chispas", "polvo"])
def test_particles_render_for_each_kind(image, kind):
    """Cada tipo de partícula deja marcas que cambian entre frames sobre una imagen estática."""
    mask = np.zeros(image.shape[:2], np.uint8)
    mood = neutral_mood(particle_kind=kind, particle_density=400.0)
    frames = render(image, mask, None, neutral_config(mood=mood))
    assert mae(frames[0], image) > 0.0
    assert mae(frames[0], frames[4]) > 0.0


def test_particle_density_scales_count(image):
    mask = np.zeros(image.shape[:2], np.uint8)

    def touched(density):
        mood = neutral_mood(particle_kind="polvo", particle_density=density)
        frame = render(image, mask, None, neutral_config(mood=mood))[4]
        return int(np.count_nonzero(np.any(frame != image, axis=2)))

    assert touched(800.0) > touched(100.0) > 0


def test_moods_table_matches_valid_values():
    for name in MOODS:
        mood = get_mood(name)
        assert mood.name == name
        assert mood.speed > 0 and mood.saturation > 0 and mood.contrast > 0
        assert -1.0 <= mood.warmth <= 1.0
        assert 0.0 <= mood.vignette <= 1.0
        assert mood.particle_kind in ("none", "polen", "brasas", "chispas", "polvo")
        assert 0.0 <= mood.parallax_amplitude <= 0.1


# ---------- Generador y pureza ----------


def test_animate_image_is_lazy(image, mask):
    produced = 0
    gen = animate_image(image, mask, None, neutral_config())
    assert iter(gen) is gen
    for _ in gen:
        produced += 1
        break
    assert produced == 1


FORBIDDEN = {"fastapi", "onnxruntime", "sqlite3"}


def _source_files() -> list[Path]:
    return [
        p for p in ENGINE_DIR.rglob("*.py")
        if ".venv" not in p.parts and "tests" not in p.parts
    ]


def test_source_has_no_forbidden_imports():
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                assert root not in FORBIDDEN, f"{path}: importa {name}"
                assert root in {"__future__", "contextlib", "cv2", "numpy", "dataclasses", "typing", "math", "os", "zlib"}, (
                    f"{path}: dependencia no permitida {name}"
                )


def test_import_does_not_load_forbidden_modules():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import animation_engine; "
        "bad = [m for m in ('fastapi', 'onnxruntime', 'sqlite3') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(ENGINE_DIR.parent)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
