"""Renderiza muestras del motor de animación en samples_output/ (ignorado por git).

Usa fotos reales con licencia libre de Wikimedia Commons y máscaras sintéticas
(polígonos y umbrales de color hechos para cada foto; no vienen del modelo).

    python scripts/render_samples.py --download        # baja las fotos que falten
    python scripts/render_samples.py                   # todas las muestras
    python scripts/render_samples.py --only lake_calma horse_epico
    python scripts/render_samples.py --list
    python scripts/render_samples.py --bench           # rendimiento por mood
    python scripts/render_samples.py --masks           # solo superpone las máscaras

Por cada muestra escribe keyframes PNG, una hoja de contacto con el mapa de
movimiento, la métrica del empalme y, si hay FFmpeg, un MP4 H.264.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from animation_engine import BrushStroke, RenderConfig, animate_image, animate_video, get_mood  # noqa: E402
from animation_engine.constants import AGUA, ANIMAL, CIELO, FONDO, FUEGO_LUZ, PERSONA, VEGETACION  # noqa: E402
from animation_engine.types import MoodParams  # noqa: E402

OUT = ROOT / "samples_output"
PHOTOS = OUT / "photos"
WIDTH, HEIGHT = 1280, 720
COMMONS = "https://upload.wikimedia.org/wikipedia/commons/thumb"

# Fuente, autor y licencia de cada foto (Wikimedia Commons). Las fotos no se versionan.
PHOTO_SOURCES = {
    "lake": (
        f"{COMMONS}/0/01/Lake_Mary_Mammoth_September_2016.jpg/1920px-Lake_Mary_Mammoth_September_2016.jpg",
        "King of Hearts", "CC BY-SA 4.0",
    ),
    "forest": (
        f"{COMMONS}/1/12/Quinault_Rain_Forest_June_2018_011.jpg/1920px-Quinault_Rain_Forest_June_2018_011.jpg",
        "King of Hearts", "CC BY-SA 4.0",
    ),
    "horse": (
        f"{COMMONS}/b/be/Domestic_horse_at_Suryachaur_and_the_mountains_in_the_back1.jpg/"
        "1920px-Domestic_horse_at_Suryachaur_and_the_mountains_in_the_back1.jpg",
        "Shadow Ayush", "CC BY-SA 4.0",
    ),
    "fisherman": (
        f"{COMMONS}/8/8b/Inle_Lake%2C_Fisherman_on_the_boat%2C_Myanmar.jpg/"
        "1920px-Inle_Lake%2C_Fisherman_on_the_boat%2C_Myanmar.jpg",
        "Vyacheslav Argenberg", "CC BY 4.0",
    ),
    "night_camp": (
        f"{COMMONS}/b/be/Campsite_38_in_Bowron_Lake_Provincial_Park_at_night_%28DSCF3026%29.jpg/"
        "1920px-Campsite_38_in_Bowron_Lake_Provincial_Park_at_night_%28DSCF3026%29.jpg",
        "Trougnouf", "CC BY 4.0",
    ),
}


def download_photos() -> None:
    PHOTOS.mkdir(parents=True, exist_ok=True)
    for name, (url, author, license_name) in PHOTO_SOURCES.items():
        path = PHOTOS / f"{name}.jpg"
        if path.exists():
            continue
        print(f"Descargando {name} ({author}, {license_name})")
        request = urllib.request.Request(url, headers={"User-Agent": "KINETIA-student-project/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())


def load_photo(name: str) -> np.ndarray:
    """Foto recortada al centro a 16:9 y reducida a 1280x720."""
    path = PHOTOS / f"{name}.jpg"
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Falta {path}. Ejecuta primero con --download")
    h, w = image.shape[:2]
    target_h = int(round(w * HEIGHT / WIDTH))
    if target_h <= h:
        y0 = (h - target_h) // 2
        image = image[y0 : y0 + target_h]
    return cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


# ---------- Máscaras sintéticas por foto ----------


def _channels(image: np.ndarray):
    b, g, r = [c.astype(np.float32) for c in cv2.split(image)]
    return b, g, r, 0.114 * b + 0.587 * g + 0.299 * r


def _clean(binary: np.ndarray, size: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    out = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel) > 0


def _connected_to_top(binary: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=4)
    keep = np.unique(labels[0][labels[0] > 0])
    return np.isin(labels, keep)


def _polygon(points_norm, shape=(HEIGHT, WIDTH)) -> np.ndarray:
    h, w = shape
    pts = np.array([[int(x * w), int(y * h)] for x, y in points_norm], np.int32)
    out = np.zeros(shape, np.uint8)
    cv2.fillPoly(out, [pts], 1)
    return out > 0


def _waterline(points_norm, shape=(HEIGHT, WIDTH)) -> np.ndarray:
    """Región bajo una polilínea (x, y) normalizada."""
    h, w = shape
    xs = np.arange(w) / max(w - 1, 1)
    line = np.interp(xs, [p[0] for p in points_norm], [p[1] for p in points_norm]) * h
    return np.arange(h)[:, None] >= line[None, :]


def mask_lake(image: np.ndarray) -> np.ndarray:
    b, g, r, luma = _channels(image)
    mask = np.full(image.shape[:2], FONDO, np.uint8)
    water = _waterline([(0.0, 0.565), (0.12, 0.525), (0.3, 0.505), (0.6, 0.505), (0.82, 0.50), (1.0, 0.515)])
    above = ~water
    sky = _connected_to_top(above & (b > r - 8) & (luma > 95))
    mask[above] = VEGETACION
    hill = above & ~sky & (r > b + 38) & (r > g + 14)
    mask[_clean(hill)] = FONDO
    mask[sky] = CIELO
    mask[water] = AGUA
    return mask


def mask_forest(image: np.ndarray) -> np.ndarray:
    b, g, r, luma = _channels(image)
    mask = np.full(image.shape[:2], FONDO, np.uint8)
    green = _clean((g > r * 1.0) & (g > b * 1.15) & (luma > 28), 5)
    mask[green] = VEGETACION
    falls = _polygon(
        [(0.46, 0.0), (0.58, 0.0), (0.55, 0.42), (0.47, 0.66), (0.43, 0.62), (0.47, 0.32)]
    ) | _polygon([(0.12, 0.72), (0.25, 0.72), (0.25, 1.0), (0.1, 1.0)]) | _polygon(
        [(0.36, 0.75), (0.52, 0.72), (0.8, 0.8), (0.8, 0.98), (0.4, 0.98)]
    )
    white = (luma > 150) & (np.abs(r - b) < 30)
    water = cv2.dilate((falls & white).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    mask[_clean(water & falls, 3)] = AGUA
    return mask


def mask_horse(image: np.ndarray) -> np.ndarray:
    mask = np.full(image.shape[:2], FONDO, np.uint8)
    h, w = image.shape[:2]
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    rect = _polygon([(0.05, 0.12), (0.73, 0.12), (0.73, 0.90), (0.05, 0.90)])
    gc[~rect] = cv2.GC_BGD
    body = _polygon(
        [(0.26, 0.36), (0.52, 0.33), (0.60, 0.28), (0.66, 0.34), (0.66, 0.50), (0.58, 0.55), (0.28, 0.58)]
    )
    gc[rect] = cv2.GC_PR_BGD
    gc[body] = cv2.GC_PR_FGD
    b, g, r, luma = _channels(image)
    white_horse = (luma > 165) & (np.abs(r - b) < 45) & _polygon(
        [(0.05, 0.25), (0.73, 0.2), (0.73, 0.9), (0.05, 0.9)]
    ) & _polygon([(0.05, 0.35), (0.3, 0.30), (0.73, 0.2), (0.73, 0.9), (0.2, 0.9)])
    gc[white_horse & body] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.setRNGSeed(12345)
    cv2.grabCut(image, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    subject = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    # GrabCut confunde la montaña bajo el cuello con el caballo: el pelaje no es azulado.
    subject &= ~(b > r + 6)
    # La parte baja de la cola es oscura y GrabCut la pierde; se recupera por color.
    tail = _polygon([(0.05, 0.28), (0.24, 0.28), (0.24, 0.72), (0.05, 0.72)]) & (r > b + 8) & (luma < 165)
    subject |= tail
    subject = _clean(subject, 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(subject.astype(np.uint8), connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        subject = labels == largest
    # Rellena agujeros internos (zonas en sombra azulada del pelaje).
    contours, _ = cv2.findContours(subject.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(subject.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    subject = filled > 0
    sky = ~_waterline([(0.0, 0.50), (0.35, 0.53), (0.7, 0.525), (1.0, 0.50)])
    mask[sky] = CIELO
    mask[subject] = ANIMAL
    return mask


def mask_fisherman(image: np.ndarray) -> np.ndarray:
    b, g, r, luma = _channels(image)
    mask = np.full(image.shape[:2], FONDO, np.uint8)
    water = _waterline([(0.0, 0.31), (1.0, 0.31)])
    sky = ~_waterline([(0.0, 0.285), (1.0, 0.285)])
    mask[sky] = CIELO
    mask[water] = AGUA
    zone = _polygon(
        [(0.20, 0.42), (0.51, 0.20), (0.85, 0.08), (0.90, 0.08), (0.90, 0.56), (0.20, 0.50)]
    )
    dark = zone & (luma < 88)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    subject = np.zeros_like(dark)
    if count > 1:
        subject = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask[subject] = PERSONA
    return mask


def mask_night_camp(image: np.ndarray) -> np.ndarray:
    b, g, r, luma = _channels(image)
    mask = np.full(image.shape[:2], FONDO, np.uint8)
    leaves = _clean((luma > 22) & (g > b * 1.2) & _polygon([(0, 0), (1, 0), (1, 0.62), (0, 0.62)]), 3)
    mask[leaves] = VEGETACION
    pit = _polygon([(0.54, 0.55), (0.76, 0.55), (0.76, 0.92), (0.54, 0.92)]) & (luma > 95) & (r > b + 10)
    lights = _polygon([(0.10, 0.28), (0.92, 0.28), (0.92, 0.82), (0.10, 0.82)]) & (luma > 150)
    fire = cv2.dilate((pit | lights).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    mask[fire] = FUEGO_LUZ
    return mask


def make_night(image: np.ndarray) -> np.ndarray:
    """Noche simulada a partir de una foto de día: oscurece y enfría."""
    out = image.astype(np.float32)
    luma = out.mean(axis=2, keepdims=True)
    out = 0.35 * out + 0.65 * luma
    out = out * np.array([0.30, 0.20, 0.15], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def synthetic_depth(mask: np.ndarray, far_above: float = 0.0) -> np.ndarray:
    """Profundidad sintética: abajo está más cerca; cielo lejos; sujeto al frente."""
    h, w = mask.shape
    rows = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    depth = np.tile(0.15 + 0.75 * rows**1.4, (1, w))
    depth[mask == CIELO] = far_above
    subject = (mask == PERSONA) | (mask == ANIMAL)
    if subject.any():
        depth[subject] = max(0.85, float(np.percentile(depth[subject], 80)))
    return cv2.GaussianBlur(depth, (0, 0), 3.0).astype(np.float32)


# ---------- Catálogo de muestras ----------


@dataclass(frozen=True)
class Sample:
    name: str
    photo: str
    mask_fn: Callable[[np.ndarray], np.ndarray]
    description: str
    mood: str = "calma"
    use_depth: bool = True
    night: bool = False
    neutral: bool = False  # sin color, sin partículas y sin parallax: efecto aislado
    strokes: tuple[BrushStroke, ...] = ()
    brush_mode: str = "add"
    seed: int = 7
    mood_overrides: dict = field(default_factory=dict)


WATERFALL = tuple(
    BrushStroke(points=pts, radius=r, direction_deg=d, strength=s)
    for pts, r, d, s in (
        (((0.555, 0.03), (0.525, 0.20), (0.495, 0.34)), 0.016, 100.0, 1.0),
        (((0.49, 0.42), (0.495, 0.62)), 0.016, 90.0, 1.0),
    )
)
LAKE_BRUSH = (
    BrushStroke(points=((0.05, 0.8), (0.95, 0.8)), radius=0.05, direction_deg=0.0, strength=0.8),
)
HORSE_TAIL = (
    BrushStroke(points=((0.14, 0.42), (0.20, 0.56)), radius=0.035, direction_deg=20.0, strength=0.5),
)


def _catalog() -> list[Sample]:
    samples = [
        Sample("effect_cielo_agua_vegetacion", "lake", mask_lake,
               "Cielo, agua y vegetación aislados (sin color, partículas ni parallax)", neutral=True, use_depth=False),
        Sample("effect_vegetacion_agua_hojas", "forest", mask_forest,
               "Vegetación > FOREST_MIN_PCT, con hojas que caen", mood="calma", use_depth=False,
               mood_overrides={"particle_density": 0.0, "parallax_amplitude": 0.0, "vignette": 0.0}),
        Sample("effect_fuego_luz", "night_camp", mask_night_camp,
               "Parpadeo y brillo en fogata y luces", neutral=True, use_depth=False),
        Sample("effect_parallax_animal", "horse", mask_horse,
               "Parallax 2.5D con animal", neutral=False, mood="calma",
               mood_overrides={"parallax_amplitude": 0.022, "particle_density": 0.0, "speed": 1.0}),
        Sample("effect_parallax_persona", "fisherman", mask_fisherman,
               "Parallax 2.5D con persona y agua", mood="calma",
               mood_overrides={"parallax_amplitude": 0.022, "particle_density": 0.0, "speed": 1.0}),
        Sample("effect_noche_estrellas", "lake", mask_lake,
               "Regla de noche: estrellas (foto de día oscurecida)", neutral=True, use_depth=False, night=True),
        Sample("effect_pincel_only", "forest", mask_forest,
               "Pincel en modo only: solo se mueve la cascada pintada", neutral=True, use_depth=False,
               strokes=WATERFALL, brush_mode="only"),
        Sample("effect_pincel_add", "lake", mask_lake,
               "Pincel en modo add sobre el agua", neutral=True, use_depth=False, strokes=LAKE_BRUSH),
    ]
    for mood in ("calma", "epico", "magico", "nostalgico"):
        samples.append(Sample(f"mood_{mood}_lake", "lake", mask_lake, f"Mood {mood} en lago", mood=mood))
        samples.append(Sample(f"mood_{mood}_horse", "horse", mask_horse, f"Mood {mood} en caballo", mood=mood))
    samples.append(Sample("mood_epico_fogata", "night_camp", mask_night_camp, "Brasas desde la fogata", mood="epico"))
    samples.append(Sample("mood_magico_bosque", "forest", mask_forest, "Chispas en bosque", mood="magico"))
    return samples


def neutral_mood(name: str = "calma") -> MoodParams:
    return MoodParams(
        name=name, speed=1.0, saturation=1.0, warmth=0.0, contrast=1.0,
        particle_kind="none", particle_density=0.0, parallax_amplitude=0.0, vignette=0.0,
    )


def build_inputs(sample: Sample):
    image = load_photo(sample.photo)
    mask = sample.mask_fn(image)
    if sample.night:
        image = make_night(image)
    depth = synthetic_depth(mask) if sample.use_depth else None
    mood = neutral_mood() if sample.neutral else get_mood(sample.mood)
    if sample.mood_overrides:
        mood = replace(mood, **sample.mood_overrides)
    config = RenderConfig(
        mood=mood, duration_s=4.0, fps=24, seed=sample.seed,
        strokes=sample.strokes, brush_mode=sample.brush_mode,  # type: ignore[arg-type]
    )
    return image, mask, depth, config


# ---------- Salidas ----------

_MASK_COLORS = {
    FONDO: (75, 63, 58), CIELO: (247, 195, 79), AGUA: (255, 121, 41), VEGETACION: (106, 187, 102),
    FUEGO_LUZ: (67, 112, 255), PERSONA: (122, 64, 236), ANIMAL: (40, 202, 255),
}


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    colors = np.zeros_like(image)
    for cls, color in _MASK_COLORS.items():
        colors[mask == cls] = color
    return cv2.addWeighted(image, 0.5, colors, 0.5, 0.0)


def find_ffmpeg() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    if local:
        for folder in (("kinetia-tools", "bin"), ("Microsoft", "WinGet", "Links")):
            candidate = Path(local).joinpath(*folder) / "ffmpeg.exe"
            if candidate.exists():
                return str(candidate)
    return None


def _label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_sample(sample: Sample, save_all: bool) -> dict:
    image, mask, depth, config = build_inputs(sample)
    folder = OUT / sample.name
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()

    n = int(round(config.duration_s * config.fps))
    keys = {0, n // 4, n // 2, 3 * n // 4, n - 1}
    ffmpeg = find_ffmpeg()
    encoder = None
    if ffmpeg:
        encoder = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{WIDTH}x{HEIGHT}", "-r", str(config.fps), "-i", "-", "-an",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
             str(OUT / f"{sample.name}.mp4")],
            stdin=subprocess.PIPE,
        )

    first = prev = last = None
    steps: list[float] = []
    energy = np.zeros(image.shape[:2], np.float32)
    keyframes: list[tuple[int, np.ndarray]] = []
    render_seconds = 0.0
    tick = time.perf_counter()
    for i, frame in enumerate(animate_image(image, mask, depth, config)):
        render_seconds += time.perf_counter() - tick
        if first is None:
            first = frame.copy()
        if prev is not None:
            diff = cv2.absdiff(prev, frame)
            steps.append(float(diff.mean()))
            energy += diff.max(axis=2).astype(np.float32)
        prev = frame.copy()
        last = prev
        if i in keys:
            keyframes.append((i, frame.copy()))
        if save_all or i in keys:
            cv2.imwrite(str(folder / f"frame_{i:03d}.png"), frame)
        if encoder is not None:
            encoder.stdin.write(frame.tobytes())
        tick = time.perf_counter()
    if encoder is not None:
        encoder.stdin.close()
        encoder.wait()

    seam = float(cv2.absdiff(first, last).mean())
    heat = cv2.applyColorMap(
        np.clip(energy / max(float(np.percentile(energy, 99.5)), 1.0) * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )
    tiles = [cv2.resize(_label(image, "original"), (640, 360), interpolation=cv2.INTER_AREA)]
    tiles.append(cv2.resize(_label(overlay_mask(image, mask), "mascara"), (640, 360), interpolation=cv2.INTER_AREA))
    tiles.append(cv2.resize(_label(heat, "movimiento acumulado"), (640, 360), interpolation=cv2.INTER_AREA))
    for i, frame in keyframes[:3]:
        tiles.append(cv2.resize(_label(frame, f"frame {i}"), (640, 360), interpolation=cv2.INTER_AREA))
    sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
    cv2.imwrite(str(OUT / f"{sample.name}_sheet.png"), sheet)
    typical = float(np.median(steps)) if steps else 0.0
    result = {
        "sample": sample.name,
        "description": sample.description,
        "render_s": round(render_seconds, 2),
        "seam_mae": round(seam, 3),
        "typical_step_mae": round(typical, 3),
        "mp4": bool(ffmpeg),
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def _bench_scenes() -> dict:
    """Escena natural con todos los efectos, y peor caso con todas las regiones de borde a borde."""
    lake = load_photo("lake")
    natural = mask_lake(lake)
    # Se añaden fuego y persona para activar todos los efectos a la vez.
    natural[300:340, 1100:1150] = FUEGO_LUZ
    cv2.ellipse(natural, (620, 460), (60, 180), 0, 0, 360, int(PERSONA), -1)

    # Como la salida de un modelo sin entrenar: bloques al azar de todas las clases.
    rng = np.random.default_rng(1)
    noisy = cv2.resize(
        rng.integers(0, 7, (40, 72), dtype=np.uint8), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST
    )
    return {
        "natural": (lake, natural, synthetic_depth(natural)),
        "peor_caso": (lake, noisy, rng.random((HEIGHT, WIDTH), dtype=np.float32)),
    }


def _measure(image, mask, depth, config, threads: Optional[int]) -> dict:
    default_threads = cv2.getNumThreads()
    if threads:
        cv2.setNumThreads(threads)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    frames = 0
    first_cpu = 0.0
    for _ in animate_image(image, mask, depth, config):
        if frames == 0:
            first_cpu = time.process_time() - cpu0
        frames += 1
    result = {
        "frames": frames,
        "cpu_s": round(time.process_time() - cpu0, 2),
        "wall_s": round(time.perf_counter() - wall0, 2),
        "setup_cpu_s": round(first_cpu, 2),
    }
    cv2.setNumThreads(default_threads)
    return result


def bench(threads: Optional[int]) -> None:
    """Mide 1280x720, 4 s a 24 fps, sin codificación, por escena y mood."""
    print(f"Hilos de OpenCV: {threads or cv2.getNumThreads()}")
    results: dict = {}
    for scene, (image, mask, depth) in _bench_scenes().items():
        for name in ("calma", "epico", "magico", "nostalgico"):
            config = RenderConfig(mood=get_mood(name), duration_s=4.0, fps=24, seed=7)
            results[f"{scene}/{name}"] = _measure(image, mask, depth, config, threads)
            print(f"{scene}/{name}", json.dumps(results[f"{scene}/{name}"]), flush=True)
    (OUT / f"bench_{threads or 'default'}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def bench_video() -> None:
    """Mide animate_video con 640x360, 48 frames, inferencia cada 5 y máscara dispersa, sin codificación."""
    photo = cv2.resize(load_photo("horse"), (720, 405), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(1)
    natural = cv2.resize(mask_horse(load_photo("horse")), (640, 360), interpolation=cv2.INTER_NEAREST)
    noisy = cv2.resize(rng.integers(0, 7, (20, 36), dtype=np.uint8), (640, 360), interpolation=cv2.INTER_NEAREST)
    depth = rng.random((360, 640), dtype=np.float32)

    def source():
        for i in range(48):
            x = int(40 * (i / 48))  # paneo lento, para que el flujo tenga trabajo real
            yield photo[20:380, x : x + 640].copy()

    for scene, mask in (("natural", natural), ("peor_caso", noisy)):
        for name in ("calma", "epico", "magico", "nostalgico"):
            config = RenderConfig(mood=get_mood(name), duration_s=4.0, fps=24, seed=7)
            cpu0, wall0 = time.process_time(), time.perf_counter()
            frames = sum(1 for _ in animate_video(source(), lambda f: mask, 5, config, depth=depth))
            print(
                f"video {scene}/{name}: {frames} frames, cpu {time.process_time() - cpu0:.2f} s, "
                f"wall {time.perf_counter() - wall0:.2f} s",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="baja las fotos que falten y termina")
    parser.add_argument("--only", nargs="*", help="nombres de muestras")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--threads", type=int, help="hilos de OpenCV en --bench")
    parser.add_argument("--bench-video", action="store_true")
    parser.add_argument("--masks", action="store_true", help="solo escribe superposiciones de máscaras")
    parser.add_argument("--all-frames", action="store_true", help="guarda todos los frames como PNG")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.download:
        download_photos()
        return
    catalog = _catalog()
    if args.list:
        for sample in catalog:
            print(f"{sample.name}: {sample.description}")
        return
    if args.bench:
        bench(args.threads)
        return
    if args.bench_video:
        bench_video()
        return
    if args.masks:
        for name, fn in (
            ("lake", mask_lake), ("forest", mask_forest), ("horse", mask_horse),
            ("fisherman", mask_fisherman), ("night_camp", mask_night_camp),
        ):
            image = load_photo(name)
            mask = fn(image)
            cover = {int(c): round(100.0 * float(np.mean(mask == c)), 1) for c in np.unique(mask)}
            print(name, cover)
            cv2.imwrite(str(OUT / f"mask_{name}.png"), overlay_mask(image, mask))
        return
    selected = [s for s in catalog if not args.only or s.name in args.only]
    results = [render_sample(s, args.all_frames) for s in selected]
    (OUT / "samples.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
