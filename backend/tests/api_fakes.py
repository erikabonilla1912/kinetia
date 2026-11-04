"""Dobles de prueba de la API. Solo viven en tests/: nada de esto llega a producción."""
from __future__ import annotations

import io
import struct
import threading
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from animation_engine.types import MoodParams
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app

API = "/api/v1"


class FakeEngine:
    """Sustituto del motor con sus mismas firmas. Devuelve la imagen sin animar."""

    def __init__(self) -> None:
        self.image_calls: list[dict] = []
        self.video_calls: list[dict] = []

    def get_mood(self, name: str) -> MoodParams:
        return MoodParams(name, 1.0, 1.0, 0.0, 1.0, "none", 0.0, 0.01, 0.0)

    def animate_image(self, image_bgr, mask, depth, config, progress=None):
        self.image_calls.append({"image": image_bgr, "mask": mask, "depth": depth, "config": config})
        total = round(config.duration_s * config.fps)
        for index in range(total):
            if progress:
                progress((index + 1) / total)
            yield image_bgr

    def animate_video(self, frames, segment, infer_every_n, config, depth=None, progress=None):
        segments = 0
        for index, frame in enumerate(frames):
            if index % infer_every_n == 0:
                segment(frame)
                segments += 1
            yield frame
        self.video_calls.append({"config": config, "depth": depth, "segments": segments, "frames": index + 1})


class FakeFrameEncoder:
    """Escribe bytes en vez de llamar a FFmpeg, para probar el ciclo del job sin FFmpeg."""

    def __init__(self, output_path: Path, output_format: str, width: int, height: int, fps: int) -> None:
        self.path = output_path
        self.output_size = (width, height)
        self.output_fps = fps
        self.frames_written = 0
        self.handle = output_path.open("wb")

    def write(self, frame) -> None:
        self.handle.write(frame[:1, :8].tobytes())
        self.frames_written += 1

    def close(self) -> None:
        self.handle.close()

    def abort(self) -> None:
        self.handle.close()


def make_settings(root: Path, models_dir: Path, **overrides) -> Settings:
    values = dict(
        models_dir=models_dir,
        results_dir=root / "results",
        db_path=root / "db" / "kinetia.db",
        tmp_dir=root / "tmp",
        allow_dummy_model=True,
        admin_token="secreto-de-prueba",
        ort_intra_op_threads=2,
        log_level="INFO",
    )
    values.update(overrides)
    return Settings(**values)


def wait_until(predicate, timeout: float = 30.0, interval: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("La condición no se cumplió a tiempo")


@contextmanager
def running_client(settings: Settings, engine_provider=None, wait_ready: bool = True):
    kwargs = {"exit_on_load_failure": False}
    if engine_provider is not None:
        kwargs["engine_provider"] = engine_provider
    app = create_app(settings, **kwargs)
    with TestClient(app, raise_server_exceptions=False) as client:
        if wait_ready:
            wait_until(lambda: client.get(f"{API}/health").status_code == 200)
        yield client


def wait_job(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"{API}/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"El job {job_id} no terminó a tiempo")


def sample_array(width: int = 640, height: int = 480) -> np.ndarray:
    """Imagen RGB con gradientes y ruido, reproducible."""
    rng = np.random.default_rng(1)
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    grid = np.zeros((height, width, 3), dtype=np.float32)
    grid[..., 0] = x
    grid[..., 1] = y[:, None]
    grid[..., 2] = 128
    noise = rng.integers(0, 32, size=(height, width, 3))
    return np.clip(grid + noise, 0, 255).astype(np.uint8)


def image_bytes(fmt: str = "JPEG", width: int = 640, height: int = 480, **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(sample_array(width, height)).save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def png_header_only(width: int, height: int) -> bytes:
    """PNG con cabecera válida que declara un tamaño enorme, sin datos reales: bomba de descompresión."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\0" * 64)) + chunk(b"IEND", b"")
    )


def post_file(client: TestClient, path: str, data: bytes, filename: str = "foto.jpg", content_type: str = "image/jpeg", **form):
    return client.post(f"{API}{path}", files={"file": (filename, data, content_type)}, data=form)


def assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert set(error) == {"code", "message", "details"}
    assert error["code"] == code
    assert isinstance(error["message"], str) and error["message"]
    assert "Traceback" not in response.text
    assert isinstance(error["details"], dict)
    return error


class Gate:
    """Proveedor de motor que se detiene hasta abrir la compuerta: mantiene un job en curso."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self):
        self.entered.set()
        self.release.wait(timeout=60)
        return self.engine
