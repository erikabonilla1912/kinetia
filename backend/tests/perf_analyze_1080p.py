"""Mide POST /api/v1/analyze con 1920×1080 contra un uvicorn real, en otro proceso.

    backend\\.venv\\Scripts\\python backend\\tests\\perf_analyze_1080p.py --models models/dev

No lo recoge pytest. Cada cifra es el tiempo de ida y vuelta medido desde el cliente.
"""
from __future__ import annotations

import argparse
import io
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

BACKEND = Path(__file__).resolve().parent.parent


def photo_like_jpeg(width: int = 1920, height: int = 1080, quality: int = 92) -> bytes:
    """Gradientes y ruido: comprime y decodifica de forma comparable a una foto, o peor."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    array = np.zeros((height, width, 3), dtype=np.float32)
    array[..., 0] = x
    array[..., 1] = y[:, None]
    array[..., 2] = 255 - x
    array += rng.normal(0, 12, size=array.shape)
    buffer = io.BytesIO()
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop_server(server: subprocess.Popen) -> None:
    """En Windows, el python.exe de un venv es un lanzador: hay que terminar también a su hijo."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(server.pid)], capture_output=True, check=False)
    else:
        server.terminate()
    server.wait(timeout=20)


def measure(client: httpx.Client, base: str, data: bytes, include_depth: bool, runs: int) -> tuple[list[float], dict]:
    times, last = [], {}
    for _ in range(runs):
        started = time.perf_counter()
        response = client.post(
            f"{base}/api/v1/analyze",
            files={"file": ("foto.jpg", data, "image/jpeg")},
            data={"include_depth": str(include_depth).lower()},
        )
        times.append((time.perf_counter() - started) * 1000.0)
        response.raise_for_status()
        last = response.json()
    return times, last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=str(BACKEND.parent / "models" / "dev"))
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    data = photo_like_jpeg()
    port = free_port()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        env = {
            **os.environ,
            "MODELS_DIR": str(Path(args.models).resolve()),
            "ALLOW_DUMMY_MODEL": "true",
            "RESULTS_DIR": str(Path(work) / "results"),
            "DB_PATH": str(Path(work) / "db" / "k.db"),
            "TMP_DIR": str(Path(work) / "tmp"),
            "LOG_LEVEL": "WARNING",
        }
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"],
            cwd=BACKEND,
            env=env,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            with httpx.Client(timeout=60) as client:
                deadline = time.monotonic() + 120
                started = time.perf_counter()
                while True:
                    try:
                        if client.get(f"{base}/api/v1/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        raise SystemExit("El servidor no quedó listo a tiempo")
                    time.sleep(0.05)
                print(f"Listo en {time.perf_counter() - started:.2f} s. Imagen: 1920x1080 JPEG, {len(data) / 1e6:.2f} MB")
                for include_depth in (False, True):
                    times, body = measure(client, base, data, include_depth, args.runs)
                    print(
                        f"include_depth={include_depth!s:5} modelo={body['model_version']} "
                        f"min={min(times):.0f} ms mediana={statistics.median(times):.0f} ms máx={max(times):.0f} ms "
                        f"| último inference_ms={body['inference_ms']} tamaño de trabajo={body['width']}x{body['height']}"
                    )
        finally:
            stop_server(server)


if __name__ == "__main__":
    main()
