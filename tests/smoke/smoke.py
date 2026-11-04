#!/usr/bin/env python3
"""Smoke test de KINETIA a través de Service y Nginx. Solo biblioteca estándar y ffprobe.

Sale con 1 si alguna comprobación falla. Las imágenes se generan al vuelo (ruido en un PNG sin
comprimir), para no depender de archivos de datos.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zlib

FAILURES: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'OK' if ok else 'FALLA'}] {name}{': ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)
    return ok


def request(url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 180):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def noise_png(width: int, height: int) -> bytes:
    """PNG RGB de ruido sin comprimir: su tamaño es aproximadamente width * height * 3 bytes."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    raw = b"".join(b"\x00" + os.urandom(width * 3) for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 0)) + chunk(b"IEND", b"")


def multipart(file_bytes: bytes, fields: dict[str, str]) -> tuple[bytes, dict]:
    boundary = uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="smoke.png"\r\n'
        "Content-Type: image/png\r\n\r\n".encode()
    )
    parts.extend([file_bytes, f"\r\n--{boundary}--\r\n".encode()])
    return b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def run_job(base: str, image: bytes, label: str, timeout_s: float) -> str | None:
    """Anima, espera el job, descarga el MP4 y comprueba H.264. Devuelve la URL del resultado."""
    body, headers = multipart(image, {"mood": "calma", "output_format": "mp4", "duration_s": "2.0", "seed": "5"})
    status, _, payload = request(f"{base}/api/v1/animate", body, headers)
    if not report(f"{label}: POST /animate da 202", status == 202, f"HTTP {status}"):
        print(payload[:300].decode("utf-8", "replace"))
        return None
    job_url = json.loads(payload)["status_url"]
    started = time.monotonic()
    job = {}
    while time.monotonic() - started < timeout_s:
        _, _, job_payload = request(f"{base}{job_url}")
        job = json.loads(job_payload)
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(1)
    elapsed = time.monotonic() - started
    if not report(f"{label}: job termina bien", job.get("status") == "succeeded", f"{job.get('status')} en {elapsed:.1f} s; error={job.get('error')}"):
        return None
    result_url = job["result"]["url"]
    status, _, video = request(f"{base}{result_url}")
    report(f"{label}: descarga del MP4", status == 200 and len(video) > 0, f"HTTP {status}, {len(video)} bytes")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        report(f"{label}: ffprobe disponible", False, "no está en PATH")
        return result_url
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "out.mp4")
        with open(path, "wb") as handle:
            handle.write(video)
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt,width,height", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=False,
        )
    report(f"{label}: ffprobe confirma H.264", probe.stdout.startswith("h264"), probe.stdout.strip())
    return result_url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--metrics-url", default="", help="URL directa a la API (port-forward) para /metrics")
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--full", action="store_true", help="Añade las subidas de 12 MB y más de 15 MB")
    parser.add_argument("--job-timeout", type=float, default=240)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    status, _, payload = request(f"{base}/api/v1/health")
    report("GET /api/v1/health da 200", status == 200, payload.decode("utf-8", "replace")[:220])
    if args.health_only:
        return 1 if FAILURES else 0

    status, _, _ = request(f"{base}/metrics")
    report("/metrics no se publica por Nginx", status == 404, f"HTTP {status}")
    if args.metrics_url:
        status, _, payload = request(f"{args.metrics_url.rstrip('/')}/metrics")
        report("/metrics responde directo en :8000", status == 200 and b"kinetia_jobs_total" in payload, f"HTTP {status}")
    status, _, payload = request(f"{base}/no-existe/ruta")
    report("Ruta desconocida vuelve a index.html", status == 200 and b"<html" in payload.lower(), f"HTTP {status}")
    status, _, payload = request(f"{base}/api/v1/no-existe")
    report("Ruta de API desconocida da 404 en JSON", status == 404 and b'"NOT_FOUND"' in payload, f"HTTP {status}")

    small = noise_png(640, 480)
    body, headers = multipart(small, {"include_depth": "true"})
    status, _, payload = request(f"{base}/api/v1/analyze", body, headers)
    detail = f"HTTP {status}"
    if status == 200:
        data = json.loads(payload)
        detail += f", {data['width']}x{data['height']}, modelo {data['model_version']}, inference_ms={data['inference_ms']}"
    report("POST /analyze da 200", status == 200, detail)
    run_job(base, small, "Imagen 640x480", args.job_timeout)

    if args.full:
        big = noise_png(2000, 2000)
        run_job(base, big, f"Subida de {len(big) / 1e6:.1f} MB", args.job_timeout)
        too_big = noise_png(2400, 2300)
        body, headers = multipart(too_big, {})
        status, response_headers, payload = request(f"{base}/api/v1/animate", body, headers)
        try:
            code = json.loads(payload)["error"]["code"]
        except (ValueError, KeyError, TypeError):
            code = payload[:80].decode("utf-8", "replace")
        report(
            f"Subida de {len(too_big) / 1e6:.1f} MB da 413 JSON de la API",
            status == 413 and code == "FILE_TOO_LARGE",
            f"HTTP {status}, code={code}, content-type={response_headers.get('Content-Type')}",
        )

    print(f"\nResultado: {'TODO OK' if not FAILURES else 'FALLAS: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
