"""Todo lo que habla con FFmpeg: codificación por pipe, ffprobe y lectura de frames.

Los parámetros de salida son los de CONTRATOS, sección 5. cv2.VideoWriter está prohibido.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..config import GIF_FPS, GIF_MAX_SIDE

PROBE_TIMEOUT_S = 30
FIRST_FRAME_TIMEOUT_S = 60


class FFmpegMissing(RuntimeError):
    """FFmpeg o ffprobe no están instalados en este equipo."""


class EncoderError(RuntimeError):
    """FFmpeg terminó con error o recibió frames inválidos."""


class ProbeError(RuntimeError):
    """ffprobe o ffmpeg no pudieron leer el archivo."""


def find_tool(name: str) -> str:
    """Busca en PATH y en la carpeta de enlaces de winget (no siempre está en PATH)."""
    found = shutil.which(name)
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    if local:
        found = shutil.which(name, path=str(Path(local) / "Microsoft" / "WinGet" / "Links"))
        if found:
            return found
        packages = Path(local) / "Microsoft" / "WinGet" / "Packages"
        if packages.is_dir():
            for candidate in packages.glob(f"*FFmpeg*/**/{name}.exe"):
                return str(candidate)
    raise FFmpegMissing(f"No se encontró {name}. Instala FFmpeg o añádelo al PATH.")


def ffmpeg_available() -> bool:
    try:
        find_tool("ffmpeg")
        find_tool("ffprobe")
    except FFmpegMissing:
        return False
    return True


@dataclass(frozen=True)
class VideoInfo:
    width: int  # ya con la rotación del contenedor aplicada
    height: int
    duration_s: float
    fps: float


def _parse_rate(text: str | None) -> float:
    try:
        numerator, _, denominator = (text or "").partition("/")
        value = float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return value if value > 0 else 0.0


def _rotation_of(stream: dict) -> int:
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            return int(float(side["rotation"]))
    try:
        return int(float((stream.get("tags") or {}).get("rotate", 0)))
    except ValueError:
        return 0


def probe_video(path: Path) -> VideoInfo:
    """Lee el contenedor con ffprobe. El demuxer se fuerza a mov/mp4, ya confirmado por magic bytes."""
    command = [
        find_tool("ffprobe"),
        "-v", "error",
        "-protocol_whitelist", "file",
        "-f", "mov",
        "-select_streams", "v:0",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=PROBE_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ProbeError("ffprobe excedió el tiempo límite") from exc
    if completed.returncode != 0:
        raise ProbeError("ffprobe no pudo leer el archivo")
    try:
        data = json.loads(completed.stdout.decode("utf-8", "replace"))
    except ValueError as exc:
        raise ProbeError("ffprobe devolvió una salida inválida") from exc
    streams = data.get("streams") or []
    if not streams:
        raise ProbeError("El archivo no tiene pista de video")
    stream = streams[0]
    try:
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ProbeError("Pista de video sin dimensiones") from exc
    if width <= 0 or height <= 0:
        raise ProbeError("Pista de video sin dimensiones")
    if abs(_rotation_of(stream)) % 180 == 90:
        width, height = height, width

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    duration = 0.0
    for candidate in ((data.get("format") or {}).get("duration"), stream.get("duration")):
        try:
            duration = float(candidate)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            break
    if duration <= 0 and fps > 0:
        try:
            duration = int(stream.get("nb_frames", 0)) / fps
        except (TypeError, ValueError):
            duration = 0.0
    if duration <= 0:
        raise ProbeError("No se pudo determinar la duración")
    return VideoInfo(width=width, height=height, duration_s=duration, fps=fps or 24.0)


def read_first_frame(path: Path) -> np.ndarray:
    """Primer frame BGR, con la rotación del contenedor aplicada por FFmpeg."""
    command = [
        find_tool("ffmpeg"),
        "-nostdin", "-v", "error",
        "-protocol_whitelist", "file,pipe",
        "-f", "mov",
        "-i", str(path),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-c:v", "png",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=FIRST_FRAME_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ProbeError("FFmpeg excedió el tiempo límite") from exc
    if completed.returncode != 0 or not completed.stdout:
        raise ProbeError("FFmpeg no pudo decodificar el primer frame")
    frame = cv2.imdecode(np.frombuffer(completed.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ProbeError("FFmpeg devolvió un frame ilegible")
    return frame


def iter_video_frames(path: Path, width: int, height: int, fps: int, max_frames: int) -> Iterator[np.ndarray]:
    """Genera frames BGR uno a uno, reescalados y remuestreados, sin acumularlos en memoria."""
    command = [
        find_tool("ffmpeg"),
        "-nostdin", "-v", "error",
        "-protocol_whitelist", "file,pipe",
        "-f", "mov",
        "-i", str(path),
        "-vf", f"fps={fps},scale={width}:{height}:flags=area",
        "-frames:v", str(max_frames),
        "-an",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    frame_bytes = width * height * 3
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
        try:
            while True:
                buffer = process.stdout.read(frame_bytes)
                if len(buffer) < frame_bytes:
                    break
                yield np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3).copy()
            if process.wait(timeout=30) != 0:
                stderr.seek(0)
                tail = stderr.read()[-500:].decode("utf-8", "replace")
                raise ProbeError(f"FFmpeg falló al leer el video: {tail}")
        finally:
            if process.poll() is None:
                process.kill()
            process.stdout.close()
            process.wait()


def even_size(width: int, height: int) -> tuple[int, int]:
    """H.264 con yuv420p exige dimensiones pares."""
    return max(2, width - width % 2), max(2, height - height % 2)


def gif_size(width: int, height: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= GIF_MAX_SIDE:
        return width, height
    scale = GIF_MAX_SIDE / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def _output_args(output_format: str, width: int, height: int) -> tuple[list[str], tuple[int, int], int | None]:
    """Devuelve (argumentos, tamaño de salida, fps de salida o None si conserva el de entrada)."""
    if output_format == "mp4":
        out_w, out_h = even_size(width, height)
        args = [
            "-vf", f"crop={out_w}:{out_h}:0:0",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            "-f", "mp4",
        ]
        return args, (out_w, out_h), None
    if output_format == "gif":
        out_w, out_h = gif_size(width, height)
        graph = (
            f"fps={GIF_FPS},scale={out_w}:{out_h}:flags=lanczos,"
            "split[a][b];[a]palettegen[p];[b][p]paletteuse"
        )
        args = ["-filter_complex", graph, "-loop", "0", "-an", "-f", "gif"]
        return args, (out_w, out_h), GIF_FPS
    raise ValueError(f"Formato de salida desconocido: {output_format}")


class FrameEncoder:
    """Envía frames BGR crudos a FFmpeg por stdin. El error va a un archivo para evitar bloqueos."""

    def __init__(self, output_path: Path, output_format: str, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3
        args, self.output_size, output_fps = _output_args(output_format, width, height)
        self.output_fps = output_fps or fps
        self.frames_written = 0
        self._stderr = tempfile.TemporaryFile()
        command = [
            find_tool("ffmpeg"),
            "-nostdin", "-y", "-v", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            *args,
            str(output_path),
        ]
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr
        )

    def _stderr_tail(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read()[-800:].decode("utf-8", "replace")

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise EncoderError(f"Frame inválido: {frame.dtype} {frame.shape}")
        try:
            self._process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            self._process.wait()
            raise EncoderError(f"FFmpeg terminó antes de tiempo: {self._stderr_tail()}") from exc
        self.frames_written += 1

    def close(self) -> None:
        """Cierra el pipe y espera a que FFmpeg termine de escribir el archivo."""
        try:
            self._process.stdin.close()
        except OSError:
            pass
        code = self._process.wait()
        tail = self._stderr_tail() if code != 0 else ""
        self._stderr.close()
        if code != 0:
            raise EncoderError(f"FFmpeg terminó con código {code}: {tail}")

    def abort(self) -> None:
        """Descarta la codificación en curso."""
        if self._process.poll() is None:
            self._process.kill()
        try:
            self._process.stdin.close()
        except OSError:
            pass
        self._process.wait()
        if not self._stderr.closed:
            self._stderr.close()
