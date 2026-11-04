"""Validación de subidas, en el orden del contrato: tamaño, formato, decodificación, resolución, duración."""
from __future__ import annotations

import io
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import UploadFile
from PIL import Image
from starlette.concurrency import run_in_threadpool

from .. import errors
from ..config import Settings
from ..observability import log_event
from . import encoder

# Pillow no debe decidir el límite: lo aplicamos nosotros con MAX_PIXELS a partir de la cabecera,
# antes de descomprimir un solo píxel.
Image.MAX_IMAGE_PIXELS = None

HEAD_BYTES = 32
IMAGE_FORMATS = {"jpg": "JPEG", "png": "PNG", "webp": "WEBP"}
PIL_MODES_WITH_ALPHA = ("RGBA", "LA", "PA")
# ftyp de imágenes y audio: tienen la misma estructura que un mp4, pero no son video.
NON_VIDEO_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"avif", b"avis",
                    b"M4A ", b"M4B ", b"M4P ", b"F4A ", b"F4B "}
LEGACY_MOV_ATOMS = {b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}


def sniff_format(head: bytes) -> Optional[str]:
    """Formato según los magic bytes: jpg, png, webp, mp4, mov o None."""
    if head[:3] == b"\xff\xd8\xff":
        return "jpg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in NON_VIDEO_BRANDS:
            return None
        return "mov" if brand == b"qt  " else "mp4"
    if head[4:8] in LEGACY_MOV_ATOMS:
        return "mov"
    return None


def apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 2:
        return cv2.flip(image, 1)
    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(image, 0)
    if orientation == 5:
        return cv2.transpose(image)
    if orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(image), -1)
    if orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    if pil_image.mode in PIL_MODES_WITH_ALPHA or "transparency" in pil_image.info:
        rgba = pil_image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, (255, 255, 255))
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        rgb = flattened
    else:
        rgb = pil_image.convert("RGB")
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def decode_image(data: bytes, fmt: str, settings: Settings) -> np.ndarray:
    """Imagen BGR uint8 con la orientación EXIF aplicada, o ApiError.

    El límite de píxeles se comprueba con la cabecera, antes de decodificar. Pillow verifica la
    integridad (cv2.imdecode acepta un JPEG truncado) y OpenCV entrega los píxeles, igual que en ml/.
    """
    try:
        pil_image = Image.open(io.BytesIO(data), formats=[IMAGE_FORMATS[fmt]])
        width, height = pil_image.size
    except Exception as exc:
        raise errors.corrupt_file() from exc
    if width <= 0 or height <= 0:
        raise errors.corrupt_file()
    if width * height > settings.max_pixels:
        raise errors.resolution_too_high(width, height, settings.max_pixels)
    try:
        pil_image.load()
    except Exception as exc:
        raise errors.corrupt_file() from exc
    try:
        orientation = int(pil_image.getexif().get(0x0112, 1))
    except Exception:  # Un EXIF dañado no invalida la imagen.
        orientation = 1

    array = None
    try:
        flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    except cv2.error:
        array = None  # Por ejemplo, WebP animado: se usa su primer frame, ya decodificado por Pillow.
    if array is None:
        try:
            array = _pil_to_bgr(pil_image)
        except Exception as exc:
            raise errors.corrupt_file() from exc
    array = apply_exif_orientation(array, orientation)

    final_height, final_width = array.shape[:2]
    if min(final_width, final_height) < settings.min_side_px:
        raise errors.resolution_too_low(final_width, final_height, settings.min_side_px)
    return array


def _validate_video(path: Path, settings: Settings) -> tuple[np.ndarray, encoder.VideoInfo]:
    try:
        info = encoder.probe_video(path)
    except encoder.ProbeError as exc:
        raise errors.corrupt_file() from exc
    if info.width * info.height > settings.max_pixels:
        raise errors.resolution_too_high(info.width, info.height, settings.max_pixels)
    try:
        frame = encoder.read_first_frame(path)
    except encoder.ProbeError as exc:
        raise errors.corrupt_file() from exc
    height, width = frame.shape[:2]
    if min(width, height) < settings.min_side_px:
        raise errors.resolution_too_low(width, height, settings.min_side_px)
    if info.duration_s > settings.max_video_seconds:
        raise errors.video_too_long(info.duration_s, settings.max_video_seconds)
    return frame, info


@dataclass
class ValidatedMedia:
    media: str  # "image" o "video"
    fmt: str
    image_bgr: np.ndarray  # la foto o el primer frame del video
    video: Optional[encoder.VideoInfo] = None


def validate_media(path: Path, settings: Settings) -> ValidatedMedia:
    """Valida un archivo en disco. Se usa tanto al recibirlo como en el hilo de jobs."""
    size = path.stat().st_size
    if size > settings.max_upload_bytes:
        raise errors.file_too_large(size, settings.max_upload_bytes)
    with path.open("rb") as handle:
        head = handle.read(HEAD_BYTES)
    fmt = sniff_format(head)
    if fmt is None:
        raise errors.unsupported_format()
    if fmt in IMAGE_FORMATS:
        return ValidatedMedia("image", fmt, decode_image(path.read_bytes(), fmt, settings))
    try:
        frame, info = _validate_video(path, settings)
    except encoder.FFmpegMissing as exc:
        log_event("ffmpeg_missing", logging.ERROR, error_code="INTERNAL_ERROR")
        raise errors.internal_error("Este servidor no puede procesar videos en este momento.") from exc
    return ValidatedMedia("video", fmt, frame, info)


async def spool_upload(upload: Optional[UploadFile], settings: Settings) -> Path:
    """Copia la subida a TMP_DIR. Tamaño es el primer error que se comprueba."""
    if upload is None or not isinstance(upload.size, int) or upload.size == 0:
        raise errors.missing_file()
    if upload.size > settings.max_upload_bytes:
        raise errors.file_too_large(upload.size, settings.max_upload_bytes)
    # El nombre del cliente nunca se usa en el disco.
    destination = settings.tmp_dir / "uploads" / uuid.uuid4().hex

    def _copy() -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        upload.file.seek(0)
        with destination.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle, 1024 * 1024)

    await run_in_threadpool(_copy)
    return destination
