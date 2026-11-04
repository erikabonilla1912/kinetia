"""POST /api/v1/analyze."""
from __future__ import annotations

import base64
import functools
import time
from pathlib import Path
from typing import Any, Optional

import anyio.to_thread
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile

from .. import errors
from ..schemas import AnalyzeResponse
from ..services.preprocess import resize_to_work
from ..services.validation import spool_upload, validate_media
from .deps import AppServices, get_services

router = APIRouter()


def _png_base64(array: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError("OpenCV no pudo codificar el PNG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _analyze_sync(services: AppServices, path: Path, include_depth: bool, started: float) -> dict[str, Any]:
    settings = services.settings
    media = validate_media(path, settings)
    image = resize_to_work(media.image_bgr, settings.work_max_side)
    segmentation = services.models.segment(image)
    depth_ms = None
    depth_png = None
    if include_depth:
        depth, depth_ms = services.models.estimate_depth(image)
        depth_png = _png_base64(np.round(depth * 255.0).astype(np.uint8))
    mask_png = _png_base64(segmentation.mask)
    return {
        "width": image.shape[1],
        "height": image.shape[0],
        "mask_png_base64": mask_png,
        "classes": segmentation.classes,
        "inference_ms": {
            "segmentation": round(segmentation.elapsed_ms, 1),
            "depth": round(depth_ms, 1) if depth_ms is not None else None,
            "total": round((time.perf_counter() - started) * 1000.0, 1),
        },
        "depth_png_base64": depth_png,
        "model_version": services.models.version,
        "model_sha256": services.models.seg_sha256,
        "conf_threshold": settings.conf_threshold,
    }


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: Optional[UploadFile] = File(None),
    include_depth: bool = Form(False),
    services: AppServices = Depends(get_services),
) -> dict[str, Any]:
    started = time.perf_counter()
    if not services.models.ready:
        raise errors.model_not_ready()
    path = await spool_upload(file, services.settings)
    try:
        # Fuera del bucle de eventos: /health y el resto de la API siguen respondiendo.
        return await anyio.to_thread.run_sync(
            functools.partial(_analyze_sync, services, path, include_depth, started),
            limiter=services.analyze_limiter,
        )
    finally:
        path.unlink(missing_ok=True)
