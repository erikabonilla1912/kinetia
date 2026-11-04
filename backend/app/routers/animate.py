"""POST /api/v1/animate."""
from __future__ import annotations

import functools
import math
import secrets
from typing import Literal, Optional

import anyio.to_thread
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse

from .. import errors
from ..config import API_PREFIX
from ..schemas import AnimateAccepted
from ..services.brush import parse_strokes, strokes_to_json
from ..services.validation import spool_upload, validate_media
from .deps import AppServices, get_services

router = APIRouter()

MIN_DURATION_S = 2.0
MAX_DURATION_S = 6.0
MAX_SEED = 2**32 - 1


@router.post("/animate", status_code=202, response_model=AnimateAccepted)
async def animate(
    file: Optional[UploadFile] = File(None),
    mood: Literal["calma", "epico", "magico", "nostalgico"] = Form("calma"),
    output_format: Literal["mp4", "gif"] = Form("mp4"),
    duration_s: float = Form(4.0),
    mask_strokes: Optional[str] = Form(None),
    brush_mode: Literal["add", "only"] = Form("add"),
    gallery_opt_in: bool = Form(False),
    seed: Optional[int] = Form(None, ge=0, le=MAX_SEED),
    services: AppServices = Depends(get_services),
) -> JSONResponse:
    if not services.models.ready:
        raise errors.model_not_ready()
    path = await spool_upload(file, services.settings)
    try:
        services.jobs.ensure_capacity()
        media = await anyio.to_thread.run_sync(
            functools.partial(validate_media, path, services.settings),
            limiter=services.analyze_limiter,
        )
        # duration_s solo aplica a imágenes: en un video se ignora, sea cual sea su valor.
        if media.media == "image" and (
            not math.isfinite(duration_s) or not MIN_DURATION_S <= duration_s <= MAX_DURATION_S
        ):
            raise errors.invalid_params(
                f"duration_s debe estar entre {MIN_DURATION_S:g} y {MAX_DURATION_S:g} segundos.",
                [{"field": "duration_s", "problem": "Fuera de rango."}],
            )
        strokes = parse_strokes(mask_strokes)
        if brush_mode == "only" and not strokes:
            raise errors.invalid_params(
                "brush_mode 'only' necesita al menos un trazo en mask_strokes.",
                [{"field": "mask_strokes", "problem": "Falta para este modo."}],
            )
        job_id, position = services.jobs.submit(
            upload_path=path,
            media=media.media,
            mood=mood,
            output_format=output_format,
            duration_s=duration_s if media.media == "image" else 0.0,
            seed=seed if seed is not None else secrets.randbelow(MAX_SEED + 1),
            brush_mode=brush_mode,
            strokes_json=strokes_to_json(strokes) if strokes else None,
            gallery_opt_in=gallery_opt_in,
        )
    finally:
        path.unlink(missing_ok=True)
    body = AnimateAccepted(
        job_id=job_id,
        status="queued",
        queue_position=position,
        status_url=f"{API_PREFIX}/jobs/{job_id}",
    )
    return JSONResponse(body.model_dump(), status_code=202)
