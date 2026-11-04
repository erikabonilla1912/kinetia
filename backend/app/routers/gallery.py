"""Galería para la pantalla grande y moderación durante la Expo."""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, Query, Response

from .. import errors
from ..observability import log_event
from ..schemas import GalleryResponse
from ..services.job_manager import result_path, result_url, to_iso
from .deps import AppServices, get_services

router = APIRouter()


def _parse_after(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    # Un "+" sin codificar en la URL llega como espacio.
    text = value.strip().replace(" ", "+")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise errors.invalid_params(
            "El parámetro after debe ser una fecha ISO 8601.",
            [{"field": "after", "problem": "Fecha inválida."}],
        ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return to_iso(moment)


@router.get("/gallery", response_model=GalleryResponse)
def list_gallery(
    limit: int = Query(24, ge=1, le=100),
    after: Optional[str] = Query(None),
    services: AppServices = Depends(get_services),
) -> dict:
    rows = services.store.gallery(limit, _parse_after(after))
    return {
        "items": [
            {
                "job_id": row["job_id"],
                "mood": row["mood"],
                "result_url": result_url(row["job_id"], row["output_format"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    }


@router.delete("/gallery/{job_id}", status_code=204, response_class=Response)
def moderate_gallery(
    job_id: str,
    x_admin_token: Optional[str] = Header(None),
    services: AppServices = Depends(get_services),
) -> Response:
    """Retira el resultado de la galería y borra su archivo: una imagen moderada no debe seguir accesible."""
    configured = services.settings.admin_token
    if not configured or not x_admin_token or not secrets.compare_digest(
        x_admin_token.encode("utf-8"), configured.encode("utf-8")
    ):
        raise errors.unauthorized()
    row = services.store.get(job_id)
    if row is None or not row["gallery_opt_in"]:
        raise errors.job_not_found(job_id)
    if not row["gallery_removed"]:
        services.store.update(job_id, gallery_removed=1)
        result_path(services.settings, job_id, row["output_format"]).unlink(missing_ok=True)
        log_event("gallery_item_removed", logging.WARNING, job_id=job_id)
    return Response(status_code=204)
