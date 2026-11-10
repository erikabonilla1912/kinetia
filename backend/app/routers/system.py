"""Salud, configuración pública y métricas (del modelo y de Prometheus)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .. import errors
from ..config import API_PREFIX, CLASS_TABLE, MOODS, UPLOAD_FORMATS
from ..observability import REGISTRY
from ..schemas import ConfigResponse, HealthResponse, LiveResponse
from .deps import AppServices, get_services

router = APIRouter(prefix=API_PREFIX)
metrics_router = APIRouter()


@router.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health(services: AppServices = Depends(get_services)) -> JSONResponse:
    models = services.models
    if not models.ready:
        return JSONResponse(
            {
                "status": "loading",
                "model_loaded": False,
                "depth_loaded": False,
                "model_version": None,
                "model_sha256": None,
                "depth_sha256": None,
            },
            status_code=503,
        )
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": True,
            "depth_loaded": False,  # sin modelo de profundidad desde el contrato v1.8
            "model_version": models.version,
            "model_sha256": models.seg_sha256,
            "depth_sha256": None,
        }
    )


@router.get("/health/live", response_model=LiveResponse)
async def health_live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/model/metrics")
def model_metrics(services: AppServices = Depends(get_services)) -> Response:
    """Contenido literal de metrics.json: los bytes del archivo, sin transformar."""
    try:
        content = services.settings.metrics_path.read_bytes()
    except OSError as exc:
        raise errors.metrics_not_found() from exc
    return Response(content=content, media_type="application/json")


@router.get("/config", response_model=ConfigResponse)
def public_config(services: AppServices = Depends(get_services)) -> dict:
    settings = services.settings
    return {
        "public_base_url": settings.public_base_url,
        "limits": {
            "max_upload_bytes": settings.max_upload_bytes,
            "max_video_seconds": settings.max_video_seconds,
            "min_side_px": settings.min_side_px,
            "formats": list(UPLOAD_FORMATS),
        },
        "moods": list(MOODS),
        "classes": [
            {"id": class_id, "name": name, "label": label, "color": color}
            for class_id, name, label, color in CLASS_TABLE
        ],
    }


@metrics_router.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
