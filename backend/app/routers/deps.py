"""Acceso a los servicios creados en el arranque de la aplicación."""
from __future__ import annotations

from dataclasses import dataclass

import anyio
from starlette.requests import Request

from ..config import Settings
from ..services.inference import ModelStore
from ..services.job_manager import JobManager, JobStore


@dataclass
class AppServices:
    settings: Settings
    models: ModelStore
    store: JobStore
    jobs: JobManager
    # Limita los /analyze simultáneos: cada uno puede ocupar cientos de MB al decodificar.
    analyze_limiter: anyio.CapacityLimiter


def get_services(request: Request) -> AppServices:
    return request.app.state.services
