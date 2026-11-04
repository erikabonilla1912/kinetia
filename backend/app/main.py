"""Aplicación FastAPI de KINETIA. Contrato: docs/CONTRATOS.md, sección 4.

    uvicorn app.main:app --app-dir backend --env-file .env --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

import anyio
from fastapi import FastAPI

from . import errors
from .config import API_PREFIX, Settings
from .observability import configure_logging, error_code_var, log_event, record_error, request_id_var
from .routers import analyze, animate, gallery, jobs, results, system
from .routers.deps import AppServices
from .services.encoder import ffmpeg_available
from .services.inference import ModelStartupError, ModelStore
from .services.job_manager import JobManager, JobStore, load_engine

ANALYZE_CONCURRENCY = 2
# Margen sobre MAX_UPLOAD_BYTES para el resto del formulario (mask_strokes y separadores).
FORM_OVERHEAD_BYTES = 1024 * 1024
# Si se responde sin leer el cuerpo, el cliente que aún sube ve la conexión cortada y no el 413.
MAX_DRAIN_BYTES = 64 * 1024 * 1024
UPLOAD_PATHS = (f"{API_PREFIX}/analyze", f"{API_PREFIX}/animate")
QUIET_PATHS = (f"{API_PREFIX}/health", f"{API_PREFIX}/health/live", "/metrics")


class RequestContextMiddleware:
    """Asigna request_id, registra cada petición y corta las subidas demasiado grandes."""

    def __init__(self, app: Any, max_upload_bytes: int) -> None:
        self.app = app
        self.max_upload_bytes = max_upload_bytes

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        request_id_var.set(request_id)
        error_code_var.set(None)
        started = time.perf_counter()
        status_holder = {"status": 500}

        async def send_with_context(message: dict) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode("ascii")))
            await send(message)

        path = scope.get("path", "")
        try:
            if path in UPLOAD_PATHS:
                await self._handle_upload(scope, receive, send_with_context)
            else:
                await self.app(scope, receive, send_with_context)
        finally:
            level = logging.DEBUG if path in QUIET_PATHS else logging.INFO
            log_event(
                "http_request",
                level,
                method=scope.get("method"),
                path=path,
                status=status_holder["status"],
                duration_ms=round((time.perf_counter() - started) * 1000.0, 1),
            )

    async def _handle_upload(self, scope: dict, receive: Callable, send: Callable) -> None:
        limit = self.max_upload_bytes + FORM_OVERHEAD_BYTES
        headers = dict(scope.get("headers") or [])
        try:
            declared = int(headers.get(b"content-length", b""))
        except ValueError:
            declared = None

        if declared is not None and declared > limit:
            # Se descarta lo que llega, sin escribirlo a disco, y luego se responde.
            drained = 0
            while drained < MAX_DRAIN_BYTES:
                message = await receive()
                if message["type"] != "http.request":
                    break
                drained += len(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            await self._send_too_large(declared, send)
            return

        received = 0

        async def limited_receive() -> dict:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:  # Subida sin Content-Length.
                    raise errors.file_too_large(received, self.max_upload_bytes)
            return message

        await self.app(scope, limited_receive, send)

    async def _send_too_large(self, size: int, send: Callable) -> None:
        error = errors.file_too_large(size, self.max_upload_bytes)
        record_error(error.code)
        body = json.dumps(errors.error_body(error.code, error.message, error.details)).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": error.status_code,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})


def create_app(
    settings: Optional[Settings] = None,
    engine_provider: Callable[[], Any] = load_engine,
    exit_on_load_failure: bool = True,
) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)  # uvicorn ya configuró sus loggers antes de importar la app.
        settings.tmp_dir.mkdir(parents=True, exist_ok=True)
        # El sistema de archivos raíz es de solo lectura en el pod: las subidas también van a TMP_DIR.
        tempfile.tempdir = str(settings.tmp_dir)

        models = ModelStore(settings)
        models.preflight()  # ModelStartupError aquí impide arrancar: uvicorn termina con error.

        store = JobStore(settings.db_path)
        manager = JobManager(settings, store, models, engine_provider)
        manager.start()
        app.state.services = AppServices(
            settings=settings,
            models=models,
            store=store,
            jobs=manager,
            analyze_limiter=anyio.CapacityLimiter(ANALYZE_CONCURRENCY),
        )
        if not ffmpeg_available():
            log_event("ffmpeg_missing", logging.ERROR, error_code="ENCODER_UNAVAILABLE")

        def load_models() -> None:
            try:
                models.load()
            except (ModelStartupError, OSError) as exc:
                log_event("model_load_failed", logging.CRITICAL, exc_info=exc, reason=str(exc))
                if exit_on_load_failure:
                    logging.shutdown()
                    os._exit(1)

        # En segundo plano: /health responde 503 y /health/live 200 mientras se cargan.
        threading.Thread(target=load_models, name="kinetia-model-loader", daemon=True).start()
        log_event("api_started", version=models.version)
        try:
            yield
        finally:
            manager.stop()
            store.close()
            log_event("api_stopped")

    app = FastAPI(title="KINETIA API", version="1.1.0", lifespan=lifespan)
    app.state.settings = settings
    errors.register_error_handlers(app)
    app.add_middleware(RequestContextMiddleware, max_upload_bytes=settings.max_upload_bytes)
    for module in (analyze, animate, jobs, results, gallery):
        app.include_router(module.router, prefix=API_PREFIX)
    app.include_router(system.router)
    app.include_router(system.metrics_router)
    return app


app = create_app()
