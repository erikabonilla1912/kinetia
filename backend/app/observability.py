"""Logs JSON y métricas de Prometheus (docs/CONTRATOS.md, sección 8).

Nunca se registra el contenido de las imágenes ni datos de los visitantes: por eso se apaga
el log de acceso de uvicorn, que incluye la IP del cliente.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, disable_created_metrics

disable_created_metrics()

# Cada petición o job fija su contexto; el formateador lo añade a cada línea.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)
error_code_var: ContextVar[str | None] = ContextVar("error_code", default=None)

logger = logging.getLogger("kinetia")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "event": record.getMessage(),
        }
        fields = dict(getattr(record, "fields", {}))
        request_id = request_id_var.get()
        job_id = fields.pop("job_id", None) or job_id_var.get()
        error_code = fields.pop("error_code", None) or error_code_var.get()
        if request_id:
            entry["request_id"] = request_id
        if job_id:
            entry["job_id"] = job_id
        if "duration_ms" in fields:
            entry["duration_ms"] = fields.pop("duration_ms")
        if error_code:
            entry["error_code"] = error_code
        entry.update(fields)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class _KinetiaHandler(logging.StreamHandler):
    """Tipo propio para poder reemplazar solo este manejador sin tocar los ajenos."""


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, _KinetiaHandler):
            root.removeHandler(handler)
    handler = _KinetiaHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    # En DEBUG este parser describe cada parte de la subida: nunca debe llegar a los logs.
    logging.getLogger("python_multipart").setLevel(logging.WARNING)

    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    access.disabled = True


def log_event(event: str, level: int = logging.INFO, exc_info: Any = None, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields}, exc_info=exc_info)


REGISTRY = CollectorRegistry()

INFERENCE_SECONDS = Histogram(
    "kinetia_inference_seconds",
    "Duración de la inferencia por modelo.",
    ["model"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    registry=REGISTRY,
)
JOB_DURATION_SECONDS = Histogram(
    "kinetia_job_duration_seconds",
    "Duración total de los jobs por tipo de medio.",
    ["media"],
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300, 600),
    registry=REGISTRY,
)
JOBS_TOTAL = Counter("kinetia_jobs_total", "Jobs terminados por estado.", ["status"], registry=REGISTRY)
ERRORS_TOTAL = Counter("kinetia_errors_total", "Errores por código estable.", ["code"], registry=REGISTRY)
QUEUE_DEPTH = Gauge("kinetia_queue_depth", "Jobs en cola.", registry=REGISTRY)

# Crea las series conocidas para que existan en /metrics desde el arranque, en cero.
for _model in ("seg", "depth"):
    INFERENCE_SECONDS.labels(model=_model)
for _media in ("image", "video"):
    JOB_DURATION_SECONDS.labels(media=_media)
for _status in ("succeeded", "failed"):
    JOBS_TOTAL.labels(status=_status)


def record_error(code: str) -> None:
    ERRORS_TOTAL.labels(code=code).inc()
    error_code_var.set(code)
