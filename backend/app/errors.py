"""Formato único de error (docs/CONTRATOS.md, sección 4)."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .observability import log_event, record_error, request_id_var

MIB = 1024 * 1024


def _mb(value: float) -> str:
    rounded = round(value / MIB, 1)
    return f"{rounded:g}"


class ApiError(StarletteHTTPException):
    """Error con código estable. Hereda de HTTPException para atravesar el parser de FastAPI."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.code = code
        self.message = message
        self.details = details or {}


def error_body(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def missing_file() -> ApiError:
    return ApiError(400, "MISSING_FILE", "No llegó ningún archivo. Selecciona una foto o un video.")


def corrupt_file(reason: str = "No se pudo leer el contenido del archivo.") -> ApiError:
    return ApiError(400, "CORRUPT_FILE", f"El archivo parece dañado. {reason}")


def unauthorized() -> ApiError:
    return ApiError(401, "UNAUTHORIZED", "Falta el token de administración o es incorrecto.")


def job_not_found(job_id: str) -> ApiError:
    return ApiError(404, "JOB_NOT_FOUND", "El trabajo solicitado no existe.", {"job_id": job_id})


def result_not_found() -> ApiError:
    return ApiError(404, "RESULT_NOT_FOUND", "El resultado no existe o ya expiró.")


def metrics_not_found() -> ApiError:
    return ApiError(404, "METRICS_NOT_FOUND", "Todavía no hay métricas del modelo disponibles.")


def job_not_ready(job_id: str, status: str) -> ApiError:
    return ApiError(409, "JOB_NOT_READY", "El trabajo aún no ha terminado.", {"job_id": job_id, "status": status})


def file_too_large(size: int, max_bytes: int) -> ApiError:
    return ApiError(
        413,
        "FILE_TOO_LARGE",
        f"El archivo pesa {_mb(size)} MB; el máximo es {_mb(max_bytes)} MB.",
        {"max_bytes": max_bytes},
    )


def unsupported_format() -> ApiError:
    return ApiError(415, "UNSUPPORTED_FORMAT", "Formato no admitido. Usa JPG, PNG, WEBP, MP4 o MOV.")


def resolution_too_low(width: int, height: int, min_side: int) -> ApiError:
    return ApiError(
        422,
        "RESOLUTION_TOO_LOW",
        f"La imagen mide {width}×{height} px; el lado menor debe tener al menos {min_side} px.",
        {"width": width, "height": height, "min_side_px": min_side},
    )


def resolution_too_high(width: int, height: int, max_pixels: int) -> ApiError:
    return ApiError(
        422,
        "RESOLUTION_TOO_HIGH",
        f"La imagen mide {width}×{height} px; el máximo es {max_pixels} píxeles en total.",
        {"width": width, "height": height, "max_pixels": max_pixels},
    )


def video_too_long(duration_s: float, max_seconds: float) -> ApiError:
    return ApiError(
        422,
        "VIDEO_TOO_LONG",
        f"El video dura {duration_s:.1f} s; el máximo es {max_seconds:g} s.",
        {"duration_s": round(duration_s, 3), "max_video_seconds": max_seconds},
    )


def invalid_params(message: str, fields: list[dict[str, str]] | None = None) -> ApiError:
    return ApiError(422, "INVALID_PARAMS", message, {"fields": fields or []})


def queue_full(retry_after_s: int, max_queue: int) -> ApiError:
    return ApiError(
        429,
        "QUEUE_FULL",
        f"Hay muchos trabajos en cola. Intenta de nuevo en unos {retry_after_s} segundos.",
        {"max_queue": max_queue, "retry_after_s": retry_after_s},
        headers={"Retry-After": str(retry_after_s)},
    )


def model_not_ready() -> ApiError:
    return ApiError(503, "MODEL_NOT_READY", "El modelo aún se está cargando. Intenta de nuevo en unos segundos.")


def internal_error(message: str = "Ocurrió un error inesperado. Intenta de nuevo.") -> ApiError:
    return ApiError(500, "INTERNAL_ERROR", message)


_FIELD_MESSAGES = {
    "literal_error": "Valor no permitido.",
    "bool_parsing": "Debe ser verdadero o falso.",
    "bool_type": "Debe ser verdadero o falso.",
    "float_parsing": "Debe ser un número.",
    "int_parsing": "Debe ser un número entero.",
    "greater_than_equal": "El valor es menor que el mínimo.",
    "less_than_equal": "El valor es mayor que el máximo.",
    "missing": "Falta este campo.",
}


def _json_error(status_code: int, code: str, message: str, details=None, headers=None) -> JSONResponse:
    return JSONResponse(error_body(code, message, details), status_code=status_code, headers=headers)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        record_error(exc.code)
        return _json_error(exc.status_code, exc.code, exc.message, exc.details, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # `input` se descarta a propósito: podría contener el archivo enviado.
        fields = []
        for item in exc.errors():
            loc = item.get("loc", ())
            if loc[:2] == ("body", "file"):
                error = missing_file()
                record_error(error.code)
                return _json_error(error.status_code, error.code, error.message)
            field = ".".join(str(part) for part in loc[1:]) or "body"
            fields.append({"field": field, "problem": _FIELD_MESSAGES.get(item.get("type", ""), "Valor inválido.")})
        names = ", ".join(entry["field"] for entry in fields)
        record_error("INVALID_PARAMS")
        return _json_error(422, "INVALID_PARAMS", f"Parámetro inválido: {names}.", {"fields": fields})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Errores que produce el propio framework, fuera de la tabla de códigos del contrato.
        detail = str(exc.detail)
        if exc.status_code == 400 and "maximum size" in detail.lower():
            error = invalid_params("Uno de los campos del formulario es demasiado grande.")
        elif exc.status_code == 400:
            error = missing_file()
        elif exc.status_code == 404:
            error = ApiError(404, "NOT_FOUND", "La ruta solicitada no existe.")
        elif exc.status_code == 405:
            error = ApiError(405, "METHOD_NOT_ALLOWED", "Método no permitido en esta ruta.")
        else:
            error = internal_error()
            error.status_code = exc.status_code
        record_error(error.code)
        return _json_error(error.status_code, error.code, error.message, error.details, exc.headers)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # La traza va al log y nunca a la respuesta.
        record_error("INTERNAL_ERROR")
        log_event("unhandled_exception", logging.ERROR, exc_info=exc, error_code="INTERNAL_ERROR")
        error = internal_error()
        headers = {"X-Request-ID": request_id_var.get() or ""}
        return _json_error(500, error.code, error.message, headers=headers)
