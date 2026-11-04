"""GET /api/v1/results/{job_id}.{ext}, con soporte de cabeceras Range (lo exige Safari en iPhone)."""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Iterator, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .. import errors
from ..observability import record_error
from ..services.job_manager import result_path
from .deps import AppServices, get_services

router = APIRouter()

MEDIA_TYPES = {"mp4": "video/mp4", "gif": "image/gif"}
CHUNK_BYTES = 256 * 1024
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class _Unsatisfiable(Exception):
    pass


def parse_range(header: Optional[str], size: int) -> Optional[tuple[int, int]]:
    """Devuelve (inicio, fin) inclusivos, o None si se debe enviar el archivo entero.

    Un Range malformado o con varios rangos se ignora, como permite el RFC 9110.
    """
    if not header:
        return None
    match = _RANGE.match(header.strip())
    if match is None:
        return None
    first, last = match.groups()
    if not first and not last:
        return None
    if not first:
        suffix = int(last)
        if suffix == 0 or size == 0:
            raise _Unsatisfiable()
        return max(0, size - suffix), size - 1
    start = int(first)
    if start >= size:
        raise _Unsatisfiable()
    end = min(int(last), size - 1) if last else size - 1
    if end < start:
        return None
    return start, end


def _iter_file(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk


def _resolve(filename: str, services: AppServices) -> tuple[Path, str]:
    job_id, dot, extension = filename.partition(".")
    if not dot or extension not in MEDIA_TYPES:
        raise errors.result_not_found()
    try:
        canonical = str(uuid.UUID(job_id))
    except ValueError as exc:
        raise errors.result_not_found() from exc
    row = services.store.get(canonical)
    if row is None or row["output_format"] != extension:
        raise errors.result_not_found()
    if row["status"] in ("queued", "running"):
        raise errors.job_not_ready(canonical, row["status"])
    path = result_path(services.settings, canonical, extension)
    if row["status"] != "succeeded" or row["expired"] or not path.is_file():
        raise errors.result_not_found()
    return path, extension


@router.head("/results/{filename}", response_class=Response, include_in_schema=False)
@router.get("/results/{filename}", response_class=Response)
def get_result(filename: str, request: Request, services: AppServices = Depends(get_services)) -> Response:
    path, extension = _resolve(filename, services)
    stat = path.stat()
    size = stat.st_size
    etag = f'"{size:x}-{int(stat.st_mtime):x}"'
    headers = {
        "Accept-Ranges": "bytes",
        "ETag": etag,
        "Content-Disposition": f'inline; filename="kinetia.{extension}"',
    }
    range_header = request.headers.get("range")
    if_range = request.headers.get("if-range")
    if if_range is not None and if_range != etag:
        range_header = None

    try:
        byte_range = parse_range(range_header, size)
    except _Unsatisfiable:
        record_error("RANGE_NOT_SATISFIABLE")
        body = errors.error_body(
            "RANGE_NOT_SATISFIABLE", "El rango solicitado está fuera del archivo.", {"size_bytes": size}
        )
        return JSONResponse(body, status_code=416, headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"})

    if byte_range is None:
        start, length, status = 0, size, 200
    else:
        start, end = byte_range
        length, status = end - start + 1, 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(length)

    if request.method == "HEAD":
        return Response(status_code=status, headers=headers, media_type=MEDIA_TYPES[extension])
    return StreamingResponse(
        _iter_file(path, start, length), status_code=status, headers=headers, media_type=MEDIA_TYPES[extension]
    )
