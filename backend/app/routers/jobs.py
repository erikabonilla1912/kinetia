"""GET /api/v1/jobs/{id}."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from fastapi import APIRouter, Depends

from .. import errors
from ..schemas import JobResponse
from .deps import AppServices, get_services

router = APIRouter()


def job_to_response(row: sqlite3.Row) -> dict[str, Any]:
    failed = row["status"] == "failed"
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "stage": row["stage"],
        "progress": row["progress"],
        "mood": row["mood"],
        "seed": row["seed"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "result": None if failed or not row["result_json"] else json.loads(row["result_json"]),
        "timings_ms": json.loads(row["timings_json"]) if row["timings_json"] else {},
        "error": json.loads(row["error_json"]) if failed and row["error_json"] else None,
    }


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, services: AppServices = Depends(get_services)) -> dict[str, Any]:
    row = services.store.get(job_id)
    if row is None:
        raise errors.job_not_found(job_id)
    return job_to_response(row)
