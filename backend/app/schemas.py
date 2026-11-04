"""Esquemas de las respuestas y del campo mask_strokes (docs/CONTRATOS.md, sección 4)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_STROKES = 200
MAX_STROKE_POINTS = 2000


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassStat(_Strict):
    id: int
    name: str
    pixel_pct: float
    confidence: float


class InferenceMs(_Strict):
    segmentation: float
    depth: Optional[float]
    total: float


class AnalyzeResponse(_Strict):
    width: int
    height: int
    mask_png_base64: str
    classes: list[ClassStat]
    inference_ms: InferenceMs
    depth_png_base64: Optional[str]
    model_version: str
    model_sha256: str
    conf_threshold: float

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class AnimateAccepted(_Strict):
    job_id: str
    status: Literal["queued"]
    queue_position: int
    status_url: str


class JobResult(_Strict):
    url: str
    format: Literal["mp4", "gif"]
    width: int
    height: int
    duration_s: float
    fps: int
    size_bytes: int


class JobTimings(_Strict):
    """Un tiempo es null mientras no se haya medido."""

    segmentation: Optional[float] = None
    depth: Optional[float] = None
    render: Optional[float] = None
    encode: Optional[float] = None
    total: Optional[float] = None


class JobErrorInfo(_Strict):
    code: str
    message: str


class JobResponse(_Strict):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: Literal["queued", "segmenting", "depth", "rendering", "encoding", "done"]
    progress: float
    mood: str
    seed: int
    created_at: str
    updated_at: str
    result: Optional[JobResult]
    timings_ms: JobTimings
    error: Optional[JobErrorInfo]


class HealthResponse(_Strict):
    status: Literal["ok", "loading"]
    model_loaded: bool
    depth_loaded: bool
    model_version: Optional[str]
    model_sha256: Optional[str]
    depth_sha256: Optional[str]

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class LiveResponse(_Strict):
    status: Literal["alive"]


class ConfigLimits(_Strict):
    max_upload_bytes: int
    max_video_seconds: float
    min_side_px: int
    formats: list[str]


class ConfigClass(_Strict):
    id: int
    name: str
    label: str
    color: str


class ConfigResponse(_Strict):
    public_base_url: str
    limits: ConfigLimits
    moods: list[str]
    classes: list[ConfigClass]


class GalleryItem(_Strict):
    job_id: str
    mood: str
    result_url: str
    created_at: str


class GalleryResponse(_Strict):
    items: list[GalleryItem]


class ErrorDetail(_Strict):
    code: str
    message: str
    details: dict


class ErrorResponse(_Strict):
    error: ErrorDetail


class StrokeIn(_Strict):
    points: list[tuple[float, float]] = Field(min_length=1)
    radius: float = Field(ge=0.005, le=0.2)
    direction_deg: float = Field(ge=0.0, le=360.0)
    strength: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _points_in_range(self) -> "StrokeIn":
        for x, y in self.points:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("Las coordenadas de los puntos deben estar entre 0 y 1.")
        return self


class StrokesIn(_Strict):
    strokes: list[StrokeIn] = Field(max_length=MAX_STROKES)

    @model_validator(mode="after")
    def _point_limit(self) -> "StrokesIn":
        total = sum(len(stroke.points) for stroke in self.strokes)
        if total > MAX_STROKE_POINTS:
            raise ValueError(f"Máximo {MAX_STROKE_POINTS} puntos en total; llegaron {total}.")
        return self
