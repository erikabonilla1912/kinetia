"""Configuración de la API: variables de entorno de docs/CONTRATOS.md, sección 6."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

API_PREFIX = "/api/v1"
DUMMY_VERSION = "dummy-dev"
NUM_CLASSES = 7

# (id, name, label, color). Sección 1 del contrato: el orden es fijo.
CLASS_TABLE: tuple[tuple[int, str, str, str], ...] = (
    (0, "fondo", "Fondo", "#3A3F4B"),
    (1, "cielo", "Cielo", "#4FC3F7"),
    (2, "agua", "Agua", "#2979FF"),
    (3, "vegetacion", "Vegetación", "#66BB6A"),
    (4, "fuego_luz", "Fuego / luz", "#FF7043"),
    (5, "persona", "Persona", "#EC407A"),
    (6, "animal", "Animal", "#FFCA28"),
)
CLASS_NAMES: tuple[str, ...] = tuple(row[1] for row in CLASS_TABLE)
MOODS: tuple[str, ...] = ("calma", "epico", "magico", "nostalgico")
UPLOAD_FORMATS: tuple[str, ...] = ("jpg", "png", "webp", "mp4", "mov")

IMAGE_FPS = 24
GIF_FPS = 12
GIF_MAX_SIDE = 480


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False, protected_namespaces=())

    models_dir: Path = Path("/models")
    seg_model_file: str = "kinetia_seg.onnx"
    results_dir: Path = Path("/data/results")
    db_path: Path = Path("/data/db/kinetia.db")
    tmp_dir: Path = Path("/shared-tmp/api")
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_upload_bytes: int = Field(15728640, gt=0)
    max_video_seconds: float = Field(10, gt=0)
    max_frames: int = Field(300, gt=0)
    min_side_px: int = Field(256, gt=0)
    max_pixels: int = Field(40000000, gt=0)
    work_max_side: int = Field(1280, gt=0)
    video_infer_every_n: int = Field(5, gt=0)
    night_luma_threshold: float = Field(60, ge=0, le=255)
    forest_min_pct: float = Field(25, ge=0, le=100)
    max_queue: int = Field(8, gt=0)
    result_ttl_hours: float = Field(96, gt=0)
    ort_intra_op_threads: int = Field(4, gt=0)
    public_base_url: str = "http://localhost:8080"
    allow_dummy_model: bool = False
    log_level: str = "INFO"
    admin_token: str = ""

    @property
    def seg_model_path(self) -> Path:
        return self.models_dir / self.seg_model_file

    @property
    def manifest_path(self) -> Path:
        return self.models_dir / "manifest.json"

    @property
    def metrics_path(self) -> Path:
        return self.models_dir / "metrics.json"
