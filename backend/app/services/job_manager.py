"""Jobs de animación: un solo hilo de trabajo, cola en memoria y estado en SQLite WAL."""
from __future__ import annotations

import importlib
import json
import logging
import math
import os
import shutil
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from animation_engine.types import RenderConfig

from .. import errors
from ..config import API_PREFIX, IMAGE_FPS, Settings
from ..observability import JOB_DURATION_SECONDS, JOBS_TOTAL, QUEUE_DEPTH, job_id_var, log_event, record_error
from . import encoder
from .brush import rasterize_strokes, strokes_from_json
from .inference import ModelStore
from .preprocess import resize_to_work, work_size
from .validation import validate_media

PROGRESS_MIN_INTERVAL_S = 0.25
CLEANUP_INTERVAL_S = 600
TMP_MAX_AGE_S = 6 * 3600
DEFAULT_RETRY_AFTER_S = 30
RENDER_PROGRESS_START = 0.1
RENDER_PROGRESS_END = 0.95


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(moment: datetime) -> str:
    """ISO 8601 UTC con milisegundos. El ancho fijo permite ordenar y comparar como texto."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class JobFailure(Exception):
    """Fallo del job con un código estable y un mensaje apto para el usuario."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _Interrupted(Exception):
    """La API se está apagando con este job en curso."""


class JobStore:
    """Persistencia de jobs. Una conexión protegida por un candado: las operaciones son breves."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    media TEXT NOT NULL,
                    mood TEXT NOT NULL,
                    output_format TEXT NOT NULL,
                    duration_s REAL NOT NULL,
                    seed INTEGER NOT NULL,
                    brush_mode TEXT NOT NULL,
                    strokes_json TEXT,
                    gallery_opt_in INTEGER NOT NULL,
                    gallery_removed INTEGER NOT NULL DEFAULT 0,
                    expired INTEGER NOT NULL DEFAULT 0,
                    input_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    result_json TEXT,
                    timings_json TEXT,
                    error_json TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_gallery ON jobs (gallery_opt_in, status, created_at)"
            )

    def journal_mode(self) -> str:
        with self._lock:
            return self._conn.execute("PRAGMA journal_mode").fetchone()[0]

    def insert(self, **fields: Any) -> None:
        columns = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with self._lock, self._conn:
            self._conn.execute(f"INSERT INTO jobs ({columns}) VALUES ({marks})", tuple(fields.values()))

    def update(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = to_iso(utc_now())
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._lock, self._conn:
            self._conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ?", (*fields.values(), job_id))

    def get(self, job_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()

    def mark_interrupted(self) -> int:
        """Lo que estaba en cola o en curso al reiniciar ya no se ejecutará."""
        error = json.dumps({"code": "JOB_INTERRUPTED", "message": "El servidor se reinició antes de terminar el trabajo."})
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE jobs SET status = 'failed', error_json = ?, updated_at = ?, input_path = NULL "
                "WHERE status IN ('queued', 'running')",
                (error, to_iso(utc_now())),
            )
            return cursor.rowcount

    def gallery(self, limit: int, after: Optional[str]) -> list[sqlite3.Row]:
        query = (
            "SELECT * FROM jobs WHERE gallery_opt_in = 1 AND gallery_removed = 0 AND expired = 0 "
            "AND status = 'succeeded'"
        )
        parameters: list[Any] = []
        if after is not None:
            query += " AND created_at > ?"
            parameters.append(after)
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(limit)
        with self._lock:
            return self._conn.execute(query, parameters).fetchall()

    def finished_before(self, moment: str) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM jobs WHERE status IN ('succeeded', 'failed') AND expired = 0 AND updated_at < ?",
                (moment,),
            ).fetchall()

    def delete_expired_before(self, moment: str) -> int:
        with self._lock, self._conn:
            return self._conn.execute(
                "DELETE FROM jobs WHERE expired = 1 AND updated_at < ?", (moment,)
            ).rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def result_extension(row: sqlite3.Row) -> str:
    return row["output_format"]


def result_path(settings: Settings, job_id: str, extension: str) -> Path:
    return settings.results_dir / f"{job_id}.{extension}"


def result_url(job_id: str, extension: str) -> str:
    return f"{API_PREFIX}/results/{job_id}.{extension}"


def load_engine() -> Any:
    """Importa el motor real. Si aún no existe o está incompleto, el job falla con error claro."""
    try:
        module = importlib.import_module("animation_engine")
    except Exception as exc:
        raise JobFailure("ENGINE_UNAVAILABLE", "El motor de animación no está disponible en este servidor.") from exc
    for name in ("get_mood", "animate_image", "animate_video"):
        if not hasattr(module, name):
            raise JobFailure("ENGINE_UNAVAILABLE", "El motor de animación no está disponible en este servidor.")
    return module


@dataclass
class _Timings:
    segmentation: Optional[float] = None
    depth: Optional[float] = None
    render: Optional[float] = None
    encode: Optional[float] = None
    total: Optional[float] = None

    def as_json(self) -> str:
        return json.dumps(
            {name: (round(value, 1) if value is not None else None) for name, value in self.__dict__.items()}
        )


class JobManager:
    def __init__(
        self,
        settings: Settings,
        store: JobStore,
        models: ModelStore,
        engine_provider: Callable[[], Any] = load_engine,
    ) -> None:
        self._settings = settings
        self._store = store
        self._models = models
        self._engine_provider = engine_provider
        self._condition = threading.Condition()
        self._queue: deque[str] = deque()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._janitor: Optional[threading.Thread] = None
        self._average_job_s: Optional[float] = None

    @property
    def queued(self) -> int:
        with self._condition:
            return len(self._queue)

    def start(self) -> None:
        """Marca lo interrumpido, limpia y arranca los hilos."""
        interrupted = self._store.mark_interrupted()
        if interrupted:
            log_event("jobs_interrupted", logging.WARNING, count=interrupted, error_code="JOB_INTERRUPTED")
        self._settings.results_dir.mkdir(parents=True, exist_ok=True)
        self._settings.tmp_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self._settings.tmp_dir / "jobs", ignore_errors=True)
        shutil.rmtree(self._settings.tmp_dir / "uploads", ignore_errors=True)
        self.cleanup()
        QUEUE_DEPTH.set(0)
        self._stop.clear()
        self._worker = threading.Thread(target=self._work_loop, name="kinetia-job-worker", daemon=True)
        self._worker.start()
        self._janitor = threading.Thread(target=self._janitor_loop, name="kinetia-janitor", daemon=True)
        self._janitor.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        for thread in (self._worker, self._janitor):
            if thread is not None:
                thread.join(timeout=timeout)

    def retry_after_seconds(self) -> int:
        """Estimación basada en la duración medida de los jobs anteriores."""
        per_job = self._average_job_s if self._average_job_s is not None else DEFAULT_RETRY_AFTER_S
        return int(min(600, max(5, math.ceil(per_job))))

    def ensure_capacity(self) -> None:
        with self._condition:
            if len(self._queue) >= self._settings.max_queue:
                raise errors.queue_full(self.retry_after_seconds(), self._settings.max_queue)

    def submit(
        self,
        upload_path: Path,
        media: str,
        mood: str,
        output_format: str,
        duration_s: float,
        seed: int,
        brush_mode: str,
        strokes_json: Optional[str],
        gallery_opt_in: bool,
    ) -> tuple[str, int]:
        """Encola el job y devuelve (job_id, posición). La posición 1 es el siguiente en ejecutarse."""
        job_id = str(uuid.uuid4())
        with self._condition:
            if len(self._queue) >= self._settings.max_queue:
                raise errors.queue_full(self.retry_after_seconds(), self._settings.max_queue)
            job_dir = self._settings.tmp_dir / "jobs" / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            input_path = job_dir / "input"
            os.replace(upload_path, input_path)
            now = to_iso(utc_now())
            self._store.insert(
                job_id=job_id,
                status="queued",
                stage="queued",
                progress=0.0,
                media=media,
                mood=mood,
                output_format=output_format,
                duration_s=duration_s,
                seed=seed,
                brush_mode=brush_mode,
                strokes_json=strokes_json,
                gallery_opt_in=int(gallery_opt_in),
                input_path=str(input_path),
                created_at=now,
                updated_at=now,
                timings_json=_Timings().as_json(),
            )
            self._queue.append(job_id)
            position = len(self._queue)
            QUEUE_DEPTH.set(position)
            self._condition.notify_all()
        log_event("job_queued", job_id=job_id, media=media, output_format=output_format)
        return job_id, position

    def _work_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stop.is_set():
                    self._condition.wait()
                if self._stop.is_set():
                    return
                job_id = self._queue.popleft()
                QUEUE_DEPTH.set(len(self._queue))
            job_id_var.set(job_id)
            try:
                self._execute(job_id)
            except Exception:  # El hilo de trabajo no puede morir por un job.
                log_event("job_worker_error", logging.ERROR, exc_info=True, error_code="INTERNAL_ERROR")
            finally:
                job_id_var.set(None)

    def _execute(self, job_id: str) -> None:
        row = self._store.get(job_id)
        if row is None:
            return
        started = time.perf_counter()
        timings = _Timings()
        self._store.update(job_id, status="running", stage="segmenting", progress=0.01)
        log_event("job_started", job_id=job_id)
        media = row["media"]
        try:
            result = self._run(row, timings)
        except _Interrupted:
            self._finish_failed(row, timings, started, "JOB_INTERRUPTED", "El servidor se reinició antes de terminar el trabajo.")
        except JobFailure as failure:
            self._finish_failed(row, timings, started, failure.code, failure.message)
        except encoder.FFmpegMissing as exc:
            self._finish_failed(row, timings, started, "ENCODER_UNAVAILABLE", "FFmpeg no está disponible en este servidor.", exc)
        except Exception as exc:
            self._finish_failed(row, timings, started, "INTERNAL_ERROR", "No se pudo generar la animación.", exc)
        else:
            self._remove_job_files(job_id)
            elapsed = time.perf_counter() - started
            timings.total = elapsed * 1000.0
            self._store.update(
                job_id,
                status="succeeded",
                stage="done",
                progress=1.0,
                result_json=json.dumps(result),
                timings_json=timings.as_json(),
                input_path=None,
            )
            JOBS_TOTAL.labels(status="succeeded").inc()
            JOB_DURATION_SECONDS.labels(media=media).observe(elapsed)
            previous = self._average_job_s
            self._average_job_s = elapsed if previous is None else 0.7 * previous + 0.3 * elapsed
            log_event("job_succeeded", job_id=job_id, duration_ms=round(elapsed * 1000.0, 1))

    def _remove_job_files(self, job_id: str) -> None:
        """Se llama antes de publicar el estado final: quien lo lea ya no encuentra temporales."""
        shutil.rmtree(self._settings.tmp_dir / "jobs" / job_id, ignore_errors=True)
        for leftover in self._settings.results_dir.glob(f".{job_id}.*"):
            leftover.unlink(missing_ok=True)

    def _finish_failed(
        self,
        row: sqlite3.Row,
        timings: _Timings,
        started: float,
        code: str,
        message: str,
        exc: Optional[BaseException] = None,
    ) -> None:
        job_id = row["job_id"]
        self._remove_job_files(job_id)
        elapsed = time.perf_counter() - started
        timings.total = elapsed * 1000.0
        self._store.update(
            job_id,
            status="failed",
            error_json=json.dumps({"code": code, "message": message}),
            timings_json=timings.as_json(),
            input_path=None,
        )
        JOBS_TOTAL.labels(status="failed").inc()
        JOB_DURATION_SECONDS.labels(media=row["media"]).observe(elapsed)
        record_error(code)
        log_event(
            "job_failed",
            logging.ERROR,
            exc_info=exc,
            job_id=job_id,
            error_code=code,
            duration_ms=round(elapsed * 1000.0, 1),
        )

    def _progress_reporter(self, job_id: str) -> Callable[[float], None]:
        state = {"last": 0.0, "value": RENDER_PROGRESS_START}

        def report(fraction: float) -> None:
            fraction = min(1.0, max(0.0, float(fraction)))
            value = RENDER_PROGRESS_START + (RENDER_PROGRESS_END - RENDER_PROGRESS_START) * fraction
            now = time.monotonic()
            if value > state["value"] and now - state["last"] >= PROGRESS_MIN_INTERVAL_S:
                state["last"] = now
                state["value"] = value
                self._store.update(job_id, progress=round(value, 4))

        return report

    def _run(self, row: sqlite3.Row, timings: _Timings) -> dict[str, Any]:
        job_id = row["job_id"]
        engine = self._engine_provider()
        encoder.find_tool("ffmpeg")
        media = validate_media(Path(row["input_path"]), self._settings)
        strokes = strokes_from_json(row["strokes_json"])
        mood = engine.get_mood(row["mood"])
        report = self._progress_reporter(job_id)
        source = None

        if media.media == "image":
            image = resize_to_work(media.image_bgr, self._settings.work_max_side)
            segmentation = self._models.segment(image)
            timings.segmentation = segmentation.elapsed_ms
            self._log_brush(job_id, strokes, image.shape[1], image.shape[0])
            config = RenderConfig(
                mood=mood,
                duration_s=row["duration_s"],
                fps=IMAGE_FPS,
                seed=row["seed"],
                strokes=strokes,
                brush_mode=row["brush_mode"],
            )
            # Sin modelo de profundidad: el motor no la usa (sin parallax, contrato v1.7).
            frames = engine.animate_image(image, segmentation.mask, None, config, report)
            width, height = image.shape[1], image.shape[0]
            fps = IMAGE_FPS
            expected = round(config.duration_s * fps)
            segment_seconds = lambda: 0.0  # noqa: E731
        else:
            info = media.video
            width, height = work_size(info.width, info.height, self._settings.work_max_side)
            # Se remuestrea a fps entero; nunca más de MAX_FRAMES en total.
            fps = max(1, min(round(info.fps) or IMAGE_FPS, math.floor(self._settings.max_frames / info.duration_s)))
            expected = max(1, min(self._settings.max_frames, round(info.duration_s * fps)))
            self._log_brush(job_id, strokes, width, height)
            config = RenderConfig(
                mood=mood,
                duration_s=expected / fps,
                fps=fps,
                seed=row["seed"],
                strokes=strokes,
                brush_mode=row["brush_mode"],
            )
            seg_clock = {"seconds": 0.0}

            def segment(frame: np.ndarray) -> np.ndarray:
                segmentation = self._models.segment(frame)
                seg_clock["seconds"] += segmentation.elapsed_ms / 1000.0
                return segmentation.mask

            source = encoder.iter_video_frames(Path(row["input_path"]), width, height, fps, self._settings.max_frames)
            frames = engine.animate_video(source, segment, self._settings.video_infer_every_n, config, None, report)
            segment_seconds = lambda: seg_clock["seconds"]  # noqa: E731

        self._store.update(job_id, stage="rendering", progress=RENDER_PROGRESS_START, timings_json=timings.as_json())
        extension = row["output_format"]
        partial = self._settings.results_dir / f".{job_id}.{extension}.part"
        sink = encoder.FrameEncoder(partial, extension, width, height, fps)
        try:
            written, render_s, encode_s = self._pump(frames, sink, expected, report)
            if written == 0:
                raise JobFailure("INTERNAL_ERROR", "El motor de animación no produjo ningún frame.")
            self._store.update(job_id, stage="encoding", progress=RENDER_PROGRESS_END)
            started_close = time.perf_counter()
            sink.close()
            encode_s += time.perf_counter() - started_close
        except BaseException:
            sink.abort()
            close = getattr(frames, "close", None)
            if close is not None:
                close()
            if source is not None:
                source.close()  # También termina el FFmpeg que lee el video.
            raise

        spent_on_segmentation = segment_seconds()
        if media.media == "video":
            timings.segmentation = spent_on_segmentation * 1000.0
        timings.render = max(0.0, render_s - spent_on_segmentation) * 1000.0
        timings.encode = encode_s * 1000.0

        final = result_path(self._settings, job_id, extension)
        os.replace(partial, final)
        out_width, out_height = sink.output_size
        return {
            "url": result_url(job_id, extension),
            "format": extension,
            "width": out_width,
            "height": out_height,
            "duration_s": round(written / fps, 3),
            "fps": sink.output_fps,
            "size_bytes": final.stat().st_size,
        }

    def _pump(
        self,
        frames: Iterable[np.ndarray],
        sink: encoder.FrameEncoder,
        expected: int,
        report: Callable[[float], None],
    ) -> tuple[int, float, float]:
        """Pasa los frames del motor al codificador midiendo por separado cada lado."""
        iterator = iter(frames)
        render_s = encode_s = 0.0
        written = 0
        while True:
            if self._stop.is_set():
                raise _Interrupted()
            started = time.perf_counter()
            try:
                frame = next(iterator)
            except StopIteration:
                render_s += time.perf_counter() - started
                return written, render_s, encode_s
            render_s += time.perf_counter() - started
            started = time.perf_counter()
            sink.write(frame)
            encode_s += time.perf_counter() - started
            written += 1
            report(written / max(expected, 1))

    @staticmethod
    def _log_brush(job_id: str, strokes: tuple, width: int, height: int) -> None:
        if not strokes:
            return
        raster = rasterize_strokes(strokes, width, height)
        log_event(
            "brush_rasterized",
            job_id=job_id,
            strokes=len(strokes),
            coverage_pct=round(100.0 * float((raster.coverage > 0).mean()), 2),
        )

    def cleanup(self) -> None:
        """Borra resultados vencidos, filas muy antiguas y temporales huérfanos."""
        settings = self._settings
        ttl = timedelta(hours=settings.result_ttl_hours)
        now = utc_now()
        removed = 0
        for row in self._store.finished_before(to_iso(now - ttl)):
            result_path(settings, row["job_id"], row["output_format"]).unlink(missing_ok=True)
            self._store.update(row["job_id"], expired=1)
            removed += 1
        self._store.delete_expired_before(to_iso(now - 2 * ttl))

        oldest_result = time.time() - ttl.total_seconds()
        if settings.results_dir.is_dir():
            for path in settings.results_dir.iterdir():
                if path.is_file() and path.stat().st_mtime < oldest_result:
                    path.unlink(missing_ok=True)
        oldest_tmp = time.time() - TMP_MAX_AGE_S
        for folder in (settings.tmp_dir / "uploads", settings.tmp_dir / "jobs"):
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if path.stat().st_mtime < oldest_tmp:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
        if removed:
            log_event("results_expired", count=removed)

    def _janitor_loop(self) -> None:
        while not self._stop.wait(CLEANUP_INTERVAL_S):
            try:
                self.cleanup()
            except Exception:
                log_event("cleanup_failed", logging.ERROR, exc_info=True)
