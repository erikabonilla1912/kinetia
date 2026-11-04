"""Carga de los modelos ONNX e inferencia de segmentación (CONTRATOS, secciones 2 y 3).

Los modelos se cargan una sola vez. `InferenceSession.run` es seguro entre hilos y libera el
GIL, así que el endpoint y el hilo de jobs comparten las sesiones sin ningún candado.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from ..config import CLASS_NAMES, DUMMY_VERSION, NUM_CLASSES, Settings
from ..observability import INFERENCE_SECONDS, log_event
from .depth import DepthModel
from .preprocess import SEG_INPUT_SIZE, preprocess_seg


class ModelStartupError(RuntimeError):
    """El modelo no es válido o su uso no está permitido: la API no debe arrancar."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_session(path: Path, intra_op_threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Sin espera activa: el hilo de render y FFmpeg comparten estos mismos núcleos.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def postprocess_logits(logits: np.ndarray, conf_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Logits (7, H, W) -> (clases uint8 (H, W), max_prob float32 (H, W)). Pasos 1 a 3 del contrato."""
    probabilities = softmax(logits, axis=0)
    classes = probabilities.argmax(axis=0).astype(np.uint8)
    max_prob = probabilities.max(axis=0).astype(np.float32)
    classes[max_prob < conf_threshold] = 0
    return classes, max_prob


def class_stats(mask: np.ndarray, classes_grid: np.ndarray, max_prob: np.ndarray) -> list[dict[str, Any]]:
    """`pixel_pct` sale de la máscara final; `confidence` es el promedio de max_prob por clase."""
    counts = np.bincount(mask.ravel(), minlength=NUM_CLASSES)
    total = int(counts.sum())
    stats = []
    for class_id, count in enumerate(counts[:NUM_CLASSES]):
        if count == 0:
            continue
        assigned = classes_grid == class_id
        confidence = float(max_prob[assigned].mean()) if assigned.any() else 0.0
        stats.append(
            {
                "id": class_id,
                "name": CLASS_NAMES[class_id],
                "pixel_pct": round(100.0 * int(count) / total, 4),
                "confidence": round(confidence, 4),
            }
        )
    stats.sort(key=lambda item: item["pixel_pct"], reverse=True)
    return stats


@dataclass
class SegmentationResult:
    mask: np.ndarray
    classes: list[dict[str, Any]]
    elapsed_ms: float


class ModelStore:
    """Dueño del manifiesto y de las dos sesiones ONNX."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ready = threading.Event()
        self._seg_session: ort.InferenceSession | None = None
        self._depth: DepthModel | None = None
        self.manifest: dict[str, Any] = {}
        self.version: str | None = None
        self.seg_sha256: str | None = None
        self.depth_sha256: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def preflight(self) -> None:
        """Comprobaciones rápidas previas a abrir el puerto. Lanza ModelStartupError si fallan."""
        settings = self._settings
        path = settings.manifest_path
        if not path.is_file():
            raise ModelStartupError(f"No existe {path}. Revisa MODELS_DIR.")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            version = manifest["version"]
            files = manifest["files"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ModelStartupError(f"manifest.json inválido: {exc}") from exc

        if version == DUMMY_VERSION and not settings.allow_dummy_model:
            raise ModelStartupError(
                f"Se encontró el modelo '{DUMMY_VERSION}' y ALLOW_DUMMY_MODEL no es true. "
                "La API se niega a arrancar con pesos aleatorios."
            )
        if list(manifest.get("classes", [])) != list(CLASS_NAMES):
            raise ModelStartupError("Las clases de manifest.json no coinciden con las del contrato.")
        if list(manifest.get("input_size", [])) != [SEG_INPUT_SIZE, SEG_INPUT_SIZE]:
            raise ModelStartupError("input_size de manifest.json no coincide con el contrato.")

        hashes = {}
        for model_path in (settings.seg_model_path, settings.depth_model_path):
            if not model_path.is_file():
                raise ModelStartupError(f"No existe el modelo {model_path}.")
            actual = sha256_file(model_path)
            expected = (files.get(model_path.name) or {}).get("sha256")
            if expected != actual:
                raise ModelStartupError(
                    f"El SHA-256 de {model_path.name} ({actual}) no coincide con manifest.json ({expected})."
                )
            hashes[model_path] = actual

        self.manifest = manifest
        self.version = version
        self.seg_sha256 = hashes[settings.seg_model_path]
        self.depth_sha256 = hashes[settings.depth_model_path]
        if version == DUMMY_VERSION:
            log_event("dummy_model_allowed", logging.WARNING, version=version)

    def load(self) -> None:
        """Crea las sesiones y comprueba su firma. Marca el estado listo al final, de una vez."""
        settings = self._settings
        started = time.perf_counter()
        try:
            seg_session = make_session(settings.seg_model_path, settings.ort_intra_op_threads)
            depth_session = make_session(settings.depth_model_path, settings.ort_intra_op_threads)
        except Exception as exc:  # ONNX Runtime usa excepciones propias para archivos inválidos.
            raise ModelStartupError(f"No se pudo abrir un modelo ONNX: {exc}") from exc
        self._check_seg_signature(seg_session)
        try:
            depth = DepthModel(depth_session)
            size = depth_session.get_inputs()[0].shape
            depth.infer(np.zeros((size[2], size[3], 3), dtype=np.uint8))
        except Exception as exc:
            raise ModelStartupError(f"El modelo de profundidad no cumple la firma: {exc}") from exc
        self._seg_session = seg_session
        self._depth = depth
        self._ready.set()
        log_event(
            "models_loaded",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 1),
            version=self.version,
            model_sha256=self.seg_sha256,
            depth_sha256=self.depth_sha256,
        )

    @staticmethod
    def _check_seg_signature(session: ort.InferenceSession) -> None:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if [node.name for node in inputs] != ["input"] or outputs[0].name != "logits":
            raise ModelStartupError("El modelo de segmentación debe tener la entrada `input` y la salida `logits`.")
        probe = np.zeros((1, 3, SEG_INPUT_SIZE, SEG_INPUT_SIZE), dtype=np.float32)
        try:
            logits = session.run(["logits"], {"input": probe})[0]
        except Exception as exc:
            raise ModelStartupError(f"El modelo de segmentación no acepta la entrada del contrato: {exc}") from exc
        expected = (1, NUM_CLASSES, SEG_INPUT_SIZE, SEG_INPUT_SIZE)
        if logits.shape != expected:
            raise ModelStartupError(f"Salida `logits` con forma {logits.shape}; se esperaba {expected}.")

    def segment(self, image_bgr: np.ndarray) -> SegmentationResult:
        """Máscara uint8 del tamaño de la imagen, con el posproceso del contrato. Tiempo medido."""
        if self._seg_session is None:
            raise RuntimeError("El modelo de segmentación no está cargado")
        started = time.perf_counter()
        tensor = preprocess_seg(image_bgr)
        logits = self._seg_session.run(["logits"], {"input": tensor})[0][0]
        grid, max_prob = postprocess_logits(logits, self._settings.conf_threshold)
        height, width = image_bgr.shape[:2]
        mask = cv2.resize(grid, (width, height), interpolation=cv2.INTER_NEAREST)
        stats = class_stats(mask, grid, max_prob)
        elapsed = time.perf_counter() - started
        INFERENCE_SECONDS.labels(model="seg").observe(elapsed)
        return SegmentationResult(mask=mask, classes=stats, elapsed_ms=elapsed * 1000.0)

    def estimate_depth(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        if self._depth is None:
            raise RuntimeError("El modelo de profundidad no está cargado")
        return self._depth.predict(image_bgr)
