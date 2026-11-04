"""/config, /model/metrics, /metrics, logs, pincel, preproceso y restricciones de código."""
from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from api_fakes import API, assert_error, image_bytes, make_settings, post_file, running_client, wait_job
from conftest import BACKEND_DIR, REPO_ROOT

from app.errors import ApiError
from app.observability import REGISTRY, JsonFormatter, job_id_var, log_event, request_id_var
from app.services.brush import parse_strokes, rasterize_strokes
from app.services.preprocess import preprocess_seg, work_size


def stroke(points=None, radius=0.05, direction_deg=90.0, strength=0.5):
    return {
        "points": points if points is not None else [[0.5, 0.5]],
        "radius": radius,
        "direction_deg": direction_deg,
        "strength": strength,
    }


def strokes_json(*items):
    return json.dumps({"strokes": list(items)})


# Configuración pública


def test_config_exact_shape(tmp_path, dummy_models, engine):
    settings = make_settings(tmp_path, dummy_models, public_base_url="http://192.168.0.10:8080")
    with running_client(settings, lambda: engine) as client:
        body = client.get(f"{API}/config").json()
    assert body == {
        "public_base_url": "http://192.168.0.10:8080",
        "limits": {
            "max_upload_bytes": 15728640,
            "max_video_seconds": 10,
            "min_side_px": 256,
            "formats": ["jpg", "png", "webp", "mp4", "mov"],
        },
        "moods": ["calma", "epico", "magico", "nostalgico"],
        "classes": [
            {"id": 0, "name": "fondo", "label": "Fondo", "color": "#3A3F4B"},
            {"id": 1, "name": "cielo", "label": "Cielo", "color": "#4FC3F7"},
            {"id": 2, "name": "agua", "label": "Agua", "color": "#2979FF"},
            {"id": 3, "name": "vegetacion", "label": "Vegetación", "color": "#66BB6A"},
            {"id": 4, "name": "fuego_luz", "label": "Fuego / luz", "color": "#FF7043"},
            {"id": 5, "name": "persona", "label": "Persona", "color": "#EC407A"},
            {"id": 6, "name": "animal", "label": "Animal", "color": "#FFCA28"},
        ],
    }


def test_env_example_lists_every_contract_variable():
    names = {
        line.split("=", 1)[0].strip()
        for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    assert names == {
        "MODELS_DIR", "SEG_MODEL_FILE", "DEPTH_MODEL_FILE", "RESULTS_DIR", "DB_PATH", "TMP_DIR", "CONF_THRESHOLD",
        "MAX_UPLOAD_BYTES", "MAX_VIDEO_SECONDS", "MAX_FRAMES", "MIN_SIDE_PX", "MAX_PIXELS", "WORK_MAX_SIDE",
        "VIDEO_INFER_EVERY_N", "NIGHT_LUMA_THRESHOLD", "FOREST_MIN_PCT", "MAX_QUEUE", "RESULT_TTL_HOURS",
        "ORT_INTRA_OP_THREADS", "PUBLIC_BASE_URL", "ALLOW_DUMMY_MODEL", "LOG_LEVEL", "ADMIN_TOKEN",
    }
    assert "ALLOW_DUMMY_MODEL=false" in (REPO_ROOT / ".env.example").read_text(encoding="utf-8")


def test_settings_read_environment_and_contract_defaults(monkeypatch):
    from app.config import Settings

    for name in ("MODELS_DIR", "MAX_QUEUE", "ALLOW_DUMMY_MODEL", "CONF_THRESHOLD"):
        monkeypatch.delenv(name, raising=False)
    defaults = Settings()
    assert defaults.models_dir == Path("/models")
    assert defaults.max_queue == 8
    assert defaults.allow_dummy_model is False
    assert defaults.conf_threshold == 0.5
    assert defaults.max_upload_bytes == 15728640
    monkeypatch.setenv("MAX_QUEUE", "3")
    monkeypatch.setenv("ALLOW_DUMMY_MODEL", "true")
    configured = Settings()
    assert configured.max_queue == 3 and configured.allow_dummy_model is True


# Métricas del modelo


def test_model_metrics_are_literal_file_contents(tmp_path, dummy_models, engine):
    import shutil

    models = tmp_path / "m"
    shutil.copytree(dummy_models, models)
    literal = b'{ "miou":   0.123456789012345678, "orden": [3,2,1] }\n'
    (models / "metrics.json").write_bytes(literal)
    with running_client(make_settings(tmp_path, models), lambda: engine) as client:
        response = client.get(f"{API}/model/metrics")
        assert response.status_code == 200
        assert response.content == literal
        assert response.headers["content-type"].startswith("application/json")
        (models / "metrics.json").unlink()
        assert_error(client.get(f"{API}/model/metrics"), 404, "METRICS_NOT_FOUND")


def test_dummy_metrics_have_schema_and_are_marked(client, dummy_models):
    body = client.get(f"{API}/model/metrics").json()
    assert body["dummy"] is True
    assert set(body) - {"dummy"} == {
        "model_sha256", "dataset_manifest_sha256", "split", "num_images", "evaluated_at", "miou", "dice_macro",
        "pixel_accuracy", "per_class", "confusion_matrix", "classes",
    }
    assert body["miou"] is None and body["dice_macro"] is None and body["pixel_accuracy"] is None
    assert body["num_images"] == 0
    assert set(body["per_class"]) == set(body["classes"]) and len(body["classes"]) == 7
    for entry in body["per_class"].values():
        assert entry == {"iou": None, "dice": None, "support_pixels": 0}
    assert np.array(body["confusion_matrix"]).shape == (7, 7)
    assert np.array(body["confusion_matrix"]).sum() == 0


# Prometheus


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_prometheus_exposes_exact_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    declared = {}
    for line in text.splitlines():
        if line.startswith("# TYPE"):
            _, _, name, kind = line.split()
            declared[name] = kind
    assert declared == {
        "kinetia_inference_seconds": "histogram",
        "kinetia_job_duration_seconds": "histogram",
        "kinetia_jobs_total": "counter",
        "kinetia_errors_total": "counter",
        "kinetia_queue_depth": "gauge",
    }
    for label in ('model="seg"', 'model="depth"', 'media="image"', 'media="video"', 'status="succeeded"', 'status="failed"'):
        assert label in text


def test_metrics_count_real_activity(client, fake_encoder):
    seg_before = sample("kinetia_inference_seconds_count", model="seg")
    depth_before = sample("kinetia_inference_seconds_count", model="depth")
    assert post_file(client, "/analyze", image_bytes(), include_depth="true").status_code == 200
    assert sample("kinetia_inference_seconds_count", model="seg") == seg_before + 1
    assert sample("kinetia_inference_seconds_count", model="depth") == depth_before + 1

    errors_before = sample("kinetia_errors_total", code="UNSUPPORTED_FORMAT")
    post_file(client, "/analyze", b"texto")
    assert sample("kinetia_errors_total", code="UNSUPPORTED_FORMAT") == errors_before + 1

    invalid_before = sample("kinetia_errors_total", code="INVALID_PARAMS")
    post_file(client, "/animate", image_bytes(), mood="x")
    assert sample("kinetia_errors_total", code="INVALID_PARAMS") == invalid_before + 1

    succeeded_before = sample("kinetia_jobs_total", status="succeeded")
    duration_before = sample("kinetia_job_duration_seconds_count", media="image")
    job_id = post_file(client, "/animate", image_bytes(), duration_s="2").json()["job_id"]
    assert wait_job(client, job_id)["status"] == "succeeded"
    assert sample("kinetia_jobs_total", status="succeeded") == succeeded_before + 1
    assert sample("kinetia_job_duration_seconds_count", media="image") == duration_before + 1


def test_failed_job_is_counted(settings):
    from app.services.job_manager import JobFailure

    def unavailable():
        raise JobFailure("ENGINE_UNAVAILABLE", "Sin motor")

    failed_before = sample("kinetia_jobs_total", status="failed")
    code_before = sample("kinetia_errors_total", code="ENGINE_UNAVAILABLE")
    with running_client(settings, unavailable) as client:
        wait_job(client, post_file(client, "/animate", image_bytes()).json()["job_id"])
    assert sample("kinetia_jobs_total", status="failed") == failed_before + 1
    assert sample("kinetia_errors_total", code="ENGINE_UNAVAILABLE") == code_before + 1


# Logs


def format_record(event, **fields):
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    logger = logging.getLogger("kinetia")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        log_event(event, **fields)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    return json.loads(JsonFormatter().format(records[0]))


def test_log_line_has_contract_fields():
    token_request = request_id_var.set("req-1")
    token_job = job_id_var.set("job-1")
    try:
        entry = format_record("evento", duration_ms=12.5, error_code="QUEUE_FULL")
    finally:
        request_id_var.reset(token_request)
        job_id_var.reset(token_job)
    assert entry["event"] == "evento"
    assert entry["level"] == "INFO"
    assert entry["ts"].endswith("Z")
    assert entry["request_id"] == "req-1"
    assert entry["job_id"] == "job-1"
    assert entry["duration_ms"] == 12.5
    assert entry["error_code"] == "QUEUE_FULL"


def test_optional_log_fields_are_absent_when_not_applicable():
    entry = format_record("evento")
    assert {"job_id", "duration_ms", "error_code"}.isdisjoint(entry)


def test_real_server_logs_are_json_without_image_content(tmp_path, dummy_models):
    """Corre la app en otro proceso para leer su salida real, línea a línea."""
    script = f"""
import sys, base64
sys.path.insert(0, r"{BACKEND_DIR / 'tests'}")
from api_fakes import image_bytes, make_settings, post_file, running_client, API
from pathlib import Path
settings = make_settings(Path(r"{tmp_path}"), Path(r"{dummy_models}"))
data = image_bytes("PNG")
with running_client(settings, lambda: None) as client:
    post_file(client, "/analyze", data, filename="nombre_privado_visitante.png")
    post_file(client, "/analyze", b"texto")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=BACKEND_DIR, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines
    entries = [json.loads(line) for line in lines]
    requests = [entry for entry in entries if entry["event"] == "http_request"]
    assert len(requests) >= 2
    assert all({"ts", "level", "event", "request_id", "duration_ms"} <= set(entry) for entry in requests)
    assert any(entry.get("error_code") == "UNSUPPORTED_FORMAT" for entry in requests)
    assert "nombre_privado_visitante" not in completed.stdout
    assert "base64" not in completed.stdout


def test_request_id_header(client):
    assert client.get(f"{API}/config").headers["x-request-id"]


# Pincel


def test_parse_strokes_valid():
    parsed = parse_strokes(strokes_json(stroke(points=[[0.12, 0.40], [0.18, 0.38]], radius=0.04, strength=0.7)))
    assert parsed[0].points == ((0.12, 0.40), (0.18, 0.38))
    assert parsed[0].radius == 0.04
    assert parsed[0].direction_deg == 90.0
    assert parsed[0].strength == 0.7
    assert parse_strokes(None) == ()
    assert parse_strokes(" ") == ()


@pytest.mark.parametrize(
    "payload",
    [
        strokes_json(*[stroke()] * 201),
        strokes_json(*[stroke(points=[[0.5, 0.5]] * 200)] * 11),
        strokes_json(stroke(points=[])),
        strokes_json(stroke(points=[[1.5, 0.5]])),
        strokes_json(stroke(points=[[0.5, -0.1]])),
        strokes_json(stroke(points=[[0.5]])),
        strokes_json(stroke(radius=0.004)),
        strokes_json(stroke(radius=0.21)),
        strokes_json(stroke(direction_deg=-1)),
        strokes_json(stroke(direction_deg=361)),
        strokes_json(stroke(strength=1.1)),
        strokes_json(stroke(strength=-0.1)),
        '{"strokes": [{"points": [[NaN, 0.5]], "radius": 0.05, "direction_deg": 0, "strength": 0.5}]}',
        '{"strokes": "nada"}',
        '{"otra": []}',
        "[]",
        "{no json",
    ],
)
def test_parse_strokes_rejects(payload):
    with pytest.raises(ApiError) as info:
        parse_strokes(payload)
    assert info.value.status_code == 422
    assert info.value.code == "INVALID_PARAMS"


def test_stroke_limits_accept_exact_maximum():
    assert len(parse_strokes(strokes_json(*[stroke()] * 200))) == 200
    assert len(parse_strokes(strokes_json(*[stroke(points=[[0.5, 0.5]] * 10)] * 200))) == 200


def test_too_many_strokes_via_api(client):
    response = post_file(client, "/animate", image_bytes(), mask_strokes=strokes_json(*[stroke()] * 201))
    assert_error(response, 422, "INVALID_PARAMS")


def test_rasterize_uses_top_left_origin_and_longest_side_radius():
    raster = rasterize_strokes(parse_strokes(strokes_json(stroke(points=[[0.0, 0.0]], radius=0.1))), 200, 100)
    assert raster.coverage.shape == (100, 200)
    assert raster.coverage.dtype == np.uint8
    assert raster.coverage[0, 0] == 255
    # radio = 0.1 * 200 = 20 px
    assert raster.coverage[0, 20] == 255 and raster.coverage[0, 22] == 0
    assert raster.coverage[20, 0] == 255 and raster.coverage[22, 0] == 0
    assert raster.coverage[99, 199] == 0
    assert raster.strength[0, 0] == 0.5
    assert raster.direction_deg[0, 0] == 90.0
    assert raster.strength[99, 199] == 0.0


def test_rasterize_connects_points_and_last_stroke_wins():
    first = stroke(points=[[0.1, 0.5], [0.9, 0.5]], radius=0.05, direction_deg=10.0, strength=0.2)
    second = stroke(points=[[0.5, 0.5]], radius=0.05, direction_deg=200.0, strength=0.9)
    raster = rasterize_strokes(parse_strokes(strokes_json(first, second)), 100, 100)
    assert raster.coverage[50, 30] == 255  # entre los dos puntos del primer trazo
    assert raster.direction_deg[50, 30] == 10.0
    assert raster.direction_deg[50, 50] == 200.0
    assert abs(raster.strength[50, 50] - 0.9) < 1e-6


def test_rasterizer_agrees_with_engine():
    """El motor rasteriza los mismos trazos que recibe en RenderConfig: ambos deben coincidir."""
    from animation_engine.effects.brush import rasterize_strokes as engine_rasterize

    strokes = parse_strokes(
        strokes_json(
            stroke(points=[[0.1, 0.2], [0.4, 0.6], [0.7, 0.3]], radius=0.03),
            stroke(points=[[0.9, 0.9]], radius=0.1, direction_deg=270.0, strength=1.0),
        )
    )
    ours = rasterize_strokes(strokes, 320, 200).coverage > 0
    painted = engine_rasterize(strokes, (200, 320))[0] > 0
    assert (ours != painted).mean() < 0.001


# Preproceso


def test_preprocess_matches_contract_formula():
    image = np.random.default_rng(0).integers(0, 256, size=(123, 77, 3), dtype=np.uint8)
    tensor = preprocess_seg(image)
    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 3, 320, 320)
    resized = cv2.resize(image, (320, 320), interpolation=cv2.INTER_LINEAR)
    rgb = resized[..., ::-1].astype(np.float64) / 255.0
    expected = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    assert np.abs(tensor[0].transpose(1, 2, 0) - expected).max() < 1e-5


def test_preprocess_converts_bgr_to_rgb():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[..., 2] = 255  # rojo en BGR
    tensor = preprocess_seg(image)
    assert tensor[0, 0, 0, 0] > 2.0 and tensor[0, 2, 0, 0] < -1.5


def test_preprocess_rejects_wrong_input():
    with pytest.raises(ValueError):
        preprocess_seg(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        preprocess_seg(np.zeros((10, 10, 3), dtype=np.float32))


def test_work_size_never_upscales():
    assert work_size(1920, 1080, 1280) == (1280, 720)
    assert work_size(1080, 1920, 1280) == (720, 1280)
    assert work_size(640, 480, 1280) == (640, 480)


# Restricciones de código


def _app_python_files():
    return list((BACKEND_DIR / "app").rglob("*.py")) + [REPO_ROOT / "ml" / "make_dummy_model.py"]


def test_no_torch_dependency_and_no_videowriter():
    for path in _app_python_files():
        source = path.read_text(encoding="utf-8")
        assert "VideoWriter(" not in source, path
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] != "torch" for alias in node.names), path
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] != "torch", path
    lines = (BACKEND_DIR / "requirements.txt").read_text(encoding="utf-8").lower().splitlines()
    assert not [line for line in lines if "torch" in line and not line.startswith("#")]


def test_requirements_are_pinned():
    lines = (BACKEND_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert all("==" in line for line in packages)
    assert "opencv-python-headless==4.10.0.84" in packages


def test_uploaded_filename_never_reaches_disk(client, settings):
    response = post_file(client, "/animate", image_bytes(), filename="../../escapar.jpg", duration_s="2")
    assert response.status_code == 202
    assert not (settings.tmp_dir.parent / "escapar.jpg").exists()
    assert not list(settings.tmp_dir.rglob("escapar*"))
