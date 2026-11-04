"""POST /api/v1/analyze: formato campo a campo y posproceso."""
from __future__ import annotations

import base64

import cv2
import numpy as np
from api_fakes import API, image_bytes, post_file

from app.services.inference import class_stats, postprocess_logits, sha256_file

EXPECTED_KEYS = {
    "width",
    "height",
    "mask_png_base64",
    "classes",
    "inference_ms",
    "depth_png_base64",
    "model_version",
    "model_sha256",
    "conf_threshold",
}


def decode_png(text: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(text), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def test_analyze_response_field_by_field(client, settings):
    response = post_file(client, "/analyze", image_bytes(width=640, height=480))
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == EXPECTED_KEYS

    assert body["width"] == 640 and isinstance(body["width"], int)
    assert body["height"] == 480 and isinstance(body["height"], int)
    assert body["model_version"] == "dummy-dev"
    assert body["model_sha256"] == sha256_file(settings.seg_model_path)
    assert len(body["model_sha256"]) == 64
    assert body["conf_threshold"] == settings.conf_threshold
    assert body["depth_png_base64"] is None

    timing = body["inference_ms"]
    assert set(timing) == {"segmentation", "depth", "total"}
    assert timing["depth"] is None
    assert timing["segmentation"] > 0
    assert timing["total"] >= timing["segmentation"]

    mask = decode_png(body["mask_png_base64"])
    assert mask.dtype == np.uint8 and mask.ndim == 2
    assert mask.shape == (480, 640)
    assert set(np.unique(mask)).issubset(set(range(7)))

    classes = body["classes"]
    assert classes, "Debe haber al menos una clase"
    names = ["fondo", "cielo", "agua", "vegetacion", "fuego_luz", "persona", "animal"]
    for entry in classes:
        assert set(entry) == {"id", "name", "pixel_pct", "confidence"}
        assert entry["name"] == names[entry["id"]]
        assert entry["pixel_pct"] > 0
        assert 0.0 <= entry["confidence"] <= 1.0
    percentages = [entry["pixel_pct"] for entry in classes]
    assert percentages == sorted(percentages, reverse=True)
    assert abs(sum(percentages) - 100.0) < 0.01

    # El reporte coincide con la máscara devuelta: nada está precalculado.
    counts = np.bincount(mask.ravel(), minlength=7)
    assert {entry["id"] for entry in classes} == {int(i) for i in np.nonzero(counts)[0]}
    for entry in classes:
        assert abs(entry["pixel_pct"] - 100.0 * counts[entry["id"]] / mask.size) < 1e-3


def test_analyze_include_depth(client):
    response = post_file(client, "/analyze", image_bytes(), include_depth="true")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["inference_ms"]["depth"] > 0
    depth = decode_png(body["depth_png_base64"])
    assert depth.dtype == np.uint8 and depth.shape == (480, 640)
    assert depth.max() == 255 and depth.min() == 0


def test_work_size_reduces_but_never_upscales(client, settings):
    big = post_file(client, "/analyze", image_bytes(width=1920, height=1080)).json()
    assert (big["width"], big["height"]) == (settings.work_max_side, 720)
    assert decode_png(big["mask_png_base64"]).shape == (720, 1280)
    small = post_file(client, "/analyze", image_bytes(width=300, height=256)).json()
    assert (small["width"], small["height"]) == (300, 256)


def test_formats_are_accepted_by_magic_bytes(client):
    for fmt, name in (("JPEG", "a.jpg"), ("PNG", "a.png"), ("WEBP", "a.webp")):
        assert post_file(client, "/analyze", image_bytes(fmt), filename=name).status_code == 200


def test_mask_depends_on_input(client):
    first = post_file(client, "/analyze", image_bytes(width=640, height=480)).json()
    other = np.full((480, 640, 3), 30, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", other)
    assert ok
    second = post_file(client, "/analyze", encoded.tobytes(), filename="b.png").json()
    assert first["mask_png_base64"] != second["mask_png_base64"]


def test_postprocess_threshold_sends_low_confidence_to_background():
    logits = np.zeros((7, 2, 2), dtype=np.float32)
    logits[3, 0, 0] = 10.0  # vegetación con alta confianza
    logits[5, 0, 1] = 0.1  # casi uniforme: max_prob < 0.5
    logits[1, 1, 0] = 10.0
    logits[1, 1, 1] = 10.0
    classes, max_prob = postprocess_logits(logits, 0.5)
    assert classes.tolist() == [[3, 0], [1, 1]]
    assert max_prob[0, 0] > 0.99 and max_prob[0, 1] < 0.5
    assert classes.dtype == np.uint8


def test_softmax_is_numerically_stable_with_large_logits():
    logits = np.zeros((7, 1, 1), dtype=np.float32)
    logits[2, 0, 0] = 5000.0
    classes, max_prob = postprocess_logits(logits, 0.5)
    assert classes[0, 0] == 2 and np.isfinite(max_prob).all()


def test_confidence_is_average_of_max_prob_over_assigned_pixels():
    grid = np.array([[1, 1], [2, 0]], dtype=np.uint8)
    max_prob = np.array([[0.8, 0.6], [0.9, 0.4]], dtype=np.float32)
    stats = {entry["id"]: entry for entry in class_stats(grid, grid, max_prob)}
    assert abs(stats[1]["confidence"] - 0.7) < 1e-4
    assert abs(stats[2]["confidence"] - 0.9) < 1e-4
    assert stats[1]["pixel_pct"] == 50.0
    assert 3 not in stats


def test_openapi_lists_the_contract_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    for route in ("analyze", "animate", "jobs/{job_id}", "results/{filename}", "gallery", "gallery/{job_id}", "config", "health", "health/live", "model/metrics"):
        assert f"{API}/{route}" in paths
