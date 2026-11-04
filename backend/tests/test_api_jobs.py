"""Ciclo de jobs, cola, reinicio y limpieza. FFmpeg se sustituye: su prueba real está en test_api_ffmpeg.py."""
from __future__ import annotations

import json
import os
import re
import time
from datetime import timedelta
from pathlib import Path

import pytest
from api_fakes import (
    API,
    FakeEngine,
    Gate,
    assert_error,
    image_bytes,
    make_settings,
    post_file,
    running_client,
    wait_job,
    wait_until,
)

from app.observability import REGISTRY
from app.services import job_manager as jm

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
JOB_KEYS = {"job_id", "status", "stage", "progress", "mood", "seed", "created_at", "updated_at", "result", "timings_ms", "error"}


def test_animate_accepted_response_shape(client, fake_encoder):
    response = post_file(client, "/animate", image_bytes())
    assert response.status_code == 202, response.text
    body = response.json()
    assert set(body) == {"job_id", "status", "queue_position", "status_url"}
    assert UUID_RE.match(body["job_id"])
    assert body["status"] == "queued"
    assert isinstance(body["queue_position"], int) and body["queue_position"] >= 1
    assert body["status_url"] == f"{API}/jobs/{body['job_id']}"


def test_job_cycle_field_by_field(client, engine, fake_encoder, settings):
    job_id = post_file(client, "/animate", image_bytes(), mood="epico", duration_s="2.0", seed="7").json()["job_id"]
    body = wait_job(client, job_id)
    assert set(body) == JOB_KEYS
    assert body["job_id"] == job_id
    assert body["status"] == "succeeded", body
    assert body["stage"] == "done"
    assert body["progress"] == 1.0
    assert body["mood"] == "epico"
    assert body["seed"] == 7 and isinstance(body["seed"], int)
    assert ISO_RE.match(body["created_at"]) and ISO_RE.match(body["updated_at"])
    assert body["error"] is None

    result = body["result"]
    assert set(result) == {"url", "format", "width", "height", "duration_s", "fps", "size_bytes"}
    assert result["url"] == f"{API}/results/{job_id}.mp4"
    assert result["format"] == "mp4"
    assert (result["width"], result["height"]) == (640, 480)
    assert result["duration_s"] == 2.0
    assert result["fps"] == 24
    assert result["size_bytes"] == (settings.results_dir / f"{job_id}.mp4").stat().st_size

    timings = body["timings_ms"]
    assert set(timings) == {"segmentation", "depth", "render", "encode", "total"}
    assert all(isinstance(value, (int, float)) and value >= 0 for value in timings.values())
    assert timings["total"] >= timings["segmentation"] + timings["depth"]

    call = engine.image_calls[0]
    assert call["image"].shape == (480, 640, 3)
    assert call["mask"].shape == (480, 640) and call["mask"].dtype.name == "uint8"
    assert call["depth"].shape == (480, 640) and call["depth"].dtype.name == "float32"
    assert 0.0 <= call["depth"].min() <= call["depth"].max() <= 1.0
    assert call["config"].seed == 7
    assert call["config"].mood.name == "epico"
    assert call["config"].duration_s == 2.0

    # El temporal de entrada se elimina al terminar.
    assert not (settings.tmp_dir / "jobs" / job_id).exists()


def test_random_seed_is_generated_and_stored(client, engine, fake_encoder):
    job_id = post_file(client, "/animate", image_bytes(), duration_s="2").json()["job_id"]
    body = wait_job(client, job_id)
    used = engine.image_calls[0]["config"].seed
    assert isinstance(body["seed"], int) and 0 <= body["seed"] <= 2**32 - 1
    assert body["seed"] == used
    assert client.app.state.services.store.get(job_id)["seed"] == used


def test_seed_is_visible_in_every_job_state(tmp_path, dummy_models, engine, fake_encoder):
    settings = make_settings(tmp_path, dummy_models)
    gate = Gate(engine)
    try:
        with running_client(settings, gate) as client:
            running = post_file(client, "/animate", image_bytes(), duration_s="2", seed="123").json()["job_id"]
            assert gate.entered.wait(10)
            queued = post_file(client, "/animate", image_bytes(), duration_s="2").json()["job_id"]

            running_body = client.get(f"{API}/jobs/{running}").json()
            assert running_body["status"] == "running" and running_body["seed"] == 123
            queued_body = client.get(f"{API}/jobs/{queued}").json()
            assert queued_body["status"] == "queued" and isinstance(queued_body["seed"], int)

            gate.release.set()
            finished = wait_job(client, queued)
            # Al azar, pero fijo desde que se aceptó el job y es el que recibió el motor.
            assert finished["seed"] == queued_body["seed"] == engine.image_calls[1]["config"].seed
    finally:
        gate.release.set()


def test_two_random_seeds_differ(client, fake_encoder):
    seeds = {
        wait_job(client, post_file(client, "/animate", image_bytes(), duration_s="2").json()["job_id"])["seed"]
        for _ in range(3)
    }
    assert len(seeds) > 1


def test_seed_survives_restart(settings, engine):
    store = jm.JobStore(settings.db_path)
    _insert_stale(store, "interrumpido", "running", seed=99)
    store.close()
    with running_client(settings, lambda: engine) as client:
        body = client.get(f"{API}/jobs/interrumpido").json()
    assert body["status"] == "failed" and body["seed"] == 99


def test_strokes_reach_the_engine(client, engine, fake_encoder):
    strokes = {"strokes": [{"points": [[0.1, 0.2], [0.3, 0.4]], "radius": 0.05, "direction_deg": 90.0, "strength": 0.7}]}
    job_id = post_file(client, "/animate", image_bytes(), duration_s="2", brush_mode="only", mask_strokes=json.dumps(strokes)).json()["job_id"]
    assert wait_job(client, job_id)["status"] == "succeeded"
    config = engine.image_calls[0]["config"]
    assert config.brush_mode == "only"
    assert config.strokes[0].points == ((0.1, 0.2), (0.3, 0.4))
    assert config.strokes[0].direction_deg == 90.0


def test_job_not_found(client):
    error = assert_error(client.get(f"{API}/jobs/no-existe"), 404, "JOB_NOT_FOUND")
    assert error["details"]["job_id"] == "no-existe"


def test_engine_missing_fails_job_with_clear_error(settings):
    def unavailable():
        raise jm.JobFailure("ENGINE_UNAVAILABLE", "El motor de animación no está disponible en este servidor.")

    with running_client(settings, unavailable) as client:
        job_id = post_file(client, "/animate", image_bytes()).json()["job_id"]
        body = wait_job(client, job_id)
        assert body["status"] == "failed"
        assert body["result"] is None
        assert body["error"]["code"] == "ENGINE_UNAVAILABLE"
        assert set(body["error"]) == {"code", "message"}
        assert_error(client.get(f"{API}/results/{job_id}.mp4"), 404, "RESULT_NOT_FOUND")


def test_load_engine_reports_incomplete_package(monkeypatch):
    import types

    monkeypatch.setattr(jm.importlib, "import_module", lambda name: types.SimpleNamespace(get_mood=None))
    with pytest.raises(jm.JobFailure) as info:
        jm.load_engine()
    assert info.value.code == "ENGINE_UNAVAILABLE"

    def fail(name):
        raise ImportError("sin motor")

    monkeypatch.setattr(jm.importlib, "import_module", fail)
    with pytest.raises(jm.JobFailure):
        jm.load_engine()


def test_ffmpeg_missing_fails_job_with_clear_error(client, monkeypatch):
    from app.services import encoder

    def missing(_):
        raise encoder.FFmpegMissing("no hay")

    monkeypatch.setattr(encoder, "find_tool", missing)
    body = wait_job(client, post_file(client, "/animate", image_bytes()).json()["job_id"])
    assert body["status"] == "failed"
    assert body["error"]["code"] == "ENCODER_UNAVAILABLE"


class ExplodingEngine(FakeEngine):
    def animate_image(self, image_bgr, mask, depth, config, progress=None):
        yield image_bgr
        raise RuntimeError("fallo interno con detalle privado")


def test_engine_exception_fails_job_and_worker_survives(settings, fake_encoder):
    engine = ExplodingEngine()
    current = {"engine": engine}
    with running_client(settings, lambda: current["engine"]) as client:
        body = wait_job(client, post_file(client, "/animate", image_bytes()).json()["job_id"])
        assert body["status"] == "failed" and body["result"] is None
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "privado" not in body["error"]["message"]
        assert not list(settings.results_dir.glob("*"))

        current["engine"] = FakeEngine()
        body = wait_job(client, post_file(client, "/animate", image_bytes(), duration_s="2").json()["job_id"])
        assert body["status"] == "succeeded"


class NoFramesEngine(FakeEngine):
    def animate_image(self, image_bgr, mask, depth, config, progress=None):
        return iter(())


def test_engine_without_frames_fails(settings, fake_encoder):
    with running_client(settings, NoFramesEngine) as client:
        body = wait_job(client, post_file(client, "/animate", image_bytes()).json()["job_id"])
        assert body["status"] == "failed"


def test_queue_full_returns_429_with_retry_after(tmp_path, dummy_models, engine, fake_encoder):
    settings = make_settings(tmp_path, dummy_models, max_queue=2)
    gate = Gate(engine)
    try:
        with running_client(settings, gate) as client:
            first = post_file(client, "/animate", image_bytes(), duration_s="2")
            assert first.status_code == 202
            assert gate.entered.wait(10)  # el primero ya salió de la cola y está en curso

            status = client.get(f"{API}/jobs/{first.json()['job_id']}").json()
            assert status["status"] == "running"

            second = post_file(client, "/animate", image_bytes(), duration_s="2")
            third = post_file(client, "/animate", image_bytes(), duration_s="2")
            assert second.json()["queue_position"] == 1
            assert third.json()["queue_position"] == 2
            assert REGISTRY.get_sample_value("kinetia_queue_depth") == 2

            refused = post_file(client, "/animate", image_bytes(), duration_s="2")
            error = assert_error(refused, 429, "QUEUE_FULL")
            assert int(refused.headers["Retry-After"]) >= 1
            assert error["details"]["max_queue"] == 2

            # /analyze y /health no se bloquean con el worker ocupado.
            assert post_file(client, "/analyze", image_bytes()).status_code == 200
            assert client.get(f"{API}/health").status_code == 200

            # Resultado sin terminar.
            job_id = second.json()["job_id"]
            assert_error(client.get(f"{API}/results/{job_id}.mp4"), 409, "JOB_NOT_READY")

            gate.release.set()
            for job in (first, second, third):
                assert wait_job(client, job.json()["job_id"])["status"] == "succeeded"
            assert REGISTRY.get_sample_value("kinetia_queue_depth") == 0
            assert post_file(client, "/animate", image_bytes(), duration_s="2").status_code == 202
    finally:
        gate.release.set()


def test_analyze_during_inference_on_worker_does_not_deadlock(settings, fake_encoder):
    class SlowEngine(FakeEngine):
        def animate_image(self, image_bgr, mask, depth, config, progress=None):
            for frame in super().animate_image(image_bgr, mask, depth, config, progress):
                time.sleep(0.01)
                yield frame

    with running_client(settings, SlowEngine) as client:
        job_id = post_file(client, "/animate", image_bytes(), duration_s="2").json()["job_id"]
        for _ in range(5):
            assert post_file(client, "/analyze", image_bytes(), include_depth="true").status_code == 200
        assert wait_job(client, job_id)["status"] == "succeeded"


def _insert_stale(store, job_id, status, **extra):
    now = jm.to_iso(jm.utc_now())
    values = dict(
        job_id=job_id,
        status=status,
        stage="rendering" if status == "running" else "queued",
        progress=0.4,
        media="image",
        mood="calma",
        output_format="mp4",
        duration_s=4.0,
        seed=1,
        brush_mode="add",
        gallery_opt_in=0,
        created_at=now,
        updated_at=now,
    )
    values.update(extra)
    store.insert(**values)


def test_restart_marks_running_and_queued_as_interrupted(settings, engine):
    store = jm.JobStore(settings.db_path)
    _insert_stale(store, "running-job", "running")
    _insert_stale(store, "queued-job", "queued")
    _insert_stale(store, "done-job", "succeeded", stage="done")
    store.close()

    with running_client(settings, lambda: engine) as client:
        for job_id in ("running-job", "queued-job"):
            body = client.get(f"{API}/jobs/{job_id}").json()
            assert body["status"] == "failed"
            assert body["result"] is None
            assert body["error"]["code"] == "JOB_INTERRUPTED"
            assert body["error"]["message"]
        assert client.get(f"{API}/jobs/done-job").json()["status"] == "succeeded"


def test_real_restart_with_job_in_progress(settings, engine, fake_encoder):
    gate = Gate(engine)
    with running_client(settings, gate) as client:
        job_id = post_file(client, "/animate", image_bytes()).json()["job_id"]
        assert gate.entered.wait(10)
        queued_id = post_file(client, "/animate", image_bytes()).json()["job_id"]
        client.app.state.services.jobs._worker.join(timeout=0)  # sigue vivo
    # Tras apagar, el proveedor sigue bloqueado como un proceso muerto; se libera al final.
    try:
        with running_client(settings, lambda: engine) as client:
            for job in (job_id, queued_id):
                body = client.get(f"{API}/jobs/{job}").json()
                assert body["status"] == "failed", body
                assert body["error"]["code"] == "JOB_INTERRUPTED"
    finally:
        gate.release.set()


def test_database_uses_wal(client):
    assert client.app.state.services.store.journal_mode().lower() == "wal"


def test_cleanup_removes_expired_results(tmp_path, dummy_models, engine, fake_encoder):
    settings = make_settings(tmp_path, dummy_models, result_ttl_hours=1)
    with running_client(settings, lambda: engine) as client:
        services = client.app.state.services
        job_id = post_file(client, "/animate", image_bytes(), duration_s="2", gallery_opt_in="true").json()["job_id"]
        assert wait_job(client, job_id)["status"] == "succeeded"
        path = settings.results_dir / f"{job_id}.mp4"
        assert path.exists()
        assert client.get(f"{API}/results/{job_id}.mp4").status_code == 200
        assert len(client.get(f"{API}/gallery").json()["items"]) == 1

        services.jobs.cleanup()
        assert path.exists(), "No debe borrar lo que aún no vence"

        old = jm.to_iso(jm.utc_now() - timedelta(hours=2))
        with services.store._conn:
            services.store._conn.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?", (old, job_id))
        services.jobs.cleanup()
        assert not path.exists()
        assert_error(client.get(f"{API}/results/{job_id}.mp4"), 404, "RESULT_NOT_FOUND")
        assert client.get(f"{API}/gallery").json()["items"] == []

        # Las filas muy antiguas también se purgan.
        ancient = jm.to_iso(jm.utc_now() - timedelta(hours=5))
        with services.store._conn:
            services.store._conn.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?", (ancient, job_id))
        services.jobs.cleanup()
        assert_error(client.get(f"{API}/jobs/{job_id}"), 404, "JOB_NOT_FOUND")


def test_startup_purges_temporary_files(settings, engine):
    stale = settings.tmp_dir / "uploads" / "viejo"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"x")
    job_dir = settings.tmp_dir / "jobs" / "abc"
    job_dir.mkdir(parents=True)
    (job_dir / "input").write_bytes(b"x")
    with running_client(settings, lambda: engine):
        assert not stale.exists()
        assert not job_dir.exists()


def test_cleanup_removes_old_orphan_temporaries(client, settings):
    orphan = settings.tmp_dir / "uploads" / "huerfano"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"x")
    recent = settings.tmp_dir / "uploads" / "reciente"
    recent.write_bytes(b"x")
    old = time.time() - jm.TMP_MAX_AGE_S - 10
    os.utime(orphan, (old, old))
    client.app.state.services.jobs.cleanup()
    assert not orphan.exists()
    assert recent.exists()


def test_rejected_uploads_leave_no_temporaries(client, settings):
    post_file(client, "/analyze", b"texto")
    post_file(client, "/animate", b"texto")
    post_file(client, "/analyze", image_bytes())
    assert list(Path(settings.tmp_dir / "uploads").glob("*")) == []


def test_progress_reporter_is_monotonic(client, fake_encoder, settings):
    seen = []
    job_id = post_file(client, "/animate", image_bytes(), duration_s="6").json()["job_id"]

    def finished():
        body = client.get(f"{API}/jobs/{job_id}").json()
        seen.append(body["progress"])
        return body["status"] in ("succeeded", "failed")

    wait_until(finished)
    assert seen == sorted(seen)
    assert seen[-1] == 1.0


def test_real_engine_through_the_job_pipeline(settings, fake_encoder):
    """Motor real con modelos dummy; FFmpeg sustituido. La versión con FFmpeg está en test_api_ffmpeg.py."""
    with running_client(settings) as client:
        strokes = {"strokes": [{"points": [[0.2, 0.2], [0.6, 0.6]], "radius": 0.05, "direction_deg": 45.0, "strength": 0.8}]}
        job_id = post_file(
            client,
            "/animate",
            image_bytes("PNG", 320, 256),
            duration_s="2",
            seed="11",
            mood="magico",
            mask_strokes=json.dumps(strokes),
        ).json()["job_id"]
        body = wait_job(client, job_id, 240)
    assert body["status"] == "succeeded", body
    assert body["result"]["duration_s"] == 2.0
    assert all(value > 0 for value in body["timings_ms"].values())
