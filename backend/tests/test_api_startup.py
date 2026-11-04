"""Arranque: política del dummy, carga única, salud y sustitución del modelo."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytest
from api_fakes import API, assert_error, image_bytes, make_settings, post_file, running_client, wait_until
from conftest import BACKEND_DIR, generate_dummy
from fastapi.testclient import TestClient
from onnx import TensorProto, helper

from app.main import create_app
from app.services import inference
from app.services.inference import ModelStartupError, ModelStore, sha256_file


def copy_models(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def test_dummy_refused_without_flag(tmp_path, dummy_models):
    settings = make_settings(tmp_path, dummy_models, allow_dummy_model=False)
    app = create_app(settings, exit_on_load_failure=False)
    with pytest.raises(ModelStartupError, match="ALLOW_DUMMY_MODEL"):
        with TestClient(app):
            pass


def test_real_uvicorn_refuses_dummy_without_flag(tmp_path, dummy_models):
    env = {
        **os.environ,
        "MODELS_DIR": str(dummy_models),
        "ALLOW_DUMMY_MODEL": "false",
        "RESULTS_DIR": str(tmp_path / "results"),
        "DB_PATH": str(tmp_path / "db" / "k.db"),
        "TMP_DIR": str(tmp_path / "tmp"),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "0", "--log-level", "info"],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        timeout=60,
        text=True,
    )
    assert completed.returncode != 0
    assert "ALLOW_DUMMY_MODEL" in completed.stdout + completed.stderr


def test_flag_allows_dummy_and_version_is_reported(client):
    body = client.get(f"{API}/health").json()
    assert body["model_version"] == "dummy-dev"


def test_non_dummy_version_does_not_need_flag(tmp_path, dummy_models, engine):
    models = generate_dummy(tmp_path / "models", "--version", "seg-1.0.0", "--seed", "3")
    settings = make_settings(tmp_path, models, allow_dummy_model=False)
    with running_client(settings, lambda: engine) as client:
        assert client.get(f"{API}/health").json()["model_version"] == "seg-1.0.0"
        assert post_file(client, "/analyze", image_bytes()).json()["model_version"] == "seg-1.0.0"


def test_health_ok_fields(client, settings):
    response = client.get(f"{API}/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "depth_loaded": True,
        "model_version": "dummy-dev",
        "model_sha256": sha256_file(settings.seg_model_path),
        "depth_sha256": sha256_file(settings.depth_model_path),
    }


def test_health_503_until_both_models_loaded(settings, engine, monkeypatch):
    release = threading.Event()
    original = ModelStore.load

    def slow_load(self):
        release.wait(timeout=30)
        original(self)

    monkeypatch.setattr(ModelStore, "load", slow_load)
    try:
        with running_client(settings, lambda: engine, wait_ready=False) as client:
            loading = client.get(f"{API}/health")
            assert loading.status_code == 503
            body = loading.json()
            assert body["status"] == "loading"
            assert body["model_loaded"] is False and body["depth_loaded"] is False
            assert client.get(f"{API}/health/live").json() == {"status": "alive"}
            assert_error(post_file(client, "/analyze", image_bytes()), 503, "MODEL_NOT_READY")
            assert_error(post_file(client, "/animate", image_bytes()), 503, "MODEL_NOT_READY")
            release.set()
            wait_until(lambda: client.get(f"{API}/health").status_code == 200)
            assert post_file(client, "/analyze", image_bytes()).status_code == 200
    finally:
        release.set()


def test_models_are_loaded_once(settings, engine, monkeypatch):
    calls = []
    original = inference.make_session

    def counting(path, threads):
        calls.append((path.name, threads))
        return original(path, threads)

    monkeypatch.setattr(inference, "make_session", counting)
    with running_client(settings, lambda: engine) as client:
        for _ in range(3):
            assert post_file(client, "/analyze", image_bytes(), include_depth="true").status_code == 200
    assert sorted(name for name, _ in calls) == ["kinetia_seg.onnx", "midas_small.onnx"]


def test_intra_op_threads_setting_is_applied(settings, engine, monkeypatch):
    seen = []
    original = onnxruntime.InferenceSession

    def spy(path, sess_options=None, providers=None):
        seen.append(sess_options.intra_op_num_threads)
        return original(path, sess_options=sess_options, providers=providers)

    monkeypatch.setattr(inference.ort, "InferenceSession", spy)
    with running_client(settings, lambda: engine):
        pass
    assert seen == [settings.ort_intra_op_threads] * 2


def test_health_live_does_not_touch_model_or_disk(settings, engine):
    with running_client(settings, lambda: engine, wait_ready=False) as client:
        client.app.state.services = None  # si tocara algún servicio, fallaría
        assert client.get(f"{API}/health/live").status_code == 200


def test_missing_manifest_refuses_start(tmp_path, dummy_models):
    models = copy_models(dummy_models, tmp_path / "m")
    (models / "manifest.json").unlink()
    with pytest.raises(ModelStartupError, match="manifest|No existe"):
        ModelStore(make_settings(tmp_path, models)).preflight()


def test_missing_model_file_refuses_start(tmp_path, dummy_models):
    models = copy_models(dummy_models, tmp_path / "m")
    (models / "midas_small.onnx").unlink()
    with pytest.raises(ModelStartupError, match="No existe"):
        ModelStore(make_settings(tmp_path, models)).preflight()


def test_hash_mismatch_refuses_start(tmp_path, dummy_models):
    models = copy_models(dummy_models, tmp_path / "m")
    with (models / "kinetia_seg.onnx").open("ab") as handle:
        handle.write(b"manipulado")
    with pytest.raises(ModelStartupError, match="SHA-256"):
        ModelStore(make_settings(tmp_path, models)).preflight()


def test_wrong_classes_refuse_start(tmp_path, dummy_models):
    models = copy_models(dummy_models, tmp_path / "m")
    manifest = json.loads((models / "manifest.json").read_text(encoding="utf-8"))
    manifest["classes"] = list(reversed(manifest["classes"]))
    (models / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelStartupError, match="clases"):
        ModelStore(make_settings(tmp_path, models)).preflight()


def _rewrite_manifest(models: Path, path: Path) -> None:
    manifest = json.loads((models / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    (models / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _save_bad_seg(path: Path, out_channels: int, output_name: str = "logits") -> None:
    weights = onnx.numpy_helper.from_array(np.ones((out_channels, 3, 1, 1), dtype=np.float32), "w")
    node = helper.make_node("Conv", ["input", "w"], [output_name], kernel_shape=[1, 1])
    graph = helper.make_graph(
        [node],
        "bad",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 320, 320])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None)],
        initializer=[weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


@pytest.mark.parametrize("channels, name, message", [(5, "logits", "forma"), (7, "salida", "logits")])
def test_wrong_seg_signature_is_rejected_at_load(tmp_path, dummy_models, channels, name, message):
    models = copy_models(dummy_models, tmp_path / "m")
    _save_bad_seg(models / "kinetia_seg.onnx", channels, name)
    _rewrite_manifest(models, models / "kinetia_seg.onnx")
    store = ModelStore(make_settings(tmp_path, models))
    store.preflight()
    with pytest.raises(ModelStartupError, match=message):
        store.load()
    assert store.ready is False


def test_load_failure_keeps_health_503(tmp_path, dummy_models, engine):
    models = copy_models(dummy_models, tmp_path / "m")
    (models / "kinetia_seg.onnx").write_bytes(b"esto no es un onnx")
    _rewrite_manifest(models, models / "kinetia_seg.onnx")
    settings = make_settings(tmp_path, models)
    with running_client(settings, lambda: engine, wait_ready=False) as client:
        time.sleep(0.5)
        assert client.get(f"{API}/health").status_code == 503


def _run_uvicorn_until_exit(models: Path, work: Path, timeout: float = 90.0) -> tuple[int, str]:
    env = {
        **os.environ,
        "MODELS_DIR": str(models),
        "ALLOW_DUMMY_MODEL": "true",
        "RESULTS_DIR": str(work / "results"),
        "DB_PATH": str(work / "db" / "k.db"),
        "TMP_DIR": str(work / "tmp"),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "0"],
        cwd=BACKEND_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # El python.exe de un venv en Windows es un lanzador: hay que terminar también a su hijo.
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, check=False)
        else:
            process.kill()
        process.communicate()
        pytest.fail("El servidor siguió vivo con un modelo que no se puede cargar")
    return process.returncode, output


@pytest.mark.parametrize("kind", ["not_onnx", "wrong_signature"])
def test_real_uvicorn_exits_with_code_1_when_model_load_fails(tmp_path, dummy_models, kind):
    """Ruta os._exit: pasa las comprobaciones previas, porque el hash coincide, y falla al cargar."""
    models = copy_models(dummy_models, tmp_path / "m")
    path = models / "kinetia_seg.onnx"
    if kind == "not_onnx":
        path.write_bytes(b"esto no es un onnx")
    else:
        _save_bad_seg(path, 5)
    _rewrite_manifest(models, path)
    ModelStore(make_settings(tmp_path, models)).preflight()  # no lanza: el fallo es de la carga

    code, output = _run_uvicorn_until_exit(models, tmp_path / "work")

    assert code == 1, output
    failures = [json.loads(line) for line in output.splitlines() if '"model_load_failed"' in line]
    assert len(failures) == 1, output
    assert failures[0]["level"] == "CRITICAL"
    assert "models_loaded" not in output


def test_swapping_model_with_same_signature_needs_no_code_change(tmp_path, engine):
    """Criterio 4: otro ONNX, con otra arquitectura y otros pesos, misma firma."""
    first = generate_dummy(tmp_path / "a", "--seed", "1")
    second = generate_dummy(tmp_path / "b", "--seed", "2", "--heavy", "--version", "seg-9.9.9")
    image = image_bytes()

    with running_client(make_settings(tmp_path / "s1", first), lambda: engine) as client:
        one = post_file(client, "/analyze", image, include_depth="true").json()
    with running_client(make_settings(tmp_path / "s2", second), lambda: engine) as client:
        two = post_file(client, "/analyze", image, include_depth="true").json()

    assert one["model_sha256"] != two["model_sha256"]
    assert two["model_version"] == "seg-9.9.9"
    assert one["mask_png_base64"] != two["mask_png_base64"]
    assert set(one) == set(two)


def test_official_style_depth_signature_is_supported(tmp_path, dummy_models, engine):
    """El MiDaS oficial usa otros nombres de entrada y salida, y salida con canal explícito."""
    models = copy_models(dummy_models, tmp_path / "m")
    path = models / "midas_small.onnx"
    weights = onnx.numpy_helper.from_array(np.ones((1, 3, 1, 1), dtype=np.float32), "w")
    node = helper.make_node("Conv", ["0", "w"], ["1"], kernel_shape=[1, 1])
    graph = helper.make_graph(
        [node],
        "official_like",
        [helper.make_tensor_value_info("0", TensorProto.FLOAT, [1, 3, 256, 256])],
        [helper.make_tensor_value_info("1", TensorProto.FLOAT, [1, 1, 256, 256])],
        initializer=[weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    _rewrite_manifest(models, path)
    with running_client(make_settings(tmp_path, models), lambda: engine) as client:
        response = post_file(client, "/analyze", image_bytes(), include_depth="true")
        assert response.status_code == 200
        assert response.json()["depth_png_base64"]


def test_dummy_models_follow_contract(dummy_models):
    manifest = json.loads((dummy_models / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "dummy-dev"
    assert set(manifest) == {
        "version", "created_at", "classes", "input_size", "mean", "std", "files", "dataset_manifest_sha256",
    }
    assert manifest["classes"] == ["fondo", "cielo", "agua", "vegetacion", "fuego_luz", "persona", "animal"]
    for name in ("kinetia_seg.onnx", "midas_small.onnx"):
        entry = manifest["files"][name]
        assert entry["sha256"] == sha256_file(dummy_models / name)
        assert entry["bytes"] == (dummy_models / name).stat().st_size

    seg = onnx.load(str(dummy_models / "kinetia_seg.onnx"))
    assert seg.opset_import[0].version >= 17
    assert seg.graph.input[0].name == "input"
    assert seg.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert seg.graph.output[0].name == "logits"
    session = onnxruntime.InferenceSession(str(dummy_models / "kinetia_seg.onnx"), providers=["CPUExecutionProvider"])
    for batch in (1, 3):
        out = session.run(None, {"input": np.zeros((batch, 3, 320, 320), np.float32)})[0]
        assert out.shape == (batch, 7, 320, 320)
    depth = onnxruntime.InferenceSession(str(dummy_models / "midas_small.onnx"), providers=["CPUExecutionProvider"])
    assert depth.get_inputs()[0].shape == [1, 3, 256, 256]
