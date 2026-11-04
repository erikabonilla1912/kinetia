""".onnx y .pt coinciden; evaluate.py y los esquemas cumplen el contrato."""
from __future__ import annotations

import json
import re
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

import evaluate as ev
import export as ex
import train as tr
import validate_artifacts as va
from dataset import CLASSES, file_sha256, manifest_sha256, read_image_bgr, preprocess_seg
from model import build_model


@pytest.fixture(scope="module")
def exported(mini_dataset, tmp_path_factory):
    _, manifest, _ = mini_dataset
    base = tmp_path_factory.mktemp("export")
    torch.manual_seed(0)
    model = build_model(pretrained=False)
    model.eval()
    tr.write_best_model(model, base, 1, 0.1, manifest_sha256(manifest))
    midas = base / "midas_small.onnx"
    midas.write_bytes(b"archivo de relleno: esta prueba solo necesita su hash")
    out = base / "out"
    ex.main(["--checkpoint", str(base / "kinetia_seg.pt"), "--midas", str(midas), "--manifest", str(manifest), "--out-dir", str(out), "--version", "test"])
    return {"out": out, "model": model, "manifest": manifest}


def test_onnx_matches_pt_on_real_preprocessing(exported, mini_dataset):
    data, _, rows = mini_dataset
    model, _ = ex.load_trained(exported["out"] / "kinetia_seg.pt")
    session = ort.InferenceSession(str(exported["out"] / "kinetia_seg.onnx"), providers=["CPUExecutionProvider"])
    x = np.concatenate([preprocess_seg(read_image_bgr(data, r)) for r in rows[:3]])
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    got = session.run(["logits"], {"input": x})[0]
    assert np.abs(expected - got).max() <= 1e-4


def test_verify_detects_a_different_model(exported):
    other = build_model(pretrained=False)
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.5)
    other.eval()
    with pytest.raises(RuntimeError):
        ex.verify_pt_vs_onnx(other, exported["out"] / "kinetia_seg.onnx")


def test_onnx_signature_matches_contract(exported):
    graph = onnx.load(str(exported["out"] / "kinetia_seg.onnx"))
    assert max(o.version for o in graph.opset_import if o.domain in ("", "ai.onnx")) >= 17
    inp, out = graph.graph.input[0], graph.graph.output[0]
    assert inp.name == "input" and out.name == "logits"
    assert inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    assert isinstance(dims[0], str) and dims[1:] == [3, 320, 320]
    assert "Softmax" not in {n.op_type for n in graph.graph.node}
    session = ort.InferenceSession(str(exported["out"] / "kinetia_seg.onnx"), providers=["CPUExecutionProvider"])
    for n in (1, 4):
        got = session.run(["logits"], {"input": np.zeros((n, 3, 320, 320), dtype=np.float32)})[0]
        assert got.shape == (n, 7, 320, 320) and got.dtype == np.float32


def test_hashes_file_format(exported):
    text = (exported["out"] / "hashes.sha256").read_text(encoding="ascii")
    lines = text.splitlines()
    assert [line.split("  ")[1] for line in lines] == ["kinetia_seg.pt", "kinetia_seg.onnx"]
    assert all(re.fullmatch(r"[0-9a-f]{64}  \S+", line) for line in lines)
    assert lines[1].split()[0] == file_sha256(exported["out"] / "kinetia_seg.onnx")


def test_manifest_json_matches_schema(exported):
    data = json.loads((exported["out"] / "manifest.json").read_text(encoding="utf-8"))
    assert va.validate_manifest(data) == []


@pytest.fixture(scope="module")
def evaluated(exported, mini_dataset):
    data, manifest, _ = mini_dataset
    code = ev.main(["--models-dir", str(exported["out"]), "--data-root", str(data), "--manifest", str(manifest), "--batch-size", "2"])
    assert code == 0
    return {**exported, "data": data}


def test_metrics_json_matches_schema(evaluated):
    assert va.validate_dir(evaluated["out"]) == []
    metrics = json.loads((evaluated["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_sha256"] == file_sha256(evaluated["out"] / "kinetia_seg.onnx")
    assert metrics["classes"] == CLASSES


def _base_args(evaluated):
    return ["--models-dir", str(evaluated["out"]), "--data-root", str(evaluated["data"]), "--manifest", str(evaluated["manifest"]), "--batch-size", "2"]


def test_verify_accepts_same_results(evaluated):
    assert ev.main([*_base_args(evaluated), "--verify", str(evaluated["out"] / "metrics.json")]) == 0


def test_verify_rejects_altered_metrics(evaluated, tmp_path):
    altered = json.loads((evaluated["out"] / "metrics.json").read_text(encoding="utf-8"))
    altered["miou"] += 0.01
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    assert ev.main([*_base_args(evaluated), "--verify", str(path)]) == 1


def test_small_difference_is_inside_tolerance(evaluated):
    a = json.loads((evaluated["out"] / "metrics.json").read_text(encoding="utf-8"))
    b = json.loads(json.dumps(a))
    b["miou"] += 5e-4
    assert ev.compare_metrics(a, b) == []
    b["miou"] += 1e-3
    assert ev.compare_metrics(a, b)


def test_manifest_hash_mismatch_is_an_error(evaluated, tmp_path, capsys):
    other = tmp_path / "manifest.csv"
    other.write_text(evaluated["manifest"].read_text(encoding="utf-8") + "x,coco_stuff,train\n", encoding="utf-8")
    args = _base_args(evaluated)
    args[args.index("--manifest") + 1] = str(other)
    assert ev.main(args) == 3
    assert "no coincide" in capsys.readouterr().err


def test_modified_onnx_is_an_error(evaluated, tmp_path):
    models = tmp_path / "models"
    shutil.copytree(evaluated["out"], models)
    with open(models / "kinetia_seg.onnx", "ab") as f:
        f.write(b"\0")
    args = _base_args(evaluated)
    args[args.index("--models-dir") + 1] = str(models)
    assert ev.main(args) == 3


def test_limit_prints_not_official(evaluated, tmp_path, capsys):
    out = tmp_path / "limited.json"
    assert ev.main([*_base_args(evaluated), "--limit", "1", "--out", str(out)]) == 0
    assert "NO son las oficiales" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["num_images"] == 1


def test_missing_images_are_a_clear_error(evaluated, tmp_path, capsys):
    args = _base_args(evaluated)
    args[args.index("--data-root") + 1] = str(tmp_path / "vacio")
    assert ev.main(args) == 2
    assert "fetch_test_set" in capsys.readouterr().err
