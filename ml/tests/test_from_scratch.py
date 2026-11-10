"""El modelo se entrena desde cero: sin transfer learning y sin nada tomado de ImageNet."""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

import compute_norm_stats as cns
import export as ex
import train as tr
import validate_artifacts as va
from dataset import INIT, MEAN, STD, manifest_sha256
from model import build_model

ML_DIR = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def test_weights_are_random_not_loaded():
    """Pesos cargados de un archivo serian iguales con cualquier semilla; aleatorios, no."""
    torch.manual_seed(1)
    a = build_model().state_dict()
    torch.manual_seed(2)
    b = build_model().state_dict()
    torch.manual_seed(1)
    c = build_model().state_dict()
    conv = next(k for k in a if k.startswith("encoder.") and k.endswith("weight") and a[k].dim() == 4)
    assert not torch.equal(a[conv], b[conv]), "El encoder no depende de la semilla: parecen pesos cargados"
    assert torch.equal(a[conv], c[conv])


def test_no_pretrained_option_left():
    for option in ("--no-pretrained", "--encoder-lr", "--freeze-encoder-epochs"):
        with pytest.raises(SystemExit):
            tr.parse_args([option, "1"])


def test_ml_sources_never_ask_for_imagenet_weights():
    for name in ("model.py", "train.py", "export.py"):
        tree = ast.parse((ML_DIR / name).read_text(encoding="utf-8"))
        strings = {n.value.lower() for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert "imagenet" not in strings, f"{name} pide pesos de ImageNet"


def test_normalization_is_not_imagenet():
    assert not np.allclose(MEAN, IMAGENET_MEAN, atol=1e-3)
    assert not np.allclose(STD, IMAGENET_STD, atol=1e-3)


def test_norm_stats_match_numpy(mini_dataset):
    data, _, rows = mini_dataset
    train_rows = [r for r in rows if r["split"] == "train"]
    stats = cns.channel_stats(data, train_rows)
    import cv2
    from dataset import read_image_bgr, resize_image

    pixels = np.concatenate([
        resize_image(cv2.cvtColor(read_image_bgr(data, r), cv2.COLOR_BGR2RGB)).reshape(-1, 3) / 255.0 for r in train_rows
    ])
    assert stats["mean"] == pytest.approx(pixels.mean(axis=0).tolist(), abs=1e-4)
    assert stats["std"] == pytest.approx(pixels.std(axis=0).tolist(), abs=1e-4)
    assert stats["num_images"] == len(train_rows)


def _run(mini_dataset, out, *extra):
    data, manifest, _ = mini_dataset
    return tr.train(tr.parse_args([
        "--data-root", str(data), "--manifest", str(manifest), "--out-dir", str(out), "--batch-size", "2",
        "--num-workers", "0", "--device", "cpu", "--limit-train", "4", "--limit-val", "2", *extra,
    ]))


def test_checkpoints_carry_the_mark_and_old_ones_are_refused(mini_dataset, tmp_path):
    run = tmp_path / "run"
    _run(mini_dataset, run, "--epochs", "2", "--stop-after-epochs", "1")
    ckpt = torch.load(run / tr.LAST_CKPT, weights_only=False)
    assert ckpt["init"] == INIT
    assert torch.load(run / tr.BEST_MODEL, weights_only=False)["init"] == INIT

    del ckpt["init"]  # asi se ve un checkpoint del modelo anterior
    torch.save(ckpt, run / tr.LAST_CKPT)
    (run / tr.PREV_CKPT).unlink(missing_ok=True)
    with pytest.raises(tr.InitMismatchError):
        _run(mini_dataset, run, "--epochs", "2")


def test_export_refuses_a_model_without_the_mark(mini_dataset, tmp_path):
    _, manifest, _ = mini_dataset
    tr.write_best_model(build_model(), tmp_path, 1, 0.1, manifest_sha256(manifest))
    payload = torch.load(tmp_path / tr.BEST_MODEL, weights_only=False)
    del payload["init"]
    old = tmp_path / "anterior.pt"
    torch.save(payload, old)
    with pytest.raises(ex.NotFromScratchError):
        ex.load_trained(old)


def test_manifest_schema_requires_the_mark_and_no_midas():
    good = {
        "version": "seg-2.0.0", "created_at": "2026-11-09T00:00:00Z", "init": INIT,
        "classes": va.CLASSES, "input_size": [320, 320],
        "mean": [round(float(v), 4) for v in MEAN], "std": [round(float(v), 4) for v in STD],
        "files": {"kinetia_seg.onnx": {"sha256": "a" * 64, "bytes": 10}},
        "dataset_manifest_sha256": "b" * 64,
    }
    assert va.validate_manifest(good) == []
    assert va.validate_manifest({**good, "init": "imagenet"})
    with_midas = {**good, "files": {**good["files"], "midas_small.onnx": {"sha256": "c" * 64, "bytes": 10}}}
    assert va.validate_manifest(with_midas)
    assert va.validate_manifest({**good, "mean": IMAGENET_MEAN, "std": IMAGENET_STD})
