"""train.py no puede abrir datos de la particion de prueba."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from dataset import TRAINING_SPLITS, ForbiddenSplitError, SegDataset, read_manifest, rows_for_split

ML_DIR = Path(__file__).resolve().parents[1]

# Registra con un evento de auditoria cada archivo abierto mientras train.py entrena de verdad.
AUDIT_RUNNER = r"""
import json, sys
opened = []
sys.addaudithook(lambda e, a: opened.append(str(a[0])) if e == "open" and a and isinstance(a[0], (str, bytes)) else None)
sys.path.insert(0, sys.argv[1])
import train
train.train(train.parse_args(sys.argv[3:]))
json.dump(sorted(set(p for p in opened if isinstance(p, str))), open(sys.argv[2], "w"))
"""


def test_training_really_runs_without_opening_test_files(mini_dataset, tmp_path):
    data, manifest, rows = mini_dataset
    log = tmp_path / "opened.json"
    cmd = [
        sys.executable, "-c", AUDIT_RUNNER, str(ML_DIR), str(log),
        "--data-root", str(data), "--manifest", str(manifest), "--out-dir", str(tmp_path / "run"),
        "--epochs", "1", "--batch-size", "2", "--num-workers", "0", "--no-pretrained", "--device", "cpu",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    opened = json.loads(log.read_text())
    png = [Path(p).name for p in opened if str(data) in p and p.endswith(".png")]
    train_ids = {r["image_id"] for r in rows if r["split"] == "train"}
    val_ids = {r["image_id"] for r in rows if r["split"] == "val"}
    test_ids = {r["image_id"] for r in rows if r["split"] == "test"}
    opened_ids = {Path(p).stem for p in png}
    assert opened_ids & train_ids, "El registro debe ver las imagenes de train"
    assert opened_ids & val_ids, "El registro debe ver las imagenes de val"
    assert not opened_ids & test_ids, f"train.py abrio datos de prueba: {opened_ids & test_ids}"


def test_dataset_refuses_test_rows_at_construction(mini_dataset):
    data, _, rows = mini_dataset
    test_rows = [r for r in rows if r["split"] == "test"]
    with pytest.raises(ForbiddenSplitError):
        SegDataset(data, test_rows, TRAINING_SPLITS)


def test_dataset_refuses_test_rows_at_read(mini_dataset):
    data, _, rows = mini_dataset
    row = dict(next(r for r in rows if r["split"] == "train"))
    ds = SegDataset(data, [row], TRAINING_SPLITS)
    row["split"] = "test"
    ds.rows[0]["split"] = "test"
    with pytest.raises(ForbiddenSplitError):
        ds[0]


def test_rows_for_split_refuses_test():
    with pytest.raises(ForbiddenSplitError):
        rows_for_split(read_manifest(), "test", TRAINING_SPLITS)


def test_training_splits_exclude_test():
    assert "test" not in TRAINING_SPLITS


def test_train_source_never_names_test_split():
    tree = ast.parse((ML_DIR / "train.py").read_text(encoding="utf-8"))
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "test" not in strings, "train.py no debe pedir la particion test"
    assert "EVALUATION_SPLITS" not in (ML_DIR / "train.py").read_text(encoding="utf-8")
