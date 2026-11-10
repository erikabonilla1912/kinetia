"""La reanudacion continua en la misma epoca y con el mismo estado."""
from __future__ import annotations

import shutil

import pytest
import torch

import dataset as ds
import train as tr

COMMON = ["--batch-size", "2", "--num-workers", "0", "--device", "cpu", "--limit-train", "4", "--limit-val", "2"]


def _args(mini, out, *extra):
    data, manifest, _ = mini
    return tr.parse_args(["--data-root", str(data), "--manifest", str(manifest), "--out-dir", str(out), *COMMON, *extra])


@pytest.fixture(scope="module")
def runs(mini_dataset, tmp_path_factory):
    """A: 3 epocas seguidas. B: 2 epocas, un proceso nuevo logico, y luego hasta 3."""
    base = tmp_path_factory.mktemp("resume")
    a = base / "a"
    tr.train(_args(mini_dataset, a, "--epochs", "3"))
    b = base / "b"
    tr.train(_args(mini_dataset, b, "--epochs", "3", "--stop-after-epochs", "2"))
    checkpoint_after_2 = torch.load(b / tr.LAST_CKPT, weights_only=False)
    shutil.copytree(b, base / "b_stop")
    state = tr.train(_args(mini_dataset, b, "--epochs", "3"))
    return {"a": a, "b": b, "after2": checkpoint_after_2, "state_b": state, "stop": base / "b_stop"}


def test_checkpoint_holds_all_required_state(runs):
    ckpt = runs["after2"]
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "best_miou", "bad_epochs", "history", "epoch"):
        assert key in ckpt, f"Falta {key}"
    assert ckpt["epoch"] == 2
    assert {"python", "numpy", "torch"} <= set(ckpt["rng"])


def test_resume_continues_at_next_epoch(runs):
    history = runs["state_b"]["history"]
    assert [h["epoch"] for h in history] == [1, 2, 3], "No debe repetir ni saltarse epocas"
    assert history[:2] == runs["after2"]["history"], "El historial de antes del corte debe conservarse"


def test_resumed_run_matches_uninterrupted_run(runs):
    a = torch.load(runs["a"] / tr.LAST_CKPT, weights_only=False)
    b = torch.load(runs["b"] / tr.LAST_CKPT, weights_only=False)
    assert a["epoch"] == b["epoch"] == 3
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), f"Pesos distintos en {key}"
    for p, q in zip(a["optimizer"]["state"].values(), b["optimizer"]["state"].values()):
        assert torch.equal(p["step"], q["step"])
        assert torch.equal(p["exp_avg"], q["exp_avg"])
    assert a["scheduler"] == b["scheduler"]
    assert a["best_miou"] == b["best_miou"]
    assert a["bad_epochs"] == b["bad_epochs"]
    for ha, hb in zip(a["history"], b["history"]):
        assert ha["train_loss"] == pytest.approx(hb["train_loss"], abs=1e-12)
        assert ha["val_miou"] == pytest.approx(hb["val_miou"], abs=1e-12)


def test_finished_run_does_nothing_on_resume(runs, mini_dataset):
    before = (runs["a"] / tr.LAST_CKPT).read_bytes()
    state = tr.train(_args(mini_dataset, runs["a"], "--epochs", "3"))
    assert state["finished"] is True
    assert (runs["a"] / tr.LAST_CKPT).read_bytes() == before


def test_corrupt_last_checkpoint_falls_back_to_previous(runs):
    run = runs["stop"]
    (run / tr.LAST_CKPT).write_bytes(b"cortado a medias")
    ckpt = tr.load_checkpoint(run)
    assert ckpt is not None and ckpt["epoch"] == 1


def test_crash_while_writing_keeps_previous_file(tmp_path):
    path = tmp_path / "best.pt"
    tr.save_torch_atomic({"v": 1}, path)

    def dies_half_way(tmp: str) -> None:
        with open(tmp, "wb") as f:
            f.write(b"medio archivo")
        raise KeyboardInterrupt("corte de Colab")

    with pytest.raises(KeyboardInterrupt):
        ds.atomic_write(path, dies_half_way)
    assert torch.load(path, weights_only=False) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["best.pt"], "No deben quedar temporales"


def test_resume_refuses_other_manifest(runs, mini_dataset, tmp_path):
    data, manifest, _ = mini_dataset
    other = tmp_path / "other.csv"
    other.write_text(manifest.read_text() + "x,coco_stuff,val\n", encoding="utf-8")
    run = tmp_path / "run"
    shutil.copytree(runs["a"], run)
    args = _args(mini_dataset, run, "--epochs", "3")
    args.manifest = other
    args.limit_val = 2
    with pytest.raises(ds.ManifestMismatchError):
        tr.train(args)
