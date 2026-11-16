"""La demostración en vivo usa train.py tal cual y nunca toca el modelo de la app."""
from pathlib import Path

import pytest

import demo_entrenamiento as demo
import train


def test_usa_los_valores_de_train_salvo_datos_epocas_y_cpu(tmp_path):
    args = demo.parse_args([])
    got = demo.train_args(args, tmp_path / "demo")
    defaults = train.parse_args([])
    assert (got.limit_train, got.limit_val, got.epochs, got.device) == (32, 16, 8, "cpu")
    # Lo que define el entrenamiento es igual al de Colab.
    for name in ("lr", "weight_decay", "min_delta", "patience", "seed", "manifest"):
        assert getattr(got, name) == getattr(defaults, name), name


def test_por_defecto_escribe_en_data_runs_y_no_en_models(monkeypatch):
    seen = {}
    monkeypatch.setattr(train, "train", lambda a: seen.setdefault("out", Path(a.out_dir)) and {"history": []})
    demo.main([])
    out = seen["out"].resolve()
    assert out.parent == (demo.REPO / "data" / "runs").resolve()
    assert out.name.startswith("demo_")
    assert (demo.REPO / "models").resolve() not in out.parents


def test_no_reanuda_una_carpeta_con_entrenamiento(tmp_path, monkeypatch):
    (tmp_path / train.LAST_CKPT).write_bytes(b"x")
    monkeypatch.setattr(train, "train", lambda a: pytest.fail("no debe entrenar"))
    with pytest.raises(SystemExit):
        demo.main(["--out-dir", str(tmp_path)])
