from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset import INPUT_SIZE, MEAN, STD, preprocess_seg, resize_image

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend" / "app" / "services" / "preprocess.py"


def _photo(h=480, w=640, seed=1):
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    return base


def test_shape_type_and_contract_formula():
    x = preprocess_seg(_photo())
    assert x.shape == (1, 3, INPUT_SIZE, INPUT_SIZE) and x.dtype == np.float32
    bgr = cv2.resize(_photo(), (320, 320), interpolation=cv2.INTER_LINEAR)
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    manual = np.transpose((rgb - MEAN) / STD, (2, 0, 1))[None]
    assert np.abs(x - manual).max() <= 1e-6


def test_uses_rgb_not_bgr():
    red_bgr = np.zeros((10, 10, 3), np.uint8)
    red_bgr[..., 2] = 255
    x = preprocess_seg(red_bgr)[0]
    red_expected = (1.0 - MEAN[0]) / STD[0]
    blue_expected = (0.0 - MEAN[2]) / STD[2]
    assert x[0].mean() == pytest.approx(red_expected, abs=1e-5), "El rojo debe caer en el primer canal"
    assert x[2].mean() == pytest.approx(blue_expected, abs=1e-5)


def test_stored_png_gives_same_input_as_original_photo(tmp_path):
    """El PNG de 320x320 en disco produce la misma entrada que la foto original en produccion."""
    photo = _photo()
    path = tmp_path / "x.png"
    assert cv2.imwrite(str(path), resize_image(photo))
    assert np.array_equal(preprocess_seg(cv2.imread(str(path), cv2.IMREAD_COLOR)), preprocess_seg(photo))


@pytest.mark.skipif(not BACKEND.exists(), reason="backend aun no existe")
def test_matches_backend_within_contract_tolerance():
    spec = importlib.util.spec_from_file_location("backend_preprocess", BACKEND)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for shape in ((480, 640), (1080, 1920), (320, 320), (333, 777)):
        photo = _photo(*shape)
        assert np.abs(preprocess_seg(photo) - module.preprocess_seg(photo)).max() <= 1e-5
