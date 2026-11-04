"""Fixtures: un mini dataset con mascaras REALES de COCO-Stuff y BoWFire.

Las mascaras de tests/fixtures son archivos sin modificar de stuffthingmaps_trainval2017.zip y
de BoWFireDataset.zip. Las imagenes se sintetizan desde la mascara para no redistribuir fotos:
sirven para probar el flujo, no para medir nada del modelo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from dataset import (  # noqa: E402
    CLASSES,
    SOURCE_BOWFIRE,
    SOURCE_COCO,
    build_coco_lut,
    image_path,
    load_categories,
    load_mapping,
    mask_path,
    remap_bowfire_mask,
    remap_coco_mask,
    resize_image,
    resize_mask,
    write_manifest,
)

FIXTURES = ML_DIR / "tests" / "fixtures"


@pytest.fixture(scope="session")
def real_coco_masks() -> dict[str, np.ndarray]:
    return {p.stem: cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in sorted((FIXTURES / "coco_stuff").glob("*.png"))}


@pytest.fixture(scope="session")
def real_bowfire_masks() -> dict[str, np.ndarray]:
    return {p.stem.replace("_gt", ""): cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted((FIXTURES / "bowfire").glob("*.png"))}


@pytest.fixture(scope="session")
def mapping() -> dict:
    return load_mapping()


@pytest.fixture(scope="session")
def lut(mapping) -> np.ndarray:
    return build_coco_lut(mapping, load_categories())


@pytest.fixture(scope="session")
def mini_dataset(tmp_path_factory, real_coco_masks, real_bowfire_masks, mapping, lut):
    """Devuelve (raiz_de_datos, manifiesto, filas). Asigna train/val/test de forma fija."""
    root = tmp_path_factory.mktemp("mini")
    data = root / "data"
    rows = []
    coco_ids = list(real_coco_masks)
    fire_ids = list(real_bowfire_masks)
    plan = {
        **{i: "train" for i in coco_ids[:6]},
        **{coco_ids[6]: "val", coco_ids[7]: "val"},
        **{i: "test" for i in coco_ids[8:]},
        fire_ids[0]: "train",
        fire_ids[1]: "test",
    }
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, size=(256, 3), dtype=np.uint8)
    for image_id, split in plan.items():
        is_fire = image_id in real_bowfire_masks
        source = SOURCE_BOWFIRE if is_fire else SOURCE_COCO
        row = {"image_id": image_id, "source": source, "split": split}
        mask = remap_bowfire_mask(real_bowfire_masks[image_id], mapping) if is_fire else remap_coco_mask(real_coco_masks[image_id], lut)
        image = palette[mask]
        image = np.clip(image.astype(np.int16) + rng.integers(-15, 15, image.shape), 0, 255).astype(np.uint8)
        for path, content in ((image_path(data, row), resize_image(image)), (mask_path(data, row), resize_mask(mask))):
            path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(path), content)
        rows.append(row)
    manifest = root / "manifest.csv"
    write_manifest(rows, manifest)
    return data, manifest, rows


@pytest.fixture(scope="session")
def classes() -> list[str]:
    return CLASSES
