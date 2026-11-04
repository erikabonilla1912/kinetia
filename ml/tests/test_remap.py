from __future__ import annotations

import numpy as np
import pytest

from dataset import (
    CLASSES,
    IGNORE_INDEX,
    build_coco_lut,
    load_categories,
    load_mapping,
    remap_bowfire_mask,
    remap_coco_mask,
)
from seg_metrics import confusion_update, metrics_from_confusion


def test_real_coco_masks_give_only_valid_indices(real_coco_masks, lut):
    assert len(real_coco_masks) >= 10
    seen = set()
    for raw in real_coco_masks.values():
        values = set(np.unique(remap_coco_mask(raw, lut)).tolist())
        assert values <= set(range(7)) | {IGNORE_INDEX}
        seen |= values
    assert seen == set(range(7)) | {IGNORE_INDEX}, f"Las fixtures deben cubrir todas las clases: {seen}"


def test_remap_follows_category_names(real_coco_masks, lut):
    """Comprobacion independiente de la LUT, usando los nombres de categoria."""
    names = load_categories()
    groups = load_mapping()["coco_stuff"]["mapping"]
    expected = {name: CLASSES.index(c) for c, group in groups.items() for name in group}
    for raw in real_coco_masks.values():
        remapped = remap_coco_mask(raw, lut)
        for value in np.unique(raw).tolist():
            target = IGNORE_INDEX if value == 255 else expected.get(names[value + 1], 0)
            assert (remapped[raw == value] == target).all(), f"{value} -> {target}"


def test_mapping_uses_only_real_categories():
    categories = set(load_categories().values())
    for group in load_mapping()["coco_stuff"]["mapping"].values():
        assert set(group) <= categories


def test_no_fire_category_in_coco_stuff():
    names = load_categories().values()
    assert not [n for n in names if n in ("fire", "flame", "flames", "bonfire", "campfire", "smoke", "candle")]
    assert "light" in names and "fire hydrant" in names


def test_unknown_png_value_raises(lut):
    bad = np.array([[11]], dtype=np.uint8)  # id 12 no existe en COCO: valor 11
    with pytest.raises(ValueError):
        remap_coco_mask(bad, lut)


def test_lut_rejects_unknown_category():
    mapping = load_mapping()
    mapping["coco_stuff"]["mapping"]["cielo"] = ["no-existe"]
    with pytest.raises(KeyError):
        build_coco_lut(mapping, load_categories())


def test_real_bowfire_masks_only_fire_counts(real_bowfire_masks, mapping):
    assert len(real_bowfire_masks) >= 2
    for gt in real_bowfire_masks.values():
        out = remap_bowfire_mask(gt, mapping)
        assert set(np.unique(out).tolist()) == {CLASSES.index("fuego_luz"), IGNORE_INDEX}
        assert (out[gt > 127] == CLASSES.index("fuego_luz")).all()
        assert (out[gt <= 127] == IGNORE_INDEX).all()


def test_metrics_omit_ignore_and_match_hand_computed_values():
    truth = np.array([[0, 0, 1, 255]], dtype=np.uint8)
    pred = np.array([[0, 1, 1, 1]], dtype=np.int64)
    matrix = np.zeros((7, 7), dtype=np.int64)
    confusion_update(matrix, truth, pred)
    assert matrix.sum() == 3
    m = metrics_from_confusion(matrix)
    assert m["per_class"]["fondo"]["iou"] == pytest.approx(1 / 2)
    assert m["per_class"]["cielo"]["iou"] == pytest.approx(1 / 2)
    assert m["pixel_accuracy"] == pytest.approx(2 / 3)
    assert m["miou"] == pytest.approx(1 / 2), "Las clases sin datos no deben promediar"
    assert m["per_class"]["fondo"]["dice"] == pytest.approx(2 / 3)
