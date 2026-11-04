from __future__ import annotations

import collections

import numpy as np

from dataset import (
    DEFAULT_MANIFEST,
    SOURCES,
    limit_rows,
    manifest_sha256,
    read_manifest,
    write_manifest,
)
from make_splits import RATIOS, split_coco, split_groups


def test_columns_and_sources():
    rows = read_manifest()
    assert list(rows[0]) == ["image_id", "source", "split"]
    assert {r["source"] for r in rows} == set(SOURCES)
    assert {r["split"] for r in rows} == set(RATIOS)


def test_partitions_are_disjoint():
    rows = read_manifest()
    keys = [(r["source"], r["image_id"]) for r in rows]
    assert len(keys) == len(set(keys)), "Hay imagenes repetidas"
    by_split = {s: {(r["source"], r["image_id"]) for r in rows if r["split"] == s} for s in RATIOS}
    assert not by_split["train"] & by_split["val"]
    assert not by_split["train"] & by_split["test"]
    assert not by_split["val"] & by_split["test"]


def test_proportions_70_15_15_within_one_percent():
    rows = read_manifest()
    counts = collections.Counter(r["split"] for r in rows)
    for split, target in RATIOS.items():
        assert abs(counts[split] / len(rows) - target) <= 0.01, f"{split}: {counts[split] / len(rows):.3f}"


def test_coco_subset_size():
    rows = read_manifest()
    assert sum(r["source"] == "coco_stuff" for r in rows) == 15000


def test_bowfire_near_duplicates_stay_together():
    import json

    from make_splits import BOWFIRE_GROUPS

    groups = json.loads(BOWFIRE_GROUPS.read_text(encoding="utf-8"))
    split_of = {r["image_id"]: r["split"] for r in read_manifest() if r["source"] == "bowfire"}
    assert sorted(i for g in groups for i in g) == sorted(split_of), "Los grupos deben cubrir todo BoWFire"
    for group in groups:
        assert len({split_of[i] for i in group}) == 1, f"Grupo repartido entre particiones: {group}"


def test_hash_ignores_line_endings(tmp_path):
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    data = DEFAULT_MANIFEST.read_bytes().replace(b"\r\n", b"\n")
    lf.write_bytes(data)
    crlf.write_bytes(data.replace(b"\n", b"\r\n"))
    assert manifest_sha256(lf) == manifest_sha256(crlf)


def test_split_functions_are_reproducible():
    ids = [f"{i:06d}" for i in range(1000)]
    strata = [i % 5 for i in range(1000)]
    a = split_coco(ids, strata, np.random.default_rng(42))
    b = split_coco(ids, strata, np.random.default_rng(42))
    c = split_coco(ids, strata, np.random.default_rng(7))
    assert a == b and a != c
    groups = [[f"g{i}a", f"g{i}b"] for i in range(40)]
    assert split_groups(groups, np.random.default_rng(42)) == split_groups(groups, np.random.default_rng(42))


def test_limit_rows_keeps_every_source_and_is_stable():
    rows = read_manifest()
    subset = limit_rows(rows, 10)
    assert len(subset) == 10
    assert {r["source"] for r in subset} == set(SOURCES)
    assert subset == limit_rows(list(reversed(rows)), 10)


def test_write_manifest_uses_lf(tmp_path):
    path = tmp_path / "m.csv"
    write_manifest([{"image_id": "1", "source": "coco_stuff", "split": "train"}], path)
    assert b"\r" not in path.read_bytes()
