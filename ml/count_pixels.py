"""Conteo real de pixeles por clase tras el remapeo. Escribe ml/reports/class_pixel_counts.json.

Sale de leer cada mascara de COCO-Stuff y BoWFire a su resolucion nativa, sin estimaciones.
Tambien busca una categoria de fuego en la lista real de categorias y comprueba que
valor de PNG = category_id - 1 comparando con las anotaciones de instancias de COCO.

Uso:
    python ml/count_pixels.py --stuff-zip data/raw/stuffthingmaps_trainval2017.zip \
                              --bowfire-zip data/_fire_candidates/bowfire/BoWFireDataset.zip
"""
from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

from dataset import (
    CLASSES,
    IGNORE_INDEX,
    SOURCE_BOWFIRE,
    SOURCE_COCO,
    atomic_write_json,
    load_categories,
    load_mapping,
    manifest_sha256,
    read_manifest,
    remap_bowfire_mask,
)
from make_splits import class_histograms, compute_histograms

FIRE_PATTERN = re.compile(r"fire|flame|burn|blaze|ember|smoke|candle|torch", re.I)
INSTANCES_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


class _HttpFile(io.RawIOBase):
    """Archivo de solo lectura sobre HTTP Range, para leer un JSON de un zip enorme."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.size = int(head.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req, timeout=120).read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def verify_png_offset(names: np.ndarray, hist: np.ndarray) -> dict:
    """Compara el area de `person` en el PNG con el area de las instancias reales de val2017."""
    with zipfile.ZipFile(io.BufferedReader(_HttpFile(INSTANCES_ZIP), buffer_size=1 << 20)) as zf:
        data = json.loads(zf.read("annotations/instances_val2017.json"))
    categories = {c["name"]: c["id"] for c in data["categories"]}
    index = {int(Path(str(n)).stem): i for i, n in enumerate(names) if "val2017" in str(n)}
    results = {}
    for name in ("person", "dog", "airplane", "toilet"):
        cid = categories[name]
        per_image: dict[int, float] = {}
        for ann in data["annotations"]:
            if ann["category_id"] == cid and not ann["iscrowd"]:
                per_image[ann["image_id"]] = per_image.get(ann["image_id"], 0.0) + ann["area"]
        ids = [i for i in per_image if i in index]
        polygon_area = np.array([per_image[i] for i in ids])
        png_area = np.array([hist[index[i], cid - 1] for i in ids], dtype=np.float64)
        results[name] = {
            "category_id": cid,
            "images": len(ids),
            "correlation_png_value_id_minus_1": float(np.corrcoef(polygon_area, png_area)[0, 1]),
            "total_area_ratio_png_over_annotations": float(png_area.sum() / polygon_area.sum()),
        }
    return results


def table(per_class: np.ndarray, unlabeled: int) -> dict:
    labeled = int(per_class.sum())
    total = labeled + int(unlabeled)
    return {
        "per_class": {
            name: {
                "pixels": int(per_class[i]),
                "pct_of_labeled": round(100.0 * int(per_class[i]) / labeled, 4) if labeled else 0.0,
            }
            for i, name in enumerate(CLASSES)
        },
        "ignore_255_pixels": int(unlabeled),
        "ignore_pct_of_total": round(100.0 * int(unlabeled) / total, 4) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stuff-zip", type=Path, required=True)
    parser.add_argument("--bowfire-zip", type=Path, required=True)
    parser.add_argument("--hist-cache", type=Path, default=Path("data/raw/coco_png_histograms.npz"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "reports" / "class_pixel_counts.json")
    parser.add_argument("--skip-offset-check", action="store_true")
    args = parser.parse_args()

    categories = load_categories()
    names, hist = compute_histograms(args.stuff_zip, args.hist_cache)
    per_class, unlabeled = class_histograms(hist)
    ids = {Path(str(n)).stem: i for i, n in enumerate(names)}

    raw_totals = hist.sum(axis=0)
    fire_matches = {name: int(raw_totals[cid - 1]) for cid, name in categories.items() if FIRE_PATTERN.search(name)}
    light_id = next(c for c, n in categories.items() if n == "light")

    manifest = read_manifest()
    report = {
        "generated_by": "ml/count_pixels.py",
        "resolution": "nativa de cada mascara",
        "dataset_manifest_sha256": manifest_sha256(),
        "fire_category_check": {
            "categories_total": len(categories),
            "pattern_searched": FIRE_PATTERN.pattern,
            "categories_matching_with_pixels": fire_matches,
            "has_fire_category": any(name != "fire hydrant" for name in fire_matches),
            "note": "`fire hydrant` es un hidrante. COCO-Stuff no tiene categoria de fuego.",
            "light_category_id": light_id,
            "light_pixels": int(raw_totals[light_id - 1]),
            "images_with_light": int((hist[:, light_id - 1] > 0).sum()),
        },
        "all_coco_stuff_trainval": {"num_images": len(names), **table(per_class.sum(axis=0), unlabeled.sum())},
    }

    mapping = load_mapping()
    fire_px: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(args.bowfire_zip) as zf:
        for row in manifest:
            if row["source"] != SOURCE_BOWFIRE:
                continue
            gt = cv2.imdecode(
                np.frombuffer(zf.read(f"BoWFireDataset/dataset/gt/{row['image_id']}_gt.png"), np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
            remapped = remap_bowfire_mask(gt, mapping)
            fire_px[row["image_id"]] = np.bincount(remapped.ravel(), minlength=256)

    for split in ("all", "train", "val", "test"):
        rows = [r for r in manifest if split == "all" or r["split"] == split]
        coco_rows = [ids[r["image_id"]] for r in rows if r["source"] == SOURCE_COCO]
        fire_rows = [fire_px[r["image_id"]] for r in rows if r["source"] == SOURCE_BOWFIRE]
        counts = per_class[coco_rows].sum(axis=0)
        ignored = int(unlabeled[coco_rows].sum())
        fire_only = np.zeros(len(CLASSES), dtype=np.int64)
        for h in fire_rows:
            fire_only += h[: len(CLASSES)]
            ignored += int(h[IGNORE_INDEX])
        report[f"selected_{split}"] = {
            "num_images": {SOURCE_COCO: len(coco_rows), SOURCE_BOWFIRE: len(fire_rows)},
            "fuego_luz_from_coco_light": int(counts[CLASSES.index("fuego_luz")]),
            "fuego_luz_from_bowfire_fire": int(fire_only[CLASSES.index("fuego_luz")]),
            **table(counts + fire_only, ignored),
        }

    if not args.skip_offset_check:
        report["png_value_equals_category_id_minus_1_check"] = verify_png_offset(names, hist)

    atomic_write_json(args.out, report)
    print(json.dumps({k: report[k] for k in report if k != "dataset_manifest_sha256"}, indent=1, ensure_ascii=False))
    print(f"Escrito {args.out}")


if __name__ == "__main__":
    main()
