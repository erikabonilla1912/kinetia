"""Calcula la media y la desviacion por canal de NUESTRAS imagenes de entrenamiento.

El modelo se entrena desde cero, sin nada de ImageNet (decision 14 de docs/PROGRESO.md), asi
que la normalizacion tampoco usa los valores de ImageNet: sale de la particion `train`, con
el mismo preprocesado que el modelo (RGB, 320x320 con INTER_LINEAR, /255). Validacion y
prueba no se miran.

El resultado se copia a mano en `MEAN` y `STD` de ml/dataset.py y de
backend/app/services/preprocess.py. El notebook lo recalcula y se detiene si no coincide.

Uso:
    python ml/compute_norm_stats.py               # baja lo que falte a data/processed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from dataset import DEFAULT_MANIFEST, image_path, read_image_bgr, read_manifest, resize_image, rows_for_split, TRAINING_SPLITS

DECIMALS = 4


def channel_stats(data_root: Path, rows: list[dict]) -> dict:
    """Media y desviacion por canal RGB de todos los pixeles, en [0, 1], acumuladas en float64."""
    total = np.zeros(3, np.float64)
    total_sq = np.zeros(3, np.float64)
    count = 0
    for row in rows:
        rgb = cv2.cvtColor(read_image_bgr(data_root, row), cv2.COLOR_BGR2RGB)
        x = resize_image(rgb).reshape(-1, 3).astype(np.float64) / 255.0
        total += x.sum(axis=0)
        total_sq += (x * x).sum(axis=0)
        count += x.shape[0]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 0.0))
    return {
        "mean": [round(float(v), DECIMALS) for v in mean],
        "std": [round(float(v), DECIMALS) for v in std],
        "num_images": len(rows),
        "num_pixels": int(count),
        "split": "train",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--no-download", action="store_true", help="Falla si falta alguna imagen en vez de bajarla")
    args = parser.parse_args(argv)

    rows = rows_for_split(read_manifest(args.manifest), "train", TRAINING_SPLITS)
    missing = [r for r in rows if not image_path(args.data_root, r).exists()]
    if missing:
        if args.no_download:
            print(f"ERROR: faltan {len(missing)} imagenes de train en {args.data_root}", file=sys.stderr)
            return 2
        from prepare_data import prepare_rows

        print(f"Bajando {len(missing)} imagenes de train que faltan...", flush=True)
        prepare_rows(missing, args.data_root, args.raw_dir)
    stats = channel_stats(args.data_root, rows)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
