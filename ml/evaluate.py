"""Evalua el .onnx exportado sobre la particion de prueba y escribe metrics.json.

Uso:
    python ml/evaluate.py --models-dir models --data-root data/test                 # escribe metrics.json
    python ml/evaluate.py --models-dir models --data-root data/test --verify models/metrics.json
    python ml/evaluate.py --models-dir models --data-root data/test --limit 50      # NO oficial

Las metricas se calculan a 320x320 (la resolucion del modelo) y omiten IGNORE_INDEX.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from dataset import (
    CLASSES,
    DEFAULT_MANIFEST,
    EVALUATION_SPLITS,
    FUEGO_LUZ,
    NUM_CLASSES,
    SOURCE_BOWFIRE,
    SOURCE_COCO,
    atomic_write_json,
    file_sha256,
    image_path,
    limit_rows,
    manifest_sha256,
    mask_path,
    preprocess_seg,
    read_image_bgr,
    read_manifest,
    read_mask,
    resize_mask,
    rows_for_split,
)
from seg_metrics import confusion_update, metrics_from_confusion

VERIFY_TOLERANCE = 1e-3
NOT_OFFICIAL_BANNER = (
    "\n" + "=" * 72 + "\n"
    "  AVISO: se uso --limit. Estas cifras NO son las oficiales: no sirven para el\n"
    "  documento tecnico ni para la sustentacion. Las oficiales salen sin --limit.\n"
    + "=" * 72 + "\n"
)


def evaluate(onnx_path: Path, rows: list[dict], data_root: Path, batch_size: int = 8) -> dict:
    """Evalua y devuelve metricas, tiempos y una matriz de confusion por fuente."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    by_source = {SOURCE_COCO: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64), SOURCE_BOWFIRE: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)}
    start = time.time()
    last_print = start
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        x = np.concatenate([preprocess_seg(read_image_bgr(data_root, r)) for r in batch], axis=0)
        logits = session.run(["logits"], {"input": x})[0]
        pred = logits.argmax(axis=1)
        for row, p in zip(batch, pred):
            confusion_update(by_source[row["source"]], resize_mask(read_mask(data_root, row)), p)
        if time.time() - last_print > 30:
            last_print = time.time()
            done = i + len(batch)
            print(f"  {done}/{len(rows)} imagenes; {(last_print - start) / done:.3f} s/imagen", flush=True)
    return {"by_source": by_source, "seconds": time.time() - start}


def build_metrics(onnx_path: Path, manifest_hash: str, matrix: np.ndarray, num_images: int) -> dict:
    """Solo los campos del esquema de metrics.json del contrato."""
    m = metrics_from_confusion(matrix)
    return {
        "model_sha256": file_sha256(onnx_path),
        "dataset_manifest_sha256": manifest_hash,
        "split": "test",
        "num_images": num_images,
        "evaluated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "miou": m["miou"],
        "dice_macro": m["dice_macro"],
        "pixel_accuracy": m["pixel_accuracy"],
        "per_class": m["per_class"],
        "confusion_matrix": matrix.tolist(),
        "classes": CLASSES,
    }


def compare_metrics(expected: dict, got: dict, tolerance: float = VERIFY_TOLERANCE) -> list[str]:
    """Lista de diferencias: hashes y conteo deben ser iguales; las cifras, dentro de tolerancia."""
    problems: list[str] = []
    for key in ("model_sha256", "dataset_manifest_sha256", "split", "num_images", "classes"):
        if expected.get(key) != got.get(key):
            problems.append(f"{key}: archivo={expected.get(key)} calculado={got.get(key)}")
    for key in ("miou", "dice_macro", "pixel_accuracy"):
        if not math.isclose(expected[key], got[key], abs_tol=tolerance):
            problems.append(f"{key}: archivo={expected[key]:.6f} calculado={got[key]:.6f}")
    for name in CLASSES:
        for key in ("iou", "dice"):
            a, b = expected["per_class"][name][key], got["per_class"][name][key]
            if not math.isclose(a, b, abs_tol=tolerance):
                problems.append(f"{name}.{key}: archivo={a:.6f} calculado={b:.6f}")
    # La matriz se compara en proporcion del total, porque los conteos son enteros grandes.
    a = np.asarray(expected["confusion_matrix"], dtype=np.float64)
    b = np.asarray(got["confusion_matrix"], dtype=np.float64)
    if a.shape != b.shape:
        problems.append(f"confusion_matrix: formas {a.shape} y {b.shape}")
    else:
        worst = float(np.abs(a / max(a.sum(), 1) - b / max(b.sum(), 1)).max())
        if worst > tolerance:
            problems.append(f"confusion_matrix: diferencia maxima de proporcion {worst:.6f}")
    return problems


def print_summary(metrics: dict, by_source: dict, seconds: float, limited: bool) -> None:
    print(f"\nImagenes: {metrics['num_images']} | tiempo medido: {seconds:.1f} s ({seconds / max(metrics['num_images'], 1):.3f} s/imagen)")
    print(f"mIoU {metrics['miou']:.4f} | Dice macro {metrics['dice_macro']:.4f} | exactitud de pixel {metrics['pixel_accuracy']:.4f}")
    for name in CLASSES:
        c = metrics["per_class"][name]
        print(f"  {name:<11} IoU {c['iou']:.4f}  Dice {c['dice']:.4f}  soporte {c['support_pixels']}")
    fire = by_source[SOURCE_BOWFIRE]
    if fire.sum():
        recall = fire[FUEGO_LUZ, FUEGO_LUZ] / fire[FUEGO_LUZ].sum()
        print(f"Solo imagenes de fuego (BoWFire): {recall:.4f} de los pixeles de fuego se predicen como fuego_luz.")
        print("  Es recall: en BoWFire el resto de la imagen se ignora, asi que no mide falsos positivos.")
    light = by_source[SOURCE_COCO]
    if light[FUEGO_LUZ].sum():
        print(f"Solo COCO-Stuff: {light[FUEGO_LUZ, FUEGO_LUZ] / light[FUEGO_LUZ].sum():.4f} de los pixeles de luz se predicen como fuego_luz.")
    if limited:
        print(NOT_OFFICIAL_BANNER)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--data-root", type=Path, default=Path("data/test"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=None, help="Por defecto, <models-dir>/metrics.json")
    parser.add_argument("--verify", type=Path, default=None, help="Compara con este metrics.json, sin escribir")
    parser.add_argument("--limit", type=int, default=None, help="Solo N imagenes. Las cifras NO son las oficiales")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    onnx_path = args.models_dir / "kinetia_seg.onnx"
    model_manifest_path = args.models_dir / "manifest.json"
    for path in (onnx_path, model_manifest_path, args.manifest):
        if not path.exists():
            print(f"ERROR: no existe {path}", file=sys.stderr)
            return 2
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    manifest_hash = manifest_sha256(args.manifest)
    if model_manifest["dataset_manifest_sha256"] != manifest_hash:
        print(
            "ERROR: el hash de ml/splits/manifest.csv no coincide con el de manifest.json.\n"
            f"  manifest.csv : {manifest_hash}\n  manifest.json: {model_manifest['dataset_manifest_sha256']}",
            file=sys.stderr,
        )
        return 3
    actual = file_sha256(onnx_path)
    if model_manifest["files"]["kinetia_seg.onnx"]["sha256"] != actual:
        print(f"ERROR: {onnx_path} no coincide con el hash de manifest.json ({actual})", file=sys.stderr)
        return 3

    rows = rows_for_split(read_manifest(args.manifest), EVALUATION_SPLITS[0], EVALUATION_SPLITS)
    limited = args.limit is not None
    if limited:
        print(NOT_OFFICIAL_BANNER, flush=True)
        rows = limit_rows(rows, args.limit)

    missing = [r["image_id"] for r in rows if not (image_path(args.data_root, r).exists() and mask_path(args.data_root, r).exists())]
    if missing:
        print(f"ERROR: faltan {len(missing)} imagenes en {args.data_root} (ej. {missing[:3]}). Ejecuta ml/fetch_test_set.py.", file=sys.stderr)
        return 2

    result = evaluate(onnx_path, rows, args.data_root, args.batch_size)
    matrix = sum(result["by_source"].values())
    metrics = build_metrics(onnx_path, manifest_hash, matrix, len(rows))
    print_summary(metrics, result["by_source"], result["seconds"], limited)

    if args.verify is not None:
        expected = json.loads(args.verify.read_text(encoding="utf-8"))
        problems = compare_metrics(expected, metrics)
        if problems:
            print(f"VERIFICACION FALLIDA contra {args.verify}:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print(f"VERIFICACION CORRECTA: coincide con {args.verify} (tolerancia {VERIFY_TOLERANCE}).")
        return 0

    out = args.out or args.models_dir / "metrics.json"
    if limited and out.resolve().parent == Path("models").resolve():
        print("ERROR: unas cifras con --limit no pueden escribirse en models/.", file=sys.stderr)
        return 2
    atomic_write_json(out, metrics)
    print(f"Escrito {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
