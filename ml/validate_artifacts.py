"""Comprueba que los 5 archivos que entrega el notebook cumplen los esquemas del contrato."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dataset import CLASSES, INIT, MEAN, STD, file_sha256

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
ARTIFACTS = ("kinetia_seg.pt", "kinetia_seg.onnx", "metrics.json", "manifest.json", "hashes.sha256")


def validate_manifest(data: dict) -> list[str]:
    p: list[str] = []
    keys = {"version", "created_at", "init", "classes", "input_size", "mean", "std", "files", "dataset_manifest_sha256"}
    if set(data) != keys:
        p.append(f"manifest.json: campos {sorted(set(data) ^ keys)} sobran o faltan")
        return p
    if data["init"] != INIT:
        p.append(f"manifest.json: init debe ser {INIT!r} (modelo entrenado desde cero)")
    if data["classes"] != CLASSES:
        p.append("manifest.json: clases incorrectas")
    if data["input_size"] != [320, 320]:
        p.append("manifest.json: input_size incorrecto")
    if data["mean"] != [round(float(v), 4) for v in MEAN] or data["std"] != [round(float(v), 4) for v in STD]:
        p.append("manifest.json: mean/std no son los de ml/dataset.py")
    if not ISO.match(data["created_at"]):
        p.append("manifest.json: created_at no es ISO 8601 UTC")
    if set(data["files"]) != {"kinetia_seg.onnx"}:
        p.append("manifest.json: files incorrecto")
    for name, info in data["files"].items():
        if not HEX64.match(info["sha256"]) or not isinstance(info["bytes"], int) or info["bytes"] <= 0:
            p.append(f"manifest.json: entrada invalida para {name}")
    if not HEX64.match(data["dataset_manifest_sha256"]):
        p.append("manifest.json: dataset_manifest_sha256 invalido")
    return p


def validate_metrics(data: dict) -> list[str]:
    p: list[str] = []
    keys = {
        "model_sha256", "dataset_manifest_sha256", "split", "num_images", "evaluated_at", "miou",
        "dice_macro", "pixel_accuracy", "per_class", "confusion_matrix", "classes",
    }
    if set(data) != keys:
        p.append(f"metrics.json: campos {sorted(set(data) ^ keys)} sobran o faltan")
        return p
    if not HEX64.match(data["model_sha256"]) or not HEX64.match(data["dataset_manifest_sha256"]):
        p.append("metrics.json: hash invalido")
    if data["split"] != "test":
        p.append("metrics.json: split debe ser test")
    if data["classes"] != CLASSES:
        p.append("metrics.json: clases incorrectas")
    if not ISO.match(data["evaluated_at"]):
        p.append("metrics.json: evaluated_at no es ISO 8601 UTC")
    if not isinstance(data["num_images"], int) or data["num_images"] <= 0:
        p.append("metrics.json: num_images invalido")
    for key in ("miou", "dice_macro", "pixel_accuracy"):
        if not 0.0 <= data[key] <= 1.0:
            p.append(f"metrics.json: {key} fuera de [0, 1]")
    if list(data["per_class"]) != CLASSES:
        p.append("metrics.json: per_class debe traer las 7 clases en orden")
    else:
        for name, c in data["per_class"].items():
            if set(c) != {"iou", "dice", "support_pixels"}:
                p.append(f"metrics.json: campos invalidos en {name}")
            elif not (0.0 <= c["iou"] <= 1.0 and 0.0 <= c["dice"] <= 1.0 and isinstance(c["support_pixels"], int)):
                p.append(f"metrics.json: valores invalidos en {name}")
    matrix = data["confusion_matrix"]
    if len(matrix) != 7 or any(len(r) != 7 or not all(isinstance(v, int) for v in r) for r in matrix):
        p.append("metrics.json: confusion_matrix debe ser 7x7 de enteros")
    elif [sum(r) for r in matrix] != [data["per_class"][n]["support_pixels"] for n in CLASSES]:
        p.append("metrics.json: el soporte no coincide con la matriz")
    return p


def validate_dir(directory: Path) -> list[str]:
    directory = Path(directory)
    problems = [f"Falta {name}" for name in ARTIFACTS if not (directory / name).exists()]
    if problems:
        return problems
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    problems += validate_manifest(manifest) + validate_metrics(metrics)
    onnx_hash = file_sha256(directory / "kinetia_seg.onnx")
    if manifest["files"]["kinetia_seg.onnx"]["sha256"] != onnx_hash:
        problems.append("manifest.json no coincide con kinetia_seg.onnx")
    if metrics["model_sha256"] != onnx_hash:
        problems.append("metrics.json no se calculo sobre este kinetia_seg.onnx")
    if metrics["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"]:
        problems.append("metrics.json y manifest.json usan manifiestos de datos distintos")
    expected_hashes = "".join(f"{file_sha256(directory / n)}  {n}\n" for n in ("kinetia_seg.pt", "kinetia_seg.onnx"))
    if (directory / "hashes.sha256").read_bytes().decode("ascii") != expected_hashes:
        problems.append("hashes.sha256 no coincide con los modelos")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    problems = validate_dir(args.directory)
    for problem in problems:
        print(f"ERROR: {problem}")
    if not problems:
        print(f"Los 5 archivos de {args.directory} cumplen los esquemas del contrato.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
