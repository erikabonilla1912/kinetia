"""Exporta el mejor checkpoint a .pt y .onnx, y escribe manifest.json y hashes.sha256.

Uso:
    python ml/export.py --checkpoint <carpeta>/kinetia_seg.pt --midas models/midas_small.onnx --out-dir models
"""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from dataset import (
    CLASSES,
    DEFAULT_MANIFEST,
    INPUT_SIZE,
    MEAN,
    NUM_CLASSES,
    STD,
    atomic_write,
    atomic_write_json,
    file_sha256,
    manifest_sha256,
)
from model import build_model

OPSET = 17
VERIFY_TOLERANCE = 1e-4
SEG_ONNX = "kinetia_seg.onnx"
SEG_PT = "kinetia_seg.pt"
MIDAS_ONNX = "midas_small.onnx"


def load_trained(path: Path) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    """Grafo con N dinamico, entradas `input` y salidas `logits`, sin softmax."""
    dummy = torch.zeros(2, 3, INPUT_SIZE, INPUT_SIZE)
    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )
    # PyTorch >= 2.9 usa por defecto otro exportador; se pide el clasico cuando existe la opcion,
    # y no se reinstala PyTorch en Colab.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    atomic_write(path, lambda tmp: torch.onnx.export(model, dummy, tmp, **kwargs))


def check_onnx_signature(path: Path) -> None:
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    opset = max(o.version for o in graph.opset_import if o.domain in ("", "ai.onnx"))
    if opset < OPSET:
        raise RuntimeError(f"Opset {opset} menor que {OPSET}")
    names_in = [i.name for i in graph.graph.input]
    names_out = [o.name for o in graph.graph.output]
    if names_in != ["input"] or names_out != ["logits"]:
        raise RuntimeError(f"Nombres incorrectos: {names_in} -> {names_out}")
    if any(n.op_type == "Softmax" for n in graph.graph.node):
        raise RuntimeError("El ONNX contiene un Softmax: el contrato pide logits")


def verify_pt_vs_onnx(model: torch.nn.Module, path: Path, tolerance: float = VERIFY_TOLERANCE) -> float:
    """Compara con lotes de distinto tamano. Devuelve la mayor diferencia absoluta."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for n in (1, 2, 3):
        x = rng.normal(size=(n, 3, INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        got = session.run(["logits"], {"input": x})[0]
        if got.shape != (n, NUM_CLASSES, INPUT_SIZE, INPUT_SIZE) or got.dtype != np.float32:
            raise RuntimeError(f"Salida ONNX inesperada: {got.shape} {got.dtype}")
        difference = np.abs(expected - got)
        if not np.isfinite(difference).all():  # NaN pasaria un max() y parecería que coinciden
            raise RuntimeError("Hay valores no finitos en la salida del .pt o del .onnx")
        worst = max(worst, float(difference.max()))
    if worst > tolerance:
        raise RuntimeError(f"ONNX y .pt difieren en {worst:.2e} (tolerancia {tolerance:.0e})")
    return worst


def write_manifest_json(out_dir: Path, midas: Path, manifest_csv: Path, version: str) -> dict:
    seg = out_dir / SEG_ONNX
    data = {
        "version": version,
        "created_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "classes": CLASSES,
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "mean": [round(float(v), 3) for v in MEAN],
        "std": [round(float(v), 3) for v in STD],
        "files": {
            SEG_ONNX: {"sha256": file_sha256(seg), "bytes": seg.stat().st_size},
            MIDAS_ONNX: {"sha256": file_sha256(midas), "bytes": Path(midas).stat().st_size},
        },
        "dataset_manifest_sha256": manifest_sha256(manifest_csv),
    }
    atomic_write_json(out_dir / "manifest.json", data)
    return data


def write_hashes(out_dir: Path) -> None:
    lines = "".join(f"{file_sha256(out_dir / name)}  {name}\n" for name in (SEG_PT, SEG_ONNX))

    def writer(tmp: str) -> None:
        with open(tmp, "w", encoding="ascii", newline="\n") as f:
            f.write(lines)

    atomic_write(out_dir / "hashes.sha256", writer)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--midas", type=Path, required=True, help="midas_small.onnx (ver fetch_depth_model.py)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--version", default="seg-1.0.0")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, payload = load_trained(args.checkpoint)
    if payload["dataset_manifest_sha256"] != manifest_sha256(args.manifest):
        sys.exit("ERROR: el checkpoint se entreno con otro manifiesto de datos.")

    pt_out = args.out_dir / SEG_PT
    if args.checkpoint.resolve() != pt_out.resolve():
        atomic_write(pt_out, lambda tmp: shutil.copyfile(args.checkpoint, tmp))
    onnx_out = args.out_dir / SEG_ONNX
    export_onnx(model, onnx_out)
    check_onnx_signature(onnx_out)
    worst = verify_pt_vs_onnx(model, onnx_out)
    print(f".pt vs .onnx: diferencia maxima {worst:.2e} (tolerancia {VERIFY_TOLERANCE:.0e}), N dinamico comprobado")

    manifest = write_manifest_json(args.out_dir, args.midas, args.manifest, args.version)
    write_hashes(args.out_dir)
    print(f"SHA-256 kinetia_seg.pt:   {file_sha256(pt_out)}")
    print(f"SHA-256 kinetia_seg.onnx: {manifest['files'][SEG_ONNX]['sha256']}")
    print(f"Escritos en {args.out_dir}: {SEG_PT}, {SEG_ONNX}, manifest.json, hashes.sha256")


if __name__ == "__main__":
    main()
