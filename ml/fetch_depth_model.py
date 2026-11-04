"""Descarga MiDaS v2.1 small ONNX de la release oficial y verifica su SHA-256.

No es un modelo entrenado por el equipo: se usa preentrenado, solo para el parallax.

Uso:
    python ml/fetch_depth_model.py                # escribe models/midas_small.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnxruntime as ort

from dataset import file_sha256
from prepare_data import download

MIDAS_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
MIDAS_BYTES = 66764249
# Calculado sobre el archivo descargado de la release oficial el 2026-11-02.
# La release no publica hash; este fija el archivo para que todas las maquinas usen el mismo.
MIDAS_SHA256 = "2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58"


def fetch(out: Path) -> Path:
    download(MIDAS_URL, out, MIDAS_BYTES)
    actual = file_sha256(out)
    if actual != MIDAS_SHA256:
        out.unlink()
        raise RuntimeError(f"SHA-256 inesperado en MiDaS: {actual}; se borro el archivo.")
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    if list(shape[1:]) != [3, 256, 256]:
        raise RuntimeError(f"Entrada de MiDaS inesperada: {shape}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models/midas_small.onnx"))
    args = parser.parse_args(argv)
    fetch(args.out)
    print(f"{args.out}: {args.out.stat().st_size} bytes, SHA-256 {MIDAS_SHA256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
