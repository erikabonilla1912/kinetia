"""Genera modelos ONNX dummy para desarrollo (docs/CONTRATOS.md, sección 2.3).

Solo necesita onnx y numpy. Los pesos son aleatorios: las máscaras y la profundidad
que producen NO significan nada y nunca deben presentarse como resultado.

    python ml/make_dummy_model.py                  # escribe en models/dev/
    python ml/make_dummy_model.py --out DIR --seed 7
    python ml/make_dummy_model.py --out DIR --heavy   # solo para medir rendimiento

`--heavy` añade capas convolucionales para que el costo de cómputo sea del orden de un
modelo real (unos 3 GMAC). Sigue siendo aleatorio y se usa solo para pruebas de carga.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CLASSES = ["fondo", "cielo", "agua", "vegetacion", "fuego_luz", "persona", "animal"]
SEG_SIZE = 320
DEPTH_SIZE = 256
OPSET = 17
IR_VERSION = 8
DUMMY_VERSION = "dummy-dev"
ZERO_HASH = "0" * 64
HEAVY_CHANNELS = 32
HEAVY_LAYERS = 3


def _conv(graph_nodes, inits, rng, name, source, in_ch, out_ch, gain=1.0):
    """Añade una convolución 3x3 con pesos aleatorios y devuelve el nombre de su salida."""
    weights = rng.normal(0.0, gain / np.sqrt(in_ch), size=(out_ch, in_ch, 3, 3)).astype(np.float32)
    bias = rng.normal(0.0, 0.1, size=(out_ch,)).astype(np.float32)
    inits.append(numpy_helper.from_array(weights, f"{name}_w"))
    inits.append(numpy_helper.from_array(bias, f"{name}_b"))
    output = f"{name}_out"
    graph_nodes.append(
        helper.make_node(
            "Conv",
            [source, f"{name}_w", f"{name}_b"],
            [output],
            name=name,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    )
    return output


def _backbone(nodes, inits, rng, source, heavy):
    channels = HEAVY_CHANNELS if heavy else 8
    current = _conv(nodes, inits, rng, "conv_in", source, 3, channels)
    for index in range(HEAVY_LAYERS if heavy else 0):
        current = _conv(nodes, inits, rng, f"conv_{index}", current, channels, channels)
        activated = f"relu_{index}"
        nodes.append(helper.make_node("Relu", [current], [activated], name=f"relu_node_{index}"))
        current = activated
    return current, channels


def _save(graph, path: Path) -> None:
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="kinetia-make-dummy-model",
    )
    model.ir_version = IR_VERSION
    model.doc_string = "Modelo dummy con pesos aleatorios. Solo desarrollo."
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def build_seg(path: Path, rng: np.random.Generator, heavy: bool) -> None:
    """Entrada `input` [N,3,320,320]; salida `logits` [N,7,320,320], sin softmax."""
    nodes, inits = [], []
    current, channels = _backbone(nodes, inits, rng, "input", heavy)
    # La ganancia alta produce logits separados, de modo que la máscara no quede toda en fondo.
    last = _conv(nodes, inits, rng, "conv_out", current, channels, len(CLASSES), gain=6.0)
    nodes.append(helper.make_node("Identity", [last], ["logits"], name="logits_node"))
    graph = helper.make_graph(
        nodes,
        "kinetia_seg_dummy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, SEG_SIZE, SEG_SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", len(CLASSES), SEG_SIZE, SEG_SIZE])],
        initializer=inits,
    )
    _save(graph, path)


def build_depth(path: Path, rng: np.random.Generator, heavy: bool) -> None:
    """Entrada [1,3,256,256]; salida [1,256,256], igual que el MiDaS v2.1 small oficial."""
    nodes, inits = [], []
    current, channels = _backbone(nodes, inits, rng, "input", heavy)
    last = _conv(nodes, inits, rng, "conv_out", current, channels, 1)
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "squeeze_axes"))
    nodes.append(helper.make_node("Squeeze", [last, "squeeze_axes"], ["depth"], name="squeeze"))
    graph = helper.make_graph(
        nodes,
        "midas_small_dummy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, DEPTH_SIZE, DEPTH_SIZE])],
        [helper.make_tensor_value_info("depth", TensorProto.FLOAT, [1, DEPTH_SIZE, DEPTH_SIZE])],
        initializer=inits,
    )
    _save(graph, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_manifest(out: Path, seg: Path, depth: Path, version: str) -> None:
    manifest = {
        "version": version,
        "created_at": _now(),
        "classes": CLASSES,
        "input_size": [SEG_SIZE, SEG_SIZE],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "files": {
            seg.name: {"sha256": _sha256(seg), "bytes": seg.stat().st_size},
            depth.name: {"sha256": _sha256(depth), "bytes": depth.stat().st_size},
        },
        "dataset_manifest_sha256": ZERO_HASH,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_metrics(out: Path, seg: Path) -> None:
    """Mismo esquema de metrics.json, sin ningún resultado: métricas nulas y `dummy: true`."""
    metrics = {
        "dummy": True,
        "model_sha256": _sha256(seg),
        "dataset_manifest_sha256": ZERO_HASH,
        "split": "dummy",
        "num_images": 0,
        "evaluated_at": _now(),
        "miou": None,
        "dice_macro": None,
        "pixel_accuracy": None,
        "per_class": {name: {"iou": None, "dice": None, "support_pixels": 0} for name in CLASSES},
        "confusion_matrix": [[0] * len(CLASSES) for _ in CLASSES],
        "classes": CLASSES,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=root / "models" / "dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default=DUMMY_VERSION)
    parser.add_argument("--heavy", action="store_true", help="Costo de cómputo similar al real; solo para medir")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    seg = args.out / "kinetia_seg.onnx"
    depth = args.out / "midas_small.onnx"
    build_seg(seg, rng, args.heavy)
    build_depth(depth, rng, args.heavy)
    write_manifest(args.out, seg, depth, args.version)
    write_metrics(args.out, seg)
    print(f"Modelos dummy ({args.version}) escritos en {args.out}. Pesos aleatorios: no son resultados.")


if __name__ == "__main__":
    main()
