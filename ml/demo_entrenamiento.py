"""Demostración en vivo del entrenamiento, para la sustentación.

Entrena una copia NUEVA del modelo, desde cero (pesos aleatorios), con pocas fotos de la
partición de entrenamiento y en la CPU del portátil, usando exactamente train.py: el mismo
código, la misma pérdida, el mismo optimizador y los mismos aumentos que corrieron en Colab.
Solo cambian la cantidad de datos y de épocas, para que termine en pocos minutos.

No reemplaza el modelo de la app: escribe en data/runs/demo_<fecha>, nunca en models/.
Sus cifras NO son las del proyecto: con tan pocas fotos el modelo apenas empieza a aprender.
Lo que sirve mostrar es que la pérdida baja época tras época.

    ml\\.venv\\Scripts\\python.exe ml\\demo_entrenamiento.py
    ml\\.venv\\Scripts\\python.exe ml\\demo_entrenamiento.py --images 48 --epochs 6
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
REPO = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

import train  # noqa: E402  (después de ajustar sys.path, igual que los otros scripts de ml/)

BANNER = (
    "DEMOSTRACION: copia nueva del modelo, desde cero, con {images} fotos de train y {val} de validacion, "
    "{epochs} epocas en la CPU. No es el modelo de la app (ese se entreno 100 epocas en Colab con 10 583 fotos) "
    "y no lo reemplaza."
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=32, help="Fotos de entrenamiento (por defecto 32)")
    parser.add_argument("--val-images", type=int, default=16, help="Fotos de validacion (por defecto 16)")
    parser.add_argument("--epochs", type=int, default=8, help="Por defecto 8: unos 3 minutos en el portatil")
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=None, help="Por defecto data/runs/demo_<fecha>")
    return parser.parse_args(argv)


def train_args(args: argparse.Namespace, out_dir: Path) -> argparse.Namespace:
    """Los argumentos de train.py: sus valores por defecto, salvo datos, épocas y CPU."""
    return train.parse_args(
        [
            "--data-root", str(args.data_root),
            "--out-dir", str(out_dir),
            "--epochs", str(args.epochs),
            "--limit-train", str(args.images),
            "--limit-val", str(args.val_images),
            "--batch-size", "8",
            "--num-workers", "0",
            "--device", "cpu",
        ]
    )


def main(argv=None) -> dict:
    args = parse_args(argv)
    out_dir = args.out_dir or REPO / "data" / "runs" / f"demo_{datetime.now():%Y%m%d_%H%M%S}"
    if (Path(out_dir) / train.LAST_CKPT).exists():
        raise SystemExit(f"{out_dir} ya tiene un entrenamiento. La demo empieza siempre desde cero: usa otra carpeta.")
    print(BANNER.format(images=args.images, val=args.val_images, epochs=args.epochs), flush=True)
    print(f"Inicializacion: {train.INIT} (pesos aleatorios). Salida: {out_dir}", flush=True)
    start = time.time()
    state = train.train(train_args(args, Path(out_dir)))
    history = state["history"]
    if history:
        first, last = history[0], history[-1]
        print(
            f"\nResumen: perdida de entrenamiento {first['train_loss']:.4f} -> {last['train_loss']:.4f} "
            f"en {len(history)} epocas, {time.time() - start:.0f} s en total. "
            "Con tan pocas fotos el mIoU no es comparable con el del modelo de la app.",
            flush=True,
        )
    return state


if __name__ == "__main__":
    main()
