"""Descarga SOLO el conjunto de prueba a data/test, para correr evaluate.py en este portatil.

Baja cada imagen de prueba por separado, y una vez el zip de mascaras de COCO-Stuff (659 MB)
y el de BoWFire (291 MB), porque esos autores no ofrecen mascaras sueltas. Solo guarda lo de
prueba. Con --clean-raw borra los zip al terminar.

Uso:
    python ml/fetch_test_set.py
    python ml/evaluate.py --models-dir models --data-root data/test --verify models/metrics.json
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from dataset import DEFAULT_MANIFEST, EVALUATION_SPLITS, image_path, limit_rows, mask_path, read_manifest, rows_for_split
from prepare_data import check_prepared, prepare_rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/test"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Solo N imagenes: sirve para probar, no para las cifras oficiales")
    parser.add_argument("--clean-raw", action="store_true", help="Borra los zip descargados al terminar")
    args = parser.parse_args(argv)

    rows = rows_for_split(read_manifest(args.manifest), EVALUATION_SPLITS[0], EVALUATION_SPLITS)
    rows = limit_rows(rows, args.limit)
    prepare_rows(rows, args.out, args.raw_dir, args.threads)
    missing = check_prepared(rows, args.out)
    if missing:
        print(f"ERROR: faltan {len(missing)} imagenes: {missing[:5]}", file=sys.stderr)
        return 1
    size = sum(image_path(args.out, r).stat().st_size + mask_path(args.out, r).stat().st_size for r in rows)
    print(f"Conjunto de prueba listo en {args.out}: {len(rows)} imagenes, {size / 1e6:.0f} MB (medido)")
    if args.clean_raw:
        shutil.rmtree(args.raw_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
