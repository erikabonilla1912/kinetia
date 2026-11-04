"""Selecciona ~15 000 imagenes de COCO-Stuff y genera ml/splits/manifest.csv (70/15/15, seed=42).

Por que no se usan las particiones oficiales de COCO:
  1. El conjunto `test2017` oficial no tiene mascaras publicas, asi que no sirve para medir.
  2. Solo se usa un subconjunto reforzado en clases escasas; train2017 y val2017 se tratan
     como una sola bolsa y se reparten de nuevo con semilla fija. Asi las 3 particiones
     tienen la misma distribucion de clases y BoWFire puede repartirse en las tres.

El manifiesto se genera SOLO con anotaciones: no hace falta descargar imagenes de COCO.

Uso:
    python ml/make_splits.py --stuff-zip data/raw/stuffthingmaps_trainval2017.zip \
                             --bowfire-zip data/_fire_candidates/bowfire/BoWFireDataset.zip
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from dataset import (
    CLASSES,
    DEFAULT_MANIFEST,
    FUEGO_LUZ,
    NUM_CLASSES,
    SOURCE_BOWFIRE,
    SOURCE_COCO,
    build_coco_lut,
    load_categories,
    load_mapping,
    write_manifest,
)

SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
BOWFIRE_GROUPS = Path(__file__).resolve().parent / "splits" / "bowfire_groups.json"

# Imagenes pedidas para cada clase escasa y area minima (fraccion de la imagen) para contar.
# Se piden en orden de escasez. Las imagenes que ya entraron por otra clase cuentan para esta.
QUOTAS = [
    ("fuego_luz", 4000, 0.001),
    ("agua", 3000, 0.02),
    ("animal", 3500, 0.01),
    ("cielo", 2500, 0.05),
    ("vegetacion", 2500, 0.05),
    ("persona", 2500, 0.02),
]
HIST_SIZE = 256


def _hist_worker(args: tuple[str, list[str]]) -> tuple[list[str], np.ndarray]:
    zip_path, names = args
    out = np.zeros((len(names), HIST_SIZE), dtype=np.int64)
    with zipfile.ZipFile(zip_path) as zf:
        for i, name in enumerate(names):
            raw = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_UNCHANGED)
            out[i] = np.bincount(raw.ravel(), minlength=HIST_SIZE)
    return names, out


def compute_histograms(zip_path: Path, cache: Path, workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Histograma de valores de PNG por imagen. Devuelve (nombres, hist) y usa cache."""
    if cache.exists():
        data = np.load(cache)
        return data["names"], data["hist"]
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".png") and "__MACOSX" not in n)
    print(f"Leyendo {len(names)} mascaras reales de {zip_path} ...", flush=True)
    chunks = [names[i : i + 2000] for i in range(0, len(names), 2000)]
    out_names: list[str] = []
    out_hist: list[np.ndarray] = []
    with ProcessPoolExecutor(workers) as pool:
        for i, (chunk_names, chunk_hist) in enumerate(pool.map(_hist_worker, [(str(zip_path), c) for c in chunks])):
            out_names += chunk_names
            out_hist.append(chunk_hist)
            print(f"  {min((i + 1) * 2000, len(names))}/{len(names)}", flush=True)
    hist = np.concatenate(out_hist)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, names=np.array(out_names), hist=hist)
    return np.array(out_names), hist


def class_histograms(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pixeles por clase 0..6, pixeles unlabeled) de cada imagen, por remapeo real."""
    lut = build_coco_lut(load_mapping(), load_categories())
    unknown = np.where((lut == 254) & (hist.sum(axis=0) > 0))[0]
    if unknown.size:
        raise ValueError(f"Valores de PNG sin categoria en COCO-Stuff: {unknown.tolist()}")
    per_class = np.zeros((hist.shape[0], NUM_CLASSES), dtype=np.int64)
    for value in range(HIST_SIZE):
        if lut[value] < NUM_CLASSES:
            per_class[:, lut[value]] += hist[:, value]
    return per_class, hist[:, 255]


def select_images(per_class: np.ndarray, n_target: int, rng: np.random.Generator) -> np.ndarray:
    """Cuotas por clase escasa y resto al azar. Devuelve indices de las imagenes elegidas."""
    frac = per_class / np.maximum(per_class.sum(axis=1, keepdims=True), 1)
    chosen = np.zeros(len(per_class), dtype=bool)
    for name, quota, min_frac in QUOTAS:
        c = CLASSES.index(name)
        eligible = frac[:, c] >= min_frac
        missing = quota - int((chosen & eligible).sum())
        pool = np.where(eligible & ~chosen)[0]
        if missing > 0:
            take = rng.permutation(pool)[: min(missing, n_target - int(chosen.sum()))]
            chosen[take] = True
        print(f"  cuota {name}: {int(eligible.sum())} elegibles, {int((chosen & eligible).sum())} incluidas")
    remaining = n_target - int(chosen.sum())
    if remaining > 0:
        pool = np.where(~chosen & (per_class.sum(axis=1) > 0))[0]
        chosen[rng.permutation(pool)[:remaining]] = True
    return np.where(chosen)[0]


def stratum(row_hist: np.ndarray) -> int:
    """Clase mas escasa presente en la imagen, para repartir las escasas en las 3 particiones."""
    frac = row_hist / max(row_hist.sum(), 1)
    for name, _, min_frac in QUOTAS:
        c = CLASSES.index(name)
        if frac[c] >= min_frac:
            return c
    return 0


def split_groups(groups: list[list[str]], rng: np.random.Generator) -> dict[str, str]:
    """Reparte grupos enteros buscando 70/15/15 de imagenes."""
    total = sum(len(g) for g in groups)
    targets = {k: v * total for k, v in RATIOS.items()}
    counts = {k: 0 for k in RATIOS}
    out: dict[str, str] = {}
    order = rng.permutation(len(groups))
    # Primero los grupos grandes: son los dificiles de encajar.
    for i in sorted(order, key=lambda j: -len(groups[j])):
        group = groups[i]
        split = max(RATIOS, key=lambda k: (targets[k] - counts[k]) / targets[k])
        for item in group:
            out[item] = split
        counts[split] += len(group)
    return out


def split_coco(ids: list[str], strata: list[int], rng: np.random.Generator) -> dict[str, str]:
    out: dict[str, str] = {}
    strata_arr = np.array(strata)
    for value in sorted(set(strata)):
        members = np.where(strata_arr == value)[0]
        members = rng.permutation(members)
        n_train = int(round(len(members) * RATIOS["train"]))
        n_val = int(round(len(members) * RATIOS["val"]))
        for pos, idx in enumerate(members):
            out[ids[idx]] = "train" if pos < n_train else "val" if pos < n_train + n_val else "test"
    return out


def bowfire_fire_ids(zip_path: Path) -> list[str]:
    """Imagenes con fuego. Las `not*` no tienen fuego y sin el quedarian enteras en ignore."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            n
            for n in zf.namelist()
            if "/dataset/img/" in n and "__MACOSX" not in n and Path(n).name.startswith("fire") and n.endswith((".png", ".jpg"))
        ]
    return sorted(Path(n).stem for n in names)


def compute_bowfire_groups(zip_path: Path, max_distance: int = 12) -> list[list[str]]:
    """Agrupa casi-duplicados con dHash de 64 bits.

    Muchas fotos de BoWFire son rafagas de la misma escena. Si caen en train y test, el
    conjunto de prueba mediria memoria y no generalizacion, asi que van juntas.
    """
    ids = bowfire_fire_ids(zip_path)
    hashes = []
    with zipfile.ZipFile(zip_path) as zf:
        by_stem = {
            Path(n).stem: n for n in zf.namelist() if "/dataset/img/" in n and "__MACOSX" not in n and not n.endswith("/")
        }
        for image_id in ids:
            img = cv2.imdecode(np.frombuffer(io.BytesIO(zf.read(by_stem[image_id])).getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
            small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
            hashes.append((small[:, 1:] > small[:, :-1]).ravel())
    parent = list(range(len(ids)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if int((hashes[i] != hashes[j]).sum()) <= max_distance:
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i, image_id in enumerate(ids):
        groups.setdefault(find(i), []).append(image_id)
    return sorted(groups.values(), key=lambda g: g[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stuff-zip", type=Path, required=True)
    parser.add_argument("--bowfire-zip", type=Path, required=True)
    parser.add_argument("--n-coco", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--hist-cache", type=Path, default=Path("data/raw/coco_png_histograms.npz"))
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    names, hist = compute_histograms(args.stuff_zip, args.hist_cache, args.workers)
    per_class, _ = class_histograms(hist)
    ids = [Path(str(n)).stem for n in names]

    rng = np.random.default_rng(args.seed)
    print(f"Imagenes en COCO-Stuff (train2017 + val2017): {len(ids)}")
    selected = select_images(per_class, args.n_coco, rng)
    selected_ids = [ids[i] for i in selected]
    coco_split = split_coco(selected_ids, [stratum(per_class[i]) for i in selected], rng)

    if BOWFIRE_GROUPS.exists():
        groups = json.loads(BOWFIRE_GROUPS.read_text(encoding="utf-8"))
    else:
        groups = compute_bowfire_groups(args.bowfire_zip)
        BOWFIRE_GROUPS.write_text(json.dumps(groups, indent=1) + "\n", encoding="utf-8", newline="\n")
    fire_split = split_groups(groups, rng)

    rows = [{"image_id": k, "source": SOURCE_COCO, "split": v} for k, v in coco_split.items()]
    rows += [{"image_id": k, "source": SOURCE_BOWFIRE, "split": v} for k, v in fire_split.items()]
    write_manifest(rows, args.out)

    for source in (SOURCE_COCO, SOURCE_BOWFIRE):
        sub = [r for r in rows if r["source"] == source]
        counts = {s: sum(r["split"] == s for r in sub) for s in RATIOS}
        print(source, len(sub), {k: f"{v} ({v / len(sub):.1%})" for k, v in counts.items()})
    print(f"Fuego en COCO (clase {FUEGO_LUZ}) solo significa `light`: ver ml/reports/class_pixel_counts.json")
    print(f"Escrito {args.out}")


if __name__ == "__main__":
    main()
