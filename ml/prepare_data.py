"""Descarga y preprocesa las imagenes del manifiesto, y empaqueta/extrae el tar de Drive.

Las imagenes se guardan como PNG de 320x320 (resize directo INTER_LINEAR del contrato) y las
mascaras como PNG de 1 canal con indices 0-6 y 255 (INTER_NEAREST). Se usa PNG porque no
pierde informacion: el pixel en disco es exactamente la entrada que veria `preprocess_seg`.

Salida:
    <out>/coco_stuff/images/<id>.png, <out>/coco_stuff/masks/<id>.png
    <out>/bowfire/images/<id>.png,    <out>/bowfire/masks/<id>.png
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from dataset import (
    DEFAULT_MANIFEST,
    IGNORE_INDEX,
    NUM_CLASSES,
    SOURCE_BOWFIRE,
    SOURCE_COCO,
    atomic_write_json,
    build_coco_lut,
    image_path,
    limit_rows,
    load_categories,
    load_mapping,
    manifest_sha256,
    mask_path,
    read_manifest,
    read_mask,
    remap_bowfire_mask,
    remap_coco_mask,
    resize_image,
    resize_mask,
)

STUFF_ZIP_URL = "https://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip"
STUFF_ZIP_BYTES = 659008564
BOWFIRE_ZIP_URL = "https://bitbucket.org/gbdi/bowfire-dataset/downloads/BoWFireDataset.zip"
BOWFIRE_ZIP_BYTES = 291275697
COCO_IMAGE_URL = "http://images.cocodataset.org/{subset}/{image_id}.jpg"
USER_AGENT = "kinetia-dataset-fetch/1.0"


def download(url: str, dst: Path, expected_bytes: Optional[int] = None, retries: int = 8) -> Path:
    """Descarga reanudable con Range. Si el archivo ya esta completo, no hace nada."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and (expected_bytes is None or dst.stat().st_size == expected_bytes):
        return dst
    part = dst.with_name(dst.name + ".part")
    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as resp:
                mode = "ab" if start and resp.status == 206 else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            if expected_bytes is None or part.stat().st_size == expected_bytes:
                os.replace(part, dst)
                return dst
            print(f"Descarga incompleta de {dst.name}: {part.stat().st_size}/{expected_bytes} bytes; reintentando")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(f"Intento {attempt}/{retries} de {dst.name} fallo: {exc}")
        time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"No se pudo descargar {url}")


def _fetch_bytes(url: str, retries: int = 6) -> bytes:
    last: Exception = RuntimeError("sin intentos")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            time.sleep(min(20, 0.5 * 2**attempt) + random.random())
    raise RuntimeError(f"No se pudo descargar {url}: {last}")


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise IOError(f"No se pudo codificar {path}")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)


def _prepare_one(row: dict, out: Path, stuff: Optional[zipfile.ZipFile], index: dict, bow: Optional[zipfile.ZipFile], lut, mapping) -> None:
    img_out, mask_out = image_path(out, row), mask_path(out, row)
    if img_out.exists() and mask_out.exists():
        return
    image_id = row["image_id"]
    if row["source"] == SOURCE_COCO:
        member = index[image_id]  # p. ej. train2017/000000000139.png
        subset = member.split("/")[0]
        raw_image = _fetch_bytes(COCO_IMAGE_URL.format(subset=subset, image_id=image_id))
        raw_mask = stuff.read(member)
        mask = remap_coco_mask(cv2.imdecode(np.frombuffer(raw_mask, np.uint8), cv2.IMREAD_UNCHANGED), lut)
    elif row["source"] == SOURCE_BOWFIRE:
        raw_image = bow.read(index[f"bowfire/{image_id}"])
        raw_mask = bow.read(f"BoWFireDataset/dataset/gt/{image_id}_gt.png")
        mask = remap_bowfire_mask(cv2.imdecode(np.frombuffer(raw_mask, np.uint8), cv2.IMREAD_GRAYSCALE), mapping)
    else:
        raise ValueError(f"Fuente desconocida: {row['source']}")
    image = cv2.imdecode(np.frombuffer(raw_image, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Imagen ilegible: {image_id}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Imagen y mascara de {image_id} con tamanos distintos: {image.shape[:2]} {mask.shape[:2]}")
    _write_png(img_out, resize_image(image))
    _write_png(mask_out, resize_mask(mask))


def prepare_rows(rows: Sequence[dict], out: Path, raw_dir: Path, threads: int = 16) -> dict:
    """Prepara las filas dadas. Reanuda sola: lo que ya existe no se vuelve a bajar."""
    out, raw_dir = Path(out), Path(raw_dir)
    for source in (SOURCE_COCO, SOURCE_BOWFIRE):
        (out / source / "images").mkdir(parents=True, exist_ok=True)
        (out / source / "masks").mkdir(parents=True, exist_ok=True)
    pending = [r for r in rows if not (image_path(out, r).exists() and mask_path(out, r).exists())]
    print(f"Imagenes: {len(rows)} en total, {len(pending)} por preparar", flush=True)
    if not pending:
        return {"prepared": 0, "failed": 0, "seconds": 0.0}

    mapping = load_mapping()
    lut = build_coco_lut(mapping, load_categories())
    index: dict[str, str] = {}
    stuff = bow = None
    if any(r["source"] == SOURCE_COCO for r in pending):
        stuff = zipfile.ZipFile(download(STUFF_ZIP_URL, raw_dir / "stuffthingmaps_trainval2017.zip", STUFF_ZIP_BYTES))
        for name in stuff.namelist():
            if name.endswith(".png"):
                index[Path(name).stem] = name
    if any(r["source"] == SOURCE_BOWFIRE for r in pending):
        bow = zipfile.ZipFile(download(BOWFIRE_ZIP_URL, raw_dir / "BoWFireDataset.zip", BOWFIRE_ZIP_BYTES))
        for name in bow.namelist():
            if "/dataset/img/" in name and "__MACOSX" not in name and not name.endswith("/"):
                index[f"bowfire/{Path(name).stem}"] = name

    start = time.time()
    failed: list[tuple[str, str]] = []
    done = 0
    with ThreadPoolExecutor(threads) as pool:
        futures = {pool.submit(_prepare_one, r, out, stuff, index, bow, lut, mapping): r for r in pending}
        for future in as_completed(futures):
            row = futures[future]
            try:
                future.result()
            except Exception as exc:  # se reporta al final: una imagen no debe tumbar 15 000
                failed.append((row["image_id"], repr(exc)))
            done += 1
            if done % 500 == 0 or done == len(pending):
                rate = done / max(time.time() - start, 1e-6)
                print(f"  {done}/{len(pending)} ({rate:.0f} img/s medidos; faltan ~{(len(pending) - done) / rate / 60:.1f} min)", flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} imagenes fallaron; vuelve a ejecutar para reintentar. Primeras: {failed[:5]}")
    return {"prepared": len(pending), "failed": 0, "seconds": time.time() - start}


def make_tar(out: Path, rows: Sequence[dict], tar_path: Path) -> dict:
    """Tar sin comprimir (los PNG ya lo estan). Se escribe en temporal y se renombra."""
    out, tar_path = Path(out), Path(tar_path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tar_path.with_name(tar_path.name + ".tmp")
    with tarfile.open(tmp, "w") as tar:
        for row in sorted(rows, key=lambda r: (r["source"], r["image_id"])):
            for path in (image_path(out, row), mask_path(out, row)):
                tar.add(path, arcname=path.relative_to(out).as_posix())
    os.replace(tmp, tar_path)
    return {"bytes": tar_path.stat().st_size, "num_images": len(rows)}


def extract_tar(tar_path: Path, out: Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        for member in members:  # el tar lo escribimos nosotros, pero se valida igual
            target = (out / member.name).resolve()
            if not str(target).startswith(str(out.resolve())) or not member.isfile():
                raise ValueError(f"Entrada sospechosa en el tar: {member.name}")
        tar.extractall(out, members=members)


def compute_class_weights(root: Path, train_rows: Sequence[dict]) -> dict:
    """Peso por clase a partir de los pixeles de TRAIN (raiz de la frecuencia mediana)."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for row in train_rows:
        hist = np.bincount(read_mask(root, row).ravel(), minlength=256)
        counts += hist[:NUM_CLASSES]
    freq = counts / max(counts.sum(), 1)
    present = freq[freq > 0]
    median = np.median(present) if present.size else 1.0
    weights = np.where(freq > 0, np.sqrt(median / np.maximum(freq, 1e-12)), 1.0)
    weights = np.clip(weights, 0.5, 8.0)
    return {"pixels": counts.tolist(), "weights": [float(w) for w in weights]}


def check_prepared(rows: Sequence[dict], root: Path) -> list[str]:
    return [
        r["image_id"]
        for r in rows
        if not (image_path(root, r).exists() and mask_path(root, r).exists())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--tar", type=Path, default=None, help="Ademas, empaqueta en este tar")
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    rows: list[dict] = []
    for split in args.splits.split(","):
        rows += limit_rows([r for r in manifest if r["split"] == split], args.limit_per_split)
    summary = prepare_rows(rows, args.out, args.raw_dir, args.threads)
    missing = check_prepared(rows, args.out)
    if missing:
        raise RuntimeError(f"Faltan {len(missing)} imagenes: {missing[:5]}")
    print(json.dumps(summary))
    total_bytes = sum(image_path(args.out, r).stat().st_size + mask_path(args.out, r).stat().st_size for r in rows)
    print(f"Tamano en disco de {len(rows)} imagenes + mascaras: {total_bytes / 1e9:.3f} GB (medido)")
    if args.tar:
        info = make_tar(args.out, rows, args.tar)
        info["manifest_sha256"] = manifest_sha256(args.manifest)
        atomic_write_json(args.tar.with_suffix(".json"), info)
        print(f"Tar {args.tar}: {info['bytes'] / 1e9:.3f} GB (medido)")
    assert IGNORE_INDEX == 255


if __name__ == "__main__":
    main()
