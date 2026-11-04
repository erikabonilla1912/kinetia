"""Dataset, remapeo de mascaras, preprocesado y utilidades de particion de KINETIA.

Fuente de verdad: docs/CONTRATOS.md, secciones 1, 2 y 3.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import cv2
import numpy as np
import yaml

CLASSES = ["fondo", "cielo", "agua", "vegetacion", "fuego_luz", "persona", "animal"]
NUM_CLASSES = 7
IGNORE_INDEX = 255
FUEGO_LUZ = CLASSES.index("fuego_luz")
INPUT_SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SOURCE_COCO = "coco_stuff"
SOURCE_BOWFIRE = "bowfire"
SOURCES = (SOURCE_COCO, SOURCE_BOWFIRE)

# train.py solo puede usar estas particiones. Ver ForbiddenSplitError.
TRAINING_SPLITS = ("train", "val")
EVALUATION_SPLITS = ("test",)

ML_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ML_DIR / "splits" / "manifest.csv"
DEFAULT_MAPPING = ML_DIR / "class_mapping.yaml"
DEFAULT_CATEGORIES = ML_DIR / "coco_categories.json"


class ForbiddenSplitError(RuntimeError):
    """Se intento usar una particion que este proceso no tiene permitida."""


class ManifestMismatchError(RuntimeError):
    """El manifiesto de datos no coincide con el hash esperado."""


# --------------------------------------------------------------------------
# Preprocesado (contrato, seccion 2.1). Debe coincidir con
# backend/app/services/preprocess.py::preprocess_seg con tolerancia 1e-5.
# --------------------------------------------------------------------------
def resize_image(image: np.ndarray) -> np.ndarray:
    """Resize directo a 320x320 con INTER_LINEAR, sin letterbox."""
    return cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask: np.ndarray) -> np.ndarray:
    """Las mascaras usan INTER_NEAREST: interpolar indices de clase los corromperia."""
    return cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)


def normalize_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> float32 CHW normalizado con la media y std de ImageNet."""
    x = image_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))


def preprocess_seg(image_bgr: np.ndarray) -> np.ndarray:
    """BGR de OpenCV -> tensor float32 de forma [1, 3, 320, 320], listo para el ONNX."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return normalize_rgb(resize_image(rgb))[None, ...]


# --------------------------------------------------------------------------
# Remapeo de mascaras (siempre por nombre de categoria)
# --------------------------------------------------------------------------
def load_mapping(path: Path = DEFAULT_MAPPING) -> dict:
    with open(path, encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    if mapping["classes"] != CLASSES:
        raise ValueError(f"Las clases de {path} no coinciden con el contrato: {mapping['classes']}")
    return mapping


def load_categories(path: Path = DEFAULT_CATEGORIES) -> dict[int, str]:
    """id de categoria -> nombre, extraidos de los JSON oficiales de COCO."""
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


# Valor en la LUT para un valor de PNG que no corresponde a ninguna categoria conocida.
_UNKNOWN = 254


def build_coco_lut(mapping: dict, categories: dict[int, str]) -> np.ndarray:
    """Tabla de 256 entradas: valor del PNG de COCO-Stuff -> indice de clase."""
    cfg = mapping["coco_stuff"]
    by_name = {name: cid for cid, name in categories.items()}
    lut = np.full(256, _UNKNOWN, dtype=np.uint8)
    default = CLASSES.index(cfg["default_class"])
    for cid in categories:
        lut[cid + cfg["png_value_offset"]] = default
    for class_name, names in cfg["mapping"].items():
        for name in names:
            if name not in by_name:
                raise KeyError(f"La categoria '{name}' no existe en COCO-Stuff")
            lut[by_name[name] + cfg["png_value_offset"]] = CLASSES.index(class_name)
    lut[cfg["unlabeled_png_value"]] = IGNORE_INDEX
    return lut


def remap_coco_mask(raw: np.ndarray, lut: np.ndarray) -> np.ndarray:
    out = lut[raw]
    if (out == _UNKNOWN).any():
        bad = sorted(set(raw[out == _UNKNOWN].tolist()))
        raise ValueError(f"Valores de mascara sin categoria conocida: {bad}")
    return out


def remap_bowfire_mask(gt: np.ndarray, mapping: dict) -> np.ndarray:
    """Solo cuenta el fuego; todo lo demas queda en IGNORE_INDEX."""
    cfg = mapping["external_sources"][SOURCE_BOWFIRE]
    if gt.ndim == 3:
        gt = gt[..., 0]
    out = np.full(gt.shape, IGNORE_INDEX, dtype=np.uint8)
    out[gt > cfg["foreground_threshold"]] = CLASSES.index(cfg["foreground_class"])
    return out


# --------------------------------------------------------------------------
# Manifiesto de particion
# --------------------------------------------------------------------------
def manifest_sha256(path: Path = DEFAULT_MANIFEST) -> str:
    """SHA-256 del manifiesto con finales de linea normalizados a LF.

    git puede convertir el CSV a CRLF al clonar en Windows; sin normalizar, el mismo
    manifiesto daria un hash distinto en Colab y en el portatil.
    """
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_manifest(path: Path = DEFAULT_MANIFEST) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and list(rows[0].keys()) != ["image_id", "source", "split"]:
        raise ValueError(f"Columnas inesperadas en {path}: {list(rows[0].keys())}")
    return rows


def write_manifest(rows: Iterable[dict], path: Path = DEFAULT_MANIFEST) -> None:
    rows = sorted(rows, key=lambda r: (r["source"], r["image_id"]))
    text = "image_id,source,split\n" + "".join(
        f"{r['image_id']},{r['source']},{r['split']}\n" for r in rows
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def rows_for_split(rows: Sequence[dict], split: str, allowed: Sequence[str]) -> list[dict]:
    """Filas de una particion; falla si el proceso no tiene permitida esa particion."""
    if split not in allowed:
        raise ForbiddenSplitError(f"La particion '{split}' no esta permitida aqui (permitidas: {tuple(allowed)})")
    return [r for r in rows if r["split"] == split]


def limit_rows(rows: Sequence[dict], limit: Optional[int]) -> list[dict]:
    """Subconjunto estable alternando fuentes, para que un limite incluya a todas."""
    if limit is None or limit >= len(rows):
        return list(rows)
    by_source: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda r: (r["source"], r["image_id"])):
        by_source.setdefault(row["source"], []).append(row)
    out: list[dict] = []
    queues = [list(reversed(by_source[s])) for s in sorted(by_source)]
    while len(out) < limit:
        for queue in queues:
            if queue and len(out) < limit:
                out.append(queue.pop())
    return out


# --------------------------------------------------------------------------
# Lectura de disco
# --------------------------------------------------------------------------
def image_path(root: Path, row: dict) -> Path:
    return Path(root) / row["source"] / "images" / f"{row['image_id']}.png"


def mask_path(root: Path, row: dict) -> Path:
    return Path(root) / row["source"] / "masks" / f"{row['image_id']}.png"


def read_png(path: Path, flags: int) -> np.ndarray:
    # open() de Python y no cv2.imread: soporta rutas no ASCII en Windows y deja rastro
    # en los eventos de auditoria, que es como el test comprueba que train.py no abre test.
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    image = cv2.imdecode(buf, flags)
    if image is None:
        raise IOError(f"No se pudo decodificar {path}")
    return image


def read_image_bgr(root: Path, row: dict) -> np.ndarray:
    return read_png(image_path(root, row), cv2.IMREAD_COLOR)


def read_mask(root: Path, row: dict) -> np.ndarray:
    return read_png(mask_path(root, row), cv2.IMREAD_GRAYSCALE)


class SegDataset:
    """Dataset map-style compatible con torch.utils.data.DataLoader.

    `allowed_splits` es obligatorio: quien construye el dataset declara que particiones
    puede leer, y cualquier otra fila provoca ForbiddenSplitError antes de abrir un archivo.
    """

    def __init__(
        self,
        root: Path,
        rows: Sequence[dict],
        allowed_splits: Sequence[str],
        transform: Optional[Callable] = None,
    ):
        for row in rows:
            if row["split"] not in allowed_splits:
                raise ForbiddenSplitError(
                    f"La imagen {row['image_id']} es de '{row['split']}' y solo se permite {tuple(allowed_splits)}"
                )
        self.root = Path(root)
        self.rows = list(rows)
        self.allowed_splits = tuple(allowed_splits)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.rows[index]
        if row["split"] not in self.allowed_splits:  # las filas pueden mutar tras construir
            raise ForbiddenSplitError(f"Acceso prohibido a la particion '{row['split']}'")
        image_bgr = read_image_bgr(self.root, row)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = resize_image(rgb)
        mask = resize_mask(read_mask(self.root, row))
        if self.transform is not None:
            augmented = self.transform(image=rgb, mask=mask)
            rgb, mask = augmented["image"], augmented["mask"]
        return normalize_rgb(rgb), mask.astype(np.int64)


# --------------------------------------------------------------------------
# Escritura atomica (Colab puede cortar en medio de una escritura)
# --------------------------------------------------------------------------
def atomic_write(path: Path, writer: Callable[[str], None]) -> None:
    """Escribe en un temporal de la misma carpeta y renombra.

    Si el proceso muere en medio de `writer`, el archivo anterior queda intacto.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(tmp)
        with open(tmp, "rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_json(path: Path, data: object) -> None:
    def writer(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    atomic_write(path, writer)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
