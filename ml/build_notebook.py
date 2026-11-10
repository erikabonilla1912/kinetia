"""Genera notebooks/kinetia_training.ipynb incrustando los archivos de ml/ con %%writefile.

El notebook se sube solo a Colab: todo el codigo va dentro y a la vista. Para no duplicar,
el codigo vive en ml/ y aqui se copia. ml/tests/test_notebook_sync.py falla si se desincronizan.

Uso:
    python ml/build_notebook.py
"""
from __future__ import annotations

import base64
import gzip
import json
import textwrap
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
REPO = ML_DIR.parent
NOTEBOOK = REPO / "notebooks" / "kinetia_training.ipynb"

# Orden en que aparecen en el notebook. Todo ml/*.py debe estar aqui, salvo este script.
EMBEDDED = [
    "requirements.txt",
    "class_mapping.yaml",
    "coco_categories.json",
    "splits/bowfire_groups.json",
    "dataset.py",
    "seg_metrics.py",
    "model.py",
    "prepare_data.py",
    "make_splits.py",
    "count_pixels.py",
    "train.py",
    "export.py",
    "evaluate.py",
    "compute_norm_stats.py",
    "fetch_test_set.py",
    "validate_artifacts.py",
]


def _read(rel: str) -> str:
    return (ML_DIR / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(textwrap.dedent(text).strip("\n"))}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(textwrap.dedent(text).strip("\n")),
    }


def writefile_cell(rel: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(f"%%writefile ml/{rel}\n" + _read(rel)),
    }


def manifest_cell() -> dict:
    data = (ML_DIR / "splits" / "manifest.csv").read_bytes().replace(b"\r\n", b"\n")
    compressed = gzip.compress(data, compresslevel=9, mtime=0)
    blob = base64.b64encode(compressed).decode("ascii")
    wrapped = "\n".join(textwrap.wrap(blob, 100))
    import hashlib

    digest = hashlib.sha256(data).hexdigest()
    source = (
        "# Particion fija (ml/splits/manifest.csv), comprimida para no ocupar 15 000 lineas.\n"
        "# Es la misma del repositorio: https://github.com/erikabonilla1912/kinetia\n"
        "import base64, gzip, hashlib\n"
        f'MANIFEST_SHA256 = "{digest}"\n'
        f'_MANIFEST_B64 = """\n{wrapped}\n"""\n'
        "_data = gzip.decompress(base64.b64decode(_MANIFEST_B64))\n"
        'assert hashlib.sha256(_data).hexdigest() == MANIFEST_SHA256, "Manifiesto incrustado danado"\n'
        'Path("ml/splits").mkdir(parents=True, exist_ok=True)\n'
        'Path("ml/splits/manifest.csv").write_bytes(_data)\n'
        'print("Manifiesto escrito; SHA-256:", MANIFEST_SHA256, "| filas:", _data.count(b"\\n") - 1)\n'
    )
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(source)}


INTRO = """
# KINETIA: entrenamiento del modelo de segmentación

U-Net con encoder MobileNetV3 Large **entrenada desde cero**: todos los pesos arrancan aleatorios, sin *transfer learning* ni nada de ImageNet. 7 clases, entrada 320×320.

**Cómo usarlo:** activa la GPU T4 (*Entorno de ejecución → Cambiar tipo de entorno*) y pulsa *Entorno de ejecución → Ejecutar todo*. Si la sesión se corta, vuelve a pulsar *Ejecutar todo*: continúa sola desde la última época guardada en Drive. La guía completa está en `docs/COLAB_PASO_A_PASO.md`.

**Índice:** 1) Configuración · 2) Código del proyecto · 3) Dependencias y GPU · 4) Google Drive · 5) Datos · 6) Análisis exploratorio · 7) Preprocesamiento · 8) Entrenamiento y validación · 9) Curvas y sobreajuste/subajuste · 10) Exportación · 11) Evaluación · 12) Predicción sobre una imagen nueva · 13) Conclusiones · 14) Descarga.

**Ninguna cifra de este notebook está escrita a mano:** todas salen de ejecutar las celdas. Las estimaciones de tiempo se marcan como estimaciones.
"""

CONFIG = """
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0").strip().lower() in ("1", "true", "yes")
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if not SMOKE_TEST and not IN_COLAB:
    raise RuntimeError("Este notebook se entrena en Google Colab. Para una prueba local usa la variable de entorno SMOKE_TEST=1.")

# --- Parámetros del entrenamiento. Ajustados para la T4 gratis: AMP, 2 trabajadores.
EPOCHS = 2 if SMOKE_TEST else 100          # máximo; desde cero hacen falta más épocas
BATCH_SIZE = 4 if SMOKE_TEST else 32
NUM_WORKERS = 0 if SMOKE_TEST else 2       # Colab gratis tiene 2 vCPU
PATIENCE = 15                              # épocas sin mejorar el mIoU de validación
SMOKE_LIMITS = {"train": 16, "val": 6, "test": 6}  # solo se usan en SMOKE_TEST

if SMOKE_TEST:
    # Sin KINETIA_WORKDIR va a una carpeta temporal nueva, para no ensuciar el repositorio.
    WORK = Path(os.environ.get("KINETIA_WORKDIR") or tempfile.mkdtemp(prefix="kinetia_smoke_")).resolve()
else:
    WORK = Path("/content/kinetia")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
Path("ml/splits").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WORK / "ml"))

RAW_DIR = Path(os.environ.get("KINETIA_RAW_DIR", WORK / "raw")).resolve()
DATA_DIR = WORK / "data" / "processed"
OUT_DIR = WORK / "output"                 # los 5 archivos finales
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_VERSION = "smoke-test-no-oficial" if SMOKE_TEST else "seg-2.0.0"

print(f"Modo: {'SMOKE_TEST (prueba del flujo; sus cifras no son resultados)' if SMOKE_TEST else 'REAL'}")
print(f"Carpeta de trabajo: {WORK}")


def run(*cmd):
    \"\"\"Ejecuta un script mostrando su salida en vivo; un error detiene el notebook.\"\"\"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "NO_ALBUMENTATIONS_UPDATE": "1"}
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env, cwd=WORK)
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"El comando falló con código {code}: {' '.join(cmd)}")
"""

INSTALL = """
# PyTorch ya viene en Colab y NO se reinstala. Solo se instala lo que falta o difiere.
# Colab trae varias variantes de OpenCV que comparten la carpeta cv2, y por eso se quitan antes;
# si no, se mezclarían versiones y el resize dejaría de ser el del contrato.
from importlib import metadata

pins = []
for line in Path("ml/requirements.txt").read_text(encoding="utf-8").splitlines():
    spec = line.split("#")[0].strip()
    if not spec:
        continue
    name = spec.split("==")[0]
    if name in ("torch", "torchvision", "numpy", "pytest", "nbconvert", "nbformat", "ipykernel", "jupyter-client"):
        continue
    if IN_COLAB and "# colab" not in line:
        continue
    pins.append(spec)


def installed(spec):
    name, version = spec.split("==")
    try:
        return metadata.version(name) == version
    except metadata.PackageNotFoundError:
        return False


missing = [p for p in pins if not installed(p)]
if missing:
    print("Instalando:", missing)
    if IN_COLAB and any(p.startswith("opencv") for p in missing):
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "opencv-python", "opencv-contrib-python", "opencv-python-headless"], check=False)
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("pip no pudo instalar las dependencias. Envía este mensaje al orquestador:\\n" + (result.stdout + result.stderr)[-3000:])
else:
    print("Dependencias ya instaladas.")

# Se comprueba en un proceso nuevo: es lo que verán los scripts.
probe = subprocess.run(
    [sys.executable, "-c", "import numpy,cv2,onnx,onnxruntime,albumentations,segmentation_models_pytorch as s,torch;print(numpy.__version__,cv2.__version__,onnx.__version__,onnxruntime.__version__,albumentations.__version__,s.__version__,torch.__version__)"],
    capture_output=True, text=True, env={**os.environ, "NO_ALBUMENTATIONS_UPDATE": "1"},
)
if probe.returncode != 0:
    raise RuntimeError("Las dependencias no se importan:\\n" + probe.stderr[-2000:])
versions = probe.stdout.split()
print("numpy, cv2, onnx, onnxruntime, albumentations, smp, torch =", versions)
if versions[1] != "4.10.0":
    raise RuntimeError(f"OpenCV debe ser 4.10.0 y es {versions[1]}. Ve a Entorno de ejecución > Reiniciar sesión y pulsa Ejecutar todo otra vez.")
# Si pip cambió numpy con este kernel ya usándolo, hay que reiniciar; al repetir ya no se instala nada.
if "numpy" in sys.modules and sys.modules["numpy"].__version__ != versions[0]:
    raise RuntimeError("La instalación cambió numpy. Ve a Entorno de ejecución > Reiniciar sesión y pulsa Ejecutar todo otra vez: no se pierde nada.")
"""

GPU = """
import torch

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
elif SMOKE_TEST:
    print("SMOKE_TEST: se usa CPU.")
else:
    raise RuntimeError(
        "NO HAY GPU, y entrenar sin ella tardaría días. Para activarla: menú Entorno de ejecución > "
        "Cambiar tipo de entorno de ejecución > Acelerador de hardware: GPU T4 > Guardar. "
        "Después pulsa Entorno de ejecución > Ejecutar todo. Si Colab dice que no hay GPU disponible, "
        "se agotó la cuota gratuita: inténtalo más tarde (ver docs/COLAB_PASO_A_PASO.md)."
    )
"""

DRIVE = """
# Drive guarda lo que debe sobrevivir a un corte: el dataset en un tar y los checkpoints.
if SMOKE_TEST:
    DRIVE_DIR = WORK / "drive_simulado"
    print("SMOKE_TEST: sin Drive; se usa una carpeta local que hace sus veces.")
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/kinetia")
# Carpeta nueva: la antigua "run" tiene el modelo con transfer learning y no se puede reanudar.
RUN_DIR = DRIVE_DIR / "run_desde_cero"
TAR_PATH = DRIVE_DIR / "dataset" / "kinetia_dataset_320.tar"
TAR_META = TAR_PATH.with_suffix(".json")
RUN_DIR.mkdir(parents=True, exist_ok=True)
TAR_PATH.parent.mkdir(parents=True, exist_ok=True)
usage = shutil.disk_usage(DRIVE_DIR)
print(f"Carpeta persistente: {DRIVE_DIR} | libre: {usage.free / 1e9:.1f} GB de {usage.total / 1e9:.1f} GB")
print(f"Checkpoints de este entrenamiento: {RUN_DIR}")
"""

DATA_MD = """
## 5. Datos

- **COCO-Stuff 164k:** unas 15 000 imágenes elegidas por script para reforzar las clases escasas. Las particiones oficiales de COCO se mezclan y se reparten de nuevo con semilla fija (`seed=42`), porque `test2017` no tiene máscaras públicas y porque así BoWFire se reparte en las tres.
- **BoWFire** (Chino et al., SIBGRAPI 2015): 119 imágenes de fuego con máscara manual. En ellas **solo cuenta el fuego**; el resto es `IGNORE_INDEX=255`. Las fotos casi idénticas se mantienen en la misma partición para que no haya fuga entre entrenamiento y prueba.
- **Licencias:** COCO usa imágenes de Flickr con licencias variadas; BoWFire declara imágenes de Flickr bajo Creative Commons y pide citar su artículo.

La primera sesión descarga y preprocesa, y guarda un tar en Drive. Las siguientes solo lo extraen a `/content`, porque leer de Drive directamente es lento.
"""

DATA = """
from dataset import limit_rows, manifest_sha256, read_manifest
from prepare_data import check_prepared, extract_tar, make_tar, prepare_rows

manifest = read_manifest("ml/splits/manifest.csv")
if SMOKE_TEST:
    rows = [r for split, n in SMOKE_LIMITS.items() for r in limit_rows([m for m in manifest if m["split"] == split], n)]
else:
    rows = manifest
print(f"Imágenes a usar en esta ejecución: {len(rows)}")
t0 = time.time()


def tar_is_valid():
    if not (TAR_PATH.exists() and TAR_META.exists()):
        return False
    meta = json.loads(TAR_META.read_text(encoding="utf-8"))
    return meta["manifest_sha256"] == MANIFEST_SHA256 and meta["bytes"] == TAR_PATH.stat().st_size and meta["num_images"] == len(rows)


if not check_prepared(rows, DATA_DIR):
    print("Datos ya presentes en esta sesión.")
elif tar_is_valid():
    print(f"Extrayendo {TAR_PATH} ...", flush=True)
    extract_tar(TAR_PATH, DATA_DIR)
    print(f"Extraído en {time.time() - t0:.0f} s (medido).")
else:
    print("Primera vez: descargo y preproceso. El tiempo depende de la red; se imprime el medido.", flush=True)
    prepare_rows(rows, DATA_DIR, RAW_DIR)
    local_tar = WORK / TAR_PATH.name
    info = make_tar(DATA_DIR, rows, local_tar)
    info["manifest_sha256"] = MANIFEST_SHA256
    free = shutil.disk_usage(TAR_PATH.parent).free
    print(f"Tar: {info['bytes'] / 1e9:.2f} GB (medido); libre en la carpeta persistente: {free / 1e9:.1f} GB")
    if free < info["bytes"] * 1.1:
        raise RuntimeError(f"No cabe el tar en Drive: necesita {info['bytes'] / 1e9:.2f} GB. Libera espacio y vacía la papelera.")
    tmp = TAR_PATH.with_name(TAR_PATH.name + ".tmp")
    shutil.copyfile(local_tar, tmp)
    os.replace(tmp, TAR_PATH)
    TAR_META.write_text(json.dumps(info, indent=2), encoding="utf-8")
    local_tar.unlink()
    print(f"Tar guardado en {TAR_PATH}. Total: {(time.time() - t0) / 60:.1f} min (medido).")

missing = check_prepared(rows, DATA_DIR)
if missing:
    raise RuntimeError(f"Faltan {len(missing)} imágenes tras preparar: {missing[:5]}")
print("Datos listos.")
"""

EDA = """
# Análisis exploratorio sobre las máscaras reales. Solo se abre train: la prueba no se mira.
import collections
import numpy as np
import matplotlib.pyplot as plt
from dataset import CLASSES, IGNORE_INDEX, NUM_CLASSES, SOURCE_BOWFIRE, image_path, read_image_bgr, read_mask

COLORS = ["#3A3F4B", "#4FC3F7", "#2979FF", "#66BB6A", "#FF7043", "#EC407A", "#FFCA28"]  # contrato, sección 1
table = collections.Counter((r["split"], r["source"]) for r in manifest)
print("Imágenes por partición y fuente (manifiesto completo):")
for key in sorted(table):
    print(f"  {key[0]:<5} {key[1]:<11} {table[key]}")

train_rows = [r for r in rows if r["split"] == "train"]
pixels = np.zeros(256, dtype=np.int64)
presence = np.zeros(NUM_CLASSES, dtype=np.int64)
for r in train_rows:
    h = np.bincount(read_mask(DATA_DIR, r).ravel(), minlength=256)
    pixels += h
    presence += (h[:NUM_CLASSES] > 0).astype(np.int64)
labeled = pixels[:NUM_CLASSES].sum()
print(f"\\nIgnorados (255): {pixels[IGNORE_INDEX] / pixels.sum():.2%} de los píxeles de train")
for i, name in enumerate(CLASSES):
    print(f"  {name:<11} {pixels[i] / labeled:7.2%} de los píxeles etiquetados | presente en {presence[i]} imágenes")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
axes[0].barh(CLASSES[::-1], (pixels[:NUM_CLASSES] / labeled * 100)[::-1], color=COLORS[::-1])
axes[0].set_xscale("log"); axes[0].set_xlabel("% de píxeles etiquetados (escala log)"); axes[0].set_title("Píxeles por clase en train")
axes[1].barh(CLASSES[::-1], presence[::-1], color=COLORS[::-1])
axes[1].set_xlabel("imágenes"); axes[1].set_title("Imágenes de train con la clase")
plt.tight_layout(); plt.show()

palette = np.array([tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in COLORS], dtype=np.uint8)
samples = [r for r in train_rows if r["source"] == SOURCE_BOWFIRE][:2] + [r for r in train_rows if r["source"] != SOURCE_BOWFIRE][:4]
fig, axes = plt.subplots(2, len(samples), figsize=(2.6 * len(samples), 5.2))
for col, r in enumerate(samples):
    mask = read_mask(DATA_DIR, r)
    shown = palette[np.where(mask == IGNORE_INDEX, 0, mask)].copy()
    shown[mask == IGNORE_INDEX] = 255  # blanco = ignorado
    axes[0, col].imshow(read_image_bgr(DATA_DIR, r)[..., ::-1]); axes[0, col].set_title(r["source"], fontsize=8)
    axes[1, col].imshow(shown)
    axes[0, col].axis("off"); axes[1, col].axis("off")
plt.suptitle("Muestras de train: en blanco lo ignorado (255)"); plt.tight_layout(); plt.show()

# La normalización sale de NUESTRAS imágenes de train, no de ImageNet. Se recalcula aquí y
# se comprueba contra la que usan el modelo y la API (MEAN y STD de ml/dataset.py).
from compute_norm_stats import channel_stats
from dataset import MEAN, STD
stats = channel_stats(DATA_DIR, train_rows)
print(f"\\nNormalización medida en {stats['num_images']} imágenes de train: media {stats['mean']} | desviación {stats['std']}")
print(f"Normalización del modelo (ml/dataset.py):          media {[round(float(v), 4) for v in MEAN]} | desviación {[round(float(v), 4) for v in STD]}")
if SMOKE_TEST:
    print("SMOKE_TEST: con pocas imágenes no se compara; en la corrida real deben coincidir.")
elif not (np.allclose(stats["mean"], MEAN, atol=1e-3) and np.allclose(stats["std"], STD, atol=1e-3)):
    raise RuntimeError("La normalización de ml/dataset.py no coincide con la de estas imágenes. Envía este mensaje al orquestador.")
else:
    print("Coinciden.")
"""

PREPROCESS = """
# La entrada es exactamente la del contrato: RGB, resize directo INTER_LINEAR a 320×320,
# /255 y normalización con la media y desviación de nuestro train. Los aumentos solo se aplican en entrenamiento.
from dataset import MEAN, STD, SegDataset, preprocess_seg
from train import build_train_transform

r = train_rows[0]
x = preprocess_seg(read_image_bgr(DATA_DIR, r))
print("Forma:", x.shape, "| tipo:", x.dtype, f"| mín {x.min():.3f} | máx {x.max():.3f} | media {x.mean():.3f}")

augmented = SegDataset(DATA_DIR, [r], ("train", "val"), build_train_transform())
fig, axes = plt.subplots(1, 5, figsize=(13, 2.8))
for ax in axes:
    img, m = augmented[0]
    rgb = np.transpose(img, (1, 2, 0)) * STD + MEAN
    ax.imshow(np.clip(rgb, 0, 1)); ax.axis("off")
plt.suptitle("Aumentos de una misma imagen"); plt.tight_layout(); plt.show()
print("Clases en la máscara tras aumentar:", sorted(set(np.unique(m).tolist())))
"""

TRAIN_MD = """
## 8. Entrenamiento y validación

**Desde cero, sin *transfer learning*:** todos los pesos arrancan aleatorios y todo el modelo se entrena desde la primera época, con la misma tasa de aprendizaje. Pérdida: entropía cruzada ponderada + Dice, ambas ignorando el 255. Se guarda el mejor modelo según el mIoU de validación y el entrenamiento para con *early stopping* (15 épocas sin mejora).

**Esta celda reanuda sola.** Si existe un checkpoint en Drive, continúa desde la época siguiente, sin editar nada. Con 100 épocas es probable que necesites varias sesiones de Colab: cuando se corte, vuelve a pulsar *Ejecutar todo*. Cada línea muestra el tiempo medido de la época y una estimación del tiempo restante.
"""

TRAIN = """
args = ["--data-root", DATA_DIR, "--out-dir", RUN_DIR, "--epochs", EPOCHS, "--batch-size", BATCH_SIZE,
        "--num-workers", NUM_WORKERS, "--patience", PATIENCE, "--manifest", "ml/splits/manifest.csv"]
if SMOKE_TEST:
    args += ["--limit-train", SMOKE_LIMITS["train"], "--limit-val", SMOKE_LIMITS["val"], "--device", "cpu"]
run(sys.executable, "ml/train.py", *args)
"""

CURVES_MD = """
## 9. Curvas y análisis de sobreajuste/subajuste

El diagnóstico se calcula con el historial medido de esta corrida.
"""

CURVES = """
history = json.loads((RUN_DIR / "history.json").read_text(encoding="utf-8"))
epochs = [h["epoch"] for h in history]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="train"); axes[0].plot(epochs, [h["val_loss"] for h in history], "o-", label="val")
axes[0].set_title("Pérdida"); axes[0].set_xlabel("época"); axes[0].legend()
axes[1].plot(epochs, [h["train_miou"] for h in history], "o-", label="train"); axes[1].plot(epochs, [h["val_miou"] for h in history], "o-", label="val")
axes[1].set_title("mIoU"); axes[1].set_xlabel("época"); axes[1].legend()
for i, name in enumerate(CLASSES):
    axes[2].plot(epochs, [h["val_iou_per_class"][name] for h in history], color=COLORS[i], label=name)
axes[2].set_title("IoU por clase en val"); axes[2].set_xlabel("época"); axes[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT_DIR / "curvas.png", dpi=120); plt.show()

best = max(history, key=lambda h: h["val_miou"])
last = history[-1]
gap_miou = last["train_miou"] - last["val_miou"]
gap_loss = last["val_loss"] - last["train_loss"]
best_val_loss = min(h["val_loss"] for h in history)
print(f"Épocas completadas: {len(history)} | tiempo total medido: {sum(h['seconds'] for h in history) / 60:.1f} min")
print(f"Mejor mIoU de val: {best['val_miou']:.4f} en la época {best['epoch']}")
print(f"Última época: mIoU train {last['train_miou']:.4f}, val {last['val_miou']:.4f} (brecha {gap_miou:+.4f}); pérdida train {last['train_loss']:.4f}, val {last['val_loss']:.4f} (brecha {gap_loss:+.4f})")
if SMOKE_TEST:
    diagnosis = "SMOKE_TEST: con tan pocas imágenes y épocas no se puede diagnosticar nada. Esto solo prueba el flujo."
elif gap_miou > 0.15 and last["val_loss"] > best_val_loss * 1.1:
    diagnosis = "Señal de SOBREAJUSTE: train sigue mejor que val y la pérdida de val se aleja de su mínimo."
elif last["train_miou"] < 0.5:
    diagnosis = "Señal de SUBAJUSTE: ni en train el mIoU es alto; el modelo o las épocas se quedan cortos."
else:
    diagnosis = "Sin señal clara de sobreajuste ni de subajuste con los umbrales usados (brecha > 0.15; mIoU de train < 0.5)."
print(diagnosis)
"""

EXPORT_MD = """
## 10. Exportación

`export.py` genera `kinetia_seg.pt`, `kinetia_seg.onnx` (opset 17, N dinámico, salida `logits` sin softmax), `manifest.json` y `hashes.sha256`, y verifica que `.pt` y `.onnx` coincidan con tolerancia `1e-4`. Solo exporta un modelo entrenado desde cero.
"""

EXPORT = """
run(sys.executable, "ml/export.py", "--checkpoint", RUN_DIR / "kinetia_seg.pt",
    "--out-dir", OUT_DIR, "--version", MODEL_VERSION, "--manifest", "ml/splits/manifest.csv")
"""

EVAL_MD = """
## 11. Evaluación sobre el conjunto de prueba

Se evalúa el **`.onnx` exportado** con ONNX Runtime, no el `.pt`, para medir lo mismo que corre en producción. La partición de prueba solo se abre aquí. `metrics.json` es la única fuente de métricas para la API, la UI y el documento técnico.
"""

EVAL = """
cmd = ["--models-dir", OUT_DIR, "--data-root", DATA_DIR, "--manifest", "ml/splits/manifest.csv"]
if SMOKE_TEST:
    cmd += ["--limit", SMOKE_LIMITS["test"], "--out", OUT_DIR / "metrics.json"]
run(sys.executable, "ml/evaluate.py", *cmd)

metrics = json.loads((OUT_DIR / "metrics.json").read_text(encoding="utf-8"))
matrix = np.array(metrics["confusion_matrix"], dtype=np.float64)
normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
im = axes[0].imshow(normalized, cmap="Blues", vmin=0, vmax=1)
axes[0].set_xticks(range(7), CLASSES, rotation=45, ha="right"); axes[0].set_yticks(range(7), CLASSES)
axes[0].set_xlabel("predicción"); axes[0].set_ylabel("verdad"); axes[0].set_title("Matriz de confusión (por fila)")
for i in range(7):
    for j in range(7):
        axes[0].text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if normalized[i, j] > 0.5 else "black")
fig.colorbar(im, ax=axes[0], fraction=0.046)
axes[1].barh(CLASSES[::-1], [metrics["per_class"][c]["iou"] for c in CLASSES[::-1]], color=COLORS[::-1])
axes[1].axvline(metrics["miou"], color="black", linestyle="--", label=f"mIoU = {metrics['miou']:.3f}")
axes[1].set_xlim(0, 1); axes[1].set_title("IoU por clase en prueba"); axes[1].legend()
plt.tight_layout(); plt.savefig(OUT_DIR / "evaluacion.png", dpi=120); plt.show()
"""

PREDICT_MD = """
## 12. Predicción sobre una imagen nueva, sin reentrenar

Carga el modelo exportado y predice una imagen que no está en el manifiesto. Para probar una foto propia, súbela al panel de archivos de Colab y pon su ruta en `NEW_IMAGE_PATH`.
"""

PREDICT = """
import urllib.request
import cv2
import onnxruntime as ort
from dataset import preprocess_seg

NEW_IMAGE_PATH = ""  # p. ej. "/content/mi_foto.jpg"; vacío usa una imagen de COCO test2017, que no tiene máscaras públicas
NEW_IMAGE_URL = "http://images.cocodataset.org/test2017/000000000016.jpg"
CONF_THRESHOLD = 0.5  # contrato, sección 6

if NEW_IMAGE_PATH:
    image_bgr = cv2.imread(NEW_IMAGE_PATH)
else:
    with urllib.request.urlopen(NEW_IMAGE_URL, timeout=60) as resp:
        image_bgr = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_COLOR)
if image_bgr is None:
    raise RuntimeError("No se pudo leer la imagen.")

session = ort.InferenceSession(str(OUT_DIR / "kinetia_seg.onnx"), providers=["CPUExecutionProvider"])
t0 = time.time()
logits = session.run(["logits"], {"input": preprocess_seg(image_bgr)})[0][0]
inference_ms = (time.time() - t0) * 1000
exp = np.exp(logits - logits.max(axis=0, keepdims=True))
probs = exp / exp.sum(axis=0, keepdims=True)
pred = probs.argmax(axis=0)
pred[probs.max(axis=0) < CONF_THRESHOLD] = 0  # posproceso del contrato: confianza baja pasa a fondo
print(f"Inferencia medida en este equipo: {inference_ms:.0f} ms")
for i, name in enumerate(CLASSES):
    share = (pred == i).mean()
    if share > 0:
        print(f"  {name:<11} {share:7.2%} de los píxeles | confianza media {probs.max(axis=0)[pred == i].mean():.3f}")
rgb = cv2.cvtColor(cv2.resize(image_bgr, (320, 320), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(rgb); axes[0].set_title("Imagen nueva")
axes[1].imshow(palette[pred]); axes[1].set_title("Máscara predicha")
axes[2].imshow((0.55 * rgb + 0.45 * palette[pred]).astype(np.uint8)); axes[2].set_title("Superpuesta")
for ax in axes:
    ax.axis("off")
plt.tight_layout(); plt.show()
"""

CONCLUSIONS_MD = """
## 13. Conclusiones

Esta celda redacta las conclusiones a partir de `metrics.json` y del historial. Ninguna cifra está escrita a mano.
"""

CONCLUSIONS = """
ranking = sorted(CLASSES, key=lambda c: metrics["per_class"][c]["iou"], reverse=True)
print(f"Sobre {metrics['num_images']} imágenes de prueba: mIoU {metrics['miou']:.4f}, Dice macro {metrics['dice_macro']:.4f}, exactitud de píxel {metrics['pixel_accuracy']:.4f}.")
print(f"Mejor clase: {ranking[0]} (IoU {metrics['per_class'][ranking[0]]['iou']:.4f}). Peor clase: {ranking[-1]} (IoU {metrics['per_class'][ranking[-1]]['iou']:.4f}).")
print("Objetivo del proyecto: mIoU ≥ 0.55 →", "se cumple." if metrics["miou"] >= 0.55 else "NO se cumple; se reporta la cifra real.")
print("Diagnóstico de la corrida:", diagnosis)
print("Limitaciones: fuego_luz mezcla la categoría light de COCO-Stuff con BoWFire, que tiene pocas imágenes y de incidentes; "
      "en esas imágenes el resto se ignora, así que allí no se miden falsos positivos de fuego.")
print("Noche y bosque no son clases: los deduce el motor con reglas (contrato, sección 1).")
if SMOKE_TEST:
    print("\\nSMOKE_TEST: estas cifras NO son resultados del modelo; solo prueban que el flujo funciona.")
"""

FINAL_MD = """
## 14. Descarga de los archivos

Crea `kinetia_artifacts.zip` con los 5 archivos, guarda una copia en Drive e imprime los hashes. Coloca el contenido en la carpeta `models/` del repositorio.
"""

FINAL = """
import zipfile
from dataset import file_sha256
from validate_artifacts import ARTIFACTS, validate_dir

problems = validate_dir(OUT_DIR)
if problems:
    raise RuntimeError("Los archivos no cumplen el contrato: " + "; ".join(problems))
print("Los 5 archivos cumplen los esquemas del contrato.")

zip_path = OUT_DIR / "kinetia_artifacts.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in ARTIFACTS:
        zf.write(OUT_DIR / name, arcname=name)

final_dir = DRIVE_DIR / "artifacts_desde_cero"
final_dir.mkdir(parents=True, exist_ok=True)
for name in (*ARTIFACTS, "kinetia_artifacts.zip"):
    tmp = final_dir / (name + ".tmp")
    shutil.copyfile(OUT_DIR / name, tmp)
    os.replace(tmp, final_dir / name)

print(f"Copia en: {final_dir}")
print(f"SHA-256 kinetia_seg.pt:   {file_sha256(OUT_DIR / 'kinetia_seg.pt')}")
print(f"SHA-256 kinetia_seg.onnx: {file_sha256(OUT_DIR / 'kinetia_seg.onnx')}")
print(f"Tamaño del zip: {zip_path.stat().st_size / 1e6:.1f} MB")
if IN_COLAB and not SMOKE_TEST:
    from google.colab import files
    files.download(str(zip_path))
"""


def build() -> dict:
    cells = [md(INTRO), md("## 1. Configuración"), code(CONFIG)]
    cells.append(md("## 2. Código del proyecto\n\nCada celda escribe un archivo de `ml/`. El notebook no necesita ningún otro archivo."))
    cells += [writefile_cell(rel) for rel in EMBEDDED]
    cells.append(manifest_cell())
    cells += [md("## 3. Dependencias y GPU"), code(INSTALL), code(GPU)]
    cells += [md("## 4. Google Drive"), code(DRIVE)]
    cells += [md(DATA_MD), code(DATA)]
    cells += [md("## 6. Análisis exploratorio"), code(EDA)]
    cells += [md("## 7. Preprocesamiento y aumentos"), code(PREPROCESS)]
    cells += [md(TRAIN_MD), code(TRAIN)]
    cells += [md(CURVES_MD), code(CURVES)]
    cells += [md(EXPORT_MD), code(EXPORT)]
    cells += [md(EVAL_MD), code(EVAL)]
    cells += [md(PREDICT_MD), code(PREDICT)]
    cells += [md(CONCLUSIONS_MD), code(CONCLUSIONS)]
    cells += [md(FINAL_MD), code(FINAL)]
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:03d}"
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render() -> str:
    return json.dumps(build(), ensure_ascii=False, indent=1) + "\n"


def main() -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK, "w", encoding="utf-8", newline="\n") as f:
        f.write(render())
    print(f"Escrito {NOTEBOOK} ({NOTEBOOK.stat().st_size / 1e6:.2f} MB, {len(build()['cells'])} celdas)")


if __name__ == "__main__":
    main()
