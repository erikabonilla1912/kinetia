# KINETIA — Contrato de interfaces

**Versión:** 1.7.0 · **Fecha:** 2026-11-02 · **Dueño:** agente orquestador

Este archivo es la fuente de verdad entre las áreas. Todos los subagentes lo respetan.

**Cómo se cambia:** el subagente propone el cambio al orquestador con su motivo. El orquestador decide, edita este archivo, sube la versión y lo registra en el historial del final. Ningún subagente lo edita por su cuenta.

---

## 1. Clases del modelo

El orden es fijo: estos índices aparecen en las máscaras, en el ONNX, en la API y en el motor.

| Índice | Nombre (`name`) | Etiqueta en UI | Color UI (Rayos X) | Efecto en el motor |
|---|---|---|---|---|
| 0 | `fondo` | Fondo | `#3A3F4B` | Ninguno: queda quieto |
| 1 | `cielo` | Cielo | `#4FC3F7` | Nubes por desplazamiento de textura; estrellas si es de noche |
| 2 | `agua` | Agua | `#2979FF` | Ondas y reflejos |
| 3 | `vegetacion` | Vegetación | `#66BB6A` | Vaivén leve y hojas que caen |
| 4 | `fuego_luz` | Fuego / luz | `#FF7043` | Parpadeo y brillo |
| 5 | `persona` | Persona | `#EC407A` | Queda quieta, por encima de las partículas traseras |
| 6 | `animal` | Animal | `#FFCA28` | Queda quieto, por encima de las partículas traseras |

- `NUM_CLASSES = 7`.
- `IGNORE_INDEX = 255`: píxel sin etiqueta. Solo existe en entrenamiento y evaluación; no entra en la pérdida ni en las métricas. **Nunca** aparece en una máscara de inferencia.
- **"Noche" no es una clase.** El motor la deduce con una regla: la luminancia media (canal Y) de los píxeles `cielo` es menor que `NIGHT_LUMA_THRESHOLD`. Esto debe decirse así en la sustentación.
- **"Bosque" tampoco es una clase.** Se activa si `vegetacion` cubre al menos `FOREST_MIN_PCT` de la imagen.

### Remapeo del dataset (propuesta sobre COCO-Stuff 164k)

Se implementa en `ml/class_mapping.yaml`, por **nombre** de categoría y nunca por id numérico.

| Clase | Categorías de origen |
|---|---|
| `cielo` | `sky-other`, `clouds` |
| `agua` | `sea`, `river`, `water-other` |
| `vegetacion` | `tree`, `grass`, `bush`, `plant-other`, `flower`, `leaves`, `moss` |
| `fuego_luz` | `light` + máscaras de fuego de BoWFire (Chino et al., SIBGRAPI 2015; hay que citarlo) |
| `persona` | `person` |
| `animal` | `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe` |
| `fondo` | Cualquier otra categoría etiquetada |
| `255` | `unlabeled` de COCO-Stuff. En imágenes de BoWFire: todo píxel que no sea fuego |

**Limitación de `fuego_luz` que se declara en la demo y en el documento:** BoWFire tiene 119 imágenes de incendios a nivel de suelo, casi sin fogatas y sin velas ni chimeneas vistas en la revisión. Una vela o una chimenea pueden no detectarse. Y como en esas imágenes el resto queda en 255, allí no se miden falsos positivos de fuego: el modelo podría confundir objetos naranjas con fuego.

---

## 2. Modelos: entrada y salida

### 2.1 Segmentación (`kinetia_seg.onnx`) — entrenado por nosotros

| Propiedad | Valor |
|---|---|
| Arquitectura | U-Net con encoder MobileNetV3 Large preentrenado en ImageNet (`segmentation_models_pytorch`) |
| Tensor de entrada | nombre `input`, `float32`, forma `[N, 3, 320, 320]` (NCHW), `N` dinámico |
| Color | RGB (OpenCV entrega BGR: hay que convertir) |
| Resize | `cv2.INTER_LINEAR` directo a 320×320, **sin letterbox**, igual en entrenamiento e inferencia |
| Normalización | Valores `/255`, luego `(x - mean) / std`, con `mean=[0.485, 0.456, 0.406]` y `std=[0.229, 0.224, 0.225]` |
| Tensor de salida | nombre `logits`, `float32`, forma `[N, 7, 320, 320]`. **Sin softmax**: se aplica en la API |
| Opset ONNX | ≥ 17 |

Función compartida (única implementación de la inferencia): `backend/app/services/preprocess.py::preprocess_seg(image_bgr) -> np.ndarray`. `ml/` debe producir exactamente el mismo resultado, y QA lo prueba con una tolerancia de `1e-5`.

**Posproceso en la API:**
1. Softmax por píxel.
2. `argmax` para obtener la clase.
3. Si `max_prob < CONF_THRESHOLD`, el píxel pasa a `fondo` (0).
4. Se reescala con `INTER_NEAREST` al tamaño de trabajo.

**Definición de confianza por clase:** promedio de `max_prob` sobre los píxeles asignados a esa clase. No es la probabilidad de que la clase esté en la imagen, y la UI no debe presentarla así.

### 2.2 Profundidad (`midas_small.onnx`) — preentrenado, no entrenado por nosotros

| Propiedad | Valor |
|---|---|
| Modelo | MiDaS v2.1 small |
| Entrada | según el modelo oficial: RGB, `[1, 3, 256, 256]` |
| Salida interna | profundidad inversa relativa, sin unidades |
| Salida hacia el motor | `depth`: `float32`, forma `(H, W)`, rango `[0, 1]` por min-max, donde **1.0 = más cerca de la cámara** |

Es una medida relativa: no sirve para medir distancias. **Desde la versión 1.7.0 el motor no la usa**, porque el equipo eligió retirar el parallax. El modelo se conserva por ahora; su retirada se decide al integrar el modelo real.

### 2.3 Modelo dummy (solo desarrollo)

- `ml/make_dummy_model.py` genera en `models/dev/` (fuera de git) un `kinetia_seg.onnx` y un `midas_small.onnx` con las firmas de esta sección, pesos aleatorios y `version = "dummy-dev"` en su `manifest.json`. Solo necesita `onnx` y `numpy`, no PyTorch. En desarrollo: `MODELS_DIR=models/dev`.
- La API **se niega a arrancar** con él salvo que `ALLOW_DUMMY_MODEL=true`. Ese valor está prohibido en `k8s/` y en `docker-compose.yml`, y QA lo verifica.
- Con el dummy, `/api/v1/analyze` devuelve `model_version: "dummy-dev"` y la UI muestra un aviso visible.
- Su `metrics.json` lleva además `dummy: true` y `split: "dummy"`. Si la UI ve esos campos, muestra que no son resultados del modelo.

---

## 3. Artefactos

| Ruta (dentro del pod) | Genera | Se usa en producción | Contenido |
|---|---|---|---|
| `/models/kinetia_seg.onnx` | `ml/export.py` | Sí | Segmentación |
| `/models/midas_small.onnx` | `ml/fetch_depth_model.py` | Sí | Profundidad |
| `/models/manifest.json` | `ml/export.py` | Sí | Versión y metadatos |
| `/models/metrics.json` | `ml/evaluate.py` | Sí | Métricas del conjunto de prueba |
| `models/kinetia_seg.pt` (repo) | `ml/train.py` | **No** | Mejor checkpoint; la imagen `api` no incluye PyTorch |
| `models/hashes.sha256` (repo) | `ml/export.py` | **No** | SHA-256 de `kinetia_seg.pt` y `kinetia_seg.onnx`, formato `sha256sum`. Es lo que se copia a `k8s/base/model-hashes.env` |

Los binarios no se versionan en git. Se publican como asset de un GitHub Release y `models/` queda en `.gitignore`, salvo un `README` con los hashes.

**Entrenamiento:** lo ejecuta Erika Bonilla en Google Colab gratis. Al terminar, el notebook descarga `kinetia_seg.pt`, `kinetia_seg.onnx`, `metrics.json`, `manifest.json` y `hashes.sha256`, que se colocan en `models/` del repositorio.

**De dónde salen las métricas:** `metrics.json` se calcula evaluando el **`.onnx` exportado** con ONNX Runtime, no el `.pt`, para que mida lo mismo que corre en producción. `evaluate.py --verify models/metrics.json` repite la evaluación en otra máquina y compara con tolerancia `1e-3`. Pequeñas diferencias numéricas entre máquinas son esperables; el documento técnico cita siempre `metrics.json`.

**Versión de OpenCV:** `opencv-python-headless==4.10.0.84` en `ml/`, en el notebook y en `backend/`, porque el resize es parte de la entrada del modelo.

### `manifest.json`

```json
{
  "version": "seg-1.0.0",
  "created_at": "2026-11-10T12:00:00Z",
  "classes": ["fondo", "cielo", "agua", "vegetacion", "fuego_luz", "persona", "animal"],
  "input_size": [320, 320],
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "files": {
    "kinetia_seg.onnx": { "sha256": "<64 hex>", "bytes": 0 },
    "midas_small.onnx": { "sha256": "<64 hex>", "bytes": 0 }
  },
  "dataset_manifest_sha256": "<64 hex>"
}
```

### `metrics.json`

`evaluate.py` lo escribe. Es la **única** fuente de métricas para la API, la UI y el documento técnico.

```json
{
  "model_sha256": "<64 hex>",
  "dataset_manifest_sha256": "<64 hex>",
  "split": "test",
  "num_images": 0,
  "evaluated_at": "2026-11-10T12:00:00Z",
  "miou": 0.0,
  "dice_macro": 0.0,
  "pixel_accuracy": 0.0,
  "per_class": {
    "fondo": { "iou": 0.0, "dice": 0.0, "support_pixels": 0 }
  },
  "confusion_matrix": [[0]],
  "classes": ["fondo", "cielo", "agua", "vegetacion", "fuego_luz", "persona", "animal"]
}
```

`per_class` trae las 7 clases y `confusion_matrix` es de 7×7 (filas = verdad, columnas = predicción), en conteo de píxeles. Los valores anteriores solo ilustran el formato: no son resultados.

### Partición de datos

- `ml/splits/manifest.csv`, con columnas `image_id, source, split`. Partición 70/15/15 con `seed=42`, generada una sola vez y versionada en git. Se puede generar solo con las anotaciones, sin descargar las imágenes. Valores de `source`: `coco_stuff` y `bowfire`.
- `dataset_manifest_sha256` es el SHA-256 del archivo con los finales de línea normalizados a LF, para que Windows y Colab den el mismo hash.
- Las métricas se calculan a 320×320, sin los píxeles 255. mIoU y Dice promedian solo las clases con datos.
- `evaluated_at` y `created_at` de los archivos de modelo van sin milisegundos: la regla de milisegundos es de las fechas que genera la API. `/model/metrics` devuelve el archivo sin cambios.
- `train.py` escribe el `.pt`, y `export.py` lo copia a la carpeta de artefactos.
- `evaluate.py` termina con error si el hash del manifiesto no coincide con el de `manifest.json`.
- `train.py` no puede abrir imágenes de `split=test`. La prueba es un test automatizado.

---

## 4. API

**Base:** `/api/v1`. Todo es JSON, salvo las subidas (`multipart/form-data`) y `/metrics`. Las fechas van en ISO 8601 UTC con milisegundos (`2026-11-25T15:00:00.123Z`), para que `after` en la galería no pierda jobs creados en el mismo segundo.

### Formato único de error

```json
{ "error": { "code": "FILE_TOO_LARGE", "message": "El archivo pesa 22 MB; el máximo es 15 MB.", "details": { "max_bytes": 15728640 } } }
```

`message` está en español y es apto para mostrarse al usuario. `code` es estable y es lo que el frontend evalúa.

| HTTP | `code` | Cuándo |
|---|---|---|
| 400 | `MISSING_FILE` | No llegó archivo |
| 400 | `CORRUPT_FILE` | La extensión y los magic bytes son válidos, pero no se puede decodificar |
| 401 | `UNAUTHORIZED` | `X-Admin-Token` falta o es incorrecto |
| 404 | `JOB_NOT_FOUND` | El job no existe |
| 404 | `RESULT_NOT_FOUND` | El resultado no existe o expiró |
| 404 | `METRICS_NOT_FOUND` | Falta `metrics.json` |
| 409 | `JOB_NOT_READY` | Se pidió el resultado de un job sin terminar |
| 413 | `FILE_TOO_LARGE` | Más de `MAX_UPLOAD_BYTES` |
| 415 | `UNSUPPORTED_FORMAT` | Los magic bytes no son jpg, png, webp, mp4 ni mov |
| 422 | `RESOLUTION_TOO_LOW` | Lado menor por debajo de `MIN_SIDE_PX` |
| 422 | `RESOLUTION_TOO_HIGH` | Más de `MAX_PIXELS` (protege de bombas de descompresión) |
| 422 | `VIDEO_TOO_LONG` | Más de `MAX_VIDEO_SECONDS` |
| 422 | `INVALID_PARAMS` | Parámetro inválido. Reemplaza el 422 por defecto de FastAPI |
| 429 | `QUEUE_FULL` | Cola con `MAX_QUEUE` jobs. Incluye la cabecera `Retry-After` |
| 503 | `MODEL_NOT_READY` | El modelo aún no está cargado |
| 500 | `INTERNAL_ERROR` | Error inesperado, sin traza en la respuesta (la traza va al log) |
| 404 | `NOT_FOUND` | Ruta inexistente |
| 405 | `METHOD_NOT_ALLOWED` | Método no permitido en esa ruta |
| 416 | `RANGE_NOT_SATISFIABLE` | Cabecera `Range` fuera del archivo |

Solo dentro de `error` de un job fallido: `JOB_INTERRUPTED`, `ENGINE_UNAVAILABLE`, `ENCODER_UNAVAILABLE`.

**Validación:** el formato se decide por **magic bytes**, nunca por la extensión ni por el `Content-Type`. Los errores se comprueban en este orden: tamaño, formato, decodificación, resolución, duración.

**Normalización obligatoria de la entrada:** las fotos se rotan según su orientación EXIF antes de cualquier proceso, porque los celulares guardan así las fotos verticales.

### `POST /api/v1/analyze`

Multipart: `file` (imagen o video; de un video se analiza el primer frame) e `include_depth` (bool, por defecto `false`).

**200:**

```json
{
  "width": 1280,
  "height": 720,
  "mask_png_base64": "<PNG 8 bits, 1 canal, valor de píxel = índice de clase>",
  "classes": [
    { "id": 1, "name": "cielo", "pixel_pct": 41.2, "confidence": 0.93 }
  ],
  "inference_ms": { "segmentation": 84.1, "depth": null, "total": 112.5 },
  "depth_png_base64": null,
  "model_version": "seg-1.0.0",
  "model_sha256": "<64 hex>",
  "conf_threshold": 0.5
}
```

- `width` y `height` son el tamaño de trabajo: la entrada reescalada a un lado mayor ≤ `WORK_MAX_SIDE`, sin ampliar nunca.
- `classes` incluye solo las clases con `pixel_pct > 0`, en orden descendente de `pixel_pct`. Los valores anteriores ilustran el formato.
- `inference_ms` se **mide** en cada petición.

### `POST /api/v1/animate`

Multipart:

| Campo | Tipo | Por defecto | Reglas |
|---|---|---|---|
| `file` | archivo | requerido | jpg, png, webp, mp4, mov |
| `mood` | string | `calma` | `calma`, `epico`, `magico`, `nostalgico` |
| `output_format` | string | `mp4` | `mp4`, `gif` |
| `duration_s` | float | `4.0` | 2.0 a 6.0. Solo imágenes; en un video se ignora |
| `mask_strokes` | string JSON | ausente | Máscara opcional del pincel, ver abajo |
| `brush_mode` | string | `add` | `add`: suma al resultado del modelo. `only`: anima solo lo pintado |
| `gallery_opt_in` | bool | `false` | Consentimiento para mostrar el resultado en la galería |
| `seed` | int | aleatorio | Hace reproducible la animación; se guarda en el job |

La máscara viaja como **trazos vectoriales** y no como PNG, porque cada trazo debe llevar también su dirección. El servidor la rasteriza.

```json
{
  "strokes": [
    {
      "points": [[0.12, 0.40], [0.18, 0.38]],
      "radius": 0.04,
      "direction_deg": 90.0,
      "strength": 0.7
    }
  ]
}
```

- `points`: coordenadas normalizadas de 0 a 1 respecto a la imagen; origen arriba a la izquierda.
- `radius`: fracción del lado mayor (0.005 a 0.2).
- `direction_deg`: 0 a 360; 0 = derecha, 90 = abajo (coordenadas de imagen).
- `strength`: 0 a 1.
- Límites: máximo 200 trazos y 2000 puntos en total.

**202:**

```json
{ "job_id": "0b7c3b0e-5a1e-4f7a-9d55-2a9a7f1c9e10", "status": "queued", "queue_position": 2, "status_url": "/api/v1/jobs/0b7c3b0e-5a1e-4f7a-9d55-2a9a7f1c9e10" }
```

### `GET /api/v1/jobs/{id}`

**200:**

```json
{
  "job_id": "<uuid>",
  "status": "succeeded",
  "stage": "done",
  "progress": 1.0,
  "mood": "calma",
  "seed": 5,
  "created_at": "2026-11-25T15:00:00Z",
  "updated_at": "2026-11-25T15:00:21Z",
  "result": {
    "url": "/api/v1/results/<uuid>.mp4",
    "format": "mp4",
    "width": 1280,
    "height": 720,
    "duration_s": 4.0,
    "fps": 24,
    "size_bytes": 0
  },
  "timings_ms": { "segmentation": 0, "depth": 0, "render": 0, "encode": 0, "total": 0 },
  "error": null
}
```

- `status`: `queued`, `running`, `succeeded`, `failed`.
- `stage`: `queued`, `segmenting`, `depth`, `rendering`, `encoding`, `done`.
- Si falla, `result` es `null` y `error` tiene la forma `{ "code", "message" }`.
- En `timings_ms`, una etapa aún no medida vale `null`, nunca 0.
- Un job que estaba en curso cuando se reinició el pod queda en `failed` con código `JOB_INTERRUPTED`.
- El frontend consulta cada 1 s mientras el job no termine.

### Otros endpoints

| Endpoint | Respuesta |
|---|---|
| `GET /api/v1/health` | **200** `{ "status": "ok", "model_loaded": true, "depth_loaded": true, "model_version": "seg-1.0.0", "model_sha256": "<hex>", "depth_sha256": "<hex>" }`. **503** con `status: "loading"` y los campos `*_loaded` en `false` hasta que **ambos** modelos estén cargados |
| `GET /api/v1/health/live` | **200** `{ "status": "alive" }`. No toca el modelo ni el disco |
| `GET /api/v1/model/metrics` | **200** con el contenido literal de `metrics.json`, sin transformar |
| `GET /metrics` | Formato de texto Prometheus; ver sección 8 |

### Endpoints añadidos a la lista original

Son necesarios para cumplir las funciones pedidas (resultado, QR, galería).

| Endpoint | Para qué | Respuesta |
|---|---|---|
| `GET /api/v1/results/{job_id}.{ext}` | Descargar o reproducir. **Debe soportar cabeceras `Range`**: Safari en iPhone no reproduce video sin ellas | Binario, `video/mp4` o `image/gif` |
| `GET /api/v1/config` | Configuración pública para el frontend y el QR | `{ "public_base_url": "http://192.168.0.10:8080", "limits": { "max_upload_bytes": 15728640, "max_video_seconds": 10, "min_side_px": 256, "formats": ["jpg","png","webp","mp4","mov"] }, "moods": ["calma","epico","magico","nostalgico"], "classes": [{ "id": 0, "name": "fondo", "label": "Fondo", "color": "#3A3F4B" }] }` |
| `GET /api/v1/gallery?limit=24&after=<ISO>` | Pantalla grande | `{ "items": [{ "job_id": "<uuid>", "mood": "calma", "result_url": "/api/v1/results/<uuid>.mp4", "created_at": "<ISO>" }] }`. Solo jobs con `gallery_opt_in=true`, en orden del más reciente al más antiguo |
| `DELETE /api/v1/gallery/{job_id}` | Moderación durante la Expo. Requiere la cabecera `X-Admin-Token` | **204** |

---

## 5. Motor de animación

Paquete `backend/animation_engine/`. **Es puro:** solo usa NumPy y OpenCV, y no importa FastAPI, ONNX Runtime ni SQLite. Recibe las máscaras ya calculadas, de modo que puede probarse con máscaras sintéticas.

### Formatos

| Dato | Formato |
|---|---|
| `image_bgr` / frame | `np.ndarray`, `uint8`, forma `(H, W, 3)`, BGR |
| `mask` | `np.ndarray`, `uint8`, forma `(H, W)`, valores de 0 a 6 (índices de la sección 1) |
| `depth` | `np.ndarray`, `float32`, forma `(H, W)`, rango `[0, 1]`, 1 = cerca |

`mask` y `depth` tienen el mismo `(H, W)` que la imagen. Si no, se lanza `ValueError`.

### Firmas

```python
# backend/animation_engine/types.py
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Literal, Optional
import numpy as np

MoodName = Literal["calma", "epico", "magico", "nostalgico"]
ParticleKind = Literal["none", "polen", "brasas", "chispas", "polvo"]

@dataclass(frozen=True)
class MoodParams:
    name: MoodName
    speed: float              # multiplica la velocidad de todos los efectos
    saturation: float         # multiplica la saturación HSV
    warmth: float             # -1 (frío) a 1 (cálido)
    contrast: float           # multiplica el contraste
    particle_kind: ParticleKind
    particle_density: float   # partículas por megapíxel
    parallax_amplitude: float # desplazamiento máximo, fracción del ancho
    vignette: float           # 0 a 1

@dataclass(frozen=True)
class BrushStroke:
    points: tuple[tuple[float, float], ...]  # normalizados de 0 a 1
    radius: float
    direction_deg: float
    strength: float

@dataclass(frozen=True)
class RenderConfig:
    mood: MoodParams
    duration_s: float = 4.0
    fps: int = 24
    seed: int = 0
    crossfade_s: float = 0.75
    strokes: tuple[BrushStroke, ...] = ()
    brush_mode: Literal["add", "only"] = "add"

ProgressFn = Callable[[float], None]  # recibe valores de 0 a 1
SegmentFn = Callable[[np.ndarray], np.ndarray]  # frame BGR -> mask (H, W)
```

```python
# backend/animation_engine/__init__.py
def get_mood(name: MoodName) -> MoodParams: ...

def animate_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    depth: Optional[np.ndarray],   # se valida, pero no cambia el resultado
    config: RenderConfig,
    progress: Optional[ProgressFn] = None,
) -> Iterator[np.ndarray]:
    """Genera round(duration_s * fps) frames BGR. Es un loop: el último frame
    empalma con el primero mediante crossfade."""

def animate_video(
    frames: Iterable[np.ndarray],
    segment: SegmentFn,
    infer_every_n: int,
    config: RenderConfig,
    depth: Optional[np.ndarray] = None,
    progress: Optional[ProgressFn] = None,
) -> Iterator[np.ndarray]:
    """Llama a segment() cada infer_every_n frames y, entre medias, propaga la
    máscara con flujo óptico Farneback. No acumula todos los frames en memoria."""
```

**Reglas:**
- Con el mismo `seed` y la misma entrada, la salida debe ser idéntica byte a byte.
- Los generadores producen frames de uno en uno, para no agotar la memoria.
- El motor no codifica: `backend/app/services/encoder.py` envía los frames a FFmpeg.

### Codificación de salida

| Formato | Parámetros |
|---|---|
| MP4 | H.264 (`libx264`), `yuv420p`, `+faststart`, dimensiones pares. `cv2.VideoWriter` queda prohibido porque su `mp4v` no se reproduce en navegadores |
| GIF | Máximo 480 px de lado mayor, 12 fps, paleta generada con `palettegen`/`paletteuse` |

### Presets de mood (valores iniciales)

Son valores de partida, ajustables por el Agente Motor tras verlos en pantalla. Si los cambia, lo reporta, y la tabla se actualiza.

| Mood | speed | saturation | warmth | contrast | particle_kind | particle_density | parallax_amplitude | vignette |
|---|---|---|---|---|---|---|---|---|
| `calma` | 0.6 | 0.95 | 0.05 | 1.00 | `polen` | 30 | 0.0 | 0.10 |
| `epico` | 1.4 | 1.15 | -0.10 | 1.20 | `brasas` | 75 | 0.0 | 0.35 |
| `magico` | 1.0 | 1.25 | -0.05 | 1.05 | `chispas` | 110 | 0.0 | 0.20 |
| `nostalgico` | 0.8 | 0.70 | 0.35 | 0.95 | `polvo` | 50 | 0.0 | 0.40 |

`parallax_amplitude` se conserva en 0 para no romper la firma; el motor lo ignora.

**Comportamientos precisados:**

- **`fuego_luz`:** el brillo sale de la región, hasta unos 280 px con un lado mayor de 1280 (`glow_reach_px`). Es la única excepción a "los efectos no tocan otras clases".
- **Modo `only`:** sin parallax ni efectos de clase. Lo pintado se mueve según el pincel; el resto lleva el color del mood, pero no cambia en el tiempo.
- **`particle_kind="none"`:** apaga también las hojas de la regla de bosque.
- **Video:** noche, bosque, partículas y encuadre se fijan con el primer frame (en noche, el primero que tenga cielo), para evitar parpadeos. `duration_s` es el ciclo de los efectos. Un video de entrada no cierra en loop.
- **Validación:** `animate_image` lanza `ValueError` al llamarla, no en el primer `next`, y rechaza máscaras que no sean `uint8` o con valores mayores que 6.
- **Hilos:** el motor limita OpenCV a 4 hilos durante el render. `cv2.setNumThreads` es global, y se acepta que afecte a otro código de OpenCV durante ese tiempo: cambia su duración, no su resultado.

---

## 6. Variables de entorno

**Contenedor `api`.** Fuente: ConfigMap `kinetia-config`, excepto `ADMIN_TOKEN`, que viene del Secret `kinetia-secret`. Se documentan en `.env.example`.

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `MODELS_DIR` | `/models` | Carpeta de modelos |
| `SEG_MODEL_FILE` | `kinetia_seg.onnx` | Modelo de segmentación |
| `DEPTH_MODEL_FILE` | `midas_small.onnx` | Modelo de profundidad |
| `RESULTS_DIR` | `/data/results` | Resultados |
| `DB_PATH` | `/data/db/kinetia.db` | SQLite (modo WAL) |
| `TMP_DIR` | `/shared-tmp/api` | Temporales |
| `CONF_THRESHOLD` | `0.5` | Umbral de confianza por píxel |
| `MAX_UPLOAD_BYTES` | `15728640` | 15 MiB |
| `MAX_VIDEO_SECONDS` | `10` | Duración máxima |
| `MAX_FRAMES` | `300` | Frames máximos; un video con más fps se remuestrea |
| `MIN_SIDE_PX` | `256` | Resolución mínima |
| `MAX_PIXELS` | `40000000` | Resolución máxima de entrada |
| `WORK_MAX_SIDE` | `1280` | Lado mayor de trabajo |
| `VIDEO_INFER_EVERY_N` | `5` | Inferencia cada N frames |
| `NIGHT_LUMA_THRESHOLD` | `60` | Regla de noche (0 a 255) |
| `FOREST_MIN_PCT` | `25` | Regla de bosque (%) |
| `MAX_QUEUE` | `8` | Jobs en cola antes de responder 429 |
| `RESULT_TTL_HOURS` | `96` | Limpieza de resultados |
| `ORT_INTRA_OP_THREADS` | `4` | Hilos de ONNX Runtime |
| `PUBLIC_BASE_URL` | `http://localhost:8080` | URL del QR. `make demo` la fija con la IP de la red local |
| `ALLOW_DUMMY_MODEL` | `false` | `true` solo en desarrollo local |
| `LOG_LEVEL` | `INFO` | Nivel de log |
| `ADMIN_TOKEN` | sin valor | Moderación de galería. Secreto |

**InitContainer `model-loader`:** `MODELS_DIR=/models`, `EXPECTED_SHA256_SEG` y `EXPECTED_SHA256_DEPTH` (ConfigMap `kinetia-model-hashes`).

**Build del frontend:** sin variables; llama a la ruta relativa `/api/v1` y lee el resto en `GET /api/v1/config`.

---

## 7. Kubernetes

### Pod único

| Contenedor | Imagen | Puerto | Usuario |
|---|---|---|---|
| `model-loader` (init) | `ghcr.io/<owner>/kinetia-model-loader:<sha>` | ninguno | 1000 |
| `frontend` | `ghcr.io/<owner>/kinetia-frontend:<sha>` | **80** | 101 |
| `api` | `ghcr.io/<owner>/kinetia-api:<sha>` | **8000** | 1000 |

- **`model-loader`:** la imagen contiene los `.onnx`. Si faltan en `/models`, los copia al PVC. Después calcula el SHA-256 y lo compara con los valores **del ConfigMap**, no con los de la imagen. Si no coincide, termina con código 1 y el pod no arranca. Sin esta copia, un clúster limpio tendría el PVC vacío y el pod nunca iniciaría.
- **Puerto 80 sin root:** se cumple con `securityContext.sysctls: net.ipv4.ip_unprivileged_port_start=80` a nivel de pod, un sysctl considerado seguro por Kubernetes.
- **Nginx:**
  - `location /api/` hace proxy a `http://127.0.0.1:8000`.
  - Rutas desconocidas vuelven a `index.html`.
  - `client_max_body_size 20m`, mayor que el límite de la API, para que el 413 lo responda la API en JSON.
  - `/metrics` **no** se publica a través de Nginx: `location = /metrics { return 404; }`, porque si no el fallback a `index.html` respondería 200.
  - `proxy_read_timeout 120s`.

### Volúmenes

| Volumen | Tipo | `model-loader` | `frontend` | `api` |
|---|---|---|---|---|
| `kinetia-data` (subPath `models`) | PVC RWO, 5 Gi | `/models` rw | no | `/models` **ro** |
| `kinetia-data` (subPath `results`) | mismo PVC | no | no | `/data/results` rw |
| `kinetia-data` (subPath `db`) | mismo PVC | no | no | `/data/db` rw |
| `shared-tmp` | `emptyDir` (`sizeLimit: 2Gi`) | no | `/shared-tmp` | `/shared-tmp` |
| `nginx-run` | `emptyDir` | no | `/var/cache/nginx`, `/var/run` (por subPath) | no |

Nginx guarda en `/shared-tmp/nginx` los cuerpos de subida en tránsito. La API usa `/shared-tmp/api`. El pod declara `fsGroup: 1000`, y `shared-tmp` debe permitir escritura a ambos usuarios.

`readOnlyRootFilesystem: true` y `runAsNonRoot: true` en todos los contenedores.

### Recursos

| Nombre | Tipo | Valor |
|---|---|---|
| `kinetia` | Namespace | |
| `kinetia` | Deployment | `replicas: 1`, `strategy: Recreate`, sin HPA |
| `kinetia` | Service | `NodePort`, `port: 80`, `targetPort: 80`, `nodePort: 30080` |
| `kinetia` | Ingress (solo `overlays/local`) | Host `kinetia.local` |
| `kinetia-config` | ConfigMap | Sección 6 |
| `kinetia-model-hashes` | ConfigMap | Hashes esperados, generado desde `k8s/base/model-hashes.env` |
| `kinetia-secret` | Secret | Se crea con un script desde `.env`; nunca se versiona |
| `kinetia-data` | PersistentVolumeClaim | 5 Gi, RWO |

**Clúster k3d:** k3s se arranca con `--disable-cloud-controller`, porque `cloud-controller-manager` provocaba un bucle de reinicios del nodo en k3s v1.35.5 (causa raíz sin determinar). El mapeo de puertos requiere el proxy de k3d. Los modelos del CI se descargan del Release indicado en la variable de repositorio `MODELS_RELEASE_TAG` (`models-latest` por defecto).

**Modelos antiguos en el PVC:** si el hash del volumen no coincide con el ConfigMap y el de la imagen sí, `model-loader` reemplaza el archivo del volumen y vuelve a verificar. Si reemplaza el modelo de segmentación, también reemplaza `manifest.json` y `metrics.json`, porque lo describen. Si tampoco coincide el de la imagen, termina con código 1.

**Exposición local (k3d):** `-p "8080:30080@server:0"`, es decir, `http://<IP-del-portátil>:8080`. El QR usa la **IP**, porque los celulares no resuelven `kinetia.local`.

### Probes (contenedor `api`, directo al puerto 8000)

| Probe | Ruta | Parámetros iniciales |
|---|---|---|
| `startupProbe` | `GET /api/v1/health` | `periodSeconds: 5`, `failureThreshold: 36` (hasta 180 s) |
| `readinessProbe` | `GET /api/v1/health` | `periodSeconds: 10`, `failureThreshold: 3` |
| `livenessProbe` | `GET /api/v1/health/live` | `periodSeconds: 20`, `failureThreshold: 3` |

### `requests` y `limits`

**No se fijan aquí.** El Agente DevOps los obtiene de una medición real (`kubectl top`, con carga de QA) y los registra con su evidencia en `PROGRESO.md`. Hasta entonces, los manifiestos llevan valores provisionales marcados con el comentario `# PROVISIONAL`, que no pueden quedar en la entrega final.

### Plan C (`docker-compose.yml`)

Usa las mismas imágenes. El servicio `frontend` declara `network_mode: "service:api"` para compartir red con la API, igual que en el pod, y así `127.0.0.1:8000` sigue funcionando sin cambiar la configuración de Nginx. El mapeo `8080:80` se publica en el servicio `api`.

---

## 8. Observabilidad

**Logs:** JSON, una línea por evento, con los campos `ts`, `level`, `event`, `request_id`, `job_id` (si aplica), `duration_ms` (si aplica) y `error_code` (si aplica). **No se registra el contenido de las imágenes** ni datos de los visitantes.

**Métricas de Prometheus** (librería `prometheus_client`; el `/metrics` se consulta con `kubectl port-forward`):

| Métrica | Tipo | Etiquetas |
|---|---|---|
| `kinetia_inference_seconds` | Histogram | `model` = `seg` o `depth` |
| `kinetia_job_duration_seconds` | Histogram | `media` = `image` o `video` |
| `kinetia_jobs_total` | Counter | `status` = `succeeded` o `failed` |
| `kinetia_errors_total` | Counter | `code` |
| `kinetia_queue_depth` | Gauge | ninguna |

No se despliega un servidor Prometheus, porque sería otro pod y rompería la restricción. Se expone el endpoint, y `kubectl top` usa el `metrics-server` que k3s ya trae en `kube-system`.

---

## 9. Frontend

| Ruta | Vista |
|---|---|
| `/` | Estudio: subir, Rayos X, mood, pincel, comparador y exportar |
| `/expo/subir` | Versión móvil a la que lleva el QR; incluye la casilla de consentimiento para la galería |
| `/expo/pantalla` | Muro para la pantalla grande con el QR visible |
| `/modelo` | Métricas del modelo, tal como las devuelve `GET /api/v1/model/metrics` |

- Los colores, las clases y los límites se leen de `GET /api/v1/config`; no se duplican en el código.
- La cámara se abre con `<input type="file" accept="image/*,video/*" capture>`. No se usa `getUserMedia`, porque exige HTTPS y el QR va por HTTP en la red local.
- `/expo/subir` tiene además un botón "Elegir de mi galería", con un segundo input sin `capture`, porque `capture` obliga a usar la cámara en móvil.

---

## 10. Reglas para todos

1. **Nada simulado.** Ninguna predicción, métrica o tiempo puede estar escrito a mano ni precargado. Los valores numéricos de este archivo que parecen resultados solo ilustran el formato.
2. **No se inventan métricas.** Si un criterio no se cumple, se reporta la cifra real.
3. Los identificadores del código van en inglés, salvo los nombres de clases y moods definidos aquí. Los mensajes al usuario y la documentación van en español.
4. Python 3.11, dependencias con versión fija.

---

## Historial

| Versión | Fecha | Cambio | Motivo |
|---|---|---|---|
| 1.0.0-borrador | 2026-11-02 | Versión inicial | Fase 0 |
| 1.7.0 | 2026-11-04 | Sin parallax: el sujeto y el fondo quedan quietos; `depth` no cambia el resultado | Decisión de Erika tras ver los videos en movimiento (opción A) |
| 1.6.0 | 2026-11-04 | `/metrics` en Nginx, bandera de k3s, Release de modelos y reemplazo de modelos antiguos en el PVC | Propuestas del Agente DevOps |
| 1.5.0 | 2026-11-03 | Ruta `/modelo` y botón de galería en el frontend | Propuestas del Agente Frontend. Pendientes para la Fase 2: límites de duración, pincel y `max_pixels` en `/config`, y `queue_position` en jobs |
| 1.4.0 | 2026-11-03 | BoWFire como fuente de fuego, con su limitación; valores de `source`; definición del hash del manifiesto y de las métricas | Decisión D2 y propuestas del Agente ML |
| 1.3.0 | 2026-11-03 | Densidades finales de partículas y comportamientos precisados del motor | Propuestas del Agente Motor, aceptadas |
| 1.2.0 | 2026-11-02 | `label` en `/config`; `seed` en jobs; códigos de error añadidos; `stage: done`; `null` en tiempos sin medir; fechas con milisegundos; marca del `metrics.json` dummy | Propuestas del Agente Backend, aceptadas |
| 1.1.0 | 2026-11-02 | `metrics_test.json` pasa a `metrics.json`; se añaden `hashes.sha256`, modelos dummy en `models/dev/`, evaluación sobre el ONNX y versión fija de OpenCV; se quita `sha256_image` del manifiesto | Aprobación del equipo y flujo de entrenamiento en Colab |
