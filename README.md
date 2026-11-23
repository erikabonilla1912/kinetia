# KINETIA — Dale vida a tus imágenes

Aplicación web que anima fotos y videos cortos. Una red de segmentación semántica, **entrenada desde cero por el equipo**, clasifica cada píxel en 7 clases (fondo, cielo, agua, vegetación, fuego/luz, persona, animal), y un motor de animación mueve cada zona según su clase. Proyecto de Inteligencia Artificial para ExpoUNINPAHU 2026.

**Equipo:** Erika Julieth Bonilla Florez, Juan Felipe Delgadillo Vanegas y Jeisson David Hernandez Tibata.

## Contenido del repositorio

| Ruta | Qué hay |
| --- | --- |
| `notebooks/kinetia_training.ipynb` | Notebook de Colab: exploración, preprocesamiento, entrenamiento, validación, evaluación, gráficas y conclusiones |
| `ml/` | Código de entrenamiento (`train.py`), evaluación (`evaluate.py`), exportación a ONNX (`export.py`) y pruebas |
| `backend/` | API en FastAPI y motor de animación (`animation_engine/`) |
| `frontend/` | Interfaz en Angular 20 |
| `model-loader/` | Contenedor que verifica el SHA-256 del modelo antes de arrancar la API |
| `k8s/` | Manifiestos de Kubernetes (Kustomize) |
| `scripts/` | Scripts de despliegue para Windows (PowerShell) |
| `docs/` | Documento técnico, contratos de interfaces y bitácora del proyecto |

## Modelo entrenado

- **Arquitectura:** U-Net con encoder MobileNetV3 Large (6,69 M de parámetros), pesos iniciales aleatorios, sin *transfer learning*.
- **Datos:** 10 583 imágenes de entrenamiento de COCO-Stuff y BoWFire; 2268 de validación y 2268 de prueba.
- **Resultado en prueba (2268 imágenes no vistas):** mIoU 0,6347 · Dice 0,7616 · exactitud de píxel 0,8535. Detalle en `models/metrics.json`.

Los archivos del modelo no se guardan en git. Están en el Release **[models-latest](https://github.com/erikabonilla1912/kinetia/releases/tag/models-latest)**:

| Archivo | Para qué |
| --- | --- |
| `kinetia_seg.onnx` | Modelo que usa la aplicación (ONNX Runtime) |
| `kinetia_seg.pt` | Checkpoint de PyTorch |
| `manifest.json` | Versión, clases, normalización y hashes |
| `metrics.json` | Métricas del conjunto de prueba y matriz de confusión |
| `hashes.sha256` | SHA-256 de los archivos |

Descárgalos y colócalos en la carpeta `models/`. El SHA-256 del `.onnx` debe ser el de `k8s/base/model-hashes.env`; si no coincide, la aplicación no arranca.

## Requisitos

- Windows 10/11 con PowerShell
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) encendido
- [k3d](https://k3d.io/) y `kubectl`
- 8 GB de RAM libres como mínimo. No se necesita GPU
- Opcional: `make` (si no está, se usan los scripts `.ps1` directamente)

## Ejecutar la aplicación

```powershell
# 1. Clonar el repositorio
git clone https://github.com/erikabonilla1912/kinetia.git
cd kinetia

# 2. Copiar los archivos del modelo (del Release) a models/

# 3. Levantar todo: construye las imágenes, crea el clúster k3d y despliega un pod
make demo
#    sin make:
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\demo.ps1
```

Cuando termina, la aplicación queda en **http://localhost:8080**.

| Acción | Con make | Sin make |
| --- | --- | --- |
| Ver el estado | `make status` | `scripts\status.ps1` |
| Prueba de humo | `make smoke` | `tests\smoke\smoke.ps1` |
| Ver los logs | `make logs` | `kubectl --context k3d-kinetia -n kinetia logs deploy/kinetia --all-containers` |
| Apagar | `make down` | `scripts\down.ps1` |
| Sin Kubernetes (Docker Compose) | `make planc` | `scripts\planc.ps1 -Action up` |

### Acceso desde otros dispositivos

Para probar desde un celular se puede exponer con Cloudflare Tunnel y volver a desplegar con la URL pública:

```powershell
cloudflared tunnel --url http://localhost:8080
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\demo.ps1 -PublicBaseUrl https://<url-que-entrega-cloudflared>
```

## Configuración

La configuración está en `k8s/base/config.env` (Docker Compose usa el mismo archivo). Valores principales:

| Variable | Valor | Significado |
| --- | --- | --- |
| `CONF_THRESHOLD` | 0.5 | Píxeles con menos confianza pasan a fondo |
| `MAX_UPLOAD_BYTES` | 15728640 | Tamaño máximo del archivo (15 MB) |
| `MIN_SIDE_PX` | 256 | Lado mínimo de la imagen |
| `MAX_VIDEO_SECONDS` | 10 | Duración máxima del video |
| `MAX_QUEUE` | 8 | Trabajos en cola como máximo |
| `RESULT_TTL_HOURS` | 96 | Horas que se guardan los videos generados |

El token de administrador (para retirar publicaciones del muro) se define como `ADMIN_TOKEN` en un archivo `.env` que **no** se sube al repositorio. `scripts\create-secret.ps1` lo carga como Secret de Kubernetes; si no existe, genera uno aleatorio.

## Usar el modelo sin la aplicación

El modelo se carga sin reentrenar y predice sobre imágenes nuevas:

- **En el notebook:** sección 12, "Predicción sobre una imagen nueva, sin reentrenar".
- **Con la API:** `POST /api/v1/analyze` con un archivo devuelve la máscara en PNG, el porcentaje y la confianza de cada clase.

```powershell
curl.exe -F "file=@mi_foto.jpg" http://localhost:8080/api/v1/analyze
```

- **Repetir la evaluación:** descarga solo las imágenes de prueba y vuelve a calcular las métricas con el `.onnx`, comparándolas con las publicadas (tolerancia 0,001):

```powershell
python ml/fetch_test_set.py
python ml/evaluate.py --models-dir models --data-root data/test --verify models/metrics.json
```

## Endpoints de la API

| Método | Ruta | Qué hace |
| --- | --- | --- |
| POST | `/api/v1/analyze` | Segmenta una imagen y responde de inmediato |
| POST | `/api/v1/animate` | Crea un trabajo de animación (responde 202) |
| GET | `/api/v1/jobs/{id}` | Estado y progreso del trabajo |
| GET | `/api/v1/results/{archivo}` | Descarga el MP4 o GIF |
| GET | `/api/v1/gallery` | Publicaciones del muro |
| DELETE | `/api/v1/gallery/{id}` | Retira una publicación (requiere token) |
| GET | `/api/v1/health` | Estado del servicio y del modelo |
| GET | `/api/v1/model/metrics` | Métricas del modelo |

## Pruebas

```powershell
# API y motor de animación
cd backend
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python -m pytest animation_engine/tests -q

# Entrenamiento y evaluación (desde la raíz)
python -m pip install -r ml/requirements.txt
python -m pytest ml/tests -q

# Interfaz
cd frontend
npm ci
npm test
```

Las pruebas de la API cubren entradas válidas e inválidas: formato no admitido, archivo demasiado grande, resolución baja o excesiva, video demasiado largo y cola llena. GitHub Actions ejecuta las pruebas de la API, del motor y de la interfaz en cada cambio.

## Entrenar de nuevo (opcional)

El entrenamiento completo se hizo en Google Colab con GPU (100 épocas, 4,4 horas): abrir `notebooks/kinetia_training.ipynb` y ejecutar las celdas en orden. Para ver el proceso en vivo en un portátil hay una demostración corta en CPU (32 imágenes, 8 épocas) que no toca el modelo de la aplicación. Necesita los datos preparados en `data/processed`:

```powershell
python ml/demo_entrenamiento.py
```

## Documentación

- Documento técnico y arquitectónico (stack, C4, UML, modelo): `docs/entrega/`
- Contratos de interfaces: [docs/CONTRATOS.md](docs/CONTRATOS.md)
- Bitácora y decisiones: [docs/PROGRESO.md](docs/PROGRESO.md)
