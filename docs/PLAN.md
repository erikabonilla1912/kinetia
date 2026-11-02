# KINETIA — Plan de ejecución (Fase 0)

**Fecha:** 2026-11-02 · **Expo:** 25 a 28 de noviembre de 2026 · **Estado:** esperando aprobación

## 1. Punto de partida (revisado hoy en el portátil)

| Elemento | Estado | Consecuencia |
|---|---|---|
| CPU / RAM | i5-1235U, 15.7 GB | Justifica MobileNetV3. La memoria queda justa con Docker, k3d y el navegador abiertos |
| GPU | Intel Iris Xe, sin NVIDIA | No se puede entrenar aquí: el entrenamiento real se hace en Colab |
| Disco libre | 61 GB | Suficiente; en el portátil solo se guarda el conjunto de prueba |
| Python 3.11.9, Node 22.14, Angular CLI, git | Instalados | Listos |
| Docker Desktop, kubectl | Instalados; el motor de Docker no estaba encendido | Hay que iniciarlo |
| k3d, make, ffmpeg, gh | **No instalados** | Se instalan al aprobar el plan (ffmpeg solo hace falta dentro del contenedor y para desarrollo local) |
| Repositorio | Carpeta vacía, sin git | Hay que crear el repositorio en GitHub |

## 2. Fases y fechas

Quedan 23 días. El lunes 16 de noviembre es festivo en Colombia. El **congelamiento de código es el 22 de noviembre**: después solo se corrigen fallos.

| Fase | Fechas | Qué se hace | Cierra con |
|---|---|---|---|
| **0** | 2–3 nov | Plan, estructura, contrato | Tu aprobación |
| **1** | 3–11 nov | ML, Backend, Frontend y Motor en paralelo | Modelo entrenado; API con dummy; UI; efectos con máscaras de prueba |
| **2** | 11–18 nov | Integración, QA, DevOps | App completa en un solo pod en k3d |
| **3** | 19–24 nov | Documentación, verificación cruzada, ensayos | Entregables listos; demo ensayada |
| **Expo** | 25–28 nov | Presentación | |

### Hitos de control

| Fecha | Hito | Si falla |
|---|---|---|
| 5 nov | **Compuerta de datos:** conteo real de píxeles por clase y decisión sobre fuego | Se aplica la alternativa de D2 |
| 6–10 nov | **Entrenamiento en Colab** (lo ejecutan ustedes) | Modelo base el 8 nov como respaldo; se reanuda desde Drive si Colab corta la sesión |
| 11 nov | **Compuerta del modelo:** `evaluate.py` sobre el conjunto de prueba | Si mIoU < 0.55: una sola ronda de mejora hasta el 13 nov y luego se reporta la cifra real |
| 13 nov | Flujo completo funcionando fuera del clúster con el modelo real | Se recorta alcance según la sección 6 |
| 18 nov | Un pod `Running` en k3d, con prueba de borrado y recreación | Se ensaya el Plan C (docker-compose) y se sigue depurando |
| 22 nov | Congelamiento y prueba de integración final en clúster limpio | |
| 23–24 nov | Dos ensayos completos con fotos tomadas en el momento | |

## 3. Tareas por subagente

| # | Subagente | Fase | Entregables | Criterios de aceptación |
|---|---|---|---|---|
| T1 | ML: datos | 1 | `ml/dataset.py`, `ml/class_mapping.yaml`, `ml/splits/manifest.csv`, `ml/make_dummy_model.py` | Partición 70/15/15 reproducible; conteo de píxeles por clase generado por script; test automatizado que impide a `train.py` abrir datos de prueba |
| T2 | ML: entrenamiento | 1 | `notebooks/kinetia_training.ipynb`, `ml/train.py`, `ml/export.py`, `ml/evaluate.py`, `ml/fetch_depth_model.py` | Ciclo completo probado con una muestra mínima en CPU antes de ir a Colab; tras Colab: mIoU en test ≥ 0.55; ONNX y `.pt` coinciden con tolerancia `1e-4`; la celda final predice sin reentrenar |
| T3 | Backend | 1, 2 | `backend/app/`, `backend/tests/`, dependencias | Todos los endpoints y errores de CONTRATOS; `/analyze` en menos de 2 s con 1080p; arranca solo con modelo válido |
| T4 | Motor | 1 | `backend/animation_engine/`, pruebas, galería de muestras | Seis efectos y cuatro moods visibles; salida idéntica con el mismo `seed`; imagen de 4 s en 30 s o menos en este portátil, a medir |
| T5 | Frontend | 1, 2 | `frontend/` | `ng build` y pruebas pasan; pincel, comparador, Rayos X, QR y galería; funciona en un celular real |
| T6 | QA | 2 | `backend/tests/`, `tests/smoke/`, pruebas del modelo | 100 % de pruebas en verde; casos inválidos; inferencia determinística; informe de rendimiento con cifras reales |
| T7 | DevOps | 2 | `Dockerfile`s, `k8s/`, `.github/workflows/`, `docker-compose.yml`, `Makefile`, `scripts/` | `make demo` desde clúster limpio; 1 pod `Running`; probes pasan; el pod se recrea; un hash incorrecto impide el arranque; pipeline en verde; recursos medidos |
| T8 | Documentación | 3 | `docs/documento_tecnico.md`, diagramas, `README.md`, guion, banco de preguntas | Métricas idénticas a `evaluate.py`; cada elemento de los diagramas existe en el código o en los manifiestos |

**Independencia de QA:** el Agente QA arranca sin contexto de ML. Recibe solo CONTRATOS.md y el `.onnx`, y se le prohíbe leer `ml/` y `notebooks/`. Es una regla por instrucción, no una barrera técnica, así que reviso su registro de archivos leídos.

## 4. Estructura del repositorio

```text
/
├── notebooks/kinetia_training.ipynb
├── ml/
│   ├── class_mapping.yaml
│   ├── dataset.py  train.py  evaluate.py  export.py
│   ├── fetch_depth_model.py  make_dummy_model.py
│   ├── splits/manifest.csv
│   └── requirements.txt
├── models/                      # binarios fuera de git; se publican en un Release
├── backend/
│   ├── app/
│   │   ├── main.py  config.py  schemas.py  errors.py
│   │   ├── routers/             # analyze, animate, jobs, health, metrics, gallery, config
│   │   └── services/            # inference, preprocess, validation, job_manager, encoder, depth
│   ├── animation_engine/        # puro: NumPy y OpenCV
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                    # Angular, Tailwind, GSAP; Dockerfile y nginx.conf
├── model-loader/                # Dockerfile y verify.sh del initContainer
├── k8s/
│   ├── base/
│   └── overlays/{local,cloud}/
├── tests/smoke/
├── scripts/                     # demo.ps1, generación de secretos, verificación
├── .github/workflows/{ci.yml,deploy.yml}
├── docs/
│   ├── CONTRATOS.md  PLAN.md  PROGRESO.md
│   ├── documento_tecnico.md  guion_demo.md  banco_preguntas.md
│   └── diagramas/
├── .env.example
├── docker-compose.yml
├── Makefile
└── README.md
```

## 5. Decisiones que necesito de ustedes

### D1. Dataset — recomiendo COCO-Stuff 164k en lugar de ADE20K

- **Por qué:** trae 10 categorías de animales etiquetadas por separado, incluidos perros y gatos, y personas bien anotadas. Las mascotas son clave en la demo, y en ADE20K (150 clases) "animal" es una sola categoría pequeña.
- **Cómo:** un subconjunto de unas 15 000 imágenes elegidas por script para reforzar las clases escasas. Se descargan una por una, sin bajar los 18 GB del zip completo.
- **Costo:** el reentrenamiento y la partición se hacen sobre datos nuevos en vez de los splits oficiales, lo cual se explica en la sustentación. Licencias: las imágenes de COCO tienen licencias variadas de Flickr, y se debe citar la fuente.

### D2. Clase fuego/luz — el riesgo técnico más grande

Hasta donde sé, ni COCO-Stuff ni ADE20K (150 clases) tienen una categoría "fuego". Solo hay lámparas y luces. Si se entrena solo con eso, **el modelo reconocerá lámparas, pero probablemente no una fogata**, y esta clase será la que más baje el mIoU. El Agente ML lo confirma con datos el 5 de noviembre.

- **Opción A (recomendada):** añadir un dataset público de segmentación de fuego, si se encuentra uno con licencia utilizable. En esas imágenes solo cuenta el fuego; el resto queda con `IGNORE_INDEX` para no enseñar errores. Si para el 5 de noviembre no hay uno viable, se pasa a B.
- **Opción B:** dejar la clase solo como luz (lámparas, luces, velas si las hay) y decir con claridad en la demo que el fuego no se detecta de forma fiable.

Rechazo un detector de fuego por color: sería una regla manual presentada como si fuera el modelo, y un jurado podría objetarlo.

### D3. Entrenamiento — ustedes deben ejecutarlo en Colab

No puedo operar Colab, y este portátil no tiene GPU compatible. El Agente ML entrega el notebook probado con una muestra mínima, y **alguien del equipo lo ejecuta en Colab con su cuenta** entre el 6 y el 10 de noviembre, y luego coloca los artefactos en `models/`. El notebook guarda en Drive cada época y puede reanudarse, porque Colab gratis puede cortar la sesión.

Necesito saber quién lo hará y si tienen Colab gratis o de pago.

### D4. GitHub — repositorio público (recomendado)

Necesito el usuario u organización y el nombre del repositorio. Lo recomiendo público porque Actions no gasta minutos y las imágenes de GHCR se descargan sin secreto. También necesito permiso para instalar `make`, `k3d`, `gh` y `ffmpeg` con winget.

### D5. Datos para la entrega

- Nombres de los integrantes, para repartir los temas de la sustentación.
- Fecha de entrega del documento técnico, si es anterior a la Expo.
- Día y hora de su presentación dentro de la Expo, si ya se conoce.

## 6. Riesgos

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| Sin datos de fuego (D2) | Alta | Medio | Compuerta del 5 nov; opción B |
| mIoU por debajo de 0.55 por clases escasas | Media | Alto | Muestreo reforzado y pérdida ponderada; reportar la cifra real con IoU por clase |
| Colab corta la sesión o agota la cuota | Media | Alto | Reanudación desde Drive; empezar el 6 nov deja margen; modelo base el 8 nov |
| **Wifi de la universidad bloquea el tráfico entre dispositivos**, y el QR deja de funcionar | Alta | Alto | Llevar router propio o usar el hotspot de un celular; probarlo antes de la Expo. Regla de firewall de Windows en `make demo` |
| RAM justa en el portátil | Media | Medio | Limitar WSL a unos 8 GB; cerrar otras aplicaciones; medir en la Fase 2 |
| Jobs lentos en CPU con muchos visitantes | Media | Medio | Cola máxima de 8 con mensaje claro; tamaño de trabajo limitado a 1280 px |
| Sin internet en la Expo | Media | Alto | `k3d image import`: las imágenes se cargan antes y no se descarga nada en vivo |
| Fotos de visitantes y menores en una pantalla pública | Media | Alto | Galería solo con consentimiento; moderación con token; borrado de resultados al terminar |
| Kubernetes falla en vivo | Baja | Alto | Plan C con docker-compose, ensayado |

### Orden de recorte si falta tiempo

Se recorta de arriba hacia abajo. Los requisitos de la rúbrica no se tocan.

1. Overlay `cloud` y despliegue en la nube (ya opcional).
2. Ingress `kinetia.local` (ya opcional).
3. Video con flujo óptico: se deja inferencia cada N frames más simple, y se conserva el flujo si ya funciona.
4. Efectos GSAP avanzados de la interfaz.
5. Galería: el QR se conserva y el muro pasa a una lista simple.

## 7. Diferencias con lo que pidieron

Todas están en CONTRATOS.md. Las señalo para que las aprueben o las rechacen.

| Cambio | Motivo |
|---|---|
| **Endpoints añadidos:** resultados, configuración y galería | Sin ellos no se pueden servir los videos, generar el QR ni mostrar el muro |
| **Prometheus:** solo el endpoint `/metrics`, sin servidor | Un servidor sería un segundo pod y rompería la restricción |
| **Imagen `model-loader`:** además de verificar, copia los modelos al PVC | Sin eso, un clúster limpio tiene el PVC vacío y el pod nunca arranca |
| **Máscara opcional:** se envía como trazos JSON y no como PNG | El pincel necesita enviar también la dirección de movimiento |
| **Despliegue de CI:** se hace en un clúster k3d temporal dentro del runner | GitHub Actions no puede alcanzar el clúster de un portátil. Es un despliegue real con smoke test real, pero no actualiza el portátil de la Expo |
| **`make demo` en Windows:** el Makefile invoca `scripts/demo.ps1` | Este portátil usa Windows y no trae `make` |
