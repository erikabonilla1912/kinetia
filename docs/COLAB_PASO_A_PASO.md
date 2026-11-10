# Entrenar el modelo en Google Colab, paso a paso

Guía para Erika. No hace falta experiencia previa en Colab.

**Qué vas a conseguir:** 5 archivos (`kinetia_seg.pt`, `kinetia_seg.onnx`, `metrics.json`, `manifest.json`, `hashes.sha256`), que luego se colocan en la carpeta `models/` del repositorio.

**Qué necesitas:**
- Una cuenta de Google con **al menos 6 GB libres en Drive** (más detalle en la sección "Espacio en Drive").
- El archivo `notebooks/kinetia_training.ipynb` del repositorio. Es el único archivo que hay que subir.
- Conexión estable para el inicio. Después, el trabajo corre en los servidores de Google.

**Cuánto se espera:** en el primer entrenamiento, del 4 de noviembre, cada época tardó unos 158 s medidos en la T4. Con un máximo de 100 épocas serían unas 4,4 horas, menos si el *early stopping* para antes. Esto es una estimación y probablemente necesitarás 2 o 3 sesiones. Google no publica los límites de la cuenta gratis. Cada época imprime el tiempo real medido, y esa cifra manda sobre esta estimación.

## Segundo entrenamiento: desde cero (9 de noviembre)

El profesor no permite *transfer learning*, así que el modelo del 4 de noviembre no se usa. Lo que cambia para ti:

- **Notebook:** sube el `notebooks/kinetia_training.ipynb` **nuevo** y no uses el anterior. La primera línea del notebook dice "entrenada desde cero".
- **Carpetas:** los checkpoints van a `kinetia/run_desde_cero` y los archivos finales a `kinetia/artifacts_desde_cero`. Las carpetas antiguas, `kinetia/run` y `kinetia/artifacts`, **no se borran ni se usan**.
- **Dataset:** se reutiliza el tar de `kinetia/dataset`, así que no vuelve a descargar las imágenes.
- **Épocas:** el máximo es 100, y el entrenamiento para solo tras 15 épocas sin mejora.
- **Comprobación previa:** antes de entrenar, la sección 6 imprime `Normalización ... Coinciden.`. Si dice otra cosa, se detiene: avisa al orquestador.
- **Antes de empezar, vacía la papelera de Drive** (sección 8). Con 100 épocas, la papelera puede crecer bastante.

---

## 1. Abrir Colab y subir el notebook

1. Entra en <https://colab.research.google.com> con tu cuenta de Google.
2. En la ventana que aparece, elige la pestaña **Subir** (*Upload*). Si no aparece: menú **Archivo → Subir cuaderno**.
3. Arrastra `kinetia_training.ipynb`. Se abre y queda guardado en tu Drive, en la carpeta **Colab Notebooks**.

La próxima vez, ábrelo desde esa carpeta o desde **Archivo → Abrir cuaderno**. No lo subas de nuevo: si el orquestador te da una versión nueva, sí, súbela y usa esa.

## 2. Activar la GPU T4

1. Menú **Entorno de ejecución → Cambiar tipo de entorno de ejecución**.
2. En **Acelerador de hardware**, elige **GPU T4** (*T4 GPU*).
3. Pulsa **Guardar**.

Sin GPU, el notebook se detiene con el mensaje **"NO HAY GPU"**, que explica esto mismo. No hace falta cambiar nada del código.

## 3. Ejecutar todo

Menú **Entorno de ejecución → Ejecutar todo**.

Si aparece un aviso de que el cuaderno no lo creó Google, pulsa **Ejecutar de todas formas**.

## 4. Autorizar Google Drive

A los pocos minutos sale una ventana que pide permiso para acceder a Google Drive:

1. Pulsa **Conectar con Google Drive**.
2. Elige tu cuenta y acepta todos los permisos.

Debes estar frente al computador en ese momento: el notebook espera tu respuesta. El notebook crea la carpeta **Mi unidad/kinetia**, con:

| Carpeta | Contenido |
|---|---|
| `kinetia/dataset/` | El dataset preparado, en un tar |
| `kinetia/run_desde_cero/` | Checkpoints e historial. **No la borres mientras entrenas** |
| `kinetia/artifacts_desde_cero/` | Los 5 archivos finales, al terminar |
| `kinetia/run/` y `kinetia/artifacts/` | Del modelo anterior, con *transfer learning*. No se usan; no las borres |

## 5. Qué ver mientras corre

**Primera sesión:** descarga y prepara el dataset. Verás líneas como:

```
500/15119 (120 img/s medidos; faltan ~2.0 min)
```

Luego guarda el tar en Drive. Las sesiones siguientes ya no descargan: solo dicen `Extrayendo ...`.

**Entrenamiento:** una línea por época, con el formato:

```
Epoca 3/100 | train loss ... mIoU ... | val loss ... mIoU ... | mejor ... (epoca 3) | sin mejora 0/15 | tiempo medido ... s | restante estimado ... | checkpoint guardado
```

- **`val ... mIoU`** debe tender a subir. Desde cero sube **más despacio** que en el primer entrenamiento, y las primeras épocas pueden dar cifras bajas: es normal. Puede bajar un poco de vez en cuando.
- **`sin mejora`** cuenta las épocas sin mejorar. Al llegar a 15, el entrenamiento para solo. Es normal: se conserva el mejor modelo.
- **`tiempo medido`** es real. **`restante estimado`** es una estimación y asume que llega hasta la época 100.

**Mantén la pestaña abierta y el computador sin suspender.** En la cuenta gratis, la sesión se puede cerrar si detecta inactividad. Interactúa de vez en cuando, por ejemplo abriendo una celda. No uses scripts para simular actividad: va contra las normas de Colab.

## 6. Si se corta la sesión

Es normal en Colab gratis y **no se pierde lo guardado**: cada época queda en Drive.

1. Si aparece **"Conectar"** o un aviso de desconexión, pulsa **Conectar** o **Volver a conectar**.
2. Comprueba que siga la GPU T4 (paso 2).
3. Pulsa **Entorno de ejecución → Ejecutar todo**.
4. Vuelve a autorizar Drive.

Reanuda solo, sin editar nada. Busca esta línea:

```
REANUDANDO: continuo en la epoca N de 100; mejor mIoU de val: ...; epocas sin mejora: ...
```

Solo se repite la época que estaba a medias.

**Si el notebook pide reiniciar la sesión** (mensaje que menciona `numpy` u `OpenCV`): **Entorno de ejecución → Reiniciar sesión**, y luego **Ejecutar todo** otra vez. No se pierde nada.

## 7. Si se agota la cuota de GPU

Colab dirá algo como **"No se puede conectar a un backend de GPU"** o que alcanzaste el límite de uso.

- **No uses CPU para seguir:** tardaría días.
- Espera y vuelve a intentarlo. Suele liberarse en unas horas o al día siguiente (estimación; Google no publica la cifra).
- Al volver, repite los pasos de la sección 6. Los checkpoints siguen en Drive.
- Plan B: pide a otra persona del equipo que lo reanude. Debe tener acceso a la misma carpeta `kinetia/run_desde_cero` en **su propio** "Mi unidad" (copiándola entera). **Avísale al orquestador antes**, para que quede registrado.

## 8. Espacio en Drive

El notebook imprime el espacio libre al arrancar y se detiene con un mensaje claro si el tar no cabe.

- **Tamaño del dataset en Drive:** ver "Tamaños medidos" al final de esta guía.
- **Checkpoints:** ocupan poco cada uno, pero al reemplazarse cada época **el archivo anterior puede ir a la papelera de Drive, y la papelera cuenta en tu cuota**. Si Drive se llena, ve a <https://drive.google.com/drive/trash> y pulsa **Vaciar papelera**. Hazlo con confianza: la carpeta `kinetia/run_desde_cero` no está en la papelera.

Si Drive se llena en plena escritura, el checkpoint anterior sigue intacto, porque cada archivo se escribe aparte y solo se reemplaza al terminar.

## 9. Descargar los archivos

Al final, el notebook crea **`kinetia_artifacts.zip`** e intenta descargarlo.

- Si tu navegador bloquea la descarga, permítela y vuelve a ejecutar la última celda.
- También queda una copia en **Mi unidad/kinetia/artifacts_desde_cero/**. Puedes descargarla desde Drive.

El notebook imprime el SHA-256 de cada modelo. **Cópialos en un mensaje para el orquestador.**

## 10. Dónde poner los archivos

Descomprime el zip y coloca los 5 archivos en la carpeta `models/` de la raíz del repositorio:

```text
proyecto-machine/
└── models/
    ├── kinetia_seg.pt
    ├── kinetia_seg.onnx
    ├── metrics.json
    ├── manifest.json
    └── hashes.sha256
```

**No los subas a git:** `models/` está excluida a propósito. Se publican en un GitHub Release.

Después, en el portátil, se comprueba todo:

```powershell
ml\.venv\Scripts\python ml\validate_artifacts.py models
ml\.venv\Scripts\python ml\fetch_test_set.py
ml\.venv\Scripts\python ml\evaluate.py --models-dir models --data-root data/test --verify models/metrics.json
```

La última línea debe ser `VERIFICACION CORRECTA`.

## 11. Señales de problema

| Lo que ves | Qué significa | Qué hacer |
|---|---|---|
| `NO HAY GPU` | Falta activar la GPU, o no hay cuota | Sección 2 o sección 7 |
| `val mIoU` sigue en menos de 0.1 después de 10 épocas | El modelo no está aprendiendo | Detén y avisa al orquestador con una captura |
| `no viene de un entrenamiento desde cero` | Se intentó reanudar el modelo anterior | Usas el notebook viejo: sube el nuevo. **No borres nada** |
| `La normalización de ml/dataset.py no coincide` | Los datos no son los esperados | Avisa al orquestador con una captura |
| `La perdida no es finita` | La pérdida se volvió inválida | Avisa al orquestador. No lo reinicies en bucle |
| `train loss` baja, `val loss` sube muchas épocas seguidas | Sobreajuste | Déjalo: el *early stopping* conserva el mejor modelo. El notebook lo analiza en la sección 9 |
| `CUDA out of memory` | No cabe en la memoria de la GPU | Avisa al orquestador; se reduce `BATCH_SIZE` en la celda de configuración. **Reanudar con otro tamaño es seguro**, pero debe quedar registrado |
| `No cabe el tar en Drive` | Falta espacio | Libera espacio y vacía la papelera |
| `imagenes fallaron` durante la descarga | Falló la red o el servidor de COCO | Vuelve a ejecutar: solo reintenta las que faltan |
| `No se pudo descargar ... calvin.inf.ed.ac.uk` o `bitbucket` | El servidor de los datos no responde | Espera un rato; si sigue, avisa al orquestador |
| `El checkpoint se entreno con otro manifiesto` | Cambió el notebook o la partición | **No borres nada.** Avisa al orquestador |
| Se repite `AVISO: no se pudo leer last.ckpt` | El último checkpoint quedó dañado; se usa el anterior | Es esperado tras un corte. Si los dos fallan, avisa |
| El tiempo por época supera los 20 minutos | Probablemente no se está usando bien la GPU | Comprueba que dice `GPU: ...T4` y avisa |
| Tras terminar, `metrics.json` da un mIoU menor de 0.55 | El modelo no alcanzó el objetivo | **No lo repitas por tu cuenta.** Avisa: hay un plan para esto, y la cifra real se reporta tal cual |

## 12. Empezar de cero (solo si el orquestador lo pide)

Borra **Mi unidad/kinetia/run_desde_cero** y vacía la papelera. No borres `kinetia/dataset`, salvo que te lo pidan.

---

## Tamaños medidos

Medido en el portátil del proyecto. El tar completo **no se generó** aquí, porque no hace falta el dataset entero en el portátil: su tamaño se extrapola de 10 956 imágenes ya preparadas. El notebook imprime el tamaño real al crearlo.

| Dato | Valor | Origen |
|---|---|---|
| Imagen + máscara de COCO-Stuff | 200.9 KB de media | Medido sobre 10 855 pares |
| Imagen + máscara de BoWFire | 187.7 KB de media | Medido sobre 101 pares |
| Tar con las 15 119 imágenes | **unos 3.1 GB** | Extrapolado |
| `last.ckpt` y `last.prev.ckpt` | 57 MB cada uno | Medido |
| `kinetia_seg.pt` | 27 MB | Medido |
| Papelera tras 100 épocas, en el peor caso | unos 5,7 GB | Estimación: 57 MB por época, si Drive conserva cada versión reemplazada. Vacíala antes de empezar y a mitad del entrenamiento |
| **Total en Drive, en el peor caso** | **unos 9 GB** de los 15 GB gratuitos | Estimación: tar, papelera y carpetas del primer entrenamiento |
