# KINETIA — Registro de avance

**Última actualización:** 2026-11-09 · **Fase actual:** 2, en curso: reentrenamiento desde cero e integración (plan aprobado por el equipo el 2026-11-02)

Estados: `pendiente`, `en curso`, `en revisión`, `aprobada`, `bloqueada`. Una tarea solo pasa a `aprobada` con evidencia que el orquestador haya verificado.

## Tareas

| # | Tarea | Subagente | Estado | Fecha | Ciclos | Evidencia |
|---|---|---|---|---|---|---|
| T0 | Plan, estructura y CONTRATOS.md | Orquestador | aprobada | 2026-11-02 | 0 | Aprobación del equipo; `docs/PLAN.md`, `docs/CONTRATOS.md` v1.1.0 |
| T1 | Dataset, remapeo, partición, búsqueda de fuego | ML | aprobada | 2026-11-03 | 0 | Orquestador: 15 119 imágenes, sin solapes entre particiones, hash `fb8a7cc4…` comprobado; 53 pruebas pasadas |
| T2 | Notebook, evaluación, exportación; entrenamiento lo ejecuta Erika Bonilla en Colab | ML + Erika | **en curso: reentrenar desde cero** (decisión 14) | 2026-11-09 | 2 | **Modelo del 4 nov descartado:** usaba *transfer learning* (encoder con pesos de ImageNet) y el profesor no lo permite. Sus cifras (mIoU 0,7495 en prueba, verificado) **no se usan** como resultado del proyecto. 9 nov, orquestador: código sin ninguna vía para cargar pesos preentrenados; normalización calculada con nuestro train; MiDaS retirado; pruebas: ML 61, backend 222 + motor 130, frontend 161, todas pasadas |
| T3 | API, validaciones, inferencia, jobs, modelo dummy | Backend | aprobada (Fase 1; se repite con el modelo real en la Fase 2) | 2026-11-03 | 1 | Orquestador: pytest 220 pasadas, 0 omitidas, con FFmpeg real; flujo real con uvicorn hasta MP4/GIF válidos; `/analyze` 1080p 999 ms **con modelo dummy** |
| T4 | Efectos, partículas, moods, video | Motor | aprobada (Fase 1; falta verlo con el modelo real): Erika aprobó en movimiento el lago y el caballo el 2026-11-04. Motor 130 y backend 220 pruebas pasadas, verificadas por el orquestador | 2026-11-04 | 3 | Orquestador: motor 108 pruebas y backend 220 pasadas; por API real con CPU al 74 % y modelo dummy, 48 frames en 13,1 s de render (antes 29,5 s), es decir, unos 26 s por 96 frames, cifra extrapolada; revisión de hojas de contacto. Máscaras de prueba, no del modelo real |
| T5 | Interfaz, pincel, comparador, Rayos X, QR, galería | Frontend | aprobada (Fase 1; falta celular real y el modelo real) | 2026-11-03 | 0 | Orquestador: build sin errores, 160 pruebas, `npm audit` 0 vulnerabilidades; capturas revisadas. Del agente, sin repetir: 30/30 contra la API real. Render por API real (lago, 4 s, modelo dummy): 15–17 s sin pincel, 23 s con `add`, 4 s con `only` |
| T7 | Docker, Kubernetes, CI/CD, monitoreo, `make demo` | DevOps | en revisión: desplegado **con el modelo real**; falta Release para el CI, recursos medidos (`# PROVISIONAL`), celular y ensayo sin Wi-Fi | 2026-11-04 | 0 | Orquestador, clúster limpio con modelo real: `make demo` 216 s, 1 pod 2/2 Running, `model_version` seg-1.0.0, smoke OK. `model-loader` reemplaza modelos antiguos del volumen: 13 casos pasados en Alpine 3.20; aún sin probar dentro del clúster. Antes, con dummy: pod recreado en 10 s con resultado intacto; hash incorrecto da `Init:CrashLoopBackOff`; subida de 16,6 MB da 413 JSON; kubeconform 7 válidos |
| T6 | Pruebas de API, modelo, rendimiento y frontend | QA | pendiente | | 0 | |
| T8 | Documento técnico, diagramas, README, guion, preguntas | Documentación | pendiente | | 0 | |
| T9 | Verificación cruzada e integración final en clúster limpio | Orquestador | pendiente | | 0 | |

## Decisiones técnicas

Las decisiones marcadas "propuesta" quedaron aprobadas por el equipo el 2026-11-02, junto con el plan.

| # | Decisión | Justificación | Estado |
|---|---|---|---|
| 1 | U-Net con encoder MobileNetV3 Large, entrada de 320×320 | La inferencia se ejecuta en la CPU de un portátil de bajo consumo (i5-1235U), sin GPU dedicada. Desde la decisión 14, con pesos aleatorios | aprobada; modificada por la 14 |
| 2 | MiDaS v2.1 small para la profundidad | Modelo ligero con versión ONNX; entrenar profundidad no es viable en 3 semanas y no es el objetivo del curso | **retirada** el 2026-11-09 (decisión 15) |
| 3 | COCO-Stuff 164k como dataset | Personas y animales mejor anotados que en ADE20K (ver D1) | propuesta |
| 4 | Noche y bosque como reglas, no como clases | Se deducen de la luminancia del cielo y del porcentaje de vegetación, sin añadir clases con pocos datos | propuesta |
| 5 | k3d en lugar de minikube | Corre sobre Docker Desktop; trae `metrics-server` y `local-path`, necesarios para `kubectl top` y el PVC | propuesta |
| 6 | `model-loader` copia los modelos al PVC y luego verifica contra el ConfigMap | Permite arrancar en un clúster limpio; el hash esperado no depende de la imagen que se verifica | propuesta |
| 7 | Sin servidor Prometheus | Sería un segundo pod; se expone solo `/metrics` | propuesta |
| 8 | Trazos JSON para el pincel | Llevan la zona y la dirección en un solo campo | propuesta |
| 9 | Despliegue de CI en un clúster k3d temporal | El runner no alcanza el portátil; así el smoke test es real | propuesta |
| 10 | Nginx en el puerto 80 sin root mediante un sysctl de pod | Cumple a la vez "puerto 80" y `runAsNonRoot` | propuesta |
| 11 | BoWFire como fuente de fuego (119 imágenes), con el resto de cada imagen ignorado | Es el único candidato a nivel de suelo con máscaras manuales y descarga libre. Su README indica Creative Commons de Flickr y pide citar; no especifica la variante por imagen, así que se usa solo en este proyecto académico, sin redistribuir las fotos. No citarlo como MIT | aprobada 2026-11-03 |
| 13 | **Sin parallax 2.5D (cambio de alcance):** sujeto y fondo quietos, estilo cinemagraph; la profundidad la dan partículas delante y detrás del sujeto | Erika rechazó dos versiones con parallax al verlas en movimiento ("patas de goma") y eligió esta opción el 2026-11-04. El requisito original lo pedía: se explica como decisión de calidad. Animar al propio sujeto (por ejemplo, un galope) exigiría un modelo generativo y GPU | aprobada 2026-11-04 |
| 14 | **Segmentación entrenada desde cero, sin *transfer learning*:** pesos aleatorios, una sola tasa de aprendizaje (1e-3), sin congelar el encoder, hasta 100 épocas con parada tras 15 sin mejora. Normalización con la media y desviación de nuestras 10 583 imágenes de train (0,4657/0,4461/0,4079 y 0,2706/0,2678/0,2831), no con las de ImageNet. Marca `init=desde_cero` que exigen `train.py`, `export.py`, `model-loader` (vía hash), `update-model-hashes.ps1` y la API | El profesor no permite el modelo con *transfer learning* (comunicado por Erika el 2026-11-09). Se conserva la arquitectura para no cambiar la API ni el despliegue. Una prueba registra las conexiones de red durante el entrenamiento y exige cero | aprobada 2026-11-09 |
| 15 | MiDaS retirado del proyecto | Era preentrenado y el motor ya no lo usaba (decisión 13). Los campos `depth` de la API quedan en `null` por compatibilidad | aprobada 2026-11-09 |
| 12 | Grupos manuales de escena en BoWFire | 33 imágenes son del mismo patio de entrenamiento: una misma escena no puede quedar en dos particiones, o la prueba se contaminaría | aprobada 2026-11-03 |

## Riesgos abiertos

La tabla completa, con su mitigación, está en [PLAN.md](PLAN.md#6-riesgos).

| Riesgo | Estado | Próxima revisión |
|---|---|---|
| Fuego con pocos datos (119 imágenes; casi sin fogatas, sin velas ni chimeneas vistas) | Mitigado con BoWFire; el riesgo restante se declara en la demo | 11 nov |
| **mIoU por debajo de 0.55 con el modelo desde cero** | **Reabierto:** sin pesos preentrenados el resultado será menor que 0,7495; no se sabe cuánto hasta entrenar | Al terminar el entrenamiento |
| Cortes o cuota de Colab con 100 épocas (unas 4,4 h estimadas a 158 s por época) | Reabierto; el notebook reanuda desde Drive | Durante el entrenamiento |
| Espacio en Drive: la papelera puede crecer unos 5,7 GB con 100 épocas | Abierto; vaciar antes y a mitad | Durante el entrenamiento |
| Clase fuego_luz débil: IoU 0,376, sobre todo por las luces de COCO-Stuff (recall 0,3753) | Abierto; se declara en la demo y en el documento | Fase 2 |
| Sobreajuste moderado: la pérdida de validación sube de 0,56 (época 9) a 0,62 mientras el mIoU de validación se mantiene | Aceptado: se guarda el modelo por mIoU de validación; se explica en la sustentación | Documento |
| El wifi de la Expo bloquea el QR | Abierto; requiere router propio | 18 nov |
| RAM justa en el portátil | Abierto | Fase 2 |
| Límite de uso de la cuenta: cuatro agentes en paralelo lo agotaron en menos de una hora y se perdió cerca de un día | Abierto; se trabaja por tandas | Cada fase |
| **Errores del modelo visibles en el video:** en la foto del caballo, montañas clasificadas como agua y cielo producen rayas fantasma junto a las patas y detrás de la cabeza. El lago no las muestra | Abierto, confirmado en fotogramas; pendiente de corrección en el motor y de revisión de Erika | Fase 2 |
| Modelo poco fiable fuera de fotos reales: en un lago dibujado clasificó casi toda el agua como fondo; un fondo oscuro desenfocado salió como vegetación | Abierto; se declara y se usan fotos en la demo | Fase 2 |
| Render muy sensible a la carga del portátil: lago 1280×720, 4 s, en el pod: 66 s y 78 s con el equipo al 100 % de CPU y en batería; 11,9 s con el equipo más libre. Caballo: 85,9 s y 17,9 s | Abierto; falta medir enchufado y con el equipo libre. En la Expo: cargador conectado y otros programas cerrados | Fase 2 |
| **El portátil entra en espera por inactividad y congela los jobs:** un render de unos 20 s tardó 1618 s por una espera de 21:10 a 21:37 del 3 nov, comprobada en el registro de Windows | Abierto; `make demo` debe impedir la espera mientras corra; también afecta el entrenamiento en Colab | Fase 2 |
| Texto de consentimiento para menores sin validar | Abierto; lo revisa el equipo | Antes de la Expo |

## Resúmenes de fase

Se completan al cerrar cada fase.
