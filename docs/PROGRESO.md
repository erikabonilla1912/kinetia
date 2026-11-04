# KINETIA — Registro de avance

**Última actualización:** 2026-11-02 · **Fase actual:** 1, en curso (plan aprobado por el equipo el 2026-11-02)

Estados: `pendiente`, `en curso`, `en revisión`, `aprobada`, `bloqueada`. Una tarea solo pasa a `aprobada` con evidencia que el orquestador haya verificado.

## Tareas

| # | Tarea | Subagente | Estado | Fecha | Ciclos | Evidencia |
|---|---|---|---|---|---|---|
| T0 | Plan, estructura y CONTRATOS.md | Orquestador | aprobada | 2026-11-02 | 0 | Aprobación del equipo; `docs/PLAN.md`, `docs/CONTRATOS.md` v1.1.0 |
| T1 | Dataset, remapeo, partición, búsqueda de fuego | ML | en curso | 2026-11-02 | 0 | |
| T2 | Notebook, evaluación, exportación; entrenamiento lo ejecuta Erika Bonilla en Colab | ML + Erika | en curso | 2026-11-02 | 0 | |
| T3 | API, validaciones, inferencia, jobs, modelo dummy | Backend | aprobada (Fase 1; se repite con el modelo real en la Fase 2) | 2026-11-03 | 1 | Orquestador: pytest 220 pasadas, 0 omitidas, con FFmpeg real; flujo real con uvicorn hasta MP4/GIF válidos; `/analyze` 1080p 999 ms **con modelo dummy** |
| T4 | Efectos, parallax, partículas, moods, video | Motor | en curso | 2026-11-02 | 0 | |
| T5 | Interfaz, pincel, comparador, Rayos X, QR, galería | Frontend | en curso | 2026-11-02 | 0 | |
| T6 | Pruebas de API, modelo, rendimiento y frontend | QA | pendiente | | 0 | |
| T7 | Docker, Kubernetes, CI/CD, monitoreo, `make demo` | DevOps | pendiente | | 0 | |
| T8 | Documento técnico, diagramas, README, guion, preguntas | Documentación | pendiente | | 0 | |
| T9 | Verificación cruzada e integración final en clúster limpio | Orquestador | pendiente | | 0 | |

## Decisiones técnicas

Las decisiones marcadas "propuesta" quedaron aprobadas por el equipo el 2026-11-02, junto con el plan.

| # | Decisión | Justificación | Estado |
|---|---|---|---|
| 1 | U-Net con encoder MobileNetV3 Large, entrada de 320×320 | La inferencia se ejecuta en la CPU de un portátil de bajo consumo (i5-1235U), sin GPU dedicada | propuesta |
| 2 | MiDaS v2.1 small para la profundidad | Modelo ligero con versión ONNX; entrenar profundidad no es viable en 3 semanas y no es el objetivo del curso | propuesta |
| 3 | COCO-Stuff 164k como dataset | Personas y animales mejor anotados que en ADE20K (ver D1) | propuesta |
| 4 | Noche y bosque como reglas, no como clases | Se deducen de la luminancia del cielo y del porcentaje de vegetación, sin añadir clases con pocos datos | propuesta |
| 5 | k3d en lugar de minikube | Corre sobre Docker Desktop; trae `metrics-server` y `local-path`, necesarios para `kubectl top` y el PVC | propuesta |
| 6 | `model-loader` copia los modelos al PVC y luego verifica contra el ConfigMap | Permite arrancar en un clúster limpio; el hash esperado no depende de la imagen que se verifica | propuesta |
| 7 | Sin servidor Prometheus | Sería un segundo pod; se expone solo `/metrics` | propuesta |
| 8 | Trazos JSON para el pincel | Llevan la zona y la dirección en un solo campo | propuesta |
| 9 | Despliegue de CI en un clúster k3d temporal | El runner no alcanza el portátil; así el smoke test es real | propuesta |
| 10 | Nginx en el puerto 80 sin root mediante un sysctl de pod | Cumple a la vez "puerto 80" y `runAsNonRoot` | propuesta |

## Riesgos abiertos

La tabla completa, con su mitigación, está en [PLAN.md](PLAN.md#6-riesgos).

| Riesgo | Estado | Próxima revisión |
|---|---|---|
| Fuego sin datos etiquetados | Abierto; decisión D2 | 5 nov |
| mIoU por debajo de 0.55 | Abierto | 11 nov |
| Cortes o cuota de Colab | Abierto | 6 nov |
| El wifi de la Expo bloquea el QR | Abierto; requiere router propio | 18 nov |
| RAM justa en el portátil | Abierto | Fase 2 |

## Resúmenes de fase

Se completan al cerrar cada fase.
