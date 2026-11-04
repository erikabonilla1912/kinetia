# KINETIA. En Windows, cada objetivo invoca un script de PowerShell (docs/PLAN.md, sección 7).
# Variables opcionales: make demo MODELS_SRC=<carpeta> OVERLAY=<ruta de overlay>
PS := powershell -NoProfile -ExecutionPolicy Bypass -File
KC := kubectl --context k3d-kinetia -n kinetia

export MODELS_SRC
export OVERLAY
export PUBLIC_BASE_URL

.PHONY: demo down status logs smoke prefetch planc planc-down

demo:
	$(PS) scripts/demo.ps1

down:
	$(PS) scripts/down.ps1

status:
	$(PS) scripts/status.ps1

logs:
	$(KC) logs deploy/kinetia --all-containers --prefix --tail=100

smoke:
	$(PS) tests/smoke/smoke.ps1

prefetch:
	$(PS) scripts/prefetch-images.ps1

planc:
	$(PS) scripts/planc.ps1 -Action up

planc-down:
	$(PS) scripts/planc.ps1 -Action down
