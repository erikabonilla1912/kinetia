# make status: estado del pod, memoria y bloqueo de espera.
$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'common.ps1')
if (-not (Test-ClusterExists)) { Write-Host 'No hay clúster. Ejecuta: make demo'; exit 0 }
kubectl --context $script:KubeContext -n kinetia get pods,svc,pvc,ingress
Write-Host "`n--- kubectl top (tarda ~1 min en estar disponible tras crear el clúster)"
kubectl --context $script:KubeContext -n kinetia top pod --containers
Write-Host "`n--- Memoria del nodo k3d"
docker stats --no-stream --format '{{.Name}} {{.MemUsage}} {{.CPUPerc}}' $script:NodeContainer
Write-Host "`n--- Bloqueo de espera"
$statusFile = Join-Path $script:StateDir 'keep-awake.status'
if (Test-Path $statusFile) { Get-Content $statusFile } else { Write-Host 'Sin estado.' }
Write-Host "`n--- Health"
try { (Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:8080/api/v1/health' -TimeoutSec 5).Content } catch { Write-Host "No responde: $($_.Exception.Message)" }
exit 0
