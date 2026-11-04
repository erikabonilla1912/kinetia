# Plan C: la aplicación con docker compose, sin Kubernetes. Mismas imágenes y sin descargas si ya existen.
# Uso: planc.ps1 [up|down]. Con -ComposeArgs se añaden archivos extra (por ejemplo, un override de pruebas).
param(
    [ValidateSet('up', 'down')][string]$Action = 'up',
    [string]$ModelsSrc = $(if ($env:MODELS_SRC) { $env:MODELS_SRC } else { 'models' }),
    [string[]]$ComposeFiles = @()
)
$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'common.ps1')
Set-Location $script:RepoRoot

$fileArgs = @('-f', 'docker-compose.yml')
foreach ($file in $ComposeFiles) { $fileArgs += @('-f', $file) }

if ($Action -eq 'down') {
    Stop-KeepAwake
    docker compose @fileArgs down -v
    exit $LASTEXITCODE
}

if (Test-ClusterExists) {
    $running = docker inspect -f '{{.State.Running}}' $script:NodeContainer
    if ($running -eq 'true') { Write-Host 'ERROR: k3d ocupa el puerto 8080. Ejecuta primero: make down' -ForegroundColor Red; exit 1 }
}
try { $tags = Build-AllImages -ModelsSrc $ModelsSrc } catch { Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red; exit 1 }

$lanIp = Get-LanIp
$env:PUBLIC_BASE_URL = if ($env:PUBLIC_BASE_URL) { $env:PUBLIC_BASE_URL } elseif ($lanIp) { "http://${lanIp}:8080" } else { 'http://localhost:8080' }
docker compose @fileArgs up -d --no-build --wait --wait-timeout 240
if ($LASTEXITCODE -ne 0) {
    docker compose @fileArgs ps -a
    docker compose @fileArgs logs --tail 20 model-loader
    Write-Host 'ERROR: Plan C no arrancó. Revisa la salida anterior.' -ForegroundColor Red
    exit 1
}
Start-KeepAwake 'kinetia-api-1'
Show-FirewallAdvice
Write-Host "Plan C activo: $($env:PUBLIC_BASE_URL)  |  terminar: make planc-down" -ForegroundColor Green
exit 0
