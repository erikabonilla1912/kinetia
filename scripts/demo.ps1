# make demo: construye (si hace falta), crea el clúster k3d y deja KINETIA en un solo pod.
# Idempotente. Sin descargas si ya se ejecutó scripts/prefetch-images.ps1 y las imágenes existen.
param(
    [string]$ModelsSrc = $(if ($env:MODELS_SRC) { $env:MODELS_SRC } else { 'models' }),
    [string]$Overlay = $(if ($env:OVERLAY) { $env:OVERLAY } else { 'k8s/overlays/local' }),
    [string]$PublicBaseUrl = $env:PUBLIC_BASE_URL,
    [string]$ServersMemory = '5g',
    [int]$WaitSeconds = 300
)
$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'common.ps1')

function Show-Help {
    Write-Host ''
    Write-Host 'Si falla:' -ForegroundColor Yellow
    Write-Host '  1. ¿Docker Desktop está encendido?  docker info'
    Write-Host '  2. Estado y registros:              make status ; make logs'
    Write-Host '  3. Empezar de cero:                 make down ; make demo'
    Write-Host '  4. Plan C sin Kubernetes:           make planc'
    Write-Host '  5. Sin Wi-Fi: comparte datos del celular o usa http://localhost:8080 en este portátil.'
}

function Stop-Demo([string]$Message) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    Show-Help
    exit 1
}

docker info --format '{{.ServerVersion}}' 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Stop-Demo 'Docker no responde. Abre Docker Desktop y vuelve a intentarlo.' }

# Modelo: con el overlay por defecto, los hashes versionados deben coincidir antes de arrancar nada.
$isDefaultOverlay = ($Overlay.Replace('\', '/').TrimEnd('/') -eq 'k8s/overlays/local')
if ($isDefaultOverlay) {
    $hashes = Get-Content (Join-Path $script:RepoRoot 'k8s\base\model-hashes.env') -Raw
    $seg = Join-Path $script:RepoRoot "$ModelsSrc\kinetia_seg.onnx"
    if (-not (Test-Path $seg)) {
        Stop-Demo "Falta $ModelsSrc/kinetia_seg.onnx. Coloca el modelo real en models/ y ejecuta: powershell -ExecutionPolicy Bypass -File scripts/update-model-hashes.ps1"
    }
    if ($hashes -notmatch (Get-FileHash $seg -Algorithm SHA256).Hash.ToLower()) {
        Stop-Demo 'k8s/base/model-hashes.env no coincide con el modelo. Ejecuta: powershell -ExecutionPolicy Bypass -File scripts/update-model-hashes.ps1'
    }
}

# Caché de imágenes de k3s.
$airgap = docker volume ls -q --filter "name=^$($script:AirgapVolume)$"
if (-not $airgap -or -not (Test-DockerImage $script:K3sImage) -or -not (Test-DockerImage $script:K3dToolsImage) -or -not (Test-DockerImage $script:K3dProxyImage)) {
    Write-Host '==> Falta la caché de imágenes de k3s; se ejecuta el paso previo (requiere internet).' -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot 'prefetch-images.ps1')
    if ($LASTEXITCODE -ne 0) { Stop-Demo 'No se pudo preparar la caché. Conéctate a internet y ejecuta: make prefetch' }
}

# Imágenes de la aplicación.
try {
    $tags = Build-AllImages -ModelsSrc $ModelsSrc
} catch {
    Stop-Demo $_.Exception.Message
}

# Clúster.
if (-not (Test-ClusterExists)) {
    Write-Host '==> Creando el clúster k3d'
    k3d cluster create $script:ClusterName --image $script:K3sImage --servers 1 --agents 0 `
        --api-port 127.0.0.1:6445 -p '8080:30080@server:0' --servers-memory $ServersMemory `
        -v "$($script:AirgapVolume):/var/lib/rancher/k3s/agent/images@server:0" `
        --k3s-arg '--disable-cloud-controller@server:0' `
        --kubeconfig-update-default --kubeconfig-switch-context --wait --timeout 240s
    if ($LASTEXITCODE -ne 0) { Stop-Demo 'k3d no pudo crear el clúster. ¿El puerto 8080 está ocupado? Revisa: netstat -ano | findstr :8080' }
} else {
    $running = docker inspect -f '{{.State.Running}}' $script:NodeContainer
    if ($running -ne 'true') {
        Write-Host '==> Iniciando el clúster existente'
        k3d cluster start $script:ClusterName --wait --timeout 180s
        if ($LASTEXITCODE -ne 0) { Stop-Demo 'k3d no pudo iniciar el clúster.' }
    }
    k3d kubeconfig merge $script:ClusterName --kubeconfig-merge-default --kubeconfig-switch-context | Out-Null
}

# Importa solo lo que falte en el nodo.
$nodeImages = docker exec $script:NodeContainer crictl images 2>$null | Out-String
$missing = @()
foreach ($name in $tags.Keys) {
    $tag = $tags[$name]
    if ($nodeImages -notmatch "$name\s+$tag") { $missing += "$script:Registry/${name}:$tag" }
}
if ($missing.Count -gt 0) {
    Write-Host "==> Importando al clúster: $($missing -join ', ')"
    k3d image import @missing -c $script:ClusterName
    if ($LASTEXITCODE -ne 0) { Stop-Demo 'k3d image import falló.' }
    # k3d puede informar éxito aunque la importación haya fallado: se comprueba en el nodo.
    $nodeImages = docker exec $script:NodeContainer crictl images 2>$null | Out-String
    foreach ($name in $tags.Keys) {
        if ($nodeImages -notmatch "$name\s+$($tags[$name])") { Stop-Demo "La imagen $name no quedó en el nodo tras importarla." }
    }
}

# URL pública.
$lanIp = Get-LanIp
if (-not $PublicBaseUrl) {
    if ($lanIp) { $PublicBaseUrl = "http://${lanIp}:8080" } else {
        $PublicBaseUrl = 'http://localhost:8080'
        Write-Host 'AVISO: no se detectó red local; el QR usará localhost y no servirá en celulares.' -ForegroundColor Yellow
    }
}

# Namespace y Secret primero; después todo el manifiesto de una vez, ya con la URL correcta.
"apiVersion: v1`nkind: Namespace`nmetadata:`n  name: kinetia`n" | kubectl --context $script:KubeContext apply -f - | Out-Null
$secretRev = & (Join-Path $PSScriptRoot 'create-secret.ps1') -KubeContext $script:KubeContext | Select-Object -Last 1
if (-not $secretRev) { Stop-Demo 'No se pudo crear kinetia-secret.' }

$overlayPath = if ([IO.Path]::IsPathRooted($Overlay)) { $Overlay } else { Join-Path $script:RepoRoot $Overlay }
$runtimeDir = Join-Path $script:StateDir 'runtime'
New-Item -ItemType Directory -Force $runtimeDir | Out-Null
$rendered = kubectl kustomize $overlayPath | Out-String
if ($LASTEXITCODE -ne 0) { Stop-Demo "kubectl kustomize falló en $overlayPath" }
Write-Utf8NoBom (Join-Path $runtimeDir 'rendered.yaml') $rendered
$imageLines = ($tags.Keys | ForEach-Object { "  - name: $script:Registry/$_`n    newTag: $($tags[$_])" }) -join "`n"
$runtimeKustomization = @"
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - rendered.yaml
images:
$imageLines
generatorOptions:
  disableNameSuffixHash: true
configMapGenerator:
  - name: kinetia-config
    namespace: kinetia
    behavior: merge
    literals:
      - PUBLIC_BASE_URL=$PublicBaseUrl
patches:
  - target:
      kind: Deployment
      name: kinetia
    patch: |-
      - op: add
        path: /spec/template/metadata/annotations
        value:
          kinetia/public-base-url: "$PublicBaseUrl"
          kinetia/secret-rev: "$secretRev"
"@
Write-Utf8NoBom (Join-Path $runtimeDir 'kustomization.yaml') $runtimeKustomization
kubectl --context $script:KubeContext apply -k $runtimeDir | Out-Host
if ($LASTEXITCODE -ne 0) { Stop-Demo 'kubectl apply falló.' }

# Espera, detectando pronto un fallo del init (hash incorrecto).
Write-Host '==> Esperando a que el pod esté Ready'
$deadline = (Get-Date).AddSeconds($WaitSeconds)
$ready = $false
while ((Get-Date) -lt $deadline) {
    $table = kubectl --context $script:KubeContext -n kinetia get pods --no-headers 2>$null | Out-String
    if ($table -match 'Init:Error|Init:CrashLoopBackOff') {
        kubectl --context $script:KubeContext -n kinetia logs deploy/kinetia -c model-loader --tail=10 | Out-Host
        Stop-Demo 'model-loader falló: el hash no coincide. Lee el mensaje de arriba.'
    }
    if ($table -match 'ImagePullBackOff|ErrImagePull|ErrImageNeverPull') { Stop-Demo "Imagen no disponible en el clúster:`n$table" }
    kubectl --context $script:KubeContext -n kinetia rollout status deploy/kinetia --timeout=5s *>$null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 5
}
if (-not $ready) {
    kubectl --context $script:KubeContext -n kinetia get pods | Out-Host
    Stop-Demo "El pod no quedó Ready en $WaitSeconds s."
}

$healthy = $false
foreach ($attempt in 1..30) {
    try {
        $health = Invoke-RestMethod -Uri 'http://localhost:8080/api/v1/health' -TimeoutSec 5
        $healthy = $true
        break
    } catch { Start-Sleep -Seconds 2 }
}
if (-not $healthy) { Stop-Demo 'http://localhost:8080/api/v1/health no responde.' }

Start-KeepAwake $script:NodeContainer
Show-FirewallAdvice
kubectl --context $script:KubeContext -n kinetia get pods | Out-Host
Write-Host ''
Write-Host "KINETIA lista. Modelo: $($health.model_version)" -ForegroundColor Green
Write-Host "  Abre:            $PublicBaseUrl"
Write-Host '  En este equipo:  http://localhost:8080'
Write-Host '  Comprobar:       make smoke | make status | make logs'
Write-Host '  Terminar:        make down  (restaura la espera del portátil)'
Write-Host '  Enchufa el cargador y no cierres la tapa: el bloqueo no cubre el cierre de tapa ni el botón de encendido.'
if ($health.model_version -eq 'dummy-dev') {
    Write-Host '  AVISO: modelo DUMMY con pesos aleatorios. No sirve para la demo ni para medir rendimiento.' -ForegroundColor Yellow
}
Show-Help
exit 0
