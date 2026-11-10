# Constantes y utilidades de los scripts de KINETIA. Se carga con dot-sourcing.
$script:RepoRoot = Split-Path -Parent $PSScriptRoot
$script:ClusterName = 'kinetia'
$script:KubeContext = 'k3d-kinetia'
$script:NodeContainer = 'k3d-kinetia-server-0'
$script:K3sVersion = 'v1.35.5-k3s1'
$script:K3sImage = "rancher/k3s:$script:K3sVersion"
$script:K3dToolsImage = 'ghcr.io/k3d-io/k3d-tools:5.9.0'
$script:K3dProxyImage = 'ghcr.io/k3d-io/k3d-proxy:5.9.0'
$script:AirgapVolume = 'kinetia-k3s-airgap'
$script:Registry = 'ghcr.io/erikabonilla1912'
$script:BaseImages = @('python:3.11-slim-bookworm', 'node:22-alpine', 'nginx:1.29-alpine', 'alpine:3.20')
$script:StateDir = Join-Path $env:TEMP 'kinetia'

try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }
$toolsBin = Join-Path $env:LOCALAPPDATA 'kinetia-tools\bin'
if (Test-Path $toolsBin) { $env:PATH = "$toolsBin;$env:PATH" }
if (-not (Test-Path $script:StateDir)) { New-Item -ItemType Directory -Force $script:StateDir | Out-Null }

function Assert-Exit([string]$What) {
    if ($LASTEXITCODE -ne 0) { throw "$What falló (código $LASTEXITCODE)." }
}

function Write-Utf8NoBom([string]$Path, [string]$Text) {
    [IO.File]::WriteAllText($Path, $Text.Replace("`r`n", "`n"), (New-Object Text.UTF8Encoding($false)))
}

function Test-DockerImage([string]$Image) {
    $found = docker images -q $Image
    return [bool]$found
}

function Test-ClusterExists {
    $names = docker ps -a --filter "name=^/$($script:NodeContainer)$" --format '{{.Names}}'
    return [bool]$names
}

# Hash corto del contenido de las entradas de una imagen: si no cambia, no hace falta reconstruir.
function Get-InputHash([string[]]$Paths) {
    $sha = [Security.Cryptography.SHA256]::Create()
    $lines = New-Object Collections.Generic.List[string]
    foreach ($path in $Paths) {
        $full = Join-Path $script:RepoRoot $path
        if (-not (Test-Path $full)) { throw "No existe $full" }
        $files = if (Test-Path $full -PathType Container) {
            Get-ChildItem $full -Recurse -File | Where-Object { $_.FullName -notmatch '[\\/](__pycache__|tests)[\\/]' }
        } else { Get-Item $full }
        foreach ($file in $files) {
            $relative = $file.FullName.Substring($script:RepoRoot.Length)
            $lines.Add("$relative $((Get-FileHash $file.FullName -Algorithm SHA256).Hash)")
        }
    }
    $lines.Sort([StringComparer]::Ordinal)
    $bytes = [Text.Encoding]::UTF8.GetBytes(($lines -join "`n"))
    return (-join ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString('x2') })).Substring(0, 12)
}

# Construye una imagen solo si no existe la etiqueta con ese hash. Devuelve la etiqueta.
function Build-Image {
    param([string]$Name, [string]$Context, [string]$Dockerfile, [string[]]$Inputs, [string[]]$BuildArgs = @(), [switch]$Force)
    $tag = 'src-' + (Get-InputHash $Inputs)
    $image = "$script:Registry/${Name}:$tag"
    if ($Force -or -not (Test-DockerImage $image)) {
        Write-Host "==> Construyendo $image"
        $arguments = @('build', '-t', $image, '-t', "$script:Registry/${Name}:local", '-f', $Dockerfile) + $BuildArgs + @($Context)
        Push-Location $script:RepoRoot
        try { & docker @arguments | Out-Host } finally { Pop-Location }
        Assert-Exit "docker build de $Name"
    } else {
        Write-Host "==> $image ya existe; no se reconstruye"
        docker tag $image "$script:Registry/${Name}:local"
    }
    return $tag
}

# Construye las tres imágenes. Devuelve un hashtable nombre -> etiqueta.
function Build-AllImages {
    param([string]$ModelsSrc = 'models', [switch]$Force)
    $src = $ModelsSrc.Replace('\', '/').TrimEnd('/')
    $modelFiles = 'kinetia_seg.onnx', 'manifest.json', 'metrics.json' | ForEach-Object { "$src/$_" }
    foreach ($file in $modelFiles) {
        if (-not (Test-Path (Join-Path $script:RepoRoot $file))) {
            throw "Falta $file. Coloca los archivos del modelo en $src/ (ver models/README.md) y ejecuta scripts/update-model-hashes.ps1."
        }
    }
    return @{
        'kinetia-api' = Build-Image -Name 'kinetia-api' -Context 'backend' -Dockerfile 'backend/Dockerfile' -Force:$Force `
            -Inputs @('backend/Dockerfile', 'backend/.dockerignore', 'backend/requirements.txt', 'backend/app', 'backend/animation_engine')
        'kinetia-frontend' = Build-Image -Name 'kinetia-frontend' -Context 'frontend' -Dockerfile 'frontend/Dockerfile' -Force:$Force `
            -Inputs @('frontend/Dockerfile', 'frontend/.dockerignore', 'frontend/nginx.conf', 'frontend/package.json', 'frontend/package-lock.json', 'frontend/angular.json', 'frontend/tsconfig.json', 'frontend/tsconfig.app.json', 'frontend/.postcssrc.json', 'frontend/public', 'frontend/src')
        'kinetia-model-loader' = Build-Image -Name 'kinetia-model-loader' -Context '.' -Dockerfile 'model-loader/Dockerfile' -Force:$Force `
            -Inputs (@('model-loader', '.dockerignore') + $modelFiles) -BuildArgs @('--build-arg', "MODELS_SRC=$src")
    }
}

# IP de la red local: interfaz activa con puerta de enlace, sin adaptadores virtuales.
function Get-LanIp {
    $configs = Get-NetIPConfiguration -ErrorAction SilentlyContinue | Where-Object {
        $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' -and
        $_.InterfaceAlias -notmatch 'vEthernet|WSL|Docker|VirtualBox|VMware|Loopback|Tailscale|ZeroTier' -and
        $_.NetAdapter.InterfaceDescription -notmatch 'Hyper-V|VirtualBox|VMware|TAP-|Tunnel'
    } | Sort-Object { $_.IPv4DefaultGateway.RouteMetric + $_.NetIPv4Interface.InterfaceMetric }
    foreach ($config in $configs) {
        foreach ($address in $config.IPv4Address) {
            if ($address.IPAddress -notmatch '^169\.254\.') { return $address.IPAddress }
        }
    }
    return $null
}

# Inicia el proceso auxiliar que impide la espera. Se detiene solo al desaparecer el contenedor.
function Start-KeepAwake([string]$WatchContainer) {
    Stop-KeepAwake
    $script = Join-Path $PSScriptRoot 'keep-awake.ps1'
    $command = "powershell.exe -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`" -WatchContainer $WatchContainer"
    # Por CIM: el proceso sobrevive a make y a la consola que lo lanzó.
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $command }
    if ($result.ReturnValue -ne 0) {
        Write-Warning "No se pudo iniciar keep-awake (código $($result.ReturnValue)). Mantén el portátil activo a mano."
        return
    }
    $pidFile = Join-Path $script:StateDir 'keep-awake.pid'
    Write-Utf8NoBom $pidFile "$($result.ProcessId)"
    Write-Host "==> Espera del portátil bloqueada mientras corra $WatchContainer (PID $($result.ProcessId)). No es un cambio permanente."
}

function Stop-KeepAwake {
    $pidFile = Join-Path $script:StateDir 'keep-awake.pid'
    if (-not (Test-Path $pidFile)) { return }
    $id = [int](Get-Content $pidFile -Raw)
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$id" -ErrorAction SilentlyContinue
    if ($process -and $process.CommandLine -match 'keep-awake\.ps1') {
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
        Write-Host "==> Bloqueo de espera liberado (PID $id)."
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

function Show-FirewallAdvice {
    $rule = Get-NetFirewallRule -DisplayName 'KINETIA 8080' -ErrorAction SilentlyContinue | Where-Object { $_.Enabled -eq 'True' }
    if ($rule) { Write-Host '==> Firewall: la regla "KINETIA 8080" está activa.'; return }
    Write-Host ''
    Write-Host 'AVISO firewall: no existe la regla "KINETIA 8080"; los celulares pueden no alcanzar el puerto 8080.' -ForegroundColor Yellow
    Write-Host 'Requiere administrador. Este script no eleva privilegios. Es una regla permanente; en una PowerShell como administrador:'
    Write-Host '  New-NetFirewallRule -DisplayName "KINETIA 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -RemoteAddress LocalSubnet -Action Allow -Profile Private,Public'
    Write-Host 'Para quitarla después:'
    Write-Host '  Remove-NetFirewallRule -DisplayName "KINETIA 8080"'
    Write-Host ''
}
