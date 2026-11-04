# PASO PREVIO, CON INTERNET: deja en caché local todo lo que `make demo` necesita para funcionar sin red.
# - Imagen del nodo k3s, k3d-proxy (la exige el mapeo -p) y k3d-tools (la usa `k3d image import`).
# - Imágenes de sistema de k3s (airgap oficial) en un volumen de Docker que el nodo importa al arrancar.
# - Imágenes base de los Dockerfile, para poder reconstruir sin red.
# Es idempotente. `make down` no borra esta caché.
$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'common.ps1')

foreach ($image in @($script:K3sImage, $script:K3dToolsImage, $script:K3dProxyImage) + $script:BaseImages) {
    if (Test-DockerImage $image) { Write-Host "==> En caché: $image"; continue }
    foreach ($attempt in 1..4) {
        docker pull $image | Out-Host
        if ($LASTEXITCODE -eq 0) { break }
        Write-Host "==> Reintento $attempt de docker pull"
        Start-Sleep -Seconds 5
    }
    Assert-Exit "docker pull $image"
}

$archive = 'k3s-airgap-images-amd64.tar.zst'
$exists = docker volume ls -q --filter "name=^$($script:AirgapVolume)$"
if ($exists) {
    docker run --rm -v "$($script:AirgapVolume):/images:ro" alpine:3.20 test -s "/images/$archive"
    if ($LASTEXITCODE -eq 0) { Write-Host "==> En caché: volumen $($script:AirgapVolume)"; exit 0 }
}

$release = $script:K3sVersion -replace '-k3s', '+k3s'
$url = "https://github.com/k3s-io/k3s/releases/download/$([uri]::EscapeDataString($release))/$archive"
$cacheDir = Join-Path $env:LOCALAPPDATA 'kinetia-tools\k3s-images'
New-Item -ItemType Directory -Force $cacheDir | Out-Null
$local = Join-Path $cacheDir $archive
if (-not (Test-Path $local)) {
    Write-Host "==> Descargando $url"
    curl.exe -fsSL --retry 3 -o "$local.part" $url
    Assert-Exit 'Descarga de las imágenes airgap'
    Move-Item -Force "$local.part" $local
}

docker volume create $script:AirgapVolume | Out-Null
$helper = 'kinetia-airgap-fill'
docker rm -f $helper 2>$null | Out-Null
docker create --name $helper -v "$($script:AirgapVolume):/images" alpine:3.20 true | Out-Null
Assert-Exit 'Crear contenedor auxiliar'
docker cp $local "${helper}:/images/$archive"
$copyExit = $LASTEXITCODE
docker rm -f $helper | Out-Null
if ($copyExit -ne 0) { throw 'No se pudo copiar el archivo airgap al volumen.' }
Write-Host "==> Caché lista: volumen $($script:AirgapVolume)"
