# make down: borra el clúster y libera el bloqueo de espera. La caché de imágenes se conserva.
$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'common.ps1')
Stop-KeepAwake
if (Test-ClusterExists) {
    k3d cluster delete $script:ClusterName
    Assert-Exit 'k3d cluster delete'
} else {
    Write-Host '==> No hay clúster que borrar.'
}
exit 0
