# Regenera k8s/base/model-hashes.env desde los archivos del modelo real en models/.
# Se niega a usar el modelo dummy o uno que no se entrenó desde cero, y comprueba el hash con manifest.json.
param([string]$ModelsDir = 'models')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$dir = Join-Path $script:RepoRoot $ModelsDir
foreach ($name in 'kinetia_seg.onnx', 'manifest.json', 'metrics.json') {
    if (-not (Test-Path (Join-Path $dir $name))) { throw "Falta $ModelsDir/$name." }
}
$manifest = Get-Content (Join-Path $dir 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.version -eq 'dummy-dev') { throw "$ModelsDir contiene el modelo dummy-dev. Este script solo acepta el modelo real." }
if ($manifest.init -ne 'desde_cero') { throw "$ModelsDir/manifest.json no declara init=desde_cero: es el modelo anterior, con transfer learning." }

$seg = (Get-FileHash (Join-Path $dir 'kinetia_seg.onnx') -Algorithm SHA256).Hash.ToLower()
if ($manifest.files.'kinetia_seg.onnx'.sha256 -ne $seg) { throw 'kinetia_seg.onnx no coincide con manifest.json.' }

$target = Join-Path $script:RepoRoot 'k8s\base\model-hashes.env'
Write-Utf8NoBom $target "EXPECTED_SHA256_SEG=$seg`n"
Write-Host "Actualizado $target (modelo $($manifest.version))."
Write-Host 'Siguiente paso: make down; make demo'
