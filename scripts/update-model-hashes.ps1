# Regenera k8s/base/model-hashes.env desde los archivos del modelo real en models/.
# Se niega a usar el modelo dummy y comprueba que los hashes coinciden con manifest.json.
param([string]$ModelsDir = 'models')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$dir = Join-Path $script:RepoRoot $ModelsDir
foreach ($name in 'kinetia_seg.onnx', 'midas_small.onnx', 'manifest.json', 'metrics.json') {
    if (-not (Test-Path (Join-Path $dir $name))) { throw "Falta $ModelsDir/$name." }
}
$manifest = Get-Content (Join-Path $dir 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.version -eq 'dummy-dev') { throw "$ModelsDir contiene el modelo dummy-dev. Este script solo acepta el modelo real." }

$seg = (Get-FileHash (Join-Path $dir 'kinetia_seg.onnx') -Algorithm SHA256).Hash.ToLower()
$depth = (Get-FileHash (Join-Path $dir 'midas_small.onnx') -Algorithm SHA256).Hash.ToLower()
if ($manifest.files.'kinetia_seg.onnx'.sha256 -ne $seg) { throw 'kinetia_seg.onnx no coincide con manifest.json.' }
if ($manifest.files.'midas_small.onnx'.sha256 -ne $depth) { throw 'midas_small.onnx no coincide con manifest.json.' }

$target = Join-Path $script:RepoRoot 'k8s\base\model-hashes.env'
Write-Utf8NoBom $target "EXPECTED_SHA256_SEG=$seg`nEXPECTED_SHA256_DEPTH=$depth`n"
Write-Host "Actualizado $target (modelo $($manifest.version))."
Write-Host 'Siguiente paso: make down; make demo'
