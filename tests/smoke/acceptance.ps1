# Criterios 3 y 4: resiliencia con PVC y rechazo de un hash incorrecto. Requiere `make demo` activo.
$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot '..\..\scripts\common.ps1')
$kc = @('--context', $script:KubeContext, '-n', 'kinetia')
$failed = $false
function Result([string]$Name, [bool]$Ok, [string]$Detail = '') {
    $word = if ($Ok) { 'OK' } else { 'FALLA'; $script:failed = $true }
    Write-Host "[$word] $Name $Detail"
}
function Wait-Ready([int]$Seconds = 240) {
    kubectl @kc rollout status deploy/kinetia "--timeout=${Seconds}s" | Out-Null
    return ($LASTEXITCODE -eq 0)
}

# --- Criterio 3: se necesita un resultado previo; se obtiene de un job real.
$python = Join-Path $script:RepoRoot 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
$probe = @'
import json, sys
sys.path.insert(0, sys.argv[1])
import smoke
small = smoke.noise_png(640, 480)
print(smoke.run_job("http://localhost:8080", small, "Previo", 240))
'@
$probeFile = Join-Path $script:StateDir 'prior-job.py'
Write-Utf8NoBom $probeFile $probe
$output = & $python $probeFile $PSScriptRoot
$resultPath = ($output | Select-Object -Last 1).Trim()
$output | Select-Object -SkipLast 1 | Write-Host
$before = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8080$resultPath" -TimeoutSec 30
$hashBefore = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($before.RawContentStream.ToArray()))

$oldPod = kubectl @kc get pods -o 'jsonpath={.items[0].metadata.name}'
$start = Get-Date
kubectl @kc delete pod $oldPod --wait=true | Out-Host
$ok = Wait-Ready
Start-Sleep -Seconds 2
$table = kubectl @kc get pods --no-headers | Out-String
Result 'Criterio 3: el pod se recrea y queda Ready' ($ok -and $table -match '2/2\s+Running') "en $([int]((Get-Date) - $start).TotalSeconds) s -> $($table.Trim())"
$after = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8080$resultPath" -TimeoutSec 30
$hashAfter = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($after.RawContentStream.ToArray()))
Result 'Criterio 3: el resultado previo sigue disponible (mismo SHA-256)' ($hashBefore -eq $hashAfter) $resultPath

# --- Criterio 4: hash incorrecto en el ConfigMap.
$original = kubectl @kc get configmap kinetia-model-hashes -o 'jsonpath={.data.EXPECTED_SHA256_SEG}'
$bad = '0' * 64
$patch = Join-Path $script:StateDir 'bad-hash.json'
Write-Utf8NoBom $patch "{`"data`":{`"EXPECTED_SHA256_SEG`":`"$bad`"}}"
kubectl @kc patch configmap kinetia-model-hashes --type merge --patch-file $patch | Out-Null
kubectl @kc delete pod --all --wait=true | Out-Null
$state = ''
foreach ($attempt in 1..30) {
    Start-Sleep -Seconds 3
    $state = kubectl @kc get pods --no-headers | Out-String
    if ($state -match 'Init:Error|Init:CrashLoopBackOff') { break }
}
Result 'Criterio 4: hash incorrecto deja el pod en Init:Error/CrashLoop' ($state -match 'Init:Error|Init:CrashLoopBackOff') $state.Trim()
$podJson = kubectl @kc get pods -o json | Out-String | ConvertFrom-Json
$apiStatus = @($podJson.items[0].status.containerStatuses | Where-Object { $_.name -eq 'api' })
$apiRan = ($apiStatus.Count -eq 1) -and ($null -ne $apiStatus[0].state.running -or $apiStatus[0].restartCount -gt 0 -or $apiStatus[0].started -eq $true)
$apiDetail = if ($apiStatus.Count -eq 1) { $apiStatus[0].state | ConvertTo-Json -Compress } else { 'sin estado del contenedor api' }
Result 'Criterio 4: la API no arranca' (($apiStatus.Count -eq 1) -and -not $apiRan) $apiDetail
kubectl @kc logs deploy/kinetia -c model-loader --tail=3 | Write-Host

Write-Utf8NoBom $patch "{`"data`":{`"EXPECTED_SHA256_SEG`":`"$original`"}}"
kubectl @kc patch configmap kinetia-model-hashes --type merge --patch-file $patch | Out-Null
kubectl @kc delete pod --all --wait=true | Out-Null
Result 'Restauración: con el hash correcto vuelve a Ready' (Wait-Ready)
Remove-Item $patch, $probeFile -ErrorAction SilentlyContinue
if ($failed) { exit 1 }
exit 0
