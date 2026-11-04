# make smoke: ejecuta smoke.py contra :8080 y comprueba /metrics con un port-forward al 8000.
param([switch]$Full, [switch]$NoKubernetes, [string]$BaseUrl = 'http://localhost:8080')
$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot '..\..\scripts\common.ps1')

$python = Join-Path $script:RepoRoot 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
$smokeArgs = @((Join-Path $PSScriptRoot 'smoke.py'), '--base-url', $BaseUrl)
if ($Full) { $smokeArgs += '--full' }

$forward = $null
if (-not $NoKubernetes) {
    $forward = Start-Process kubectl -PassThru -WindowStyle Hidden -ArgumentList '--context', $script:KubeContext, '-n', 'kinetia', 'port-forward', 'deploy/kinetia', '18000:8000'
    Start-Sleep -Seconds 3
    $smokeArgs += @('--metrics-url', 'http://127.0.0.1:18000')
}
try {
    & $python @smokeArgs
    $code = $LASTEXITCODE
} finally {
    if ($forward) { Stop-Process -Id $forward.Id -Force -ErrorAction SilentlyContinue }
}
exit $code
