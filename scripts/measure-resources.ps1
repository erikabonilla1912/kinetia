# Mide memoria y CPU reales con carga, para sustituir los valores "# PROVISIONAL" de deployment.yaml.
# Ejecutar con el modelo REAL y con `make demo` activo. Lanza N rondas de smoke con subidas grandes
# mientras muestrea kubectl top y docker stats. Guarda un CSV y muestra los picos.
param([int]$Rounds = 3, [int]$SampleSeconds = 5, [string]$OutputCsv = (Join-Path $env:TEMP 'kinetia\resources.csv'))
$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'common.ps1')

function ConvertTo-MiB([string]$Value) {
    if ($Value -match '^([\d.]+)\s*(KiB|MiB|GiB|Ki|Mi|Gi)') {
        $number = [double]$Matches[1]
        switch -Regex ($Matches[2]) { '^K' { return $number / 1024 } '^M' { return $number } '^G' { return $number * 1024 } }
    }
    return 0
}

$health = Invoke-RestMethod -Uri 'http://localhost:8080/api/v1/health' -TimeoutSec 10
$isDummy = $health.model_version -eq 'dummy-dev'
Write-Host "Modelo: $($health.model_version)"
if ($isDummy) { Write-Host 'AVISO: modelo dummy. Estas cifras NO valen como cifras finales.' -ForegroundColor Yellow }

$python = Join-Path $script:RepoRoot 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
$smoke = Join-Path $script:RepoRoot 'tests\smoke\smoke.py'
$logFile = Join-Path $script:StateDir 'measure-load.log'
$load = Start-Process -FilePath 'powershell.exe' -PassThru -WindowStyle Hidden -ArgumentList '-NoProfile', '-Command',
    "1..$Rounds | ForEach-Object { & '$python' '$smoke' --full } *> '$logFile'"

$samples = New-Object Collections.Generic.List[object]
do {
    $time = Get-Date -Format s
    foreach ($line in (kubectl --context $script:KubeContext -n kinetia top pod --containers --no-headers 2>$null)) {
        $parts = -split $line
        if ($parts.Count -ge 4) {
            $samples.Add([pscustomobject]@{ time = $time; source = $parts[1]; cpu_m = [double]($parts[2] -replace 'm', ''); mem_mib = ConvertTo-MiB $parts[3] })
        }
    }
    $node = (docker stats --no-stream --format '{{.CPUPerc}};{{.MemUsage}}' $script:NodeContainer) -split ';'
    if ($node.Count -eq 2) {
        $samples.Add([pscustomobject]@{ time = $time; source = 'nodo-k3d'; cpu_m = 10 * [double]($node[0] -replace '%', ''); mem_mib = ConvertTo-MiB $node[1] })
    }
    Start-Sleep -Seconds $SampleSeconds
} while (-not $load.HasExited)

$samples | Export-Csv -NoTypeInformation -Encoding UTF8 $OutputCsv
Write-Host "`nMuestras: $($samples.Count). CSV: $OutputCsv. Registro de la carga: $logFile"
$samples | Group-Object source | ForEach-Object {
    [pscustomobject]@{
        origen = $_.Name
        pico_mem_MiB = [math]::Round(($_.Group | Measure-Object mem_mib -Maximum).Maximum, 0)
        media_mem_MiB = [math]::Round(($_.Group | Measure-Object mem_mib -Average).Average, 0)
        pico_cpu_m = [math]::Round(($_.Group | Measure-Object cpu_m -Maximum).Maximum, 0)
    }
} | Format-Table -AutoSize
Write-Host 'kubectl top muestrea cada ~15 s: los picos breves pueden no aparecer. Deja margen en los limits.'
Write-Host 'Registra los valores con su evidencia en PROGRESO.md y quita los comentarios # PROVISIONAL.'
