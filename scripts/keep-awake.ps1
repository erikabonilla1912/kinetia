# Impide la espera del sistema y el apagado de pantalla mientras exista el contenedor vigilado.
# SetThreadExecutionState solo afecta a este hilo: al terminar el proceso, Windows lo revierte solo.
param([Parameter(Mandatory = $true)][string]$WatchContainer, [int]$IntervalSeconds = 30)

Add-Type -Namespace Kinetia -Name Power -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@

$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
$ES_DISPLAY_REQUIRED = [uint32]2
$flags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED
$statusFile = Join-Path $env:TEMP 'kinetia\keep-awake.status'

try {
    while ($true) {
        $state = docker inspect -f '{{.State.Running}}' $WatchContainer 2>$null
        if ("$state" -ne 'true') { break }
        $previous = [Kinetia.Power]::SetThreadExecutionState($flags)
        $line = "{0} pid={1} watch={2} flags=0x{3:X8} previous=0x{4:X8}" -f (Get-Date -Format s), $PID, $WatchContainer, $flags, $previous
        [IO.File]::WriteAllText($statusFile, $line)
        Start-Sleep -Seconds $IntervalSeconds
    }
} finally {
    [void][Kinetia.Power]::SetThreadExecutionState($ES_CONTINUOUS)
    [IO.File]::WriteAllText($statusFile, "$(Get-Date -Format s) pid=$PID finalizado: espera restaurada")
}
