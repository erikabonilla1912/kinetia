# Crea o actualiza el Secret kinetia-secret desde ADMIN_TOKEN de .env. Nunca se versiona.
# Si .env no lo define, conserva el token del Secret o genera uno aleatorio que solo vive en el clúster.
# Escribe en la salida una huella corta del token, para reiniciar el pod cuando cambie.
param([string]$KubeContext = 'k3d-kinetia')
. (Join-Path $PSScriptRoot 'common.ps1')

$token = ''
$envFile = Join-Path $script:RepoRoot '.env'
if (Test-Path $envFile) {
    $match = Get-Content $envFile | Select-String -Pattern '^\s*ADMIN_TOKEN\s*=\s*(.*)$' | Select-Object -First 1
    if ($match) { $token = $match.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'") }
}
if (-not $token) {
    $current = kubectl --context $KubeContext -n kinetia get secret kinetia-secret -o 'jsonpath={.data.ADMIN_TOKEN}' --ignore-not-found
    if ($current) {
        $token = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($current))
    } else {
        $bytes = New-Object byte[] 24
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $token = -join ($bytes | ForEach-Object { $_.ToString('x2') })
        Write-Host '==> .env no define ADMIN_TOKEN: se generó uno aleatorio, guardado solo en el Secret.'
    }
}

# Por stdin: el token no aparece en la línea de comandos de ningún proceso.
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($token))
$manifest = "apiVersion: v1`nkind: Secret`nmetadata:`n  name: kinetia-secret`n  namespace: kinetia`ntype: Opaque`ndata:`n  ADMIN_TOKEN: $encoded`n"
$manifest | kubectl --context $KubeContext apply -f - | Out-Host
Assert-Exit 'Crear kinetia-secret'

$sha = [Security.Cryptography.SHA256]::Create()
$digest = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($token))
Write-Output (-join ($digest | ForEach-Object { $_.ToString('x2') })).Substring(0, 12)
