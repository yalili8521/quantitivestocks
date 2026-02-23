# One-time script: persist API keys from .env file as Windows user environment variables.
# This makes them available to all processes (including detached/scheduled task processes).

$envFile = Join-Path (Split-Path -Parent $PSScriptRoot) 'settings\alpaca.env'

if (-not (Test-Path -LiteralPath $envFile)) {
    Write-Error "Env file not found: $envFile"
    exit 1
}

Get-Content -LiteralPath $envFile | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith('#')) { return }
    $idx = $line.IndexOf('=')
    if ($idx -lt 1) { return }
    $name  = $line.Substring(0, $idx).Trim()
    $value = $line.Substring($idx + 1).Trim().Trim('"')
    [System.Environment]::SetEnvironmentVariable($name, $value, 'User')
    Write-Host "Set user env var: $name"
}

Write-Host "`nDone. Re-open any terminals for the new vars to take effect."
