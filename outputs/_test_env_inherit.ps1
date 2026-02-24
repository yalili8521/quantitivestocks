# Test: does a child cmd.exe process inherit env vars loaded from the .env file?

$envFile = Join-Path (Split-Path -Parent $PSScriptRoot) 'settings\alpaca.env'

# Load .env into current process
Get-Content -LiteralPath $envFile | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith('#')) { return }
    $idx = $line.IndexOf('=')
    if ($idx -lt 1) { return }
    $name  = $line.Substring(0, $idx).Trim()
    $value = $line.Substring($idx + 1).Trim().Trim('"')
    [System.Environment]::SetEnvironmentVariable($name, $value)
}

Write-Host "In this PS process — ALPACA_API_KEY set: $([bool]$env:ALPACA_API_KEY)"

# Launch child cmd.exe with UseShellExecute=false and capture output
$psi = New-Object System.Diagnostics.ProcessStartInfo("cmd.exe")
$psi.Arguments             = "/c echo ALPACA_API_KEY=%ALPACA_API_KEY%"
$psi.UseShellExecute       = $false
$psi.RedirectStandardOutput = $true
$psi.CreateNoWindow        = $true
$p   = [System.Diagnostics.Process]::Start($psi)
$out = $p.StandardOutput.ReadToEnd()
$p.WaitForExit()
Write-Host ('Child cmd.exe output: ' + $out)
