$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python virtual environment not found: $PythonExe"
}

Set-Location -LiteralPath $ProjectRoot

$env:PYTHONUTF8      = '1'
$env:PYTHONIOENCODING = 'utf-8'

# Load env vars
function Import-DotEnvFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    Get-Content -LiteralPath $Path -ErrorAction Stop | ForEach-Object {
        $line = $_.Trim()
        if ($line.Length -eq 0 -or $line.StartsWith('#')) { return }
        $idx = $line.IndexOf('=')
        if ($idx -lt 1) { return }
        $name  = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()
        if ($name.Length -eq 0) { return }
        if ($value.StartsWith('"') -and $value.EndsWith('"') -and $value.Length -ge 2) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        Set-Item -Path ("Env:" + $name) -Value $value
    }
}

$EnvFile = Join-Path $ProjectRoot 'secrets\alpaca.env'
if (Test-Path -LiteralPath $EnvFile) {
    Import-DotEnvFile -Path $EnvFile
}

# Set IBKR connection defaults if not in env
if (-not $env:IBKR_HOST)      { $env:IBKR_HOST = '127.0.0.1' }
if (-not $env:IBKR_PORT)      { $env:IBKR_PORT = '7497' }
if (-not $env:IBKR_CLIENT_ID) { $env:IBKR_CLIENT_ID = '10' }

# Log directory
$LogDir = Join-Path $ProjectRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogOut = Join-Path $LogDir "gold_ibkr_${ts}.log"
$LogErr = Join-Path $LogDir "gold_ibkr_${ts}_err.log"

Write-Host "[$(Get-Date -Format s)] Starting IBKR gold scalper. Logs: $LogOut"

# Check if TWS/Gateway is reachable
$tcpTest = Test-NetConnection -ComputerName $env:IBKR_HOST -Port $env:IBKR_PORT -WarningAction SilentlyContinue
if (-not $tcpTest.TcpTestSucceeded) {
    Write-Host "[$(Get-Date -Format s)] ERROR: TWS/Gateway not reachable at ${env:IBKR_HOST}:${env:IBKR_PORT}"
    Write-Host "  Make sure TWS or IB Gateway is running with API enabled."
    exit 1
}

# Run the gold scalper with IBKR broker
& $PythonExe -u main.py gold-scalper --broker ibkr 1>$LogOut 2>$LogErr

Write-Host "[$(Get-Date -Format s)] IBKR gold scalper exited."
