$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python virtual environment not found: $PythonExe"
}

Set-Location -LiteralPath $ProjectRoot

# Force UTF-8 output so Unicode chars don't crash on Windows cp1252
$env:PYTHONUTF8       = '1'
$env:PYTHONIOENCODING = 'utf-8'

function Get-WritableLogDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot
    )

    $preferred = Join-Path $ProjectRoot 'logs'
    try {
        New-Item -ItemType Directory -Force -Path $preferred | Out-Null
        $probe = Join-Path $preferred (".write_test_{0}.tmp" -f ([guid]::NewGuid().ToString('N')))
        'ok' | Out-File -FilePath $probe -Encoding utf8 -Force
        Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
        return $preferred
    } catch {
        $fallback = Join-Path $env:ProgramData 'QuantitativeStocks\logs'
        New-Item -ItemType Directory -Force -Path $fallback | Out-Null
        return $fallback
    }
}

$LogDir = Get-WritableLogDir -ProjectRoot $ProjectRoot

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'

function Import-DotEnvFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    Get-Content -LiteralPath $Path -ErrorAction Stop | ForEach-Object {
        $line = $_.Trim()
        if ($line.Length -eq 0) { return }
        if ($line.StartsWith('#')) { return }

        $idx = $line.IndexOf('=')
        if ($idx -lt 1) { return }

        $name = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()
        if ($name.Length -eq 0) { return }

        if ($value.StartsWith('"') -and $value.EndsWith('"') -and $value.Length -ge 2) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        Set-Item -Path ("Env:" + $name) -Value $value
    }
}

$EnvCandidates = @(
    (Join-Path $ProjectRoot 'secrets\alpaca.env')
)

if (-not $env:ALPACA_API_KEY -or -not $env:ALPACA_API_SECRET) {
    foreach ($candidate in $EnvCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            Import-DotEnvFile -Path $candidate
            if ($env:ALPACA_API_KEY -and $env:ALPACA_API_SECRET) {
                break
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Range trader uses ALPACA_RANGE_KEY / ALPACA_RANGE_SECRET from secrets\alpaca.env
# Falls back to generic ALPACA_API_KEY / ALPACA_API_SECRET if not set.
# Add these lines to secrets\alpaca.env:
#   ALPACA_RANGE_KEY=<your range paper account key>
#   ALPACA_RANGE_SECRET=<your range paper account secret>
# ---------------------------------------------------------------------------

# Stop any existing range trader processes first
$existingTraders = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'main.py range-trade' }
foreach ($proc in $existingTraders) {
    try { Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop } catch {}
}
if ($existingTraders) {
    Write-Host "Stopped $($existingTraders.Count) existing range trader process(es)."
}

Write-Host "[$(Get-Date -Format s)] Starting range trader..."
Write-Host "Python: $PythonExe"
foreach ($candidate in $EnvCandidates) {
    if (Test-Path -LiteralPath $candidate) { Write-Host "Env file: $candidate" }
}

if (-not $env:ALPACA_API_KEY -or -not $env:ALPACA_API_SECRET) {
    Write-Error "ERROR: Missing ALPACA_API_KEY / ALPACA_API_SECRET. Set machine env vars or add secrets\alpaca.env"
    exit 1
}

# Use intraday group keys; fall back to generic keys
if (-not $env:ALPACA_INTRADAY_KEY) {
    Set-Item -Path "Env:ALPACA_INTRADAY_KEY"    -Value $env:ALPACA_API_KEY    -ErrorAction SilentlyContinue
    Set-Item -Path "Env:ALPACA_INTRADAY_SECRET" -Value $env:ALPACA_API_SECRET -ErrorAction SilentlyContinue
    Write-Host "  [range] No ALPACA_INTRADAY_KEY found - using generic keys (same account as momentum trader)"
}

$Symbols = 'SPY,QQQ,IWM,SOXX'

$ArgList = @(
    '-u', 'main.py', 'range-trade',
    '--symbols',  $Symbols,
    '--provider', 'alpaca'
)
$ArgLine = $ArgList -join ' '

$rangeLog    = Join-Path $LogDir "range_trader_$Timestamp.log"
$rangeErrLog = $rangeLog -replace '\.log$', '_err.log'

$proc = Start-Process `
    -FilePath         $PythonExe `
    -ArgumentList     $ArgLine `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $rangeLog `
    -RedirectStandardError  $rangeErrLog `
    -WindowStyle      Hidden `
    -PassThru

Write-Host "  [range] PID $($proc.Id)  log: $rangeLog"
Write-Host "[$(Get-Date -Format s)] Range trader started."
Write-Host "Logs directory: $LogDir"
