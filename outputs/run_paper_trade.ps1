$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python virtual environment not found: $PythonExe"
}

Set-Location -LiteralPath $ProjectRoot

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
        # Fall back when running as SYSTEM and the project folder isn't writable (common under OneDrive)
        $fallback = Join-Path $env:ProgramData 'QuantitativeStocks\logs'
        New-Item -ItemType Directory -Force -Path $fallback | Out-Null
        return $fallback
    }
}

$LogDir = Get-WritableLogDir -ProjectRoot $ProjectRoot

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogFile = Join-Path $LogDir "paper_trader_$Timestamp.log"

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

        # Strip optional surrounding quotes
        if ($value.StartsWith('"') -and $value.EndsWith('"') -and $value.Length -ge 2) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        Set-Item -Path ("Env:" + $name) -Value $value
    }
}

# Scheduled task context may not inherit user env vars.
# Try local env files in priority order.
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
# 3-account group configuration
# Groups split by trading MODE (not asset class):
#   intraday  — SPY/QQQ/IWM/SOXX    → 5-min LSTM, time filter active
#   swing     — EWT/GLD/EEM/SLV     → daily LSTM (TLT dropped: poor win rate)
#   expansion — EWJ/EWS/XLE/INDA    → daily LSTM (IGV/FXI dropped: <50% win rate)
# Group → env key prefix (e.g. ALPACA_INTRADAY_KEY / ALPACA_INTRADAY_SECRET)
# Fall back to generic ALPACA_API_KEY / ALPACA_API_SECRET if group keys not set.
# ---------------------------------------------------------------------------
$Groups = @(
    @{ Name = 'intraday';  EnvPrefix = 'ALPACA_INTRADAY_';  Mode = 'intraday'; Interval = '5min' },
    @{ Name = 'swing';     EnvPrefix = 'ALPACA_SWING_';     Mode = 'daily';    Interval = '5min' },
    @{ Name = 'expansion'; EnvPrefix = 'ALPACA_EXPANSION_'; Mode = 'daily';    Interval = '5min' }
)

$CommonArgs = @(
    '-u', 'main.py', 'trade',
    '--provider',         'alpaca',
    '--confidence',       '0.05',
    '--short-confidence', '0.05',
    '--exit-confidence',  '0.02',
    '--trailing-stop',    '0.05',
    '--take-profit',      '0.08'
)

# Stop any existing trader processes first
$existingTraders = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'main.py trade' }
foreach ($proc in $existingTraders) {
    try { Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop } catch {}
}
if ($existingTraders) {
    Write-Host "Stopped $($existingTraders.Count) existing trader process(es)."
}

Write-Host "[$(Get-Date -Format s)] Starting 3 paper trader account groups (intraday/swing/expansion)..."
Write-Host "Python: $PythonExe"
foreach ($candidate in $EnvCandidates) {
    if (Test-Path -LiteralPath $candidate) { Write-Host "Env file: $candidate" }
}

# Validate at least the generic fallback keys exist
if (-not $env:ALPACA_API_KEY -or -not $env:ALPACA_API_SECRET) {
    Write-Error "ERROR: Missing ALPACA_API_KEY / ALPACA_API_SECRET. Set machine env vars or add secrets\alpaca.env"
    exit 1
}

# Launch one process per group
foreach ($grp in $Groups) {
    $modeArgs  = @('--mode', $grp.Mode, '--interval', $grp.Interval)
    $groupArgs = $CommonArgs + $modeArgs + @('--group', $grp.Name)
    $ArgLine   = $groupArgs -join ' '
    $groupLog   = Join-Path $LogDir ("paper_trader_$($grp.Name)_$Timestamp.log")
    $groupErrLog = $groupLog -replace '\.log$', '_err.log'

    # Set group-specific env vars for this process if available
    $keyVar = $grp.EnvPrefix + 'KEY'
    $secVar = $grp.EnvPrefix + 'SECRET'
    if (-not (Get-Item -Path "Env:$keyVar" -ErrorAction SilentlyContinue)) {
        # Group key not set — copy generic keys so the process inherits them
        Set-Item -Path "Env:$keyVar" -Value $env:ALPACA_API_KEY   -ErrorAction SilentlyContinue
        Set-Item -Path "Env:$secVar" -Value $env:ALPACA_API_SECRET -ErrorAction SilentlyContinue
    }

    $proc = Start-Process `
        -FilePath         $PythonExe `
        -ArgumentList     $ArgLine `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $groupLog `
        -RedirectStandardError  $groupErrLog `
        -WindowStyle      Hidden `
        -PassThru
    Write-Host "  [$($grp.Name)] PID $($proc.Id)  log: $groupLog"
}

Write-Host "[$(Get-Date -Format s)] All 3 trader groups started."
Write-Host "Logs directory: $LogDir"
