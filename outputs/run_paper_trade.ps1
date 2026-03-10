$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python virtual environment not found: $PythonExe"
}

Set-Location -LiteralPath $ProjectRoot

# Force UTF-8 output so Unicode chars (arrows, symbols) don't crash on Windows cp1252
$env:PYTHONUTF8     = '1'
$env:PYTHONIOENCODING = 'utf-8'

function Get-WritableLogDir {
    param([string]$ProjectRoot)
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

function Test-MarketHours {
    $tz  = [TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
    $now = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $tz)
    $dow = $now.DayOfWeek
    if ($dow -eq 'Saturday' -or $dow -eq 'Sunday') { return $false }
    $t = $now.TimeOfDay
    # 6:20 AM - 4:25 PM ET
    return ($t -ge [TimeSpan]'06:20:00' -and $t -le [TimeSpan]'16:25:00')
}

function Test-AfterMarketClose {
    $tz  = [TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
    $now = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $tz)
    $dow = $now.DayOfWeek
    $t   = $now.TimeOfDay
    # Weekday after 4:25 PM ET - done for the day
    if ($dow -ne 'Saturday' -and $dow -ne 'Sunday' -and $t -gt [TimeSpan]'16:25:00') { return $true }
    # Weekend
    if ($dow -eq 'Saturday' -or $dow -eq 'Sunday') { return $true }
    return $false
}

$LogDir = Get-WritableLogDir -ProjectRoot $ProjectRoot

$EnvCandidates = @( (Join-Path $ProjectRoot 'secrets\alpaca.env') )
if (-not $env:ALPACA_API_KEY -or -not $env:ALPACA_API_SECRET) {
    foreach ($candidate in $EnvCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            Import-DotEnvFile -Path $candidate
            if ($env:ALPACA_API_KEY -and $env:ALPACA_API_SECRET) { break }
        }
    }
}

if (-not $env:ALPACA_API_KEY -or -not $env:ALPACA_API_SECRET) {
    Write-Error "ERROR: Missing ALPACA_API_KEY / ALPACA_API_SECRET."
    exit 1
}

# ---------------------------------------------------------------------------
# Group configuration
# ---------------------------------------------------------------------------
$Groups = @(
    @{ Name = 'intraday';  EnvPrefix = 'ALPACA_INTRADAY_';  Mode = 'intraday'; Interval = '5min';
       ExtraArgs = @(); Command = 'trade' },
    @{ Name = 'swing';     EnvPrefix = 'ALPACA_SWING_';     Mode = 'daily';    Interval = '5min';
       ExtraArgs = @(); Command = 'trade' },
    @{ Name = 'expansion'; EnvPrefix = 'ALPACA_EXPANSION_'; Mode = 'daily';    Interval = '5min';
       ExtraArgs = @(); Command = 'trade' },
    @{ Name = 'spreads';   EnvPrefix = 'ALPACA_EXPANSION_'; Mode = 'daily';    Interval = '5min';
       ExtraArgs = @('--vix-threshold', '25.0'); Command = 'spread-trade' }
)

$CommonArgs = @('-u', 'main.py')

# Stop any existing trader processes (LSTM + range)
$existingTraders = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'main\.py (trade|range-trade)' }
foreach ($proc in $existingTraders) {
    try { Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop } catch {}
}
if ($existingTraders) { Write-Host "Stopped $($existingTraders.Count) existing trader process(es)." }

# ---------------------------------------------------------------------------
# Function: launch one group process, return the Process object
# ---------------------------------------------------------------------------
function Start-TraderGroup {
    param($grp)

    $ts          = Get-Date -Format 'yyyyMMdd_HHmmss'
    $groupLog    = Join-Path $LogDir ("paper_trader_$($grp.Name)_${ts}.log")
    $groupErrLog = $groupLog -replace '\.log$', '_err.log'

    if ($grp.Command -eq 'spread-trade') {
        $groupArgs = $CommonArgs + @($grp.Command) + $grp.ExtraArgs
    } else {
        $modeArgs  = @('--mode', $grp.Mode, '--interval', $grp.Interval)
        $groupArgs = $CommonArgs + @($grp.Command) + $modeArgs + @('--group', $grp.Name) + $grp.ExtraArgs
    }
    $ArgLine   = $groupArgs -join ' '

    $keyVar = $grp.EnvPrefix + 'KEY'
    $secVar = $grp.EnvPrefix + 'SECRET'
    if (-not (Get-Item -Path "Env:$keyVar" -ErrorAction SilentlyContinue)) {
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
    return $proc
}

# ---------------------------------------------------------------------------
# Initial launch
# ---------------------------------------------------------------------------
Write-Host "[$(Get-Date -Format s)] Starting paper trader groups (intraday/swing/expansion)..."
Write-Host "Python: $PythonExe"
foreach ($c in $EnvCandidates) { if (Test-Path -LiteralPath $c) { Write-Host "Env: $c" } }

$GroupProcs = @{}
foreach ($grp in $Groups) {
    try {
        $proc = Start-TraderGroup $grp
        $GroupProcs[$grp.Name] = $proc
    } catch {
        Write-Host "  [$($grp.Name)] FAILED to start: $_"
    }
}

Write-Host "[$(Get-Date -Format s)] All groups started. Watchdog active."

# ---------------------------------------------------------------------------
# Watchdog loop - check every 2 minutes, restart dead processes during market hours
# ---------------------------------------------------------------------------
while (-not (Test-AfterMarketClose)) {
    Start-Sleep -Seconds 120

    if (-not (Test-MarketHours)) { continue }

    foreach ($grp in $Groups) {
        $proc = $GroupProcs[$grp.Name]
        if (-not $proc) { continue }
        $alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if (-not $alive) {
            Write-Host "[$(Get-Date -Format s)] [$($grp.Name)] Process $($proc.Id) died - restarting..."
            try {
                $newProc = Start-TraderGroup $grp
                $GroupProcs[$grp.Name] = $newProc
            } catch {
                Write-Host "[$(Get-Date -Format s)] [$($grp.Name)] Restart failed: $_"
            }
        }
    }
}

Write-Host "[$(Get-Date -Format s)] Market closed - watchdog exiting."
