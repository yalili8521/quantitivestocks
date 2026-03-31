# ---------------------------------------------------------------------------
# Single-instance guard: exit if another copy is already running
# (Must run BEFORE setting ErrorActionPreference to Stop)
# ---------------------------------------------------------------------------
$lockFile = Join-Path $env:TEMP 'QuantStocks-PaperTrader.lock'
try {
    $script:lockStream = [System.IO.File]::Open($lockFile, 'OpenOrCreate', 'ReadWrite', 'None')
} catch {
    Write-Host "[$(Get-Date -Format s)] Another instance is already running - exiting."
    exit 0
}

$ErrorActionPreference = 'Stop'

# Global error trap — log unhandled errors to file for S4U debugging
trap {
    $errLog = Join-Path $env:TEMP 'QuantStocks-PaperTrader-crash.log'
    "[$(Get-Date -Format s)] CRASH: $_`n$($_.ScriptStackTrace)" | Out-File -Append -FilePath $errLog -Encoding utf8
    Write-Host "[$(Get-Date -Format s)] CRASH: $_"
}

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
    $tz  = [TimeZoneInfo]::FindSystemTimeZoneById('Pacific Standard Time')
    $now = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $tz)
    $dow = $now.DayOfWeek
    if ($dow -eq 'Saturday' -or $dow -eq 'Sunday') { return $false }
    $t = $now.TimeOfDay
    # 6:20 AM - 1:25 PM PT (= 9:20 AM - 4:25 PM ET)
    return ($t -ge [TimeSpan]'06:20:00' -and $t -le [TimeSpan]'13:25:00')
}

function Test-AfterMarketClose {
    $tz  = [TimeZoneInfo]::FindSystemTimeZoneById('Pacific Standard Time')
    $now = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $tz)
    $dow = $now.DayOfWeek
    $t   = $now.TimeOfDay
    # Weekday after 1:25 PM PT (= 4:25 PM ET) - done for the day
    if ($dow -ne 'Saturday' -and $dow -ne 'Sunday' -and $t -gt [TimeSpan]'13:25:00') { return $true }
    # Weekend
    if ($dow -eq 'Saturday' -or $dow -eq 'Sunday') { return $true }
    return $false
}

$LogDir = Get-WritableLogDir -ProjectRoot $ProjectRoot

$EnvCandidates = @( (Join-Path $ProjectRoot 'secrets\alpaca.env') )
# Always load the .env file so group-specific keys (ALPACA_SWING_KEY, etc.) are available
# even if ALPACA_API_KEY is already set in the system environment.
foreach ($candidate in $EnvCandidates) {
    if (Test-Path -LiteralPath $candidate) {
        Import-DotEnvFile -Path $candidate
        break
    }
}

if (-not $env:ALPACA_INTRADAY_KEY -and (-not $env:ALPACA_API_KEY -or -not $env:ALPACA_API_SECRET)) {
    Write-Error "ERROR: Missing ALPACA_API_KEY / ALPACA_API_SECRET (and no ALPACA_INTRADAY_KEY)."
    exit 1
}

# ---------------------------------------------------------------------------
# Group configuration
# ---------------------------------------------------------------------------
# Config file: production parameters live in config/trading.json
# CLI args from the config file are loaded automatically by paper_trader.py
# Only --group, --mode, --interval are passed here; all tuning params come from config
$Groups = @(
    @{ Name = 'intraday';  EnvPrefix = 'ALPACA_INTRADAY_';  Mode = 'intraday'; Interval = '5min';
       ExtraArgs = @(); Command = 'trade'; AlwaysOn = $true },
    @{ Name = 'swing';     EnvPrefix = 'ALPACA_SWING_';     Mode = 'daily';    Interval = '5min';
       ExtraArgs = @(); Command = 'trade'; AlwaysOn = $true },
    @{ Name = 'crypto';    EnvPrefix = 'KRAKEN_';            Mode = 'daily';    Interval = '5min';
       ExtraArgs = @(); Command = 'trade'; AlwaysOn = $true; SkipKeyCheck = $true },
    @{ Name = 'crypto_intraday'; EnvPrefix = 'KRAKEN_';      Mode = 'intraday'; Interval = '5min';
       ExtraArgs = @(); Command = 'trade'; AlwaysOn = $true; SkipKeyCheck = $true },
    @{ Name = 'gold_scalper';   EnvPrefix = 'AMP_CQG_';       Mode = '';         Interval = '';
       ExtraArgs = @('--broker', 'hybrid'); Command = 'gold-scalper'; AlwaysOn = $true; SkipKeyCheck = $true }
)

$CommonArgs = @('-u', 'main.py')

# ---------------------------------------------------------------------------
# Layer 0: refresh crypto universe (CoinGecko x Kraken, runs once at startup)
# ---------------------------------------------------------------------------
Write-Host "`n  [Layer 0] Refreshing crypto universe (background, 120s timeout)..."
$screenLog = Join-Path $LogDir "universe_screen_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
try {
    $screenProc = Start-Process -FilePath $PythonExe `
        -ArgumentList "-u main.py screen-universe --top-n 250" `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $screenLog `
        -RedirectStandardError ($screenLog -replace '\.log$', '_err.log') `
        -WindowStyle Hidden -PassThru
    $exited = $screenProc.WaitForExit(120000)  # 120s timeout
    if ($exited) {
        Write-Host "  [Layer 0] Universe screen complete (see $screenLog)"
    } else {
        Stop-Process -Id $screenProc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "  [Layer 0] Universe screen timed out after 120s — using cached universe"
    }
} catch {
    Write-Warning "  [Layer 0] Universe screen failed: $_ — using cached universe"
}

# Kill orphaned processes from previous runs — PID-based, not image-name.
# This avoids killing unrelated python.exe (Jupyter, user scripts, etc.).
$pidFile = Join-Path $env:TEMP 'QuantStocks-PaperTrader.pids'
if (Test-Path -LiteralPath $pidFile) {
    $oldPids = Get-Content -Path $pidFile -ErrorAction SilentlyContinue
    foreach ($p in $oldPids) {
        $p = $p.Trim()
        if ($p -match '^\d+$') {
            try { cmd.exe /c "taskkill /F /PID $p /T >nul 2>&1" } catch {}
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2  # let processes fully terminate
    Write-Host "  Killed previous PIDs from pid file."
} else {
    Write-Host "  No previous PID file found — clean start."
}

# Log rotation: delete log files older than 7 days
$oldLogs = Get-ChildItem -Path (Join-Path $LogDir '*.log') -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) }
if ($oldLogs) {
    $oldLogs | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "  Cleaned up $($oldLogs.Count) log files older than 7 days."
}

# ---------------------------------------------------------------------------
# Function: launch one group process, return the Process object
# ---------------------------------------------------------------------------
function Start-TraderGroup {
    param($grp)

    $ts          = Get-Date -Format 'yyyyMMdd_HHmmss'
    $groupLog    = Join-Path $LogDir ("paper_trader_$($grp.Name)_${ts}.log")
    $groupErrLog = $groupLog -replace '\.log$', '_err.log'

    if ($grp.Mode) {
        $modeArgs  = @('--mode', $grp.Mode, '--interval', $grp.Interval)
        $groupArgs = $CommonArgs + @($grp.Command) + $modeArgs + @('--group', $grp.Name) + $grp.ExtraArgs
    } else {
        # Gold scalper and other standalone commands — no --mode/--interval/--group
        $groupArgs = $CommonArgs + @($grp.Command) + $grp.ExtraArgs
    }
    $ArgLine   = $groupArgs -join ' '

    $keyVar = $grp.EnvPrefix + 'KEY'
    $secVar = $grp.EnvPrefix + 'SECRET'
    $keyItem = Get-Item -Path "Env:$keyVar" -ErrorAction SilentlyContinue
    $keyVal  = if ($keyItem) { $keyItem.Value } else { $null }
    if (-not $keyVal -and -not $grp.SkipKeyCheck) {
        Write-Warning "  [$($grp.Name)] $keyVar not set - SKIPPING group (set it in secrets/alpaca.env)"
        return $null
    }

    # Use cmd /c with shell redirection instead of PowerShell's -Redirect*
    # which buffers indefinitely for long-running processes under S4U sessions.
    # NOTE: cmd.exe /c strips the outermost pair of quotes when multiple quoted
    # strings appear. Wrapping the entire command in an extra pair of quotes
    # preserves correct quoting for paths that contain spaces (e.g. OneDrive).
    $innerCmd = "`"$PythonExe`" $ArgLine > `"$groupLog`" 2> `"$groupErrLog`""
    $proc = Start-Process `
        -FilePath         "cmd.exe" `
        -ArgumentList     "/c `"$innerCmd`"" `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle      Hidden `
        -PassThru

    Write-Host "  [$($grp.Name)] PID $($proc.Id)  log: $groupLog"
    return $proc
}

function Save-PidFile {
    # Persist all active cmd.exe wrapper PIDs so the next run can kill them
    $pids = @()
    foreach ($name in $GroupProcs.Keys) {
        $p = $GroupProcs[$name]
        if ($p) { $pids += $p.Id.ToString() }
    }
    if ($pids.Count -gt 0) {
        $pids | Out-File -FilePath $pidFile -Encoding utf8 -Force
    }
}

# ---------------------------------------------------------------------------
# Initial launch
# ---------------------------------------------------------------------------
Write-Host "[$(Get-Date -Format s)] Starting paper trader groups (intraday/swing/crypto)..."
Write-Host "Python: $PythonExe"
foreach ($c in $EnvCandidates) { if (Test-Path -LiteralPath $c) { Write-Host "Env: $c" } }

$GroupProcs = @{}
$inMarketNow = Test-MarketHours
$groupIdx = 0
foreach ($grp in $Groups) {
    # Skip equity groups at startup if outside market hours
    if (-not $grp.AlwaysOn -and -not $inMarketNow) {
        Write-Host "  [$($grp.Name)] Skipped (outside market hours) - watchdog will start at 6:20 AM ET"
        continue
    }
    # Stagger launches by 60s to avoid LightGBM DLL deadlock on Windows.
    # All groups import lightgbm; concurrent DLL init causes hangs.
    if ($groupIdx -gt 0) {
        Write-Host "  Waiting 60s before next group launch (LightGBM DLL stagger)..."
        Start-Sleep -Seconds 60
    }
    try {
        $proc = Start-TraderGroup $grp
        if ($proc) { $GroupProcs[$grp.Name] = $proc }
    } catch {
        Write-Host "  [$($grp.Name)] FAILED to start: $_"
    }
    $groupIdx++
}

Save-PidFile
Write-Host "[$(Get-Date -Format s)] All groups started. Watchdog active."

# ---------------------------------------------------------------------------
# Watchdog loop - check every 2 minutes, restart dead processes
# Equity groups (intraday/swing): only during market hours.
# AlwaysOn groups (crypto): kept alive 24/7 until the script is killed.
# ---------------------------------------------------------------------------
$equityDone = $false
while ($true) {
    Start-Sleep -Seconds 120

    $afterClose = Test-AfterMarketClose
    $inMarket   = Test-MarketHours

    foreach ($grp in $Groups) {
        if (-not $grp.AlwaysOn) {
            # --- Market closed: kill running equity groups ---
            if ($afterClose -or -not $inMarket) {
                $proc = $GroupProcs[$grp.Name]
                if ($proc) {
                    $alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
                    if ($alive) {
                        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                        Write-Host "[$(Get-Date -Format s)] [$($grp.Name)] Stopped (market closed)"
                    }
                    $GroupProcs.Remove($grp.Name)
                }
                continue
            }
            # --- Market open: auto-start if not running ---
            $proc = $GroupProcs[$grp.Name]
            if (-not $proc) {
                Write-Host "[$(Get-Date -Format s)] [$($grp.Name)] Market open - launching..."
                try {
                    $newProc = Start-TraderGroup $grp
                    if ($newProc) { $GroupProcs[$grp.Name] = $newProc }
                } catch {
                    Write-Host "[$(Get-Date -Format s)] [$($grp.Name)] Launch failed: $_"
                }
                continue
            }
        }

        # Restart dead OR hung processes (AlwaysOn groups + equity groups during market hours)
        $proc = $GroupProcs[$grp.Name]
        if (-not $proc) { continue }
        $alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        $needsRestart = $false
        $reason = ''
        if (-not $alive) {
            $needsRestart = $true
            $reason = "died"
        } else {
            # Check if the process is hung: no log file updated in 10 minutes.
            # Gold scalper writes to gold_signal.log / gold_webhook.log (via Python logging),
            # NOT to the paper_trader_gold_scalper_*.log stdout redirect (which stays 0-byte).
            $logPattern    = Join-Path $LogDir ("paper_trader_$($grp.Name)_*.log")
            $errLogPattern = Join-Path $LogDir ("paper_trader_$($grp.Name)_*_err.log")
            $logCandidates = @(
                Get-ChildItem -Path $logPattern    -ErrorAction SilentlyContinue
                Get-ChildItem -Path $errLogPattern -ErrorAction SilentlyContinue
            )
            # Also check the gold scalper's own RotatingFileHandler logs
            if ($grp.Name -eq 'gold_scalper') {
                $logCandidates += @(
                    Get-Item -Path (Join-Path $LogDir 'gold_signal.log') -ErrorAction SilentlyContinue
                    Get-Item -Path (Join-Path $LogDir 'gold_webhook.log') -ErrorAction SilentlyContinue
                )
            }
            $latestLog = $logCandidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latestLog) {
                $staleMins = ((Get-Date) - $latestLog.LastWriteTime).TotalMinutes
                if ($staleMins -gt 10) {
                    $needsRestart = $true
                    $reason = "hung (log stale $([int]$staleMins)min)"
                    # Tree-kill the hung process
                    try { cmd.exe /c "taskkill /F /PID $($proc.Id) /T >nul 2>&1" } catch {}
                }
            }
        }
        if ($needsRestart) {
            Write-Host "[$(Get-Date -Format s)] [$($grp.Name)] Process $($proc.Id) $reason - restarting..."
            try {
                $newProc = Start-TraderGroup $grp
                $GroupProcs[$grp.Name] = $newProc
            } catch {
                Write-Host "[$(Get-Date -Format s)] [$($grp.Name)] Restart failed: $_"
            }
        }
    }

    # Update PID file after any restarts
    Save-PidFile

    # Log once when equity market closes
    if ($afterClose -and -not $equityDone) {
        $equityDone = $true
        Write-Host "[$(Get-Date -Format s)] Market closed - equity groups stopped. Crypto continues 24/7."
    }

    # Reset flag at market open so it logs again tomorrow
    if ($inMarket -and $equityDone) {
        $equityDone = $false
    }
}
