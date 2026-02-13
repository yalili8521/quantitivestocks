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
    (Join-Path $ProjectRoot 'secrets\alpaca.env'),
    (Join-Path $ProjectRoot 'settings\alpaca.env')
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

$Args = @(
    'main.py','trade',
    '--provider','alpaca',
    '--mode','intraday',
    '--interval','5min',
    '--confidence','0.2',
    '--trailing-stop','0.05',
    '--take-profit','0.08'
)

# Ensure a fresh daily run/log: stop any existing trader loop first.
$existingTraders = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'main.py trade' }
foreach ($proc in $existingTraders) {
    try {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
    } catch {
        # Continue; lock in paper_trader.py still prevents duplicate loops.
    }
}

"[$(Get-Date -Format s)] Starting paper trader..." | Out-File -FilePath $LogFile -Encoding utf8
"Python: $PythonExe" | Out-File -FilePath $LogFile -Encoding utf8 -Append
"Log: $LogFile" | Out-File -FilePath $LogFile -Encoding utf8 -Append
foreach ($candidate in $EnvCandidates) {
    if (Test-Path -LiteralPath $candidate) {
        "Env file present: $candidate" | Out-File -FilePath $LogFile -Encoding utf8 -Append
    }
}
"Args: $($Args -join ' ')" | Out-File -FilePath $LogFile -Encoding utf8 -Append
if ($existingTraders) {
    "Stopped existing trader process count: $($existingTraders.Count)" | Out-File -FilePath $LogFile -Encoding utf8 -Append
}

if (-not $env:ALPACA_API_KEY -or -not $env:ALPACA_API_SECRET) {
    "ERROR: Missing ALPACA_API_KEY / ALPACA_API_SECRET. Set machine env vars or add one of: $($EnvCandidates -join ', ')" | Out-File -FilePath $LogFile -Encoding utf8 -Append
    exit 1
}

# Capture Python stdout+stderr to the log (cmd redirection is the most reliable across PS versions)
$ArgLine = $Args -join ' '
$CmdLine = "`"$PythonExe`" $ArgLine >> `"$LogFile`" 2>&1"
& cmd.exe /c $CmdLine
