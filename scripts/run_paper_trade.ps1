$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python virtual environment not found: $PythonExe"
}

Set-Location -LiteralPath $ProjectRoot

# When running via Task Scheduler as SYSTEM, writing under the project folder may fail.
# ProgramData is writable by SYSTEM and typically writable by admins.
$LogDir = Join-Path $env:ProgramData 'QuantitativeStocks\logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

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

# SYSTEM won't see your per-user env vars. If keys aren't present, load them from a local file.
$SecretsEnv = Join-Path $ProjectRoot 'secrets\alpaca.env'
if (-not $env:ALPACA_API_KEY -or -not $env:ALPACA_API_SECRET) {
    if (Test-Path -LiteralPath $SecretsEnv) {
        Import-DotEnvFile -Path $SecretsEnv
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

"[$(Get-Date -Format s)] Starting paper trader..." | Out-File -FilePath $LogFile -Encoding utf8
"Python: $PythonExe" | Out-File -FilePath $LogFile -Encoding utf8 -Append
"Log: $LogFile" | Out-File -FilePath $LogFile -Encoding utf8 -Append
if (Test-Path -LiteralPath $SecretsEnv) {
    "Secrets file present: $SecretsEnv" | Out-File -FilePath $LogFile -Encoding utf8 -Append
}
"Args: $($Args -join ' ')" | Out-File -FilePath $LogFile -Encoding utf8 -Append

if (-not $env:ALPACA_API_KEY -or -not $env:ALPACA_API_SECRET) {
    "ERROR: Missing ALPACA_API_KEY / ALPACA_API_SECRET. Set them as machine env vars or create $SecretsEnv" | Out-File -FilePath $LogFile -Encoding utf8 -Append
    exit 1
}

# Capture Python stdout+stderr to the log (cmd redirection is the most reliable across PS versions)
$ArgLine = $Args -join ' '
$CmdLine = "`"$PythonExe`" $ArgLine >> `"$LogFile`" 2>&1"
& cmd.exe /c $CmdLine
