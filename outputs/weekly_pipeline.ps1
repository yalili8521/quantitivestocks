$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python virtual environment not found: $PythonExe"
}

Set-Location -LiteralPath $ProjectRoot

$env:PYTHONUTF8      = '1'
$env:PYTHONIOENCODING = 'utf-8'

# Load env vars (FRED key, Alpaca keys, Slack webhook, etc.)
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

$EnvCandidates = @( (Join-Path $ProjectRoot 'secrets\alpaca.env') )
foreach ($candidate in $EnvCandidates) {
    if (Test-Path -LiteralPath $candidate) {
        Import-DotEnvFile -Path $candidate
        break
    }
}

# Ensure log directory
$LogDir = Join-Path $ProjectRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogFile = Join-Path $LogDir "weekly_pipeline_${ts}.log"

Write-Host "[$(Get-Date -Format s)] Starting weekly pipeline. Log: $LogFile"

& $PythonExe -u scripts/weekly_pipeline.py 2>&1 | Tee-Object -FilePath $LogFile

Write-Host "[$(Get-Date -Format s)] Weekly pipeline finished."
