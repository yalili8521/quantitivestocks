# Weekly Pipeline Wrapper — runs weekly_pipeline.py with proper environment
# Scheduled via: QuantStocks-WeeklyRetrain (Sunday 2:00 AM)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Script = Join-Path $ProjectRoot 'scripts\weekly_pipeline.py'
$LogDir = Join-Path $ProjectRoot 'logs'

# Ensure log dir exists
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $LogDir "weekly_pipeline_${ts}.log"
$errFile = Join-Path $LogDir "weekly_pipeline_${ts}_err.log"

# Load environment variables from secrets/alpaca.env
$envFile = Join-Path $ProjectRoot 'secrets\alpaca.env'
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $parts = $line -split '=', 2
            $key = $parts[0].Trim()
            $val = $parts[1].Trim()
            if ($key -and $val) {
                [Environment]::SetEnvironmentVariable($key, $val, 'Process')
            }
        }
    }
}

Write-Host "Starting weekly pipeline at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Log: $logFile"

# Run the pipeline
& $Python -u $Script 2>&1 | Tee-Object -FilePath $logFile

Write-Host "Weekly pipeline completed at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Clean up old pipeline logs (> 30 days)
$oldLogs = Get-ChildItem -Path (Join-Path $LogDir 'weekly_pipeline_*.log') -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) }
if ($oldLogs) {
    $oldLogs | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "Cleaned up $($oldLogs.Count) old pipeline logs"
}
