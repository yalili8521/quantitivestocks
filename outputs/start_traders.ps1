$ErrorActionPreference = 'Stop'
$ProjectRoot = "C:\Users\yalil\OneDrive\Desktop\AI-projects\quantitivestocks"
$PythonExe   = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$LogDir      = Join-Path $ProjectRoot 'logs'

# Kill any existing trader processes
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'main\.py trade' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# Load env file
function Import-DotEnvFile($Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line.Length -eq 0 -or $line.StartsWith('#')) { return }
        $idx = $line.IndexOf('=')
        if ($idx -lt 1) { return }
        $name  = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()
        if ($value.StartsWith('"') -and $value.EndsWith('"') -and $value.Length -ge 2) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}
Import-DotEnvFile (Join-Path $ProjectRoot 'settings\alpaca.env')

Set-Location $ProjectRoot

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'

# --- Paper Trader ---
$paperLog = Join-Path $LogDir "paper_trader_${ts}.log"
$paperProc = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList "-u main.py trade --provider alpaca --mode intraday --interval 5min --confidence 0.2 --trailing-stop 0.05 --take-profit 0.08" `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $paperLog `
    -RedirectStandardError  ($paperLog -replace '\.log$', '_err.log') `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Milliseconds 1000

# --- Options Trader ---
$ts2 = Get-Date -Format 'yyyyMMdd_HHmmss'
$optLog = Join-Path $LogDir "options_trader_${ts2}.log"
$optProc = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList "-u main.py trade-options --symbols SPY,QQQ,IWM,SLV,GLD,XLE,IGV --vix-spike-threshold 15 --max-risk 5000 --confidence 0.30 --check-interval 15" `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $optLog `
    -RedirectStandardError  ($optLog -replace '\.log$', '_err.log') `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Paper trader   PID: $($paperProc.Id)  Log: $paperLog"
Write-Host "Options trader PID: $($optProc.Id)   Log: $optLog"
Write-Host ""
Write-Host "Waiting 15s for startup output..."
Start-Sleep -Seconds 15

Write-Host ""
Write-Host "Paper still running: $(-not $paperProc.HasExited)"
Write-Host "Options still running: $(-not $optProc.HasExited)"
Write-Host ""
Write-Host "=== Paper Trader Log ==="
if (Test-Path $paperLog) { Get-Content $paperLog -Tail 15 }
$paperErrLog = $paperLog -replace '\.log$', '_err.log'
if ((Test-Path $paperErrLog) -and (Get-Item $paperErrLog).Length -gt 0) {
    Write-Host "--- Paper Trader STDERR ---"
    Get-Content $paperErrLog
}

Write-Host ""
Write-Host "=== Options Trader Log ==="
if (Test-Path $optLog) { Get-Content $optLog -Tail 15 }
$optErrLog = $optLog -replace '\.log$', '_err.log'
if ((Test-Path $optErrLog) -and (Get-Item $optErrLog).Length -gt 0) {
    Write-Host "--- Options Trader STDERR ---"
    Get-Content $optErrLog
}
