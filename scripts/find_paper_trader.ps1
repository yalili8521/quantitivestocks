# Find paper trader processes and optionally stop the intraday one so user can run with legacy flags
$lockDir = $env:TEMP
$intradayLock = Join-Path $lockDir ".paper_trader_intraday.lock"

# Get all python processes
$pythons = Get-Process -Name python* -ErrorAction SilentlyContinue
$results = @()
foreach ($p in $pythons) {
    try {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)" -ErrorAction Stop).CommandLine
        $results += [PSCustomObject]@{ PID = $p.Id; CommandLine = $cmd }
    } catch {
        $results += [PSCustomObject]@{ PID = $p.Id; CommandLine = "(could not get)" }
    }
}
$outPath = Join-Path $PSScriptRoot "..\outputs\paper_trader_processes.txt"
$results | Format-Table -Wrap -AutoSize | Out-String | Set-Content $outPath -Encoding utf8
Write-Host "Python processes written to $outPath"
$results | Format-Table -Wrap -AutoSize

# If user passed -StopIntraday, try to remove stale lock (only works if process is dead)
if ($args -contains "-StopIntraday") {
    Write-Host "`nTo stop the intraday paper trader, end the process that holds the lock."
    Write-Host "Lock file: $intradayLock"
    Write-Host "Or run: Get-Process -Id <PID> | Stop-Process -Force"
}
