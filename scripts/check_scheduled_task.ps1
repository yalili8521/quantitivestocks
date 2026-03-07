# Check if the paper trader scheduled task is registered and when it runs next.
# Run this to verify automation will fire on weekdays.

$TaskName = "QuantStocks-PaperTrader"

Write-Host ""
Write-Host "  Scheduled task status" -ForegroundColor Cyan
Write-Host "  --------------------" -ForegroundColor Gray
Write-Host ""

$t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $t) {
    Write-Host "  $TaskName : NOT REGISTERED" -ForegroundColor Red
    Write-Host "    -> Run outputs\register_task.ps1 once (as admin) to register." -ForegroundColor Yellow
    Write-Host ""
} else {
    $info = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue
    $state = $t.State
    $next  = $info.NextRunTime
    $last  = $info.LastRunTime
    $result = $info.LastTaskResult

    Write-Host "  $TaskName" -ForegroundColor White
    Write-Host "    State      : $state"
    Write-Host "    Next run   : $next"
    Write-Host "    Last run   : $last"
    Write-Host "    Last result: $result (0 = success)"
    if ($state -ne "Ready") {
        Write-Host "    WARNING: Task is not Ready (e.g. Disabled). It will NOT run until enabled." -ForegroundColor Yellow
    }
    Write-Host ""
}

Write-Host "  Requirements for auto-run:" -ForegroundColor Cyan
Write-Host "    1. Task must be registered (see above) and State = Ready." -ForegroundColor Gray
Write-Host "    2. PC must be ON at 6:25 AM local time (Mon-Fri)." -ForegroundColor Gray
Write-Host "    3. secrets\alpaca.env (or ALPACA_* env vars) must be present." -ForegroundColor Gray
Write-Host ""
