# Check if the paper trader (and options trader) scheduled tasks are registered and when they run next.
# Run this to verify automation will fire without you logging in.

$PaperTask = "QuantitativeStocks-PaperTrader-Weekdays"
$OptionsTask = "QuantitativeStocks-OptionsTrader-Weekdays"

Write-Host ""
Write-Host "  Scheduled task status" -ForegroundColor Cyan
Write-Host "  --------------------" -ForegroundColor Gray
Write-Host ""

foreach ($TaskName in @($PaperTask, $OptionsTask)) {
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $t) {
        Write-Host "  $TaskName : NOT REGISTERED" -ForegroundColor Red
        Write-Host "    -> Program will NOT run automatically. Run scripts\setup_paper_task.ps1 (or setup_both_tasks.ps1) once to register." -ForegroundColor Yellow
        Write-Host ""
        continue
    }
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

Write-Host "  Requirements for auto-run without login:" -ForegroundColor Cyan
Write-Host "    1. Task must be registered (see above) and State = Ready." -ForegroundColor Gray
Write-Host "    2. PC must be ON at the scheduled time (9:25 AM paper / 6:25 AM options)." -ForegroundColor Gray
Write-Host "    3. If task runs as your user + password: runs when you're logged OFF (PC on)." -ForegroundColor Gray
Write-Host "    4. If task runs as SYSTEM: runs even with no user logged in." -ForegroundColor Gray
Write-Host "    5. secrets\alpaca.env (or ALPACA_* env vars) must be available to the task." -ForegroundColor Gray
Write-Host ""
