param([switch]$Elevated)

function Test-Admin {
    $identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal] $identity
    $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Elevated" -Verb RunAs
    exit
}

Write-Host ""
Write-Host "  Fixing scheduled task settings..." -ForegroundColor Cyan
Write-Host "    - MultipleInstances: IgnoreNew -> Parallel (fresh log each day)"
Write-Host "    - Trigger time:      9:25 AM   -> 6:25 AM  (PST = 5 min before NYSE open)"
Write-Host ""

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 72) `
    -MultipleInstances Parallel `
    -StartWhenAvailable `
    -WakeToRun:$true

$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At '06:25AM'

Set-ScheduledTask -TaskName 'QuantitativeStocks-PaperTrader-Weekdays'  -Settings $settings -Trigger $trigger | Out-Null
Set-ScheduledTask -TaskName 'QuantitativeStocks-OptionsTrader-Weekdays' -Settings $settings -Trigger $trigger | Out-Null

Write-Host "  Done!" -ForegroundColor Green
Write-Host ""

$pt = Get-ScheduledTaskInfo -TaskName 'QuantitativeStocks-PaperTrader-Weekdays'
$ot = Get-ScheduledTaskInfo -TaskName 'QuantitativeStocks-OptionsTrader-Weekdays'
Write-Host ("  PaperTrader   next run : {0}" -f $pt.NextRunTime)
Write-Host ("  OptionsTrader next run : {0}" -f $ot.NextRunTime)
Write-Host ""
Read-Host "  Press Enter to close"
