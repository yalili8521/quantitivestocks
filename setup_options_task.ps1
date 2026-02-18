# Run this script once to register the options trader as a scheduled task.
# It will self-elevate to admin if needed.

param([switch]$Elevated)

function Test-Admin {
    $identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal] $identity
    $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    # Re-launch as administrator
    Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Elevated" -Verb RunAs
    exit
}

Write-Host ""
Write-Host "  Registering QuantitativeStocks-OptionsTrader-Weekdays..." -ForegroundColor Cyan

$CmdPath = 'C:\Users\yalil\OneDrive\Desktop\AI-projects\quantitivestocks\run_options_trade.cmd'

$action = New-ScheduledTaskAction -Execute $CmdPath

$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At '09:25AM'

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 72) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -WakeToRun:$true

$principal = New-ScheduledTaskPrincipal `
    -UserId 'yalil' `
    -LogonType S4U `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName 'QuantitativeStocks-OptionsTrader-Weekdays' `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force | Out-Null

Write-Host "  Done! Task registered." -ForegroundColor Green
Write-Host ""
Write-Host "  Schedule : Mon-Fri at 9:25 AM"
Write-Host "  Runs     : automatically, no login required"
Write-Host "  Log      : logs\options_trader_YYYYMMDD_HHmmss.log"
Write-Host ""
Read-Host "  Press Enter to close"
