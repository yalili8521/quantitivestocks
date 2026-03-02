# Run this script once to register the options trader as a scheduled task.
# The task will run even when you are NOT logged in (computer must be on and online).
# It will self-elevate to admin and prompt once for your Windows password.

param([switch]$Elevated)

$ProjectRoot = 'C:\Users\yalil\OneDrive\Desktop\AI-projects\quantitivestocks'
$TaskUser    = $env:USERNAME  # e.g. yalil

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
Write-Host "  Registering QuantitativeStocks-OptionsTrader-Weekdays..." -ForegroundColor Cyan
Write-Host "  (Runs whether you are logged in or not)" -ForegroundColor Gray
Write-Host ""

$cred = $null
try {
    $cred = Get-Credential -UserName $TaskUser -Message "Enter your Windows password so the task can run when you're logged off (stored securely by Windows)"
} catch {
    Write-Host "  Cancelled." -ForegroundColor Yellow
    exit 1
}
if (-not $cred) {
    Write-Host "  Cancelled." -ForegroundColor Yellow
    exit 1
}

$password = $cred.GetNetworkCredential().Password
if ([string]::IsNullOrEmpty($password)) {
    Write-Host "  Password is required for run-without-login. Exiting." -ForegroundColor Red
    exit 1
}

$CmdPath = Join-Path $ProjectRoot 'run_options_trade.cmd'

$action = New-ScheduledTaskAction -Execute $CmdPath -WorkingDirectory $ProjectRoot

$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At '06:25AM'

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 72) `
    -MultipleInstances Parallel `
    -StartWhenAvailable `
    -WakeToRun:$true

Register-ScheduledTask `
    -TaskName 'QuantitativeStocks-OptionsTrader-Weekdays' `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -User $TaskUser `
    -Password $password `
    -Force | Out-Null

# Clear password from memory
$password = $null

Write-Host "  Done! Task registered." -ForegroundColor Green
Write-Host ""
Write-Host "  Schedule : Mon-Fri at 6:25 AM (5 min before NYSE open)"
Write-Host "  Runs     : automatically when PC is on — no login required"
Write-Host "  Log      : logs\options_trader_YYYYMMDD_HHmmss.log"
Write-Host ""
Read-Host "  Press Enter to close"
