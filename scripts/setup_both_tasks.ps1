# Run this script once to register BOTH traders as scheduled tasks.
# Both run as SYSTEM -- no password needed, works when logged off.
# Self-elevates to admin.

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

$ProjectRoot = 'C:\Users\yalil\OneDrive\Desktop\AI-projects\quantitivestocks'

Write-Host ""
Write-Host "  Registering BOTH scheduled tasks..." -ForegroundColor Cyan
Write-Host "  (Run as SYSTEM -- no password needed, works when logged off)" -ForegroundColor Gray
Write-Host ""

$optCmd   = Join-Path $ProjectRoot 'run_options_trade.cmd'
$paperCmd = Join-Path $ProjectRoot 'run_paper_trade.cmd'

# --- Options trader: Mon-Fri 6:25 AM ---
$out = schtasks /create /tn "QuantitativeStocks-OptionsTrader-Weekdays" `
    /tr "`"$optCmd`"" `
    /sc weekly /d MON,TUE,WED,THU,FRI /st 06:25 `
    /ru SYSTEM `
    /rl HIGHEST `
    /f 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Options trader : Mon-Fri 6:25 AM" -ForegroundColor Green
} else {
    Write-Host "  FAILED to register options trader:" -ForegroundColor Red
    Write-Host "  $out" -ForegroundColor Red
    Read-Host "  Press Enter to close"
    exit 1
}

# --- Paper trader: Mon-Fri 9:25 AM ---
$out = schtasks /create /tn "QuantitativeStocks-PaperTrader-Weekdays" `
    /tr "`"$paperCmd`"" `
    /sc weekly /d MON,TUE,WED,THU,FRI /st 09:25 `
    /ru SYSTEM `
    /rl HIGHEST `
    /f 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Paper trader   : Mon-Fri 9:25 AM" -ForegroundColor Green
} else {
    Write-Host "  FAILED to register paper trader:" -ForegroundColor Red
    Write-Host "  $out" -ForegroundColor Red
    Read-Host "  Press Enter to close"
    exit 1
}

Write-Host ""
Write-Host "  Done! Both tasks run as SYSTEM -- no login required." -ForegroundColor Green
Write-Host ""
Read-Host "  Press Enter to close"
