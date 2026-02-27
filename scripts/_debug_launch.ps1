$ProjectRoot = "C:\Users\yalil\OneDrive\Desktop\AI-projects\quantitivestocks"
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$LogFile = Join-Path $ProjectRoot 'logs\debug_launch.log'

# Load env
$EnvFile = Join-Path $ProjectRoot 'settings\alpaca.env'
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line.Length -eq 0 -or $line.StartsWith('#')) { return }
    $idx = $line.IndexOf('=')
    if ($idx -lt 1) { return }
    $name = $line.Substring(0, $idx).Trim()
    $value = $line.Substring($idx + 1).Trim()
    if ($value.StartsWith('"') -and $value.EndsWith('"') -and $value.Length -ge 2) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    Set-Item -Path ("Env:" + $name) -Value $value
}

"ALPACA_API_KEY set: $($env:ALPACA_API_KEY -ne $null -and $env:ALPACA_API_KEY.Length -gt 0)" | Out-File $LogFile
"ALPACA_SECRET set: $($env:ALPACA_API_SECRET -ne $null -and $env:ALPACA_API_SECRET.Length -gt 0)" | Out-File $LogFile -Append

Set-Location $ProjectRoot

# Run directly and capture output
"--- Running paper trader test ---" | Out-File $LogFile -Append
$result = & $PythonExe -u main.py trade --provider alpaca --mode intraday --interval 5min --confidence 0.2 --trailing-stop 0.05 --take-profit 0.08 2>&1
$result | Out-File $LogFile -Append
"--- Exit ---" | Out-File $LogFile -Append

Write-Host "Done. Check $LogFile"
