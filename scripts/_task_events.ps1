$logName = 'Microsoft-Windows-TaskScheduler/Operational'

# Get events for both tasks from the last 24 hours
$start = (Get-Date).AddHours(-24)

Get-WinEvent -LogName $logName -ErrorAction SilentlyContinue |
    Where-Object {
        $_.TimeCreated -ge $start -and
        $_.Message -match 'PaperTrader|OptionsTrader|QuantitativeStocks'
    } |
    Sort-Object TimeCreated |
    ForEach-Object {
        Write-Host "[$($_.TimeCreated.ToString('HH:mm:ss'))] EventID=$($_.Id) Level=$($_.LevelDisplayName)"
        Write-Host "  $($_.Message.Split([Environment]::NewLine)[0])"
        Write-Host ""
    }
