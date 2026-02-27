$tasks = Get-ScheduledTask | Where-Object { $_.TaskName -match 'QuantitativeStocks' }
foreach ($t in $tasks) {
    Write-Host "=== $($t.TaskName) ==="
    Write-Host "  StartWhenAvailable:    $($t.Settings.StartWhenAvailable)"
    Write-Host "  WakeToRun:             $($t.Settings.WakeToRun)"
    Write-Host "  MultipleInstances:     $($t.Settings.MultipleInstances)"
    Write-Host "  StopIfGoingOnBatteries:$($t.Settings.StopIfGoingOnBatteries)"
    Write-Host ""
}
