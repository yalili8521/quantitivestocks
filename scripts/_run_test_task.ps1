$action  = New-ScheduledTaskAction -Execute 'C:\Users\yalil\OneDrive\Desktop\AI-projects\quantitivestocks\outputs\_test_task.cmd'
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
Register-ScheduledTask -TaskName '_PaperTraderTest' -Action $action -Trigger $trigger -RunLevel Highest -Force | Out-Null
Write-Host "Test task registered, waiting 12 seconds..."
Start-Sleep 12
$result = (Get-ScheduledTaskInfo -TaskName '_PaperTraderTest').LastTaskResult
Write-Host "Task result code: $result"
Unregister-ScheduledTask -TaskName '_PaperTraderTest' -Confirm:$false | Out-Null

$logFile = 'C:\Users\yalil\OneDrive\Desktop\AI-projects\quantitivestocks\logs\_task_test.log'
if (Test-Path $logFile) {
    Write-Host "Log created! Contents:"
    Get-Content $logFile
    Remove-Item $logFile -Force
} else {
    Write-Host 'Log NOT created - task may have failed.'
}
