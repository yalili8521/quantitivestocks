Get-ChildItem 'C:\Users\yalil\OneDrive\Desktop\AI-projects\quantitivestocks\logs\' |
    Where-Object { $_.Name -like 'paper_trader__*' -or $_.Name -like 'paper_trader_2026022[34]*' } |
    Select-Object Name, LastWriteTime, Length |
    Format-Table -AutoSize
