@echo off
setlocal

set "ROOT=C:\Users\yalil\OneDrive\Desktop\AI-projects\quantitivestocks"
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
set "LOGDIR=%ROOT%\logs"
set "ENVFILE=%ROOT%\settings\alpaca.env"

cd /d "%ROOT%"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do if not "%%I"=="" set "DT=%%I"
set "LOGFILE=%LOGDIR%\paper_trader_%DT:~0,8%_%DT:~8,6%.log"

if exist "%ENVFILE%" (
    for /f "usebackq tokens=1* delims==" %%A in (`findstr /v /r "^#" "%ENVFILE%"`) do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

echo [%DATE% %TIME%] Starting paper trader... >> "%LOGFILE%"
echo Python: %PYTHON% >> "%LOGFILE%"
echo Log: %LOGFILE% >> "%LOGFILE%"
echo Args: -u main.py trade --provider alpaca --mode intraday --interval 5min --confidence 0.2 --trailing-stop 0.05 --take-profit 0.08 >> "%LOGFILE%"

for /f "tokens=2" %%P in ('wmic process where "name=''python.exe'' and commandline like ''%%main.py trade%%''" get processid /format:list 2^>nul ^| findstr "="') do taskkill /PID %%P /F >nul 2>&1

timeout /t 2 /nobreak >nul

"%PYTHON%" -u main.py trade --provider alpaca --mode intraday --interval 5min --confidence 0.2 --trailing-stop 0.05 --take-profit 0.08 >> "%LOGFILE%" 2>&1
