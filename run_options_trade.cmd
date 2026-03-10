@echo off
REM Options trader has been removed from this codebase.
REM This stub prevents the scheduled task from erroring.
echo [%date% %time%] Options trader disabled (module removed). >> "%~dp0logs\options_disabled.log"
exit /b 0
