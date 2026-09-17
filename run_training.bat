@echo off
REM Starting/Resuming Large-Scale JetClass Training...
REM You can stop this safely at any time by pressing Ctrl+C.
REM The script uses --resume, so it will automatically pick up from the last checkpoint when restarted.
echo Starting/Resuming Large-Scale JetClass Training...
echo You can stop this safely at any time by pressing Ctrl+C.
echo The script uses --resume, so it will automatically pick up from the last checkpoint when restarted.
echo.

if not defined PYTHONPATH set PYTHONPATH=.
if exist .\.venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe experiments\train_jetclass.py --large --arch edgeconv --save-steps 5000 --resume
) else (
    echo [INFO] .venv not found, falling back to system python.
    python experiments\train_jetclass.py --large --arch edgeconv --save-steps 5000 --resume
)
if errorlevel 1 (
    echo [ERROR] Training exited with code %errorlevel%.
    pause
    exit /b %errorlevel%
)

echo.
pause
