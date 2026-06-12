@echo off
echo Starting/Resuming Large-Scale JetClass Training...
echo You can stop this safely at any time by pressing Ctrl+C.
echo The script uses --resume, so it will automatically pick up from the last checkpoint when restarted.
echo.

set PYTHONPATH=.
.\.venv\Scripts\python.exe experiments\train_jetclass.py --large --arch edgeconv --save-steps 5000 --resume

echo.
pause
