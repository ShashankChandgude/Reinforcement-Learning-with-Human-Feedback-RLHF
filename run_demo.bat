@echo off
echo 🚀 RLHF Demo Launcher for Windows
echo ===============================================
echo.
echo ⚠️  Windows Security Note:
echo    If Windows Defender shows "Potentially Unwanted App", click "Allow"
echo    This is normal for local web servers - Gradio is safe
echo.
echo 📱 Demo will open at: http://localhost:7860
echo 🛑 Press Ctrl+C to stop when done
echo.
pause
echo.
echo Starting demo...
python run_demo.py
pause
