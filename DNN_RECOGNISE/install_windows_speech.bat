@echo off
REM Windows Speech Installation Script for CCTV System
REM Installs pyttsx3 and pywin32 for better Windows speech support

echo 🔊 Installing Windows Speech Support for CCTV System
echo ==================================================

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ❌ Please do not run this script as administrator
    pause
    exit /b 1
)

echo 📦 Installing Windows speech libraries...
echo.

REM Install pyttsx3 (best Windows TTS option)
echo Installing pyttsx3...
pip install pyttsx3

if %errorLevel% neq 0 (
    echo ❌ Failed to install pyttsx3
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo ✅ pyttsx3 installed successfully

REM Install pywin32 (Windows SAPI support)
echo Installing pywin32...
pip install pywin32

if %errorLevel% neq 0 (
    echo ❌ Failed to install pywin32
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo ✅ pywin32 installed successfully

echo.
echo 🧪 Testing Windows speech support...

REM Test pyttsx3
python -c "import pyttsx3; engine = pyttsx3.init(); print('pyttsx3: Available')" 2>nul
if %errorLevel% == 0 (
    echo ✅ pyttsx3 is working
) else (
    echo ⚠️ pyttsx3 test failed
)

REM Test pywin32
python -c "import win32com.client; print('pywin32: Available')" 2>nul
if %errorLevel% == 0 (
    echo ✅ pywin32 is working
) else (
    echo ⚠️ pywin32 test failed
)

echo.
echo 🎉 Windows speech installation completed!
echo.
echo 🔊 Available Windows speech options:
echo   1. pyttsx3 (best option) - Cross-platform TTS
echo   2. Windows SAPI (pywin32) - Native Windows speech
echo   3. winsound (fallback) - Beep sounds only
echo.
echo 🧪 Test the sound system:
echo   python test_sound.py
echo.
echo 📝 The CCTV system will automatically use the best available option:
echo   - espeak-ng (if installed) - Best quality, supports Gujarati
echo   - pyttsx3 (if installed) - Good quality, Windows native
echo   - Windows SAPI (if available) - Windows built-in speech
echo   - winsound (fallback) - Beep sounds only
echo.
pause
