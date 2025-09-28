@echo off
REM Sound System Installation Script for CCTV System (Windows)
REM Installs espeak-ng with Gujarati support

echo 🔊 Installing Sound System for CCTV Security System
echo ==================================================

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ❌ Please do not run this script as administrator
    pause
    exit /b 1
)

echo 📦 Detected Windows system
echo.

REM Check if espeak-ng is already installed
where espeak-ng >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ espeak-ng is already installed
    goto :test_installation
)

echo 📥 Installing espeak-ng for Windows...
echo.

REM Download espeak-ng for Windows
echo Downloading espeak-ng Windows binary...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-1.51-win64.zip' -OutFile 'espeak-ng.zip'}"

if not exist espeak-ng.zip (
    echo ❌ Failed to download espeak-ng
    echo Please download manually from: https://github.com/espeak-ng/espeak-ng/releases
    pause
    exit /b 1
)

echo ✅ Downloaded espeak-ng.zip

REM Extract espeak-ng
echo Extracting espeak-ng...
powershell -Command "Expand-Archive -Path 'espeak-ng.zip' -DestinationPath '.' -Force"

REM Move to system PATH
echo Installing to system PATH...
if not exist "C:\espeak-ng" mkdir "C:\espeak-ng"
xcopy /E /I /Y "espeak-ng-1.51-win64\*" "C:\espeak-ng\"

REM Add to PATH
echo Adding to system PATH...
setx PATH "%PATH%;C:\espeak-ng" /M

REM Clean up
del espeak-ng.zip
rmdir /S /Q espeak-ng-1.51-win64

echo ✅ espeak-ng installed successfully

:test_installation
echo.
echo 🧪 Testing espeak-ng installation...

REM Test if espeak-ng is available
where espeak-ng >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ espeak-ng not found in PATH
    echo Please restart your command prompt and try again
    pause
    exit /b 1
)

echo ✅ espeak-ng is available

REM Test Gujarati voice
echo 🎤 Testing Gujarati female voice...
espeak-ng -v gu+f55 -s 163 "Hello World" --stdout > test_audio.wav

if exist test_audio.wav (
    echo ✅ Voice test successful - audio file created
    del test_audio.wav
) else (
    echo ⚠️ Voice test failed, but espeak-ng is installed
)

echo.
echo 🎉 Sound system installation completed!
echo.
echo 🔊 Available voices:
espeak-ng --voices | findstr gu
echo.
echo 🎛️ Voice parameters:
echo   Language: Gujarati (gu)
echo   Speed: 163
echo   Pitch: 55 (female cute girl voice)
echo   Volume: 100
echo.
echo 🧪 Test the sound system:
echo   python src\sound_system.py
echo.
echo 📝 Integration with CCTV system:
echo   The sound system is ready to be integrated with the CCTV system
echo   Use: from src.sound_system import get_sound_system, get_messages
echo.
pause
