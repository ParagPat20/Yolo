#!/bin/bash
# Sound System Installation Script for CCTV System
# Installs espeak-ng with Gujarati support

echo "🔊 Installing Sound System for CCTV Security System"
echo "=================================================="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ Please do not run this script as root"
   exit 1
fi

# Detect OS and install espeak-ng
if command -v apt &> /dev/null; then
    echo "📦 Detected Ubuntu/Debian system"
    echo "Installing espeak-ng..."
    sudo apt update
    sudo apt install -y espeak-ng espeak-ng-data-gujarati
    echo "✅ espeak-ng installed successfully"
    
elif command -v yum &> /dev/null; then
    echo "📦 Detected CentOS/RHEL system"
    echo "Installing espeak-ng..."
    sudo yum install -y espeak-ng
    echo "✅ espeak-ng installed successfully"
    
elif command -v brew &> /dev/null; then
    echo "📦 Detected macOS system"
    echo "Installing espeak-ng..."
    brew install espeak-ng
    echo "✅ espeak-ng installed successfully"
    
else
    echo "❌ Unsupported operating system"
    echo "Please install espeak-ng manually:"
    echo "  Ubuntu/Debian: sudo apt install espeak-ng espeak-ng-data-gujarati"
    echo "  CentOS/RHEL: sudo yum install espeak-ng"
    echo "  macOS: brew install espeak-ng"
    exit 1
fi

# Test installation
echo ""
echo "🧪 Testing espeak-ng installation..."
if command -v espeak-ng &> /dev/null; then
    echo "✅ espeak-ng is available"
    
    # Test Gujarati voice
    echo "🎤 Testing Gujarati female voice..."
    espeak-ng -v gu+f55 -s 163 "હેલો વર્લ્ડ" --stdout | aplay 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅ Gujarati voice test successful"
    else
        echo "⚠️ Voice test failed, but espeak-ng is installed"
    fi
    
else
    echo "❌ espeak-ng installation failed"
    exit 1
fi

# Install aplay if not available (for audio playback)
if ! command -v aplay &> /dev/null; then
    echo "📦 Installing aplay for audio playback..."
    if command -v apt &> /dev/null; then
        sudo apt install -y alsa-utils
    elif command -v yum &> /dev/null; then
        sudo yum install -y alsa-utils
    fi
fi

echo ""
echo "🎉 Sound system installation completed!"
echo ""
echo "🔊 Available voices:"
espeak-ng --voices | grep gu
echo ""
echo "🎛️ Voice parameters:"
echo "  Language: Gujarati (gu)"
echo "  Speed: 163"
echo "  Pitch: 55 (female cute girl voice)"
echo "  Volume: 100"
echo ""
echo "🧪 Test the sound system:"
echo "  python3 src/sound_system.py"
echo ""
echo "📝 Integration with CCTV system:"
echo "  The sound system is ready to be integrated with the CCTV system"
echo "  Use: from src.sound_system import get_sound_system, get_messages"
