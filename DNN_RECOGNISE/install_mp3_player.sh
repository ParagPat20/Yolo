#!/bin/bash
# Install MP3 player for Linux systems

echo "🔊 Installing MP3 player for CCTV system..."

# Check if mpg123 is already installed
if command -v mpg123 &> /dev/null
then
    echo "✅ mpg123 is already installed"
    mpg123 --version
else
    echo "📦 Installing mpg123..."
    
    # Update package list
    sudo apt-get update
    
    # Install mpg123
    sudo apt-get install -y mpg123
    
    if command -v mpg123 &> /dev/null
    then
        echo "✅ mpg123 installed successfully"
        mpg123 --version
    else
        echo "❌ Failed to install mpg123"
        echo "Please install manually: sudo apt install mpg123"
        exit 1
    fi
fi

echo ""
echo "🧪 Testing MP3 playback..."
if mpg123 --help &> /dev/null
then
    echo "✅ mpg123 is working correctly"
else
    echo "❌ mpg123 test failed"
    exit 1
fi

echo ""
echo "🎵 MP3 player setup complete!"
echo "The CCTV system can now play alarm and verification sounds."
echo ""
echo "📁 Make sure you have these sound files:"
echo "   - sounds/alarm.mp3"
echo "   - sounds/verification_beep.mp3"
