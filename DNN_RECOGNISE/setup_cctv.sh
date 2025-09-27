#!/bin/bash

# Raspberry Pi CCTV System Setup Script
echo "🚀 Raspberry Pi CCTV System Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}❌ Please do not run this script as root${NC}"
    exit 1
fi

# Update system
echo -e "${YELLOW}📦 Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo -e "${YELLOW}🔧 Installing system dependencies...${NC}"
sudo apt install -y python3-pip python3-opencv bluetooth bluez espeak-ng festival
sudo apt install -y libatlas-base-dev libopenjp2-7 libtiff5
# Install additional festival voices for better female voice support
sudo apt install -y festvox-kallpc16k

# Install Python packages
echo -e "${YELLOW}🐍 Installing Python packages...${NC}"
pip3 install --upgrade pip --break-system-packages
pip3 install -r requirements_cctv.txt --break-system-packages

# Enable camera interface
echo -e "${YELLOW}📷 Enabling camera interface...${NC}"
sudo raspi-config nonint do_camera 0

# Setup directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p models
mkdir -p images/arcface_format
mkdir -p recordings
mkdir -p unknown_faces
mkdir -p sounds

# Download models if they don't exist
echo -e "${YELLOW}🤖 Downloading AI models...${NC}"

# YOLOv8 model
if [ ! -f "models/yolov8n.onnx" ]; then
    echo "Downloading YOLOv8 person detection model..."
    wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx -O models/yolov8n.onnx
fi

# SCRFD model
if [ ! -f "models/scrfd_10g_bnkps.onnx" ]; then
    echo "Downloading SCRFD face detection model..."
    wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.onnx -O models/scrfd_10g_bnkps.onnx
fi

# ArcFace model
if [ ! -f "models/arcface_r100.onnx" ]; then
    echo "Downloading ArcFace recognition model..."
    wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100.onnx -O models/arcface_r100.onnx
fi

echo -e "${GREEN}✅ Models downloaded${NC}"

# Setup GPIO permissions
echo -e "${YELLOW}🔌 Setting up GPIO permissions...${NC}"
sudo usermod -a -G gpio $USER
sudo usermod -a -G video $USER

# Create sample configuration
echo -e "${YELLOW}⚙️ Creating sample configuration...${NC}"
if [ ! -f "src/settings/local_settings.py" ]; then
    cat > src/settings/local_settings.py << 'EOF'
# Local hardware configuration
# Edit these values according to your hardware setup

HARDWARE = {
    'pir_pin': 17,  # GPIO pin for PIR motion sensor
    'led_brightness_pin': 18,  # GPIO pin for high brightness LED
    'led_green_pin': 22,  # GPIO pin for green status LED
    'led_yellow_pin': 23,  # GPIO pin for yellow status LED
    'led_red_pin': 24,  # GPIO pin for red status LED
    'bt_speaker_mac': 'XX:XX:XX:XX:XX:XX',  # Your Bluetooth speaker MAC
    'bt_speaker_name': 'CCTV_Speaker',  # Your Bluetooth speaker name
}

# Voice configuration for female voice
VOICE = {
    'engine': 'espeak-ng',  # 'espeak-ng', 'festival', or 'auto'
    'gender': 'female',  # 'female' or 'male'
    'espeak_voice': 'en+f3',  # Female voice options: en+f1, en+f2, en+f3, en+f4
    'festival_voice': 'cmu_us_slt_cg',  # Female voice for festival
    'speech_rate': 150,  # Words per minute
    'pitch': 50,  # Pitch (0-100, higher = more feminine)
    'volume': 80,  # Volume (0-100)
}

# Get your Bluetooth speaker MAC address:
# 1. Run: bluetoothctl devices
# 2. Find your speaker and copy the MAC address
# 3. Replace XX:XX:XX:XX:XX:XX above
EOF
    echo -e "${GREEN}✅ Sample configuration created${NC}"
    echo -e "${YELLOW}⚠️ Please edit src/settings/local_settings.py with your hardware configuration${NC}"
fi

# Create startup script
echo -e "${YELLOW}🚀 Creating startup script...${NC}"
cat > start_cctv.sh << 'EOF'
#!/bin/bash
echo "Starting Raspberry Pi CCTV System..."
cd "$(dirname "$0")"
python3 src/cctv_system.py
EOF

chmod +x start_cctv.sh

# Test TTS with female voice
echo -e "${YELLOW}🗣️ Testing female voice TTS...${NC}"
echo "Testing espeak-ng female voice..."
echo "Hello, this is a test of the female voice for CCTV system" | espeak-ng -v en+f3 -s 150 -p 50

echo ""
echo "Testing festival female voice..."
echo '(voice_cmu_us_slt_cg)' > /tmp/test_tts
echo '(SayText "Hello, this is a test of the female voice for CCTV system")' >> /tmp/test_tts
echo "" >> /tmp/test_tts
festival -b /tmp/test_tts 2>/dev/null || echo "Festival test completed"

# Test camera
echo -e "${YELLOW}📷 Testing camera...${NC}"
python3 -c "
try:
    from picamera2 import Picamera2
    import time
    picam2 = Picamera2()
    picam2.start()
    time.sleep(2)  # Allow camera to warm up
    frame = picam2.capture_array()
    if frame is not None:
        from PIL import Image
        img = Image.fromarray(frame)
        img.save('camera_test.jpg')
        print('✅ Camera test successful')
        print('📸 Test image saved as camera_test.jpg')
    else:
        print('❌ Camera test failed - no frame received')
    picam2.close()
except ImportError:
    print('❌ picamera2 module not found. Please install with: pip install picamera2')
except Exception as e:
    print(f'❌ Camera test failed: {e}')
"

# Setup complete
echo ""
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit src/settings/settings.py with your hardware configuration"
echo "2. Get your Bluetooth speaker MAC: bluetoothctl devices"
echo "3. Run face training: python3 src/face_taker.py (then python3 src/face_trainer.py)"
echo "4. Start CCTV system: ./start_cctv.sh"
echo ""
echo "For detailed instructions, see CCTV_README.md"
echo ""
echo -e "${YELLOW}🗣️ Female Voice Configuration:${NC}"
echo "  • Primary TTS: espeak-ng with female voice (en+f3)"
echo "  • Fallback TTS: festival with female voice (cmu_us_slt_cg)"
echo "  • Speech rate: 150 words per minute"
echo "  • Pitch: 50 (optimized for female voice)"
echo ""
echo -e "${YELLOW}⚠️ Remember to configure your hardware settings before running the system!${NC}"
echo -e "${YELLOW}💡 The system will now speak all messages in a natural female voice!${NC}"
