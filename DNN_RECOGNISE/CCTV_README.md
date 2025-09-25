# Raspberry Pi CCTV Camera System

## Advanced Person Tracking with Face Recognition & Smart Security

This is a comprehensive CCTV surveillance system for Raspberry Pi that combines:
- **Person Detection & Tracking** using YOLOv8
- **Face Detection** using SCRFD
- **Face Recognition** using ArcFace
- **Motion Detection** with PIR sensor
- **LED Status Indicators** for visual feedback
- **Bluetooth Audio Alerts** with **Female Voice** for smart greetings and alarms
- **Video Recording** of unknown persons
- **Time-based Greetings** (morning, afternoon, evening)

## 🗣️ Female Voice Features

The system now speaks all messages in a **natural female voice** using:
- **Primary TTS**: espeak-ng with optimized female voice (`en+f3`)
- **Fallback TTS**: Festival with female voice (`cmu_us_slt_cg`)
- **Configurable Settings**: Voice type, pitch, speed, and volume
- **Multiple Voice Options**: Choose from different female voice variants

## Hardware Requirements

### Required Components
- **Raspberry Pi 5** (recommended) or Raspberry Pi 4
- **Arducam 64MP Camera** (or compatible camera)
- **PIR Motion Sensor**
- **High Brightness LED** (for night illumination)
- **3x Status LEDs** (Green, Yellow, Red)
- **Bluetooth Speaker** (already paired)

### GPIO Pin Configuration

```python
# Update these in src/settings/settings.py
HARDWARE = {
    'pir_pin': 17,              # GPIO pin for PIR motion sensor
    'led_brightness_pin': 18,   # GPIO pin for high brightness LED
    'led_green_pin': 22,        # GPIO pin for green status LED
    'led_yellow_pin': 23,       # GPIO pin for yellow status LED
    'led_red_pin': 24,          # GPIO pin for red status LED
    'bt_speaker_mac': 'XX:XX:XX:XX:XX:XX',  # Bluetooth speaker MAC
}
```

## Software Requirements

### Installation

1. **Install system dependencies:**
```bash
sudo apt update
sudo apt install python3-pip python3-opencv bluetooth bluez espeak-ng festival
sudo apt install libatlas-base-dev libopenjp2-7 libtiff5
```

2. **Install Python packages:**
```bash
pip install -r requirements_cctv.txt
```

3. **Enable camera:**
```bash
sudo raspi-config
# Enable Camera interface
```

4. **Setup Bluetooth audio:**
```bash
bluetoothctl
# Pair your speaker (already done according to your setup)
```

## Voice Configuration

### Female Voice Settings

```python
# Configure in src/settings/settings.py
VOICE = {
    'engine': 'espeak-ng',  # 'espeak-ng', 'festival', or 'auto'
    'gender': 'female',  # 'female' or 'male'
    'espeak_voice': 'en+f3',  # Female voice options: en+f1, en+f2, en+f3, en+f4
    'festival_voice': 'cmu_us_slt_cg',  # Female voice for festival
    'speech_rate': 150,  # Words per minute (higher = faster)
    'pitch': 50,  # Pitch (0-100, 50 = natural female)
    'volume': 80,  # Volume (0-100)
}
```

### Available Female Voices

- **espeak-ng Female Voices**:
  - `en+f1`: Female voice 1 (clear)
  - `en+f2`: Female voice 2 (soft)
  - `en+f3`: Female voice 3 (natural) ← **Recommended**
  - `en+f4`: Female voice 4 (bright)

- **Festival Female Voice**:
  - `cmu_us_slt_cg`: US female voice (backup)

## Model Setup

### Download Required Models

1. **YOLOv8 Person Detection Model:**
```bash
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx -O models/yolov8n.onnx
```

2. **SCRFD Face Detection Model:**
```bash
wget https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.onnx -O models/scrfd_10g_bnkps.onnx
```

3. **ArcFace Recognition Model:**
```bash
wget https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100.onnx -O models/arcface_r100.onnx
```

4. **OpenCV Face Detection Models:**
```bash
# These should already be available in your OpenCV installation
# If not, download from OpenCV GitHub
```

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Hardware      │    │   CCTV System    │    │  Face Training  │
│   Interface     │    │   🗣️ Female Voice│    │                 │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • PIR Sensor    │    │ • Person Tracker │    │ • Face Taker    │
│ • LEDs          │    │ • Face Detector  │    │ • Face Trainer  │
│ • BT Speaker    │    │ • Face Recognizer│    │ • Data Manager  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌──────────────────┐
                    │   Smart Alerts   │
                    │   🗣️ Female TTS   │
                    │                  │
                    │ • Motion Detect  │
                    │ • LED Control    │
                    │ • Audio Alerts   │
                    │ • Time Greetings │
                    └──────────────────┘
```

## Key Features

### 1. Smart Person Verification
- **4-Second Rule**: Unknown persons get 4 seconds to show face before alert
- **8-Second Timeout**: Maximum time for face verification
- **3-Attempt System**: Multiple chances for recognition before marking as unknown
- **Trusted Memory**: Remembers verified persons for 5 minutes

### 2. Enhanced Face Recognition
- **ArcFace Model**: State-of-the-art face recognition
- **SCRFD Detection**: High-accuracy face detection
- **Quality Filtering**: Ensures only good quality faces are used for training
- **Outlier Removal**: Removes poor quality training samples

### 3. Audio Integration with Female Voice
- **Time-based Greetings** (in natural female voice):
  - Morning: "Good morning, [Name]!"
  - Afternoon: "Good afternoon, [Name]!"
  - Evening: "Good evening, [Name]!"
- **Smart Alerts** (in natural female voice): "Unknown person detected in the area"
- **Verification Requests** (in natural female voice): "Please look at the camera for verification"
- **Welcome Back**: "Welcome back, [Name]!"

### 4. LED Status System
- **Green LED**: System ready / Person verified
- **Yellow LED**: Verifying person / Processing
- **Red LED**: Alert / Unknown person detected
- **Brightness LED**: Auto-illumination when motion detected

### 5. Recording System
- **Automatic Recording**: Unknown persons are automatically recorded
- **60-Second Duration**: Records for 1 minute when unknown detected
- **High Quality**: 1280x720 resolution at 30 FPS
- **Organized Storage**: Saved in `recordings/` directory

## Usage Guide

### 1. Initial Setup

1. **Configure Hardware Settings:**
```python
# Edit src/settings/settings.py
HARDWARE = {
    'pir_pin': 17,  # Your PIR sensor pin
    'led_brightness_pin': 18,  # Your brightness LED pin
    'led_green_pin': 22,  # Your green LED pin
    'led_yellow_pin': 23,  # Your yellow LED pin
    'led_red_pin': 24,  # Your red LED pin
    'bt_speaker_mac': 'YOUR_SPEAKER_MAC',  # Your BT speaker MAC
}

# Configure female voice
VOICE = {
    'engine': 'espeak-ng',  # 'espeak-ng', 'festival', or 'auto'
    'gender': 'female',  # 'female' or 'male'
    'espeak_voice': 'en+f3',  # Female voice options: en+f1, en+f2, en+f3, en+f4
    'festival_voice': 'cmu_us_slt_cg',  # Female voice for festival
    'speech_rate': 150,  # Words per minute (150 = natural speed)
    'pitch': 50,  # Pitch (50 = natural female voice)
    'volume': 80,  # Volume level
}
```

2. **Update Bluetooth Speaker MAC:**
```bash
bluetoothctl devices
# Find your speaker's MAC address and update in settings
```

### 2. Face Training

1. **Collect Training Data:**
```bash
cd src
python face_taker.py
```

2. **Train the Models:**
```bash
python face_trainer.py
```

### 3. Run CCTV System

```bash
cd src
python cctv_system.py
```

### 4. Controls

- **ESC/Q**: Exit system
- **S**: Save current frame snapshot
- **R**: Reset all person tracks
- **M**: Toggle motion detection (if implemented)

## System Workflow

### Person Detection Flow

1. **Motion Detection**: PIR sensor detects movement
2. **Person Tracking**: YOLOv8 detects and tracks persons
3. **Face Request**: System asks person to show face (4 seconds)
4. **Face Detection**: SCRFD detects faces in person bounding box
5. **Recognition**: ArcFace recognizes the face
6. **Verification**: Multiple attempts if recognition fails
7. **Action**:
   - ✅ **Known Person**: Greet with time-based message + green LED
   - ❌ **Unknown Person**: Alert + record video + red LED + alarm sound

### Security Scenarios

#### Scenario 1: Known Person
```
Motion Detected → Person Tracked → Face Shown → Recognized ✅
→ LED: Green → 🗣️ Audio: "Good morning, John!" → No Recording
```

#### Scenario 2: Unknown Person (Cooperative)
```
Motion Detected → Person Tracked → Face Shown → Not Recognized ❌
→ LED: Yellow (verifying) → 🗣️ Audio: "Please look at camera"
→ After 3 attempts: LED: Red → 🗣️ Audio: "Unknown person detected!"
→ Start Recording → Save Unknown Face
```

#### Scenario 3: Unknown Person (Non-cooperative)
```
Motion Detected → Person Tracked → No Face Shown (4s timeout)
→ LED: Red → 🗣️ Audio: "Unknown person detected!"
→ Start Recording → Alert Activated
```

## File Structure

```
raspberry-pi-cctv/
├── src/
│   ├── cctv_system.py          # Main CCTV system
│   ├── advanced_person_tracker.py  # Person tracking & recognition
│   ├── face_taker.py           # Face data collection
│   ├── face_trainer.py         # Model training
│   ├── hardware_interface.py   # GPIO & hardware control
│   └── settings/
│       └── settings.py         # Configuration
├── models/                     # AI models
│   ├── yolov8n.onnx           # Person detection
│   ├── scrfd_10g_bnkps.onnx   # Face detection
│   ├── arcface_r100.onnx      # Face recognition
│   └── face_embeddings.pkl    # Trained embeddings
├── images/                    # Training data
│   └── arcface_format/        # Processed face data
├── recordings/                # Video recordings
├── unknown_faces/             # Unknown person images
├── sounds/                    # Audio files (if any)
├── requirements_cctv.txt      # Dependencies
└── CCTV_README.md             # This file
```

## Configuration Options

### Voice Settings

```python
VOICE = {
    'engine': 'espeak-ng',  # 'espeak-ng', 'festival', or 'auto'
    'gender': 'female',  # 'female' or 'male'
    'espeak_voice': 'en+f3',  # Female voice options: en+f1, en+f2, en+f3, en+f4
    'festival_voice': 'cmu_us_slt_cg',  # Female voice for festival
    'speech_rate': 150,  # Words per minute (150 = natural speed)
    'pitch': 50,  # Pitch (50 = natural female voice)
    'volume': 80,  # Volume level
}
```

### Available Voice Options

| Voice Engine | Voice Code | Description | Quality |
|-------------|------------|-------------|---------|
| espeak-ng | `en+f1` | Female voice 1 | ⭐⭐⭐ |
| espeak-ng | `en+f2` | Female voice 2 | ⭐⭐⭐⭐ |
| espeak-ng | `en+f3` | Female voice 3 | ⭐⭐⭐⭐⭐ ← **Recommended** |
| espeak-ng | `en+f4` | Female voice 4 | ⭐⭐⭐ |
| Festival | `cmu_us_slt_cg` | US Female | ⭐⭐⭐⭐ |

### CCTV Settings (src/settings/settings.py)

```python
CCTV = {
    'motion_detection_enabled': True,      # Enable PIR sensor
    'motion_cooldown': 5.0,               # Motion detection cooldown
    'led_auto_brightness': True,          # Auto-control brightness LED
    'recording_enabled': True,            # Enable unknown person recording
    'recording_duration': 60.0,           # Recording time (seconds)
    'greeting_enabled': True,             # Enable time-based greetings
    'verification_timeout': 8.0,          # Face verification timeout
    'unknown_timeout': 4.0,               # Unknown person timeout
    'max_verification_attempts': 3,       # Max recognition attempts
}
```

### Audio Settings

```python
AUDIO = {
    'greeting_morning': 'Good morning',
    'greeting_afternoon': 'Good afternoon',
    'greeting_evening': 'Good evening',
    'unknown_alert': 'Alert! Unknown person detected in the area',
    'verification_request': 'Please look at the camera for verification',
    'welcome_back': 'Welcome back',
    'greeting_sound_enabled': True,       # Play audio greetings
    'alert_sound_enabled': True,          # Play alert sounds
}
```

## Troubleshooting

### TTS/Voice Issues

1. **No sound from speaker:**
   - Check Bluetooth connection: `bluetoothctl info [MAC]`
   - Ensure speaker is paired and trusted
   - Test manually: `echo "Test message" | espeak-ng -v en+f3`

2. **Voice sounds robotic:**
   - Try different female voice: `en+f3` is most natural
   - Adjust pitch: Lower values (40-60) sound more natural
   - Adjust speed: 140-160 words/minute is optimal

3. **Festival not working:**
   - Install festival voices: `sudo apt install festvox-kallpc16k`
   - Check festival installation: `festival --version`

4. **espeak-ng not found:**
   - Install: `sudo apt install espeak-ng`
   - Test: `espeak-ng -v en+f3 "Hello world"`

### Common Issues

1. **Camera not detected:**
   - Check camera connection
   - Run `sudo raspi-config` and enable camera
   - Verify camera module is loaded: `lsmod | grep v4l2`

2. **GPIO errors:**
   - Run with sudo: `sudo python cctv_system.py`
   - Check GPIO pin numbers in settings
   - Verify LED connections

3. **Bluetooth speaker not connecting:**
   - Check MAC address in settings
   - Ensure speaker is paired and trusted
   - Test connection: `bluetoothctl connect [MAC]`

4. **Face recognition not working:**
   - Check model files exist in models/ directory
   - Verify training data quality
   - Retrain models: `python face_trainer.py`

5. **Poor performance:**
   - Reduce camera resolution in settings
   - Use smaller models (YOLOv8 nano)
   - Close other applications

### Performance Optimization

1. **Faster Speech:**
```python
VOICE = {
    'speech_rate': 180,  # Faster speech
    'pitch': 45,  # Slightly lower pitch
}
```

2. **Slower, Clearer Speech:**
```python
VOICE = {
    'speech_rate': 120,  # Slower speech
    'pitch': 55,  # Slightly higher pitch
}
```

3. **Reduce Resolution:**
```python
CAMERA = {
    'width': 640,    # Lower resolution
    'height': 480,   # Lower height
    'fps': 30,       # Lower FPS
}
```

4. **Use Smaller Models:**
```python
# Use YOLOv8 nano instead of larger models
PATHS = {
    'yolov8_person_model': 'models/yolov8n.onnx',
}
```

5. **Optimize Face Detection:**
```python
FACE_DETECTION = {
    'min_size': (40, 40),    # Smaller minimum face size
    'scale_factor': 1.1,     # Faster detection
}
```

## Security Considerations

1. **Network Security**: Run on isolated network
2. **Physical Security**: Secure Raspberry Pi location
3. **Data Privacy**: Unknown faces saved locally only
4. **Access Control**: Require authentication for configuration
5. **Audit Logging**: All events logged with timestamps

## Advanced Features

### Custom Voice Messages
Edit audio settings to customize greetings and alerts.

### Multiple Camera Support
Extend system to support multiple cameras for larger areas.

### Email Notifications
Add email alerts for unknown person detection.

### Web Interface
Create web dashboard for remote monitoring.

### Mobile App
Develop mobile app for remote CCTV control.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify all hardware connections
3. Test individual components separately
4. Check system logs for error messages
5. Ensure all dependencies are installed

## License

This project is provided as-is for educational and personal use. Please respect privacy laws and regulations when deploying surveillance systems.

---

**🗣️ Your CCTV system now speaks with a natural female voice!** 🎉
