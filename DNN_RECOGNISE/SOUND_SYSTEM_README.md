# 🔊 Sound System for CCTV Security System

This sound system provides text-to-speech capabilities using espeak-ng with English and Gujarati female voices for the CCTV security system.

## 🎛️ Voice Parameters

### Gujarati Voice
- **Language**: Gujarati (gu)
- **Speed**: 163 (fast but clear)
- **Pitch**: 55 (female cute girl voice)
- **Volume**: 100
- **Amplitude**: 100

### English Voice
- **Language**: English (en)
- **Speed**: 150 (clear and natural)
- **Pitch**: 50 (female voice)
- **Volume**: 100
- **Amplitude**: 100

## 📦 Installation

### Windows
```bash
# Option 1: Install espeak-ng (best quality, supports Gujarati)
install_sound.bat

# Option 2: Install Windows speech libraries (easier setup)
install_windows_speech.bat
```

### Linux/Ubuntu
```bash
# Run the installation script
chmod +x install_sound.sh
./install_sound.sh
```

### Manual Installation

#### Windows
1. Download espeak-ng from: https://github.com/espeak-ng/espeak-ng/releases
2. Extract to `C:\espeak-ng\`
3. Add `C:\espeak-ng\` to your system PATH
4. Restart command prompt

#### Linux/Ubuntu
```bash
sudo apt update
sudo apt install espeak-ng espeak-ng-data-gujarati
```

#### macOS
```bash
brew install espeak-ng
```

## 🧪 Testing

Run the test script to verify the sound system:

```bash
python test_sound.py
```

## 📝 Usage

### Basic Usage

```python
from src.sound_system import get_sound_system, get_messages

# Get sound system instance (default: Gujarati)
sound = get_sound_system('gu')  # Gujarati
sound = get_sound_system('en')  # English

# Get messages instance
messages = get_messages('gu')   # Gujarati messages
messages = get_messages('en')   # English messages

# Speak custom text
sound.speak("હેલો વર્લ્ડ!")  # Gujarati
sound.speak("Hello World!")   # English

# Use predefined messages
messages.person_detected()
messages.face_verification_request()
messages.known_person_greeting("રાજ")  # Gujarati
messages.known_person_greeting("Raj")   # English
```

### Integration with CCTV System

```python
from src.sound_system import get_sound_system, get_messages

class CCTVSystem:
    def __init__(self, language='gu'):
        self.sound = get_sound_system(language)
        self.messages = get_messages(language)
        self.language = language
    
    def set_language(self, language):
        """Change language dynamically"""
        self.sound.set_language(language)
        self.messages = get_messages(language)
        self.language = language
    
    def on_person_detected(self):
        self.messages.person_detected()
    
    def on_unknown_person(self):
        self.messages.unknown_person_alert()
    
    def on_known_person(self, name):
        self.messages.known_person_greeting(name)
```

## 🌍 Language Selection

The sound system supports both English and Gujarati languages:

### English Messages
- `person_detected()` - "Person detected. Please look at the camera."
- `face_verification_request()` - "Please show your face. Look at the camera for face verification."
- `unknown_person_alert()` - "Unknown person detected! Security alert!"
- `known_person_greeting(name)` - "Hello {name}! Welcome."
- `time_based_greeting()` - "Good morning!" / "Good afternoon!" / "Good evening!"

### Gujarati Messages
- `person_detected()` - "વ્યક્તિ શોધાઈ ગઈ છે. કૃપા કરીને કેમેરા તરફ જુઓ."

### Face Verification
- `face_verification_request()` - "કૃપા કરીને ચહેરો દેખાડો. ચહેરાની ઓળખ માટે કેમેરા તરફ જુઓ."
- `face_verification_reminder(count)` - Progressive reminders
- `verification_timeout()` - "સમય સમાપ્ત! ચહેરાની ઓળખ નિષ્ફળ."

### Security Alerts
- `unknown_person_alert()` - "અજ્ઞાત વ્યક્તિ શોધાઈ ગઈ છે! સુરક્ષા ચેતવણી!"
- `security_breach()` - "સુરક્ષા ભંગ! અનધિકૃત વ્યક્તિ શોધાઈ ગઈ છે!"

### Greetings
- `known_person_greeting(name)` - "નમસ્તે {name}! આપનું સ્વાગત છે."
- `time_based_greeting()` - Time-based greetings (morning/afternoon/evening)
- `welcome_back(name)` - "પાછા આવ્યા માટે આભાર {name}!"

### Guest Mode
- `guest_mode_activated(host_name)` - "મહેમાન મોડ સક્રિય થયો છે. {host_name} સાથે મહેમાન આવ્યા છે."
- `guest_mode_expired()` - "મહેમાન મોડ સમાપ્ત થયો છે."

## ⚙️ Configuration

### Voice Parameters
```python
sound = get_sound_system('en')  # English
sound = get_sound_system('gu')  # Gujarati

# Adjust voice parameters
sound.set_voice_params(speed=150, pitch=60, volume=120)

# Change language dynamically
sound.set_language('en')  # Switch to English
sound.set_language('gu')  # Switch to Gujarati
```

### Enable/Disable Sound
```python
# Disable sound
sound.disable()

# Enable sound
sound.enable()

# Check if sound is available
if sound.is_available():
    sound.speak("Sound is working!")
```

## 🔧 Troubleshooting

### espeak-ng not found
- **Windows**: Make sure espeak-ng is in your PATH. If not available, system will use winsound fallback (beep sounds)
- **Linux**: Install with `sudo apt install espeak-ng`
- **macOS**: Install with `brew install espeak-ng`

### No audio output
- Check if audio drivers are working
- Test with: `espeak-ng "Hello World"`
- On Windows, ensure Windows Media Player is available

### Windows Speech Fallbacks
When espeak-ng is not available on Windows, the system automatically tries these options in order:

#### 1. pyttsx3 (Best Option)
- **Installation**: `pip install pyttsx3`
- **Features**: Cross-platform TTS, good voice quality
- **Languages**: Supports multiple languages including English
- **Usage**: Automatically selected when available

#### 2. Windows SAPI (pywin32)
- **Installation**: `pip install pywin32`
- **Features**: Native Windows speech, built-in voices
- **Languages**: Uses Windows installed voices
- **Usage**: Fallback when pyttsx3 not available

#### 3. winsound (Final Fallback)
- **Features**: Beep sounds only
- **Security alerts**: 3 high-frequency beeps (800Hz)
- **Verification requests**: 1 medium-frequency beep (600Hz)  
- **Welcome messages**: 1 low-frequency beep (400Hz)
- **Default messages**: 1 standard beep (500Hz)

### Gujarati voice not working
- Install Gujarati language data: `sudo apt install espeak-ng-data-gujarati`
- Test with: `espeak-ng -v gu "હેલો"`

## 📁 File Structure

```
src/
├── sound_system.py          # Main sound system
├── advanced_person_tracker.py
└── settings/
    └── settings.py

install_sound.bat            # Windows installation
install_sound.sh             # Linux installation
test_sound.py                # Test script
SOUND_SYSTEM_README.md       # This file
```

## 🎯 Features

- ✅ Gujarati female voice (cute girl voice)
- ✅ Cross-platform support (Windows/Linux/macOS)
- ✅ Windows winsound fallback when espeak-ng unavailable
- ✅ Threaded speech (non-blocking)
- ✅ Speech queue system
- ✅ Priority speech support
- ✅ Predefined CCTV messages
- ✅ Voice parameter adjustment
- ✅ Enable/disable functionality

## 🚀 Quick Start

1. **Install espeak-ng**: Run `install_sound.bat` (Windows) or `install_sound.sh` (Linux)
2. **Test installation**: Run `python test_sound.py`
3. **Integrate**: Import and use in your CCTV system

```python
from src.sound_system import get_sound_system, get_messages

sound = get_sound_system()
messages = get_messages()

# Ready to use!
messages.person_detected()
```
