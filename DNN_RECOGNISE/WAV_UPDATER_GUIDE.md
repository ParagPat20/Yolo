# WAV File Updater Guide

## Overview
The `update_wav_files.py` script allows you to update WAV files from settings.py as needed for speech in the CCTV system.

## Features
- ✅ **Automatic WAV Generation**: Generate WAV files using Piper TTS from settings
- ✅ **Language Support**: English and Gujarati WAV files
- ✅ **Selective Updates**: Update specific WAV files or all files
- ✅ **Force Updates**: Overwrite existing WAV files
- ✅ **File Management**: List, clean, and get information about WAV files
- ✅ **Command Line Interface**: Easy-to-use command line interface

## Usage

### Basic Commands

#### 1. Update All WAV Files
```bash
# Update all missing WAV files
python update_wav_files.py --update-all

# Force update all WAV files (overwrite existing)
python update_wav_files.py --update-all --force
```

#### 2. Update Specific WAV File
```bash
# Update a specific message
python update_wav_files.py --update person_detected

# Update a specific variant
python update_wav_files.py --update time_based_greeting --variant morning
```

#### 3. List and Information
```bash
# List all existing WAV files
python update_wav_files.py --list

# Show detailed information
python update_wav_files.py --info
```

#### 4. Clean Up
```bash
# Remove all WAV files
python update_wav_files.py --clean
```

#### 5. Language Settings
```bash
# Set language to English
python update_wav_files.py --language en --update-all

# Set language to Gujarati
python update_wav_files.py --language gu --update-all
```

### Available Message Keys

#### English Messages
- `person_detected` - "Person detected. Please look at the camera."
- `face_verification_request` - "Please show your face. Look at the camera for face verification."
- `face_verification_reminder` - Multiple variants (1, 2, 3)
- `verification_timeout` - "Time's up! Face verification failed."
- `unknown_person_alert` - "Unknown person detected! Security alert!"
- `security_breach` - "Security breach! Unauthorized person detected!"
- `known_person_greeting` - "Hello {name}! Welcome."
- `time_based_greeting` - Variants: morning, afternoon, evening
- `welcome_back` - "Welcome back {name}!"
- `guest_mode_activated` - "Guest mode activated. {host_name} has a guest."
- `guest_mode_expired` - "Guest mode expired. Reverting to normal security protocols."

#### Gujarati Messages
- `person_detected` - "વ્યક્તિ શોધાઈ ગઈ છે. કૃપા કરીને કેમેરા તરફ જુઓ."
- `face_verification_request` - "કૃપા કરીને ચહેરો દેખાડો. ચહેરાની ઓળખ માટે કેમેરા તરફ જુઓ."
- And more...

### Examples

#### Generate All English WAV Files
```bash
python update_wav_files.py --language en --update-all
```

#### Generate All Gujarati WAV Files
```bash
python update_wav_files.py --language gu --update-all
```

#### Update Specific Files
```bash
# Update person detection message
python update_wav_files.py --update person_detected

# Update morning greeting
python update_wav_files.py --update time_based_greeting --variant morning

# Update first verification reminder
python update_wav_files.py --update face_verification_reminder --variant 1
```

#### Check Status
```bash
# See what WAV files exist
python update_wav_files.py --list

# Get detailed information
python update_wav_files.py --info
```

## Requirements

### For WAV Generation (Linux/Raspberry Pi)
- **Piper TTS**: Must be installed and configured
- **Model Files**: English and/or Gujarati models must be available
- **Settings**: Proper configuration in `settings.py`

### For MP3 Fallback (Windows)
- **MP3 Files**: `sounds/alarm.mp3` and `sounds/verification_beep.mp3`
- **Audio Player**: `mpg123` (Linux) or PowerShell (Windows)

## File Structure
```
sounds/
├── wav/                    # Generated WAV files
│   ├── person_detected.wav
│   ├── face_verification_request.wav
│   ├── time_based_greeting_morning.wav
│   ├── time_based_greeting_afternoon.wav
│   ├── time_based_greeting_evening.wav
│   └── ...
├── alarm.mp3              # Alarm sound
└── verification_beep.mp3  # Verification beep
```

## Integration with CCTV System

The WAV file updater works seamlessly with the CCTV system:

1. **Automatic Generation**: The CCTV system automatically generates missing WAV files when needed
2. **Settings Integration**: Uses the same settings as the CCTV system
3. **Language Support**: Supports the same languages as the CCTV system
4. **Fallback Support**: Falls back to MP3 files when WAV generation is not available

## Troubleshooting

### Piper TTS Not Available
- **Windows**: Use MP3 fallback files
- **Linux**: Install Piper TTS and models
- **Check**: Run `python update_wav_files.py --info` to see status

### WAV Files Not Generated
- Check Piper TTS installation
- Verify model files exist
- Check settings configuration
- Use `--force` flag to overwrite existing files

### Language Issues
- Use `--language` flag to set correct language
- Ensure language models are available
- Check settings.py configuration

## Advanced Usage

### Programmatic Usage
```python
from update_wav_files import WAVFileUpdater

# Initialize updater
updater = WAVFileUpdater()

# Update all files
generated, skipped = updater.update_all_wav_files()

# Update specific file
success = updater.update_specific_wav_file('person_detected')

# Get information
info = updater.get_wav_file_info()
```

### Batch Operations
```bash
# Generate all English files
python update_wav_files.py --language en --update-all

# Clean and regenerate
python update_wav_files.py --clean
python update_wav_files.py --update-all --force
```

This WAV file updater provides a complete solution for managing speech files in the CCTV system! 🎉
