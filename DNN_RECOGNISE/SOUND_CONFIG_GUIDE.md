# 🔊 Sound System Configuration Guide

This guide explains how to configure the sound system language and voice parameters in `settings.py`.

## 🌍 Language Configuration

### Quick Language Change

To change the default language, edit `src/settings/settings.py`:

```python
# Sound System Settings
SOUND_SYSTEM = {
    'enabled': True,  # Enable/disable sound system
    'language': 'gu',  # Change this: 'en' for English, 'gu' for Gujarati
    # ... rest of settings
}
```

**Options:**
- `'gu'` - Gujarati (default)
- `'en'` - English

## 🎛️ Voice Parameters

### Gujarati Voice Settings

```python
'voice_parameters': {
    'gujarati': {
        'speed': 163,      # Speech speed (50-300)
        'pitch': 55,       # Voice pitch (0-99, female cute girl voice)
        'volume': 100,     # Volume (0-200)
        'amplitude': 100   # Amplitude (0-200)
    },
    # ... english settings
}
```

### English Voice Settings

```python
'voice_parameters': {
    'english': {
        'speed': 150,      # Speech speed (50-300)
        'pitch': 50,       # Voice pitch (0-99, female voice)
        'volume': 100,     # Volume (0-200)
        'amplitude': 100   # Amplitude (0-200)
    }
}
```

## ⚙️ Advanced Settings

### Complete Configuration Example

```python
SOUND_SYSTEM = {
    'enabled': True,  # Enable/disable sound system
    'language': 'gu',  # Default language: 'en' for English, 'gu' for Gujarati
    'voice_parameters': {
        'gujarati': {
            'speed': 163,      # Speech speed (50-300)
            'pitch': 55,       # Voice pitch (0-99, female cute girl voice)
            'volume': 100,     # Volume (0-200)
            'amplitude': 100   # Amplitude (0-200)
        },
        'english': {
            'speed': 150,      # Speech speed (50-300)
            'pitch': 50,       # Voice pitch (0-99, female voice)
            'volume': 100,     # Volume (0-200)
            'amplitude': 100   # Amplitude (0-200)
        }
    },
    'winsound_fallback': True,  # Use winsound fallback on Windows when espeak-ng unavailable
    'speech_queue_enabled': True,  # Enable speech queue system
    'priority_speech_enabled': True,  # Enable priority speech (interrupts current speech)
    'auto_language_detection': False,  # Auto-detect language from system locale
}
```

## 🎯 Common Configurations

### For English Users

```python
SOUND_SYSTEM = {
    'enabled': True,
    'language': 'en',  # English
    'voice_parameters': {
        'english': {
            'speed': 150,
            'pitch': 50,
            'volume': 100,
            'amplitude': 100
        }
    }
}
```

### For Gujarati Users

```python
SOUND_SYSTEM = {
    'enabled': True,
    'language': 'gu',  # Gujarati
    'voice_parameters': {
        'gujarati': {
            'speed': 163,
            'pitch': 55,
            'volume': 100,
            'amplitude': 100
        }
    }
}
```

### Disable Sound System

```python
SOUND_SYSTEM = {
    'enabled': False,  # Disable sound system
    'language': 'gu',
    # ... rest of settings
}
```

## 🔧 Voice Parameter Tuning

### Speed (50-300)
- **50-100**: Very slow speech
- **100-150**: Normal speech
- **150-200**: Fast speech
- **200-300**: Very fast speech

### Pitch (0-99)
- **0-30**: Low pitch (male-like)
- **30-50**: Medium pitch
- **50-70**: High pitch (female-like)
- **70-99**: Very high pitch

### Volume (0-200)
- **0-50**: Very quiet
- **50-100**: Normal volume
- **100-150**: Loud
- **150-200**: Very loud

## 🚀 Quick Start

1. **Edit settings**: Open `src/settings/settings.py`
2. **Change language**: Set `'language': 'en'` for English or `'language': 'gu'` for Gujarati
3. **Adjust voice**: Modify speed, pitch, volume as needed
4. **Test**: Run `python test_sound.py` to test your configuration

## 📝 Examples

### Example 1: English with Slower Speech

```python
SOUND_SYSTEM = {
    'enabled': True,
    'language': 'en',
    'voice_parameters': {
        'english': {
            'speed': 120,    # Slower speech
            'pitch': 45,     # Slightly lower pitch
            'volume': 120,   # Louder
            'amplitude': 100
        }
    }
}
```

### Example 2: Gujarati with Faster Speech

```python
SOUND_SYSTEM = {
    'enabled': True,
    'language': 'gu',
    'voice_parameters': {
        'gujarati': {
            'speed': 180,    # Faster speech
            'pitch': 60,     # Higher pitch
            'volume': 80,    # Quieter
            'amplitude': 100
        }
    }
}
```

## 🔄 Dynamic Language Switching

You can also change language at runtime:

```python
from src.sound_system import get_sound_system, get_messages

# Get current sound system
sound = get_sound_system()
messages = get_messages()

# Change to English
sound.set_language('en')
messages = get_messages('en')

# Change to Gujarati
sound.set_language('gu')
messages = get_messages('gu')
```

## 🛠️ Troubleshooting

### Sound Not Working
1. Check `'enabled': True` in settings
2. Verify espeak-ng is installed: `python test_sound.py`
3. Check voice parameters are within valid ranges

### Wrong Language
1. Verify `'language': 'en'` or `'language': 'gu'` in settings
2. Restart the CCTV system after changing settings
3. Check that voice parameters exist for your language

### Voice Too Fast/Slow
1. Adjust `'speed'` parameter (50-300)
2. Lower values = slower speech
3. Higher values = faster speech

### Voice Too High/Low
1. Adjust `'pitch'` parameter (0-99)
2. Lower values = lower pitch
3. Higher values = higher pitch
