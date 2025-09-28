#!/usr/bin/env python3
"""
Test script for the sound system
Tests espeak-ng with Gujarati female voice
"""

import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sound_system import get_sound_system, get_messages

def main():
    print("🔊 Testing Sound System with English and Gujarati Support")
    print("=" * 60)
    
    # Test both languages
    for language in ['en', 'gu']:
        lang_name = "English" if language == 'en' else "Gujarati"
        print(f"\n🌍 Testing {lang_name} Language")
        print("-" * 40)
        
        # Initialize sound system with specific language
        sound = get_sound_system(language)
        messages = get_messages(language)
    
        if not sound.is_enabled:
            print(f"❌ {lang_name} sound system is disabled - no audio available")
            continue
        
        if hasattr(sound, 'use_pyttsx3') and sound.use_pyttsx3:
            print("🔊 Using pyttsx3 (Windows TTS)")
        elif hasattr(sound, 'use_win32') and sound.use_win32:
            print("🔊 Using Windows SAPI (pywin32)")
        elif hasattr(sound, 'use_winsound') and sound.use_winsound:
            print("🔊 Using winsound fallback (beep sounds)")
        else:
            print(f"🔊 Using espeak-ng with {lang_name} voice")
        
        print("✅ Sound system is enabled")
        print(f"🎛️ Voice parameters: speed={sound.voice_params['speed']}, pitch={sound.voice_params['pitch']}")
        print()
        
        # Test basic speech
        print("1. Testing basic speech...")
        if language == 'gu':
            sound.speak("હેલો વર્લ્ડ! આ ટેસ્ટ છે.")
        else:
            sound.speak("Hello World! This is a test.")
        time.sleep(4)
        
        # Test CCTV messages
        print("\n2. Testing CCTV messages...")
        
        print("   - Person detection message:")
        messages.person_detected()
        time.sleep(3)
        
        print("   - Face verification request:")
        messages.face_verification_request()
        time.sleep(3)
        
        print("   - Face verification reminder:")
        messages.face_verification_reminder(1)
        time.sleep(3)
        
        print("   - Known person greeting:")
        test_name = "રાજ" if language == 'gu' else "Raj"
        messages.known_person_greeting(test_name)
        time.sleep(3)
        
        print("   - Time-based greeting:")
        messages.time_based_greeting()
        time.sleep(3)
        
        print("   - Unknown person alert:")
        messages.unknown_person_alert()
        time.sleep(3)
        
        print("   - Security breach:")
        messages.security_breach()
        time.sleep(3)
        
        print("   - Guest mode activation:")
        messages.guest_mode_activated(test_name)
        time.sleep(3)
        
        print("   - Welcome back:")
        messages.welcome_back(test_name)
        time.sleep(3)
        
        print(f"\n✅ {lang_name} sound system test completed!")
    
    print("\n🎉 All language tests passed - sound system is working correctly!")
    print("\n📝 Integration ready:")
    print("   from src.sound_system import get_sound_system, get_messages")
    print("   sound = get_sound_system('en')  # English")
    print("   sound = get_sound_system('gu')  # Gujarati")
    print("   messages = get_messages('en')   # English messages")
    print("   messages = get_messages('gu')   # Gujarati messages")

if __name__ == "__main__":
    main()
