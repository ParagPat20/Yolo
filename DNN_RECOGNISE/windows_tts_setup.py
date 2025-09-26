#!/usr/bin/env python3
"""
Windows TTS Setup and Installation Script
This script helps install and configure TTS for Windows
"""

import subprocess
import sys
import os

def install_pywin32():
    """Install pywin32 for Windows SAPI TTS"""
    print("🔧 Installing pywin32 for Windows TTS support...")

    try:
        # Check if pip is available
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("📦 Installing pywin32...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pywin32'],
                         capture_output=True, text=True, timeout=60)
            print("✅ pywin32 installed successfully!")
            return True
        else:
            print("❌ pip not found. Please install manually:")
            print("   pip install pywin32")
            return False

    except Exception as e:
        print(f"❌ Error installing pywin32: {e}")
        return False

def test_windows_tts():
    """Test Windows TTS functionality"""
    print("\n🗣️ Testing Windows TTS...")

    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")

        # Test message
        test_msg = "Windows TTS is working correctly. This is a test of the female voice system."
        print(f"Speaking: '{test_msg}'")
        speaker.Speak(test_msg)

        print("✅ Windows TTS test successful!")

        # List available voices
        voices = speaker.GetVoices()
        print(f"\nAvailable voices ({len(voices)}):")

        female_voices = []
        for i in range(len(voices)):
            voice = voices.Item(i)
            desc = voice.GetDescription()
            print(f"  {i}: {desc}")

            # Look for female voices
            if any(keyword in desc.lower() for keyword in ['female', 'woman', 'girl', 'zira', 'hazel']):
                female_voices.append((i, desc))

        if female_voices:
            print(f"\n🗣️ Female voices found: {len(female_voices)}")
            for idx, desc in female_voices:
                print(f"  Voice {idx}: {desc}")

            # Test with first female voice
            if female_voices:
                idx, desc = female_voices[0]
                speaker.Voice = voices.Item(idx)
                print(f"\nTesting with: {desc}")
                speaker.Speak("This is the female voice for CCTV alerts and greetings.")
        else:
            print("\n⚠️ No female voices found, using default voice")

        return True

    except ImportError:
        print("❌ pywin32 not installed. Run install_pywin32() first.")
        return False
    except Exception as e:
        print(f"❌ Windows TTS test failed: {e}")
        return False

def manual_installation_guide():
    """Provide manual installation instructions"""
    print("\n📋 Manual Installation Guide for Windows TTS")
    print("=" * 50)
    print("If automatic installation fails, follow these steps:")
    print()
    print("1. Install Python TTS Libraries:")
    print("   pip install pywin32")
    print("   pip install pyttsx3")
    print()
    print("2. Alternative TTS Engines (Optional):")
    print("   • eSpeak for Windows: https://github.com/espeak-ng/espeak-ng/releases")
    print("   • Festival for Windows: Not natively supported")
    print()
    print("3. Test Installation:")
    print("   python -c \"import win32com.client; speaker = win32com.client.Dispatch('SAPI.SpVoice'); speaker.Speak('TTS is working')\"")
    print()
    print("4. Check Available Voices:")
    print("   python -c \"import win32com.client; speaker = win32com.client.Dispatch('SAPI.SpVoice'); voices = speaker.GetVoices(); [print(f'{i}: {v.GetDescription()}') for i,v in enumerate(voices)]\"")

def main():
    """Main installation function"""
    print("🎯 Windows TTS Setup for CCTV System")
    print("=" * 50)

    # Install pywin32
    install_success = install_pywin32()

    if install_success:
        # Test TTS
        test_success = test_windows_tts()

        if test_success:
            print("\n🎉 Windows TTS setup completed successfully!")
            print("Your CCTV system is ready to speak with Windows TTS.")
        else:
            print("\n⚠️ TTS installation may need manual setup.")
            manual_installation_guide()
    else:
        manual_installation_guide()

    print("\n📝 Next Steps:")
    print("1. Run the Windows voice test: python test_female_voice_windows.py")
    print("2. Configure voice settings in src/settings/settings.py")
    print("3. Run your CCTV system: python src/cctv_system.py")

if __name__ == "__main__":
    main()
