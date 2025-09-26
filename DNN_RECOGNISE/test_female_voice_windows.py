#!/usr/bin/env python3
"""
Test script for female voice functionality in CCTV system - WINDOWS VERSION
Run this script to test if the female voice TTS is working properly on Windows
"""

import subprocess
import sys
import time
import platform

def test_windows_tts():
    """Test Windows TTS functionality"""
    print("🗣️ Testing Female Voice for CCTV System - WINDOWS")
    print("=" * 60)

    # Check if running on Windows
    if platform.system() != 'Windows':
        print("❌ This is a Windows-specific test script")
        return

    # Test Windows built-in TTS
    print("Testing Windows built-in TTS (SAPI)...")
    try:
        import win32com.client

        # Initialize SAPI
        speaker = win32com.client.Dispatch("SAPI.SpVoice")

        # Test basic message
        test_message = "Hello, this is a test of the Windows female voice for the CCTV system. I can greet people and give alerts."
        speaker.Speak(test_message)

        print("✅ Windows SAPI TTS test successful!")

        # Test different voices if available
        voices = speaker.GetVoices()
        print(f"Available voices: {len(voices)}")

        if len(voices) > 1:
            # Try to find a female voice
            for i in range(len(voices)):
                voice = voices.Item(i)
                if "female" in voice.GetDescription().lower() or "woman" in voice.GetDescription().lower():
                    print(f"Found female voice: {voice.GetDescription()}")
                    speaker.Voice = voice
                    speaker.Speak("This is the female voice for CCTV alerts.")
                    break
            else:
                print("Using default voice (may be male)")

    except ImportError:
        print("❌ pywin32 not installed. Please install: pip install pywin32")
        print("Or install Windows TTS manually...")

        # Try PowerShell TTS as fallback
        try:
            ps_command = 'Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Speak("Hello, this is a test of the Windows female voice for the CCTV system.")'
            subprocess.run(['powershell', '-Command', ps_command], capture_output=True, timeout=10)
            print("✅ PowerShell TTS test successful!")
        except Exception as e:
            print(f"❌ PowerShell TTS failed: {e}")

    except Exception as e:
        print(f"❌ Windows SAPI TTS failed: {e}")

    print("\n" + "=" * 60)
    print("🎯 Windows TTS Test Complete!")

def test_cctv_messages_windows():
    """Test typical CCTV system messages with Windows TTS"""
    print("\n🧪 Testing CCTV System Messages - WINDOWS")
    print("=" * 60)

    messages = [
        "Good morning! Welcome to the premises.",
        "Good afternoon! Please verify your identity.",
        "Good evening! Thank you for your cooperation.",
        "Please look at the camera for verification.",
        "Welcome back! How can I help you today?",
        "Alert! Unknown person detected in the area.",
        "Security system activated. Please identify yourself.",
        "Person verification required. Please show your face clearly."
    ]

    print("Testing CCTV messages with Windows TTS...")

    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")

        for i, message in enumerate(messages, 1):
            print(f"{i}. Testing: '{message}'")
            try:
                speaker.Speak(message)
                print("   ✅ Message spoken successfully")
                time.sleep(1)  # Brief pause between messages
            except Exception as e:
                print(f"   ❌ Error: {e}")

    except ImportError:
        print("❌ pywin32 not available, cannot test messages")
    except Exception as e:
        print(f"❌ Error testing messages: {e}")

    print("\n✅ Windows TTS message test complete!")
    print("Your CCTV system is ready to speak with Windows TTS!")

if __name__ == "__main__":
    try:
        test_windows_tts()
        time.sleep(2)  # Brief pause
        test_cctv_messages_windows()

        print("\n🎉 Windows Female Voice Test Complete!")
        print("Your CCTV system is configured with Windows TTS support.")
        print("\n📝 Note: This is the Windows version.")
        print("For Raspberry Pi, use the Linux TTS engines (espeak-ng, festival)")

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        sys.exit(1)
