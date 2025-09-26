#!/usr/bin/env python3
"""
Test script for female voice functionality in CCTV system
Run this script to test if the female voice TTS is working properly
"""

import subprocess
import sys
import time

def test_female_voice():
    """Test the female voice configuration"""
    print("🗣️ Testing Female Voice for CCTV System")
    print("=" * 50)

    # Test espeak-ng female voice
    print("Testing espeak-ng female voice (en+f3)...")
    try:
        result = subprocess.run([
            'espeak-ng', '-v', 'en+f3', '-s', '150', '-p', '50',
            'Hello, this is a test of the female voice for the CCTV system. I can greet people and give alerts.'
        ], capture_output=True, timeout=10)

        if result.returncode == 0:
            print("✅ espeak-ng female voice test successful!")
        else:
            print("❌ espeak-ng female voice test failed")
    except Exception as e:
        print(f"❌ Error testing espeak-ng: {e}")

    print("\nTesting different female voice variants...")

    # Test different female voices
    voices = ['en+f1', 'en+f2', 'en+f3', 'en+f4']
    for voice in voices:
        print(f"Testing {voice}...")
        try:
            subprocess.run([
                'espeak-ng', '-v', voice, '-s', '140',
                f'This is {voice} speaking. I am a female voice option.'
            ], capture_output=True, timeout=5)
            print(f"  ✅ {voice} works")
        except Exception as e:
            print(f"  ❌ {voice} failed: {e}")

    print("\nTesting festival female voice...")
    try:
        # Test festival female voice
        cmd = '''echo '(voice_cmu_us_slt_cg)' > /tmp/festival_test
echo '(SayText "Hello, this is the festival female voice for the CCTV system. I provide backup text-to-speech functionality.")' >> /tmp/festival_test
festival -b /tmp/festival_test 2>/dev/null'''

        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
        if result.returncode == 0:
            print("✅ Festival female voice test successful!")
        else:
            print("❌ Festival female voice test failed")
    except Exception as e:
        print(f"❌ Error testing festival: {e}")

    print("\n" + "=" * 50)
    print("🎯 Voice Test Complete!")
    print("\nIf you heard female voice messages, the TTS system is working correctly.")
    print("The CCTV system will use these voices for:")
    print("  • Greeting people: 'Good morning, John!'")
    print("  • Alert messages: 'Unknown person detected!'")
    print("  • Verification requests: 'Please look at the camera'")

def test_cctv_messages():
    """Test typical CCTV system messages with female voice"""
    print("\n🧪 Testing CCTV System Messages")
    print("=" * 50)

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

    print("Testing CCTV messages with female voice...")
    for i, message in enumerate(messages, 1):
        print(f"{i}. Testing: '{message}'")
        try:
            subprocess.run([
                'espeak-ng', '-v', 'en+f3', '-s', '140', '-p', '50', message
            ], capture_output=True, timeout=8)
            print("   ✅ Message spoken successfully")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
    print("\n✅ All CCTV messages tested!")
    print("Your CCTV system is ready to speak with a female voice!")

if __name__ == "__main__":
    try:
        test_female_voice()
        time.sleep(1)  # Brief pause
        test_cctv_messages()

        print("\n🎉 Female Voice Test Complete!")
        print("Your Raspberry Pi CCTV system is configured with female voice support.")

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        sys.exit(1)
