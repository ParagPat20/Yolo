#!/usr/bin/env python3
"""
Test MP3 Sound System
Tests alarm and verification beep sounds
"""
import os
import sys
import time

# Add src to path
sys.path.append('src')

def test_sound_files():
    """Test if sound files exist"""
    print("🔍 Checking sound files...")
    
    alarm_path = 'sounds/alarm.mp3'
    beep_path = 'sounds/verification_beep.mp3'
    
    if os.path.exists(alarm_path):
        print(f"✅ Alarm sound found: {alarm_path}")
    else:
        print(f"❌ Alarm sound not found: {alarm_path}")
    
    if os.path.exists(beep_path):
        print(f"✅ Verification beep found: {beep_path}")
    else:
        print(f"❌ Verification beep not found: {beep_path}")
    
    return os.path.exists(alarm_path) and os.path.exists(beep_path)

def test_sound_player():
    """Test sound player functionality"""
    print("\n🔊 Testing sound player...")
    
    try:
        from sound_player import get_sound_player, play_alarm, play_verification_beep
        
        player = get_sound_player()
        print(f"✅ Sound player initialized")
        print(f"🔊 MP3 support: {player.mp3_available}")
        
        if player.mp3_available:
            print("🎵 Testing verification beep...")
            play_verification_beep()
            time.sleep(2)
            
            print("🚨 Testing alarm sound...")
            play_alarm()
            time.sleep(3)
            
            print("✅ Sound tests completed!")
        else:
            print("⚠️ MP3 support not available - using fallback")
            print("🔍 BEEP: Please show your face for verification")
            print("🚨 ALARM: Unknown person detected!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing sound player: {e}")
        return False

def main():
    """Main test function"""
    print("🔊 MP3 Sound System Test")
    print("=" * 40)
    
    # Test sound files
    files_ok = test_sound_files()
    
    if not files_ok:
        print("\n❌ Sound files missing. Please ensure:")
        print("   - sounds/alarm.mp3 exists")
        print("   - sounds/verification_beep.mp3 exists")
        return 1
    
    # Test sound player
    player_ok = test_sound_player()
    
    if player_ok:
        print("\n✅ All sound tests passed!")
        return 0
    else:
        print("\n❌ Sound player test failed!")
        return 1

if __name__ == "__main__":
    exit(main())
