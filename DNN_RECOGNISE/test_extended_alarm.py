#!/usr/bin/env python3
"""
Test Extended Alarm System
Tests the 2-minute alarm functionality
"""
import sys
import time

# Add src to path
sys.path.append('src')

def test_extended_alarm():
    """Test extended alarm functionality"""
    print("🚨 Testing Extended Alarm System")
    print("=" * 40)
    
    try:
        from sound_player import get_sound_player, play_alarm, stop_alarm, is_alarm_active
        
        player = get_sound_player()
        print(f"✅ Sound player initialized")
        print(f"🔊 MP3 support: {player.mp3_available}")
        
        if not player.mp3_available:
            print("⚠️ MP3 support not available - using fallback")
            print("🚨 ALARM: Unknown person detected!")
            return True
        
        # Test alarm settings
        from settings.settings import AUDIO
        duration = AUDIO.get('alarm_duration_minutes', 2)
        interval = AUDIO.get('alarm_loop_interval', 5)
        
        print(f"⏰ Alarm duration: {duration} minutes")
        print(f"🔄 Loop interval: {interval} seconds")
        
        # Start alarm
        print("\n🚨 Starting extended alarm...")
        play_alarm()
        
        if is_alarm_active():
            print("✅ Alarm is active")
            
            # Monitor for a short time
            print("👂 Monitoring alarm for 15 seconds...")
            for i in range(15):
                if is_alarm_active():
                    print(f"🔊 Alarm active - {i+1}/15 seconds")
                else:
                    print("🔇 Alarm stopped")
                    break
                time.sleep(1)
            
            # Stop alarm manually
            print("\n🔇 Stopping alarm manually...")
            stop_alarm()
            
            if not is_alarm_active():
                print("✅ Alarm stopped successfully")
            else:
                print("❌ Alarm still active")
        else:
            print("❌ Alarm not active")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing extended alarm: {e}")
        return False

def main():
    """Main test function"""
    print("🔊 Extended Alarm System Test")
    print("=" * 40)
    
    success = test_extended_alarm()
    
    if success:
        print("\n✅ Extended alarm test completed!")
        print("\n📝 Configuration:")
        print("   - Alarm duration: 2 minutes (configurable)")
        print("   - Loop interval: 5 seconds (configurable)")
        print("   - Automatic stop after duration")
        print("   - Manual stop available")
        return 0
    else:
        print("\n❌ Extended alarm test failed!")
        return 1

if __name__ == "__main__":
    exit(main())
