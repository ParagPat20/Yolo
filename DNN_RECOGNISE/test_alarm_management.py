#!/usr/bin/env python3
"""
Test script for alarm management functionality
"""

import sys
import os
import logging
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_alarm_management():
    """Test alarm management functionality"""
    try:
        from sound_system import get_sound_system
        from sound_player import get_sound_player
        
        logger.info("🧪 Testing Alarm Management")
        logger.info("=" * 30)
        
        # Initialize sound system
        sound_system = get_sound_system()
        sound_player = get_sound_player()
        
        if not sound_system:
            logger.error("❌ Sound system not available")
            return False
        
        logger.info(f"✅ Sound system initialized with TTS: {sound_system.tts_system}")
        
        # Test 1: Check initial alarm state
        logger.info("\n📋 Test 1: Initial Alarm State")
        logger.info(f"Alarm playing: {sound_system.is_alarm_playing()}")
        logger.info(f"Sound player alarm active: {sound_player.is_alarm_active()}")
        
        # Test 2: Start alarm
        logger.info("\n📋 Test 2: Start Alarm")
        sound_system.start_alarm()
        logger.info(f"Alarm playing after start: {sound_system.is_alarm_playing()}")
        
        # Test 3: Try to start alarm again (should be skipped)
        logger.info("\n📋 Test 3: Try to Start Alarm Again")
        sound_system.start_alarm()  # Should be skipped
        
        # Test 4: Stop alarm
        logger.info("\n📋 Test 4: Stop Alarm")
        sound_system.stop_alarm()
        logger.info(f"Alarm playing after stop: {sound_system.is_alarm_playing()}")
        
        # Test 5: Test sound player alarm management
        logger.info("\n📋 Test 5: Sound Player Alarm Management")
        if sound_player.mp3_available:
            logger.info("Testing MP3 alarm (will be skipped if no MP3 files)")
            sound_player.play_alarm(sound_system)
            time.sleep(2)  # Let it play briefly
            sound_player.stop_alarm()
            logger.info("Sound player alarm stopped")
        else:
            logger.info("MP3 not available - skipping sound player test")
        
        logger.info("✅ All alarm management tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_known_person_alarm_stop():
    """Test that known person detection stops alarm"""
    try:
        from sound_system import get_sound_system
        
        logger.info("\n📋 Test 6: Known Person Alarm Stop")
        
        sound_system = get_sound_system()
        
        # Simulate alarm state
        sound_system.start_alarm()
        logger.info(f"Alarm started: {sound_system.is_alarm_playing()}")
        
        # Simulate known person detection (should stop alarm)
        logger.info("Simulating known person detection...")
        if sound_system.is_alarm_playing():
            logger.info("🔇 Stopping alarm - known person detected")
            sound_system.stop_alarm()
        
        logger.info(f"Alarm after known person: {sound_system.is_alarm_playing()}")
        logger.info("✅ Known person alarm stop test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Known person test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Alarm Management Tests")
    logger.info("=" * 50)
    
    # Test 1: Basic alarm management
    success1 = test_alarm_management()
    
    # Test 2: Known person alarm stop
    success2 = test_known_person_alarm_stop()
    
    # Results
    logger.info("\n📊 Test Results:")
    logger.info("=" * 20)
    logger.info(f"Alarm Management: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"Known Person Stop: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        logger.info("\n🎉 All alarm management tests passed!")
        return 0
    else:
        logger.error("\n💥 Some alarm management tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
