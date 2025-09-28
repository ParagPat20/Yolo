#!/usr/bin/env python3
"""
Test script for the new WAV file generation and playback system
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

def test_wav_generation():
    """Test WAV file generation with Piper TTS"""
    try:
        from sound_system import get_sound_system
        
        logger.info("🧪 Testing WAV File Generation System")
        logger.info("=" * 50)
        
        # Initialize sound system
        sound_system = get_sound_system()
        
        if not sound_system.is_enabled:
            logger.error("❌ Sound system not available")
            return False
        
        logger.info(f"✅ Sound system initialized with language: {sound_system.get_language()}")
        logger.info(f"📁 WAV files directory: {sound_system.wav_files_dir}")
        
        # Check if WAV files were generated
        wav_files = [f for f in os.listdir(sound_system.wav_files_dir) if f.endswith('.wav')]
        logger.info(f"📄 Generated {len(wav_files)} WAV files:")
        for wav_file in sorted(wav_files):
            logger.info(f"  - {wav_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_wav_playback():
    """Test WAV file playback"""
    try:
        from sound_system import get_sound_system
        
        logger.info("\n🧪 Testing WAV File Playback")
        logger.info("=" * 40)
        
        sound_system = get_sound_system()
        
        if not sound_system.is_enabled:
            logger.error("❌ Sound system not available")
            return False
        
        # Test different WAV files
        test_files = [
            "person_detected.wav",
            "face_verification_request.wav",
            "unknown_person_alert.wav",
            "time_based_greeting_morning.wav"
        ]
        
        for wav_file in test_files:
            filepath = os.path.join(sound_system.wav_files_dir, wav_file)
            if os.path.exists(filepath):
                logger.info(f"🔊 Testing playback: {wav_file}")
                sound_system.play_wav_file(wav_file)
                time.sleep(2)  # Wait for playback
            else:
                logger.warning(f"⚠️ WAV file not found: {wav_file}")
        
        logger.info("✅ WAV playback test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Playback test failed: {e}")
        return False

def test_sound_player():
    """Test sound player functionality"""
    try:
        from sound_player import get_sound_player
        
        logger.info("\n🧪 Testing Sound Player")
        logger.info("=" * 30)
        
        player = get_sound_player()
        
        if not player.is_enabled:
            logger.warning("⚠️ Sound player disabled")
            return False
        
        logger.info("✅ Sound player initialized")
        
        # Test WAV file playback
        logger.info("🔊 Testing WAV file playback...")
        player.play_person_detected()
        time.sleep(2)
        
        player.play_verification_request()
        time.sleep(2)
        
        player.play_unknown_person_alert()
        time.sleep(2)
        
        # Test MP3 playback (if available)
        if os.path.exists(player.verification_beep_path):
            logger.info("🔊 Testing MP3 file playback...")
            player.play_verification_beep()
            time.sleep(2)
        else:
            logger.warning("⚠️ MP3 files not found - skipping MP3 test")
        
        logger.info("✅ Sound player test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sound player test failed: {e}")
        return False

def test_language_switching():
    """Test language switching and WAV regeneration"""
    try:
        from sound_system import get_sound_system
        
        logger.info("\n🧪 Testing Language Switching")
        logger.info("=" * 40)
        
        # Test English
        sound_system = get_sound_system('en')
        logger.info(f"✅ English sound system: {sound_system.get_language()}")
        
        # Test Gujarati
        sound_system = get_sound_system('gu')
        logger.info(f"✅ Gujarati sound system: {sound_system.get_language()}")
        
        logger.info("✅ Language switching test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Language switching test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting WAV System Tests")
    logger.info("=" * 60)
    
    # Test 1: WAV generation
    success1 = test_wav_generation()
    
    # Test 2: WAV playback
    success2 = test_wav_playback()
    
    # Test 3: Sound player
    success3 = test_sound_player()
    
    # Test 4: Language switching
    success4 = test_language_switching()
    
    # Results
    logger.info("\n📊 Test Results:")
    logger.info("=" * 20)
    logger.info(f"WAV Generation: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"WAV Playback: {'✅ PASS' if success2 else '❌ FAIL'}")
    logger.info(f"Sound Player: {'✅ PASS' if success3 else '❌ FAIL'}")
    logger.info(f"Language Switching: {'✅ PASS' if success4 else '❌ FAIL'}")
    
    if all([success1, success2, success3, success4]):
        logger.info("\n🎉 All WAV system tests passed!")
        logger.info("💡 The new system is ready for use!")
        return 0
    else:
        logger.error("\n💥 Some WAV system tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
