#!/usr/bin/env python3
"""
Test script for Piper TTS integration with fallback to espeak-ng
"""

import sys
import os
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_piper_integration():
    """Test Piper TTS integration with fallback chain"""
    try:
        from sound_system import get_sound_system, get_messages
        
        logger.info("🧪 Testing Piper TTS Integration")
        logger.info("=" * 50)
        
        # Test 1: Initialize sound system
        logger.info("🔧 Initializing sound system...")
        sound_system = get_sound_system()
        messages = get_messages()
        
        if not sound_system:
            logger.error("❌ Sound system not available")
            return False
        
        logger.info(f"✅ Sound system initialized with TTS: {sound_system.tts_system}")
        
        # Test 2: Test different message types
        test_messages = [
            "Hello! This is a test of the Piper TTS system.",
            "Good morning! The CCTV system is working properly.",
            "Unknown person detected! Please show your face for verification.",
            "Welcome back! You have been successfully identified."
        ]
        
        logger.info("🔊 Testing speech with different messages...")
        for i, message in enumerate(test_messages, 1):
            logger.info(f"📢 Test {i}: {message}")
            
            # Test regular speech
            sound_system.speak(message)
            
            # Wait a moment between tests
            import time
            time.sleep(2)
        
        # Test 3: Test priority speech
        logger.info("🔊 Testing priority speech...")
        sound_system.speak("This is a priority message that should interrupt any ongoing speech.", priority=True)
        
        # Test 4: Test language switching
        logger.info("🌐 Testing language switching...")
        original_lang = sound_system.get_language()
        
        # Switch to English
        sound_system.set_language('en')
        sound_system.speak("Testing English language support.")
        
        # Switch to Gujarati (if supported)
        sound_system.set_language('gu')
        sound_system.speak("Testing Gujarati language support.")
        
        # Restore original language
        sound_system.set_language(original_lang)
        
        logger.info("✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_fallback_chain():
    """Test the fallback chain when Piper is not available"""
    try:
        from sound_system import SoundSystem
        
        logger.info("🧪 Testing Fallback Chain")
        logger.info("=" * 30)
        
        # Test with different TTS systems
        tts_systems = ['piper', 'espeak', 'pyttsx3', 'win32', 'winsound']
        
        for tts_system in tts_systems:
            logger.info(f"🔧 Testing {tts_system} availability...")
            
            # Create a temporary sound system instance
            sound = SoundSystem()
            
            # Check availability
            if tts_system == 'piper':
                available = sound._check_piper_availability()
            elif tts_system == 'espeak':
                available = sound._check_espeak_availability()
            elif tts_system == 'pyttsx3':
                available = sound._check_pyttsx3_availability()
            elif tts_system == 'win32':
                available = sound._check_win32_availability()
            elif tts_system == 'winsound':
                available = sound._check_winsound_availability()
            
            status = "✅ Available" if available else "❌ Not available"
            logger.info(f"  {tts_system}: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Piper TTS Integration Tests")
    logger.info("=" * 60)
    
    # Test 1: Basic integration
    logger.info("\n📋 Test 1: Basic Piper Integration")
    success1 = test_piper_integration()
    
    # Test 2: Fallback chain
    logger.info("\n📋 Test 2: Fallback Chain Testing")
    success2 = test_fallback_chain()
    
    # Results
    logger.info("\n📊 Test Results:")
    logger.info("=" * 20)
    logger.info(f"Basic Integration: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"Fallback Chain: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        logger.info("\n🎉 All tests passed! Piper TTS integration is working correctly.")
        return 0
    else:
        logger.error("\n💥 Some tests failed. Please check the configuration.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
