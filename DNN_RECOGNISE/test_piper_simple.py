#!/usr/bin/env python3
"""
Simple test for Piper TTS integration
"""

import sys
import os
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_piper_simple():
    """Test simple Piper integration"""
    try:
        from sound_system import get_sound_system
        
        logger.info("🧪 Testing Simple Piper Integration")
        logger.info("=" * 40)
        
        # Initialize sound system
        sound_system = get_sound_system()
        
        if not sound_system:
            logger.error("❌ Sound system not available")
            return False
        
        logger.info(f"✅ Sound system initialized with TTS: {sound_system.tts_system}")
        
        # Test simple speech
        test_message = "Hello! This is a test of the Piper TTS system."
        logger.info(f"🔊 Testing speech: {test_message}")
        
        # Speak the message
        sound_system.speak(test_message)
        
        logger.info("✅ Piper test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Simple Piper Test")
    logger.info("=" * 30)
    
    success = test_piper_simple()
    
    if success:
        logger.info("\n🎉 Piper integration test completed!")
        return 0
    else:
        logger.error("\n💥 Piper integration test failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
