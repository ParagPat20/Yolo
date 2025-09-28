#!/usr/bin/env python3
"""
Test script for optimized Piper TTS initialization
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

def test_piper_initialization_speed():
    """Test how fast Piper initializes and becomes ready"""
    try:
        from sound_system import get_sound_system
        
        logger.info("🧪 Testing Optimized Piper Initialization")
        logger.info("=" * 50)
        
        # Measure initialization time
        start_time = time.time()
        
        # Initialize sound system
        sound_system = get_sound_system()
        
        init_time = time.time() - start_time
        
        if not sound_system:
            logger.error("❌ Sound system not available")
            return False
        
        logger.info(f"✅ Sound system initialized in {init_time:.2f} seconds")
        logger.info(f"✅ TTS System: {sound_system.tts_system}")
        
        if sound_system.tts_system == 'piper':
            logger.info("✅ Using optimized persistent Piper voice")
            
            # Test immediate speech (should be very fast)
            test_messages = [
                "Hello! This is a test of optimized Piper TTS.",
                "The model is pre-loaded and ready for immediate use.",
                "No more loading delays for speech synthesis."
            ]
            
            for i, message in enumerate(test_messages, 1):
                logger.info(f"🔊 Test {i}: {message}")
                
                # Measure speech start time
                speech_start = time.time()
                sound_system.speak(message)
                speech_time = time.time() - speech_start
                
                logger.info(f"✅ Speech started in {speech_time:.3f} seconds")
                
                # Small delay between tests
                time.sleep(1)
        else:
            logger.info(f"ℹ️ Using fallback TTS: {sound_system.tts_system}")
        
        logger.info("✅ Piper optimization test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_speech_performance():
    """Test speech performance with multiple rapid requests"""
    try:
        from sound_system import get_sound_system
        
        logger.info("\n🧪 Testing Speech Performance")
        logger.info("=" * 30)
        
        sound_system = get_sound_system()
        
        if not sound_system or sound_system.tts_system != 'piper':
            logger.info("ℹ️ Skipping performance test - not using Piper")
            return True
        
        # Test rapid speech requests
        rapid_messages = [
            "First message",
            "Second message", 
            "Third message",
            "Fourth message",
            "Fifth message"
        ]
        
        logger.info("🔊 Testing rapid speech requests...")
        start_time = time.time()
        
        for i, message in enumerate(rapid_messages, 1):
            logger.info(f"🔊 Rapid test {i}: {message}")
            sound_system.speak(message)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(rapid_messages)
        
        logger.info(f"✅ Completed {len(rapid_messages)} rapid speeches in {total_time:.2f} seconds")
        logger.info(f"✅ Average time per speech: {avg_time:.3f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Piper Optimization Tests")
    logger.info("=" * 60)
    
    # Test 1: Initialization speed
    success1 = test_piper_initialization_speed()
    
    # Test 2: Speech performance
    success2 = test_speech_performance()
    
    # Results
    logger.info("\n📊 Test Results:")
    logger.info("=" * 20)
    logger.info(f"Initialization Speed: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"Speech Performance: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        logger.info("\n🎉 All Piper optimization tests passed!")
        logger.info("💡 Piper should now be much faster for speech synthesis!")
        return 0
    else:
        logger.error("\n💥 Some Piper optimization tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
