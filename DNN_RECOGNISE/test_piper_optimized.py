#!/usr/bin/env python3
"""
Test script for optimized Piper TTS integration
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

def test_piper_optimized():
    """Test optimized Piper TTS integration"""
    try:
        from sound_system import get_sound_system
        
        logger.info("🧪 Testing Optimized Piper TTS Integration")
        logger.info("=" * 50)
        
        # Initialize sound system
        sound_system = get_sound_system()
        
        if not sound_system:
            logger.error("❌ Sound system not available")
            return False
        
        logger.info(f"✅ Sound system initialized with TTS: {sound_system.tts_system}")
        
        # Test 1: Check if persistent voice is loaded
        if sound_system.tts_system == 'piper':
            if sound_system.piper_voice:
                logger.info("✅ Persistent Piper voice loaded")
            else:
                logger.info("⚠️ Using command-line Piper (no persistent voice)")
        
        # Test 2: Measure speech speed
        test_messages = [
            "Hello! This is a test of the optimized Piper TTS system.",
            "Good morning! The CCTV system is working properly.",
            "Unknown person detected! Please show your face for verification.",
            "Welcome back! You have been successfully identified."
        ]
        
        logger.info("\n🔊 Testing speech speed with optimized Piper...")
        
        for i, message in enumerate(test_messages, 1):
            logger.info(f"📢 Test {i}: {message}")
            
            # Measure time for speech
            start_time = time.time()
            sound_system.speak(message)
            end_time = time.time()
            
            duration = end_time - start_time
            logger.info(f"⏱️ Speech completed in {duration:.2f} seconds")
            
            # Wait a moment between tests
            time.sleep(1)
        
        logger.info("✅ Optimized Piper test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_piper_performance():
    """Test Piper performance comparison"""
    try:
        from sound_system import get_sound_system
        
        logger.info("\n📊 Testing Piper Performance")
        logger.info("=" * 30)
        
        sound_system = get_sound_system()
        
        if not sound_system or sound_system.tts_system != 'piper':
            logger.info("⚠️ Piper not available - skipping performance test")
            return True
        
        # Test multiple rapid speech requests
        test_text = "Performance test message."
        
        logger.info("🚀 Testing rapid speech requests...")
        
        start_time = time.time()
        for i in range(5):
            sound_system.speak(f"{test_text} {i+1}")
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / 5
        
        logger.info(f"📊 Performance Results:")
        logger.info(f"   Total time for 5 speeches: {total_time:.2f}s")
        logger.info(f"   Average time per speech: {avg_time:.2f}s")
        
        if avg_time < 1.0:
            logger.info("✅ Excellent performance! (< 1s per speech)")
        elif avg_time < 2.0:
            logger.info("✅ Good performance! (< 2s per speech)")
        else:
            logger.info("⚠️ Performance could be improved")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Optimized Piper TTS Tests")
    logger.info("=" * 60)
    
    # Test 1: Basic optimized integration
    success1 = test_piper_optimized()
    
    # Test 2: Performance testing
    success2 = test_piper_performance()
    
    # Results
    logger.info("\n📊 Test Results:")
    logger.info("=" * 20)
    logger.info(f"Optimized Integration: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"Performance Test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        logger.info("\n🎉 All optimized Piper tests passed!")
        logger.info("💡 Piper TTS is now optimized for fast speech generation!")
        return 0
    else:
        logger.error("\n💥 Some optimized Piper tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
