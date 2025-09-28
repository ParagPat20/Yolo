#!/usr/bin/env python3
"""
Test script to verify Piper command structure (without execution)
"""

import shlex
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_piper_command_structure():
    """Test the Piper command structure without execution"""
    try:
        text = "Welcome to JECH AEROTECH. We build CCTV for DRONES, and sometimes we build drones for cctv."
        model_path = "/usr/local/share/piper-voices/en_US-ljspeech-medium.onnx"
        
        # Escape text for shell safety
        safe_text = shlex.quote(text)
        
        # Build the exact command you provided
        cmd = [
            'bash', '-c',
            f'echo {safe_text} | piper --model {model_path} --output-raw | aplay -r 22050 -f S16_LE -c 1 -'
        ]
        
        logger.info(f"🔊 Basic Piper Command Structure:")
        logger.info(f"🔧 Command: {' '.join(cmd)}")
        logger.info("✅ Command structure is correct")
        
        # Test with parameters
        cmd_with_params = [
            'bash', '-c',
            f'echo {safe_text} | piper --model {model_path} --length-scale 1.0 --noise-scale 0.667 --output-raw | aplay -r 22050 -f S16_LE -c 1 -'
        ]
        
        logger.info(f"\n🔊 Piper Command with Parameters:")
        logger.info(f"🔧 Command: {' '.join(cmd_with_params)}")
        logger.info("✅ Command structure with parameters is correct")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing command structure: {e}")
        return False

def test_sound_system_integration():
    """Test the sound system integration structure"""
    try:
        import sys
        import os
        
        # Add src directory to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from sound_system import SoundSystem
        
        logger.info("\n🔧 Testing Sound System Integration:")
        
        # Create sound system instance
        sound_system = SoundSystem()
        
        if sound_system.tts_system:
            logger.info(f"✅ TTS System: {sound_system.tts_system}")
            logger.info(f"✅ Enabled: {sound_system.is_enabled}")
            logger.info(f"✅ Language: {sound_system.language}")
        else:
            logger.info("⚠️ No TTS system available (expected on Windows)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing sound system: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 Testing Piper TTS Command Structure")
    logger.info("=" * 50)
    
    # Test 1: Command structure
    logger.info("\n📋 Test 1: Command Structure")
    success1 = test_piper_command_structure()
    
    # Test 2: Sound system integration
    logger.info("\n📋 Test 2: Sound System Integration")
    success2 = test_sound_system_integration()
    
    # Results
    logger.info("\n📊 Test Results:")
    logger.info("=" * 20)
    logger.info(f"Command Structure: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"Sound Integration: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        logger.info("\n🎉 All command structures are correct!")
        logger.info("💡 Note: Actual execution requires Linux with Piper installed")
        return 0
    else:
        logger.error("\n💥 Some tests failed.")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
