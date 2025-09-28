#!/usr/bin/env python3
"""
Direct test of Piper command execution
"""

import subprocess
import shlex
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_piper_direct():
    """Test Piper command directly like sound_system.py does"""
    try:
        text = "Hello! This is a direct test of the Piper TTS system."
        model_path = "/usr/local/share/piper-voices/en_US-ljspeech-medium.onnx"
        
        # Escape text for shell safety
        safe_text = shlex.quote(text)
        
        # Build the exact command from sound_system.py
        cmd = [
            'bash', '-c',
            f'echo {safe_text} | piper --model {model_path} --length-scale 1.0 --noise-scale 0.667 --output-raw | aplay -r 22050 -f S16_LE -c 1 -'
        ]
        
        logger.info(f"🔊 Testing direct Piper command: {text}")
        logger.debug(f"🔧 Command: {' '.join(cmd)}")
        
        # Execute exactly like sound_system.py does
        logger.info("🔊 Speaking with Piper...")
        process = subprocess.Popen(cmd, stdout=None, stderr=None)
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("✅ Piper command executed successfully")
            return True
        else:
            logger.error(f"❌ Piper command failed with return code: {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing Piper: {e}")
        return False

def test_piper_with_capture():
    """Test Piper command with output capture (for debugging)"""
    try:
        text = "Hello! This is a test with output capture."
        model_path = "/usr/local/share/piper-voices/en_US-ljspeech-medium.onnx"
        
        # Escape text for shell safety
        safe_text = shlex.quote(text)
        
        # Build command
        cmd = [
            'bash', '-c',
            f'echo {safe_text} | piper --model {model_path} --length-scale 1.0 --noise-scale 0.667 --output-raw | aplay -r 22050 -f S16_LE -c 1 -'
        ]
        
        logger.info(f"🔊 Testing Piper with capture: {text}")
        
        # Execute with capture for debugging
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        logger.info(f"Return code: {result.returncode}")
        if result.stdout:
            logger.info(f"STDOUT: {result.stdout}")
        if result.stderr:
            logger.info(f"STDERR: {result.stderr}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Piper command timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing Piper with capture: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 Testing Direct Piper Execution")
    logger.info("=" * 40)
    
    # Test 1: Direct execution (like sound_system.py)
    logger.info("\n📋 Test 1: Direct Piper Execution")
    success1 = test_piper_direct()
    
    # Test 2: With capture (for debugging)
    logger.info("\n📋 Test 2: Piper with Output Capture")
    success2 = test_piper_with_capture()
    
    # Results
    logger.info("\n📊 Test Results:")
    logger.info("=" * 20)
    logger.info(f"Direct Execution: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"With Capture: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1:
        logger.info("\n🎉 Direct Piper execution works!")
        return 0
    else:
        logger.error("\n💥 Direct Piper execution failed!")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
