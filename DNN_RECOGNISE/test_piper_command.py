#!/usr/bin/env python3
"""
Test script to verify Piper command works correctly
"""

import subprocess
import shlex
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_piper_command():
    """Test the exact Piper command that should work"""
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
        
        logger.info(f"🔊 Testing Piper command: {text}")
        logger.info(f"🔧 Command: {' '.join(cmd)}")
        
        # Execute command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✅ Piper command executed successfully")
            return True
        else:
            logger.error(f"❌ Piper command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Piper command timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing Piper: {e}")
        return False

def test_piper_with_parameters():
    """Test Piper with all parameters"""
    try:
        text = "Hello! This is a test of the Piper TTS system with parameters."
        model_path = "/usr/local/share/piper-voices/en_US-ljspeech-medium.onnx"
        
        # Escape text for shell safety
        safe_text = shlex.quote(text)
        
        # Build command with all parameters (correct Piper parameters)
        cmd = [
            'bash', '-c',
            f'echo {safe_text} | piper --model {model_path} --length-scale 1.0 --noise-scale 0.667 --output-raw | aplay -r 22050 -f S16_LE -c 1 -'
        ]
        
        logger.info(f"🔊 Testing Piper with parameters: {text}")
        logger.info(f"🔧 Command: {' '.join(cmd)}")
        
        # Execute command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✅ Piper command with parameters executed successfully")
            return True
        else:
            logger.error(f"❌ Piper command with parameters failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Piper command with parameters timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing Piper with parameters: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 Testing Piper TTS Commands")
    logger.info("=" * 40)
    
    # Test 1: Basic command
    logger.info("\n📋 Test 1: Basic Piper Command")
    success1 = test_piper_command()
    
    # Test 2: Command with parameters
    logger.info("\n📋 Test 2: Piper Command with Parameters")
    success2 = test_piper_with_parameters()
    
    # Results
    logger.info("\n📊 Test Results:")
    logger.info("=" * 20)
    logger.info(f"Basic Command: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"With Parameters: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        logger.info("\n🎉 All Piper commands work correctly!")
        return 0
    else:
        logger.error("\n💥 Some Piper commands failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
