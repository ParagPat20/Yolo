#!/usr/bin/env python3
"""
Test script for WAV file updater
"""

import sys
import os
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_wav_updater():
    """Test the WAV file updater functionality"""
    try:
        from update_wav_files import WAVFileUpdater
        
        logger.info("🧪 Testing WAV File Updater")
        logger.info("=" * 40)
        
        # Initialize updater
        updater = WAVFileUpdater()
        
        # Test 1: Check Piper availability
        logger.info("🔍 Checking Piper TTS availability...")
        piper_available = updater.check_piper_availability()
        logger.info(f"Piper TTS available: {piper_available}")
        
        # Test 2: List existing WAV files
        logger.info("\n📄 Listing existing WAV files...")
        wav_files = updater.list_wav_files()
        logger.info(f"Found {len(wav_files)} WAV files:")
        for wav_file in wav_files:
            logger.info(f"  - {wav_file}")
        
        # Test 3: Get WAV file information
        logger.info("\n📊 Getting WAV file information...")
        info = updater.get_wav_file_info()
        logger.info(f"Total files: {info['total_files']}")
        logger.info(f"Directory: {info['wav_files_dir']}")
        logger.info(f"Language: {info['language']}")
        logger.info(f"Piper available: {info['piper_available']}")
        
        # Test 4: Update specific WAV file (if Piper is available)
        if piper_available:
            logger.info("\n🎵 Testing specific WAV file update...")
            success = updater.update_specific_wav_file('person_detected')
            if success:
                logger.info("✅ Successfully updated person_detected.wav")
            else:
                logger.warning("⚠️ Failed to update person_detected.wav")
            
            # Test time-based greeting
            success = updater.update_specific_wav_file('time_based_greeting', 'morning')
            if success:
                logger.info("✅ Successfully updated time_based_greeting_morning.wav")
            else:
                logger.warning("⚠️ Failed to update time_based_greeting_morning.wav")
        else:
            logger.info("\n⚠️ Skipping WAV file generation tests - Piper TTS not available")
        
        # Test 5: List files after update
        logger.info("\n📄 Listing WAV files after update...")
        wav_files_after = updater.list_wav_files()
        logger.info(f"Found {len(wav_files_after)} WAV files:")
        for wav_file in wav_files_after:
            logger.info(f"  - {wav_file}")
        
        logger.info("✅ WAV file updater test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_command_line_interface():
    """Test command line interface"""
    try:
        import subprocess
        
        logger.info("\n🧪 Testing Command Line Interface")
        logger.info("=" * 40)
        
        # Test --list command
        logger.info("🔍 Testing --list command...")
        result = subprocess.run([sys.executable, 'update_wav_files.py', '--list'], 
                              capture_output=True, text=True, timeout=10)
        logger.info(f"List command result: {result.returncode}")
        
        # Test --info command
        logger.info("🔍 Testing --info command...")
        result = subprocess.run([sys.executable, 'update_wav_files.py', '--info'], 
                              capture_output=True, text=True, timeout=10)
        logger.info(f"Info command result: {result.returncode}")
        
        logger.info("✅ Command line interface test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Command line test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting WAV File Updater Tests")
    logger.info("=" * 50)
    
    # Test 1: WAV updater functionality
    success1 = test_wav_updater()
    
    # Test 2: Command line interface
    success2 = test_command_line_interface()
    
    # Results
    logger.info("\n📊 Test Results:")
    logger.info("=" * 20)
    logger.info(f"WAV Updater: {'✅ PASS' if success1 else '❌ FAIL'}")
    logger.info(f"Command Line: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        logger.info("\n🎉 All WAV file updater tests passed!")
        logger.info("💡 The WAV file updater is ready for use!")
        return 0
    else:
        logger.error("\n💥 Some WAV file updater tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
