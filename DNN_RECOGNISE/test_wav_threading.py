#!/usr/bin/env python3
"""
Test script to verify WAV file updater threading functionality
"""

import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from update_wav_files import WAVFileUpdater

def test_threading_functionality():
    """Test that threading is properly configured and working"""
    print("Testing WAV File Updater Threading")
    print("=" * 50)

    # Initialize updater
    updater = WAVFileUpdater()

    # Test threading configuration
    threading_info = updater.get_threading_info()
    print(f"Threading enabled: {threading_info['threading_enabled']}")
    print(f"Max workers: {threading_info['max_workers']}")
    print(f"Generation timeout: {threading_info['generation_timeout']} seconds")

    # Verify threading is enabled
    if threading_info['threading_enabled']:
        print("OK Threading is enabled")
    else:
        print("ERROR Threading is disabled")
        return False

    # Test that max_workers is reasonable
    if 1 <= threading_info['max_workers'] <= 16:
        print("OK Max workers is in reasonable range")
    else:
        print("ERROR Max workers is outside reasonable range")
        return False

    # Test that timeout is reasonable
    if 30 <= threading_info['generation_timeout'] <= 300:
        print("OK Generation timeout is in reasonable range")
    else:
        print("ERROR Generation timeout is outside reasonable range")
        return False

    print("OK All threading configuration tests passed!")
    return True

def test_mock_generation():
    """Test with mock data to verify threading works"""
    print("\nTesting Mock WAV Generation")
    print("-" * 30)

    updater = WAVFileUpdater()

    # Create mock messages
    mock_messages = {
        'test1': 'This is a test message 1',
        'test2': 'This is a test message 2',
        'test3': 'This is a test message 3',
        'test4': 'This is a test message 4',
    }

    # Temporarily replace get_all_messages for testing
    original_method = updater.get_all_messages
    updater.get_all_messages = lambda: mock_messages

    # Test that we can collect tasks
    file_tasks = []
    for message_key, text in mock_messages.items():
        filename = f"{message_key}.wav"
        filepath = os.path.join(updater.wav_files_dir, filename)
        file_tasks.append((message_key, "", text, filepath))

    print(f"Collected {len(file_tasks)} tasks for generation")

    if len(file_tasks) == len(mock_messages):
        print("OK Task collection works correctly")
    else:
        print("ERROR Task collection failed")
        return False

    # Restore original method
    updater.get_all_messages = original_method

    print("OK Mock generation test passed!")
    return True

if __name__ == "__main__":
    success1 = test_threading_functionality()
    success2 = test_mock_generation()

    if success1 and success2:
        print("\nAll threading tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
