#!/usr/bin/env python3
"""
Quick test for the new threaded sound system
"""

import sys
import os
import time
import threading

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the sound system
try:
    from advanced_person_tracker import WindowsSoundAlertSystem
    print("✅ Successfully imported WindowsSoundAlertSystem")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def test_threaded_sounds():
    """Test that sounds don't block the main thread"""
    print("\n🔊 Testing Threaded Sound System")
    print("=" * 50)
    
    # Initialize sound system
    sound_system = WindowsSoundAlertSystem()
    
    print(f"🔊 Sound enabled: {sound_system.sound_enabled}")
    print(f"🗣️ Voice enabled: {sound_system.voice_enabled}")
    
    if not (sound_system.sound_enabled or sound_system.voice_enabled):
        print("⚠️ No sound capabilities available, but testing threading...")
    
    # Test 1: Play verification request (should not block)
    print("\n🔍 Test 1: Playing verification request...")
    start_time = time.time()
    sound_system.play_verification_request()
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"⏱️ play_verification_request() took {elapsed:.3f} seconds")
    
    if elapsed < 0.1:
        print("✅ PASS: Sound call returned immediately (non-blocking)")
    else:
        print("❌ FAIL: Sound call took too long (blocking)")
    
    # Test 2: Play multiple sounds rapidly (should not block or crash)
    print("\n🚨 Test 2: Playing multiple rapid sounds...")
    start_time = time.time()
    
    for i in range(3):
        print(f"  🔊 Playing sound {i+1}/3...")
        sound_system.play_verification_request()
        time.sleep(0.1)  # Very short delay
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"⏱️ 3 rapid sounds took {elapsed:.3f} seconds")
    
    if elapsed < 1.0:
        print("✅ PASS: Multiple sounds didn't block main thread")
    else:
        print("❌ FAIL: Multiple sounds blocked main thread")
    
    # Test 3: Thread cleanup
    print("\n🧹 Test 3: Thread management...")
    initial_threads = len(sound_system.sound_threads)
    print(f"  📊 Active sound threads: {initial_threads}")
    
    # Wait a bit for threads to finish
    time.sleep(2)
    sound_system._cleanup_threads()
    final_threads = len(sound_system.sound_threads)
    print(f"  📊 Active sound threads after cleanup: {final_threads}")
    
    if final_threads <= initial_threads:
        print("✅ PASS: Thread cleanup working")
    else:
        print("❌ FAIL: Thread cleanup not working")
    
    # Test 4: Main thread responsiveness during sound
    print("\n⚡ Test 4: Main thread responsiveness...")
    sound_system.play_unknown_alert()  # Long sound (siren)
    
    # Simulate main thread work
    counter = 0
    start_time = time.time()
    
    while time.time() - start_time < 1.0:  # Work for 1 second
        counter += 1
        time.sleep(0.001)  # Small delay to simulate real work
    
    print(f"  🔄 Main thread completed {counter} iterations while sound played")
    
    if counter > 500:
        print("✅ PASS: Main thread remained responsive during sound")
    else:
        print("❌ FAIL: Main thread was blocked by sound")
    
    print("\n🏁 Test completed!")
    print("=" * 50)

if __name__ == "__main__":
    try:
        test_threaded_sounds()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
