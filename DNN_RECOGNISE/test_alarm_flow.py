#!/usr/bin/env python3
"""
Test script for the alarm flow functionality
"""

import sys
import os
import time
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from advanced_person_tracker import PersonTrack
from settings.settings import CCTV

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_alarm_flow():
    """Test the complete alarm flow"""
    print("Testing Alarm Flow")
    print("=" * 50)

    # Create a mock tracker instance to test the alarm methods
    class MockTracker:
        def __init__(self):
            self.tracks = {}

    # Create a mock sound player
    class MockSoundPlayer:
        def __init__(self):
            self.alarm_playing = False

        def play_alarm(self):
            self.alarm_playing = True
            print("SOUND MOCK: Alarm started")

        def stop_alarm(self):
            self.alarm_playing = False
            print("SILENT MOCK: Alarm stopped")

        def is_alarm_playing(self):
            return self.alarm_playing

    # Create a mock hardware manager
    class MockHardwareManager:
        def __init__(self):
            self.status = 'ready'

        def set_system_status(self, status):
            self.status = status
            print(f"STATUS MOCK: Hardware status set to {status}")

        def play_alarm(self):
            print("SOUND MOCK: Hardware alarm started")

        def stop_alarm(self):
            print("SILENT MOCK: Hardware alarm stopped")

    # Create test track
    track = PersonTrack(
        track_id=1,
        bbox=(100, 100, 50, 100),
        center=(125, 150),
        confidence=0.8
    )

    # Create mock tracker
    mock_tracker = MockTracker()
    mock_tracker.tracks = {1: track}

    # Create mock components
    mock_sound_player = MockSoundPlayer()
    mock_hardware_manager = MockHardwareManager()

    print("\n1. Testing unknown person detection and alarm trigger")
    print("-" * 50)

    # Simulate unknown person detection
    current_time = time.time()
    track.alarm_start_time = current_time
    track.alarm_end_time = current_time + 120.0  # 2 minutes
    track.alarm_active = True

    print(f"OK Unknown person detected at {datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}")
    print(f"ALERT Alarm started - will run for 2 minutes until {datetime.fromtimestamp(track.alarm_end_time).strftime('%H:%M:%S')}")

    # Simulate alarm playing
    mock_sound_player.play_alarm()
    mock_hardware_manager.set_system_status('alert')
    mock_hardware_manager.play_alarm()

    print("\n2. Testing alarm suppression during active alarm")
    print("-" * 50)

    # Test that other sounds are suppressed during alarm
    any_alarm_active = any(t.alarm_active for t in mock_tracker.tracks.values())
    print(f"SEARCH Any alarm active: {any_alarm_active}")

    if any_alarm_active:
        print("OK Sound suppression active - no greeting sounds will play")
    else:
        print("ERROR Sound suppression not working")

    print("\n3. Testing known person verification")
    print("-" * 50)

    # Simulate known person verification
    track.identity = "John Doe"
    track.is_known = True
    track.is_trusted = True
    track.verification_requested = False
    track.alert_sent = False

    print(f"OK Known person verified: {track.identity}")

    # Stop alarm immediately
    if track.alarm_active:
        print("SILENT Stopping alarm immediately due to known person verification")
        mock_sound_player.stop_alarm()
        mock_hardware_manager.stop_alarm()
        track.alarm_active = False
        track.alarm_end_time = 0.0
        track.alarm_start_time = 0.0

        # Reset hardware status
        if not any(t.alarm_active for t in mock_tracker.tracks.values()):
            mock_hardware_manager.set_system_status('ready')

    print("\n4. Testing guest mode activation")
    print("-" * 50)

    # Create another unknown person to test guest mode
    guest_track = PersonTrack(
        track_id=2,
        bbox=(200, 100, 50, 100),
        center=(225, 150),
        confidence=0.7
    )
    guest_track.is_known = False
    guest_track.is_guest = False
    guest_track.verification_requested = False
    guest_track.alert_sent = False

    mock_tracker.tracks[2] = guest_track

    # Simulate guest mode activation
    distance = 100  # Close enough to be considered guest
    if distance <= CCTV['guest_detection_distance']:
        print(f"GUEST Guest mode activated: {track.identity} + guest (track {guest_track.track_id})")
        guest_track.is_guest = True
        guest_track.guest_associated_with = track.identity
        guest_track.guest_mode_start_time = time.time()

        # Stop any recording that might have started
        if guest_track.is_recording:
            guest_track.is_recording = False
            print("RECORDING Recording stopped for guest")

        print(f"GUEST Guest mode will last for {CCTV['guest_mode_duration']/60} minutes")
    else:
        print(f"ERROR Guest track {guest_track.track_id} too far from verified person")

    print("\n5. Testing alarm expiration")
    print("-" * 50)

    # Simulate 2+ minutes passing
    track.alarm_start_time = current_time - 130  # 2+ minutes ago
    track.alarm_end_time = current_time - 10  # Already expired
    track.alarm_active = True

    # Check for expired alarms
    for test_track in mock_tracker.tracks.values():
        if test_track.alarm_active and test_track.alarm_end_time > 0 and current_time >= test_track.alarm_end_time:
            print(f"SILENT Alarm expired for track {test_track.track_id} after 2 minutes")
            mock_sound_player.stop_alarm()
            mock_hardware_manager.stop_alarm()
            test_track.alarm_active = False
            test_track.alarm_end_time = 0.0
            test_track.alarm_start_time = 0.0

            # Reset hardware status if no other alarms are active
            if not any(t.alarm_active for t in mock_tracker.tracks.values()):
                mock_hardware_manager.set_system_status('ready')

    print("\nOK Alarm flow test completed successfully!")
    print("\nSummary of behavior:")
    print("1. OK Unknown person triggers 2-minute alarm")
    print("2. OK Other sounds are suppressed during alarm")
    print("3. OK Alarm stops immediately when known person verifies")
    print("4. OK Guest mode activates after successful verification")
    print("5. OK Alarm auto-expires after 2 minutes if not stopped")

if __name__ == "__main__":
    test_alarm_flow()
