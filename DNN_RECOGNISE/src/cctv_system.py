# Raspberry Pi CCTV Camera System
# Advanced person tracking with face recognition, motion detection, and smart alerts
import sys
sys.path.append('/usr/lib/python3/dist-packages')

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import logging
import time
import threading
from datetime import datetime
from typing import Optional, Tuple, List
# Unused imports removed

from settings.settings import (
    CAMERA, CCTV, HARDWARE, PATHS
)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import our components
try:
    from hardware_interface import get_hardware_manager, PIRMotionDetector
    HARDWARE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("🔧 Hardware interface available")
except ImportError:
    HARDWARE_AVAILABLE = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("🔧 Hardware interface not available")

try:
    from advanced_person_tracker import AdvancedPersonTracker, PersonTrack
    TRACKER_AVAILABLE = True
    logger.info("🎯 Advanced person tracker available")
except ImportError:
    TRACKER_AVAILABLE = False
    logger.error("❌ Advanced person tracker not available")

# Try to import sound player
try:
    from sound_player import get_sound_player
    SOUND_PLAYER_AVAILABLE = True
    logger.info("🔊 Sound player available")
except ImportError:
    SOUND_PLAYER_AVAILABLE = False
    logger.warning("🔊 Sound player not available")

class CCTVSystem:
    """Main CCTV System integrating all components"""

    def __init__(self):
        logger.info("🚀 Initializing Raspberry Pi CCTV System...")

        # Initialize hardware
        self.hardware_manager = get_hardware_manager() if HARDWARE_AVAILABLE else None

        # Initialize person tracker
        if not TRACKER_AVAILABLE:
            raise ImportError("Advanced person tracker is required for CCTV system")

        self.person_tracker = AdvancedPersonTracker()
        
        # Set CCTV system reference in person tracker for background sound
        self.person_tracker.set_cctv_system(self)

        # Initialize sound player
        if SOUND_PLAYER_AVAILABLE:
            try:
                self.sound_player = get_sound_player()
                logger.info("🔊 Sound player initialized")
            except Exception as e:
                logger.error(f"Failed to initialize sound player: {e}")
                self.sound_player = None
        else:
            self.sound_player = None

        # System state
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        self.motion_detected = False
        self.last_motion_time = 0

        # Guest mode tracking
        self.guest_mode_active = False

        # Camera
        self.camera = None
        self._continuous_af_enabled = False

        logger.info("✅ CCTV System initialized successfully")

    def initialize_camera(self) -> bool:
        """Initialize camera with enhanced settings"""
        try:
            # Try to use `picamera2` first
            try:
                from picamera2 import Picamera2, Preview
                try:
                    from libcamera import controls as LIBCAM_CONTROLS
                except Exception:
                    LIBCAM_CONTROLS = None
                self.camera = Picamera2()

                # Configure with high resolution for CCTV
                preview_config = self.camera.create_preview_configuration(
                    main={"size": (CAMERA['width'], CAMERA['height'])},
                    lores={"size": (640, 360)},
                    display="lores"
                )
                self.camera.configure(preview_config)
                # Prefer continuous autofocus if available, else normal AF
                if LIBCAM_CONTROLS and hasattr(LIBCAM_CONTROLS, 'AfModeEnum'):
                    try:
                        self.camera.set_controls({"AfMode": LIBCAM_CONTROLS.AfModeEnum.Continuous, "FrameRate": CAMERA['fps']})
                        self._continuous_af_enabled = True
                    except Exception as e:
                        logger.warning(f"Continuous AF set failed, using normal AF: {e}")
                        self.camera.set_controls({"AfMode": 1, "AfTrigger": 0, "FrameRate": CAMERA['fps']})
                        self._continuous_af_enabled = False
                else:
                    self.camera.set_controls({"AfMode": 1, "AfTrigger": 0, "FrameRate": CAMERA['fps']})
                    self._continuous_af_enabled = False
                self.camera.start()

                logger.info(f"✅ Picamera2 initialized: {CAMERA['width']}x{CAMERA['height']} @ {CAMERA['fps']}fps with autofocus")
                return True

            except ImportError:
                logger.warning("📷 Picamera2 not available, falling back to OpenCV")

            # Fallback to OpenCV
            self.camera = cv2.VideoCapture(CAMERA['index'])
            if not self.camera.isOpened():
                logger.error("❌ Could not open camera")
                return False

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
            self.camera.set(cv2.CAP_PROP_FPS, CAMERA['fps'])

            logger.info(f"✅ OpenCV camera initialized: {CAMERA['width']}x{CAMERA['height']} @ {CAMERA['fps']}fps")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize camera: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera"""
        try:
            if hasattr(self.camera, 'capture_array'):
                # Picamera2
                frame = self.camera.capture_array("main")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame
            else:
                # OpenCV
                return self.camera.read()
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None

    def handle_motion_detection(self):
        """Handle motion detection event"""
        current_time = time.time()

        if current_time - self.last_motion_time < CCTV['motion_cooldown']:
            return

        self.last_motion_time = current_time
        self.motion_detected = True

        logger.info("🚶 Motion detected - activating enhanced monitoring")

        # Turn on brightness LED
        if self.hardware_manager:
            self.hardware_manager.led_controller.turn_on_brightness(CCTV['led_brightness_duration'])

        # Set status to alert
        if self.hardware_manager:
            self.hardware_manager.set_system_status('alert')

    def run(self):
        """Main CCTV system loop"""
        if not self.initialize_camera():
            logger.error("❌ Failed to initialize camera")
            return

        self.running = True
        logger.info("🎯 CCTV System started - monitoring for persons and faces")

        # Set initial status
        if self.hardware_manager:
            self.hardware_manager.set_system_status('ready')

        try:
            last_af_trigger = time.time()
            while self.running:
                # Read frame
                ret, frame = self.read_frame()
                if not ret or frame is None:
                    logger.warning("Failed to grab frame")
                    continue

                # If continuous AF not available, periodically trigger AF in auto mode
                if hasattr(self.camera, 'set_controls') and not self._continuous_af_enabled:
                    current_time = time.time()
                    if current_time - last_af_trigger > 2.0:
                        try:
                            self.camera.set_controls({"AfTrigger": 0})
                            last_af_trigger = current_time
                        except Exception as e:
                            logger.debug(f"AF trigger failed: {e}")

                # Process frame with person tracker
                annotated_frame, tracks = self.person_tracker.process_frame(frame)

                # Update guest mode status
                self.update_guest_mode_status(tracks)

                # Update FPS counter
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_update)
                    self.frame_count = 0
                    self.last_fps_update = current_time

                # Add FPS to frame
                cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Show frame
                cv2.imshow('Raspberry Pi CCTV System', annotated_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_snapshot(annotated_frame)
                elif key == ord('r'):
                    self.reset_system()
                elif key == ord('m'):
                    self.toggle_motion_detection()

            logger.info("👋 CCTV System stopped")

        except KeyboardInterrupt:
            logger.info("⚠️ CCTV System interrupted by user")
        except Exception as e:
            logger.error(f"❌ CCTV System error: {e}")
        finally:
            self.cleanup()

    def save_snapshot(self, frame: np.ndarray):
        """Save current frame as snapshot"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cctv_snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"📸 Snapshot saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")

    def reset_system(self):
        """Reset the CCTV system"""
        logger.info("🔄 Resetting CCTV system...")

        # Clear all tracks
        if hasattr(self.person_tracker, 'tracker'):
            self.person_tracker.tracker.tracks.clear()
            self.person_tracker.tracker.next_id = 1

        # Reset hardware status
        if self.hardware_manager:
            self.hardware_manager.set_system_status('ready')

        logger.info("✅ System reset complete")

    def toggle_motion_detection(self):
        """Toggle motion detection"""
        if self.hardware_manager and self.hardware_manager.pir_detector:
            # Toggle would require modifying the PIR detector
            logger.info("Motion detection toggle not implemented yet")

    def update_guest_mode_status(self, tracks: List):
        """Update guest mode status based on current tracks"""
        # Check if any track is currently in guest mode
        current_guest_mode = any(track.is_guest for track in tracks)

        if current_guest_mode != self.guest_mode_active:
            self.guest_mode_active = current_guest_mode
            if self.hardware_manager:
                if current_guest_mode:
                    # Guest mode activated - hardware manager already handles this
                    pass
                else:
                    # Guest mode deactivated - revert to ready status
                    self.hardware_manager.set_system_status('ready')
            logger.info(f"👥 Guest mode {'activated' if current_guest_mode else 'deactivated'}")

    def play_background_sound(self, sound_type: str, **kwargs):
        """Play sound in background thread to avoid blocking main CCTV process"""
        if not self.sound_player:
            return
        
        def play_sound():
            try:
                if sound_type == 'person_detected':
                    self.sound_player.play_person_detected()
                elif sound_type == 'face_verification_request':
                    self.sound_player.play_verification_request()
                elif sound_type == 'face_verification_reminder':
                    count = kwargs.get('count', 1)
                    self.sound_player.play_verification_reminder(count)
                elif sound_type == 'unknown_person_alert':
                    self.sound_player.play_unknown_person_alert()
                elif sound_type == 'security_breach':
                    self.sound_player.play_security_breach()
                elif sound_type == 'known_person_greeting':
                    name = kwargs.get('name', '')
                    self.sound_player.play_known_person_greeting(name)
                elif sound_type == 'time_based_greeting':
                    self.sound_player.play_time_based_greeting()
                elif sound_type == 'welcome_back':
                    name = kwargs.get('name', '')
                    self.sound_player.play_welcome_back(name)
                elif sound_type == 'guest_mode_activated':
                    host_name = kwargs.get('host_name', '')
                    self.sound_player.play_guest_mode_activated(host_name)
                elif sound_type == 'guest_mode_expired':
                    self.sound_player.play_guest_mode_expired()
                elif sound_type == 'verification_timeout':
                    self.sound_player.play_verification_timeout()
            except Exception as e:
                logger.error(f"Error playing background sound {sound_type}: {e}")
        
        # Start sound in background thread
        sound_thread = threading.Thread(target=play_sound, daemon=True)
        sound_thread.start()

    def play_priority_sound(self, sound_type: str, **kwargs):
        """Play priority sound that interrupts current speech"""
        if not self.sound_player:
            return
        
        def play_priority_sound():
            try:
                if sound_type == 'unknown_person_alert':
                    self.sound_player.play_unknown_person_alert()
                elif sound_type == 'security_breach':
                    self.sound_player.play_security_breach()
                elif sound_type == 'verification_timeout':
                    self.sound_player.play_verification_timeout()
                elif sound_type == 'time_based_greeting':
                    self.sound_player.play_time_based_greeting()
                elif sound_type == 'known_person_greeting':
                    name = kwargs.get('name', '')
                    self.sound_player.play_known_person_greeting(name)
            except Exception as e:
                logger.error(f"Error playing priority sound {sound_type}: {e}")
        
        # Start priority sound in background thread
        sound_thread = threading.Thread(target=play_priority_sound, daemon=True)
        sound_thread.start()

    def cleanup(self):
        """Cleanup system resources"""
        logger.info("🧹 Cleaning up CCTV system...")

        self.running = False

        # Stop any active recordings
        if hasattr(self.person_tracker, 'recording_active'):
            for track_id, writer in self.person_tracker.recording_active.items():
                try:
                    writer.release()
                    logger.info(f"📹 Stopped recording for track {track_id}")
                except Exception as e:
                    logger.error(f"Error stopping recording: {e}")

        # Cleanup camera
        if self.camera:
            if hasattr(self.camera, 'stop'):
                # Picamera2
                self.camera.stop()
                self.camera.close()
            else:
                # OpenCV
                self.camera.release()

        # Cleanup hardware
        if self.hardware_manager:
            self.hardware_manager.cleanup()

        cv2.destroyAllWindows()
        logger.info("✅ CCTV system cleanup complete")


def main():
    """Main function to run the CCTV system"""
    logger.info("🚀 Starting Raspberry Pi CCTV Camera System")
    logger.info("=" * 60)
    logger.info("📷 Advanced Person Tracking with Face Recognition")
    logger.info("🔍 Visual Security Alerts and Time-based Greetings")
    logger.info("💡 Motion Detection with LED Control")
    logger.info("📹 Recording of Unknown Persons")
    logger.info("=" * 60)

    try:
        cctv_system = CCTVSystem()
        cctv_system.run()

    except Exception as e:
        logger.error(f"❌ Fatal error in CCTV system: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
