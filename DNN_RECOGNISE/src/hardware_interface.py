# Raspberry Pi Hardware Interface for CCTV System
import RPi.GPIO as GPIO
import threading
import time
import logging
import subprocess
import os
from typing import Optional, Callable
from settings.settings import HARDWARE, CCTV

logger = logging.getLogger(__name__)

class PIRMotionDetector:
    """PIR Motion Sensor Interface"""

    def __init__(self, pin: int, callback: Optional[Callable] = None):
        self.pin = pin
        self.callback = callback
        self.last_motion_time = 0
        self.cooldown = CCTV['motion_cooldown']
        self.enabled = CCTV['motion_detection_enabled']

        if self.enabled:
            self._setup_gpio()

    def _setup_gpio(self):
        """Setup GPIO pin for PIR sensor"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            GPIO.add_event_detect(self.pin, GPIO.RISING, callback=self._motion_detected, bouncetime=1000)
            logger.info(f"✅ PIR motion sensor initialized on pin {self.pin}")
        except Exception as e:
            logger.error(f"Failed to setup PIR sensor: {e}")
            self.enabled = False

    def _motion_detected(self, channel):
        """Handle motion detection event"""
        current_time = time.time()

        if current_time - self.last_motion_time > self.cooldown:
            self.last_motion_time = current_time
            logger.info(f"🚶 Motion detected on pin {self.pin}")

            if self.callback:
                self.callback()

    def cleanup(self):
        """Cleanup GPIO resources"""
        if self.enabled:
            try:
                GPIO.remove_event_detect(self.pin)
                logger.info("🧹 PIR motion detector cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up PIR: {e}")


class LEDController:
    """LED Control Interface for status indicators and brightness"""

    def __init__(self):
        self.brightness_pin = HARDWARE['led_brightness_pin']
        self.green_pin = HARDWARE['led_green_pin']
        self.yellow_pin = HARDWARE['led_yellow_pin']
        self.red_pin = HARDWARE['led_red_pin']

        self.brightness_timer = None
        self.brightness_on = False
        self.brightness_enabled = CCTV['led_auto_brightness']

        # Guest mode variables
        self.guest_mode_active = False
        self.guest_mode_timer = None
        self.guest_mode_pulse_interval = CCTV['guest_mode_yellow_pulse_interval']

        self._setup_gpio()
        self._init_led_states()

    def _setup_gpio(self):
        """Setup GPIO pins for LEDs"""
        try:
            GPIO.setmode(GPIO.BCM)

            # Setup output pins
            pins = [self.brightness_pin, self.green_pin, self.yellow_pin, self.red_pin]
            for pin in pins:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)

            logger.info("✅ LED controller initialized")
        except Exception as e:
            logger.error(f"Failed to setup LED controller: {e}")
            self.brightness_enabled = False

    def _init_led_states(self):
        """Initialize LED states"""
        self.set_status('off')  # All LEDs off initially
        if self.brightness_enabled:
            GPIO.output(self.brightness_pin, GPIO.LOW)

    def set_status(self, status: str):
        """Set status LEDs based on system status"""
        # Turn off all status LEDs first
        self._set_green(False)
        self._set_yellow(False)
        self._set_red(False)

        if status == 'ready':
            # System working
            self._set_green(True)
        elif status == 'verifying':
            # Person detected / verification in progress
            self._set_red(True)
        elif status == 'guest':
            # Guest mode
            self._set_yellow(True)
        elif status == 'alert':
            # Alert state
            self._set_red(True)
        elif status == 'recording':
            # Flash red LED for recording
            self._flash_red()
        elif status == 'off':
            pass  # All off

    def _set_green(self, state: bool):
        """Control green LED"""
        try:
            GPIO.output(self.green_pin, GPIO.HIGH if state else GPIO.LOW)
        except Exception as e:
            logger.error(f"Error controlling green LED: {e}")

    def _set_yellow(self, state: bool):
        """Control yellow LED"""
        try:
            GPIO.output(self.yellow_pin, GPIO.HIGH if state else GPIO.LOW)
        except Exception as e:
            logger.error(f"Error controlling yellow LED: {e}")

    def _set_red(self, state: bool):
        """Control red LED"""
        try:
            GPIO.output(self.red_pin, GPIO.HIGH if state else GPIO.LOW)
        except Exception as e:
            logger.error(f"Error controlling red LED: {e}")

    def _flash_red(self):
        """Flash red LED for recording indication"""
        def flash():
            for _ in range(3):  # Flash 3 times
                self._set_red(True)
                time.sleep(0.5)
                self._set_red(False)
                time.sleep(0.5)

        thread = threading.Thread(target=flash, daemon=True)
        thread.start()

    def turn_on_brightness(self, duration: Optional[float] = None):
        """Turn on high brightness LED"""
        if not self.brightness_enabled:
            return

        try:
            GPIO.output(self.brightness_pin, GPIO.HIGH)
            logger.info("💡 High brightness LED turned on")
            self.brightness_on = True

            if duration:
                if self.brightness_timer:
                    self.brightness_timer.cancel()

                self.brightness_timer = threading.Timer(duration, self.turn_off_brightness)
                self.brightness_timer.start()
                logger.info(f"⏰ Brightness LED will turn off in {duration} seconds")

        except Exception as e:
            logger.error(f"Error controlling brightness LED: {e}")

    def turn_off_brightness(self):
        """Turn off high brightness LED"""
        if not self.brightness_enabled:
            return

        try:
            GPIO.output(self.brightness_pin, GPIO.LOW)
            logger.info("💡 High brightness LED turned off")
            self.brightness_on = False

            if self.brightness_timer:
                self.brightness_timer.cancel()
                self.brightness_timer = None

        except Exception as e:
            logger.error(f"Error controlling brightness LED: {e}")

    def start_guest_mode_pulse(self):
        """Guest mode indicator: set steady yellow HIGH"""
        if self.guest_mode_active:
            return  # Already active
        self.guest_mode_active = True
        logger.info("💛 Guest mode active - yellow LED ON")
        try:
            self._set_yellow(True)
        except Exception as e:
            logger.error(f"Error setting guest mode LED: {e}")

    def stop_guest_mode_pulse(self):
        """Stop guest mode indicator and turn yellow OFF"""
        if not self.guest_mode_active:
            return

        self.guest_mode_active = False
        logger.info("💛 Guest mode ended - yellow LED OFF")

        if self.guest_mode_timer:
            try:
                self.guest_mode_timer.join(timeout=2.0)
            except Exception:
                pass
            self.guest_mode_timer = None

        # Turn off yellow LED
        self._set_yellow(False)

    def cleanup(self):
        """Cleanup GPIO resources"""
        try:
            # Stop guest mode pulse
            self.stop_guest_mode_pulse()

            # Turn off all LEDs
            self.set_status('off')
            self.turn_off_brightness()

            logger.info("🧹 LED controller cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up LED controller: {e}")


class ButtonEmulator:
    """Emulate a momentary button by pulling a GPIO LOW for a duration."""

    def __init__(self, pin: int):
        self.pin = pin
        self._setup_gpio()

    def _setup_gpio(self):
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.HIGH)
            logger.info(f"✅ Button emulator ready on pin {self.pin}")
        except Exception as e:
            logger.error(f"Failed to setup button emulator on pin {self.pin}: {e}")

    def press(self, press_ms: int):
        try:
            GPIO.output(self.pin, GPIO.LOW)
            time.sleep(press_ms / 1000.0)
            GPIO.output(self.pin, GPIO.HIGH)
            time.sleep(0.05)  # debounce gap
        except Exception as e:
            logger.error(f"Button press failed on pin {self.pin}: {e}")

    def double_click(self, press_ms: int, gap_ms: int):
        self.press(press_ms)
        time.sleep(gap_ms / 1000.0)
        self.press(press_ms)

    def hold(self, hold_ms: int):
        self.press(hold_ms)


class SpeakerController:
    """Control Bluetooth speaker power using its button line (hold LOW to power on)."""

    def __init__(self, pin: int):
        self.button = ButtonEmulator(pin)
        self.last_power_on = 0.0

    def power_on(self):
        try:
            now = time.time()
            # Avoid repeated holds; only attempt if >5s since last
            if now - self.last_power_on < 5.0:
                return
            logger.info("🔊 Powering on Bluetooth speaker (hold 2s)")
            self.button.hold(HARDWARE['speaker_power_hold_ms'])
            self.last_power_on = now
        except Exception as e:
            logger.error(f"Failed to power on speaker: {e}")


class ExternalLightsController:
    """Deprecated click-based controller retained for compatibility (no-op)."""
    def __init__(self, pin: int):
        self.mode = 'off'

    def motion_high_for(self, seconds: int):
        # Cancel previous timer
        if self._motion_timer and self._motion_timer.is_alive():
            self._motion_timer.cancel()
        self.set_high()
        def _turn_off():
            try:
                self.set_off()
                logger.info("💡 Motion window ended - lights OFF")
            except Exception as e:
                logger.error(f"Failed to turn off lights after motion: {e}")
        self._motion_timer = threading.Timer(seconds, _turn_off)
        self._motion_timer.daemon = True
        self._motion_timer.start()



class HardwareManager:
    """Main hardware interface manager"""

    def __init__(self):
        self.pir_detector = None
        self.led_controller = None
        self.speaker_controller = None
        self.lights_controller = None
        self._motion_suppress_until = 0.0

        self._init_hardware()

    def _init_hardware(self):
        """Initialize all hardware components"""
        try:
            # Initialize PIR motion detector
            if CCTV['motion_detection_enabled']:
                self.pir_detector = PIRMotionDetector(HARDWARE['pir_pin'], self._on_motion_detected)
            else:
                logger.info("📍 Motion detection disabled")

            # Initialize LED controller
            self.led_controller = LEDController()

            # Initialize external devices
            self.speaker_controller = SpeakerController(HARDWARE['speaker_button_pin'])
            # Remove click-based lights; use LEDController.brightness_pin MOSFET directly

            # Startup behavior: power on speaker, ensure MOSFET LED is OFF
            self.speaker_controller.power_on()
            if self.led_controller:
                self.led_controller.turn_off_brightness()

            logger.info("✅ Hardware manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize hardware manager: {e}")
            raise

    def _on_motion_detected(self):
        """Handle motion detection event"""
        current_time = time.time()
        # Suppress repeated motion actions while lights are already on for the motion window
        if current_time < getattr(self, '_motion_suppress_until', 0.0):
            return

        if self.led_controller:
            self.led_controller.set_status('verifying')

        # Drive MOSFET LED ON for motion window
        if self.led_controller:
            self.led_controller.turn_on_brightness(HARDWARE['motion_light_duration_s'])

        # Suppress further motion triggers until this window ends
        self._motion_suppress_until = current_time + HARDWARE['motion_light_duration_s']

    def set_system_status(self, status: str):
        """Set overall system status"""
        if self.led_controller:
            self.led_controller.set_status(status)

    # Convenience methods for MOSFET LED
    def lights_off(self):
        if self.led_controller:
            self.led_controller.turn_off_brightness()

    def lights_high_for_motion(self):
        if self.led_controller:
            self.led_controller.turn_on_brightness(HARDWARE['motion_light_duration_s'])

    def lights_on(self):
        if self.led_controller:
            self.led_controller.turn_on_brightness()

    def cleanup(self):
        """Cleanup all hardware resources"""
        logger.info("🧹 Cleaning up hardware interfaces...")

        if self.pir_detector:
            self.pir_detector.cleanup()

        if self.led_controller:
            self.led_controller.cleanup()

        try:
            GPIO.cleanup()
            logger.info("✅ GPIO cleanup completed")
        except Exception as e:
            logger.error(f"Error during GPIO cleanup: {e}")

    # Convenience methods
    def motion_detected(self):
        """Check if motion was recently detected"""
        if self.pir_detector:
            return time.time() - self.pir_detector.last_motion_time < 1.0
        return False

    def activate_guest_mode(self, host_name: str):
        """Activate guest mode with yellow pulsing LED"""
        if self.led_controller:
            self.led_controller.start_guest_mode_pulse()

    def revert_guest_mode(self):
        """Revert from guest mode back to normal security"""
        if self.led_controller:
            self.led_controller.stop_guest_mode_pulse()

    


# Global hardware manager instance
hardware_manager = None

def get_hardware_manager() -> HardwareManager:
    """Get or create hardware manager instance"""
    global hardware_manager
    if hardware_manager is None:
        hardware_manager = HardwareManager()
    return hardware_manager
