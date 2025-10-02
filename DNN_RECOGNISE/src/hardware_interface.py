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
    """LED control: green/yellow/red status and brightness MOSFET"""

    def __init__(self):
        self.brightness_pin = HARDWARE['led_brightness_pin']
        self.green_pin = HARDWARE['led_green_pin']
        self.yellow_pin = HARDWARE['led_yellow_pin']
        self.red_pin = HARDWARE['led_red_pin']

        self.brightness_timer = None
        self.brightness_on = False
        self.brightness_enabled = CCTV['led_auto_brightness']

        # Blink control
        self._green_blink = False
        self._yellow_blink = False
        self._green_thread = None
        self._yellow_thread = None

        self._setup_gpio()
        self._init_led_states()

    def _setup_gpio(self):
        """Setup GPIO pins for brightness and status LEDs"""
        try:
            GPIO.setmode(GPIO.BCM)

            # Setup output pins
            for pin in [self.brightness_pin, self.green_pin, self.yellow_pin, self.red_pin]:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)

            logger.info("✅ LED controller initialized")
        except Exception as e:
            logger.error(f"Failed to setup LED controller: {e}")
            self.brightness_enabled = False

    def _init_led_states(self):
        """Initialize LED states: system on -> green blink"""
        try:
            # All off initially
            self._set_green(False)
            self._set_yellow(False)
            self._set_red(False)
        except Exception:
            pass
        if self.brightness_enabled:
            GPIO.output(self.brightness_pin, GPIO.LOW)
        # System on indication
        self._start_green_blink()

    def set_status(self, status: str):
        """Set status LEDs based on system status"""
        # Stop blinks by default
        self._stop_yellow_blink()
        # Clear solid states
        self._set_green(False)
        self._set_yellow(False)
        self._set_red(False)

        if status == 'ready':
            # System running -> green blink
            self._start_green_blink()
        elif status == 'verifying':
            # Person detection -> yellow blink
            # self._stop_green_blink()
            self._start_yellow_blink()
        elif status == 'guest':
            # Guest mode -> yellow solid HIGH
            # self._stop_green_blink()
            self._set_yellow(True)
        elif status == 'alert':
            # Unknown person detected -> red solid HIGH
            # self._stop_green_blink()
            self._set_red(True)
        elif status == 'off':
            # System off -> all LEDs off
            self._stop_green_blink()
            self._stop_yellow_blink()
            self._set_green(False)
            self._set_yellow(False)
            self._set_red(False)
        elif status == 'recording':
            # Recording unknown person -> red solid HIGH
            # self._stop_green_blink()
            self._set_red(True)

    def _set_green(self, state: bool):
        try:
            GPIO.output(self.green_pin, GPIO.HIGH if state else GPIO.LOW)
        except Exception as e:
            logger.error(f"Error controlling green LED: {e}")

    def _set_yellow(self, state: bool):
        try:
            GPIO.output(self.yellow_pin, GPIO.HIGH if state else GPIO.LOW)
        except Exception as e:
            logger.error(f"Error controlling yellow LED: {e}")

    def _set_red(self, state: bool):
        try:
            GPIO.output(self.red_pin, GPIO.HIGH if state else GPIO.LOW)
        except Exception as e:
            logger.error(f"Error controlling red LED: {e}")

    def _start_green_blink(self):
        if self._green_blink:
            return
        self._green_blink = True
        def _blink():
            while self._green_blink:
                self._set_green(True)
                time.sleep(0.5)
                self._set_green(False)
                time.sleep(0.5)
        self._green_thread = threading.Thread(target=_blink, daemon=True)
        self._green_thread.start()

    def _stop_green_blink(self):
        if not self._green_blink:
            return
        self._green_blink = False
        try:
            if self._green_thread:
                self._green_thread.join(timeout=0.1)
        except Exception:
            pass
        self._green_thread = None

    def _start_yellow_blink(self):
        if self._yellow_blink:
            return
        self._yellow_blink = True
        def _blink():
            while self._yellow_blink:
                self._set_yellow(True)
                time.sleep(0.3)
                self._set_yellow(False)
                time.sleep(0.3)
        self._yellow_thread = threading.Thread(target=_blink, daemon=True)
        self._yellow_thread.start()

    def _stop_yellow_blink(self):
        if not self._yellow_blink:
            return
        self._yellow_blink = False
        try:
            if self._yellow_thread:
                self._yellow_thread.join(timeout=0.1)
        except Exception:
            pass
        self._yellow_thread = None

    # Recording flash removed; use red solid during alerts/recording

    def turn_on_brightness(self, duration: Optional[float] = None):
        """Turn on high brightness LED"""
        if not self.brightness_enabled:
            return

        try:
            GPIO.output(self.brightness_pin, GPIO.LOW)
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
            GPIO.output(self.brightness_pin, GPIO.HIGH)
            logger.info("💡 High brightness LED turned off")
            self.brightness_on = False

            if self.brightness_timer:
                self.brightness_timer.cancel()
                self.brightness_timer = None

        except Exception as e:
            logger.error(f"Error controlling brightness LED: {e}")

    def start_guest_mode_pulse(self):
        """Guest mode -> yellow solid ON"""
        self._stop_yellow_blink()
        self._stop_green_blink()
        self._set_red(False)
        self._set_yellow(True)

    def stop_guest_mode_pulse(self):
        """Exit guest mode -> yellow LOW, back to ready (green blink)"""
        self._set_yellow(False)
        self._start_green_blink()

    def person_detected(self):
        """Handle person detection -> yellow blink"""
        self._stop_green_blink()
        self._start_yellow_blink()
        logger.info("🟡 Person detected - yellow LED blinking")

    def unknown_person_detected(self):
        """Handle unknown person detection -> red solid"""
        self._stop_green_blink()
        self._stop_yellow_blink()
        self._set_red(True)
        logger.info("🔴 Unknown person detected - red LED solid")

    def known_person_detected(self):
        """Handle known person detection -> back to ready (green blink)"""
        self._stop_yellow_blink()
        self._set_red(False)
        self._start_green_blink()
        logger.info("🟢 Known person detected - back to green blink")

    def cleanup(self):
        """Cleanup GPIO resources"""
        try:
            # Stop blinks and turn off LEDs
            self._stop_green_blink()
            self._stop_yellow_blink()
            try:
                self._set_green(False)
                self._set_yellow(False)
                self._set_red(False)
            except Exception:
                pass
            # Turn off brightness
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
        # LDR state
        self._ldr_enabled = HARDWARE.get('ldr_enabled', False)
        self._ldr_pin = HARDWARE.get('ldr_pin', 23)
        self._ldr_active_high_means_bright = HARDWARE.get('ldr_active_high_means_bright', True)
        self._ldr_motion_enable_in_dark_only = HARDWARE.get('ldr_motion_enable_in_dark_only', True)
        self._ldr_last_change_ts = 0.0
        self._ldr_bright = None

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

            # Initialize LDR GPIO after other setups
            if self._ldr_enabled:
                try:
                    GPIO.setmode(GPIO.BCM)
                    # Digital LDR output: pull-down to avoid floating
                    GPIO.setup(self._ldr_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
                    # Read initial state
                    raw = GPIO.input(self._ldr_pin)
                    self._ldr_bright = bool(raw) if self._ldr_active_high_means_bright else not bool(raw)
                    self._ldr_last_change_ts = time.time()
                    # Edge detect to track changes
                    GPIO.add_event_detect(self._ldr_pin, GPIO.BOTH, callback=self._on_ldr_changed, bouncetime=100)
                    logger.info(f"✅ LDR initialized on pin {self._ldr_pin} (bright={self._ldr_bright})")
                except Exception as e:
                    self._ldr_enabled = False
                    logger.error(f"Failed to setup LDR on pin {self._ldr_pin}: {e}")

            logger.info("✅ Hardware manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize hardware manager: {e}")
            raise

    def _on_motion_detected(self):
        """Handle motion detection event"""
        current_time = time.time()
        # If LDR gating is enabled, decide whether to allow PIR
        if self._ldr_enabled:
            # Suppress immediately after a light-level change
            if current_time - self._ldr_last_change_ts < HARDWARE.get('ldr_change_suppress_s', 1.0):
                logger.info("⏱️ PIR suppressed due to recent light change")
                return
            if self._ldr_motion_enable_in_dark_only:
                # Allow PIR only when dark
                if self._ldr_bright is True:
                    logger.info("🌞 PIR suppressed because it's bright (LDR)")
                    return
        # Suppress repeated motion actions while lights are already on for the motion window
        if current_time < getattr(self, '_motion_suppress_until', 0.0):
            return
        # Indicate person detection (yellow blink)
        if self.led_controller:
            self.led_controller.set_status('verifying')
        # Drive MOSFET LED ON for motion window
        if self.led_controller:
            self.led_controller.turn_on_brightness(HARDWARE['motion_light_duration_s'])

        # Suppress further motion triggers until this window ends
        self._motion_suppress_until = current_time + HARDWARE['motion_light_duration_s']

    def _on_ldr_changed(self, channel):
        """Handle LDR state change (brightness change)."""
        try:
            raw = GPIO.input(self._ldr_pin)
            bright = bool(raw) if self._ldr_active_high_means_bright else not bool(raw)
            if bright != self._ldr_bright:
                self._ldr_bright = bright
                self._ldr_last_change_ts = time.time()
                logger.info(f"🔆 LDR changed: bright={self._ldr_bright}")
        except Exception as e:
            logger.error(f"Error reading LDR state: {e}")

    def set_system_status(self, status: str):
        """Set overall system status LEDs"""
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
        """Activate guest mode -> yellow solid ON"""
        if self.led_controller:
            self.led_controller.start_guest_mode_pulse()

    def revert_guest_mode(self):
        """Revert guest mode -> back to ready (green blink)"""
        if self.led_controller:
            self.led_controller.stop_guest_mode_pulse()

    def person_detected(self):
        """Handle person detection -> yellow blink"""
        if self.led_controller:
            self.led_controller.person_detected()

    def unknown_person_detected(self):
        """Handle unknown person detection -> red solid"""
        if self.led_controller:
            self.led_controller.unknown_person_detected()

    def known_person_detected(self):
        """Handle known person detection -> back to ready (green blink)"""
        if self.led_controller:
            self.led_controller.known_person_detected()

    


# Global hardware manager instance
hardware_manager = None

def get_hardware_manager() -> HardwareManager:
    """Get or create hardware manager instance"""
    global hardware_manager
    if hardware_manager is None:
        hardware_manager = HardwareManager()
    return hardware_manager
