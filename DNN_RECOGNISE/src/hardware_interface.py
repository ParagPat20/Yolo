# Raspberry Pi Hardware Interface for CCTV System
import RPi.GPIO as GPIO
import threading
import time
import logging
import subprocess
import os
from datetime import datetime
from typing import Optional, Callable
from settings.settings import HARDWARE, CCTV, AUDIO, VOICE

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
            self._set_green(True)
        elif status == 'verifying':
            self._set_yellow(True)
        elif status == 'alert':
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

            if self.brightness_timer:
                self.brightness_timer.cancel()
                self.brightness_timer = None

        except Exception as e:
            logger.error(f"Error controlling brightness LED: {e}")

    def start_guest_mode_pulse(self):
        """Start pulsing yellow LED for guest mode"""
        if self.guest_mode_active:
            return  # Already active

        self.guest_mode_active = True
        logger.info("💛 Starting guest mode yellow LED pulse")

        def pulse():
            while self.guest_mode_active:
                self._set_yellow(True)
                time.sleep(self.guest_mode_pulse_interval)
                self._set_yellow(False)
                time.sleep(self.guest_mode_pulse_interval)

        self.guest_mode_timer = threading.Thread(target=pulse, daemon=True)
        self.guest_mode_timer.start()

    def stop_guest_mode_pulse(self):
        """Stop pulsing yellow LED for guest mode"""
        if not self.guest_mode_active:
            return

        self.guest_mode_active = False
        logger.info("💛 Stopping guest mode yellow LED pulse")

        if self.guest_mode_timer:
            self.guest_mode_timer.join(timeout=2.0)  # Wait for thread to finish
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


class BluetoothSpeaker:
    """Bluetooth Speaker Interface"""

    def __init__(self):
        self.mac_address = HARDWARE['bt_speaker_mac']
        self.name = HARDWARE['bt_speaker_name']
        self.connected = False
        self.skip_connect = HARDWARE.get('bt_skip_connect', False)
        self.use_system_audio = HARDWARE.get('bt_use_system_audio', False)

        # If configured to skip explicit bluetooth connection, assume audio path is ready
        if self.skip_connect:
            logger.info("🔊 Skipping Bluetooth connect step, using system audio output")
            self.connected = True
        else:
            self._ensure_connected()

    def _ensure_connected(self):
        """Ensure Bluetooth speaker is connected"""
        try:
            # Check if already connected
            result = subprocess.run(['bluetoothctl', 'info', self.mac_address],
                                  capture_output=True, text=True, timeout=10)

            if 'Connected: yes' in result.stdout:
                self.connected = True
                logger.info("✅ Bluetooth speaker already connected")
                return

            # Try to connect
            logger.info(f"🔗 Connecting to Bluetooth speaker {self.name}...")
            subprocess.run(['bluetoothctl', 'connect', self.mac_address],
                         capture_output=True, timeout=10)

            # Verify connection
            result = subprocess.run(['bluetoothctl', 'info', self.mac_address],
                                  capture_output=True, text=True, timeout=5)

            if 'Connected: yes' in result.stdout:
                self.connected = True
                logger.info("✅ Bluetooth speaker connected successfully")
            else:
                logger.warning("⚠️ Failed to connect to Bluetooth speaker")
                self.connected = False

        except Exception as e:
            logger.error(f"Error connecting to Bluetooth speaker: {e}")
            self.connected = False

    def speak(self, text: str):
        """Speak text using Bluetooth speaker with configurable female voice (non-blocking)"""
        if not self.connected:
            logger.warning("Bluetooth speaker not connected, cannot speak")
            return

        def _speak_worker(message: str):
            try:
                # Get voice configuration
                engine = VOICE.get('engine', 'espeak-ng')
                gender = VOICE.get('gender', 'female')
                speech_rate = VOICE.get('speech_rate', 150)
                pitch = VOICE.get('pitch', 50)
                volume = VOICE.get('volume', 80)

                if engine == 'espeak-ng':
                    try:
                        if gender == 'female':
                            espeak_voice = VOICE.get('espeak_voice', 'en+f3')
                            cmd = ['espeak-ng', '-v', espeak_voice, '-s', str(speech_rate), '-p', str(pitch), '-a', str(volume), message]
                        else:
                            cmd = ['espeak-ng', '-v', 'en', '-s', str(speech_rate), '-p', str(pitch), '-a', str(volume), message]
                        subprocess.run(cmd, capture_output=True, timeout=15)
                        logger.info("🗣️♀️ Spoken (espeak-ng)")
                        return
                    except Exception as e:
                        logger.debug(f"espeak-ng {gender} voice failed: {e}")

                if engine == 'festival':
                    try:
                        if gender == 'female':
                            festival_voice = VOICE.get('festival_voice', 'cmu_us_slt_cg')
                            cmd = ['bash', '-lc', f"echo '(voice_{festival_voice}) (SayText \"{message}\")' | festival"]
                        else:
                            cmd = ['bash', '-lc', f"echo '(voice_cmu_us_rms_cg) (SayText \"{message}\")' | festival"]
                        subprocess.run(cmd, capture_output=True, timeout=15)
                        logger.info("🗣️♀️ Spoken (festival)")
                        return
                    except Exception as e:
                        logger.debug(f"festival {gender} voice failed: {e}")

                # Final fallback
                try:
                    subprocess.run(['espeak-ng', '-v', 'en+f3', '-s', '140', message], capture_output=True, timeout=10)
                    logger.info("🗣️♀️ Spoken (fallback)")
                except Exception as e:
                    logger.error(f"All TTS methods failed: {e}")
            except Exception as e:
                logger.error(f"Error in speak worker: {e}")

        # Start TTS in background thread so main loop is not blocked
        threading.Thread(target=_speak_worker, args=(text,), daemon=True).start()

        try:
            # Get voice configuration
            engine = VOICE.get('engine', 'espeak-ng')
            gender = VOICE.get('gender', 'female')
            speech_rate = VOICE.get('speech_rate', 150)
            pitch = VOICE.get('pitch', 50)
            volume = VOICE.get('volume', 80)

            # Try TTS engines based on configuration
            if engine == 'espeak-ng':
                try:
                    if gender == 'female':
                        # Use configured female voice for espeak-ng
                        espeak_voice = VOICE.get('espeak_voice', 'en+f3')
                        cmd = ['espeak-ng', '-v', espeak_voice, '-s', str(speech_rate), '-p', str(pitch), '-a', str(volume), text]
                    else:
                        # Use male voice
                        cmd = ['espeak-ng', '-v', 'en', '-s', str(speech_rate), '-p', str(pitch), '-a', str(volume), text]

                    result = subprocess.run(cmd, capture_output=True, timeout=15)
                    if result.returncode == 0:
                        voice_type = "female" if gender == "female" else "male"
                        logger.info(f"🗣️♀️ Spoken (espeak-ng {voice_type}): {text}")
                        return
                except Exception as e:
                    logger.debug(f"espeak-ng {gender} voice failed: {e}")

            # Try festival as fallback
            if engine == 'festival':
                try:
                    if gender == 'female':
                        # Use configured female voice for festival
                        festival_voice = VOICE.get('festival_voice', 'cmu_us_slt_cg')
                        cmd = ['echo', f'(voice_{festival_voice})', f'(SayText "{text}")', '|', 'festival']
                    else:
                        # Use male voice for festival
                        cmd = ['echo', f'(voice_cmu_us_rms_cg)', f'(SayText "{text}")', '|', 'festival']

                    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=15)
                    if result.returncode == 0:
                        voice_type = "female" if gender == "female" else "male"
                        logger.info(f"🗣️♀️ Spoken (festival {voice_type}): {text}")
                        return
                except Exception as e:
                    logger.debug(f"festival {gender} voice failed: {e}")

            # Final fallback - try basic espeak-ng
            try:
                cmd = ['espeak-ng', '-v', 'en+f3', '-s', '140', text]  # Default to female voice
                subprocess.run(cmd, capture_output=True, timeout=10)
                logger.info(f"🗣️♀️ Spoken (espeak-ng fallback): {text}")
            except Exception as e:
                logger.error(f"All TTS methods failed: {e}")

        except Exception as e:
            logger.error(f"Error speaking text: {e}")

    def play_sound_file(self, filepath: str):
        """Play sound file through Bluetooth speaker"""
        if not self.connected or not os.path.exists(filepath):
            logger.warning(f"Cannot play sound file: connected={self.connected}, file exists={os.path.exists(filepath)}")
            return

        try:
            # Use aplay or mpg123 to play the sound file
            # Adjust this based on your sound file format
            cmd = ['aplay', filepath] if filepath.endswith('.wav') else ['mpg123', filepath]
            subprocess.run(cmd, capture_output=True, timeout=30)
            logger.info(f"🔊 Played sound file: {filepath}")

        except Exception as e:
            logger.error(f"Error playing sound file: {e}")

    def play_alarm(self):
        """Play alarm sound"""
        if AUDIO['alert_sound_enabled']:
            alarm_file = AUDIO['alarm_sound_file']
            if os.path.exists(alarm_file):
                self.play_sound_file(alarm_file)
            else:
                # Fallback to speaking the alert
                self.speak(AUDIO['unknown_alert'])

    def greet_person(self, name: str):
        """Greet a person with appropriate time-based message"""
        if not AUDIO['greeting_sound_enabled']:
            return

        current_hour = datetime.now().hour

        if 5 <= current_hour < 12:
            message = f"{AUDIO['greeting_morning']}, {name}!"
        elif 12 <= current_hour < 17:
            message = f"{AUDIO['greeting_afternoon']}, {name}!"
        else:
            message = f"{AUDIO['greeting_evening']}, {name}!"

        self.speak(message)

    def welcome_back(self, name: str):
        """Welcome back a recognized person"""
        if AUDIO['greeting_sound_enabled']:
            self.speak(f"{AUDIO['welcome_back']}, {name}!")

    def request_verification(self):
        """Request face verification"""
        if AUDIO['greeting_sound_enabled']:
            self.speak(AUDIO['verification_request'])


class HardwareManager:
    """Main hardware interface manager"""

    def __init__(self):
        self.pir_detector = None
        self.led_controller = None
        self.bt_speaker = None

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

            # Initialize Bluetooth speaker
            self.bt_speaker = BluetoothSpeaker()

            logger.info("✅ Hardware manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize hardware manager: {e}")
            raise

    def _on_motion_detected(self):
        """Handle motion detection event"""
        if self.led_controller:
            self.led_controller.turn_on_brightness(CCTV['led_brightness_duration'])

        if self.led_controller:
            self.led_controller.set_status('verifying')

    def set_system_status(self, status: str):
        """Set overall system status"""
        if self.led_controller:
            self.led_controller.set_status(status)

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

    def play_alarm(self):
        """Play alarm sound"""
        if self.bt_speaker:
            self.bt_speaker.play_alarm()

    def greet_person(self, name: str):
        """Greet a person"""
        if self.bt_speaker:
            self.bt_speaker.greet_person(name)

    def welcome_back(self, name: str):
        """Welcome back a recognized person"""
        if self.bt_speaker:
            self.bt_speaker.welcome_back(name)

    def request_verification(self):
        """Request face verification"""
        if self.bt_speaker:
            self.bt_speaker.request_verification()

    def activate_guest_mode(self, host_name: str):
        """Activate guest mode with yellow pulsing LED and announcement"""
        if self.led_controller:
            self.led_controller.start_guest_mode_pulse()
        if self.bt_speaker:
            message = f"Welcome back, {host_name}. I've noticed you have a guest. System is now in guest mode for the next 15 minutes."
            self.bt_speaker.speak(message)

    def revert_guest_mode(self):
        """Revert from guest mode back to normal security"""
        if self.led_controller:
            self.led_controller.stop_guest_mode_pulse()
        if self.bt_speaker:
            self.bt_speaker.speak("Guest mode expired. Reverting to normal security protocols.")


# Global hardware manager instance
hardware_manager = None

def get_hardware_manager() -> HardwareManager:
    """Get or create hardware manager instance"""
    global hardware_manager
    if hardware_manager is None:
        hardware_manager = HardwareManager()
    return hardware_manager
