#!/usr/bin/env python3
"""
Sound System for CCTV Security System
Pre-generates WAV files using Piper TTS during initialization
Plays audio files quickly using mpg123 or aplay
"""

import subprocess
import logging
import threading
import time
import os
import platform
import shlex
from typing import Optional, Dict
from datetime import datetime

# Import settings
try:
    from settings.settings import SOUND_SYSTEM, AUDIO
except ImportError:
    # Fallback settings if import fails
    SOUND_SYSTEM = {
        'enabled': True,
        'language': 'en',
        'wav_files_dir': '/home/jecon/yolo/DNN_RECOGNISE/sounds/wav',
        'piper': {
            'model_path': '/usr/local/share/piper-voices/en_US-ljspeech-medium.onnx',
            'models': {
                'en': '/usr/local/share/piper-voices/en_US-ljspeech-medium.onnx',
                'gu': '/usr/local/share/piper-voices/gu_IN-cmu-indic_medium.onnx',
            },
            'noise': 0.667,
            'length_penalty': 1.0,
        }
    }
    AUDIO = {
        'wav_files_dir': '/home/jecon/yolo/DNN_RECOGNISE/sounds/wav',
        'alarm_sound_path': '/home/jecon/yolo/DNN_RECOGNISE/sounds/alarm.mp3',
        'verification_beep_path': '/home/jecon/yolo/DNN_RECOGNISE/sounds/verification_beep.mp3',
    }

logger = logging.getLogger(__name__)

class SoundSystem:
    """Sound system using pre-generated WAV files with Piper TTS"""
    
    def __init__(self, language: str = None):
        """Initialize sound system and generate WAV files"""
        self.is_enabled = False
        self.language = language or SOUND_SYSTEM.get('language', 'en')
        self.wav_files_dir = SOUND_SYSTEM.get('wav_files_dir', '/home/jecon/yolo/DNN_RECOGNISE/sounds/wav')
        self.alarm_active = False
        self.current_process: Optional[subprocess.Popen] = None
        
        # Check if sound system is enabled
        if not SOUND_SYSTEM.get('enabled', True):
            logger.info("🔇 Sound system disabled in settings")
            return
        
        # Create WAV files directory
        os.makedirs(self.wav_files_dir, exist_ok=True)
        
        # Initialize TTS system
        self._initialize_tts_system()
        
        if self.is_enabled:
            lang_name = "Gujarati" if self.language == 'gu' else "English"
            logger.info(f"🔊 Sound System initialized with {lang_name} using pre-generated WAV files")
        else:
            logger.warning("🔇 Sound System disabled - no audio available")
    
    def _initialize_tts_system(self):
        """Initialize TTS system and generate WAV files"""
        try:
            # Check if Piper is available
            if self._check_piper_availability():
                self.is_enabled = True
                logger.info("🔊 Using Piper TTS for WAV file generation")
                self._generate_missing_wav_files()
            else:
                logger.warning("🔇 Piper TTS not available - using MP3 fallback")
                self.is_enabled = True  # Still enable for MP3 playback
        except Exception as e:
            logger.error(f"❌ Error initializing TTS system: {e}")
            self.is_enabled = True  # Enable for MP3 fallback
    
    def _check_piper_availability(self):
        """Check if Piper TTS is available"""
        try:
            result = subprocess.run(['piper', '--help'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Check if model file exists
                piper_config = SOUND_SYSTEM.get('piper', {})
                model_path = piper_config.get('models', {}).get(self.language, 
                                                               piper_config.get('model_path', ''))
                if os.path.exists(model_path):
                    logger.info("✅ Piper TTS is available with model")
                    return True
                else:
                    logger.warning(f"⚠️ Piper model not found: {model_path}")
                    return False
            else:
                logger.warning("⚠️ Piper command not found")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            logger.warning("⚠️ Piper TTS not available")
            return False
    
    def _generate_missing_wav_files(self):
        """Generate only missing WAV files using Piper TTS"""
        logger.info("🎵 Checking and generating missing WAV files with Piper TTS...")
        
        # Get Piper configuration
        piper_config = SOUND_SYSTEM.get('piper', {})
        model_path = piper_config.get('models', {}).get(self.language, 
                                                       piper_config.get('model_path', ''))
        noise_scale = piper_config.get('noise', 0.667)
        length_scale = piper_config.get('length_penalty', 1.0)
        
        # Define all messages to generate
        messages = self._get_all_messages()
        
        # Generate only missing WAV files
        missing_files = []
        for message_key, message_data in messages.items():
            if isinstance(message_data, dict):
                # Handle messages with multiple variants (like greetings)
                for variant_key, text in message_data.items():
                    filename = f"{message_key}_{variant_key}.wav"
                    filepath = os.path.join(self.wav_files_dir, filename)
                    if not os.path.exists(filepath):
                        missing_files.append((text, filepath))
            elif isinstance(message_data, list):
                # Handle messages with multiple variants (like verification reminders)
                for i, text in enumerate(message_data):
                    filename = f"{message_key}_{i+1}.wav"
                    filepath = os.path.join(self.wav_files_dir, filename)
                    if not os.path.exists(filepath):
                        missing_files.append((text, filepath))
            else:
                # Handle simple text messages
                filename = f"{message_key}.wav"
                filepath = os.path.join(self.wav_files_dir, filename)
                if not os.path.exists(filepath):
                    missing_files.append((message_data, filepath))
        
        if missing_files:
            logger.info(f"📝 Generating {len(missing_files)} missing WAV files...")
            for text, filepath in missing_files:
                self._generate_wav_file(text, filepath, model_path, noise_scale, length_scale)
            logger.info("✅ Missing WAV files generation completed")
        else:
            logger.info("✅ All WAV files already exist")
    
    def _generate_wav_files(self):
        """Generate all required WAV files using Piper TTS (legacy method)"""
        logger.info("🎵 Generating all WAV files with Piper TTS...")
        
        # Get Piper configuration
        piper_config = SOUND_SYSTEM.get('piper', {})
        model_path = piper_config.get('models', {}).get(self.language, 
                                                       piper_config.get('model_path', ''))
        noise_scale = piper_config.get('noise', 0.667)
        length_scale = piper_config.get('length_penalty', 1.0)
        
        # Define all messages to generate
        messages = self._get_all_messages()
        
        # Generate WAV files for each message
        for message_key, message_data in messages.items():
            if isinstance(message_data, dict):
                # Handle messages with multiple variants (like greetings)
                for variant_key, text in message_data.items():
                    filename = f"{message_key}_{variant_key}.wav"
                    filepath = os.path.join(self.wav_files_dir, filename)
                    self._generate_wav_file(text, filepath, model_path, noise_scale, length_scale)
            elif isinstance(message_data, list):
                # Handle messages with multiple variants (like verification reminders)
                for i, text in enumerate(message_data):
                    filename = f"{message_key}_{i+1}.wav"
                    filepath = os.path.join(self.wav_files_dir, filename)
                    self._generate_wav_file(text, filepath, model_path, noise_scale, length_scale)
            else:
                # Handle simple text messages
                filename = f"{message_key}.wav"
                filepath = os.path.join(self.wav_files_dir, filename)
                self._generate_wav_file(message_data, filepath, model_path, noise_scale, length_scale)
        
        logger.info("✅ WAV file generation completed")
    
    def _get_all_messages(self):
        """Get all messages that need WAV files"""
        if self.language == 'gu':
            return {
                'person_detected': "વ્યક્તિ શોધાઈ ગઈ છે. કૃપા કરીને કેમેરા તરફ જુઓ.",
                'face_verification_request': "કૃપા કરીને ચહેરો દેખાડો. ચહેરાની ઓળખ માટે કેમેરા તરફ જુઓ.",
                'face_verification_reminder': [
                    "કૃપા કરીને ચહેરો દેખાડો.",
                    "ચહેરાની ઓળખ જરૂરી છે - કેમેરા તરફ જુઓ.",
                    "અંતિમ ચેતવણી - હવે ચહેરો દેખાડો."
                ],
                'verification_timeout': "સમય સમાપ્ત! ચહેરાની ઓળખ નિષ્ફળ.",
                'unknown_person_alert': "અજ્ઞાત વ્યક્તિ શોધાઈ ગઈ છે! સુરક્ષા ચેતવણી!",
                'security_breach': "સુરક્ષા ભંગ! અનધિકૃત વ્યક્તિ શોધાઈ ગઈ છે!",
                'known_person_greeting': "નમસ્તે {name}! આપનું સ્વાગત છે.",
                'time_based_greeting': {
                    'morning': "સુપ્રભાત!",
                    'afternoon': "શુભ બપોર!",
                    'evening': "શુભ સાંજ!"
                },
                'welcome_back': "પાછા આવ્યા માટે આભાર {name}!",
                'guest_mode_activated': "મહેમાન મોડ સક્રિય થયો છે. {host_name} સાથે મહેમાન આવ્યા છે.",
                'guest_mode_expired': "મહેમાન મોડ સમાપ્ત થયો છે. સામાન્ય સુરક્ષા પ્રોટોકોલ પર પાછા ફરી રહ્યા છીએ."
            }
        else:  # English
            return {
                'person_detected': "Hello!",
                'face_verification_request': "Please look at the camera for face verification.",
                'face_verification_reminder': [
                    "2.",
                    "4.",
                    "Final warning - Show your face now."
                ],
                'verification_timeout': "Time's up! Face verification failed.",
                'unknown_person_alert': "Unknown person detected! Security alert!",
                'security_breach': "Security breach! Unauthorized person detected!",
                'known_person_greeting': "Welcome.",
                'time_based_greeting': {
                    'morning': "Good morning!",
                    'afternoon': "Good afternoon!",
                    'evening': "Good evening!"
                },
                'welcome_back': "Welcome back!",
                'guest_mode_activated': "Guest mode activated.",
                'guest_mode_expired': "Guest mode expired."
            }
    
    def _generate_wav_file(self, text: str, filepath: str, model_path: str, noise_scale: float, length_scale: float):
        """Generate a single WAV file using Piper TTS"""
        try:
            # Escape text for shell safety
            safe_text = shlex.quote(text)
            
            # Build Piper command to generate WAV file
            cmd = [
                'bash', '-c',
                f'echo {safe_text} | piper --model {model_path} --length-scale {length_scale} --noise-scale {noise_scale} --output-file {filepath}'
            ]
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(filepath):
                logger.debug(f"✅ Generated WAV file: {os.path.basename(filepath)}")
            else:
                logger.error(f"❌ Failed to generate WAV file: {os.path.basename(filepath)}")
                logger.error(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout generating WAV file: {os.path.basename(filepath)}")
        except Exception as e:
            logger.error(f"❌ Error generating WAV file {os.path.basename(filepath)}: {e}")
    
    def _generate_single_wav_file(self, filename: str):
        """Generate a single WAV file based on filename"""
        try:
            # Get Piper configuration
            piper_config = SOUND_SYSTEM.get('piper', {})
            model_path = piper_config.get('models', {}).get(self.language, 
                                                           piper_config.get('model_path', ''))
            noise_scale = piper_config.get('noise', 0.667)
            length_scale = piper_config.get('length_penalty', 1.0)
            
            # Get all messages to find the right text
            messages = self._get_all_messages()
            
            # Find the text for this filename
            text = None
            for message_key, message_data in messages.items():
                if isinstance(message_data, dict):
                    # Handle messages with multiple variants (like greetings)
                    for variant_key, msg_text in message_data.items():
                        expected_filename = f"{message_key}_{variant_key}.wav"
                        if expected_filename == filename:
                            text = msg_text
                            break
                elif isinstance(message_data, list):
                    # Handle messages with multiple variants (like verification reminders)
                    for i, msg_text in enumerate(message_data):
                        expected_filename = f"{message_key}_{i+1}.wav"
                        if expected_filename == filename:
                            text = msg_text
                            break
                else:
                    # Handle simple text messages
                    expected_filename = f"{message_key}.wav"
                    if expected_filename == filename:
                        text = message_data
                        break
                
                if text:
                    break
            
            if text:
                filepath = os.path.join(self.wav_files_dir, filename)
                self._generate_wav_file(text, filepath, model_path, noise_scale, length_scale)
                logger.info(f"✅ Generated WAV file: {filename}")
            else:
                logger.warning(f"⚠️ Could not find text for filename: {filename}")
                
        except Exception as e:
            logger.error(f"❌ Error generating single WAV file {filename}: {e}")
    
    def play_wav_file(self, filename: str):
        """Play a WAV file using aplay or mpg123, generate if missing"""
        if not self.is_enabled:
            return
        
        filepath = os.path.join(self.wav_files_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ WAV file not found: {filename}")
            # Try to generate the missing WAV file
            if self._check_piper_availability():
                logger.info(f"🎵 Generating missing WAV file: {filename}")
                self._generate_single_wav_file(filename)
            else:
                logger.warning(f"⚠️ Cannot generate {filename} - Piper TTS not available")
                return
        
        try:
            # Stop any current audio
            self.stop_audio()
            
            # Use aplay for WAV files (Linux/Mac) or mpg123 for MP3
            if filename.endswith('.wav'):
                cmd = ['aplay', filepath]
            else:
                cmd = ['mpg123', '-q', filepath]  # -q for quiet mode
            
            # Start audio playback
            self.current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.debug(f"🔊 Playing audio: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error playing audio {filename}: {e}")
    
    def play_mp3_file(self, filepath: str):
        """Play an MP3 file using mpg123"""
        if not self.is_enabled or not os.path.exists(filepath):
            return
        
        try:
            # Stop any current audio
            self.stop_audio()
            
            # Use mpg123 for MP3 files
            cmd = ['mpg123', '-q', filepath]
            self.current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.debug(f"🔊 Playing MP3: {os.path.basename(filepath)}")
            
        except Exception as e:
            logger.error(f"❌ Error playing MP3 {filepath}: {e}")
    
    def stop_audio(self):
        """Stop current audio playback"""
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=2)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    self.current_process.kill()
                except ProcessLookupError:
                    pass
            finally:
                self.current_process = None
    
    def is_alarm_playing(self):
        """Check if alarm is currently playing"""
        return self.alarm_active
    
    def stop_alarm(self):
        """Stop any ongoing alarm"""
        if self.alarm_active:
            self.stop_audio()
            self.alarm_active = False
            logger.info("🔇 Alarm stopped")
    
    def start_alarm(self):
        """Start alarm if not already playing"""
        if not self.alarm_active:
            self.alarm_active = True
            logger.info("🚨 Alarm started")
        else:
            logger.info("🚨 Alarm already playing - skipping")
    
    def set_language(self, language: str):
        """Change language and regenerate WAV files"""
        if language != self.language:
            self.language = language.lower()
            logger.info(f"🔊 Language changed to {self.language}")
            if self.is_enabled:
                self._generate_wav_files()
    
    def get_language(self) -> str:
        """Get current language"""
        return self.language
    
    def disable(self):
        """Disable sound system"""
        self.is_enabled = False
        self.stop_audio()
        logger.info("🔇 Sound system disabled")
    
    def enable(self):
        """Enable sound system"""
        self.is_enabled = True
        logger.info("🔊 Sound system enabled")


# Global sound system instance
_sound_system = None

def get_sound_system(language: str = None) -> SoundSystem:
    """Get global sound system instance"""
    global _sound_system
    if language is None:
        language = SOUND_SYSTEM.get('language', 'en')
    
    if _sound_system is None or _sound_system.get_language() != language:
        _sound_system = SoundSystem(language)
    return _sound_system

def play_wav(filename: str):
    """Quick play WAV file function"""
    get_sound_system().play_wav_file(filename)

def play_mp3(filepath: str):
    """Quick play MP3 file function"""
    get_sound_system().play_mp3_file(filepath)

def stop_sound():
    """Stop all sound"""
    get_sound_system().stop_audio()

def enable_sound():
    """Enable sound system"""
    get_sound_system().enable()

def disable_sound():
    """Disable sound system"""
    get_sound_system().disable()


if __name__ == "__main__":
    # Test the sound system
    logging.basicConfig(level=logging.INFO)
    
    print("🔊 Testing Sound System with Pre-generated WAV Files")
    print("=" * 60)
    
    # Initialize sound system
    sound = get_sound_system()
    
    if sound.is_enabled:
        print("✅ Sound system is enabled")
        
        # Test WAV file playback
        print("\n1. Testing WAV file playback...")
        sound.play_wav_file("person_detected.wav")
        time.sleep(3)
        
        sound.play_wav_file("face_verification_request.wav")
        time.sleep(3)
        
        sound.play_wav_file("unknown_person_alert.wav")
        time.sleep(3)
        
        print("\n✅ Sound system test completed!")
    else:
        print("❌ Sound system is disabled - Piper TTS not available")
        print("Please install Piper TTS:")
        print("  See: https://github.com/rhasspy/piper")