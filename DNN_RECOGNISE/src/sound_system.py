#!/usr/bin/env python3
"""
Sound System for CCTV Security System
Uses espeak-ng for text-to-speech with Gujarati female voice
"""

import subprocess
import logging
import threading
import time
import os
import platform
from typing import Optional

# Import settings
try:
    from settings.settings import SOUND_SYSTEM
except ImportError:
    # Fallback settings if import fails
    SOUND_SYSTEM = {
        'enabled': True,
        'language': 'gu',
        'voice_parameters': {
            'gujarati': {'speed': 163, 'pitch': 55, 'volume': 100, 'amplitude': 100},
            'english': {'speed': 150, 'pitch': 50, 'volume': 100, 'amplitude': 100}
        },
        'windows_speech_fallback': True,
        'speech_queue_enabled': True,
        'priority_speech_enabled': True
    }

# Windows-specific imports
if platform.system() == 'Windows':
    try:
        import pyttsx3
        PYTTSX3_AVAILABLE = True
    except ImportError:
        PYTTSX3_AVAILABLE = False
    
    try:
        import win32com.client
        WIN32_AVAILABLE = True
    except ImportError:
        WIN32_AVAILABLE = False
    
    try:
        import winsound
        WINSOUND_AVAILABLE = True
    except ImportError:
        WINSOUND_AVAILABLE = False
else:
    PYTTSX3_AVAILABLE = False
    WIN32_AVAILABLE = False
    WINSOUND_AVAILABLE = False

logger = logging.getLogger(__name__)

class SoundSystem:
    """Sound system using espeak-ng with English and Gujarati support"""
    
    def __init__(self, language: str = None):
        """Initialize sound system
        :param language: 'en' for English, 'gu' for Gujarati. If None, uses settings default.
        """
        self.is_enabled = False
        self.is_speaking = False
        self.sound_queue = []
        self.current_process: Optional[subprocess.Popen] = None
        self.use_winsound = False
        
        # Use language from parameter or settings
        if language is None:
            self.language = SOUND_SYSTEM.get('language', 'gu').lower()
        else:
            self.language = language.lower()
        
        # Check if sound system is enabled in settings
        if not SOUND_SYSTEM.get('enabled', True):
            logger.info("🔇 Sound system disabled in settings")
            return
        
        # Get voice parameters from settings
        lang_key = 'gujarati' if self.language == 'gu' else 'english'
        voice_config = SOUND_SYSTEM.get('voice_parameters', {}).get(lang_key, {})
        
        # espeak-ng parameters for voice
        self.voice_params = {
            'language': self.language,
            'speed': voice_config.get('speed', 163 if self.language == 'gu' else 150),
            'pitch': voice_config.get('pitch', 55 if self.language == 'gu' else 50),
            'volume': voice_config.get('volume', 100),
            'amplitude': voice_config.get('amplitude', 100)
        }
        
        # Check if espeak-ng is available
        self._check_espeak_availability()
        
        # If espeak-ng not available, check for Windows speech fallbacks
        if (not self.is_enabled and 
            platform.system() == 'Windows' and 
            SOUND_SYSTEM.get('windows_speech_fallback', True)):
            
            # Try pyttsx3 first (best option)
            if PYTTSX3_AVAILABLE:
                logger.info("🔊 espeak-ng not available, using pyttsx3 on Windows")
                self.is_enabled = True
                self.use_pyttsx3 = True
            # Try Windows SAPI (win32com.client)
            elif WIN32_AVAILABLE:
                logger.info("🔊 espeak-ng not available, using Windows SAPI on Windows")
                self.is_enabled = True
                self.use_win32 = True
            # Fallback to winsound beeps
            elif WINSOUND_AVAILABLE:
                logger.info("🔊 espeak-ng not available, using winsound beeps on Windows")
                self.is_enabled = True
                self.use_winsound = True
        
        if self.is_enabled:
            lang_name = "Gujarati" if self.language == 'gu' else "English"
            logger.info(f"🔊 Sound System initialized with {lang_name} female voice")
        else:
            logger.warning("🔇 Sound System disabled - no audio available")
    
    def _check_espeak_availability(self):
        """Check if espeak-ng is installed and available"""
        try:
            result = subprocess.run(['espeak-ng', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ espeak-ng is available")
                self.is_enabled = True
            else:
                logger.warning("⚠️ espeak-ng not found, sound disabled")
                self.is_enabled = False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            logger.warning("⚠️ espeak-ng not available, sound disabled")
            self.is_enabled = False
    
    def _build_espeak_command(self, text: str) -> list:
        """Build espeak-ng command with parameters"""
        import platform
        
        # Build voice parameter based on language
        if self.language == 'gu':
            voice_param = f'gu+f{self.voice_params["pitch"]}'  # Gujarati female voice
        else:
            voice_param = f'en+f{self.voice_params["pitch"]}'  # English female voice
        
        cmd = [
            'espeak-ng',
            '-v', voice_param,                              # Voice with pitch
            '-s', str(self.voice_params['speed']),         # Speed
            '-a', str(self.voice_params['amplitude']),     # Amplitude
            '-g', '10',                                     # Gap between words
        ]
        
        # Windows compatibility
        if platform.system() == 'Windows':
            cmd.extend(['-w', 'temp_audio.wav'])  # Output to WAV file on Windows
        else:
            cmd.append('--stdout')  # Output to stdout on Linux/Mac
        
        # Add text as argument
        cmd.append(text)
        return cmd
    
    def speak(self, text: str, priority: bool = False):
        """Speak text using espeak-ng or winsound fallback"""
        if not self.is_enabled:
            logger.debug("🔇 Sound disabled, skipping speech")
            return
        
        if not text or not text.strip():
            logger.debug("🔇 Empty text, skipping speech")
            return
        
        # If already speaking and not priority, queue the text (if queue enabled)
        if (self.is_speaking and not priority and 
            SOUND_SYSTEM.get('speech_queue_enabled', True)):
            self.sound_queue.append(text)
            logger.debug(f"📝 Queued speech: {text[:50]}...")
            return
        
        # Speak immediately
        self._speak_immediately(text)
    
    def _speak_immediately(self, text: str):
        """Speak text immediately"""
        try:
            # Stop any current speech
            self.stop_speaking()
            
            logger.info(f"🔊 Speaking: {text}")
            
            # Use Windows speech fallbacks if espeak-ng not available
            if hasattr(self, 'use_pyttsx3') and self.use_pyttsx3:
                self._speak_with_pyttsx3(text)
                return
            elif hasattr(self, 'use_win32') and self.use_win32:
                self._speak_with_win32(text)
                return
            elif hasattr(self, 'use_winsound') and self.use_winsound:
                self._speak_with_winsound(text)
                return
            
            # Build command
            cmd = self._build_espeak_command(text)
            
            # Start speaking in a separate thread
            def speak_thread():
                try:
                    self.is_speaking = True
                    import platform
                    
                    if platform.system() == 'Windows':
                        # Windows: Generate WAV file and play with Windows Media Player
                        espeak_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        espeak_process.wait()
                        
                        # Play the generated WAV file
                        if os.path.exists('temp_audio.wav'):
                            play_process = subprocess.Popen(['powershell', '-c', 
                                '(New-Object Media.SoundPlayer "temp_audio.wav").PlaySync()'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            self.current_process = play_process
                            play_process.wait()
                            
                            # Clean up temp file
                            try:
                                os.remove('temp_audio.wav')
                            except:
                                pass
                    else:
                        # Linux/Mac: Use aplay to play the audio output from espeak-ng
                        espeak_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        aplay_process = subprocess.Popen(['aplay'], stdin=espeak_process.stdout, stderr=subprocess.PIPE)
                        
                        self.current_process = aplay_process
                        
                        # Wait for completion
                        aplay_process.wait()
                        espeak_process.wait()
                    
                except Exception as e:
                    logger.error(f"❌ Error in speech thread: {e}")
                finally:
                    self.is_speaking = False
                    self.current_process = None
                    
                    # Process next in queue
                    if self.sound_queue:
                        next_text = self.sound_queue.pop(0)
                        logger.debug(f"📝 Processing queued speech: {next_text[:50]}...")
                        self._speak_immediately(next_text)
            
            # Start speech thread
            speech_thread = threading.Thread(target=speak_thread, daemon=True)
            speech_thread.start()
            
        except Exception as e:
            logger.error(f"❌ Error starting speech: {e}")
            self.is_speaking = False

    def _speak_with_pyttsx3(self, text: str):
        """Speak using pyttsx3 (best Windows TTS option)"""
        try:
            self.is_speaking = True
            
            # Initialize pyttsx3 engine
            engine = pyttsx3.init()
            
            # Set voice properties based on language
            if self.language == 'gu':
                # Try to find a female voice for Gujarati
                voices = engine.getProperty('voices')
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                engine.setProperty('rate', 163)  # Speed
                engine.setProperty('volume', 1.0)  # Volume
            else:  # English
                # Try to find a female voice for English
                voices = engine.getProperty('voices')
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                engine.setProperty('rate', 150)  # Speed
                engine.setProperty('volume', 1.0)  # Volume
            
            # Speak the text
            engine.say(text)
            engine.runAndWait()
            
            logger.info(f"🔊 Spoke with pyttsx3: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error with pyttsx3: {e}")
        finally:
            self.is_speaking = False
            
            # Process next in queue
            if self.sound_queue:
                next_text = self.sound_queue.pop(0)
                logger.debug(f"📝 Processing queued speech: {next_text[:50]}...")
                self._speak_immediately(next_text)
    
    def _speak_with_win32(self, text: str):
        """Speak using Windows SAPI (win32com.client)"""
        try:
            self.is_speaking = True
            
            # Initialize Windows SAPI
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            
            # Set voice properties
            if self.language == 'gu':
                # Try to find a female voice
                voices = speaker.GetVoices()
                for i in range(voices.Count):
                    voice = voices.Item(i)
                    if 'female' in voice.GetDescription().lower() or 'zira' in voice.GetDescription().lower():
                        speaker.Voice = voice
                        break
                speaker.Rate = 2  # Speed (adjust as needed)
                speaker.Volume = 100  # Volume
            else:  # English
                # Try to find a female voice
                voices = speaker.GetVoices()
                for i in range(voices.Count):
                    voice = voices.Item(i)
                    if 'female' in voice.GetDescription().lower() or 'zira' in voice.GetDescription().lower():
                        speaker.Voice = voice
                        break
                speaker.Rate = 0  # Speed (adjust as needed)
                speaker.Volume = 100  # Volume
            
            # Speak the text
            speaker.Speak(text)
            
            logger.info(f"🔊 Spoke with Windows SAPI: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error with Windows SAPI: {e}")
        finally:
            self.is_speaking = False
            
            # Process next in queue
            if self.sound_queue:
                next_text = self.sound_queue.pop(0)
                logger.debug(f"📝 Processing queued speech: {next_text[:50]}...")
                self._speak_immediately(next_text)
    
    def _speak_with_winsound(self, text: str):
        """Speak using winsound fallback on Windows"""
        try:
            self.is_speaking = True
            
            # Map different message types to different beep patterns
            if "unknown" in text.lower() or "alert" in text.lower() or "security" in text.lower():
                # Security alert - multiple beeps
                for _ in range(3):
                    winsound.Beep(800, 200)  # High frequency beep
                    time.sleep(0.1)
            elif "verification" in text.lower() or "face" in text.lower():
                # Verification request - single beep
                winsound.Beep(600, 300)  # Medium frequency beep
            elif "welcome" in text.lower() or "greeting" in text.lower():
                # Welcome message - pleasant beep
                winsound.Beep(400, 200)  # Lower frequency beep
            else:
                # Default - single beep
                winsound.Beep(500, 250)
            
            logger.info(f"🔊 Played winsound for: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error with winsound: {e}")
        finally:
            self.is_speaking = False
            
            # Process next in queue
            if self.sound_queue:
                next_text = self.sound_queue.pop(0)
                logger.debug(f"📝 Processing queued speech: {next_text[:50]}...")
                self._speak_immediately(next_text)
    
    def stop_speaking(self):
        """Stop current speech"""
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
                self.is_speaking = False
    
    def clear_queue(self):
        """Clear speech queue"""
        self.sound_queue.clear()
        logger.debug("🧹 Speech queue cleared")
    
    def set_voice_params(self, speed: int = None, pitch: int = None, volume: int = None):
        """Update voice parameters"""
        if speed is not None:
            self.voice_params['speed'] = max(80, min(500, speed))
        if pitch is not None:
            self.voice_params['pitch'] = max(0, min(99, pitch))
        if volume is not None:
            self.voice_params['volume'] = max(0, min(200, volume))
        
        logger.info(f"🎛️ Voice params updated: speed={self.voice_params['speed']}, "
                   f"pitch={self.voice_params['pitch']}, volume={self.voice_params['volume']}")
    
    def enable(self):
        """Enable sound system"""
        self.is_enabled = True
        logger.info("🔊 Sound system enabled")
    
    def set_language(self, language: str):
        """Change language (en or gu)"""
        old_lang = self.language
        self.language = language.lower()
        
        # Get voice parameters from settings
        lang_key = 'gujarati' if self.language == 'gu' else 'english'
        voice_config = SOUND_SYSTEM.get('voice_parameters', {}).get(lang_key, {})
        
        # Update voice parameters
        self.voice_params = {
            'language': self.language,
            'speed': voice_config.get('speed', 163 if self.language == 'gu' else 150),
            'pitch': voice_config.get('pitch', 55 if self.language == 'gu' else 50),
            'volume': voice_config.get('volume', 100),
            'amplitude': voice_config.get('amplitude', 100)
        }
        
        lang_name = "Gujarati" if self.language == 'gu' else "English"
        logger.info(f"🔊 Language changed from {old_lang} to {lang_name}")
    
    def get_language(self) -> str:
        """Get current language"""
        return self.language
    
    def disable(self):
        """Disable sound system"""
        self.is_enabled = False
        self.stop_speaking()
        self.clear_queue()
        logger.info("🔇 Sound system disabled")
    
    def is_available(self) -> bool:
        """Check if sound system is available"""
        return self.is_enabled and not self.is_speaking


# Predefined messages for CCTV system
class CCTVMessages:
    """Predefined messages for CCTV system in English and Gujarati"""
    
    def __init__(self, sound_system: SoundSystem):
        self.sound = sound_system
        
        # Message templates in both languages
        self.messages = {
            'en': {
                'person_detected': "Person detected. Please look at the camera.",
                'face_verification_request': "Please show your face. Look at the camera for face verification.",
                'face_verification_reminder': [
                    "Please show your face for verification.",
                    "Face verification required - Look at camera.",
                    "Final warning - Show your face now."
                ],
                'verification_timeout': "Time's up! Face verification failed.",
                'unknown_person_alert': "Unknown person detected! Security alert!",
                'security_breach': "Security breach! Unauthorized person detected!",
                'known_person_greeting': "Hello {name}! Welcome.",
                'time_based_greeting': [
                    "Good morning!",
                    "Good afternoon!",
                    "Good evening!"
                ],
                'welcome_back': "Welcome back {name}!",
                'guest_mode_activated': "Guest mode activated. {host_name} has a guest.",
                'guest_mode_expired': "Guest mode expired. Reverting to normal security protocols."
            },
            'gu': {
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
                'time_based_greeting': [
                    "સુપ્રભાત!",
                    "શુભ બપોર!",
                    "શુભ સાંજ!"
                ],
                'welcome_back': "પાછા આવ્યા માટે આભાર {name}!",
                'guest_mode_activated': "મહેમાન મોડ સક્રિય થયો છે. {host_name} સાથે મહેમાન આવ્યા છે.",
                'guest_mode_expired': "મહેમાન મોડ સમાપ્ત થયો છે. સામાન્ય સુરક્ષા પ્રોટોકોલ પર પાછા ફરી રહ્યા છીએ."
            }
        }
    
    def person_detected(self, location: tuple = None):
        """Announce person detection"""
        language = self.sound.get_language()
        message = self.messages[language]['person_detected']
        self.sound.speak(message)
    
    def face_verification_request(self):
        """Request face verification"""
        language = self.sound.get_language()
        message = self.messages[language]['face_verification_request']
        self.sound.speak(message)
    
    def face_verification_reminder(self, count: int):
        """Reminder for face verification"""
        language = self.sound.get_language()
        reminders = self.messages[language]['face_verification_reminder']
        
        if count <= 2:
            message = reminders[0]
        elif count <= 4:
            message = reminders[1]
        else:
            message = reminders[2]
        
        self.sound.speak(message)
    
    def unknown_person_alert(self):
        """Alert for unknown person"""
        language = self.sound.get_language()
        message = self.messages[language]['unknown_person_alert']
        self.sound.speak(message, priority=True)
    
    def security_breach(self):
        """Security breach alert"""
        language = self.sound.get_language()
        message = self.messages[language]['security_breach']
        self.sound.speak(message, priority=True)
    
    def known_person_greeting(self, name: str):
        """Greet known person"""
        language = self.sound.get_language()
        message = self.messages[language]['known_person_greeting'].format(name=name)
        self.sound.speak(message, priority=True)
    
    def time_based_greeting(self):
        """Time-based greeting"""
        from datetime import datetime
        current_hour = datetime.now().hour
        language = self.sound.get_language()
        greetings = self.messages[language]['time_based_greeting']
        
        if 5 <= current_hour < 12:
            message = greetings[0]
        elif 12 <= current_hour < 17:
            message = greetings[1]
        else:
            message = greetings[2]
        
        self.sound.speak(message, priority=True)
    
    def guest_mode_activated(self, host_name: str):
        """Guest mode activation"""
        language = self.sound.get_language()
        message = self.messages[language]['guest_mode_activated'].format(host_name=host_name)
        self.sound.speak(message)
    
    def guest_mode_expired(self):
        """Guest mode expiration"""
        language = self.sound.get_language()
        message = self.messages[language]['guest_mode_expired']
        self.sound.speak(message)
    
    def welcome_back(self, name: str):
        """Welcome back message"""
        language = self.sound.get_language()
        message = self.messages[language]['welcome_back'].format(name=name)
        self.sound.speak(message)
    
    def verification_timeout(self):
        """Verification timeout warning"""
        language = self.sound.get_language()
        message = self.messages[language]['verification_timeout']
        self.sound.speak(message, priority=True)


# Global sound system instance
_sound_system = None
_messages = None

def get_sound_system(language: str = None) -> SoundSystem:
    """Get global sound system instance
    :param language: 'en' for English, 'gu' for Gujarati. If None, uses settings default.
    """
    global _sound_system
    if language is None:
        language = SOUND_SYSTEM.get('language', 'gu')
    
    if _sound_system is None or _sound_system.get_language() != language:
        _sound_system = SoundSystem(language)
        # Update messages if they exist
        global _messages
        if _messages is not None:
            _messages = CCTVMessages(_sound_system)
    return _sound_system

def get_messages(language: str = None) -> CCTVMessages:
    """Get global messages instance
    :param language: 'en' for English, 'gu' for Gujarati. If None, uses settings default.
    """
    global _messages
    if language is None:
        language = SOUND_SYSTEM.get('language', 'gu')
    
    if _messages is None or get_sound_system().get_language() != language:
        _messages = CCTVMessages(get_sound_system(language))
    return _messages

def speak(text: str, priority: bool = False):
    """Quick speak function"""
    get_sound_system().speak(text, priority)

def stop_sound():
    """Stop all sound"""
    get_sound_system().stop_speaking()

def enable_sound():
    """Enable sound system"""
    get_sound_system().enable()

def disable_sound():
    """Disable sound system"""
    get_sound_system().disable()


if __name__ == "__main__":
    # Test the sound system
    logging.basicConfig(level=logging.INFO)
    
    print("🔊 Testing Sound System with Gujarati Female Voice")
    print("=" * 50)
    
    # Initialize sound system
    sound = get_sound_system()
    messages = get_messages()
    
    if sound.is_enabled:
        print("✅ Sound system is enabled")
        
        # Test basic speech
        print("\n1. Testing basic speech...")
        sound.speak("હેલો વર્લ્ડ! આ ટેસ્ટ છે.")
        time.sleep(3)
        
        # Test messages
        print("\n2. Testing CCTV messages...")
        messages.person_detected()
        time.sleep(3)
        
        messages.face_verification_request()
        time.sleep(3)
        
        messages.known_person_greeting("રાજ")
        time.sleep(3)
        
        messages.time_based_greeting()
        time.sleep(3)
        
        print("\n✅ Sound system test completed!")
    else:
        print("❌ Sound system is disabled - espeak-ng not available")
        print("Please install espeak-ng:")
        print("  Ubuntu/Debian: sudo apt install espeak-ng")
        print("  CentOS/RHEL: sudo yum install espeak-ng")
        print("  macOS: brew install espeak-ng")
