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
import shlex
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
        self.tts_system = None  # Current TTS system being used
        self.piper_voice = None  # Persistent Piper voice instance
        self.alarm_active = False  # Track if alarm is currently playing
        
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
        
        # Initialize TTS system with fallback chain
        self._initialize_tts_system()
        
        # Initialize persistent voice if using Piper
        if self.tts_system == 'piper':
            self._initialize_piper_voice()
        
        if self.is_enabled:
            lang_name = "Gujarati" if self.language == 'gu' else "English"
            logger.info(f"🔊 Sound System initialized with {lang_name} female voice using {self.tts_system}")
        else:
            logger.warning("🔇 Sound System disabled - no audio available")
    
    def _initialize_tts_system(self):
        """Initialize TTS system with fallback chain"""
        primary_tts = SOUND_SYSTEM.get('primary_tts', 'piper')
        fallback_chain = SOUND_SYSTEM.get('fallback_chain', ['espeak', 'pyttsx3', 'win32', 'winsound'])
        
        # Try primary TTS system first
        if primary_tts == 'piper' and self._check_piper_availability():
            self.tts_system = 'piper'
            self.is_enabled = True
            logger.info("🔊 Using Piper TTS as primary system")
            return
        
        # Try fallback chain
        for tts_system in fallback_chain:
            if tts_system == 'espeak' and self._check_espeak_availability():
                self.tts_system = 'espeak'
                self.is_enabled = True
                logger.info("🔊 Using espeak-ng as fallback system")
                return
            elif tts_system == 'pyttsx3' and self._check_pyttsx3_availability():
                self.tts_system = 'pyttsx3'
                self.is_enabled = True
                logger.info("🔊 Using pyttsx3 as fallback system")
                return
            elif tts_system == 'win32' and self._check_win32_availability():
                self.tts_system = 'win32'
                self.is_enabled = True
                logger.info("🔊 Using Windows SAPI as fallback system")
                return
            elif tts_system == 'winsound' and self._check_winsound_availability():
                self.tts_system = 'winsound'
                self.is_enabled = True
                logger.info("🔊 Using winsound as fallback system")
                return
        
        # No TTS system available
        self.tts_system = None
        self.is_enabled = False
        logger.warning("🔇 No TTS system available")
    
    def _check_piper_availability(self):
        """Check if Piper TTS is available"""
        try:
            # Check if piper command exists
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
    
    def _check_espeak_availability(self):
        """Check if espeak-ng is installed and available"""
        try:
            result = subprocess.run(['espeak-ng', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ espeak-ng is available")
                return True
            else:
                logger.warning("⚠️ espeak-ng not found")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            logger.warning("⚠️ espeak-ng not available")
            return False
    
    def _check_pyttsx3_availability(self):
        """Check if pyttsx3 is available"""
        return PYTTSX3_AVAILABLE and platform.system() == 'Windows'
    
    def _check_win32_availability(self):
        """Check if Windows SAPI is available"""
        return WIN32_AVAILABLE and platform.system() == 'Windows'
    
    def _check_winsound_availability(self):
        """Check if winsound is available"""
        return WINSOUND_AVAILABLE and platform.system() == 'Windows'
    
    def _initialize_piper_voice(self):
        """Initialize persistent Piper voice for faster speech"""
        try:
            # Get Piper configuration
            piper_config = SOUND_SYSTEM.get('piper', {})
            model_path = piper_config.get('models', {}).get(self.language, 
                                                          piper_config.get('model_path', ''))
            
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Piper model not found: {model_path}")
                return
            
            # Load Piper voice model once
            try:
                from piper import PiperVoice
                logger.info("🔊 Loading Piper TTS model...")
                self.piper_voice = PiperVoice.load(model_path)
                logger.info(f"✅ Piper TTS model loaded successfully from {model_path}")
            except ImportError:
                logger.info("🔊 Using command-line Piper (piper library not available)")
                self.piper_voice = None
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Piper TTS model: {e}")
                self.piper_voice = None
                
        except Exception as e:
            logger.error(f"❌ Error initializing Piper voice: {e}")
            self.piper_voice = None
    
    def is_alarm_playing(self):
        """Check if alarm is currently playing"""
        return self.alarm_active
    
    def stop_alarm(self):
        """Stop any ongoing alarm"""
        if self.alarm_active:
            try:
                # Stop current process if it's an alarm
                if self.current_process and self.current_process.poll() is None:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=5)
                self.alarm_active = False
                logger.info("🔇 Alarm stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping alarm: {e}")
    
    def start_alarm(self):
        """Start alarm if not already playing"""
        if not self.alarm_active:
            self.alarm_active = True
            logger.info("🚨 Alarm started")
        else:
            logger.info("🚨 Alarm already playing - skipping")
    
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
        """Speak text immediately using the selected TTS system"""
        try:
            # Stop any current speech
            self.stop_speaking()
            
            logger.info(f"🔊 Speaking: {text}")
            
            # Use the selected TTS system
            if self.tts_system == 'piper':
                self._speak_with_piper(text)
            elif self.tts_system == 'espeak':
                self._speak_with_espeak(text)
            elif self.tts_system == 'pyttsx3':
                self._speak_with_pyttsx3(text)
            elif self.tts_system == 'win32':
                self._speak_with_win32(text)
            elif self.tts_system == 'winsound':
                self._speak_with_winsound(text)
            else:
                logger.warning("🔇 No TTS system available")
                
        except Exception as e:
            logger.error(f"❌ Error speaking: {e}")
            self.is_speaking = False
    
    def _speak_with_piper(self, text: str):
        """Speak using Piper TTS with audio output"""
        try:
            self.is_speaking = True
            
            # Get Piper configuration
            piper_config = SOUND_SYSTEM.get('piper', {})
            model_path = piper_config.get('models', {}).get(self.language, 
                                                          piper_config.get('model_path', ''))
            noise_scale = piper_config.get('noise', 0.667)
            length_scale = piper_config.get('length_penalty', 1.0)
            
            # Try to use persistent voice first (much faster)
            if self.piper_voice is not None:
                self._speak_with_persistent_piper(text, noise_scale, length_scale)
            else:
                # Fallback to command-line Piper
                self._speak_with_piper_command(text, model_path, noise_scale, length_scale)
            
        except Exception as e:
            logger.error(f"❌ Error with Piper: {e}")
        finally:
            self.is_speaking = False
            
            # Process next in queue
            if self.sound_queue:
                next_text = self.sound_queue.pop(0)
                logger.debug(f"📝 Processing queued speech: {next_text[:50]}...")
                self._speak_immediately(next_text)
    
    def _speak_with_persistent_piper(self, text: str, noise_scale: float, length_scale: float):
        """Speak using persistent Piper voice (fastest method)"""
        try:
            logger.info(f"🔊 Speaking with persistent Piper: {text[:50]}...")
            
            # Generate audio using persistent voice (Piper doesn't support parameters in synthesize)
            audio_data = self.piper_voice.synthesize(text)
            
            # Play audio using aplay with model's sample rate
            sample_rate = self.piper_voice.config.sample_rate
            
            # Use aplay to play the audio data
            cmd = ['aplay', '-r', str(sample_rate), '-f', 'S16_LE', '-c', '1', '-']
            
            self.current_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.current_process.communicate(input=audio_data)
            
            logger.info(f"🔊 Spoke with persistent Piper: {text[:50]}...")
                
        except Exception as e:
            logger.error(f"❌ Error with persistent Piper: {e}")
            # Fallback to command-line method
            self._speak_with_piper_command(text, None, noise_scale, length_scale)
    
    def _speak_with_piper_command(self, text: str, model_path: str, noise_scale: float, length_scale: float):
        """Speak using command-line Piper (fallback method)"""
        try:
            # Get model path if not provided
            if model_path is None:
                piper_config = SOUND_SYSTEM.get('piper', {})
                model_path = piper_config.get('models', {}).get(self.language, 
                                                              piper_config.get('model_path', ''))
            
            # Escape text for shell safety
            safe_text = shlex.quote(text)
            
            # Get audio configuration
            piper_config = SOUND_SYSTEM.get('piper', {})
            use_aplay = piper_config.get('use_aplay', True)
            sample_rate = piper_config.get('sample_rate', 22050)
            channels = piper_config.get('channels', 1)
            audio_format = piper_config.get('format', 'S16_LE')
            
            # Build Piper command with correct parameters
            if use_aplay:
                cmd = [
                    'bash', '-c',
                    f'echo {safe_text} | piper --model {model_path} --length-scale {length_scale} --noise-scale {noise_scale} --output-raw | aplay -r {sample_rate} -f {audio_format} -c {channels} -'
                ]
            else:
                # Fallback to direct piper output
                cmd = [
                    'bash', '-c',
                    f'echo {safe_text} | piper --model {model_path} --length-scale {length_scale} --noise-scale {noise_scale}'
                ]
            
            # Execute command (don't capture stdout/stderr for audio output)
            logger.info(f"🔊 Speaking with command-line Piper: {text[:50]}...")
            logger.debug(f"🔧 Piper command: {' '.join(cmd)}")
            
            # Start the process without capturing output (for audio)
            self.current_process = subprocess.Popen(cmd, stdout=None, stderr=None)
            
            # Wait for completion
            return_code = self.current_process.wait()
            
            if return_code == 0:
                logger.info(f"🔊 Spoke with command-line Piper: {text[:50]}...")
            else:
                logger.error(f"❌ Piper command failed with return code: {return_code}")
                
        except Exception as e:
            logger.error(f"❌ Error with command-line Piper: {e}")
    
    def _speak_with_espeak(self, text: str):
        """Speak using espeak-ng"""
        try:
            # Build command
            cmd = self._build_espeak_command(text)
            
            # Execute command
            self.current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.current_process.wait()
            
            logger.info(f"🔊 Spoke with espeak-ng: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error with espeak-ng: {e}")
        finally:
            self.is_speaking = False
            
            # Process next in queue
            if self.sound_queue:
                next_text = self.sound_queue.pop(0)
                logger.debug(f"📝 Processing queued speech: {next_text[:50]}...")
                self._speak_immediately(next_text)

    def _speak_with_pyttsx3(self, text: str):
        """Speak using pyttsx3 (best Windows TTS option)"""
        try:
            self.is_speaking = True
            
            # Initialize pyttsx3 engine with error handling
            try:
                engine = pyttsx3.init()
            except Exception as init_error:
                logger.warning(f"pyttsx3 init failed: {init_error}")
                # Try to use existing engine or fallback
                return
            
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
                engine.setProperty('rate', 163)  # Speed
                engine.setProperty('volume', 1.0)  # Volume
            
            # Speak the text with timeout protection
            engine.say(text)
            try:
                engine.runAndWait()
            except Exception as run_error:
                if "run loop already started" in str(run_error):
                    logger.warning("pyttsx3 run loop conflict - skipping speech")
                else:
                    raise run_error
            
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
