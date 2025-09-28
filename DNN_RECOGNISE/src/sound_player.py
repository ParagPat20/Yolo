#!/usr/bin/env python3
"""
Sound Player for CCTV Security System
Plays pre-generated WAV files and MP3 sounds quickly
"""

import subprocess
import logging
import threading
import time
import os
import platform
from typing import Optional
from datetime import datetime

# Import settings
try:
    from settings.settings import AUDIO, SOUND_SYSTEM
except ImportError:
    # Fallback settings if import fails
    AUDIO = {
        'wav_files_dir': 'sounds/wav',
        'alarm_sound_path': 'sounds/alarm.mp3',
        'verification_beep_path': 'sounds/verification_beep.mp3',
        'use_mp3_sounds': True,
        'mp3_player_linux': 'mpg123',
        'mp3_player_windows': 'powershell',
        'alarm_duration_minutes': 2,
        'alarm_loop_interval': 5
    }
    SOUND_SYSTEM = {
        'enabled': True,
        'language': 'en',
        'wav_files_dir': 'sounds/wav'
    }

logger = logging.getLogger(__name__)

class SoundPlayer:
    """Sound player for pre-generated audio files"""
    
    def __init__(self):
        """Initialize sound player"""
        self.is_enabled = AUDIO.get('use_mp3_sounds', True)
        self.wav_files_dir = SOUND_SYSTEM.get('wav_files_dir', 'sounds/wav')
        self.alarm_sound_path = AUDIO.get('alarm_sound_path', 'sounds/alarm.mp3')
        self.verification_beep_path = AUDIO.get('verification_beep_path', 'sounds/verification_beep.mp3')
        self.alarm_duration_minutes = AUDIO.get('alarm_duration_minutes', 2)
        self.alarm_loop_interval = AUDIO.get('alarm_loop_interval', 5)
        
        self.alarm_active = False
        self.current_process: Optional[subprocess.Popen] = None
        self.alarm_thread: Optional[threading.Thread] = None
        
        # Determine audio player based on platform
        if platform.system() == 'Windows':
            self.mp3_player = AUDIO.get('mp3_player_windows', 'powershell')
        else:
            self.mp3_player = AUDIO.get('mp3_player_linux', 'mpg123')
        
        if self.is_enabled:
            logger.info("🔊 Sound player initialized")
        else:
            logger.warning("🔇 Sound player disabled")
    
    def play_wav_file(self, filename: str):
        """Play a WAV file using aplay, generate if missing"""
        if not self.is_enabled:
            return
        
        filepath = os.path.join(self.wav_files_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ WAV file not found: {filename}")
            # Try to generate the missing WAV file using sound system
            try:
                from sound_system import get_sound_system
                sound_system = get_sound_system()
                if sound_system.is_enabled:
                    logger.info(f"🎵 Generating missing WAV file: {filename}")
                    sound_system._generate_single_wav_file(filename)
                    # Check if file was created
                    if os.path.exists(filepath):
                        logger.info(f"✅ WAV file generated: {filename}")
                    else:
                        logger.warning(f"⚠️ Failed to generate WAV file: {filename}")
                        return
                else:
                    logger.warning(f"⚠️ Cannot generate {filename} - sound system not available")
                    return
            except Exception as e:
                logger.error(f"❌ Error generating WAV file {filename}: {e}")
                return
        
        try:
            # Stop any current audio
            self.stop_audio()
            
            # Use aplay for WAV files
            cmd = ['aplay', filepath]
            self.current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.debug(f"🔊 Playing WAV: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error playing WAV {filename}: {e}")
    
    def play_mp3_file(self, filepath: str):
        """Play an MP3 file using mpg123 or PowerShell"""
        if not self.is_enabled or not os.path.exists(filepath):
            return
        
        try:
            # Stop any current audio
            self.stop_audio()
            
            if platform.system() == 'Windows' and self.mp3_player == 'powershell':
                # Use PowerShell for Windows
                cmd = ['powershell', '-c', f'(New-Object Media.SoundPlayer "{filepath}").PlaySync()']
            else:
                # Use mpg123 for Linux/Mac
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
            self.alarm_active = False
            self.stop_audio()
            if self.alarm_thread and self.alarm_thread.is_alive():
                # Let the alarm thread finish naturally
                pass
            logger.info("🔇 Alarm stopped")
    
    def play_alarm(self, sound_system=None):
        """Play extended alarm sound"""
        if self.alarm_active:
            logger.info("🚨 Alarm already playing - skipping")
            return
        
        if not self.is_enabled:
            return
        
        try:
            self.alarm_active = True
            
            # Start alarm in background thread
            self.alarm_thread = threading.Thread(target=self._play_extended_alarm, daemon=True)
            self.alarm_thread.start()
            
            # Notify sound system if provided
            if sound_system:
                sound_system.start_alarm()
            
            logger.info("🚨 Extended alarm started")
            
        except Exception as e:
            logger.error(f"❌ Error starting alarm: {e}")
            self.alarm_active = False
    
    def _play_extended_alarm(self, sound_system=None):
        """Play extended alarm sound for configured duration"""
        try:
            start_time = time.time()
            end_time = start_time + (self.alarm_duration_minutes * 60)
            
            while self.alarm_active and time.time() < end_time:
                # Play alarm sound
                self.play_mp3_file(self.alarm_sound_path)
                
                # Wait for loop interval
                time.sleep(self.alarm_loop_interval)
            
            # Alarm finished
            self.alarm_active = False
            if sound_system:
                sound_system.stop_alarm()
            
            logger.info("🔇 Extended alarm completed")
            
        except Exception as e:
            logger.error(f"❌ Error in extended alarm: {e}")
            self.alarm_active = False
    
    def play_verification_beep(self):
        """Play verification beep sound"""
        if not self.is_enabled:
            return
        
        try:
            self.play_mp3_file(self.verification_beep_path)
            logger.debug("🔊 Played verification beep")
        except Exception as e:
            logger.error(f"❌ Error playing verification beep: {e}")
    
    # WAV file playback methods
    def play_person_detected(self):
        """Play person detected message"""
        self.play_wav_file("person_detected.wav")
    
    def play_verification_request(self):
        """Play face verification request"""
        self.play_wav_file("face_verification_request.wav")
    
    def play_verification_reminder(self, count: int):
        """Play verification reminder with progressive urgency"""
        if count <= 2:
            self.play_wav_file("face_verification_reminder_1.wav")
        elif count <= 4:
            self.play_wav_file("face_verification_reminder_2.wav")
        else:
            self.play_wav_file("face_verification_reminder_3.wav")
    
    def play_verification_timeout(self):
        """Play verification timeout message"""
        self.play_wav_file("verification_timeout.wav")
    
    def play_unknown_person_alert(self):
        """Play unknown person alert"""
        self.play_wav_file("unknown_person_alert.wav")
    
    def play_security_breach(self):
        """Play security breach alert"""
        self.play_wav_file("security_breach.wav")
    
    def play_known_person_greeting(self, name: str):
        """Play known person greeting (uses template)"""
        # For now, play generic greeting - could be enhanced to use name-specific files
        self.play_wav_file("known_person_greeting.wav")
    
    def play_time_based_greeting(self):
        """Play time-based greeting"""
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            self.play_wav_file("time_based_greeting_morning.wav")
        elif 12 <= current_hour < 17:
            self.play_wav_file("time_based_greeting_afternoon.wav")
        else:
            self.play_wav_file("time_based_greeting_evening.wav")
    
    def play_welcome_back(self, name: str):
        """Play welcome back message (uses template)"""
        # For now, play generic welcome back - could be enhanced to use name-specific files
        self.play_wav_file("welcome_back.wav")
    
    def play_guest_mode_activated(self, host_name: str):
        """Play guest mode activation message"""
        self.play_wav_file("guest_mode_activated.wav")
    
    def play_guest_mode_expired(self):
        """Play guest mode expiration message"""
        self.play_wav_file("guest_mode_expired.wav")


# Global sound player instance
_sound_player = None

def get_sound_player() -> SoundPlayer:
    """Get global sound player instance"""
    global _sound_player
    if _sound_player is None:
        _sound_player = SoundPlayer()
    return _sound_player

def play_alarm(sound_system=None):
    """Quick play alarm function"""
    get_sound_player().play_alarm(sound_system)

def play_verification_beep():
    """Quick play verification beep function"""
    get_sound_player().play_verification_beep()

def stop_alarm():
    """Stop alarm function"""
    get_sound_player().stop_alarm()

def is_alarm_playing():
    """Check if alarm is playing function"""
    return get_sound_player().is_alarm_playing()


if __name__ == "__main__":
    # Test the sound player
    logging.basicConfig(level=logging.INFO)
    
    print("🔊 Testing Sound Player")
    print("=" * 30)
    
    # Initialize sound player
    player = get_sound_player()
    
    if player.is_enabled:
        print("✅ Sound player is enabled")
        
        # Test WAV file playback
        print("\n1. Testing WAV file playback...")
        player.play_person_detected()
        time.sleep(3)
        
        player.play_verification_request()
        time.sleep(3)
        
        player.play_unknown_person_alert()
        time.sleep(3)
        
        # Test MP3 playback
        print("\n2. Testing MP3 file playback...")
        player.play_verification_beep()
        time.sleep(2)
        
        print("\n✅ Sound player test completed!")
    else:
        print("❌ Sound player is disabled")