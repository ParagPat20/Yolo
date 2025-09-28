"""
Sound Player Utility for MP3 files
Handles cross-platform MP3 playback for alarm and verification sounds
"""
import os
import subprocess
import platform
import logging
import threading
import time
from settings.settings import AUDIO

logger = logging.getLogger(__name__)

class SoundPlayer:
    """Cross-platform MP3 sound player"""
    
    def __init__(self):
        self.system = platform.system()
        self.mp3_available = self._check_mp3_support()
        self.alarm_active = False
        self.alarm_thread = None
        
        if self.mp3_available:
            logger.info(f"🔊 MP3 sound player initialized for {self.system}")
        else:
            logger.warning("🔊 MP3 sound player not available - using fallback")
    
    def _check_mp3_support(self) -> bool:
        """Check if MP3 playback is available"""
        if not AUDIO.get('use_mp3_sounds', True):
            return False
            
        if self.system == 'Linux':
            # Check if mpg123 is available
            try:
                result = subprocess.run(['which', 'mpg123'], 
                                      capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            except:
                return False
        elif self.system == 'Windows':
            # Windows can use PowerShell for basic audio
            return True
        else:
            return False
    
    def play_alarm(self):
        """Play alarm sound for unknown person detection"""
        if not self.mp3_available:
            logger.info("🚨 ALARM: Unknown person detected!")
            return
            
        alarm_path = AUDIO.get('alarm_sound_path', 'sounds/alarm.mp3')
        if not os.path.exists(alarm_path):
            logger.warning(f"Alarm sound file not found: {alarm_path}")
            logger.info("🚨 ALARM: Unknown person detected!")
            return
        
        # Stop any existing alarm
        self.stop_alarm()
        
        # Start extended alarm
        self.alarm_active = True
        self.alarm_thread = threading.Thread(target=self._play_extended_alarm, 
                                            args=(alarm_path,), daemon=True)
        self.alarm_thread.start()
        logger.info("🚨 Extended alarm started")
    
    def _play_extended_alarm(self, alarm_path: str):
        """Play alarm sound for extended duration"""
        duration_minutes = AUDIO.get('alarm_duration_minutes', 2)
        loop_interval = AUDIO.get('alarm_loop_interval', 5)
        end_time = time.time() + (duration_minutes * 60)
        
        logger.info(f"🚨 Playing alarm for {duration_minutes} minutes (every {loop_interval}s)")
        
        while self.alarm_active and time.time() < end_time:
            try:
                if self.system == 'Linux':
                    # Use mpg123 on Linux
                    subprocess.run(['mpg123', '-q', alarm_path], 
                                 capture_output=True, timeout=10)
                elif self.system == 'Windows':
                    # Use PowerShell on Windows
                    cmd = f'(New-Object Media.SoundPlayer "{alarm_path}").PlaySync()'
                    subprocess.run(['powershell', '-c', cmd], 
                                 capture_output=True, timeout=10)
                
                logger.info("🔊 Alarm sound played")
                
                # Wait before next loop
                if self.alarm_active and time.time() < end_time:
                    time.sleep(loop_interval)
                    
            except subprocess.TimeoutExpired:
                logger.warning("Alarm sound playback timed out")
            except Exception as e:
                logger.error(f"Error playing alarm sound: {e}")
                break
        
        self.alarm_active = False
        logger.info("🚨 Extended alarm completed")
    
    def stop_alarm(self):
        """Stop extended alarm playback"""
        if self.alarm_active:
            self.alarm_active = False
            logger.info("🔇 Alarm stopped")
    
    def is_alarm_active(self):
        """Check if alarm is currently playing"""
        return self.alarm_active
    
    def play_verification_beep(self):
        """Play verification beep for face verification requests"""
        if not self.mp3_available:
            logger.info("🔍 BEEP: Please show your face for verification")
            return
            
        beep_path = AUDIO.get('verification_beep_path', 'sounds/verification_beep.mp3')
        if not os.path.exists(beep_path):
            logger.warning(f"Verification beep file not found: {beep_path}")
            logger.info("🔍 BEEP: Please show your face for verification")
            return
        
        def play_sound():
            try:
                if self.system == 'Linux':
                    # Use mpg123 on Linux
                    subprocess.run(['mpg123', '-q', beep_path], 
                                 capture_output=True, timeout=5)
                    logger.info("🔊 Played verification beep")
                elif self.system == 'Windows':
                    # Use PowerShell on Windows
                    cmd = f'(New-Object Media.SoundPlayer "{beep_path}").PlaySync()'
                    subprocess.run(['powershell', '-c', cmd], 
                                 capture_output=True, timeout=5)
                    logger.info("🔊 Played verification beep")
            except subprocess.TimeoutExpired:
                logger.warning("Verification beep playback timed out")
            except Exception as e:
                logger.error(f"Error playing verification beep: {e}")
                logger.info("🔍 BEEP: Please show your face for verification")
        
        # Play in background thread
        sound_thread = threading.Thread(target=play_sound, daemon=True)
        sound_thread.start()

# Global sound player instance
_sound_player = None

def get_sound_player():
    """Get global sound player instance"""
    global _sound_player
    if _sound_player is None:
        _sound_player = SoundPlayer()
    return _sound_player

def play_alarm():
    """Play alarm sound"""
    get_sound_player().play_alarm()

def stop_alarm():
    """Stop alarm sound"""
    get_sound_player().stop_alarm()

def is_alarm_active():
    """Check if alarm is currently playing"""
    return get_sound_player().is_alarm_active()

def play_verification_beep():
    """Play verification beep"""
    get_sound_player().play_verification_beep()
