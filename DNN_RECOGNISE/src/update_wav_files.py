#!/usr/bin/env python3
"""
WAV File Updater for CCTV Security System
Updates WAV files from settings.py as needed for speech
"""

import sys
import os
import logging
import subprocess
import shlex
from datetime import datetime
from typing import Dict, List, Tuple

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import settings
try:
    from settings.settings import SOUND_SYSTEM, AUDIO
except ImportError:
    logger.error("❌ Could not import settings. Please ensure settings.py is available.")
    sys.exit(1)

class WAVFileUpdater:
    """Utility class to update WAV files from settings"""
    
    def __init__(self):
        """Initialize WAV file updater"""
        self.language = SOUND_SYSTEM.get('language', 'en')
        self.wav_files_dir = SOUND_SYSTEM.get('wav_files_dir', '/home/jecon/yolo/DNN_RECOGNISE/sounds/wav')
        self.piper_config = SOUND_SYSTEM.get('piper', {})
        
        # Create WAV files directory
        os.makedirs(self.wav_files_dir, exist_ok=True)
        
        logger.info(f"🔊 WAV File Updater initialized")
        logger.info(f"📁 WAV files directory: {self.wav_files_dir}")
        logger.info(f"🌐 Language: {self.language}")
    
    def check_piper_availability(self) -> bool:
        """Check if Piper TTS is available"""
        try:
            result = subprocess.run(['piper', '--help'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Check if model file exists
                model_path = self.piper_config.get('models', {}).get(self.language, 
                                                                   self.piper_config.get('model_path', ''))
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
    
    def get_all_messages(self) -> Dict:
        """Get all messages that need WAV files"""
        if self.language == 'gu':
            return {
                'welcome_to_jech_aerotech': "જેચ એરોટેકમાં આપનું સ્વાગત છે.",
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
                'welcome_to_jech_aerotech': "Welcome to Jech Aerotech.",
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
    
    def generate_wav_file(self, text: str, filepath: str) -> bool:
        """Generate a single WAV file using Piper TTS"""
        try:
            # Get Piper configuration
            model_path = self.piper_config.get('models', {}).get(self.language, 
                                                               self.piper_config.get('model_path', ''))
            noise_scale = self.piper_config.get('noise', 0.667)
            length_scale = self.piper_config.get('length_penalty', 1.0)
            
            # Escape text for shell safety
            safe_text = shlex.quote(text)
            
            # Build Piper command to generate WAV file
            cmd = [
                'bash', '-c',
                f'echo {safe_text} | piper --model {model_path} --length-scale {length_scale} --noise-scale {noise_scale} --output-file {filepath}'
            ]
            
            # Execute command
            logger.info(f"🎵 Generating WAV file: {os.path.basename(filepath)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(filepath):
                logger.info(f"✅ Generated WAV file: {os.path.basename(filepath)}")
                return True
            else:
                logger.error(f"❌ Failed to generate WAV file: {os.path.basename(filepath)}")
                logger.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout generating WAV file: {os.path.basename(filepath)}")
            return False
        except Exception as e:
            logger.error(f"❌ Error generating WAV file {os.path.basename(filepath)}: {e}")
            return False
    
    def update_all_wav_files(self, force_update: bool = False) -> Tuple[int, int]:
        """Update all WAV files from settings"""
        logger.info("🎵 Updating WAV files from settings...")
        
        if not self.check_piper_availability():
            logger.error("❌ Piper TTS not available - cannot generate WAV files")
            return 0, 0
        
        messages = self.get_all_messages()
        generated_count = 0
        skipped_count = 0
        
        for message_key, message_data in messages.items():
            if isinstance(message_data, dict):
                # Handle messages with multiple variants (like greetings)
                for variant_key, text in message_data.items():
                    filename = f"{message_key}_{variant_key}.wav"
                    filepath = os.path.join(self.wav_files_dir, filename)
                    
                    if not os.path.exists(filepath) or force_update:
                        if self.generate_wav_file(text, filepath):
                            generated_count += 1
                        else:
                            logger.warning(f"⚠️ Failed to generate: {filename}")
                    else:
                        logger.info(f"⏭️ Skipping existing file: {filename}")
                        skipped_count += 1
                        
            elif isinstance(message_data, list):
                # Handle messages with multiple variants (like verification reminders)
                for i, text in enumerate(message_data):
                    filename = f"{message_key}_{i+1}.wav"
                    filepath = os.path.join(self.wav_files_dir, filename)
                    
                    if not os.path.exists(filepath) or force_update:
                        if self.generate_wav_file(text, filepath):
                            generated_count += 1
                        else:
                            logger.warning(f"⚠️ Failed to generate: {filename}")
                    else:
                        logger.info(f"⏭️ Skipping existing file: {filename}")
                        skipped_count += 1
            else:
                # Handle simple text messages
                filename = f"{message_key}.wav"
                filepath = os.path.join(self.wav_files_dir, filename)
                
                if not os.path.exists(filepath) or force_update:
                    if self.generate_wav_file(message_data, filepath):
                        generated_count += 1
                    else:
                        logger.warning(f"⚠️ Failed to generate: {filename}")
                else:
                    logger.info(f"⏭️ Skipping existing file: {filename}")
                    skipped_count += 1
        
        return generated_count, skipped_count
    
    def update_specific_wav_file(self, message_key: str, variant: str = None) -> bool:
        """Update a specific WAV file"""
        logger.info(f"🎵 Updating specific WAV file: {message_key}")
        
        if not self.check_piper_availability():
            logger.error("❌ Piper TTS not available - cannot generate WAV files")
            return False
        
        messages = self.get_all_messages()
        
        if message_key not in messages:
            logger.error(f"❌ Message key not found: {message_key}")
            return False
        
        message_data = messages[message_key]
        
        if isinstance(message_data, dict):
            if variant and variant in message_data:
                text = message_data[variant]
                filename = f"{message_key}_{variant}.wav"
            else:
                logger.error(f"❌ Variant not found: {variant}")
                return False
        elif isinstance(message_data, list):
            if variant and variant.isdigit():
                index = int(variant) - 1
                if 0 <= index < len(message_data):
                    text = message_data[index]
                    filename = f"{message_key}_{variant}.wav"
                else:
                    logger.error(f"❌ Invalid variant index: {variant}")
                    return False
            else:
                logger.error(f"❌ Variant must be a number for list messages")
                return False
        else:
            text = message_data
            filename = f"{message_key}.wav"
        
        filepath = os.path.join(self.wav_files_dir, filename)
        return self.generate_wav_file(text, filepath)
    
    def list_wav_files(self) -> List[str]:
        """List all existing WAV files"""
        if not os.path.exists(self.wav_files_dir):
            return []
        
        wav_files = [f for f in os.listdir(self.wav_files_dir) if f.endswith('.wav')]
        return sorted(wav_files)
    
    def clean_wav_files(self) -> int:
        """Clean all WAV files"""
        if not os.path.exists(self.wav_files_dir):
            return 0
        
        wav_files = [f for f in os.listdir(self.wav_files_dir) if f.endswith('.wav')]
        
        for wav_file in wav_files:
            filepath = os.path.join(self.wav_files_dir, wav_file)
            try:
                os.remove(filepath)
                logger.info(f"🗑️ Removed: {wav_file}")
            except Exception as e:
                logger.error(f"❌ Error removing {wav_file}: {e}")
        
        return len(wav_files)
    
    def get_wav_file_info(self) -> Dict:
        """Get information about WAV files"""
        wav_files = self.list_wav_files()
        
        info = {
            'total_files': len(wav_files),
            'wav_files_dir': self.wav_files_dir,
            'language': self.language,
            'piper_available': self.check_piper_availability(),
            'files': []
        }
        
        for wav_file in wav_files:
            filepath = os.path.join(self.wav_files_dir, wav_file)
            try:
                stat = os.stat(filepath)
                info['files'].append({
                    'name': wav_file,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
            except Exception as e:
                logger.error(f"❌ Error getting info for {wav_file}: {e}")
        
        return info


def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update WAV files from settings.py')
    parser.add_argument('--update-all', action='store_true', 
                       help='Update all WAV files from settings')
    parser.add_argument('--force', action='store_true',
                       help='Force update even if files exist')
    parser.add_argument('--update', type=str, metavar='MESSAGE_KEY',
                       help='Update specific WAV file by message key')
    parser.add_argument('--variant', type=str, metavar='VARIANT',
                       help='Variant for specific message (e.g., morning, afternoon, evening)')
    parser.add_argument('--list', action='store_true',
                       help='List all existing WAV files')
    parser.add_argument('--info', action='store_true',
                       help='Show WAV files information')
    parser.add_argument('--clean', action='store_true',
                       help='Clean all WAV files')
    parser.add_argument('--language', type=str, choices=['en', 'gu'],
                       help='Set language (en/gu)')
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = WAVFileUpdater()
    
    # Set language if specified
    if args.language:
        updater.language = args.language
        logger.info(f"🌐 Language set to: {args.language}")
    
    # Handle different operations
    if args.update_all:
        logger.info("🎵 Updating all WAV files...")
        generated, skipped = updater.update_all_wav_files(force_update=args.force)
        logger.info(f"✅ Generated: {generated}, Skipped: {skipped}")
        
    elif args.update:
        logger.info(f"🎵 Updating specific WAV file: {args.update}")
        success = updater.update_specific_wav_file(args.update, args.variant)
        if success:
            logger.info("✅ WAV file updated successfully")
        else:
            logger.error("❌ Failed to update WAV file")
            
    elif args.list:
        logger.info("📄 Listing WAV files...")
        wav_files = updater.list_wav_files()
        if wav_files:
            for wav_file in wav_files:
                logger.info(f"  - {wav_file}")
        else:
            logger.info("No WAV files found")
            
    elif args.info:
        logger.info("📊 WAV files information...")
        info = updater.get_wav_file_info()
        logger.info(f"Total files: {info['total_files']}")
        logger.info(f"Directory: {info['wav_files_dir']}")
        logger.info(f"Language: {info['language']}")
        logger.info(f"Piper available: {info['piper_available']}")
        
        if info['files']:
            logger.info("Files:")
            for file_info in info['files']:
                logger.info(f"  - {file_info['name']} ({file_info['size']} bytes, {file_info['modified']})")
                
    elif args.clean:
        logger.info("🗑️ Cleaning WAV files...")
        removed_count = updater.clean_wav_files()
        logger.info(f"✅ Removed {removed_count} WAV files")
        
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()
