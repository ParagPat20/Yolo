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
import threading
import time
from datetime import datetime
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        self.wav_files_dir = SOUND_SYSTEM.get('wav_files_dir', 'sounds/wav')
        self.piper_config = SOUND_SYSTEM.get('piper', {})

        # Threading configuration
        self.max_workers = 4  # Maximum concurrent WAV file generations
        self.generation_timeout = 60  # Timeout per file generation (seconds)

        # Create WAV files directory
        os.makedirs(self.wav_files_dir, exist_ok=True)

        logger.info(f"🔊 WAV File Updater initialized")
        logger.info(f"📁 WAV files directory: {self.wav_files_dir}")
        logger.info(f"🌐 Language: {self.language}")
        logger.info(f"⚡ Max concurrent workers: {self.max_workers}")
    
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
                'time_based_greeting': {
                    'morning': "Good morning!",
                    'afternoon': "Good afternoon!",
                    'evening': "Good evening!"
                },
                'welcome_back': "Welcome back {name}!",
                'guest_mode_activated': "Guest mode activated. {host_name} has a guest.",
                'guest_mode_expired': "Guest mode expired. Reverting to normal security protocols."
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.generation_timeout)

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

    def _generate_single_file_task(self, message_key: str, variant: str, text: str, filepath: str) -> Tuple[str, bool]:
        """Generate a single WAV file (used by threading)"""
        filename = os.path.basename(filepath)
        try:
            logger.info(f"🎵 [Thread] Starting generation: {filename}")
            success = self.generate_wav_file(text, filepath)
            if success:
                logger.info(f"✅ [Thread] Completed: {filename}")
            else:
                logger.error(f"❌ [Thread] Failed: {filename}")
            return filename, success
        except Exception as e:
            logger.error(f"❌ [Thread] Exception for {filename}: {e}")
            return filename, False
    
    def update_all_wav_files(self, force_update: bool = False) -> Tuple[int, int]:
        """Update all WAV files from settings using threading for concurrent generation"""
        logger.info("🎵 Updating WAV files from settings with threading...")
        start_time = time.time()

        if not self.check_piper_availability():
            logger.error("❌ Piper TTS not available - cannot generate WAV files")
            return 0, 0

        messages = self.get_all_messages()
        file_tasks = []

        # Collect all file generation tasks
        for message_key, message_data in messages.items():
            if isinstance(message_data, dict):
                # Handle messages with multiple variants (like greetings)
                for variant_key, text in message_data.items():
                    filename = f"{message_key}_{variant_key}.wav"
                    filepath = os.path.join(self.wav_files_dir, filename)

                    if not os.path.exists(filepath) or force_update:
                        file_tasks.append((message_key, variant_key, text, filepath))
                    else:
                        logger.info(f"⏭️ Skipping existing file: {filename}")

            elif isinstance(message_data, list):
                # Handle messages with multiple variants (like verification reminders)
                for i, text in enumerate(message_data):
                    filename = f"{message_key}_{i+1}.wav"
                    filepath = os.path.join(self.wav_files_dir, filename)

                    if not os.path.exists(filepath) or force_update:
                        file_tasks.append((message_key, str(i+1), text, filepath))
                    else:
                        logger.info(f"⏭️ Skipping existing file: {filename}")
            else:
                # Handle simple text messages
                filename = f"{message_key}.wav"
                filepath = os.path.join(self.wav_files_dir, filename)

                if not os.path.exists(filepath) or force_update:
                    file_tasks.append((message_key, "", message_data, filepath))
                else:
                    logger.info(f"⏭️ Skipping existing file: {filename}")

        total_tasks = len(file_tasks)
        if total_tasks == 0:
            logger.info("✅ All WAV files are up to date")
            return 0, len(messages)

        logger.info(f"⚡ Processing {total_tasks} files with {self.max_workers} concurrent workers")

        # Generate files using threading
        generated_count = 0
        failed_files = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._generate_single_file_task, task[0], task[1], task[2], task[3]): task
                for task in file_tasks
            }

            # Process completed tasks
            for future in as_completed(future_to_task, timeout=self.generation_timeout * total_tasks):
                try:
                    filename, success = future.result(timeout=10)
                    if success:
                        generated_count += 1
                    else:
                        failed_files.append(filename)
                except Exception as e:
                    task = future_to_task[future]
                    filename = os.path.basename(task[3])
                    logger.error(f"❌ Task failed for {filename}: {e}")
                    failed_files.append(filename)

        end_time = time.time()
        duration = end_time - start_time

        # Report results
        logger.info(f"✅ Generated: {generated_count}, Failed: {len(failed_files)}, Duration: {duration:.2f}s")

        if failed_files:
            logger.warning(f"⚠️ Failed to generate: {', '.join(failed_files[:5])}{'...' if len(failed_files) > 5 else ''}")

        return generated_count, total_tasks - generated_count
    
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

    def get_threading_info(self) -> Dict:
        """Get threading configuration information"""
        return {
            'max_workers': self.max_workers,
            'generation_timeout': self.generation_timeout,
            'threading_enabled': True
        }


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
    parser.add_argument('--workers', type=int, metavar='N',
                       help='Number of concurrent workers (default: 4)')
    parser.add_argument('--timeout', type=int, metavar='SECONDS',
                       help='Timeout per file generation in seconds (default: 60)')
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = WAVFileUpdater()

    # Set language if specified
    if args.language:
        updater.language = args.language
        logger.info(f"🌐 Language set to: {args.language}")

    # Set threading configuration if specified
    if args.workers:
        updater.max_workers = args.workers
        logger.info(f"⚡ Max workers set to: {args.workers}")

    if args.timeout:
        updater.generation_timeout = args.timeout
        logger.info(f"⏱️ Generation timeout set to: {args.timeout} seconds")
    
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
        threading_info = updater.get_threading_info()

        logger.info(f"Total files: {info['total_files']}")
        logger.info(f"Directory: {info['wav_files_dir']}")
        logger.info(f"Language: {info['language']}")
        logger.info(f"Piper available: {info['piper_available']}")
        logger.info(f"Threading: {threading_info['threading_enabled']}")
        logger.info(f"Max workers: {threading_info['max_workers']}")
        logger.info(f"Generation timeout: {threading_info['generation_timeout']}s")

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
