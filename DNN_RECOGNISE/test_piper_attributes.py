#!/usr/bin/env python3
"""
Test script to inspect AudioChunk attributes
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_audio_chunk_attributes():
    """Test what attributes AudioChunk has"""
    try:
        from sound_system import get_sound_system
        
        print("🔍 Testing AudioChunk attributes...")
        
        sound_system = get_sound_system()
        if not sound_system or not sound_system.piper_voice:
            print("❌ Piper voice not available")
            return
        
        # Generate a small audio sample
        text = "Hello"
        audio_generator = sound_system.piper_voice.synthesize(text)
        
        # Get the first chunk to inspect
        first_chunk = next(audio_generator)
        
        print(f"AudioChunk type: {type(first_chunk)}")
        print(f"AudioChunk attributes: {dir(first_chunk)}")
        
        # Try to access common attributes
        for attr in ['data', 'bytes', 'content', 'audio', 'samples', 'raw']:
            if hasattr(first_chunk, attr):
                value = getattr(first_chunk, attr)
                print(f"AudioChunk.{attr}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'no length'}")
        
        print("✅ AudioChunk inspection completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_chunk_attributes()
