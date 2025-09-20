#!/usr/bin/env python3
"""
Test script for Raspberry Pi camera integration
Run this to verify picamera2 is working correctly
"""

import time
import cv2
import numpy as np
from picamera2 import Picamera2, Preview

def test_pi_camera():
    """Test basic Pi camera functionality"""
    print("🔍 Testing Raspberry Pi Camera...")
    
    try:
        # Initialize camera
        picam2 = Picamera2()
        
        # Configure preview
        preview_config = picam2.create_preview_configuration(
            main={"size": (1920, 1080)},  # High-FPS mode
            lores={"size": (640, 360)},   # Low-res display stream
            display="lores"
        )
        picam2.configure(preview_config)
        picam2.set_controls({"FrameRate": 60})  # Target 60 fps
        
        # Start preview
        picam2.start_preview(Preview.QTGL)
        picam2.start()
        
        # Set autofocus mode
        picam2.set_controls({"AfMode": 1})  # Normal AF
        
        print("✅ Pi Camera initialized successfully!")
        print("📷 Camera is running with autofocus...")
        print("Press Ctrl+C to stop")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Trigger autofocus every second
                if frame_count % 30 == 0:  # Every 30 frames (roughly 1 second at 30fps)
                    picam2.set_controls({"AfTrigger": 1})
                
                # Capture frame
                frame = picam2.capture_array()
                
                # Convert from RGB to BGR for OpenCV compatibility
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Display frame
                cv2.imshow('Pi Camera Test', frame)
                
                frame_count += 1
                
                # Calculate and display FPS every 30 frames
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - start_time)
                    start_time = current_time
                    print(f"📊 FPS: {fps:.1f}")
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping camera...")
        
        # Cleanup
        cv2.destroyAllWindows()
        picam2.stop_preview()
        picam2.close()
        
        print("✅ Pi Camera test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing Pi camera: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_pi_camera()
