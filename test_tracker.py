#!/usr/bin/env python3
"""
Test script for YOLO Detector + Tracker
Author: AI Assistant
Description: Simple test script to verify the detector-tracker functionality
"""

import cv2
import os
import sys
from detector_tracker import YOLODetectorTracker

def test_tracker_with_webcam():
    """Test tracker with webcam"""
    print("Testing tracker with webcam...")
    print("Make sure you have a trained YOLO model in ONNX format")
    
    # Example model path (adjust as needed)
    model_path = "best.onnx"  # You'll need to convert your .pt model to .onnx first
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please convert your YOLOv8 model to ONNX format first:")
        print("python convert_to_onnx.py --model runs/detect/train/weights/best.pt --output best.onnx")
        return False
    
    try:
        # Initialize detector-tracker
        detector_tracker = YOLODetectorTracker(
            model_path=model_path,
            conf_threshold=0.5,
            nms_threshold=0.4,
            tracker_type='CSRT'
        )
        
        # Test with webcam
        detector_tracker.process_video(
            source=0,  # Webcam
            show_result=True,
            target_class=0  # Track 'mobil' (cars) by default
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_tracker_with_test_images():
    """Test tracker by creating a simple video from test images"""
    print("Testing tracker with test images...")
    
    # Check if test images exist
    test_dir = "test/images"
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    # Get test images
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"❌ No test images found in {test_dir}")
        return False
    
    print(f"Found {len(image_files)} test images")
    
    # Create a simple slideshow video for testing
    output_video = "test_slideshow.mp4"
    create_slideshow_video(test_dir, image_files, output_video)
    
    # Test with the created video
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please convert your YOLOv8 model to ONNX format first")
        return False
    
    try:
        detector_tracker = YOLODetectorTracker(
            model_path=model_path,
            conf_threshold=0.5,
            nms_threshold=0.4,
            tracker_type='CSRT'
        )
        
        detector_tracker.process_video(
            source=output_video,
            output_path="test_output.mp4",
            show_result=True
        )
        
        # Cleanup
        if os.path.exists(output_video):
            os.remove(output_video)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def create_slideshow_video(image_dir, image_files, output_path, fps=2):
    """Create a simple slideshow video from images"""
    if not image_files:
        return
    
    # Read first image to get dimensions
    first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
    if first_image is None:
        return
    
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating test video: {output_path}")
    
    for image_file in image_files[:10]:  # Use first 10 images
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        
        if frame is not None:
            # Resize frame if needed
            frame = cv2.resize(frame, (width, height))
            
            # Write frame multiple times to make it visible longer
            for _ in range(fps * 2):  # 2 seconds per image
                writer.write(frame)
    
    writer.release()
    print(f"✓ Test video created: {output_path}")

def main():
    print("🧪 YOLO Detector + Tracker Test Suite")
    print("=" * 50)
    
    # Check OpenCV version
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test 1: Check if required trackers are available
    print("\n1. Testing tracker availability...")
    try:
        tracker_csrt = cv2.TrackerCSRT_create()
        print("✓ CSRT tracker available")
    except:
        print("❌ CSRT tracker not available")
    
    try:
        tracker_kcf = cv2.TrackerKCF_create()
        print("✓ KCF tracker available")
    except:
        print("❌ KCF tracker not available")
    
    try:
        tracker_mosse = cv2.legacy.TrackerMOSSE_create()
        print("✓ MOSSE tracker available")
    except:
        print("❌ MOSSE tracker not available (requires opencv-contrib-python)")
    
    # Test 2: Check model availability
    print("\n2. Checking for ONNX model...")
    if os.path.exists("best.onnx"):
        print("✓ ONNX model found: best.onnx")
    else:
        print("❌ ONNX model not found")
        print("Convert your trained model first:")
        print("python convert_to_onnx.py --model runs/detect/train/weights/best.pt --output best.onnx")
        return
    
    # Test 3: Interactive test selection
    print("\n3. Select test mode:")
    print("1. Test with webcam (live)")
    print("2. Test with test images (slideshow)")
    print("3. Skip testing")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        test_tracker_with_webcam()
    elif choice == "2":
        test_tracker_with_test_images()
    else:
        print("Skipping tests")
    
    print("\n🎉 Test suite completed!")

if __name__ == "__main__":
    main()
