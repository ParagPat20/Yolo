#!/usr/bin/env python3
"""
Simple ONNX Model Test
Quick test to isolate detection issues
"""

import cv2
import numpy as np

def test_onnx_detection(model_path):
    """Simple test of ONNX model detection"""
    print(f"Testing ONNX model: {model_path}")
    
    # Load model
    net = cv2.dnn.readNetFromONNX(model_path)
    
    # Capture from webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("📷 Camera opened. Press 'q' to quit, 's' to save frame for debugging")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        height, width = frame.shape[:2]
        
        # Create blob
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        
        # Run detection
        net.setInput(blob)
        outputs = net.forward()
        
        print(f"\nFrame {frame_count}:")
        print(f"  Frame size: {width}x{height}")
        print(f"  Blob shape: {blob.shape}")
        
        for i, output in enumerate(outputs):
            print(f"  Output {i}: {output.shape}")
            
            # Process output
            if len(output.shape) == 3:
                output = output[0].T  # (1, 7, 8400) -> (8400, 7)
            
            print(f"  Processed shape: {output.shape}")
            
            # Look at first few detections
            detections_found = 0
            for j, detection in enumerate(output[:100]):  # Check first 100 detections
                if len(detection) >= 7:  # x, y, w, h + 3 classes
                    center_x, center_y, w, h = detection[:4]
                    class_scores = detection[4:7]  # 3 classes
                    
                    max_score = np.max(class_scores)
                    if max_score > 0.3:  # Lower threshold for testing
                        detections_found += 1
                        if detections_found <= 3:  # Show first 3
                            class_id = np.argmax(class_scores)
                            print(f"    Detection {detections_found}: center=({center_x:.3f}, {center_y:.3f}), size=({w:.3f}, {h:.3f}), class={class_id}, conf={max_score:.3f}")
                            
                            # Convert to pixel coordinates
                            x_px = int((center_x - w/2) * width)
                            y_px = int((center_y - h/2) * height)
                            w_px = int(w * width)
                            h_px = int(h * height)
                            
                            print(f"      Pixel coords: x={x_px}, y={y_px}, w={w_px}, h={h_px}")
                            
                            # Draw on frame for visualization
                            if (x_px >= 0 and y_px >= 0 and w_px > 0 and h_px > 0 and 
                                x_px + w_px <= width and y_px + h_px <= height):
                                cv2.rectangle(frame, (x_px, y_px), (x_px + w_px, y_px + h_px), (0, 255, 0), 2)
                                cv2.putText(frame, f"Det{detections_found}", (x_px, y_px-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            print(f"  Total detections above 0.3 confidence: {detections_found}")
        
        # Show frame
        cv2.imshow('Simple Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'debug_frame_{frame_count}.jpg', frame)
            print(f"Saved debug_frame_{frame_count}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python simple_test.py <model.onnx>")
        sys.exit(1)
    
    test_onnx_detection(sys.argv[1])
