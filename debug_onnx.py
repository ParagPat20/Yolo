#!/usr/bin/env python3
"""
Debug ONNX Model Script
Author: AI Assistant
Description: Debug ONNX model to understand its structure and output format
"""

import cv2
import numpy as np
import argparse

def debug_onnx_model(model_path):
    """Debug ONNX model structure and output"""
    try:
        print(f"Loading ONNX model: {model_path}")
        net = cv2.dnn.readNetFromONNX(model_path)
        
        # Get layer information
        layer_names = net.getLayerNames()
        output_layers = net.getUnconnectedOutLayersNames()
        
        print(f"\n📊 Model Structure:")
        print(f"  Total layers: {len(layer_names)}")
        print(f"  Output layers: {output_layers}")
        
        # Test with dummy input
        print(f"\n🧪 Testing with dummy input...")
        dummy_frame = np.random.rand(640, 640, 3).astype(np.float32)
        blob = cv2.dnn.blobFromImage(dummy_frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        
        print(f"  Input blob shape: {blob.shape}")
        
        # Run inference
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        print(f"\n📤 Output Analysis:")
        for i, output in enumerate(outputs):
            print(f"  Output {i}:")
            print(f"    Shape: {output.shape}")
            print(f"    Data type: {output.dtype}")
            print(f"    Min value: {output.min():.6f}")
            print(f"    Max value: {output.max():.6f}")
            print(f"    Mean value: {output.mean():.6f}")
            
            # Analyze output structure
            if len(output.shape) >= 2:
                last_dim = output.shape[-1]
                print(f"    Last dimension: {last_dim}")
                
                # Common YOLO formats
                if last_dim == 85:  # COCO (80 classes + 5)
                    print(f"    → Likely COCO format (80 classes)")
                elif last_dim == 8:  # 3 classes + 5
                    print(f"    → Likely custom format (3 classes)")
                elif last_dim >= 5:
                    classes = last_dim - 5
                    print(f"    → Likely {classes} classes + bbox + conf")
                else:
                    print(f"    → Unknown format")
                
                # Show first few detections for analysis
                if len(output.shape) == 3:
                    flat_output = output[0]  # Remove batch dimension
                else:
                    flat_output = output
                    
                if flat_output.shape[0] > flat_output.shape[1]:
                    flat_output = flat_output.T  # Transpose if needed
                
                print(f"    Processed shape: {flat_output.shape}")
                if flat_output.shape[0] > 0:
                    print(f"    First detection: {flat_output[0][:min(10, flat_output.shape[1])]}")
                    print(f"    Detection format appears to be: {flat_output.shape[1]} values per detection")
        
        # Try to understand the output format
        print(f"\n🔍 Format Analysis:")
        if len(outputs) > 0:
            main_output = outputs[0]
            if len(main_output.shape) == 3:
                main_output = main_output[0]
            
            if main_output.shape[0] > main_output.shape[1]:
                main_output = main_output.T
            
            if main_output.shape[1] >= 7:  # At least x,y,w,h + 3 classes
                print(f"  ✓ Output format: [x, y, w, h, class1_score, class2_score, class3_score, ...]")
                print(f"  ✓ Number of classes: {main_output.shape[1] - 4}")
            elif main_output.shape[1] >= 8:  # x,y,w,h,conf + classes
                print(f"  ✓ Output format: [x, y, w, h, confidence, class1_score, class2_score, ...]")
                print(f"  ✓ Number of classes: {main_output.shape[1] - 5}")
            else:
                print(f"  ⚠️  Unexpected output format")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Debug ONNX model structure')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to ONNX model')
    
    args = parser.parse_args()
    
    print("🔧 ONNX Model Debugger")
    print("=" * 50)
    
    success = debug_onnx_model(args.model)
    
    if success:
        print("\n✅ Debug completed successfully!")
    else:
        print("\n❌ Debug failed!")

if __name__ == "__main__":
    main()
