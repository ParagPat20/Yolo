#!/usr/bin/env python3
"""
YOLOv8 to ONNX Converter
Author: AI Assistant
Description: Convert trained YOLOv8 model to ONNX format for OpenCV DNN usage
"""

import os
import sys
from ultralytics import YOLO
import argparse

def convert_yolo_to_onnx(model_path, output_path=None, imgsz=640, simplify=True):
    """
    Convert YOLOv8 model to ONNX format
    
    Args:
        model_path (str): Path to YOLOv8 model (.pt file)
        output_path (str): Output ONNX file path (optional)
        imgsz (int): Input image size
        simplify (bool): Simplify ONNX model
    
    Returns:
        str: Path to exported ONNX model
    """
    try:
        # Load YOLOv8 model
        print(f"Loading YOLOv8 model: {model_path}")
        model = YOLO(model_path)
        
        # Export to ONNX
        print(f"Converting to ONNX format...")
        print(f"  - Input size: {imgsz}x{imgsz}")
        print(f"  - Simplify: {simplify}")
        print(f"  - Model classes: {len(model.names)} - {list(model.names.values())}")
        
        onnx_path = model.export(
            format='onnx',
            imgsz=imgsz,
            simplify=simplify,
            dynamic=False,
            opset=11,
            verbose=True
        )
        
        # Move to custom output path if specified
        if output_path and output_path != onnx_path:
            import shutil
            shutil.move(onnx_path, output_path)
            onnx_path = output_path
        
        print(f"✓ ONNX model exported: {onnx_path}")
        
        # Verify the exported model
        verify_onnx_model(onnx_path)
        
        return onnx_path
        
    except Exception as e:
        raise Exception(f"Error converting model: {str(e)}")

def verify_onnx_model(onnx_path):
    """Verify the exported ONNX model"""
    try:
        import onnx
        
        # Load and check ONNX model
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        print(f"✓ ONNX model verification successful")
        
        # Print model info
        print(f"  - Input shape: {model.graph.input[0].type.tensor_type.shape}")
        print(f"  - Output count: {len(model.graph.output)}")
        
        # Get file size
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  - Model size: {size_mb:.1f} MB")
        
    except ImportError:
        print("⚠️  Warning: onnx package not installed, skipping verification")
        print("   Install with: pip install onnx")
    except Exception as e:
        print(f"⚠️  Warning: ONNX verification failed: {str(e)}")

def create_class_names_file(model_path, output_path="class_names.txt"):
    """Create a class names file from YOLOv8 model"""
    try:
        model = YOLO(model_path)
        
        if hasattr(model.model, 'names'):
            class_names = list(model.model.names.values())
            
            with open(output_path, 'w') as f:
                for name in class_names:
                    f.write(f"{name}\n")
            
            print(f"✓ Class names saved: {output_path}")
            print(f"  Classes: {class_names}")
            
            return class_names
        else:
            print("⚠️  Warning: Could not extract class names from model")
            return None
            
    except Exception as e:
        print(f"⚠️  Warning: Error creating class names file: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Convert YOLOv8 to ONNX format')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLOv8 model (.pt file)')
    parser.add_argument('--output', type=str,
                       help='Output ONNX file path (optional)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Do not simplify ONNX model')
    parser.add_argument('--create-names', action='store_true',
                       help='Create class names text file')
    
    args = parser.parse_args()
    
    try:
        # Check if model file exists
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found: {args.model}")
        
        print(f"{'='*60}")
        print(f"YOLOv8 TO ONNX CONVERTER")
        print(f"{'='*60}")
        
        # Convert to ONNX
        onnx_path = convert_yolo_to_onnx(
            model_path=args.model,
            output_path=args.output,
            imgsz=args.imgsz,
            simplify=not args.no_simplify
        )
        
        # Create class names file if requested
        if args.create_names:
            create_class_names_file(args.model)
        
        print(f"\n{'='*60}")
        print(f"CONVERSION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"ONNX model: {onnx_path}")
        print(f"\nUsage with detector_tracker.py:")
        print(f"python detector_tracker.py --model {onnx_path} --source 0")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
