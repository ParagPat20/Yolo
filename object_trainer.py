#!/usr/bin/env python3
"""
YOLOv8 Object Detection Training Script
Author: AI Assistant
Description: Train YOLOv8 model for vehicle detection (mobil, motor, truck)
"""

import os
import sys
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import argparse
from datetime import datetime

class YOLOv8Trainer:
    def __init__(self, data_yaml_path="data.yaml", model_size="n"):
        """
        Initialize YOLOv8 trainer
        
        Args:
            data_yaml_path (str): Path to data.yaml file
            model_size (str): Model size - 'n', 's', 'm', 'l', 'x'
        """
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        self.model = None
        self.results_dir = "runs/detect"
        
        # Validate data.yaml exists
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"Data configuration file not found: {data_yaml_path}")
        
        # Load and validate data configuration
        self.load_data_config()
        
    def load_data_config(self):
        """Load and validate data configuration from YAML file"""
        try:
            with open(self.data_yaml_path, 'r') as f:
                self.data_config = yaml.safe_load(f)
            
            # Validate required keys
            required_keys = ['train', 'val', 'nc', 'names']
            for key in required_keys:
                if key not in self.data_config:
                    raise ValueError(f"Missing required key '{key}' in {self.data_yaml_path}")
            
            print(f"✓ Data configuration loaded successfully")
            print(f"  - Classes: {self.data_config['nc']}")
            print(f"  - Class names: {self.data_config['names']}")
            print(f"  - Training data: {self.data_config['train']}")
            print(f"  - Validation data: {self.data_config['val']}")
            
        except Exception as e:
            raise Exception(f"Error loading data configuration: {str(e)}")
    
    def initialize_model(self):
        """Initialize YOLOv8 model"""
        try:
            # Model size mapping
            model_files = {
                'n': 'yolov8n.pt',  # Nano
                's': 'yolov8s.pt',  # Small
                'm': 'yolov8m.pt',  # Medium
                'l': 'yolov8l.pt',  # Large
                'x': 'yolov8x.pt'   # Extra Large
            }
            
            if self.model_size not in model_files:
                raise ValueError(f"Invalid model size. Choose from: {list(model_files.keys())}")
            
            model_file = model_files[self.model_size]
            print(f"Initializing YOLOv8{self.model_size} model...")
            
            # Load pretrained model
            self.model = YOLO(model_file)
            print(f"✓ Model loaded: {model_file}")
            
        except Exception as e:
            raise Exception(f"Error initializing model: {str(e)}")
    
    def train(self, epochs=100, imgsz=640, batch_size=16, patience=50, save_period=10, device=None):
        """
        Train the YOLOv8 model
        
        Args:
            epochs (int): Number of training epochs
            imgsz (int): Input image size
            batch_size (int): Batch size
            patience (int): Early stopping patience
            save_period (int): Save model every N epochs
            device (str): Device to use ('auto', 'cpu', '0', '1', etc.)
        """
        if self.model is None:
            self.initialize_model()
        
        try:
            # Determine device to use
            if device is None:
                # Auto-detect best device
                if torch.cuda.is_available():
                    training_device = '0'  # Use first GPU
                    device_name = torch.cuda.get_device_name(0)
                    device_info = f"GPU ({device_name})"
                else:
                    training_device = 'cpu'
                    device_info = "CPU"
            else:
                training_device = device
                if device == 'cpu':
                    device_info = "CPU (forced)"
                elif device.isdigit():
                    if torch.cuda.is_available() and int(device) < torch.cuda.device_count():
                        device_name = torch.cuda.get_device_name(int(device))
                        device_info = f"GPU {device} ({device_name})"
                    else:
                        print(f"⚠️  GPU {device} not available, falling back to CPU")
                        training_device = 'cpu'
                        device_info = "CPU (fallback)"
                else:
                    device_info = f"{device}"
            
            print(f"\n{'='*60}")
            print(f"STARTING YOLOV8 TRAINING")
            print(f"{'='*60}")
            print(f"Model: YOLOv8{self.model_size}")
            print(f"Dataset: {self.data_yaml_path}")
            print(f"Epochs: {epochs}")
            print(f"Image size: {imgsz}")
            print(f"Batch size: {batch_size}")
            print(f"Device: {device_info}")
            
            # Show GPU memory info if using GPU
            if training_device != 'cpu' and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(int(training_device)).total_memory / 1024**3
                print(f"GPU Memory: {gpu_memory:.1f} GB")
            
            print(f"{'='*60}\n")
            
            # Start training
            results = self.model.train(
                data=self.data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                patience=patience,
                save_period=save_period,
                device=training_device,
                verbose=True,
                plots=True,
                save=True
            )
            
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Results saved in: {results.save_dir}")
            print(f"Best model: {results.save_dir}/weights/best.pt")
            print(f"Last model: {results.save_dir}/weights/last.pt")
            
            return results
            
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def validate(self, model_path=None):
        """
        Validate the trained model
        
        Args:
            model_path (str): Path to model weights (optional)
        """
        try:
            if model_path:
                model = YOLO(model_path)
            else:
                model = self.model
                
            if model is None:
                raise ValueError("No model available for validation")
            
            print("Running validation...")
            results = model.val(data=self.data_yaml_path)
            
            print(f"\n{'='*40}")
            print(f"VALIDATION RESULTS")
            print(f"{'='*40}")
            print(f"mAP50: {results.box.map50:.4f}")
            print(f"mAP50-95: {results.box.map:.4f}")
            print(f"{'='*40}")
            
            return results
            
        except Exception as e:
            raise Exception(f"Validation failed: {str(e)}")
    
    def export_model(self, model_path, format='onnx'):
        """
        Export trained model to different formats
        
        Args:
            model_path (str): Path to trained model
            format (str): Export format ('onnx', 'tensorrt', 'coreml', etc.)
        """
        try:
            model = YOLO(model_path)
            exported_model = model.export(format=format)
            print(f"✓ Model exported to {format} format: {exported_model}")
            return exported_model
        except Exception as e:
            raise Exception(f"Export failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Vehicle Detection Training')
    parser.add_argument('--data', type=str, default='data.yaml', 
                       help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='Input image size')
    parser.add_argument('--batch', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=50, 
                       help='Early stopping patience')
    parser.add_argument('--validate', action='store_true', 
                       help='Run validation after training')
    parser.add_argument('--export', type=str, choices=['onnx', 'tensorrt', 'coreml'], 
                       help='Export format after training')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for training (auto, cpu, 0, 1, etc.)')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = YOLOv8Trainer(
            data_yaml_path=args.data,
            model_size=args.model
        )
        
        # Train model
        results = trainer.train(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch,
            patience=args.patience,
            device=args.device
        )
        
        # Get best model path
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        
        # Run validation if requested
        if args.validate:
            trainer.validate(str(best_model_path))
        
        # Export model if requested
        if args.export:
            trainer.export_model(str(best_model_path), args.export)
        
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"📁 Results directory: {results.save_dir}")
        print(f"🏆 Best model: {best_model_path}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
