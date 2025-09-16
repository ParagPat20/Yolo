#!/usr/bin/env python3
"""
YOLOv8 Object Detection Script
Author: AI Assistant
Description: Detect vehicles (mobil, motor, truck) using trained YOLOv8 model
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from ultralytics import YOLO
import yaml

class YOLOv8Detector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45, use_coco=False):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path (str): Path to trained model weights
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            use_coco (bool): Use COCO class names (80 classes) instead of custom
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.use_coco = use_coco
        
        # COCO dataset class names (80 classes)
        self.coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Default to custom vehicle classes or COCO classes
        if use_coco:
            self.class_names = self.coco_names
        else:
            self.class_names = ['mobil', 'motor', 'truck']  # Default custom classes
        
        # Load model
        self.load_model()
        
        # Colors for each class (BGR format)
        if use_coco:
            # Generate colors for 80 COCO classes
            np.random.seed(42)  # For consistent colors
            self.colors = {}
            for i, name in enumerate(self.class_names):
                self.colors[name] = tuple(map(int, np.random.randint(0, 255, 3)))
        else:
            # Custom colors for vehicle classes
            self.colors = {
                'mobil': (255, 0, 0),    # Blue
                'motor': (0, 255, 0),    # Green  
                'truck': (0, 0, 255),    # Red
            }
    
    def load_model(self):
        """Load the trained YOLOv8 model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            print(f"Loading model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Get class names from model if available and not using COCO override
            if hasattr(self.model.model, 'names') and not self.use_coco:
                model_names = list(self.model.model.names.values())
                self.class_names = model_names
                print(f"✓ Using class names from model: {model_names}")
            elif self.use_coco:
                print(f"✓ Using COCO class names ({len(self.class_names)} classes)")
            
            print(f"✓ Model loaded successfully")
            print(f"  - Classes ({len(self.class_names)}): {self.class_names[:5]}{'...' if len(self.class_names) > 5 else ''}")
            print(f"  - Confidence threshold: {self.conf_threshold}")
            print(f"  - IoU threshold: {self.iou_threshold}")
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def detect_image(self, image_path, save_path=None, show_result=False):
        """
        Detect objects in a single image
        
        Args:
            image_path (str): Path to input image
            save_path (str): Path to save annotated image (optional)
            show_result (bool): Whether to display the result
            
        Returns:
            dict: Detection results
        """
        try:
            # Read image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            print(f"Processing image: {image_path}")
            
            # Run inference
            start_time = time.time()
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            inference_time = time.time() - start_time
            
            # Process results
            detections = self.process_results(results[0], image.shape)
            
            # Draw annotations
            annotated_image = self.draw_annotations(image.copy(), detections)
            
            # Save annotated image if path provided
            if save_path:
                cv2.imwrite(save_path, annotated_image)
                print(f"✓ Annotated image saved: {save_path}")
            
            # Show result if requested
            if show_result:
                self.show_image(annotated_image, f"Detections - {os.path.basename(image_path)}")
            
            # Print detection summary
            self.print_detection_summary(detections, inference_time)
            
            return {
                'detections': detections,
                'inference_time': inference_time,
                'annotated_image': annotated_image
            }
            
        except Exception as e:
            raise Exception(f"Error detecting objects in image: {str(e)}")
    
    def detect_video(self, video_path, output_path=None, show_result=False):
        """
        Detect objects in video
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save annotated video (optional)
            show_result (bool): Whether to display the result
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Processing video: {video_path}")
            print(f"  - Resolution: {width}x{height}")
            print(f"  - FPS: {fps}")
            print(f"  - Total frames: {total_frames}")
            
            # Initialize video writer if output path provided
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            total_detections = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run inference
                results = self.model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # Process results
                detections = self.process_results(results[0], frame.shape)
                total_detections += len(detections)
                
                # Draw annotations
                annotated_frame = self.draw_annotations(frame.copy(), detections)
                
                # Add frame info
                info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detections)}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame if output video specified
                if writer:
                    writer.write(annotated_frame)
                
                # Show frame if requested
                if show_result:
                    cv2.imshow('Video Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Print progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"✓ Video processing completed")
            print(f"  - Processed frames: {frame_count}")
            print(f"  - Total detections: {total_detections}")
            if output_path:
                print(f"  - Output saved: {output_path}")
            
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")
    
    def detect_webcam(self, camera_id=0):
        """
        Real-time detection from webcam
        
        Args:
            camera_id (int): Camera device ID
        """
        try:
            print(f"Starting webcam detection (Camera ID: {camera_id})")
            print("Press 'q' to quit, 's' to save screenshot")
            
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise ValueError(f"Could not open camera {camera_id}")
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                frame_count += 1
                
                # Run inference
                results = self.model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # Process results
                detections = self.process_results(results[0], frame.shape)
                
                # Draw annotations
                annotated_frame = self.draw_annotations(frame.copy(), detections)
                
                # Add info
                info_text = f"Frame: {frame_count} | Detections: {len(detections)} | Press 'q' to quit"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Webcam Detection', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"Screenshot saved: {screenshot_path}")
            
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam detection stopped")
            
        except Exception as e:
            raise Exception(f"Error with webcam detection: {str(e)}")
    
    def process_results(self, result, image_shape):
        """
        Process YOLO detection results
        
        Args:
            result: YOLO result object
            image_shape: Shape of input image (H, W, C)
            
        Returns:
            list: List of detection dictionaries
        """
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                
                # Get class name
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'class_name': class_name
                }
                
                detections.append(detection)
        
        return detections
    
    def draw_annotations(self, image, detections):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            numpy.ndarray: Annotated image
        """
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = bbox
            
            # Get color for class
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                image, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                image, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
        
        return image
    
    def show_image(self, image, window_name="Detection Result"):
        """Display image in window"""
        cv2.imshow(window_name, image)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def print_detection_summary(self, detections, inference_time):
        """Print detection summary"""
        print(f"\n{'='*50}")
        print(f"DETECTION SUMMARY")
        print(f"{'='*50}")
        print(f"Inference time: {inference_time:.3f}s")
        print(f"Total detections: {len(detections)}")
        
        # Count detections by class
        class_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if class_counts:
            print(f"Detections by class:")
            for class_name, count in class_counts.items():
                print(f"  - {class_name}: {count}")
        
        print(f"{'='*50}\n")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Vehicle Detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--source', type=str, required=True,
                       help='Source: image path, video path, or "webcam"')
    parser.add_argument('--output', type=str,
                       help='Output path for annotated result')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--show', action='store_true',
                       help='Show detection results')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID for webcam detection')
    parser.add_argument('--coco', action='store_true',
                       help='Use COCO class names (80 classes) instead of model classes')
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = YOLOv8Detector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            use_coco=args.coco
        )
        
        # Process based on source type
        if args.source.lower() == 'webcam':
            detector.detect_webcam(args.camera)
        elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            detector.detect_video(args.source, args.output, args.show)
        else:
            # Assume it's an image
            detector.detect_image(args.source, args.output, args.show)
        
        print("🎉 Detection completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
