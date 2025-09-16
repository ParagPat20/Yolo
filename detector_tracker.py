#!/usr/bin/env python3
"""
YOLO Detector + OpenCV Tracker
Author: AI Assistant
Description: Lightweight object detection with OpenCV DNN + tracking for better performance
"""

import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path

class YOLODetectorTracker:
    def __init__(self, model_path, config_path=None, class_names=None, 
                 conf_threshold=0.5, nms_threshold=0.4, tracker_type='CSRT', use_coco=False):
        """
        Initialize YOLO detector with OpenCV tracker
        
        Args:
            model_path (str): Path to YOLO model (.weights, .onnx, or .pb)
            config_path (str): Path to YOLO config file (.cfg) - optional for ONNX
            class_names (list): List of class names
            conf_threshold (float): Confidence threshold for detection
            nms_threshold (float): NMS threshold
            tracker_type (str): Tracker type ('CSRT', 'KCF', 'MOSSE')
            use_coco (bool): Use COCO class names (80 classes) instead of custom
        """
        self.model_path = model_path
        self.config_path = config_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.tracker_type = tracker_type
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
        
        # Set class names based on preference
        if use_coco:
            self.class_names = self.coco_names
        elif class_names:
            self.class_names = class_names
        else:
            self.class_names = ['mobil', 'motor', 'truck']  # Default vehicle classes
        
        # Colors for each class
        if use_coco:
            # Generate colors for 80 COCO classes
            np.random.seed(42)  # For consistent colors
            self.colors = []
            for i in range(len(self.class_names)):
                color = tuple(map(int, np.random.randint(0, 255, 3)))
                self.colors.append(color)
        else:
            # Default colors for vehicle classes or custom classes
            self.colors = [
                (255, 0, 0),    # Blue for mobil
                (0, 255, 0),    # Green for motor
                (0, 0, 255),    # Red for truck
            ]
            # Extend colors if more classes than default colors
            while len(self.colors) < len(self.class_names):
                self.colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        
        # Tracking state
        self.tracker = None
        self.tracking_target = None
        self.tracking_bbox = None
        self.tracking_class = None
        self.tracking_id = None
        self.frames_since_detection = 0
        self.max_tracking_frames = 30  # Re-detect after N frames
        
        # Load YOLO model
        self.load_model()
        
    def load_model(self):
        """Load YOLO model using OpenCV DNN"""
        try:
            print(f"Loading YOLO model: {self.model_path}")
            
            # Determine model format and load accordingly
            if self.model_path.endswith('.onnx'):
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                print("✓ ONNX model loaded")
            elif self.model_path.endswith('.weights'):
                if not self.config_path:
                    raise ValueError("Config file required for .weights format")
                self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.model_path)
                print("✓ Darknet model loaded")
            elif self.model_path.endswith('.pb'):
                self.net = cv2.dnn.readNetFromTensorflow(self.model_path)
                print("✓ TensorFlow model loaded")
            else:
                raise ValueError("Unsupported model format. Use .onnx, .weights, or .pb")
            
            # Set backend and target
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get output layer names
            self.output_layers = self.net.getUnconnectedOutLayersNames()
            print(f"✓ Model loaded with {len(self.output_layers)} output layers")
            
            # Test model with dummy input to understand output format
            self._test_model_output()
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def _test_model_output(self):
        """Test model with dummy input to understand output format"""
        try:
            # Create dummy input
            dummy_input = np.random.rand(640, 640, 3).astype(np.float32)
            blob = cv2.dnn.blobFromImage(dummy_input, 1/255.0, (640, 640), swapRB=True, crop=False)
            
            # Run inference
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            
            print(f"✓ Model output analysis:")
            for i, output in enumerate(outputs):
                print(f"  Output {i}: shape {output.shape}")
                if len(output.shape) >= 2:
                    print(f"    Detection format: {output.shape[-1]} values per detection")
                    expected_classes = output.shape[-1] - 5  # Subtract x,y,w,h,conf
                    if expected_classes == len(self.class_names):
                        print(f"    ✓ Classes match: {expected_classes}")
                    else:
                        print(f"    ⚠️  Class mismatch: expected {len(self.class_names)}, got {expected_classes}")
                        
        except Exception as e:
            print(f"⚠️  Warning: Could not analyze model output: {str(e)}")
    
    def create_tracker(self):
        """Create OpenCV tracker based on specified type"""
        # Try trackers in order of preference
        tracker_types = [self.tracker_type.upper()]
        
        # Add fallbacks if the preferred tracker fails
        if self.tracker_type.upper() == 'CSRT':
            tracker_types.extend(['KCF', 'MOSSE'])
        elif self.tracker_type.upper() == 'KCF':
            tracker_types.extend(['CSRT', 'MOSSE'])
        elif self.tracker_type.upper() == 'MOSSE':
            tracker_types.extend(['KCF', 'CSRT'])
        
        for tracker_type in tracker_types:
            try:
                if tracker_type == 'CSRT':
                    tracker = cv2.TrackerCSRT_create()
                    if tracker_type != self.tracker_type.upper():
                        print(f"Note: Using {tracker_type} tracker as fallback")
                    return tracker
                elif tracker_type == 'KCF':
                    tracker = cv2.TrackerKCF_create()
                    if tracker_type != self.tracker_type.upper():
                        print(f"Note: Using {tracker_type} tracker as fallback")
                    return tracker
                elif tracker_type == 'MOSSE':
                    tracker = cv2.legacy.TrackerMOSSE_create()
                    if tracker_type != self.tracker_type.upper():
                        print(f"Note: Using {tracker_type} tracker as fallback")
                    return tracker
            except (AttributeError, cv2.error) as e:
                print(f"Warning: {tracker_type} tracker not available: {str(e)}")
                continue
        
        # Final fallback - try basic KCF
        try:
            print("Warning: Using basic KCF tracker as last resort")
            return cv2.TrackerKCF_create()
        except Exception as e:
            raise Exception(f"No trackers available: {str(e)}")
    
    def detect_objects(self, frame):
        """
        Detect objects using YOLO
        
        Args:
            frame: Input frame
            
        Returns:
            list: List of detections [(class_id, confidence, bbox), ...]
        """
        height, width = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (640, 640), swapRB=True, crop=False
        )
        
        # Set input to the network
        self.net.setInput(blob)
        
        # Run forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Parse outputs
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            # Handle YOLOv8 output format: (1, 7, 8400) -> (8400, 7)
            if len(output.shape) == 3:
                # Remove batch dimension and transpose: (1, 7, 8400) -> (7, 8400) -> (8400, 7)
                output = output[0].T
            elif len(output.shape) == 2 and output.shape[0] < output.shape[1]:
                # Transpose if needed: (7, 8400) -> (8400, 7)
                output = output.T
            
            # Debug: print output shape only once per detection cycle
            if not hasattr(self, '_debug_printed'):
                print(f"Debug: Processing output shape {output.shape}")
                self._debug_printed = True
            
            valid_detections = 0
            for detection in output:
                # YOLOv8 format: [center_x, center_y, width, height, class1_score, class2_score, class3_score]
                if len(detection) >= 4 + len(self.class_names):
                    # Extract bounding box coordinates (normalized 0-1)
                    center_x, center_y, w, h = detection[:4]
                    
                    # Extract class scores (starting from index 4)
                    class_scores = detection[4:4 + len(self.class_names)]
                    
                    # Find best class
                    if len(class_scores) > 0:
                        class_id = np.argmax(class_scores)
                        confidence = class_scores[class_id]  # Max class score is the confidence
                        
                        # Validate class_id is within bounds
                        if class_id >= len(self.class_names):
                            continue  # Skip invalid class predictions
                        
                        if confidence > self.conf_threshold:
                            # Debug: Print raw values for the first few detections
                            if valid_detections < 3:
                                print(f"Debug: Raw detection - center_x:{center_x:.3f}, center_y:{center_y:.3f}, w:{w:.3f}, h:{h:.3f}, conf:{confidence:.3f}")
                            
                            # Check if coordinates are already in pixel format or normalized (0-1)
                            if center_x > 1.0 or center_y > 1.0 or w > 1.0 or h > 1.0:
                                # Already in pixel coordinates - use directly
                                center_x_px = center_x
                                center_y_px = center_y
                                w_px = w
                                h_px = h
                                if valid_detections < 3:
                                    print(f"Debug: Using pixel coordinates directly")
                            else:
                                # Normalized coordinates - scale to frame size
                                center_x_px = center_x * width
                                center_y_px = center_y * height
                                w_px = w * width
                                h_px = h * height
                                if valid_detections < 3:
                                    print(f"Debug: Scaling normalized coordinates")
                            
                            # Calculate top-left corner
                            x = int(center_x_px - w_px/2)
                            y = int(center_y_px - h_px/2)
                            w_int = int(w_px)
                            h_int = int(h_px)
                            
                            if valid_detections < 3:
                                print(f"Debug: Calculated bbox - x:{x}, y:{y}, w:{w_int}, h:{h_int}")
                            
                            # Ensure bounding box is valid and within image bounds
                            x = max(0, min(x, width - w_int))  # Ensure x + w doesn't exceed width
                            y = max(0, min(y, height - h_int))  # Ensure y + h doesn't exceed height
                            w_int = max(1, min(w_int, width - x))
                            h_int = max(1, min(h_int, height - y))
                            
                            if valid_detections < 3:
                                print(f"Debug: Clamped bbox - x:{x}, y:{y}, w:{w_int}, h:{h_int}")
                            
                            # Final validation with detailed debugging
                            bbox_valid = (x >= 0 and y >= 0 and w_int > 0 and h_int > 0 and 
                                        x + w_int <= width and y + h_int <= height and
                                        w_int >= 20 and h_int >= 20)  # Minimum size check
                            
                            if bbox_valid:
                                boxes.append([x, y, w_int, h_int])
                                confidences.append(float(confidence))
                                class_ids.append(int(class_id))
                                valid_detections += 1
                                if valid_detections <= 3:
                                    print(f"Debug: ✓ Valid bbox accepted: [{x}, {y}, {w_int}, {h_int}]")
                            else:
                                if valid_detections < 3:
                                    print(f"Debug: ❌ Invalid bbox rejected - x:{x}, y:{y}, w:{w_int}, h:{h_int}, frame:{width}x{height}")
            
            if valid_detections > 0:
                print(f"Found {valid_detections} valid detections")
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append((class_ids[i], confidences[i], boxes[i]))
        
        return detections
    
    def select_target(self, detections, target_class=None):
        """
        Select target object to track
        
        Args:
            detections: List of detections
            target_class: Preferred class to track (optional)
            
        Returns:
            tuple: (class_id, confidence, bbox) or None
        """
        if not detections:
            return None
        
        # Filter out detections that are too close to edges or too small
        valid_detections = []
        for detection in detections:
            class_id, confidence, bbox = detection
            x, y, w, h = bbox
            
            # Check if detection is suitable for tracking
            edge_margin = 10
            min_size = 30
            
            if (x > edge_margin and y > edge_margin and 
                x + w < 640 - edge_margin and y + h < 480 - edge_margin and
                w >= min_size and h >= min_size):
                valid_detections.append(detection)
        
        # If no valid detections, fall back to all detections
        if not valid_detections:
            valid_detections = detections
            print("Warning: No ideal detections found, using all detections")
        
        # If target class specified, prioritize it among valid detections
        if target_class is not None:
            for detection in valid_detections:
                class_id, confidence, bbox = detection
                if class_id == target_class:
                    return detection
        
        # Otherwise, return the most confident valid detection
        return max(valid_detections, key=lambda x: x[1])
    
    def init_tracker(self, frame, bbox):
        """Initialize tracker with bounding box"""
        # Validate bounding box
        x, y, w, h = bbox
        frame_height, frame_width = frame.shape[:2]
        
        print(f"Debug: Attempting to init tracker with bbox: {bbox}, frame: {frame_width}x{frame_height}")
        
        # Comprehensive validation
        if x < 0 or y < 0:
            print(f"❌ Negative coordinates: x={x}, y={y}")
            return False
        if w <= 0 or h <= 0:
            print(f"❌ Invalid dimensions: w={w}, h={h}")
            return False
        if x >= frame_width or y >= frame_height:
            print(f"❌ Coordinates outside frame: x={x}, y={y}")
            return False
        if x + w > frame_width or y + h > frame_height:
            print(f"❌ Bbox extends outside frame: x+w={x+w}, y+h={y+h}")
            return False
        
        # Additional safety margin and edge detection
        if w < 20 or h < 20:
            print(f"❌ Bbox too small for reliable tracking: {w}x{h} (minimum 20x20)")
            return False
        
        # Avoid bboxes too close to edges (can cause tracker issues)
        edge_margin = 5
        if (x < edge_margin or y < edge_margin or 
            x + w > frame_width - edge_margin or y + h > frame_height - edge_margin):
            print(f"❌ Bbox too close to frame edges: {bbox}")
            return False
        
        try:
            self.tracker = self.create_tracker()
            # Convert to tuple with integer values
            bbox_tuple = (int(x), int(y), int(w), int(h))
            print(f"Debug: Initializing tracker with validated bbox: {bbox_tuple}")
            
            success = self.tracker.init(frame, bbox_tuple)
            if success:
                self.tracking_bbox = bbox
                print(f"✓ Tracker initialized successfully with bbox: {bbox}")
            else:
                print(f"❌ Tracker initialization failed (OpenCV returned False)")
                self.tracker = None
            return success
        except Exception as e:
            print(f"❌ Tracker initialization error: {str(e)}")
            self.tracker = None
            return False
    
    def update_tracker(self, frame):
        """Update tracker and return success status and bbox"""
        if self.tracker is None:
            return False, None
        
        try:
            success, bbox = self.tracker.update(frame)
            if success and bbox is not None:
                # Convert bbox to integer coordinates and validate
                x, y, w, h = bbox
                frame_height, frame_width = frame.shape[:2]
                
                # Validate the updated bbox
                if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                    x + w <= frame_width and y + h <= frame_height):
                    bbox = [int(x), int(y), int(w), int(h)]
                    self.tracking_bbox = bbox
                    return True, bbox
                else:
                    print(f"❌ Tracker returned invalid bbox: {bbox}")
                    return False, None
            else:
                return False, None
        except Exception as e:
            print(f"❌ Tracker update error: {str(e)}")
            self.tracker = None  # Reset tracker on error
            return False, None
    
    def draw_detection(self, frame, class_id, confidence, bbox, is_tracking=False):
        """Draw bounding box and label on frame"""
        x, y, w, h = bbox
        
        # Get class name and color
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
        color = self.colors[class_id % len(self.colors)]
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Prepare label
        if is_tracking:
            label = f"TRACKING: {class_name} ({confidence:.2f})"
        else:
            label = f"DETECTED: {class_name} ({confidence:.2f})"
        
        # Draw label background and text
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_video(self, source, output_path=None, show_result=True, target_class=None):
        """
        Process video with detection and tracking
        
        Args:
            source: Video source (path or camera index)
            output_path: Output video path (optional)
            show_result: Whether to display results
            target_class: Preferred class to track (optional)
        """
        # Open video source
        if isinstance(source, str):
            cap = cv2.VideoCapture(source)
            print(f"Processing video: {source}")
        else:
            cap = cv2.VideoCapture(source)
            print(f"Processing camera: {source}")
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Initialize video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_count = 0
        tracking_count = 0
        
        print(f"\nControls:")
        print(f"  - Press 'q' to quit")
        print(f"  - Press 'r' to reset tracker")
        print(f"  - Press 's' to save screenshot")
        print(f"  - Press SPACE to pause/resume")
        print(f"\nTracker: {self.tracker_type}")
        print(f"Target class: {self.class_names[target_class] if target_class is not None else 'Any'}")
        print(f"{'='*60}")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or failed to read frame")
                    break
                
                frame_count += 1
                start_time = time.time()
                
                # Decide whether to detect or track
                should_detect = (
                    self.tracker is None or 
                    self.frames_since_detection >= self.max_tracking_frames
                )
                
                if should_detect:
                    # Run YOLO detection
                    detections = self.detect_objects(frame)
                    detection_count += 1
                    
                    if detections:
                        # Select target to track
                        target = self.select_target(detections, target_class)
                        if target:
                            class_id, confidence, bbox = target
                            
                            # Initialize tracker
                            if self.init_tracker(frame, bbox):
                                self.tracking_class = class_id
                                self.tracking_target = confidence
                                self.frames_since_detection = 0
                                
                                # Draw detection
                                frame = self.draw_detection(frame, class_id, confidence, bbox, False)
                            else:
                                print("Failed to initialize tracker")
                        else:
                            print("No suitable target found")
                    else:
                        print("No objects detected")
                        self.tracker = None
                
                else:
                    # Update tracker
                    success, bbox = self.update_tracker(frame)
                    tracking_count += 1
                    
                    if success and bbox:
                        # Draw tracking result
                        frame = self.draw_detection(
                            frame, self.tracking_class, self.tracking_target, bbox, True
                        )
                        self.frames_since_detection += 1
                    else:
                        print("Tracking failed, will re-detect")
                        self.tracker = None
                        self.frames_since_detection = self.max_tracking_frames
                
                # Add frame info
                processing_time = time.time() - start_time
                fps_current = 1.0 / processing_time if processing_time > 0 else 0
                
                info_text = f"Frame: {frame_count} | FPS: {fps_current:.1f} | Mode: {'DETECT' if should_detect else 'TRACK'}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                stats_text = f"Detections: {detection_count} | Tracking: {tracking_count}"
                cv2.putText(frame, stats_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame if output video specified
            if writer and not paused:
                writer.write(frame)
            
            # Show frame if requested
            if show_result:
                cv2.imshow('YOLO Detector + Tracker', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting tracker...")
                self.tracker = None
                self.frames_since_detection = self.max_tracking_frames
            elif key == ord('s') and not paused:
                screenshot_path = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETED")
        print(f"{'='*60}")
        print(f"Total frames: {frame_count}")
        print(f"Detections: {detection_count}")
        print(f"Tracking updates: {tracking_count}")
        print(f"Detection ratio: {detection_count/frame_count*100:.1f}%")
        if output_path:
            print(f"Output saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Detector + OpenCV Tracker')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLO model (.onnx, .weights, .pb)')
    parser.add_argument('--config', type=str,
                       help='Path to YOLO config file (.cfg) - required for .weights')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (path or camera index)')
    parser.add_argument('--output', type=str,
                       help='Output video path')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4,
                       help='NMS threshold')
    parser.add_argument('--tracker', type=str, default='CSRT',
                       choices=['CSRT', 'KCF', 'MOSSE'],
                       help='Tracker type')
    parser.add_argument('--target-class', type=int,
                       help='Preferred class ID to track')
    parser.add_argument('--max-track-frames', type=int, default=30,
                       help='Max frames to track before re-detection')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display results')
    parser.add_argument('--coco', action='store_true',
                       help='Use COCO class names (80 classes) instead of custom classes')
    
    args = parser.parse_args()
    
    try:
        # Convert source to int if it's a digit (camera index)
        source = int(args.source) if args.source.isdigit() else args.source
        
        # Initialize detector-tracker
        detector_tracker = YOLODetectorTracker(
            model_path=args.model,
            config_path=args.config,
            conf_threshold=args.conf,
            nms_threshold=args.nms,
            tracker_type=args.tracker,
            use_coco=args.coco
        )
        
        # Set max tracking frames
        detector_tracker.max_tracking_frames = args.max_track_frames
        
        # Process video
        detector_tracker.process_video(
            source=source,
            output_path=args.output,
            show_result=not args.no_show,
            target_class=args.target_class
        )
        
        print("🎉 Processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
