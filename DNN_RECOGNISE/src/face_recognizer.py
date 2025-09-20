# Enhanced Face Recognition System with Advanced Person Tracking
# This file now integrates with the new advanced person tracking system
# At the top of your Python script inside venv
import sys
sys.path.append('/usr/lib/python3/dist-packages')  # Path to system Python packages

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import json
import os
import logging
import time
import math
import sys
from settings.settings import (
    CAMERA, FACE_DETECTION, PATHS, CONFIDENCE_THRESHOLD, TRACKING,
    PERSON_DETECTION, FACE_RECOGNITION, PERSON_TRACKING, FACE_TRACKING, SECURITY
)

# Try to import the advanced tracker
try:
    from advanced_person_tracker import AdvancedPersonTracker, initialize_camera
    ADVANCED_TRACKING_AVAILABLE = True
except ImportError:
    ADVANCED_TRACKING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️  Advanced tracking not available, using legacy system")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFaceDetector:
    """
    Enhanced face detection using multiple methods: Haar cascades, DNN, and ensemble
    """
    def __init__(self):
        self.haar_cascade = None
        self.haar_cascade_tree = None
        self.dnn_net = None
        self.detection_method = FACE_DETECTION['method']
        
        # Initialize Haar cascades
        try:
            self.haar_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
            if self.haar_cascade.empty():
                logger.warning("Failed to load primary Haar cascade")
                self.haar_cascade = None
        except Exception as e:
            logger.error(f"Error loading primary Haar cascade: {e}")
            
        try:
            self.haar_cascade_tree = cv2.CascadeClassifier(PATHS['cascade_alt_tree'])
            if self.haar_cascade_tree.empty():
                logger.warning("Failed to load tree Haar cascade")
                self.haar_cascade_tree = None
        except Exception as e:
            logger.error(f"Error loading tree Haar cascade: {e}")
        
        # Initialize DNN model with better error handling
        self.dnn_net = None
        try:
            if os.path.exists(PATHS['dnn_model']) and os.path.exists(PATHS['dnn_config']):
                # Try loading the DNN model
                self.dnn_net = cv2.dnn.readNetFromTensorflow(PATHS['dnn_model'], PATHS['dnn_config'])
                
                # Test the model with a dummy input to ensure it works
                test_blob = cv2.dnn.blobFromImage(np.zeros((300, 300, 3), dtype=np.uint8), 1.0, (300, 300), [104, 117, 123])
                self.dnn_net.setInput(test_blob)
                test_output = self.dnn_net.forward()
                
                logger.info("✅ DNN face detection model loaded and tested successfully")
            else:
                logger.warning("⚠️  DNN model files not found, using Haar cascades only")
                
        except Exception as e:
            logger.warning(f"⚠️  DNN model failed to load: {str(e)[:100]}...")
            logger.info("🔄 Falling back to Haar cascade detection (still very effective!)")
            self.dnn_net = None
            
        # Adjust detection method if DNN failed
        if self.detection_method in ['dnn', 'ensemble'] and self.dnn_net is None:
            if self.haar_cascade is not None or self.haar_cascade_tree is not None:
                logger.info("🔧 Switching to Haar cascade detection due to DNN unavailability")
                self.detection_method = 'haar'
            else:
                logger.error("❌ No working detection methods available!")
                raise ValueError("No face detection methods could be initialized")
                
    def get_detection_status(self):
        """Get status of available detection methods"""
        status = {
            'primary_haar': self.haar_cascade is not None,
            'tree_haar': self.haar_cascade_tree is not None,
            'dnn': self.dnn_net is not None,
            'active_method': self.detection_method
        }
        return status
    
    def detect_faces_haar(self, gray_frame, cascade):
        """Detect faces using Haar cascade"""
        if cascade is None:
            return []
        
        faces = cascade.detectMultiScale(
            gray_frame,
            scaleFactor=FACE_DETECTION['scale_factor'],
            minNeighbors=FACE_DETECTION['min_neighbors'],
            minSize=FACE_DETECTION['min_size']
        )
        return faces.tolist() if len(faces) > 0 else []
    
    def detect_faces_dnn(self, frame):
        """Detect faces using DNN model"""
        if self.dnn_net is None:
            return []
        
        try:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > FACE_DETECTION['dnn_confidence_threshold']:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    # Convert to (x, y, w, h) format
                    faces.append([x1, y1, x2 - x1, y2 - y1])
            
            return faces
        except Exception as e:
            logger.error(f"Error in DNN face detection: {e}")
            return []
    
    def merge_detections(self, detections_list):
        """Merge overlapping detections from multiple methods"""
        if not detections_list:
            return []
        
        all_faces = []
        for detections in detections_list:
            all_faces.extend(detections)
        
        if not all_faces:
            return []
        
        # Simple non-maximum suppression
        faces = np.array(all_faces)
        if len(faces) == 0:
            return []
        
        # Calculate areas
        areas = faces[:, 2] * faces[:, 3]
        
        # Sort by area (larger faces first)
        indices = np.argsort(areas)[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = faces[current]
            remaining_boxes = faces[indices[1:]]
            
            # Calculate intersection
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[0] + current_box[2], remaining_boxes[:, 0] + remaining_boxes[:, 2])
            y2 = np.minimum(current_box[1] + current_box[3], remaining_boxes[:, 1] + remaining_boxes[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            union = areas[current] + areas[indices[1:]] - intersection
            
            iou = intersection / union
            
            # Keep boxes with IoU < 0.3
            indices = indices[1:][iou < 0.3]
        
        return faces[keep].tolist()
    
    def detect_faces(self, frame, gray_frame):
        """Main face detection method"""
        if self.detection_method == 'haar':
            # Use primary Haar cascade
            return self.detect_faces_haar(gray_frame, self.haar_cascade)
        
        elif self.detection_method == 'dnn':
            # Use DNN model
            return self.detect_faces_dnn(frame)
        
        elif self.detection_method == 'ensemble':
            # Use ensemble of multiple methods
            detections_list = []
            
            # Add Haar cascade detections
            if self.haar_cascade is not None:
                haar_faces = self.detect_faces_haar(gray_frame, self.haar_cascade)
                if haar_faces:
                    detections_list.append(haar_faces)
            
            # Add tree cascade detections
            if self.haar_cascade_tree is not None:
                tree_faces = self.detect_faces_haar(gray_frame, self.haar_cascade_tree)
                if tree_faces:
                    detections_list.append(tree_faces)
            
            # Add DNN detections
            if self.dnn_net is not None:
                dnn_faces = self.detect_faces_dnn(frame)
                if dnn_faces:
                    detections_list.append(dnn_faces)
            
            # Merge all detections
            return self.merge_detections(detections_list)
        
        else:
            logger.error(f"Unknown detection method: {self.detection_method}")
            return []

class FaceTracker:
    """
    Face tracking system that maintains recognition state for detected faces
    """
    def __init__(self):
        self.tracked_faces = {}  # Dictionary to store tracked faces
        self.face_id_counter = 0
        
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two face positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def find_matching_face(self, face_center):
        """Find if current face matches any tracked face"""
        for face_id, face_data in self.tracked_faces.items():
            distance = self.calculate_distance(face_center, face_data['center'])
            if distance < TRACKING['max_distance_threshold']:
                return face_id
        return None
    
    def update_tracking(self, detected_faces, gray_frame, recognizer, names):
        """Update face tracking and recognition with improved unknown face handling"""
        current_time = time.time()
        face_results = []
        
        # Remove expired tracked faces (different duration for known vs unknown)
        expired_faces = []
        for face_id, face_data in self.tracked_faces.items():
            # Use shorter tracking duration for unknown faces
            if face_data['name'] == "Unknown":
                tracking_duration = TRACKING['unknown_tracking_duration']
            else:
                tracking_duration = TRACKING['tracking_duration']
                
            if current_time - face_data['last_seen'] > tracking_duration:
                expired_faces.append(face_id)
        
        for face_id in expired_faces:
            face_name = self.tracked_faces[face_id]['name']
            del self.tracked_faces[face_id]
            logger.info(f"Face tracking expired for ID: {face_id} ({face_name})")
        
        # Process each detected face
        for (x, y, w, h) in detected_faces:
            face_center = (x + w//2, y + h//2)
            face_rect = (x, y, w, h)
            
            # Check if this face matches any tracked face
            matching_face_id = self.find_matching_face(face_center)
            
            if matching_face_id is not None:
                # Update existing tracked face
                tracked_face = self.tracked_faces[matching_face_id]
                tracked_face['center'] = face_center
                tracked_face['rect'] = face_rect
                tracked_face['last_seen'] = current_time
                
                # Check if we should re-verify this face (for ALL faces, not just unknown)
                should_reverify = False
                
                # Always re-verify unknown faces more frequently
                if tracked_face['name'] == "Unknown":
                    should_reverify = current_time - tracked_face['recognized_at'] > TRACKING['unknown_retry_interval']
                else:
                    # For known faces, check verification interval
                    should_reverify = current_time - tracked_face['recognized_at'] > TRACKING['verification_interval']
                    
                    # Also re-verify if confidence was low during initial recognition
                    try:
                        if tracked_face['confidence'] != "N/A":
                            confidence_value = float(tracked_face['confidence'].replace('%', ''))
                            if confidence_value > TRACKING['confidence_threshold_for_reverify']:
                                # Re-verify more frequently for low-confidence recognitions
                                should_reverify = current_time - tracked_face['recognized_at'] > (TRACKING['verification_interval'] / 2)
                    except (ValueError, AttributeError):
                        pass  # If we can't parse confidence, use normal interval
                
                if should_reverify:
                    # Re-verify recognition for this face
                    face_roi = gray_frame[y:y+h, x:x+w]
                    face_roi = preprocess_face_for_recognition(face_roi)
                    id, confidence = recognizer.predict(face_roi)
                    
                    if confidence >= CONFIDENCE_THRESHOLD and confidence <= 100:
                        new_name = names.get(str(id), "Unknown")
                        new_confidence_text = f"{confidence:.1f}%"
                    else:
                        new_name = "Unknown"
                        new_confidence_text = "N/A"
                    
                    # Check if recognition changed
                    if new_name != tracked_face['name']:
                        logger.info(f"Face ID {matching_face_id} recognition changed: {tracked_face['name']} -> {new_name} (Confidence: {new_confidence_text})")
                    else:
                        logger.info(f"Face ID {matching_face_id} re-verified: {new_name} (Confidence: {new_confidence_text})")
                    
                    # Update the tracked face with new recognition
                    tracked_face['name'] = new_name
                    tracked_face['confidence'] = new_confidence_text
                    tracked_face['recognized_at'] = current_time
                
                # Use current recognition result
                face_results.append({
                    'rect': face_rect,
                    'name': tracked_face['name'],
                    'confidence': tracked_face['confidence'],
                    'is_tracked': True,
                    'was_reverified': should_reverify
                })
                
            else:
                # New face detected - perform recognition
                should_recognize = True
                
                # Check if enough time has passed since last recognition
                if hasattr(self, 'last_recognition_time'):
                    if current_time - self.last_recognition_time < TRACKING['recognition_cooldown']:
                        should_recognize = False
                
                if should_recognize:
                    # Perform face recognition
                    face_roi = gray_frame[y:y+h, x:x+w]
                    face_roi = preprocess_face_for_recognition(face_roi)
                    id, confidence = recognizer.predict(face_roi)
                    
                    if confidence >= CONFIDENCE_THRESHOLD and confidence <= 100:
                        name = names.get(str(id), "Unknown")
                        confidence_text = f"{confidence:.1f}%"
                    else:
                        name = "Unknown"
                        confidence_text = "N/A"
                    
                    # Create new tracked face
                    self.face_id_counter += 1
                    self.tracked_faces[self.face_id_counter] = {
                        'center': face_center,
                        'rect': face_rect,
                        'name': name,
                        'confidence': confidence_text,
                        'last_seen': current_time,
                        'recognized_at': current_time
                    }
                    
                    self.last_recognition_time = current_time
                    
                    face_results.append({
                        'rect': face_rect,
                        'name': name,
                        'confidence': confidence_text,
                        'is_tracked': False,
                        'was_reverified': False
                    })
                    
                    logger.info(f"New face recognized: {name} (Confidence: {confidence_text})")
                    
                else:
                    # Skip recognition, just detect
                    face_results.append({
                        'rect': face_rect,
                        'name': "Detecting...",
                        'confidence': "...",
                        'is_tracked': False,
                        'was_reverified': False
                    })
        
        return face_results
    
    def reset_tracking(self):
        """Reset all tracked faces - useful for manual reset"""
        logger.info(f"Resetting {len(self.tracked_faces)} tracked faces")
        self.tracked_faces.clear()
        self.face_id_counter = 0

def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """
    Initialize the camera with error handling
    
    Parameters:
        camera_index (int): Camera device index
    Returns:
        cv2.VideoCapture: Initialized camera object
    """
    try:
        cam = cv2.VideoCapture(camera_index)
        if not cam.isOpened():
            logger.error("Could not open webcam")
            return None
            
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return cam
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return None

def load_names(filename: str) -> dict:
    """
    Load name mappings from JSON file
    
    Parameters:
        filename (str): Path to the JSON file containing name mappings
    Returns:
        dict: Dictionary mapping IDs to names
    """
    try:
        names_json = {}
        if os.path.exists(filename):
            with open(filename, 'r') as fs:
                content = fs.read().strip()
                if content:
                    names_json = json.loads(content)
        return names_json
    except Exception as e:
        logger.error(f"Error loading names: {e}")
        return {}

def preprocess_face_for_recognition(face_roi):
    """
    Preprocess face region for better recognition accuracy
    """
    try:
        # Resize to consistent size
        face_roi = cv2.resize(face_roi, (100, 100))
        
        # Apply histogram equalization for better lighting normalization
        face_roi = cv2.equalizeHist(face_roi)
        
        # Apply Gaussian blur to reduce noise
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        
        return face_roi
    except Exception as e:
        logger.error(f"Error in face preprocessing: {e}")
        return face_roi

if __name__ == "__main__":
    try:
        # Check if advanced tracking is available
        if ADVANCED_TRACKING_AVAILABLE:
            logger.info("🚀 Starting Advanced Person Tracking System...")
            
            # Initialize advanced tracker
            tracker = AdvancedPersonTracker()
            
            # Initialize camera
            cam = initialize_camera(CAMERA['index'])
            if cam is None:
                raise ValueError("Failed to initialize camera")
            
            logger.info("🎯 Advanced Person Tracking System Active!")
            logger.info("Features:")
            logger.info("  ✅ YOLOv8 Person Detection")
            logger.info("  ✅ YOLO Face Detection") 
            logger.info("  ✅ ArcFace Recognition")
            logger.info("  ✅ ByteTrack Person Tracking")
            logger.info("  ✅ Unknown Person Danger Alerts")
            logger.info("  ✅ Persistent Identity Tracking")
            logger.info("")
            logger.info("Controls:")
            logger.info("  - ESC/Q: Exit")
            logger.info("  - 's': Save current frame")
            logger.info("  - 'r': Reset all tracks")
            logger.info("  - 'a': Switch to legacy mode")
            
            frame_count = 0
            fps_start = time.time()
            
            while True:
                ret, frame = cam.read()
                if not ret:
                    logger.warning("Failed to grab frame")
                    continue
                
                # Process frame with advanced tracking
                annotated_frame, tracks = tracker.process_frame(frame)
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start)
                    fps_start = time.time()
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                cv2.imshow('Advanced Person Tracking & Face Recognition', annotated_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or Q
                    break
                elif key == ord('s'):  # Save frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"advanced_capture_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    logger.info(f"📸 Saved: {filename}")
                elif key == ord('r'):  # Reset tracks
                    tracker.tracker.tracks.clear()
                    tracker.tracker.next_id = 1
                    logger.info("🔄 Reset all tracks")
                elif key == ord('a'):  # Switch to legacy mode
                    logger.info("🔄 Switching to legacy face recognition...")
                    cam.release()
                    cv2.destroyAllWindows()
                    # Fall through to legacy system
                    break
            
            if key != ord('a'):  # If not switching to legacy, exit
                logger.info("👋 Advanced system stopped")
                cam.release()
                cv2.destroyAllWindows()
                sys.exit(0)
        
        # Legacy system (fallback or explicit choice)
        logger.info("Starting legacy face recognition system with tracking...")
        
        # Initialize face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists(PATHS['trainer_file']):
            raise ValueError("Trainer file not found. Please train the model first.")
        recognizer.read(PATHS['trainer_file'])
        
        # Initialize enhanced face detector
        face_detector = EnhancedFaceDetector()
        
        # Display detection status
        detection_status = face_detector.get_detection_status()
        logger.info("🔍 Detection Methods Status:")
        logger.info(f"   Primary Haar Cascade: {'✅' if detection_status['primary_haar'] else '❌'}")
        logger.info(f"   Tree Haar Cascade: {'✅' if detection_status['tree_haar'] else '❌'}")
        logger.info(f"   DNN Model: {'✅' if detection_status['dnn'] else '❌'}")
        logger.info(f"   Active Method: {detection_status['active_method'].upper()}")
        
        # Initialize camera
        cam = initialize_camera(CAMERA['index'])
        if cam is None:
            raise ValueError("Failed to initialize camera")
        
        # Load names
        names = load_names(PATHS['names_file'])
        if not names:
            logger.warning("No names loaded, recognition will be limited")
        
        # Initialize face tracker
        face_tracker = FaceTracker()
        
        logger.info("🚀 Enhanced Face Recognition System with Advanced Detection!")
        logger.info(f"Detection method: {FACE_DETECTION['method']}")
        logger.info(f"Tracking duration: {TRACKING['tracking_duration']} seconds")
        logger.info(f"Verification interval: {TRACKING['verification_interval']} seconds (ALL faces)")
        logger.info(f"Unknown retry interval: {TRACKING['unknown_retry_interval']} seconds")
        logger.info(f"Low confidence re-verify threshold: {TRACKING['confidence_threshold_for_reverify']}%")
        logger.info("🔧 Available detection methods: Haar cascades, DNN, Ensemble")
        logger.info("⚡ Image preprocessing enabled for better recognition")
        logger.info("Controls:")
        logger.info("  - Press 'ESC' key to exit")
        logger.info("  - Press 'R' key to reset all face tracking")
        logger.info("  - Press 'SPACE' to force re-recognition of all faces")
        logger.info("  - Press '1' for Haar cascade detection")
        logger.info("  - Press '2' for DNN detection")
        logger.info("  - Press '3' for Ensemble detection")
        
        while True:
            ret, img = cam.read()
            if not ret:
                logger.warning("Failed to grab frame")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Use enhanced face detection
            faces = face_detector.detect_faces(img, gray)
            
            # Update face tracking and get results
            face_results = face_tracker.update_tracking(faces, gray, recognizer, names)
            
            # Draw results on the frame
            for result in face_results:
                x, y, w, h = result['rect']
                name = result['name']
                confidence = result['confidence']
                is_tracked = result['is_tracked']
                was_reverified = result.get('was_reverified', False)
                
                # Different colors for different face states
                if is_tracked:
                    if name == "Unknown":
                        # Orange for tracked unknown faces (will retry recognition)
                        rect_color = (0, 165, 255)
                        text_color = (0, 165, 255)
                        status_text = "UNKNOWN (RETRY)"
                    else:
                        if was_reverified:
                            # Purple for faces that were just re-verified
                            rect_color = (255, 0, 255)
                            text_color = (255, 0, 255)
                            status_text = "RE-VERIFIED"
                        else:
                            # Green for tracked known faces (stable recognition)
                            rect_color = (0, 255, 0)
                            text_color = (0, 255, 0)
                            status_text = "TRACKED"
                else:
                    # Blue for newly detected/recognized faces
                    rect_color = (255, 0, 0)
                    text_color = (255, 255, 255)
                    status_text = "RECOGNIZING"
                
                # Draw rectangle around face
                cv2.rectangle(img, (x, y), (x+w, y+h), rect_color, 2)
                
                # Display name
                cv2.putText(img, name, (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                
                # Display confidence
                cv2.putText(img, confidence, (x+5, y+h-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                # Display tracking status
                cv2.putText(img, status_text, (x+5, y+h-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Display system info
            info_text = f"Method: {FACE_DETECTION['method'].upper()} | Active Tracks: {len(face_tracker.tracked_faces)}"
            cv2.putText(img, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Display detection stats
            detection_info = f"Faces Detected: {len(faces)} | Verification: {TRACKING['verification_interval']}s"
            cv2.putText(img, detection_info, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow('Enhanced Face Recognition with Tracking', img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('r') or key == ord('R'):  # Reset tracking
                face_tracker.reset_tracking()
                logger.info("Manual reset triggered - all face tracking cleared")
            elif key == ord(' '):  # Space key - force re-recognition
                # Mark all tracked faces for re-recognition by updating their recognized_at time
                current_time = time.time()
                for face_data in face_tracker.tracked_faces.values():
                    face_data['recognized_at'] = current_time - TRACKING['unknown_retry_interval'] - 1
                logger.info("Forced re-recognition for all tracked faces")
            elif key == ord('1'):  # Switch to Haar cascade
                if face_detector.haar_cascade is not None:
                    face_detector.detection_method = 'haar'
                    logger.info("🔧 Switched to Haar cascade detection")
                else:
                    logger.warning("⚠️  Haar cascade not available")
            elif key == ord('2'):  # Switch to DNN
                if face_detector.dnn_net is not None:
                    face_detector.detection_method = 'dnn'
                    logger.info("🔧 Switched to DNN detection")
                else:
                    logger.warning("⚠️  DNN model not available")
            elif key == ord('3'):  # Switch to Ensemble
                available_methods = []
                if face_detector.haar_cascade is not None:
                    available_methods.append("Haar")
                if face_detector.haar_cascade_tree is not None:
                    available_methods.append("Tree")
                if face_detector.dnn_net is not None:
                    available_methods.append("DNN")
                
                if len(available_methods) > 1:
                    face_detector.detection_method = 'ensemble'
                    logger.info(f"🔧 Switched to Ensemble detection ({', '.join(available_methods)})")
                else:
                    logger.warning("⚠️  Ensemble needs multiple detection methods")
        
        logger.info("Face recognition with tracking stopped")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        
    finally:
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()