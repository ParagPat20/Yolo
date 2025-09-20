# Suppress macOS warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import json
import cv2
import os
import numpy as np
import time
from typing import Optional, Dict
import logging
from settings.settings import CAMERA, FACE_DETECTION, TRAINING, PATHS, FACE_RECOGNITION

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFaceDetector:
    """
    Enhanced face detection for training data collection
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
            if os.path.exists(PATHS.get('cascade_alt_tree', '')):
                self.haar_cascade_tree = cv2.CascadeClassifier(PATHS['cascade_alt_tree'])
                if self.haar_cascade_tree.empty():
                    logger.warning("Failed to load tree Haar cascade")
                    self.haar_cascade_tree = None
        except Exception as e:
            logger.error(f"Error loading tree Haar cascade: {e}")
        
        # Initialize DNN model
        self.dnn_net = None
        try:
            if os.path.exists(PATHS.get('dnn_model', '')) and os.path.exists(PATHS.get('dnn_config', '')):
                self.dnn_net = cv2.dnn.readNetFromTensorflow(PATHS['dnn_model'], PATHS['dnn_config'])
                # Test the model
                test_blob = cv2.dnn.blobFromImage(np.zeros((300, 300, 3), dtype=np.uint8), 1.0, (300, 300), [104, 117, 123])
                self.dnn_net.setInput(test_blob)
                test_output = self.dnn_net.forward()
                logger.info("✅ DNN model loaded for training data collection")
            else:
                logger.info("DNN model files not found, using Haar cascades for training")
        except Exception as e:
            logger.warning(f"DNN model failed to load for training: {str(e)[:100]}...")
            self.dnn_net = None
            
        # Adjust detection method if needed
        if self.detection_method == 'dnn' and self.dnn_net is None:
            logger.warning("DNN model not available, switching to Haar cascade")
            self.detection_method = 'haar'
        elif self.detection_method == 'yolo_face':
            logger.warning("YOLO face model not available, switching to DNN")
            self.detection_method = 'dnn'
            if self.dnn_net is None:
                logger.warning("DNN also not available, switching to Haar cascade")
                self.detection_method = 'haar'
        elif self.detection_method == 'ensemble' and self.dnn_net is None:
            self.detection_method = 'haar'
        
        logger.info(f"Face detection method for training: {self.detection_method.upper()}")
    
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
    
    def detect_faces(self, frame, gray_frame):
        """Main face detection method for training"""
        faces = []
        
        if self.detection_method == 'dnn' and self.dnn_net is not None:
            faces = self.detect_faces_dnn(frame)
            if faces:
                logger.debug(f"DNN detected {len(faces)} faces")
                return faces
        
        # Fallback to Haar cascade
        if self.haar_cascade is not None:
            faces = self.detect_faces_haar(gray_frame, self.haar_cascade)
            if faces:
                logger.debug(f"Primary Haar cascade detected {len(faces)} faces")
                return faces
        
        # Try alternative Haar cascade
        if self.haar_cascade_tree is not None:
            faces = self.detect_faces_haar(gray_frame, self.haar_cascade_tree)
            if faces:
                logger.debug(f"Tree Haar cascade detected {len(faces)} faces")
                return faces
        
        # Try with more relaxed parameters for Haar cascade
        if self.haar_cascade is not None:
            faces = self.haar_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,  # More relaxed
                minNeighbors=3,   # More relaxed
                minSize=(20, 20)  # Smaller minimum size
            )
            if len(faces) > 0:
                logger.debug(f"Relaxed Haar cascade detected {len(faces)} faces")
                return faces.tolist()
        
        return []

def preprocess_face_for_training(face_roi):
    """
    Preprocess face region for better training data quality
    Compatible with both LBPH and ArcFace training
    """
    try:
        # For ArcFace compatibility, resize to the configured input size
        target_size = FACE_RECOGNITION.get('input_size', (112, 112))
        if isinstance(target_size, tuple):
            face_roi = cv2.resize(face_roi, target_size)
        else:
            face_roi = cv2.resize(face_roi, (100, 100))  # Fallback
        
        # Apply histogram equalization for better lighting normalization
        face_roi = cv2.equalizeHist(face_roi)
        
        # Apply slight Gaussian blur to reduce noise
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        
        return face_roi
    except Exception as e:
        logger.error(f"Error in face preprocessing: {e}")
        return face_roi

def save_face_for_arcface(face_roi, face_id: int, count: int):
    """
    Save face in format suitable for ArcFace training
    """
    try:
        # Create ArcFace training directory
        arcface_dir = os.path.join(PATHS['image_dir'], 'arcface_format')
        person_dir = os.path.join(arcface_dir, f'person_{face_id}')
        os.makedirs(person_dir, exist_ok=True)
        
        # Save in RGB format for ArcFace
        if len(face_roi.shape) == 3:
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        
        # Save high-quality image
        filename = f'face_{count:04d}.jpg'
        filepath = os.path.join(person_dir, filename)
        cv2.imwrite(filepath, face_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        logger.info(f"💾 Saved ArcFace format: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save ArcFace format: {e}")

def create_directory(directory: str) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    except OSError as e:
        logger.error(f"Error creating directory {directory}: {e}")
        raise

def get_face_id(directory: str) -> int:
    """
    Get the first available face ID by checking existing files.
    
    Parameters:
        directory (str): The path of the directory of images.
    Returns:
        int: The next available face ID
    """
    try:
        if not os.path.exists(directory):
            return 1
            
        user_ids = []
        for filename in os.listdir(directory):
            if filename.startswith('Users-'):
                try:
                    number = int(filename.split('-')[1])
                    user_ids.append(number)
                except (IndexError, ValueError):
                    continue
                    
        return max(user_ids + [0]) + 1
    except Exception as e:
        logger.error(f"Error getting face ID: {e}")
        raise

def save_name(face_id: int, face_name: str, filename: str) -> None:
    """
    Save name-ID mapping to JSON file
    
    Parameters:
        face_id (int): The identifier of user
        face_name (str): The user name
        filename (str): Path to the JSON file
    """
    try:
        names_json: Dict[str, str] = {}
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as fs:
                    content = fs.read().strip()
                    if content:  # Only try to load if file is not empty
                        names_json = json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {filename}, starting fresh")
                names_json = {}
        
        names_json[str(face_id)] = face_name
        
        with open(filename, 'w') as fs:
            json.dump(names_json, fs, indent=4, ensure_ascii=False)
        logger.info(f"Saved name mapping for ID {face_id}")
    except Exception as e:
        logger.error(f"Error saving name mapping: {e}")
        raise

def initialize_camera(camera_index: int = 0) -> Optional[cv2.VideoCapture]:
    """
    Initialize the camera with error handling
    
    Parameters:
        camera_index (int): Camera device index
    Returns:
        cv2.VideoCapture or None: Initialized camera object
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

if __name__ == '__main__':
    try:
        # Initialize
        create_directory(PATHS['image_dir'])
        
        # Initialize enhanced face detector
        face_detector = EnhancedFaceDetector()
        
        # Display detection status
        logger.info("🎯 Enhanced Training Data Collection System")
        logger.info(f"Detection method: {face_detector.detection_method.upper()}")
        if face_detector.dnn_net is not None:
            logger.info("✅ Using DNN detection for high-quality training data")
        else:
            logger.info("✅ Using Haar cascade detection for training data")
            
        cam = initialize_camera(CAMERA['index'])
        if cam is None:
            raise ValueError("Failed to initialize camera")
            
        # Get user info
        face_name = input('\nEnter user name and press <return> -->  ').strip()
        if not face_name:
            raise ValueError("Name cannot be empty")
            
        face_id = get_face_id(PATHS['image_dir'])
        save_name(face_id, face_name, PATHS['names_file'])
        
        logger.info(f"🚀 Initializing enhanced face capture for {face_name} (ID: {face_id})")
        logger.info("💡 Look at the camera from different angles for better training data")
        logger.info("📸 System will automatically capture high-quality face images")
        
        count = 0
        quality_threshold = 50  # Minimum face size for good quality
        last_capture_time = 0
        capture_interval = 0.3  # Minimum time between captures
        
        while True:
            ret, img = cam.read()
            if not ret:
                logger.warning("Failed to grab frame")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Use enhanced face detection
            faces = face_detector.detect_faces(img, gray)
            
            current_time = time.time()
            
            # Debug information (show every 30 frames to avoid spam)
            if count == 0 and int(current_time) % 2 == 0:  # Every 2 seconds when starting
                if len(faces) == 0:
                    logger.info(f"🔍 No faces detected. Make sure you're facing the camera well-lit area.")
                else:
                    logger.info(f"👤 Detected {len(faces)} face(s)")
            
            for (x, y, w, h) in faces:
                if count < TRAINING['samples_needed']:
                    # Quality check: ensure face is large enough
                    face_size = min(w, h)
                    
                    if face_size >= quality_threshold and (current_time - last_capture_time) >= capture_interval:
                        # Color coding for quality
                        if face_size >= 80:
                            rect_color = (0, 255, 0)  # Green for excellent quality
                            quality_text = "EXCELLENT"
                        elif face_size >= 60:
                            rect_color = (0, 255, 255)  # Yellow for good quality
                            quality_text = "GOOD"
                        else:
                            rect_color = (255, 0, 0)  # Blue for acceptable quality
                            quality_text = "ACCEPTABLE"
                        
                        cv2.rectangle(img, (x, y), (x+w, y+h), rect_color, 2)
                        
                        # Extract and preprocess face
                        face_img = gray[y:y+h, x:x+w]
                        face_img = preprocess_face_for_training(face_img)
                        
                        # Save the processed face image (legacy format)
                        img_path = f'./{PATHS["image_dir"]}/Users-{face_id}-{count+1}.jpg'
                        cv2.imwrite(img_path, face_img)
                        
                        # Also save in ArcFace format for advanced training
                        save_face_for_arcface(face_img, face_id, count+1)
                        
                        count += 1
                        last_capture_time = current_time
                        
                        # Display quality and progress info
                        cv2.putText(img, quality_text, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)
                        
                        logger.info(f"📸 Captured image {count}/{TRAINING['samples_needed']} - Quality: {quality_text}")
                    else:
                        # Face too small or too soon after last capture
                        cv2.rectangle(img, (x, y), (x+w, y+h), (128, 128, 128), 1)
                        cv2.putText(img, "TOO SMALL/WAIT", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                else:
                    break
            
            # Display progress and instructions
            progress = f"Captured: {count}/{TRAINING['samples_needed']}"
            cv2.putText(img, progress, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            instructions = f"Method: {face_detector.detection_method.upper()} | Move face for variety"
            cv2.putText(img, instructions, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show face detection status
            if len(faces) == 0:
                status_text = "No face detected - Look at camera!"
                cv2.putText(img, status_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                status_text = f"Faces detected: {len(faces)}"
                cv2.putText(img, status_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Enhanced Face Capture', img)
            
            if cv2.waitKey(100) & 0xff == 27:  # ESC key
                break
            if count >= TRAINING['samples_needed']:
                break
                
        logger.info(f"✅ Successfully captured {count} high-quality images with preprocessing")
        logger.info("🎯 Training data is optimized for better recognition accuracy")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        
    finally:
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()
