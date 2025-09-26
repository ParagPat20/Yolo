# Advanced Person Tracking and Face Recognition System
# Using state-of-the-art models: YOLOv8 for person detection, YOLO-Face for face detection,
# ArcFace for face recognition, and ByteTrack for person tracking
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
import pickle
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque

from settings.settings import (
    CAMERA, PERSON_DETECTION, FACE_DETECTION, FACE_RECOGNITION,
    PERSON_TRACKING, FACE_TRACKING, SECURITY, PATHS, CCTV, AUDIO, HARDWARE
)


# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Raspberry Pi camera imports
try:
    from picamera2 import Picamera2, Preview
    PICAMERA2_AVAILABLE = True
    logger.info("📷 Picamera2 available for Raspberry Pi")
except ImportError:
    PICAMERA2_AVAILABLE = False
    logger.warning("📷 Picamera2 not available, falling back to OpenCV")

# Try to import hardware interface
try:
    from hardware_interface import get_hardware_manager
    HARDWARE_AVAILABLE = True
    logger.info("🔧 Hardware interface available")
except ImportError:
    HARDWARE_AVAILABLE = False
    logger.warning("🔧 Hardware interface not available")

# Try to import sound libraries
try:
    import winsound
    import subprocess
    SOUND_AVAILABLE = True
    logger.info("🔊 Windows sound support available")
except ImportError:
    SOUND_AVAILABLE = False
    logger.warning("🔇 Sound support not available")

# Check if Windows Speech API is available
try:
    import win32com.client
    VOICE_AVAILABLE = True
    logger.info("🗣️ Windows Speech API available")
except ImportError:
    try:
        # Fallback to PowerShell speech
        subprocess.run(['powershell', '-Command', 'Add-Type -AssemblyName System.Speech'], 
                      capture_output=True, timeout=5)
        VOICE_AVAILABLE = True
        logger.info("🗣️ PowerShell speech available")
    except:
        VOICE_AVAILABLE = False
        logger.warning("🔇 Voice support not available")

@dataclass
class PersonTrack:
    """Data class for person tracking information with verification states"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[float, float]
    confidence: float
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_embedding: Optional[np.ndarray] = None
    identity: str = "Unknown"
    identity_confidence: float = 0.0
    last_seen: float = 0.0
    frames_since_recognition: int = 0
    is_known: bool = False
    alert_sent: bool = False
    
    # New verification states
    verification_requested: bool = False  # Waiting for face verification
    verification_start_time: float = 0.0  # When verification was requested
    is_trusted: bool = False  # Person is trusted (verified before)
    last_face_verification: float = 0.0  # Last time face was verified
    needs_face_check: bool = True  # New person needs face verification
    siren_played: bool = False  # Whether siren has been played for this person
    
    # Enhanced verification system
    verification_attempts: int = 0  # Number of failed verification attempts
    max_verification_attempts: int = CCTV['max_verification_attempts']  # Configurable attempts
    last_verification_attempt: float = 0.0  # Time of last verification attempt
    verification_attempt_cooldown: float = CCTV['verification_cooldown']  # Configurable cooldown
    verification_start_time: float = 0.0  # When verification process started
    unknown_timeout: float = CCTV['unknown_timeout']  # Time before marking as unknown
    verification_timeout: float = CCTV['verification_timeout']  # Time to wait for verification
    is_recording: bool = False  # Whether this person is being recorded
    recording_start_time: float = 0.0  # When recording started
    last_greeting_time: float = 0.0  # Last time this person was greeted

    # Guest Mode fields
    is_guest: bool = False  # Whether this person is a guest
    guest_associated_with: Optional[str] = None  # Name of the verified person they're associated with
    guest_mode_start_time: float = 0.0  # When guest mode started for this person
    trajectory_history: deque = None  # Store recent trajectory points for guest detection

class YOLOv8PersonDetector:
    """YOLOv8-based person detection"""
    
    def __init__(self):
        self.net = None
        self.input_size = PERSON_DETECTION['input_size']
        self.confidence_threshold = PERSON_DETECTION['confidence_threshold']
        self.nms_threshold = PERSON_DETECTION['nms_threshold']
        
        # Try to load YOLOv8 model
        model_path = PATHS['yolov8_person_model']
        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                logger.info("✅ YOLOv8 person detection model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLOv8 model: {e}")
                self._download_yolov8_model()
        else:
            self._download_yolov8_model()
    
    def _download_yolov8_model(self):
        """Download YOLOv8 model if not available"""
        logger.info("📥 YOLOv8 model not found. Please download yolov8n.onnx")
        logger.info("You can download it from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx")
        logger.info("Place it in the 'models' directory")
        
        # Create models directory
        os.makedirs(PATHS['models_dir'], exist_ok=True)
        
        # For now, we'll fall back to a simple person detection using existing DNN
        logger.info("🔄 Falling back to basic person detection")
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect persons in the frame"""
        if self.net is None:
            return self._fallback_person_detection(frame)
        
        try:
            # Prepare input
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (self.input_size, self.input_size), 
                swapRB=True, crop=False
            )
            self.net.setInput(blob)
            outputs = self.net.forward()
            
            persons = []
            h, w = frame.shape[:2]
            
            # Parse YOLOv8 output
            for detection in outputs[0].T:
                scores = detection[4:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only process person class (class 0 in COCO)
                if class_id == 0 and confidence > self.confidence_threshold:
                    # Convert from center format to corner format
                    cx, cy, width, height = detection[:4]
                    cx *= w
                    cy *= h
                    width *= w
                    height *= h
                    
                    x = int(cx - width / 2)
                    y = int(cy - height / 2)
                    w_box = int(width)
                    h_box = int(height)
                    
                    persons.append((x, y, w_box, h_box, float(confidence)))
            
            # Apply NMS
            if persons:
                boxes = [(x, y, w, h) for x, y, w, h, _ in persons]
                confidences = [conf for _, _, _, _, conf in persons]
                
                indices = cv2.dnn.NMSBoxes(
                    boxes, confidences, 
                    self.confidence_threshold, self.nms_threshold
                )
                
                if len(indices) > 0:
                    return [persons[i] for i in indices.flatten()]
            
            return persons
            
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            return []
    
    def _fallback_person_detection(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Fallback person detection using HOG"""
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            persons, weights = hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(32, 32), scale=1.05
            )
            
            result = []
            for i, (x, y, w, h) in enumerate(persons):
                confidence = float(weights[i]) if i < len(weights) else 0.5
                result.append((x, y, w, h, confidence))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fallback person detection: {e}")
            return []

class SCRFDFaceDetector:
    """SCRFD-based face detection with robust ONNX parsing and fallbacks"""
    
    def __init__(self):
        self.net = None
        # Reuse generic face confidence threshold from settings
        self.confidence_threshold = FACE_DETECTION.get('yolo_confidence_threshold', 0.6)
        self.input_size = (640, 640)
        
        model_path = PATHS.get('yolo_face_model', '')  # points to scrfd_10g_bnkps.onnx
        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                # quick forward to verify
                test_frame = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
                blob = cv2.dnn.blobFromImage(test_frame, 1.0, self.input_size, mean=(0,0,0), swapRB=True, crop=False)
                self.net.setInput(blob)
                _ = self.net.forward(self._get_output_names()) if hasattr(self.net, 'getUnconnectedOutLayersNames') else self.net.forward()
                logger.info("✅ SCRFD face detection model loaded and tested successfully")
            except Exception as e:
                logger.warning(f"⚠️  SCRFD model failed to load/test: {str(e)[:120]}...")
                logger.info("📥 Falling back to DNN/Haar face detection")
                self.net = None
        else:
            logger.info("📥 SCRFD model not found, using fallback detection")
    
    def _get_output_names(self):
        try:
            return self.net.getUnconnectedOutLayersNames()
        except Exception:
            return None
    
    def detect_faces(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int] = None) -> List[Tuple[int, int, int, int, float]]:
        if person_bbox:
            x, y, w, h = person_bbox
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            roi = frame[y:y+h, x:x+w]
            faces = self._detect_faces_in_roi(roi)
            adjusted = [(fx + x, fy + y, fw, fh, conf) for fx, fy, fw, fh, conf in faces]
            return adjusted
        return self._detect_faces_in_roi(frame)
    
    def _detect_faces_in_roi(self, roi: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        if self.net is not None:
            faces = self._scrfd_infer(roi)
            if faces:
                return faces
        return self._fallback_face_detection(roi)
    
    def _scrfd_infer(self, roi: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        try:
            h, w = roi.shape[:2]
            blob = cv2.dnn.blobFromImage(roi, 1.0, self.input_size, mean=(0,0,0), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = None
            try:
                outputs = self.net.forward(self._get_output_names())
            except Exception:
                outputs = self.net.forward()
            faces = []
            # Try common SCRFD output parsing patterns
            # Pattern A: single Nx5 array [x1,y1,x2,y2,score] normalized
            for out in (outputs if isinstance(outputs, (list, tuple)) else [outputs]):
                arr = np.squeeze(out)
                if arr.ndim == 2 and arr.shape[1] in (5, 6):
                    for det in arr:
                        if arr.shape[1] == 6:
                            x1, y1, x2, y2, score, _ = det
                        else:
                            x1, y1, x2, y2, score = det
                        score = float(score)
                        if score < self.confidence_threshold:
                            continue
                        # coords may be normalized [0,1] or absolute to input_size; handle both
                        if max(x1, y1, x2, y2) <= 1.5:
                            x1 *= w; y1 *= h; x2 *= w; y2 *= h
                        else:
                            # if absolute to model size, scale to roi
                            scale_x = w / self.input_size[0]
                            scale_y = h / self.input_size[1]
                            x1 *= scale_x; x2 *= scale_x; y1 *= scale_y; y2 *= scale_y
                        x = int(max(0, min(x1, x2)))
                        y = int(max(0, min(y1, y2)))
                        fw = int(max(0, abs(x2 - x1)))
                        fh = int(max(0, abs(y2 - y1)))
                        if fw > 0 and fh > 0:
                            faces.append((x, y, fw, fh, score))
            # If nothing parsed, return empty to trigger fallback
            return faces
        except Exception as e:
            logger.debug(f"SCRFD inference parse error: {e}")
            return []
    
    def _fallback_face_detection(self, roi: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        try:
            dnn_faces = self._try_dnn_face_detection(roi)
            if dnn_faces:
                return dnn_faces
            cascade_path = PATHS['cascade_file']
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5,
                    minSize=FACE_DETECTION['min_size']
                )
                return [(x, y, w, h, 0.8) for x, y, w, h in faces]
            return []
        except Exception as e:
            logger.error(f"Error in fallback face detection: {e}")
            return []
    
    def _try_dnn_face_detection(self, roi: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        try:
            dnn_model_path = PATHS.get('dnn_model', '')
            dnn_config_path = PATHS.get('dnn_config', '')
            if os.path.exists(dnn_model_path) and os.path.exists(dnn_config_path):
                net = cv2.dnn.readNetFromTensorflow(dnn_model_path, dnn_config_path)
                h, w = roi.shape[:2]
                blob = cv2.dnn.blobFromImage(roi, 1.0, (300, 300), [104, 117, 123])
                net.setInput(blob)
                detections = net.forward()
                faces = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > FACE_DETECTION['dnn_confidence_threshold']:
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        faces.append([x1, y1, x2 - x1, y2 - y1, float(confidence)])
                return faces
        except Exception:
            pass
        return []

class ArcFaceRecognizer:
    """ArcFace-based face recognition"""
    
    def __init__(self):
        self.net = None
        self.known_embeddings = {}
        self.input_size = FACE_RECOGNITION['input_size']
        self.embedding_size = FACE_RECOGNITION['embedding_size']
        self.confidence_threshold = FACE_RECOGNITION['confidence_threshold']
        
        # Try to load ArcFace model
        model_path = PATHS.get('arcface_model', '')
        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                logger.info("✅ ArcFace recognition model loaded")
            except Exception as e:
                logger.error(f"Failed to load ArcFace model: {e}")
        else:
            logger.info("📥 ArcFace model not found, using fallback LBPH recognition")
            self._init_fallback_recognizer()
        
        # Load known face embeddings
        self._load_known_embeddings()
    
    def _init_fallback_recognizer(self):
        """Initialize fallback LBPH recognizer"""
        try:
            self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
            trainer_path = PATHS['trainer_file']
            if os.path.exists(trainer_path):
                self.lbph_recognizer.read(trainer_path)
                logger.info("✅ LBPH fallback recognizer loaded")
        except Exception as e:
            logger.error(f"Failed to load LBPH recognizer: {e}")
            self.lbph_recognizer = None
    
    def _load_known_embeddings(self):
        """Load known face embeddings from file"""
        embeddings_path = PATHS.get('face_embeddings', '')
        if os.path.exists(embeddings_path):
            try:
                with open(embeddings_path, 'rb') as f:
                    self.known_embeddings = pickle.load(f)
                logger.info(f"✅ Loaded {len(self.known_embeddings)} known face embeddings")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
        
        # Also load names from JSON for fallback
        self.names = {}
        names_path = PATHS['names_file']
        if os.path.exists(names_path):
            try:
                with open(names_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self.names = json.loads(content)
            except Exception as e:
                logger.error(f"Failed to load names: {e}")
    
    def extract_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using ArcFace"""
        if self.net is None:
            return None
        
        try:
            # Preprocess face
            face_roi = cv2.resize(face_roi, self.input_size)
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB) if len(face_roi.shape) == 3 else cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            
            # Normalize
            face_roi = face_roi.astype(np.float32) / 255.0
            face_roi = (face_roi - 0.5) / 0.5
            
            # Create blob and get embedding
            blob = cv2.dnn.blobFromImage(face_roi, swapRB=True)
            self.net.setInput(blob)
            embedding = self.net.forward()
            
            # Normalize embedding
            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def recognize_face(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Recognize face and return identity with confidence"""
        if self.net is not None:
            return self._arcface_recognition(face_roi)
        else:
            return self._fallback_recognition(face_roi)
    
    def _arcface_recognition(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """ArcFace-based recognition with strict 0.5 confidence threshold"""
        embedding = self.extract_embedding(face_roi)
        if embedding is None:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_similarity = 0.0
        
        for name, known_embedding in self.known_embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(embedding, known_embedding)
            if similarity > best_similarity and similarity > self.confidence_threshold:
                best_similarity = similarity
                best_match = name
        
        # STRICT RULE: Confidence must be >= 0.5 to be considered known
        if best_similarity < 0.5:
            logger.debug(f"ArcFace confidence {best_similarity:.3f} < 0.5 threshold, treating as unknown")
            return "Unknown", best_similarity
        
        return best_match, best_similarity
    
    def _fallback_recognition(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Fallback LBPH recognition with strict 0.5 confidence threshold"""
        if not hasattr(self, 'lbph_recognizer') or self.lbph_recognizer is None:
            return "Unknown", 0.0
        
        try:
            # Preprocess for LBPH
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            gray = cv2.resize(gray, (100, 100))
            gray = cv2.equalizeHist(gray)
            
            id_pred, confidence = self.lbph_recognizer.predict(gray)
            
            # LBPH uses distance (lower is better), convert to similarity
            if confidence <= FACE_RECOGNITION['lbph_threshold']:
                name = self.names.get(str(id_pred), f"Person_{id_pred}")
                similarity = max(0, (100 - confidence) / 100)  # Convert distance to similarity
                
                # STRICT RULE: Confidence must be >= 0.5 to be considered known
                if similarity < 0.5:
                    logger.debug(f"LBPH confidence {similarity:.3f} < 0.5 threshold, treating as unknown")
                    return "Unknown", similarity
                
                return name, similarity
            else:
                return "Unknown", 0.0
                
        except Exception as e:
            logger.error(f"Error in LBPH recognition: {e}")
            return "Unknown", 0.0
    
    def add_known_face(self, name: str, face_roi: np.ndarray):
        """Add a new known face to the database"""
        if self.net is not None:
            embedding = self.extract_embedding(face_roi)
            if embedding is not None:
                self.known_embeddings[name] = embedding
                self._save_embeddings()
                logger.info(f"Added new face embedding for {name}")
    
    def _save_embeddings(self):
        """Save embeddings to file"""
        try:
            embeddings_path = PATHS.get('face_embeddings', '')
            os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.known_embeddings, f)
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")

class ByteTracker:
    """Improved ByteTrack implementation for person tracking"""
    
    def __init__(self, parent_tracker=None):
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = PERSON_TRACKING['max_disappeared']
        self.max_distance = PERSON_TRACKING['max_distance']
        self.frame_count = 0
        self.frames_without_detection = 0
        self.parent_tracker = parent_tracker
    
    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> List[PersonTrack]:
        """Update tracks with improved logic"""
        self.frame_count += 1
        current_time = time.time()
        
        # Track frames without detections for cleanup
        if len(detections) == 0:
            self.frames_without_detection += 1
        else:
            self.frames_without_detection = 0
        
        # If no detections for too long, clear all tracks
        if self.frames_without_detection > self.max_disappeared:
            if self.tracks:
                logger.info(f"🧹 Clearing {len(self.tracks)} stale tracks - no detections for {self.frames_without_detection} frames")
                self.tracks.clear()
            return []
        
        # Convert detections to potential tracks
        new_detections = []
        for x, y, w, h, conf in detections:
            center = (x + w/2, y + h/2)
            new_detections.append({
                'bbox': (x, y, w, h),
                'center': center,
                'confidence': conf
            })
        
        # If no detections this frame, just return existing tracks (but mark them as aging)
        if not new_detections:
            for track in self.tracks.values():
                track.frames_since_recognition += 1
            return list(self.tracks.values())
        
        # Association using improved matching
        matched_tracks = []
        unmatched_detections = list(range(len(new_detections)))
        
        if self.tracks:
            # Calculate cost matrix (distance + confidence)
            track_ids = list(self.tracks.keys())
            costs = np.full((len(track_ids), len(new_detections)), float('inf'))
            
            for i, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                for j, detection in enumerate(new_detections):
                    distance = self._calculate_distance(track.center, detection['center'])
                    
                    if distance < self.max_distance:
                        # Cost combines distance and confidence difference
                        confidence_diff = abs(track.confidence - detection['confidence'])
                        cost = distance + (confidence_diff * 50)  # Weight confidence difference
                        costs[i, j] = cost
            
            # Greedy assignment (could use Hungarian algorithm for optimal)
            used_detections = set()
            for i, track_id in enumerate(track_ids):
                if len(unmatched_detections) == 0:
                    break
                
                # Find best available detection for this track
                best_cost = float('inf')
                best_detection_idx = -1
                
                for j in unmatched_detections:
                    if costs[i, j] < best_cost:
                        best_cost = costs[i, j]
                        best_detection_idx = j
                
                if best_detection_idx != -1 and best_cost < self.max_distance:
                    # Match found
                    detection = new_detections[best_detection_idx]
                    track = self.tracks[track_id]
                    
                    # Smooth position update
                    old_center = track.center
                    new_center = detection['center']
                    smoothed_center = (
                        (old_center[0] * 0.3 + new_center[0] * 0.7),
                        (old_center[1] * 0.3 + new_center[1] * 0.7)
                    )
                    
                    track.bbox = detection['bbox']
                    track.center = smoothed_center
                    track.confidence = detection['confidence']
                    track.last_seen = current_time
                    track.frames_since_recognition += 1

                    # Update trajectory history for guest detection
                    if track.trajectory_history is not None:
                        track.trajectory_history.append(smoothed_center)
                    
                    matched_tracks.append(track_id)
                    unmatched_detections.remove(best_detection_idx)
        
        # Create new tracks for unmatched detections (but be conservative)
        for detection_idx in unmatched_detections:
            detection = new_detections[detection_idx]
            
            # Only create new track if confidence is high enough
            if detection['confidence'] > 0.6:
                new_track = PersonTrack(
                    track_id=self.next_id,
                    bbox=detection['bbox'],
                    center=detection['center'],
                    confidence=detection['confidence'],
                    last_seen=current_time,
                    frames_since_recognition=0
                )

                # Initialize trajectory history for guest detection
                new_track.trajectory_history = deque(maxlen=10)  # Store last 10 trajectory points

                # Check if there's a recently trusted person via parent tracker
                if self.parent_tracker:
                    recently_trusted = self.parent_tracker._check_recently_trusted_for_new_track(current_time)
                    if recently_trusted:
                        # Give benefit of doubt - assume it's the same trusted person
                        new_track.needs_face_check = False
                        new_track.is_trusted = True
                        new_track.is_known = True
                        new_track.identity = recently_trusted
                        new_track.last_face_verification = current_time
                        logger.info(f"🆕 Created new track {self.next_id} for continuing trusted person {recently_trusted}")
                    else:
                        # New unknown person - will need verification
                        new_track.needs_face_check = True
                        new_track.verification_requested = False
                        logger.info(f"🆕 Created new track {self.next_id} with confidence {detection['confidence']:.2f}")
                else:
                    # New unknown person - will need verification  
                    new_track.needs_face_check = True
                    new_track.verification_requested = False
                    logger.info(f"🆕 Created new track {self.next_id} with confidence {detection['confidence']:.2f}")
                
                self.tracks[self.next_id] = new_track
                self.next_id += 1
        
        # Remove tracks that haven't been matched for too long
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                frames_since_seen = (current_time - track.last_seen) * 30  # Approximate frames
                if frames_since_seen > self.max_disappeared:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            track_name = self.tracks[track_id].identity if self.tracks[track_id].identity != "Unknown" else f"Track {track_id}"
            del self.tracks[track_id]
            logger.info(f"🗑️ Removed track {track_id} ({track_name}) - not seen for too long")
        
        return list(self.tracks.values())
    
    def _calculate_distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two centers"""
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

class WindowsSoundAlertSystem:
    """Windows-native sound and voice alert system with threading"""
    
    def __init__(self):
        self.sound_enabled = SECURITY.get('alert_sound', False) and SOUND_AVAILABLE
        self.voice_enabled = SECURITY.get('voice_alerts', False) and VOICE_AVAILABLE
        self.last_voice_time = 0.0  # Cooldown for voice alerts
        self.use_win32_speech = False
        
        # Threading control
        self.sound_threads = []  # Keep track of active sound threads
        self.max_concurrent_sounds = 3  # Limit concurrent sounds
        self._cleanup_threads()  # Clean up any old threads
        
        # Initialize Windows Speech API
        if self.voice_enabled:
            try:
                import win32com.client
                self.speech_engine = win32com.client.Dispatch("SAPI.SpVoice")
                self.use_win32_speech = True
                logger.info("🗣️ Windows Speech API initialized")
            except Exception as e:
                logger.info(f"Windows Speech API not available, using PowerShell: {e}")
                self.use_win32_speech = False
        
        if self.sound_enabled:
            logger.info("🔊 Windows sound system ready")
    
    def _cleanup_threads(self):
        """Clean up finished sound threads"""
        self.sound_threads = [t for t in self.sound_threads if t.is_alive()]
    
    def _start_sound_thread(self, target_func, *args, **kwargs):
        """Start a sound operation in a separate thread"""
        self._cleanup_threads()
        
        # Limit concurrent sounds to prevent system overload
        if len(self.sound_threads) >= self.max_concurrent_sounds:
            logger.warning(f"🔊 Too many concurrent sounds ({len(self.sound_threads)}), skipping")
            return
        
        thread = threading.Thread(target=target_func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        self.sound_threads.append(thread)
        logger.debug(f"🔊 Started sound thread, active threads: {len(self.sound_threads)}")
    
    def _play_windows_beep(self):
        """Play Windows system beep"""
        try:
            winsound.Beep(800, 200)  # 800Hz for 200ms
        except Exception as e:
            logger.error(f"Failed to play beep: {e}")
    
    def _play_windows_siren(self):
        """Play siren using Windows sounds"""
        try:
            # Play multiple beeps to simulate siren
            for i in range(6):  # 3 seconds worth
                freq = 800 if i % 2 == 0 else 1200
                winsound.Beep(freq, 250)  # 250ms each
        except Exception as e:
            logger.error(f"Failed to play siren: {e}")
    
    def _speak_text(self, text: str):
        """Speak text using Windows TTS"""
        try:
            if self.use_win32_speech:
                # Use Windows Speech API
                self.speech_engine.Speak(text)
            else:
                # Use PowerShell as fallback
                ps_command = f'Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Speak("{text}")'
                subprocess.run(['powershell', '-Command', ps_command], 
                             capture_output=True, timeout=10)
        except Exception as e:
            logger.error(f"Failed to speak text: {e}")
    
    def play_verification_request(self):
        """Play sound and voice alert for verification request"""
        current_time = time.time()
        
        if self.sound_enabled:
            self._start_sound_thread(self._play_windows_beep)
        
        # Voice cooldown of 3 seconds to prevent spam
        if self.voice_enabled and (current_time - self.last_voice_time > 3.0):
            message = "Person detected. Please verify your face."
            self._start_sound_thread(self._speak_text, message)
            self.last_voice_time = current_time
    
    def play_verification_attempt(self, attempt: int, max_attempts: int):
        """Play voice alert for verification attempt"""
        current_time = time.time()
        
        # Voice cooldown of 2 seconds for attempt messages
        if self.voice_enabled and (current_time - self.last_voice_time > 2.0):
            remaining = max_attempts - attempt
            if remaining > 0:
                message = f"{attempt}"
            else:
                message = "Final verification attempt failed."
            self._start_sound_thread(self._speak_text, message)
            self.last_voice_time = current_time
    
    def play_verification_reminder(self):
        """Play reminder for face verification"""
        current_time = time.time()
        
        # Voice cooldown of 4 seconds for reminders
        if self.voice_enabled and (current_time - self.last_voice_time > 4.0):
            message = "Please verify."
            self._start_sound_thread(self._speak_text, message)
            self.last_voice_time = current_time
    
    def play_unknown_alert(self):
        """Play siren and voice alert for unknown person"""
        logger.info("🔊 play_unknown_alert called")
        
        if self.sound_enabled:
            logger.info("🚨 Playing Windows siren...")
            self._start_sound_thread(self._play_windows_siren)
        else:
            logger.info("🔇 Sound is disabled")
        
        # Critical alert - bypass cooldown
        if self.voice_enabled:
            logger.info("🗣️ Attempting to play voice alert...")
            message = "Alert! Unknown face detected!"
            self._start_sound_thread(self._speak_text, message)
            self.last_voice_time = time.time()
            logger.info("✅ Voice alert started in background")
        else:
            logger.info("🔇 Voice is disabled")
    
    def play_welcome_back(self, name: str):
        """Play welcome message for known person"""
        if self.voice_enabled:
            message = f"Welcome back, {name}!"
            self._start_sound_thread(self._speak_text, message)
    
    def stop_all_sounds(self):
        """Stop all playing sounds and clean up threads"""
        if self.use_win32_speech and self.voice_enabled:
            try:
                self.speech_engine.Speak("", 1)  # Stop current speech
            except Exception as e:
                logger.error(f"Failed to stop speech: {e}")
        
        # Clean up finished threads
        self._cleanup_threads()
        logger.info(f"🔇 Stopped all sounds, active threads: {len(self.sound_threads)}")

class AdvancedPersonTracker:
    """Main class that integrates all components"""
    
    def __init__(self):
        logger.info("🚀 Initializing Advanced Person Tracking System...")

        # Initialize hardware interface
        self.hardware_manager = get_hardware_manager() if HARDWARE_AVAILABLE else None

        # Initialize components
        self.person_detector = YOLOv8PersonDetector()
        self.face_detector = SCRFDFaceDetector()
        self.face_recognizer = ArcFaceRecognizer()
        self.tracker = ByteTracker(parent_tracker=self)
        self.sound_system = WindowsSoundAlertSystem()

        # Smart verification management
        self.last_unknown_alert = {}
        self.unknown_face_counter = 0
        self.trusted_persons = {}  # Store trusted person info {name: last_seen_time}
        self.global_trusted_memory = {}  # Global memory of recently verified persons
        self.greeting_history = {}  # Track recent greetings to avoid spam

        # Recording management
        self.recording_active = {}  # Track active recordings {track_id: video_writer}

        # Create directories
        os.makedirs(PATHS.get('unknown_faces_dir', 'unknown_faces'), exist_ok=True)
        os.makedirs(PATHS.get('models_dir', 'models'), exist_ok=True)
        os.makedirs('recordings', exist_ok=True)  # For video recordings

        # Set initial status
        if self.hardware_manager:
            self.hardware_manager.set_system_status('ready')

        logger.info("✅ Advanced Person Tracking System with CCTV Integration initialized")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[PersonTrack]]:
        """Process a single frame and return annotated frame with tracks - enhanced for CCTV"""
        # Keep a reference to the latest frame for saving tasks
        self._last_frame = frame
        # Detect persons
        person_detections = self.person_detector.detect_persons(frame)

        # Update person tracking
        person_tracks = self.tracker.update(person_detections)

        # Process each tracked person
        for track in person_tracks:
            self._process_person_track(frame, track)

        # Check for guest mode activation
        if CCTV['guest_mode_enabled']:
            self._check_and_activate_guest_mode(person_tracks)

        # Handle active recordings
        self._update_recordings(frame, person_tracks)

        # Clean up old trusted memory entries
        self._cleanup_trusted_memory()

        # Check for guest mode timeout and reversion
        if CCTV['guest_mode_enabled']:
            self._check_guest_mode_timeout(person_tracks)

        # Draw annotations
        annotated_frame = self._draw_annotations(frame, person_tracks)

        return annotated_frame, person_tracks

    def _check_and_activate_guest_mode(self, tracks: List[PersonTrack]):
        """Check if guest mode should be activated based on person context"""
        current_time = time.time()

        # Find verified and unknown persons
        verified_persons = []
        unknown_persons = []

        for track in tracks:
            if track.is_trusted and track.is_known:
                verified_persons.append(track)
            elif not track.is_known and not track.is_guest and not track.verification_requested:
                # Only consider persons who are fully unknown (not in verification process)
                unknown_persons.append(track)

        # Check each unknown person to see if they're with a verified person
        for unknown_track in unknown_persons:
            for verified_track in verified_persons:
                if self._should_be_guest(unknown_track, verified_track, current_time):
                    self._activate_guest_mode(unknown_track, verified_track.identity, current_time)
                    logger.info(f"👥 Guest mode activated: {verified_track.identity} + guest (track {unknown_track.track_id})")
                    break

    def _should_be_guest(self, unknown_track: PersonTrack, verified_track: PersonTrack, current_time: float) -> bool:
        """Determine if an unknown person should be treated as a guest"""
        # Check distance proximity
        distance = self._calculate_distance(unknown_track.center, verified_track.center)
        if distance > CCTV['guest_detection_distance']:
            return False

        # Check trajectory similarity (if both have enough trajectory points)
        if (unknown_track.trajectory_history and verified_track.trajectory_history and
            len(unknown_track.trajectory_history) >= 3 and len(verified_track.trajectory_history) >= 3):

            similarity = self._calculate_trajectory_similarity(unknown_track.trajectory_history,
                                                             verified_track.trajectory_history)
            if similarity < CCTV['guest_trajectory_similarity']:
                return False

        # Additional checks: both should be moving in similar directions
        # and have similar confidence levels (not one person and one object)
        if abs(unknown_track.confidence - verified_track.confidence) > 0.3:
            return False

        return True

    def _calculate_trajectory_similarity(self, traj1: deque, traj2: deque) -> float:
        """Calculate similarity between two trajectories"""
        if len(traj1) != len(traj2) or len(traj1) < 2:
            return 0.0

        # Calculate direction vectors for each trajectory
        def get_direction_vector(traj):
            if len(traj) < 2:
                return (0, 0)

            # Get the most recent movement (last 3 points for stability)
            recent_points = list(traj)[-3:]
            if len(recent_points) < 2:
                return (0, 0)

            # Calculate average direction vector
            dx = sum(p2[0] - p1[0] for p1, p2 in zip(recent_points[:-1], recent_points[1:]))
            dy = sum(p2[1] - p1[1] for p1, p2 in zip(recent_points[:-1], recent_points[1:]))
            return (dx, dy)

        vec1 = get_direction_vector(traj1)
        vec2 = get_direction_vector(traj2)

        # Calculate cosine similarity
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        norm1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
        norm2 = math.sqrt(vec2[0]**2 + vec2[1]**2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _activate_guest_mode(self, guest_track: PersonTrack, host_name: str, current_time: float):
        """Activate guest mode for a person"""
        guest_track.is_guest = True
        guest_track.guest_associated_with = host_name
        guest_track.guest_mode_start_time = current_time

        # Stop any recording that might have started
        if guest_track.is_recording:
            self._stop_recording(guest_track.track_id)
            guest_track.is_recording = False

        # Announce guest mode activation
        if self.hardware_manager:
            self.hardware_manager.activate_guest_mode(host_name)
        elif self.sound_system and self.sound_system.voice_enabled:
            message = f"Welcome back, {host_name}. {AUDIO['guest_mode_message']}"
            self.sound_system._speak_text(message)

        logger.info(f"👥 Guest mode activated for track {guest_track.track_id} with host {host_name}")

    def _check_guest_mode_timeout(self, tracks: List[PersonTrack]):
        """Check if guest mode should timeout and handle reversion"""
        current_time = time.time()

        # Check for guest mode timeout
        guests_to_revert = []
        for track in tracks:
            if track.is_guest and (current_time - track.guest_mode_start_time > CCTV['guest_mode_duration']):
                guests_to_revert.append(track)

        # Revert guests who have timed out
        for guest_track in guests_to_revert:
            self._revert_guest_mode(guest_track, current_time)

        # Check if we need to revert due to host leaving
        self._check_host_departure(tracks, current_time)

    def _revert_guest_mode(self, guest_track: PersonTrack, current_time: float):
        """Revert a guest back to normal security monitoring"""
        logger.info(f"⏰ Guest mode expired for track {guest_track.track_id} (associated with {guest_track.guest_associated_with})")

        # Mark as no longer guest
        guest_track.is_guest = False
        guest_track.guest_associated_with = None
        guest_track.guest_mode_start_time = 0.0

        # Announce reversion
        if self.hardware_manager:
            self.hardware_manager.revert_guest_mode()
        elif self.sound_system and self.sound_system.voice_enabled:
            self.sound_system._speak_text(AUDIO['guest_mode_reverted'])

        # If this guest is still unknown, start monitoring them normally
        if not guest_track.is_known:
            logger.info(f"👀 Now monitoring former guest (track {guest_track.track_id}) as potential unknown person")

    def _check_host_departure(self, tracks: List[PersonTrack], current_time: float):
        """Check if verified host has left and revert guests accordingly"""
        # Get all current hosts (verified trusted persons)
        current_hosts = set()
        for track in tracks:
            if track.is_trusted and track.is_known:
                current_hosts.add(track.identity)

        # Check each guest to see if their host is still present
        guests_to_revert = []
        for track in tracks:
            if track.is_guest and track.guest_associated_with:
                if track.guest_associated_with not in current_hosts:
                    # Host has left, check if guest should be reverted
                    time_since_host_left = current_time - track.last_seen
                    if time_since_host_left > 10.0:  # Give 10 seconds grace period
                        guests_to_revert.append(track)

        # Revert guests whose hosts have left
        for guest_track in guests_to_revert:
            logger.info(f"🚶 Host {guest_track.guest_associated_with} has left - reverting guest (track {guest_track.track_id})")
            self._revert_guest_mode(guest_track, current_time)

    def _update_recordings(self, frame: np.ndarray, tracks: List[PersonTrack]):
        """Update active recordings for unknown persons"""
        current_time = time.time()

        for track in tracks:
            if track.is_recording:
                # Check if recording should be stopped
                recording_duration = current_time - track.recording_start_time
                if recording_duration > CCTV['recording_duration']:
                    self._stop_recording(track.track_id)
                    track.is_recording = False
                else:
                    # Continue recording
                    self._write_frame_to_recording(track.track_id, frame)
    
    def _is_face_good_quality(self, face_roi: np.ndarray) -> bool:
        """Check if face ROI has good enough quality for recognition"""
        try:
            h, w = face_roi.shape[:2]
            
            # Check size - face should be reasonably large
            if min(h, w) < 60:
                return False
            
            # Check brightness - face shouldn't be too dark
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 50:  # Too dark
                return False
            if mean_brightness > 250:  # Too bright/washed out
                return False
            
            # Check contrast - face should have some detail
            contrast = np.std(gray)
            if contrast < 15:  # Too flat/low contrast
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking face quality: {e}")
            return False
    
    def _process_person_track(self, frame: np.ndarray, track: PersonTrack):
        """Process person track with smart verification logic"""
        current_time = time.time()
        
        # Check if multiple people are present - if so, be more strict
        active_tracks = len([t for t in self.tracker.tracks.values() if current_time - t.last_seen < 2.0])
        multiple_people_present = active_tracks > 1

        # Handle guests - they don't need verification and shouldn't trigger alerts
        if track.is_guest:
            # Guests don't need face verification and shouldn't be treated as unknown
            return

        # Check if this person is already trusted (known without needing face verification)
        if track.is_trusted and not self._needs_reverification(track, current_time):
            # If multiple people are present, require more frequent re-verification
            if multiple_people_present:
                time_since_verification = current_time - track.last_face_verification
                if time_since_verification > 30.0:  # Re-verify every 30 seconds when multiple people
                    track.needs_face_check = True
                    track.is_trusted = False
                    logger.info(f"🔄 Person {track.track_id} needs re-verification (multiple people present)")
                else:
                    return
            else:
                # Single person - normal trusted behavior
                return
        
        # Check if we should attempt face recognition
        should_recognize = (
            track.frames_since_recognition >= FACE_TRACKING['face_recognition_interval'] or
            track.needs_face_check or
            track.verification_requested
        )
        
        if should_recognize:
            # Detect face within person bounding box
            faces = self.face_detector.detect_faces(frame, track.bbox)
            
            if faces:
                # Use the largest/most confident face
                best_face = max(faces, key=lambda f: f[4])  # Sort by confidence
                fx, fy, fw, fh, face_conf = best_face
                
                # Check if face is large enough and has good quality
                if min(fw, fh) >= FACE_TRACKING['min_face_size']:
                    track.face_bbox = (fx, fy, fw, fh)
                    
                    # Extract face ROI
                    face_roi = frame[fy:fy+fh, fx:fx+fw]
                    
                    # Check if face ROI is good quality (not too dark, not too small)
                    if self._is_face_good_quality(face_roi):
                        # Recognize face
                        identity, confidence = self.face_recognizer.recognize_face(face_roi)
                        
                        # Process recognition result
                        self._handle_face_recognition_result(track, identity, confidence, face_roi, current_time)
                    else:
                        # Face quality is poor, don't treat as "verified unknown"
                        logger.debug(f"Poor face quality for track {track.track_id}, skipping recognition")
                        # Continue waiting for better face quality
            else:
                # No face detected - handle based on current state
                self._handle_no_face_detected(track, current_time)
    
    def _handle_face_recognition_result(self, track: PersonTrack, identity: str, confidence: float, face_roi: np.ndarray, current_time: float):
        """Handle the result of face recognition with global trusted memory"""
        track.identity = identity
        track.identity_confidence = confidence
        track.frames_since_recognition = 0
        track.last_face_verification = current_time
        track.needs_face_check = False
        track.verification_requested = False
        
        if identity != "Unknown":
            # Known person identified - reset verification attempts
            track.is_known = True
            track.is_trusted = True
            track.verification_attempts = 0  # Reset failed attempts counter
            self.trusted_persons[identity] = current_time
            self.global_trusted_memory[identity] = current_time  # Update global memory

            # Clear any previous alerts
            track.siren_played = False  # Clear any previous unknown alerts
            track.alert_sent = False   # Reset alert state for proper welcome

            # Stop recording if it was active
            if track.is_recording:
                self._stop_recording(track.track_id)

            # Set status to ready (green LED)
            if self.hardware_manager:
                self.hardware_manager.set_system_status('ready')

            # Check if this is the first person today for time-based greeting
            current_time_dt = datetime.now()
            today_start = datetime(current_time_dt.year, current_time_dt.month, current_time_dt.day).timestamp()

            if current_time - track.last_greeting_time > 3600:  # Greet at most once per hour
                # Use time-based greeting for first identification of the day
                if track.last_greeting_time < today_start:
                    self._greet_person(track, current_time)
                else:
                    # Welcome back for subsequent identifications
                    self._welcome_back_person(track, current_time)

                track.last_greeting_time = current_time

            logger.info(f"✅ Person {track.track_id} identified as {identity} (confidence: {confidence:.2f})")

        else:
            # TRIPLE VERIFICATION: Check cooldown before incrementing attempts
            time_since_last_attempt = current_time - track.last_verification_attempt
            
            if time_since_last_attempt >= track.verification_attempt_cooldown:
                # Cooldown period passed, increment attempt counter
                track.verification_attempts += 1
                track.last_verification_attempt = current_time
                logger.info(f"👤 Person {track.track_id} verification attempt {track.verification_attempts}/{track.max_verification_attempts} - confidence {confidence:.3f} < 0.5")
                
                if track.verification_attempts >= track.max_verification_attempts:
                    # After 3 failed attempts, mark as unknown
                    logger.warning(f"🚨 Person {track.track_id} failed {track.max_verification_attempts} verification attempts - marking as UNKNOWN")
                    track.is_known = False
                    track.is_trusted = False
                    self._handle_unknown_person_verified(track, face_roi, current_time, frame)
                else:
                    # Still attempting verification - request face verification again
                    remaining_attempts = track.max_verification_attempts - track.verification_attempts
                    logger.info(f"🔍 Person {track.track_id} needs {remaining_attempts} more verification attempts (next attempt in {track.verification_attempt_cooldown}s)")
                    track.verification_requested = True
                    track.verification_start_time = current_time
                    track.needs_face_check = True
                    
                    # Play verification attempt voice message
                    if SECURITY.get('voice_alerts', False):
                        self.sound_system.play_verification_attempt(track.verification_attempts, track.max_verification_attempts)
            else:
                # Still in cooldown period - don't count as attempt yet
                remaining_cooldown = track.verification_attempt_cooldown - time_since_last_attempt
                logger.debug(f"👤 Person {track.track_id} in cooldown ({remaining_cooldown:.1f}s remaining) - confidence {confidence:.3f} < 0.5")
    
    def _check_recently_trusted(self, current_time: float):
        """Check if there's a recently trusted person who might have poor recognition"""
        # Only use fallback if there's exactly ONE person being tracked
        # This prevents applying trusted status to multiple people
        active_tracks = len([t for t in self.tracker.tracks.values() if current_time - t.last_seen < 2.0])
        
        if active_tracks > 1:
            # Multiple people present - be strict, no fallback trust
            return None
        
        trusted_memory_time = 30.0  # Reduced to 30 seconds and only for single person
        
        for name, last_seen in self.global_trusted_memory.items():
            if current_time - last_seen < trusted_memory_time:
                return name
        return None
    
    def _check_recently_trusted_for_new_track(self, current_time: float):
        """Check if there's a recently trusted person for new track creation"""
        # Only apply trusted status if there's exactly ONE active track
        # This prevents multiple people from being trusted simultaneously
        active_tracks = len([t for t in self.tracker.tracks.values() if current_time - t.last_seen < 2.0])
        
        if active_tracks > 0:
            # Other people already present - don't auto-trust new tracks
            return None
        
        # Consider trusted persons from last 5 seconds only for new track assumption
        recent_memory_time = 5.0  # 5 seconds - only for very brief disappearances
        
        for name, last_seen in self.global_trusted_memory.items():
            if current_time - last_seen < recent_memory_time:
                return name
        return None
    
    def _handle_no_face_detected(self, track: PersonTrack, current_time: float):
        """Handle when no face is detected for a person - enhanced for CCTV"""
        if track.needs_face_check and not track.verification_requested:
            # New person detected but no face visible - request verification
            track.verification_requested = True
            track.verification_start_time = current_time

            logger.info(f"🔍 Person {track.track_id} detected - requesting face verification")
            print(f"🔍 Person detected at location ({track.center[0]:.0f}, {track.center[1]:.0f}) - Please show your face")

            # Update hardware status
            if self.hardware_manager:
                self.hardware_manager.set_system_status('verifying')

            # Play verification request
            if self.hardware_manager:
                self.hardware_manager.request_verification()
            elif SECURITY.get('voice_alerts', False):
                self.sound_system.play_verification_request()

        elif track.verification_requested:
            # Enhanced verification logic with different timeouts
            time_since_request = current_time - track.verification_start_time

            # Use CCTV-specific timeout settings
            verification_timeout = track.verification_timeout
            unknown_timeout = track.unknown_timeout

            # Give voice reminder every 4 seconds during first phase
            if time_since_request < unknown_timeout and int(time_since_request) % 4 == 0 and int(time_since_request) > 0:
                if (self.hardware_manager or SECURITY.get('voice_alerts', False)) and not hasattr(track, '_last_reminder') or current_time - getattr(track, '_last_reminder', 0) > 3.0:
                    track._last_reminder = current_time
                    if self.hardware_manager:
                        self.hardware_manager.request_verification()
                    elif SECURITY.get('voice_alerts', False):
                        self.sound_system.play_verification_reminder()

            # Check if we've passed the "unknown" threshold (4 seconds)
            if time_since_request > unknown_timeout and not track.siren_played:
                # Mark as potential unknown person
                logger.warning(f"⚠️ Person {track.track_id} not verified after {unknown_timeout}s - marking as unknown")
                track.siren_played = True  # Prevent multiple alerts

                # Start recording unknown person if enabled
                if CCTV['recording_enabled']:
                    self._start_recording(track)

                # Play unknown alert
                if self.hardware_manager:
                    self.hardware_manager.play_alarm()

                # Update status LED
                if self.hardware_manager:
                    self.hardware_manager.set_system_status('alert')

            # Check if verification timeout has passed (8 seconds)
            if time_since_request > verification_timeout:
                # Full timeout - treat as unverified person
                logger.warning(f"⏰ Person {track.track_id} verification timeout after {verification_timeout}s - treating as unverified")
                self._handle_unknown_person_timeout(track, current_time)
    
    def _handle_unknown_person_verified(self, track: PersonTrack, face_roi: np.ndarray, current_time: float, frame: Optional[np.ndarray] = None):
        """Handle when unknown person shows face and is verified as unknown"""
        if not track.siren_played:
            # Play siren and alert for unknown face
            logger.warning(f"🚨 UNKNOWN FACE VERIFIED - Person {track.track_id}")
            print(f"\n🚨 ALERT! Unknown face detected at location ({track.center[0]:.0f}, {track.center[1]:.0f})")
            print("🚨 SECURITY BREACH - Unauthorized person identified!")
            
            # Play siren and voice alert
            logger.info("🔊 Attempting to play unknown alert...")
            self.sound_system.play_unknown_alert()
            
            # Save unknown face and full body
            if SECURITY['log_unknown_faces']:
                self._save_unknown_face(face_roi, track.track_id)
                # Also save full body photograph
                try:
                    frame_to_save = frame if frame is not None else getattr(self, '_last_frame', None)
                    if frame_to_save is not None:
                        self._save_unknown_person_full_body(frame_to_save, track)
                    else:
                        logger.warning("No frame available to save full body for unknown person")
                except Exception as e:
                    logger.error(f"Error saving unknown full body: {e}")
            
            track.siren_played = True
            track.alert_sent = True
    
    def _handle_unknown_person_timeout(self, track: PersonTrack, current_time: float):
        """Handle when person doesn't show face within timeout"""
        if not track.alert_sent:
            logger.warning(f"⏰ Person {track.track_id} didn't verify face - potential security concern")
            print(f"⚠️ Unverified person at location ({track.center[0]:.0f}, {track.center[1]:.0f}) - Face verification required")

            # Stop recording if it was started
            if track.is_recording:
                self._stop_recording(track.track_id)

            track.alert_sent = True

    def _start_recording(self, track: PersonTrack):
        """Start recording video of unknown person"""
        if not CCTV['recording_enabled']:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recordings/unknown_person_{track.track_id}_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = CCTV['recording_fps']
            resolution = CCTV['recording_resolution']

            video_writer = cv2.VideoWriter(filename, fourcc, fps, resolution)
            self.recording_active[track.track_id] = video_writer
            track.is_recording = True
            track.recording_start_time = time.time()

            logger.info(f"📹 Started recording person {track.track_id} to {filename}")

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")

    def _stop_recording(self, track_id: int):
        """Stop recording video of person"""
        if track_id in self.recording_active:
            try:
                self.recording_active[track_id].release()
                del self.recording_active[track_id]
                logger.info(f"📹 Stopped recording person {track_id}")
            except Exception as e:
                logger.error(f"Failed to stop recording: {e}")

    def _write_frame_to_recording(self, track_id: int, frame: np.ndarray):
        """Write frame to active recording"""
        if track_id in self.recording_active:
            try:
                # Resize frame to recording resolution if needed
                if frame.shape[:2] != CCTV['recording_resolution']:
                    frame_resized = cv2.resize(frame, CCTV['recording_resolution'])
                else:
                    frame_resized = frame

                self.recording_active[track_id].write(frame_resized)
            except Exception as e:
                logger.error(f"Failed to write frame to recording: {e}")

    def _get_time_based_greeting(self) -> str:
        """Get appropriate greeting based on current time"""
        current_hour = datetime.now().hour

        if 5 <= current_hour < 12:
            return AUDIO['greeting_morning']
        elif 12 <= current_hour < 17:
            return AUDIO['greeting_afternoon']
        else:
            return AUDIO['greeting_evening']

    def _should_greet_person(self, track: PersonTrack, current_time: float) -> bool:
        """Check if person should be greeted (avoid spam)"""
        if not CCTV['greeting_enabled']:
            return False

        cooldown = CCTV['greeting_cooldown']
        person_name = track.identity

        if person_name not in self.greeting_history:
            return True

        return current_time - self.greeting_history[person_name] > cooldown

    def _greet_person(self, track: PersonTrack, current_time: float):
        """Greet a verified person with time-based message"""
        if not self._should_greet_person(track, current_time):
            return

        person_name = track.identity
        self.greeting_history[person_name] = current_time

        greeting = self._get_time_based_greeting()
        logger.info(f"👋 Greeting {person_name}: {greeting}")

        # Use hardware manager for greeting if available
        if self.hardware_manager:
            self.hardware_manager.greet_person(person_name)
        else:
            # Fallback to text output
            print(f"{greeting}, {person_name}!")

    def _welcome_back_person(self, track: PersonTrack, current_time: float):
        """Welcome back a recognized person"""
        if not self._should_greet_person(track, current_time):
            return

        person_name = track.identity

        # Mark as greeted to avoid immediate re-greeting
        self.greeting_history[person_name] = current_time

        logger.info(f"🎉 Welcoming back {person_name}")

        # Use hardware manager for welcome message if available
        if self.hardware_manager:
            self.hardware_manager.welcome_back(person_name)
        else:
            print(f"{AUDIO['welcome_back']}, {person_name}!")
    
    def _needs_reverification(self, track: PersonTrack, current_time: float) -> bool:
        """Check if trusted person needs re-verification"""
        if not track.is_trusted:
            return True
        
        # Check if too much time has passed since last face verification
        time_since_verification = current_time - track.last_face_verification
        return time_since_verification > SECURITY.get('trusted_person_memory', 300.0)
    
    def _cleanup_trusted_memory(self):
        """Clean up old entries from trusted memory"""
        current_time = time.time()
        expired_entries = []
        
        # Check for expired entries (older than 2 minutes)
        for name, last_seen in self.global_trusted_memory.items():
            if current_time - last_seen > 120.0:  # 2 minutes
                expired_entries.append(name)
        
        # Remove expired entries
        for name in expired_entries:
            del self.global_trusted_memory[name]
            logger.info(f"🧹 Cleared trusted memory for {name} (expired)")
    
    def _handle_unknown_person(self, frame: np.ndarray, track: PersonTrack, face_roi: np.ndarray, current_time: float):
        """Handle detection of unknown person"""
        if not track.alert_sent:
            # Check alert cooldown
            if (track.track_id not in self.last_unknown_alert or 
                current_time - self.last_unknown_alert[track.track_id] > FACE_TRACKING['unknown_face_alert_cooldown']):
                
                # Send danger alert
                alert_msg = f"{SECURITY['danger_alert_message']} Location: {track.center}, Track ID: {track.track_id}"
                logger.warning(alert_msg)
                print(f"\n{alert_msg}\n")
                
                # Save unknown face if enabled
                if SECURITY['log_unknown_faces']:
                    self._save_unknown_face(face_roi, track.track_id)
                
                track.alert_sent = True
                self.last_unknown_alert[track.track_id] = current_time
    
    def _save_unknown_face(self, face_roi: np.ndarray, track_id: int):
        """Save unknown face for analysis"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_{track_id}_{timestamp}_{self.unknown_face_counter}.jpg"
            filepath = os.path.join(PATHS.get('unknown_faces_dir', 'unknown_faces'), filename)
            
            cv2.imwrite(filepath, face_roi)
            self.unknown_face_counter += 1
            
            logger.info(f"💾 Saved unknown face: {filename}")
            
            # Clean up old unknown faces if too many
            self._cleanup_unknown_faces()
            
        except Exception as e:
            logger.error(f"Failed to save unknown face: {e}")
    
    def _save_unknown_person_full_body(self, frame: np.ndarray, track: PersonTrack):
        """Save full body photograph of unknown person"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_fullbody_{track.track_id}_{timestamp}.jpg"
            filepath = os.path.join(PATHS.get('unknown_faces_dir', 'unknown_faces'), filename)
            
            # Get person bounding box with some padding
            x, y, w, h = track.bbox
            padding = 50  # Add padding around the person
            
            # Expand bounding box with padding, but keep within frame bounds
            frame_h, frame_w = frame.shape[:2]
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame_w, x + w + padding)
            y_end = min(frame_h, y + h + padding)
            
            # Extract full body region
            full_body_roi = frame[y_start:y_end, x_start:x_end]
            
            # Save the full body image in color
            cv2.imwrite(filepath, full_body_roi)
            
            logger.info(f"📸 Saved unknown person full body: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save unknown person full body: {e}")
    
    def _cleanup_unknown_faces(self):
        """Clean up old unknown face and full body images"""
        try:
            unknown_dir = PATHS.get('unknown_faces_dir', 'unknown_faces')
            # Include both face and full body images
            files = [f for f in os.listdir(unknown_dir) if f.startswith('unknown_')]
            
            # Keep double the limit since we now save both face and full body
            max_files = SECURITY['max_unknown_faces_stored'] * 2
            
            if len(files) > max_files:
                # Sort by modification time and remove oldest
                files.sort(key=lambda f: os.path.getmtime(os.path.join(unknown_dir, f)))
                files_to_remove = files[:-max_files]
                
                for file in files_to_remove:
                    os.remove(os.path.join(unknown_dir, file))
                
                logger.info(f"🧹 Cleaned up {len(files_to_remove)} old unknown person images")
                
        except Exception as e:
            logger.error(f"Failed to cleanup unknown faces: {e}")
    
    def _draw_annotations(self, frame: np.ndarray, tracks: List[PersonTrack]) -> np.ndarray:
        """Draw annotations with verification status"""
        annotated = frame.copy()
        current_time = time.time()
        
        for track in tracks:
            x, y, w, h = track.bbox
            track_age = current_time - track.last_seen
            
            # Choose color and status based on verification state
            if track.is_guest:
                # Guest mode - show guest status
                color = (255, 255, 0)  # Yellow
                host_name = track.guest_associated_with or "Unknown"
                time_remaining = max(0, CCTV['guest_mode_duration'] - (current_time - track.guest_mode_start_time))
                status = f"GUEST (with {host_name[:10]}...) ({time_remaining:.0f}s)"
            elif track.is_trusted and track.is_known:
                # Trusted known person
                color = (0, 255, 0)  # Green
                status = f"TRUSTED: {track.identity}"
            elif track.is_known and not track.is_trusted:
                # Known but needs re-verification
                color = (0, 255, 255)  # Yellow
                status = f"KNOWN: {track.identity} (needs verification)"
            elif track.verification_requested:
                # Waiting for face verification - show attempt progress
                color = (255, 165, 0)  # Orange
                timeout_remaining = SECURITY.get('verification_timeout', 10.0) - (current_time - track.verification_start_time)
                
                # Check if in cooldown period
                time_since_last_attempt = current_time - track.last_verification_attempt
                if time_since_last_attempt < track.verification_attempt_cooldown:
                    cooldown_remaining = track.verification_attempt_cooldown - time_since_last_attempt
                    status = f"COOLDOWN ({track.verification_attempts}/{track.max_verification_attempts}) ({cooldown_remaining:.1f}s)"
                else:
                    status = f"VERIFY FACE ({track.verification_attempts}/{track.max_verification_attempts}) ({timeout_remaining:.1f}s)"
            elif not track.is_known and track.siren_played:
                # Unknown person verified as unknown
                color = (0, 0, 255)  # Red
                status = "🚨 UNKNOWN PERSON!"
            elif track.needs_face_check:
                # New person, needs face check
                color = (128, 0, 128)  # Purple
                if track.verification_attempts > 0:
                    # Check if in cooldown
                    time_since_last_attempt = current_time - track.last_verification_attempt
                    if time_since_last_attempt < track.verification_attempt_cooldown:
                        cooldown_remaining = track.verification_attempt_cooldown - time_since_last_attempt
                        status = f"COOLDOWN ({track.verification_attempts}/{track.max_verification_attempts}) ({cooldown_remaining:.1f}s)"
                    else:
                        status = f"VERIFYING ({track.verification_attempts}/{track.max_verification_attempts}) - SHOW FACE"
                else:
                    status = "NEW PERSON - SHOW FACE"
            else:
                # Default unknown
                color = (0, 0, 180)  # Dark red
                status = "UNKNOWN"
            
            # Line thickness based on track age
            line_thickness = 3 if track.verification_requested else 2
            if track_age > 2.0:
                line_thickness = 1
            
            # Draw person bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, line_thickness)
            
            # Draw face bounding box if available
            if track.face_bbox:
                fx, fy, fw, fh = track.face_bbox
                face_color = (0, 255, 0) if track.is_known else (0, 0, 255)
                cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), face_color, 2)
            
            # Draw labels
            label_y = y - 10 if y > 60 else y + h + 20
            
            # Track ID and age
            cv2.putText(annotated, f"ID: {track.track_id} ({track_age:.1f}s)", (x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Status
            cv2.putText(annotated, status, (x, label_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Confidence if available
            if track.identity_confidence > 0:
                cv2.putText(annotated, f"Conf: {track.identity_confidence:.2f}", 
                           (x, label_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Special indicators
            if track.verification_requested:
                # Blinking verification request
                if int(current_time * 2) % 2:  # Blink every 0.5 seconds
                    cv2.putText(annotated, "SHOW FACE!", (x + w//2 - 40, y + h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Only show alert if person is still unknown/unverified
            if track.siren_played and not track.is_known:
                cv2.putText(annotated, "🚨 ALERT", (x + w - 50, y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw system info
        trusted_count = sum(1 for t in tracks if t.is_trusted)
        verification_count = sum(1 for t in tracks if t.verification_requested)
        unknown_count = sum(1 for t in tracks if not t.is_known and not t.verification_requested)
        guest_count = sum(1 for t in tracks if t.is_guest)

        info_text = f"Trusted: {trusted_count} | Verifying: {verification_count} | Unknown: {unknown_count} | Guests: {guest_count}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw guest mode indicator
        if guest_count > 0:
            guest_mode_text = "👥 GUEST MODE ACTIVE"
            cv2.putText(annotated, guest_mode_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw verification instructions
        if verification_count > 0:
            instruction_text = "🔍 Face verification required - Look at camera!"
            cv2.putText(annotated, instruction_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw controls
        controls_text = "R=Reset | C=Clean | M=Mute | ESC=Exit"
        cv2.putText(annotated, controls_text, (10, annotated.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return annotated

def initialize_camera(camera_index: int = 0):
    """Initialize camera with enhanced settings - supports both picamera2 and OpenCV"""
    try:
        if PICAMERA2_AVAILABLE:
            # Use picamera2 for Raspberry Pi
            picam2 = Picamera2()
            
            # Configure preview with high resolution and low-res display stream
            preview_config = picam2.create_preview_configuration(
                main={"size": (CAMERA['width'], CAMERA['height'])},  # High resolution for processing
                lores={"size": (640, 360)},   # Low-res display stream for performance
                display="lores"
            )
            picam2.configure(preview_config)
            # Set autofocus mode
            picam2.set_controls({"AfMode": 1, "AfTrigger": 0, "FrameRate": CAMERA['fps']})  # Normal AF
            
            # Start camera
            picam2.start()
            
            logger.info(f"✅ Picamera2 initialized: {CAMERA['width']}x{CAMERA['height']} @ {CAMERA['fps']}fps with autofocus")
            return picam2
        else:
            # Fallback to OpenCV for non-Raspberry Pi systems
            cam = cv2.VideoCapture(camera_index)
            if not cam.isOpened():
                logger.error("Could not open webcam")
                return None
            
            # Set camera properties
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
            cam.set(cv2.CAP_PROP_FPS, CAMERA['fps'])
            
            # Try to enable auto-exposure and auto-focus if available
            cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            
            logger.info(f"✅ OpenCV camera initialized: {CAMERA['width']}x{CAMERA['height']} @ {CAMERA['fps']}fps")
            return cam
        
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return None

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting Advanced Person Tracking System...")
        
        # Initialize camera
        cam = initialize_camera(CAMERA['index'])
        if cam is None:
            raise ValueError("Failed to initialize camera")
        
        # Initialize tracker
        tracker = AdvancedPersonTracker()
        
        logger.info("🎯 Smart Security System Ready!")
        logger.info("🔊 Sound alerts: " + ("Enabled" if tracker.sound_system.sound_enabled else "Disabled"))
        logger.info("🗣️ Voice alerts: " + ("Enabled" if tracker.sound_system.voice_enabled else "Disabled"))
        logger.info("")
        logger.info("🛡️ Security Features:")
        logger.info("  ✅ Smart person tracking with memory")
        logger.info("  ✅ Face verification requests")
        logger.info("  ✅ Unknown person alerts with siren")
        logger.info("  ✅ Trusted person memory (5 minutes)")
        logger.info("")
        logger.info("🎮 Controls:")
        logger.info("  - ESC/Q: Exit")
        logger.info("  - S: Save current frame")
        logger.info("  - R: Reset all tracks")
        logger.info("  - C: Clean stale tracks")
        logger.info("  - M: Mute/Unmute sounds")
        logger.info("  - SPACE: Force cleanup")
        
        frame_count = 0
        fps_counter = time.time()
        last_af_trigger = 0  # For autofocus timing
        
        while True:
            # Read frame based on camera type
            if PICAMERA2_AVAILABLE and hasattr(cam, 'capture_array'):
                # picamera2 frame reading
                try:
                    frame = cam.capture_array("main")
                    # Convert from RGB to BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    ret = True
                except Exception as e:
                    logger.warning(f"Failed to capture frame: {e}")
                    ret = False
            else:
                # OpenCV frame reading
                ret, frame = cam.read()
            
            if not ret:
                logger.warning("Failed to grab frame")
                continue
            
            # Trigger autofocus periodically for picamera2
            if PICAMERA2_AVAILABLE and hasattr(cam, 'set_controls'):
                current_time = time.time()
                if current_time - last_af_trigger > 1.0:  # Trigger AF every second
                    try:
                        cam.set_controls({"AfTrigger": 0})
                        last_af_trigger = current_time
                    except Exception as e:
                        logger.debug(f"AF trigger failed: {e}")
            
            # Process frame
            annotated_frame, tracks = tracker.process_frame(frame)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - fps_counter)
                fps_counter = current_time
                
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Advanced Person Tracking System', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q
                break
            elif key == ord('s') or key == ord('S'):  # Save frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                logger.info(f"📸 Saved frame: {filename}")
            elif key == ord('r') or key == ord('R'):  # Reset tracks
                tracker.tracker.tracks.clear()
                tracker.tracker.next_id = 1
                tracker.tracker.frames_without_detection = 0
                logger.info("🔄 Reset all tracks")
            elif key == ord('c') or key == ord('C'):  # Clean stale tracks
                current_time = time.time()
                tracks_to_remove = []
                for track_id, track in tracker.tracker.tracks.items():
                    if current_time - track.last_seen > 2.0:  # 2 seconds
                        tracks_to_remove.append(track_id)
                
                for track_id in tracks_to_remove:
                    del tracker.tracker.tracks[track_id]
                    
                if tracks_to_remove:
                    logger.info(f"🧹 Cleaned {len(tracks_to_remove)} stale tracks")
                else:
                    logger.info("✨ No stale tracks to clean")
            elif key == ord(' '):  # Space - Force cleanup
                tracker.tracker.frames_without_detection = tracker.tracker.max_disappeared + 1
                logger.info("🧹 Forced track cleanup on next frame")
            elif key == ord('m') or key == ord('M'):  # Mute/Unmute
                if tracker.sound_system.sound_enabled or tracker.sound_system.voice_enabled:
                    # Toggle mute
                    old_sound = tracker.sound_system.sound_enabled
                    old_voice = tracker.sound_system.voice_enabled
                    
                    tracker.sound_system.sound_enabled = not (old_sound or old_voice)
                    tracker.sound_system.voice_enabled = not (old_sound or old_voice)
                    
                    if tracker.sound_system.sound_enabled:
                        logger.info("🔊 Sound alerts enabled")
                    else:
                        logger.info("🔇 Sound alerts muted")
                        tracker.sound_system.stop_all_sounds()
                else:
                    logger.info("🔇 Sound system not available")
        
        logger.info("👋 Advanced Person Tracking System stopped")
        
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        
    finally:
        if 'cam' in locals():
            if PICAMERA2_AVAILABLE and hasattr(cam, 'stop'):
                # picamera2 cleanup
                try:
                    cam.stop()
                    cam.close()
                    logger.info("📷 Picamera2 stopped and closed")
                except Exception as e:
                    logger.error(f"Error closing picamera2: {e}")
            else:
                # OpenCV cleanup
                try:
                    cam.release()
                    logger.info("📷 OpenCV camera released")
                except Exception as e:
                    logger.error(f"Error releasing camera: {e}")
        cv2.destroyAllWindows()
