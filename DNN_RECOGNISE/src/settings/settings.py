"""
Configuration settings for the advanced person tracking and face recognition system
"""
import os

# Camera settings
CAMERA = {
    'index': 0,  # Default camera (0 is usually built-in webcam)
    'width': 640,  # Higher resolution for better detection
    'height': 480,
    'fps': 60,  # Target FPS
}

# Advanced person detection settings
PERSON_DETECTION = {
    'method': 'yolov8',  # Options: 'yolov8', 'yolo_face', 'combined'
    'confidence_threshold': 0.5,  # Minimum confidence for person detection
    'nms_threshold': 0.4,  # Non-maximum suppression threshold
    'input_size': 640,  # Input size for YOLO model (416, 512, 640, 800)
    'use_gpu': False,  # Set to True if you have CUDA-enabled OpenCV
    'track_classes': [0],  # COCO class IDs to track (0 = person)
}

# Face detection settings (enhanced)
FACE_DETECTION = {
    'method': 'dnn',  # Options: 'haar', 'dnn', 'yolo_face', 'ensemble'
    'scale_factor': 1.05,  # Smaller scale factor for better detection accuracy
    'min_neighbors': 6,    # Reduced for better sensitivity
    'min_size': (30, 30),  # Minimum face size to detect
    'dnn_confidence_threshold': 0.5,  # Minimum confidence for DNN detection (lowered for better detection)
    'yolo_confidence_threshold': 0.6,  # Minimum confidence for YOLO face detection
    'ensemble_voting_threshold': 0.6,  # Minimum agreement ratio for ensemble
}

# Training settings. Number of images needed to train the model.
TRAINING = {
    'samples_needed': 10
}

# Model and data paths
PATHS = {
    'image_dir': 'images',
    'cascade_file': 'haarcascade_frontalface_alt2.xml',
    'cascade_alt_tree': 'haarcascade_frontalface_alt_tree.xml',
    'dnn_model': 'opencv_face_detector_uint8.pb',
    'dnn_config': 'opencv_face_detector.pbtxt',
    'names_file': 'names.json',
    'trainer_file': 'trainer.yml',
    
    # Advanced model paths
    'yolov8_person_model': 'models/yolov8n.onnx',  # YOLOv8 nano for person detection
    'yolo_face_model': 'models/scrfd_10g_bnkps.onnx',  # scrfd face detection model
    'arcface_model': 'models/arcface_r100.onnx',   # ArcFace recognition model
    'face_embeddings': 'models/face_embeddings.pkl',  # Stored face embeddings
    
    # Legacy YOLO paths (kept for compatibility)
    'yolo_weights': 'yolo_models/yolov4.weights',
    'yolo_config': 'yolo_models/yolov4.cfg',
    'yolo_classes': 'yolo_models/coco.names',
    
    # Directories
    'models_dir': 'models',
    'custom_models_dir': 'custom_models',
    'object_annotations_dir': 'object_annotations',
    'object_training_dir': 'object_training_data',
    'unknown_faces_dir': 'unknown_faces',  # Store unknown faces for analysis
}

# Advanced face recognition settings
FACE_RECOGNITION = {
    'method': 'arcface',  # Options: 'lbph', 'arcface', 'facenet'
    'confidence_threshold': 0.5,  # STRICT: Minimum confidence 0.5 - anything below is unknown
    'lbph_threshold': 35,  # Legacy LBPH threshold
    'embedding_size': 512,  # ArcFace embedding dimension
    'input_size': (112, 112),  # Face input size for ArcFace
}

# Advanced person tracking settings
PERSON_TRACKING = {
    'tracking_method': 'bytetrack',  # Options: 'bytetrack', 'deepsort', 'sort'
    'max_disappeared': 30,  # Max frames a person can disappear before removal (increased for stability)
    'max_distance': 150,  # Max distance for person association (increased for better tracking)
    'track_buffer': 30,  # Track buffer for ByteTrack (increased)
    'match_threshold': 0.6,  # Matching threshold for tracking (more lenient)
    'frame_rate': 60,  # Camera frame rate
}

# Face tracking and recognition integration
FACE_TRACKING = {
    'face_recognition_interval': 5,  # Frames between face recognition attempts
    'min_face_size': 50,  # Minimum face size for recognition
    'max_face_age': 100,  # Maximum frames to track a face without recognition
    'recognition_confidence_threshold': 0.4,  # Minimum confidence for face recognition
    'unknown_face_alert_cooldown': 5.0,  # Seconds between unknown face alerts
}

# Security and alerting settings
SECURITY = {
    'unknown_person_alert': True,  # Enable alerts for unknown persons
    'danger_alert_message': "🚨 DANGER: Unknown person detected!",
    'log_unknown_faces': True,  # Save unknown faces for analysis
    'alert_sound': True,  # Enable sound alerts (requires pygame)
    'voice_alerts': True,  # Enable voice alerts (requires pyttsx3)
    'alert_email': False,  # Enable email alerts (requires configuration)
    'max_unknown_faces_stored': 1000,  # Maximum unknown faces to store
    'verification_timeout': 15.0,  # Seconds to wait for face verification (increased)
    'siren_duration': 3.0,  # Seconds to play siren for unknown faces
    'trusted_person_memory': 600.0,  # Seconds to remember trusted person without face (10 minutes)
}

# CCTV Hardware Configuration
HARDWARE = {
    'pir_pin': 17,  # GPIO pin for PIR motion sensor
    'led_brightness_pin': 18,  # GPIO pin for high brightness LED
    'led_green_pin': 22,  # GPIO pin for green status LED
    'led_yellow_pin': 23,  # GPIO pin for yellow status LED
    'led_red_pin': 24,  # GPIO pin for red status LED
    'bt_speaker_mac': 'XX:XX:XX:XX:XX:XX',  # Bluetooth speaker MAC address
    'bt_speaker_name': 'CCTV_Speaker',  # Bluetooth speaker name
}

# CCTV System Settings
CCTV = {
    'motion_detection_enabled': True,  # Enable PIR-based motion detection
    'motion_cooldown': 5.0,  # Seconds between motion detections
    'led_auto_brightness': True,  # Auto-control high brightness LED
    'led_brightness_duration': 30.0,  # How long to keep LED on after motion
    'recording_enabled': True,  # Enable video recording for unknown persons
    'recording_duration': 60.0,  # How long to record unknown persons (seconds)
    'recording_fps': 30,  # Recording frame rate
    'recording_resolution': (1280, 720),  # Recording resolution
    'greeting_enabled': True,  # Enable time-based greetings
    'greeting_cooldown': 300.0,  # Seconds between greetings for same person
    'verification_timeout': 8.0,  # Time to wait for face verification (seconds)
    'unknown_timeout': 4.0,  # Time before marking as unknown person (seconds)
    'max_verification_attempts': 3,  # Maximum verification attempts
    'verification_cooldown': 2.0,  # Cooldown between verification attempts
}

# Audio Settings
AUDIO = {
    'greeting_morning': 'Good morning',
    'greeting_afternoon': 'Good afternoon',
    'greeting_evening': 'Good evening',
    'unknown_alert': 'Alert! Unknown person detected in the area',
    'verification_request': 'Please look at the camera for verification',
    'welcome_back': 'Welcome back',
    'alarm_sound_file': 'sounds/alarm.mp3',  # Alarm sound file path
    'greeting_sound_enabled': True,  # Play sound with greetings
    'alert_sound_enabled': True,  # Play sound with alerts
}

# Voice Settings for Text-to-Speech (Female Voice)
VOICE = {
    'engine': 'espeak-ng',  # Options: 'espeak-ng', 'festival', 'auto'
    'gender': 'female',  # Options: 'female', 'male'
    'espeak_voice': 'en+f3',  # espeak-ng female voice (en+f1, en+f2, en+f3, en+f4)
    'festival_voice': 'cmu_us_slt_cg',  # Festival female voice
    'speech_rate': 150,  # Speech rate (words per minute)
    'pitch': 50,  # Pitch adjustment (0-100, 50 = natural female)
    'volume': 80,  # Volume (0-100)
    'language': 'en',  # Language code
}

# Legacy compatibility settings
CONFIDENCE_THRESHOLD = 35  # For LBPH compatibility
OBJECT_DETECTION = {
    'method': 'yolo',  # Options: 'yolo', 'custom', 'combined'
    'confidence_threshold': 0.5,  # Minimum confidence for object detection
    'nms_threshold': 0.4,  # Non-maximum suppression threshold
    'input_size': 416,  # Input size for YOLO model (416, 512, 608)
    'use_gpu': False,  # Set to True if you have CUDA-enabled OpenCV
    'target_classes': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'building'],  # Classes to focus on
}

# Legacy tracking settings (kept for backward compatibility)
TRACKING = {
    'tracking_duration': 3,  # Duration in seconds to track a recognized face/object
    'unknown_tracking_duration': 0.5,  # Shorter tracking for unknown faces (allows re-recognition)
    'max_distance_threshold': 100,  # Maximum distance between face/object positions to consider it the same
    'recognition_cooldown': 1.0,  # Minimum time between recognition attempts (seconds)
    'unknown_retry_interval': 0.5,  # How often to retry recognition for unknown faces (seconds)
    'verification_interval': 3.0,  # How often to re-verify ALL faces (including known ones)
    'confidence_threshold_for_reverify': 70.0,  # Re-verify faces with confidence below this threshold more often
    'matching_threshold': 0.3,  # Minimum score for object matching in tracking
}