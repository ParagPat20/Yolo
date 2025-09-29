"""
Configuration settings for the advanced person tracking and face recognition system
"""
import os

# Camera settings
CAMERA = {
    'index': 0,  # Default camera (0 is usually built-in webcam)
    'width': 1640,  # Higher resolution for better detection
    'height': 1232,
    'fps': 10,  # Target FPS
}

# Advanced person detection settings
PERSON_DETECTION = {
    'method': 'yolov8',  # Options: 'yolov8', 'yolo_face', 'combined'
    'confidence_threshold': 0.5,  # Minimum confidence for person detection
    'nms_threshold': 0.4,  # Non-maximum suppression threshold
    'input_size': 640,  # Input size for YOLO model (416, 512, 640, 800)
    'use_gpu': False,  # Set to True if you have CUDA-enabled OpenCV
    'track_classes': [0],  # COCO class IDs to track (0 = person)z
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
    'samples_needed': 100
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
    # Tracking algorithm to use for person tracking.
    # Options: 'bytetrack' (default, robust), 'deepsort', 'sort'
    'tracking_method': 'bytetrack',

    # Maximum number of frames a person can be missing before their track is removed.
    # Higher values make tracking more stable but may keep lost tracks longer.
    'max_disappeared': 20,

    # Maximum pixel distance allowed between detections to associate them with the same person.
    # Larger values allow for more movement but may increase false associations.
    'max_distance': 170,

    # Number of frames to keep a track "alive" in ByteTrack after last detection.
    # Higher values help maintain identity through occlusions.
    'track_buffer': 40,

    # Minimum similarity threshold (0.0-1.0) for associating detections with existing tracks.
    # Lower values are more lenient and may increase false matches.
    'match_threshold': 0.6,

    # Expected camera frame rate (frames per second) for tracking logic.
    # Used to tune time-based parameters in the tracker.
    'frame_rate': 10,
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
    'alert_email': False,  # Enable email alerts (requires configuration)
    'max_unknown_faces_stored': 1000,  # Maximum unknown faces to store
    'verification_timeout': 15.0,  # Seconds to wait for face verification (increased)
    'trusted_person_memory': 600.0,  # Seconds to remember trusted person without face (10 minutes)
}

# CCTV Hardware Configuration
HARDWARE = {
    # PIR motion sensor on GPIO 4 per hardware spec
    'pir_pin': 15,

    # Existing status LED pins (if used elsewhere)
    'led_brightness_pin': 3,  # GPIO pin for high brightness LED
    'led_green_pin': 18,       # GPIO pin for green status LED
    'led_yellow_pin': 23,      # GPIO pin for yellow status LED
    'led_red_pin': 24,         # GPIO pin for red status LED

    # New external device button pins
    'speaker_button_pin': 2,   # GPIO 2: Bluetooth speaker power button (hold LOW 2s to power on)
    'lights_button_pin': 3,    # GPIO 3: Lights mode button (click/double-click cycles modes)

    # Timing configuration (milliseconds)
    'button_press_ms': 120,          # Single click press duration
    'double_click_gap_ms': 120,      # Time between the two clicks for double click
    'speaker_power_hold_ms': 2000,   # Hold LOW 2 seconds to power speaker ON

    # Motion lighting behavior
    'motion_light_duration_s': 30,   # High brightness hold on motion

    # Lights click timing
    'lights_click_gap_s': 1.0,       # Required gap between single clicks
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
    'verification_timeout': 15.0,  # Time to wait for face verification before alarm (seconds)
    'unknown_timeout': 20.0,  # Time before marking as unknown person (seconds)
    'max_verification_attempts': 3,  # Maximum verification attempts
    'verification_cooldown': 2.0,  # Cooldown between verification attempts
    # Recording behavior
    'unknown_initial_record_seconds': 600.0,  # Initial recording length for unknown detection (10 minutes)
    # Guest Mode Settings
    'guest_mode_enabled': True,  # Enable context-aware guest mode
    'guest_mode_duration': 60.0,  # Guest mode duration in seconds (15 minutes)
    'guest_detection_distance': 100.0,  # Maximum distance for guest association (pixels)
    'guest_trajectory_similarity': 0.7,  # Minimum trajectory similarity for guest detection
    'guest_mode_yellow_pulse_interval': 1.0,  # Yellow LED pulse interval in seconds
}

# Audio Settings (Visual-only messages)
AUDIO = {
    'greeting_morning': 'Good morning',
    'greeting_afternoon': 'Good afternoon',
    'greeting_evening': 'Good evening',
    'unknown_alert': 'Alert! Unknown person detected!',
    'verification_request': 'Please look at the camera for verification',
    'welcome_back': 'Welcome back',
    'guest_mode_message': "Guest Mode Activated",
    'guest_mode_reverted': 'Guest mode EXPIRED',
    'unknown_timeout': 'Unknown person timeout. Please verify face',
    'alarm_sound_path': 'sounds/alarm.mp3',
    'verification_beep_path': 'sounds/verification_beep.mp3',
    'use_mp3_sounds': True,  # Enable MP3 sound playback
    'mp3_player_linux': 'mpg123',  # MP3 player for Linux
    'mp3_player_windows': 'powershell',  # Fallback for Windows
    'alarm_duration_minutes': 2,  # Duration to play alarm sound (in minutes) - FIXED at 2 minutes
    'alarm_loop_interval': 5  # Seconds between alarm sound loops
}

# Sound System Settings - WAV File Generation
SOUND_SYSTEM = {
    'enabled': True,  # Enable/disable sound system
    'language': 'en',  # Default language: 'en' for English, 'gu' for Gujarati
    'wav_files_dir': 'sounds/wav',  # Directory for generated WAV files
    'piper': {
        'model_path': '/usr/local/share/piper-voices/en_US-ljspeech-medium.onnx',  # Default model path
        'models': {
            'en': '/usr/local/share/piper-voices/en_US-ljspeech-medium.onnx',  # English model
            'gu': '/usr/local/share/piper-voices/gu_IN-cmu-indic_medium.onnx',  # Gujarati model (if available)
        },
        'noise': 0.667,  # Noise scale (0.0-1.0) for WAV generation
        'length_penalty': 1.0,  # Length scale (0.0-2.0) for WAV generation
    }
}

# Voice Settings removed (sound system disabled)

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
    'tracking_duration': 2,  # Duration in seconds to track a recognized face/object
    'unknown_tracking_duration': 0.5,  # Shorter tracking for unknown faces (allows re-recognition)
    'max_distance_threshold': 100,  # Maximum distance between face/object positions to consider it the same
    'recognition_cooldown': 1.0,  # Minimum time between recognition attempts (seconds)
    'unknown_retry_interval': 0.5,  # How often to retry recognition for unknown faces (seconds)
    'verification_interval': 3.0,  # How often to re-verify ALL faces (including known ones)
    'confidence_threshold_for_reverify': 70.0,  # Re-verify faces with confidence below this threshold more often
    'matching_threshold': 0.3,  # Minimum score for object matching in tracking
}