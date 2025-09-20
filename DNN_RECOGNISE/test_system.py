#!/usr/bin/env python3
"""
Test script for the Advanced Person Tracking System
Tests various components and configurations
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("🧪 Testing imports...")
    
    try:
        import cv2
        logger.info(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        logger.error(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        logger.info(f"✅ NumPy version: {np.__version__}")
    except ImportError as e:
        logger.error(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from settings.settings import CAMERA, PERSON_DETECTION, FACE_RECOGNITION, SECURITY
        logger.info("✅ Settings imported successfully")
    except ImportError as e:
        logger.error(f"❌ Settings import failed: {e}")
        return False
    
    return True

def test_camera():
    """Test camera initialization"""
    logger.info("📹 Testing camera...")
    
    try:
        import cv2
        from settings.settings import CAMERA
        
        cam = cv2.VideoCapture(CAMERA['index'])
        if not cam.isOpened():
            logger.error("❌ Camera could not be opened")
            return False
        
        ret, frame = cam.read()
        if not ret:
            logger.error("❌ Could not read frame from camera")
            cam.release()
            return False
        
        logger.info(f"✅ Camera working - Frame shape: {frame.shape}")
        cam.release()
        return True
        
    except Exception as e:
        logger.error(f"❌ Camera test failed: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    logger.info("📁 Testing directories...")
    
    required_dirs = ['images', 'models', 'unknown_faces']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            logger.info(f"✅ Directory exists: {dir_name}")
        else:
            logger.warning(f"⚠️  Directory missing: {dir_name} (will be created automatically)")
    
    return True

def test_models():
    """Test if AI models are available"""
    logger.info("🤖 Testing AI models...")
    
    from settings.settings import PATHS
    
    models_to_check = {
        'yolov8_person_model': 'YOLOv8 Person Detection',
        'yolo_face_model': 'YOLO Face Detection', 
        'arcface_model': 'ArcFace Recognition',
        'dnn_model': 'OpenCV DNN Face Detection',
        'cascade_file': 'Haar Cascade'
    }
    
    available_models = 0
    total_models = len(models_to_check)
    
    for model_key, model_name in models_to_check.items():
        model_path = PATHS.get(model_key, '')
        if os.path.exists(model_path):
            logger.info(f"✅ {model_name}: {model_path}")
            available_models += 1
        else:
            logger.warning(f"⚠️  {model_name}: Not found at {model_path}")
    
    logger.info(f"📊 Models available: {available_models}/{total_models}")
    
    if available_models == 0:
        logger.error("❌ No AI models found! Run 'python download_models.py'")
        return False
    
    return True

def test_advanced_tracker():
    """Test if advanced tracker can be initialized"""
    logger.info("🚀 Testing advanced tracker...")
    
    try:
        from advanced_person_tracker import AdvancedPersonTracker
        
        # Try to initialize (this will test model loading)
        tracker = AdvancedPersonTracker()
        logger.info("✅ Advanced tracker initialized successfully")
        
        # Test with a dummy frame
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        annotated_frame, tracks = tracker.process_frame(dummy_frame)
        logger.info(f"✅ Frame processing works - Tracks: {len(tracks)}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Advanced tracker import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Advanced tracker test failed: {e}")
        return False

def test_training_data():
    """Test if training data exists"""
    logger.info("📚 Testing training data...")
    
    from settings.settings import PATHS
    
    images_dir = PATHS['image_dir']
    if not os.path.exists(images_dir):
        logger.warning(f"⚠️  Training images directory not found: {images_dir}")
        logger.info("💡 Run 'python src/face_taker.py' to collect training data")
        return False
    
    # Count training images
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f.startswith('Users-')]
    
    if len(image_files) == 0:
        logger.warning("⚠️  No training images found")
        logger.info("💡 Run 'python src/face_taker.py' to collect training data")
        return False
    
    logger.info(f"✅ Found {len(image_files)} training images")
    
    # Check if trainer file exists
    trainer_file = PATHS['trainer_file']
    if os.path.exists(trainer_file):
        logger.info(f"✅ Trained model exists: {trainer_file}")
    else:
        logger.warning(f"⚠️  Trained model not found: {trainer_file}")
        logger.info("💡 Run 'python src/face_trainer.py' to train the model")
    
    return True

def test_configuration():
    """Test system configuration"""
    logger.info("⚙️ Testing configuration...")
    
    try:
        from settings.settings import (
            CAMERA, PERSON_DETECTION, FACE_DETECTION, FACE_RECOGNITION,
            PERSON_TRACKING, FACE_TRACKING, SECURITY, PATHS
        )
        
        logger.info("✅ Configuration sections loaded:")
        logger.info(f"   📹 Camera: {CAMERA['width']}x{CAMERA['height']}")
        logger.info(f"   👤 Person Detection: {PERSON_DETECTION['method']}")
        logger.info(f"   😊 Face Detection: {FACE_DETECTION['method']}")
        logger.info(f"   🧠 Face Recognition: {FACE_RECOGNITION['method']}")
        logger.info(f"   🚨 Security Alerts: {SECURITY['unknown_person_alert']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Advanced Person Tracking System - Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Directory Test", test_directories),
        ("Configuration Test", test_configuration),
        ("Camera Test", test_camera),
        ("Model Test", test_models),
        ("Training Data Test", test_training_data),
        ("Advanced Tracker Test", test_advanced_tracker),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
        
        logger.info("-" * 40)
    
    # Summary
    logger.info(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use!")
        logger.info("\n🚀 Next steps:")
        logger.info("1. Collect training data: python src/face_taker.py")
        logger.info("2. Train models: python src/face_trainer.py")
        logger.info("3. Run system: python src/face_recognizer.py")
    else:
        logger.warning("⚠️  Some tests failed. Please address the issues above.")
        logger.info("\n💡 Common fixes:")
        logger.info("- Install missing dependencies: pip install -r requirements_advanced.txt")
        logger.info("- Download models: python download_models.py")
        logger.info("- Check camera connection")
        logger.info("- Collect training data: python src/face_taker.py")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        sys.exit(1)
