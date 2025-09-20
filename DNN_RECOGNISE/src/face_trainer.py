# Suppress macOS warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
# At the top of your Python script inside venv
import sys
sys.path.append('/usr/lib/python3/dist-packages')  # Path to system Python packages

import cv2
import numpy as np
from PIL import Image
import os
import logging
from settings.settings import PATHS, FACE_RECOGNITION

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_face_for_training(face_img):
    """
    Apply consistent preprocessing to face images for better training
    Compatible with both LBPH and ArcFace
    """
    try:
        # Get target size from settings
        if FACE_RECOGNITION['method'] == 'arcface':
            target_size = FACE_RECOGNITION.get('input_size', (112, 112))
        else:
            target_size = (100, 100)  # LBPH default
        
        # Resize to consistent size
        face_img = cv2.resize(face_img, target_size)
        
        # Apply histogram equalization for lighting normalization
        face_img = cv2.equalizeHist(face_img)
        
        # Apply slight Gaussian blur to reduce noise
        face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
        
        return face_img
    except Exception as e:
        logger.error(f"Error in face preprocessing: {e}")
        return face_img

def create_arcface_embeddings():
    """
    Create ArcFace embeddings from training images
    """
    try:
        import pickle
        
        logger.info("🧠 Creating ArcFace embeddings...")
        
        # Try to import ArcFace recognizer from advanced tracker
        try:
            from advanced_person_tracker import ArcFaceRecognizer
            recognizer = ArcFaceRecognizer()
            
            if recognizer.net is None:
                logger.warning("ArcFace model not available, skipping embedding creation")
                return False
            
        except ImportError:
            logger.warning("Advanced tracker not available, skipping ArcFace embeddings")
            return False
        
        # Process ArcFace format images
        arcface_dir = os.path.join(PATHS['image_dir'], 'arcface_format')
        if not os.path.exists(arcface_dir):
            logger.warning("No ArcFace format images found")
            return False
        
        embeddings = {}
        
        # Load names mapping
        names_path = PATHS['names_file']
        names_mapping = {}
        if os.path.exists(names_path):
            try:
                with open(names_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        names_mapping = json.loads(content)
            except Exception as e:
                logger.error(f"Failed to load names: {e}")
        
        # Process each person directory
        for person_dir in os.listdir(arcface_dir):
            person_path = os.path.join(arcface_dir, person_dir)
            if not os.path.isdir(person_path):
                continue
            
            # Extract person ID from directory name
            try:
                person_id = int(person_dir.split('_')[1])
                person_name = names_mapping.get(str(person_id), f"Person_{person_id}")
            except (IndexError, ValueError):
                logger.warning(f"Invalid person directory: {person_dir}")
                continue
            
            # Process images for this person
            person_embeddings = []
            image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in image_files:
                image_path = os.path.join(person_path, image_file)
                try:
                    # Load and process image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Extract embedding
                    embedding = recognizer.extract_embedding(image)
                    if embedding is not None:
                        person_embeddings.append(embedding)
                        
                except Exception as e:
                    logger.warning(f"Failed to process {image_path}: {e}")
            
            # Average embeddings for this person
            if person_embeddings:
                avg_embedding = np.mean(person_embeddings, axis=0)
                # Normalize
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                embeddings[person_name] = avg_embedding
                logger.info(f"Created embedding for {person_name} from {len(person_embeddings)} images")
        
        # Save embeddings
        if embeddings:
            embeddings_path = PATHS.get('face_embeddings', 'models/face_embeddings.pkl')
            os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
            
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            logger.info(f"✅ Saved {len(embeddings)} ArcFace embeddings to {embeddings_path}")
            return True
        else:
            logger.warning("No embeddings created")
            return False
            
    except Exception as e:
        logger.error(f"Failed to create ArcFace embeddings: {e}")
        return False

def get_images_and_labels(path: str):
    """
    Load face images and corresponding labels with enhanced preprocessing.

    Parameters:
        path (str): Directory path containing face images.

    Returns:
        tuple: (face_samples, ids) Lists of face samples and corresponding labels.
    """
    try:
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        faceSamples = []
        ids = []
        
        logger.info(f"📂 Found {len(imagePaths)} training images")
        logger.info("📝 Processing pre-cropped face images (no face detection needed)")

        processed_count = 0
        skipped_count = 0

        for imagePath in imagePaths:
            try:
                # Convert image to grayscale
                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
                
                # Extract the user ID from the image file name
                filename = os.path.split(imagePath)[-1]
                if not filename.startswith('Users-'):
                    logger.warning(f"Skipping file with invalid format: {filename}")
                    skipped_count += 1
                    continue
                    
                try:
                    id = int(filename.split("-")[1])
                except (IndexError, ValueError):
                    logger.warning(f"Could not extract ID from filename: {filename}")
                    skipped_count += 1
                    continue

                # For training images that are already cropped faces, just use preprocessing
                # (No need to detect faces in already-cropped face images)
                face_img = preprocess_face_for_training(img_numpy)
                faceSamples.append(face_img)
                ids.append(id)
                processed_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {imagePath}: {e}")
                skipped_count += 1
                continue

        logger.info(f"✅ Processed {processed_count} face samples")
        if skipped_count > 0:
            logger.warning(f"⚠️  Skipped {skipped_count} files due to errors")
            
        return faceSamples, ids
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise

if __name__ == "__main__":
    try:
        import json
        
        logger.info("🚀 Starting Enhanced Face Recognition Training...")
        
        # Check if training data directory exists
        if not os.path.exists(PATHS['image_dir']):
            raise ValueError(f"Training data directory not found: {PATHS['image_dir']}")
        
        # Determine training method
        training_method = FACE_RECOGNITION.get('method', 'lbph')
        logger.info(f"🎯 Training method: {training_method.upper()}")
        
        # Train LBPH model (always for backward compatibility)
        logger.info("📚 Training LBPH model for legacy compatibility...")
        
        # Initialize face recognizer with optimized parameters
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,        # Smaller radius for more detailed patterns
            neighbors=8,     # Standard neighbors
            grid_x=8,        # Grid size for histograms
            grid_y=8,
            threshold=80.0   # Lower threshold for better recognition
        )
        
        logger.info("📚 Loading and preprocessing training data...")
        
        # Get training data with enhanced preprocessing
        faces, ids = get_images_and_labels(PATHS['image_dir'])
        
        if not faces or not ids:
            raise ValueError("No training data found. Please run face_taker.py first to collect training images.")
        
        # Convert to numpy arrays
        faces_array = np.array(faces)
        ids_array = np.array(ids)
        
        # Display training statistics
        unique_ids = np.unique(ids_array)
        logger.info(f"📊 Training Statistics:")
        logger.info(f"   Total face samples: {len(faces)}")
        logger.info(f"   Unique persons: {len(unique_ids)}")
        
        for person_id in unique_ids:
            count = np.sum(ids_array == person_id)
            logger.info(f"   Person ID {person_id}: {count} samples")
        
        # Train the LBPH model
        logger.info("🧠 Training enhanced LBPH model...")
        recognizer.train(faces, ids_array)
        
        # Save the LBPH model
        logger.info("💾 Saving LBPH model...")
        recognizer.write(PATHS['trainer_file'])
        
        logger.info("✅ LBPH face recognition model training completed!")
        
        # Display model info
        model_size = os.path.getsize(PATHS['trainer_file']) / (1024 * 1024)  # Size in MB
        logger.info(f"📁 LBPH model file size: {model_size:.2f} MB")
        
        # Train ArcFace embeddings if method is arcface or if ArcFace data exists
        if training_method == 'arcface' or os.path.exists(os.path.join(PATHS['image_dir'], 'arcface_format')):
            logger.info("🚀 Creating ArcFace embeddings for advanced recognition...")
            arcface_success = create_arcface_embeddings()
            
            if arcface_success:
                logger.info("✅ ArcFace embeddings created successfully!")
                logger.info("🎯 Advanced person tracking system ready!")
            else:
                logger.warning("⚠️  ArcFace embeddings creation failed, using LBPH only")
        
        logger.info("🚀 Training completed! Models ready for use with:")
        logger.info("  - Legacy system: face_recognizer.py")
        logger.info("  - Advanced system: advanced_person_tracker.py")
        logger.info(f"🎯 Total persons trained: {len(unique_ids)}")
        logger.info(f"📊 Total samples: {len(faces)}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error("💡 Make sure you have run face_taker.py to collect training images first")
