import os
import sys
import argparse
import tempfile
import shutil
from ultralytics import YOLO


def create_mixed_dataset_yaml(
    train_dir: str,
    val_dir: str,
    test_dir: str = None
) -> str:
    """
    Create a data.yaml that includes both COCO classes and custom classes.
    Uses the simple approach: just add custom classes as new indices.
    """
    
    # All 80 COCO class names
    coco_names = [
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
    
    # Custom classes to add
    custom_names = ['NUM_PLATE', 'crane-', 'jcb']
    
    # Combine all classes
    all_names = coco_names + custom_names
    nc = len(all_names)
    
    yaml_lines = [
        f"path: {os.path.dirname(os.path.abspath(train_dir))}",
        f"train: {os.path.basename(train_dir)}",
        f"val: {os.path.basename(val_dir)}",
    ]
    
    if test_dir:
        yaml_lines.append(f"test: {os.path.basename(test_dir)}")
    
    yaml_lines.extend([
        f"",
        f"nc: {nc}",
        f"names: {all_names}"
    ])
    
    return "\n".join(yaml_lines) + "\n"


def convert_labels_for_mixed_training(
    labels_dir: str,
    output_dir: str
) -> None:
    """
    Convert existing label files to work with extended class system.
    Maps your custom classes to the new extended indices (80, 81, 82).
    """
    
    # Class mapping from your dataset to extended COCO
    class_mapping = {
        0: 2,     # CAR -> COCO car (class 2)
        1: 7,     # Heavy_Vehicle -> COCO truck (class 7)  
        2: 80,    # NUM_PLATE -> new class 80
        3: 0,     # Person -> COCO person (class 0)
        4: 81,    # crane- -> new class 81
        5: 82     # jcb -> new class 82
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    converted_count = 0
    new_class_count = 0
    
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
            
        input_path = os.path.join(labels_dir, label_file)
        output_path = os.path.join(output_dir, label_file)
        
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        converted_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                old_class = int(parts[0])
                new_class = class_mapping.get(old_class)
                
                if new_class is not None:
                    parts[0] = str(new_class)
                    converted_lines.append(' '.join(parts) + '\n')
                    converted_count += 1
                    if new_class >= 80:  # Count new classes
                        new_class_count += 1
        
        # Write file even if empty (to maintain dataset structure)
        with open(output_path, 'w') as f:
            f.writelines(converted_lines)
    
    print(f"Converted {converted_count} total annotations ({new_class_count} for new classes)")


def download_coco_subset(output_dir: str, num_images_per_class: int = 50):
    """
    Download a small subset of COCO data to help preserve original class knowledge.
    This is a placeholder - in practice you'd want to include some COCO data.
    """
    print(f"Note: For best results, consider adding some COCO images to {output_dir}")
    print("This helps preserve the model's knowledge of the original 80 classes.")
    return False  # Indicates no COCO data was added


def main():
    parser = argparse.ArgumentParser(description="Simple approach to extend YOLO11n with custom classes")
    parser.add_argument(
        "--data_root", 
        type=str, 
        default=".",
        help="Root directory containing train/, valid/, test/ folders"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolo11n.pt",
        help="Base model to extend (default: yolo11n.pt)"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size") 
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Training device")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate (lower for fine-tuning)")
    parser.add_argument(
        "--project", 
        type=str, 
        default="runs/detect",
        help="Project directory"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="extended_simple",
        help="Run name"
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Number of layers to freeze (0 to freeze none, higher numbers freeze more)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience"
    )
    
    args = parser.parse_args()
    
    # Verify directories exist
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "valid")
    test_dir = os.path.join(args.data_root, "test")
    
    for dir_path, name in [(train_dir, "train"), (val_dir, "valid")]:
        if not os.path.exists(dir_path):
            print(f"ERROR: {name} directory not found: {dir_path}")
            sys.exit(1)
    
    if not os.path.exists(test_dir):
        test_dir = None
        print("WARNING: test directory not found, proceeding without test set")
    
    # Verify model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    
    # Create temporary directories for converted labels
    temp_dir = tempfile.mkdtemp(prefix="yolo_extended_simple_")
    temp_train_labels = os.path.join(temp_dir, "train", "labels")
    temp_val_labels = os.path.join(temp_dir, "valid", "labels") 
    temp_test_labels = os.path.join(temp_dir, "test", "labels") if test_dir else None
    
    print(f"Converting labels to extended format in: {temp_dir}")
    
    # Convert labels for each split
    print("Converting training labels...")
    convert_labels_for_mixed_training(
        os.path.join(train_dir, "labels"),
        temp_train_labels
    )
    
    print("Converting validation labels...")
    convert_labels_for_mixed_training(
        os.path.join(val_dir, "labels"), 
        temp_val_labels
    )
    
    if test_dir:
        print("Converting test labels...")
        convert_labels_for_mixed_training(
            os.path.join(test_dir, "labels"),
            temp_test_labels
        )
    
    # Create symlinks/copies to image directories
    temp_train_images = os.path.join(temp_dir, "train", "images")
    temp_val_images = os.path.join(temp_dir, "valid", "images")
    temp_test_images = os.path.join(temp_dir, "test", "images") if test_dir else None
    
    try:
        os.symlink(os.path.join(train_dir, "images"), temp_train_images)
        os.symlink(os.path.join(val_dir, "images"), temp_val_images)
        if test_dir:
            os.symlink(os.path.join(test_dir, "images"), temp_test_images)
        print("Created symlinks to image directories")
    except (OSError, NotImplementedError):
        print("Symlinks not supported, copying image directories...")
        shutil.copytree(os.path.join(train_dir, "images"), temp_train_images)
        shutil.copytree(os.path.join(val_dir, "images"), temp_val_images)
        if test_dir:
            shutil.copytree(os.path.join(test_dir, "images"), temp_test_images)
        print("Copied image directories")
    
    # Create data.yaml
    data_yaml_content = create_mixed_dataset_yaml(
        temp_train_images,
        temp_val_images, 
        temp_test_images
    )
    
    data_yaml_path = os.path.join(temp_dir, "data.yaml")
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print("\nCreated data.yaml:")
    print(data_yaml_content)
    
    # Load model
    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)
    
    # Train the model
    print(f"Starting training with extended classes (83 total)...")
    print(f"Using learning rate: {args.lr0}")
    print(f"Freezing first {args.freeze} layers")
    
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,  # Lower learning rate for fine-tuning
        project=args.project,
        name=args.name,
        exist_ok=True,
        freeze=args.freeze,  # Freeze backbone layers
        patience=args.patience,  # Early stopping
        save=True,
        plots=True
    )
    
    print(f"\nTraining completed!")
    best_model_path = f"{args.project}/{args.name}/weights/best.pt"
    print(f"Best weights: {best_model_path}")
    
    # Test the model to verify it works
    print("\nTesting extended model...")
    try:
        test_model = YOLO(best_model_path)
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model has {test_model.model.nc} classes")
        print(f"🏷️  Class names: {list(test_model.names.values())}")
        
        # Show which classes are new
        if test_model.model.nc >= 83:
            print(f"🆕 New classes: {list(test_model.names.values())[80:]}")
        
    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
    
    print(f"\n📁 Temporary files in: {temp_dir}")
    print("💡 You can clean up the temporary directory after verifying the model works")
    
    return best_model_path


if __name__ == "__main__":
    main()
