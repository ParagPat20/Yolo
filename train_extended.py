import os
import sys
import argparse
import tempfile
from ultralytics import YOLO


def create_extended_data_yaml(
    train_dir: str,
    val_dir: str,
    test_dir: str = None,
    custom_classes_only: bool = False
) -> str:
    """
    Create a data.yaml that extends YOLO's 80 COCO classes with custom classes.
    
    Args:
        train_dir: Path to training images directory
        val_dir: Path to validation images directory  
        test_dir: Optional path to test images directory
        custom_classes_only: If True, only train on custom classes (NUM_PLATE, crane-, jcb)
    
    Returns:
        String content of the data.yaml file
    """
    
    # COCO class names (first 80 classes that YOLO11n already knows)
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
    
    # Custom classes to add (these will be classes 80, 81, 82)
    custom_names = ['NUM_PLATE', 'crane-', 'jcb']
    
    if custom_classes_only:
        # Train only on custom classes (renumber them to 0, 1, 2)
        all_names = custom_names
        nc = len(custom_names)
    else:
        # Extend COCO with custom classes
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


def convert_labels_for_extension(
    labels_dir: str,
    output_dir: str,
    custom_classes_only: bool = False
) -> None:
    """
    Convert existing label files to work with extended class system.
    
    Current mapping in your dataset:
    - Class 0: CAR -> COCO class 2 (car) 
    - Class 1: Heavy_Vehicle -> COCO class 7 (truck)
    - Class 2: NUM_PLATE -> New class 80 (or 0 if custom_classes_only)
    - Class 3: Person -> COCO class 0 (person)
    - Class 4: crane- -> New class 81 (or 1 if custom_classes_only)  
    - Class 5: jcb -> New class 82 (or 2 if custom_classes_only)
    """
    
    if custom_classes_only:
        # Map to custom classes only (0, 1, 2)
        class_mapping = {
            0: None,  # CAR -> skip (already in COCO)
            1: None,  # Heavy_Vehicle -> skip (already in COCO as truck)
            2: 0,     # NUM_PLATE -> class 0
            3: None,  # Person -> skip (already in COCO)
            4: 1,     # crane- -> class 1
            5: 2      # jcb -> class 2
        }
    else:
        # Map to extended COCO classes (80, 81, 82 for new classes)
        class_mapping = {
            0: 2,     # CAR -> COCO car (class 2)
            1: 7,     # Heavy_Vehicle -> COCO truck (class 7)  
            2: 80,    # NUM_PLATE -> new class 80
            3: 0,     # Person -> COCO person (class 0)
            4: 81,    # crane- -> new class 81
            5: 82     # jcb -> new class 82
        }
    
    os.makedirs(output_dir, exist_ok=True)
    
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
                
                if new_class is not None:  # Keep this annotation
                    parts[0] = str(new_class)
                    converted_lines.append(' '.join(parts) + '\n')
                # If new_class is None, skip this annotation (already covered by COCO)
        
        # Only write file if it has annotations
        if converted_lines:
            with open(output_path, 'w') as f:
                f.writelines(converted_lines)


def main():
    parser = argparse.ArgumentParser(description="Extend YOLO11n with custom classes")
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
    parser.add_argument(
        "--custom_only",
        action="store_true", 
        help="Train only on custom classes (NUM_PLATE, crane-, jcb), ignore COCO classes"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size") 
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Training device")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument(
        "--project", 
        type=str, 
        default="runs/detect",
        help="Project directory"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="train_extended",
        help="Run name"
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
    
    # Create temporary directories for converted labels
    temp_dir = tempfile.mkdtemp(prefix="yolo_extended_")
    temp_train_labels = os.path.join(temp_dir, "train", "labels")
    temp_val_labels = os.path.join(temp_dir, "valid", "labels") 
    temp_test_labels = os.path.join(temp_dir, "test", "labels") if test_dir else None
    
    print(f"Converting labels to extended format in: {temp_dir}")
    
    # Convert labels for each split
    convert_labels_for_extension(
        os.path.join(train_dir, "labels"),
        temp_train_labels,
        args.custom_only
    )
    
    convert_labels_for_extension(
        os.path.join(val_dir, "labels"), 
        temp_val_labels,
        args.custom_only
    )
    
    if test_dir:
        convert_labels_for_extension(
            os.path.join(test_dir, "labels"),
            temp_test_labels, 
            args.custom_only
        )
    
    # Copy/link image directories to temp structure
    import shutil
    temp_train_images = os.path.join(temp_dir, "train", "images")
    temp_val_images = os.path.join(temp_dir, "valid", "images")
    temp_test_images = os.path.join(temp_dir, "test", "images") if test_dir else None
    
    # Create symlinks to avoid copying large image files
    try:
        os.symlink(os.path.join(train_dir, "images"), temp_train_images)
        os.symlink(os.path.join(val_dir, "images"), temp_val_images)
        if test_dir:
            os.symlink(os.path.join(test_dir, "images"), temp_test_images)
        print("Created symlinks to image directories")
    except OSError:
        # Fallback to copying if symlinks not supported
        shutil.copytree(os.path.join(train_dir, "images"), temp_train_images)
        shutil.copytree(os.path.join(val_dir, "images"), temp_val_images)
        if test_dir:
            shutil.copytree(os.path.join(test_dir, "images"), temp_test_images)
        print("Copied image directories (symlinks not supported)")
    
    # Create data.yaml
    data_yaml_content = create_extended_data_yaml(
        temp_train_images,
        temp_val_images, 
        temp_test_images,
        args.custom_only
    )
    
    data_yaml_path = os.path.join(temp_dir, "data.yaml")
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print("Created data.yaml:")
    print(data_yaml_content)
    
    # Load model and train
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    
    model = YOLO(args.model)
    
    print(f"Starting training with {'custom classes only' if args.custom_only else 'extended COCO classes'}...")
    
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True
    )
    
    print(f"Training completed!")
    print(f"Best weights: {results.best if hasattr(results, 'best') else 'Check run directory'}")
    print(f"Temporary files in: {temp_dir}")
    print("Note: You may want to clean up the temporary directory after training")


if __name__ == "__main__":
    main()
