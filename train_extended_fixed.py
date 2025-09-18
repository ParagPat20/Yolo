import os
import sys
import argparse
import tempfile
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


def create_extended_model(base_model_path: str, num_new_classes: int = 3):
    """
    Create a new model that extends the base model with additional classes.
    This properly handles the architecture extension while preserving pretrained weights.
    """
    # Load the base model
    base_model = YOLO(base_model_path)
    
    # Get the original number of classes (80 for COCO)
    original_nc = base_model.model.nc
    new_nc = original_nc + num_new_classes
    
    print(f"Extending model from {original_nc} to {new_nc} classes")
    
    # Create new model with extended classes
    # We need to modify the model architecture to handle more classes
    model_yaml = base_model.model.yaml
    model_yaml['nc'] = new_nc
    
    # Create new model with extended architecture
    extended_model = DetectionModel(model_yaml, ch=3, nc=new_nc)
    
    # Copy weights from original model (except the final classification layers)
    base_state_dict = base_model.model.state_dict()
    extended_state_dict = extended_model.state_dict()
    
    # Copy all weights except the final classification head
    for name, param in base_state_dict.items():
        if name in extended_state_dict:
            # For classification heads, we need to handle size mismatch
            if 'cv3' in name and param.shape != extended_state_dict[name].shape:
                # This is a classification head - we need to extend it
                print(f"Extending layer {name} from {param.shape} to {extended_state_dict[name].shape}")
                
                # Initialize new weights
                new_param = extended_state_dict[name].clone()
                
                # Copy original weights for original classes
                if len(param.shape) == 4:  # Conv2d weight
                    original_classes_per_anchor = param.shape[0] // 3  # Usually 85 for COCO (80 classes + 5)
                    new_classes_per_anchor = new_param.shape[0] // 3
                    
                    for i in range(3):  # 3 anchors
                        start_orig = i * original_classes_per_anchor
                        end_orig = start_orig + original_classes_per_anchor
                        start_new = i * new_classes_per_anchor
                        end_new = start_new + original_classes_per_anchor
                        
                        new_param[start_new:end_new] = param[start_orig:end_orig]
                elif len(param.shape) == 1:  # Bias
                    original_classes_per_anchor = param.shape[0] // 3
                    new_classes_per_anchor = new_param.shape[0] // 3
                    
                    for i in range(3):  # 3 anchors
                        start_orig = i * original_classes_per_anchor
                        end_orig = start_orig + original_classes_per_anchor
                        start_new = i * new_classes_per_anchor
                        end_new = start_new + original_classes_per_anchor
                        
                        new_param[start_new:end_new] = param[start_orig:end_orig]
                
                extended_state_dict[name] = new_param
            else:
                # Copy weights directly
                extended_state_dict[name] = param
    
    # Load the extended state dict
    extended_model.load_state_dict(extended_state_dict, strict=False)
    
    # Wrap in YOLO object
    extended_yolo = YOLO(model=extended_model)
    extended_yolo.model.nc = new_nc
    
    return extended_yolo


def create_mixed_dataset_yaml(
    train_dir: str,
    val_dir: str,
    test_dir: str = None,
    include_coco_subset: bool = True
) -> str:
    """
    Create a data.yaml that includes both COCO classes and custom classes.
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
                    if new_class >= 80:  # Count new classes
                        converted_count += 1
        
        # Write file even if empty (to maintain dataset structure)
        with open(output_path, 'w') as f:
            f.writelines(converted_lines)
    
    print(f"Converted {converted_count} annotations for new classes")


def main():
    parser = argparse.ArgumentParser(description="Properly extend YOLO11n with custom classes")
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
        default="extended_fixed",
        help="Run name"
    )
    parser.add_argument(
        "--freeze_backbone",
        type=int,
        default=10,
        help="Number of backbone layers to freeze (0 to freeze none)"
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
    
    # Create extended model
    print("Creating extended model...")
    extended_model = create_extended_model(args.model, num_new_classes=3)
    
    # Create temporary directories for converted labels
    temp_dir = tempfile.mkdtemp(prefix="yolo_extended_fixed_")
    temp_train_labels = os.path.join(temp_dir, "train", "labels")
    temp_val_labels = os.path.join(temp_dir, "valid", "labels") 
    temp_test_labels = os.path.join(temp_dir, "test", "labels") if test_dir else None
    
    print(f"Converting labels to extended format in: {temp_dir}")
    
    # Convert labels for each split
    convert_labels_for_mixed_training(
        os.path.join(train_dir, "labels"),
        temp_train_labels
    )
    
    convert_labels_for_mixed_training(
        os.path.join(val_dir, "labels"), 
        temp_val_labels
    )
    
    if test_dir:
        convert_labels_for_mixed_training(
            os.path.join(test_dir, "labels"),
            temp_test_labels
        )
    
    # Create symlinks to image directories
    import shutil
    temp_train_images = os.path.join(temp_dir, "train", "images")
    temp_val_images = os.path.join(temp_dir, "valid", "images")
    temp_test_images = os.path.join(temp_dir, "test", "images") if test_dir else None
    
    try:
        os.symlink(os.path.join(train_dir, "images"), temp_train_images)
        os.symlink(os.path.join(val_dir, "images"), temp_val_images)
        if test_dir:
            os.symlink(os.path.join(test_dir, "images"), temp_test_images)
        print("Created symlinks to image directories")
    except OSError:
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
    
    print("Created data.yaml:")
    print(data_yaml_content)
    
    # Train the extended model
    print(f"Starting training with extended model (83 classes)...")
    print(f"Freezing first {args.freeze_backbone} layers for transfer learning...")
    
    results = extended_model.train(
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
        freeze=args.freeze_backbone  # Freeze backbone layers
    )
    
    print(f"Training completed!")
    print(f"Best weights: {results.best if hasattr(results, 'best') else 'Check run directory'}")
    print(f"Temporary files in: {temp_dir}")
    
    # Test the model to verify it works
    print("\nTesting extended model...")
    test_model = YOLO(results.best if hasattr(results, 'best') else f"{args.project}/{args.name}/weights/best.pt")
    print(f"Model has {test_model.model.nc} classes")
    print(f"Class names: {test_model.names}")


if __name__ == "__main__":
    main()
