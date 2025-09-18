import os
import shutil

def convert_labels_for_extended_training():
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
    
    # Process each split
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        labels_dir = os.path.join(split, 'labels')
        if not os.path.exists(labels_dir):
            print(f"Skipping {split} - labels directory not found")
            continue
            
        # Create backup
        backup_dir = f"{labels_dir}_original"
        if not os.path.exists(backup_dir):
            print(f"Creating backup: {backup_dir}")
            shutil.copytree(labels_dir, backup_dir)
        
        converted_count = 0
        new_class_count = 0
        
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            file_path = os.path.join(labels_dir, label_file)
            
            # Read original labels
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Convert labels
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
            
            # Write converted labels
            with open(file_path, 'w') as f:
                f.writelines(converted_lines)
        
        print(f"{split}: Converted {converted_count} annotations ({new_class_count} for new classes)")

if __name__ == "__main__":
    print("Converting labels for extended training...")
    convert_labels_for_extended_training()
    print("Done! Original labels backed up to *_original directories")
    print("You can now train with: yolo detect train model=yolo11n.pt data=data.yaml epochs=50 batch=16")
