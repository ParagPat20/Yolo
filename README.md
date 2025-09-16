# YOLOv8 Vehicle Detection Project

This project provides complete training and inference scripts for YOLOv8-based vehicle detection. It can detect three types of vehicles: **mobil** (cars), **motor** (motorcycles), and **truck** (trucks).

## 📁 Project Structure

```
Yolo/
├── data.yaml                 # Dataset configuration
├── object_trainer.py         # Training script
├── object_detector.py        # Detection/inference script
├── detector_tracker.py       # Lightweight detector + tracker (OpenCV DNN)
├── convert_to_onnx.py        # Convert YOLOv8 to ONNX format
├── test_tracker.py           # Test script for tracker functionality
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── train/                    # Training dataset
│   ├── images/              # Training images
│   └── labels/              # Training labels (YOLO format)
├── valid/                    # Validation dataset
│   ├── images/              # Validation images
│   └── labels/              # Validation labels
└── test/                     # Test dataset
    ├── images/              # Test images
    └── labels/              # Test labels
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Basic training with default settings
python object_trainer.py

# Advanced training with custom parameters
python object_trainer.py --model s --epochs 150 --batch 32 --imgsz 640
```

### 3. Convert Model to ONNX (for lightweight tracking)

```bash
# Convert trained YOLOv8 model to ONNX format
python convert_to_onnx.py --model runs/detect/train/weights/best.pt --output best.onnx
```

### 4. Run Detection

```bash
# Standard YOLO detection (every frame)
python object_detector.py --model runs/detect/train/weights/best.pt --source test/images/image.jpg --show

# Lightweight detection + tracking (much faster for video)
python detector_tracker.py --model best.onnx --source 0 --tracker CSRT

# Process a video with tracking
python detector_tracker.py --model best.onnx --source video.mp4 --output output_video.mp4
```

## 📖 Detailed Usage

### Training Script (`object_trainer.py`)

The training script provides comprehensive functionality for training YOLOv8 models on your vehicle dataset.

#### Command Line Arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `data.yaml` | Path to data configuration file |
| `--model` | str | `n` | Model size (n/s/m/l/x) |
| `--epochs` | int | `100` | Number of training epochs |
| `--imgsz` | int | `640` | Input image size |
| `--batch` | int | `16` | Batch size |
| `--patience` | int | `50` | Early stopping patience |
| `--validate` | flag | False | Run validation after training |
| `--export` | str | None | Export format (onnx/tensorrt/coreml) |

#### Examples:

```bash
# Train YOLOv8 nano model
python object_trainer.py --model n --epochs 100

# Train YOLOv8 small model with validation
python object_trainer.py --model s --epochs 200 --batch 32 --validate

# Train and export to ONNX format
python object_trainer.py --model m --epochs 150 --export onnx
```

#### Model Sizes:
- **n (nano)**: Fastest, smallest model
- **s (small)**: Good balance of speed and accuracy
- **m (medium)**: Better accuracy, moderate speed
- **l (large)**: High accuracy, slower
- **x (extra large)**: Highest accuracy, slowest

### Lightweight Detector + Tracker (`detector_tracker.py`)

**🚀 NEW: Ultra-fast detection with OpenCV tracking!**

This script combines YOLO detection with OpenCV tracking for dramatically improved performance:
- **Detects** objects only when needed (every ~30 frames or when tracking fails)
- **Tracks** objects between detections using lightweight OpenCV trackers
- **5-10x faster** than running YOLO on every frame
- Perfect for real-time applications and video processing

#### Key Features:
- **Smart Detection**: Only runs YOLO when necessary
- **Multiple Trackers**: CSRT (best), KCF (fast), MOSSE (fastest)
- **Automatic Re-detection**: When tracking fails or after N frames
- **Target Selection**: Focus on specific object classes
- **Real-time Performance**: Optimized for live video streams

#### Command Line Arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | Required | Path to ONNX model |
| `--source` | str | `0` | Video source (path/camera) |
| `--output` | str | None | Output video path |
| `--tracker` | str | `CSRT` | Tracker type (CSRT/KCF/MOSSE) |
| `--target-class` | int | None | Preferred class ID to track |
| `--max-track-frames` | int | `30` | Re-detect after N frames |
| `--conf` | float | `0.5` | Detection confidence threshold |
| `--nms` | float | `0.4` | NMS threshold |

#### Examples:

```bash
# Real-time webcam with CSRT tracker
python detector_tracker.py --model best.onnx --source 0 --tracker CSRT

# Process video, track cars only
python detector_tracker.py --model best.onnx --source traffic.mp4 --target-class 0 --output result.mp4

# Ultra-fast tracking with MOSSE
python detector_tracker.py --model best.onnx --source 0 --tracker MOSSE --max-track-frames 50
```

### Detection Script (`object_detector.py`)

The standard detection script runs YOLO on every frame and provides comprehensive inference options.

#### Command Line Arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | Required | Path to trained model weights |
| `--source` | str | Required | Input source (image/video/webcam) |
| `--output` | str | None | Output path for results |
| `--conf` | float | `0.5` | Confidence threshold |
| `--iou` | float | `0.45` | IoU threshold for NMS |
| `--show` | flag | False | Display results |
| `--camera` | int | `0` | Camera ID for webcam |

#### Examples:

```bash
# Detect in single image
python object_detector.py --model best.pt --source image.jpg --output result.jpg --show

# Process video with custom thresholds
python object_detector.py --model best.pt --source video.mp4 --output output.mp4 --conf 0.6 --iou 0.4

# Real-time webcam detection
python object_detector.py --model best.pt --source webcam --show --camera 0

# Batch process multiple images
for img in test/images/*.jpg; do
    python object_detector.py --model best.pt --source "$img" --output "results/$(basename $img)" --conf 0.5
done
```

## 🎯 Dataset Configuration

Your `data.yaml` file should follow this format:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 3
names: ['mobil', 'motor', 'truck']
```

### Label Format

Labels should be in YOLO format (one `.txt` file per image):
```
class_id center_x center_y width height
```

Where all coordinates are normalized (0-1).

Example:
```
0 0.5 0.5 0.3 0.4  # mobil at center, 30% width, 40% height
1 0.2 0.3 0.1 0.2  # motor at top-left area
2 0.8 0.7 0.4 0.3  # truck at bottom-right area
```

## 📊 Training Output

After training, you'll find results in `runs/detect/train*/`:

```
runs/detect/train/
├── weights/
│   ├── best.pt          # Best model weights
│   └── last.pt          # Last epoch weights
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── val_batch0_labels.jpg # Validation samples
└── ...
```

## 🔧 Customization

### Modify Class Names

Edit `data.yaml` and update the detector script:

```python
# In object_detector.py
self.class_names = ['car', 'motorcycle', 'truck']  # Your custom names
```

### Adjust Detection Colors

```python
# In object_detector.py
self.colors = {
    'mobil': (255, 0, 0),    # Blue
    'motor': (0, 255, 0),    # Green
    'truck': (0, 0, 255),    # Red
}
```

## 📈 Performance Tips

1. **Model Selection**:
   - Use YOLOv8n for real-time applications
   - Use YOLOv8s/m for balanced performance
   - Use YOLOv8l/x for highest accuracy

2. **Training Tips**:
   - Start with 100-200 epochs
   - Use larger batch sizes if GPU memory allows
   - Monitor validation loss for early stopping

3. **Detection Tips**:
   - Adjust confidence threshold based on your needs
   - Lower conf = more detections (including false positives)
   - Higher conf = fewer, more confident detections

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   python object_trainer.py --batch 8
   ```

2. **Model Not Found**:
   ```bash
   # Check model path
   python object_detector.py --model runs/detect/train/weights/best.pt --source image.jpg
   ```

3. **Low Detection Accuracy**:
   - Check label quality
   - Increase training epochs
   - Use larger model size
   - Adjust confidence threshold

## 📝 License

This project is for educational and research purposes. Please ensure compliance with your dataset's license terms.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.
