import os
import sys
import argparse
import glob
import time
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading

import cv2
import numpy as np
from ultralytics import YOLO


class ClassToggleGUI:
    def __init__(self, class_names, initial_enabled=None):
        self.class_names = class_names
        self.enabled_classes = set(range(len(class_names))) if initial_enabled is None else set(initial_enabled)
        self.result = None
        
        self.root = tk.Tk()
        self.root.title("YOLO Class Toggle")
        self.root.geometry("800x600")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Select Classes to Detect", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="Select All", command=self.select_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Deselect All", command=self.deselect_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Select Vehicles Only", command=self.select_vehicles).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Select Custom Only", command=self.select_custom).pack(side=tk.LEFT, padx=(0, 5))
        
        # Create scrollable frame for checkboxes
        canvas = tk.Canvas(main_frame, height=400)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=2, column=2, sticky=(tk.N, tk.S))
        
        # Create checkboxes
        self.checkboxes = {}
        self.checkbox_vars = {}
        
        # Group classes for better organization
        vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'motorbike', 'bicycle', 'airplane', 'aeroplane', 'boat', 'train'}
        custom_classes = {'crane-', 'jcb', 'numplate'}
        person_classes = {'person'}
        
        row = 0
        for class_idx, class_name in enumerate(class_names):
            var = tk.BooleanVar(value=class_idx in self.enabled_classes)
            self.checkbox_vars[class_idx] = var
            
            # Color code different types of classes
            if class_name.lower() in vehicle_classes:
                bg_color = "#e3f2fd"  # Light blue for vehicles
            elif class_name.lower() in custom_classes:
                bg_color = "#fff3e0"  # Light orange for custom classes
            elif class_name.lower() in person_classes:
                bg_color = "#f3e5f5"  # Light purple for person
            else:
                bg_color = "#f5f5f5"  # Light gray for others
            
            frame = ttk.Frame(scrollable_frame)
            frame.grid(row=row, column=0, sticky=(tk.W, tk.E), padx=5, pady=2)
            
            checkbox = ttk.Checkbutton(
                frame,
                text=f"{class_idx}: {class_name}",
                variable=var,
                width=50
            )
            checkbox.pack(side=tk.LEFT)
            
            self.checkboxes[class_idx] = checkbox
            row += 1
        
        # Apply and Cancel buttons
        button_frame2 = ttk.Frame(main_frame)
        button_frame2.grid(row=3, column=0, columnspan=3, pady=(20, 0))
        
        ttk.Button(button_frame2, text="Apply", command=self.apply).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame2, text="Cancel", command=self.cancel).pack(side=tk.LEFT)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
    
    def select_all(self):
        for var in self.checkbox_vars.values():
            var.set(True)
    
    def deselect_all(self):
        for var in self.checkbox_vars.values():
            var.set(False)
    
    def select_vehicles(self):
        vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'motorbike', 'bicycle', 'airplane', 'aeroplane', 'boat', 'train'}
        for class_idx, class_name in enumerate(self.class_names):
            self.checkbox_vars[class_idx].set(class_name.lower() in vehicle_classes)
    
    def select_custom(self):
        custom_classes = {'crane-', 'jcb', 'numplate'}
        for class_idx, class_name in enumerate(self.class_names):
            self.checkbox_vars[class_idx].set(class_name.lower() in custom_classes)
    
    def apply(self):
        self.result = [idx for idx, var in self.checkbox_vars.items() if var.get()]
        self.root.quit()
        self.root.destroy()
    
    def cancel(self):
        self.result = None
        self.root.quit()
        self.root.destroy()
    
    def show(self):
        self.root.mainloop()
        return self.result


def parse_classes_argument(classes_str, total_classes):
    """Parse the --classes argument to get enabled class indices."""
    if not classes_str:
        return list(range(total_classes))  # All classes enabled by default
    
    enabled = []
    parts = classes_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            # Range like "0-5"
            start, end = map(int, part.split('-'))
            enabled.extend(range(start, end + 1))
        else:
            # Single class index
            enabled.append(int(part))
    
    return list(set(enabled))  # Remove duplicates


def save_class_config(enabled_classes, filename="class_config.json"):
    """Save enabled classes to a configuration file."""
    with open(filename, 'w') as f:
        json.dump({"enabled_classes": enabled_classes}, f)


def load_class_config(filename="class_config.json"):
    """Load enabled classes from a configuration file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return data.get("enabled_classes", None)
    except FileNotFoundError:
        return None


def main():
    # Define and parse user input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                        required=True)
    parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                        image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                        required=True)
    parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                        default=0.5, type=float)
    parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                        otherwise, match source resolution',
                        default=None)
    parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                        action='store_true')
    parser.add_argument('--classes', help='Comma-separated list of class indices to detect (example: "0,2,5" or "0-10,15,20-25"). Leave empty for all classes.',
                        default=None)
    parser.add_argument('--gui', help='Show GUI for class selection before starting detection',
                        action='store_true')
    parser.add_argument('--save-config', help='Save selected classes to configuration file',
                        action='store_true')
    parser.add_argument('--load-config', help='Load previously saved class configuration',
                        action='store_true')
    
    args = parser.parse_args()
    
    # Parse user inputs
    model_path = args.model
    img_source = args.source
    min_thresh = args.thresh
    user_res = args.resolution
    record = args.record
    
    # Check if model file exists and is valid
    if not os.path.exists(model_path):
        print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
        sys.exit(1)
    
    # Load the model into memory and get label map
    print("Loading model...")
    model = YOLO(model_path, task='detect')
    labels = model.names
    total_classes = len(labels)
    
    print(f"Model loaded with {total_classes} classes:")
    for idx, name in labels.items():
        print(f"  {idx}: {name}")
    
    # Determine which classes to enable
    enabled_classes = None
    
    # Try to load from config file if requested
    if args.load_config:
        enabled_classes = load_class_config()
        if enabled_classes:
            print(f"Loaded class configuration: {enabled_classes}")
    
    # Parse command line classes argument
    if args.classes and not enabled_classes:
        try:
            enabled_classes = parse_classes_argument(args.classes, total_classes)
            print(f"Using classes from command line: {enabled_classes}")
        except ValueError as e:
            print(f"Error parsing --classes argument: {e}")
            sys.exit(1)
    
    # Show GUI if requested or if no classes specified
    if args.gui or not enabled_classes:
        print("Opening class selection GUI...")
        gui = ClassToggleGUI(list(labels.values()), enabled_classes)
        enabled_classes = gui.show()
        
        if enabled_classes is None:
            print("Class selection cancelled. Exiting.")
            sys.exit(0)
        
        print(f"Selected classes: {enabled_classes}")
    
    # Default to all classes if none specified
    if not enabled_classes:
        enabled_classes = list(range(total_classes))
    
    # Save configuration if requested
    if args.save_config:
        save_class_config(enabled_classes)
        print("Class configuration saved.")
    
    # Create set for faster lookup
    enabled_classes_set = set(enabled_classes)
    
    # Print enabled classes
    print("\nEnabled classes:")
    for class_idx in enabled_classes:
        if class_idx in labels:
            print(f"  {class_idx}: {labels[class_idx]}")
    
    # Parse input to determine if image source is a file, folder, video, or USB camera
    img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
    vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']
    
    if os.path.isdir(img_source):
        source_type = 'folder'
    elif os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        if ext in img_ext_list:
            source_type = 'image'
        elif ext in vid_ext_list:
            source_type = 'video'
        else:
            print(f'File extension {ext} is not supported.')
            sys.exit(1)
    elif 'usb' in img_source:
        source_type = 'usb'
        usb_idx = int(img_source[3:])
    elif 'picamera' in img_source:
        source_type = 'picamera'
        picam_idx = int(img_source[8:])
    else:
        print(f'Input {img_source} is invalid. Please try again.')
        sys.exit(1)
    
    # Parse user-specified display resolution
    resize = False
    if user_res:
        resize = True
        resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
    
    # Check if recording is valid and set up recording
    if record:
        if source_type not in ['video','usb']:
            print('Recording only works for video and camera sources. Please try again.')
            sys.exit(1)
        if not user_res:
            print('Please specify resolution to record video at.')
            sys.exit(1)
        
        # Set up recording
        record_name = 'demo1.avi'
        record_fps = 30
        recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))
    
    # Load or initialize image source
    if source_type == 'image':
        imgs_list = [img_source]
    elif source_type == 'folder':
        imgs_list = []
        filelist = glob.glob(img_source + '/*')
        for file in filelist:
            _, file_ext = os.path.splitext(file)
            if file_ext in img_ext_list:
                imgs_list.append(file)
    elif source_type == 'video' or source_type == 'usb':
        if source_type == 'video': 
            cap_arg = img_source
        elif source_type == 'usb': 
            cap_arg = usb_idx
        cap = cv2.VideoCapture(cap_arg)
        
        # Set camera or video resolution if specified by user
        if user_res:
            ret = cap.set(3, resW)
            ret = cap.set(4, resH)
    
    elif source_type == 'picamera':
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
        cap.start()
    
    # Set bounding box colors (using the Tableau 10 color scheme)
    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
                  (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
    
    # Special colors for custom classes
    custom_colors = {
        'crane-': (255, 165, 0),    # Orange
        'jcb': (255, 20, 147),      # Deep pink
        'numplate': (0, 255, 0)     # Green
    }
    
    # Initialize control and status variables
    avg_frame_rate = 0
    frame_rate_buffer = []
    fps_avg_len = 200
    img_count = 0
    
    print(f"\nStarting detection with confidence threshold: {min_thresh}")
    print("Press 'q' to quit, 's' to pause, 'p' to save screenshot, 'c' to change classes")
    
    # Begin inference loop
    while True:
        t_start = time.perf_counter()
        
        # Load frame from image source
        if source_type == 'image' or source_type == 'folder':
            if img_count >= len(imgs_list):
                print('All images have been processed. Exiting program.')
                break
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count = img_count + 1
        
        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret:
                print('Reached end of the video file. Exiting program.')
                break
        
        elif source_type == 'usb':
            ret, frame = cap.read()
            if (frame is None) or (not ret):
                print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
                break
        
        elif source_type == 'picamera':
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
            if (frame is None):
                print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
                break
        
        # Resize frame to desired display resolution
        if resize:
            frame = cv2.resize(frame, (resW, resH))
        
        # Run inference on frame
        results = model(frame, verbose=False)
        
        # Extract results
        detections = results[0].boxes
        
        # Initialize counters
        object_count = 0
        class_counts = {}
        
        # Go through each detection and get bbox coords, confidence, and class
        if detections is not None:
            for i in range(len(detections)):
                # Get bounding box class ID and name
                classidx = int(detections[i].cls.item())
                
                # Skip if this class is not enabled
                if classidx not in enabled_classes_set:
                    continue
                
                classname = labels[classidx]
                
                # Get bounding box confidence
                conf = detections[i].conf.item()
                
                # Draw box if confidence threshold is high enough
                if conf > min_thresh:
                    # Get bounding box coordinates
                    xyxy_tensor = detections[i].xyxy.cpu()
                    xyxy = xyxy_tensor.numpy().squeeze()
                    xmin, ymin, xmax, ymax = xyxy.astype(int)
                    
                    # Choose color
                    if classname.lower() in custom_colors:
                        color = custom_colors[classname.lower()]
                    else:
                        color = bbox_colors[classidx % len(bbox_colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    
                    # Draw label
                    label = f'{classname}: {int(conf*100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # Count objects
                    object_count += 1
                    class_counts[classname] = class_counts.get(classname, 0) + 1
        
        # Calculate and draw framerate (if using video, USB, or Picamera source)
        if source_type in ['video', 'usb', 'picamera']:
            cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        
        # Display detection results
        cv2.putText(frame, f'Total objects: {object_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        
        # Display class-specific counts
        y_offset = 80
        for class_name, count in class_counts.items():
            cv2.putText(frame, f'{class_name}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
        
        # Show enabled classes info
        enabled_class_names = [labels[idx] for idx in enabled_classes if idx in labels]
        cv2.putText(frame, f'Enabled: {len(enabled_classes)} classes', (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f'Threshold: {min_thresh}', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow('YOLO Detection Results', frame)
        if record: 
            recorder.write(frame)
        
        # Handle key presses
        if source_type in ['image', 'folder']:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(5)
        
        if key == ord('q') or key == ord('Q'):  # Press 'q' to quit
            break
        elif key == ord('s') or key == ord('S'):  # Press 's' to pause inference
            cv2.waitKey(0)
        elif key == ord('p') or key == ord('P'):  # Press 'p' to save a picture of results on this frame
            cv2.imwrite('capture.png', frame)
            print("Screenshot saved as capture.png")
        elif key == ord('c') or key == ord('C'):  # Press 'c' to change classes
            print("Opening class selection...")
            gui = ClassToggleGUI(list(labels.values()), enabled_classes)
            new_enabled = gui.show()
            if new_enabled is not None:
                enabled_classes = new_enabled
                enabled_classes_set = set(enabled_classes)
                print(f"Updated enabled classes: {enabled_classes}")
        
        # Calculate FPS for this frame
        if source_type in ['video', 'usb', 'picamera']:
            t_stop = time.perf_counter()
            frame_rate_calc = float(1/(t_stop - t_start))
            
            # Append FPS result to frame_rate_buffer
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
            
            # Calculate average FPS
            avg_frame_rate = np.mean(frame_rate_buffer)
    
    # Clean up
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    if source_type in ['video', 'usb']:
        cap.release()
    elif source_type == 'picamera':
        cap.stop()
    if record: 
        recorder.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
