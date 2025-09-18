#!/usr/bin/env python3

"""
FTP Video Downloader with YOLO Detection
Downloads video files from Raspberry Pi FTP server and performs YOLO detection
Usage: python3 download_and_detect.py [options]
"""

import ftplib
import os
import sys
import argparse
from datetime import datetime, timedelta
import re
import cv2
import json
import threading
from queue import Queue
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

def connect_ftp(host, username=None, password=None, port=21):
    """Connect to FTP server (supports anonymous login)"""
    try:
        ftp = ftplib.FTP()
        ftp.connect(host, port)
        
        # If no username provided or username is 'anonymous', use anonymous login
        if not username or username.lower() == 'anonymous':
            ftp.login()  # Anonymous login
            print(f"✅ Connected to {host} (anonymous)")
        else:
            ftp.login(username, password or '')
            print(f"✅ Connected to {host} (user: {username})")
        
        return ftp
    except Exception as e:
        print(f"❌ Failed to connect to FTP server: {e}")
        return None

def list_video_files(ftp, remote_dir="recordings"):
    """List all video files on the server"""
    try:
        # Try to change to recordings directory
        try:
            ftp.cwd(remote_dir)
            print(f"📁 Browsing directory: /{remote_dir}/")
        except:
            # If recordings dir doesn't exist, stay in root and list everything
            print(f"📁 Directory /{remote_dir}/ not found, browsing root directory")
            ftp.cwd("/")
        
        files = []
        ftp.retrlines('LIST', files.append)
        
        video_files = []
        for file_info in files:
            parts = file_info.split()
            if len(parts) >= 9:
                filename = parts[-1]
                # Check for video extensions
                if filename.lower().endswith(('.mp4', '.h264', '.avi', '.mov', '.mkv')):
                    try:
                        size = int(parts[4])
                    except:
                        size = 0
                    
                    date_str = ' '.join(parts[5:8]) if len(parts) >= 8 else 'Unknown'
                    video_files.append({
                        'name': filename,
                        'size': size,
                        'date': date_str,
                        'info': file_info
                    })
        
        return video_files
    except Exception as e:
        print(f"❌ Failed to list files: {e}")
        return []

def download_file(ftp, remote_file, local_file):
    """Download a single file"""
    try:
        with open(local_file, 'wb') as f:
            ftp.retrbinary(f'RETR {remote_file}', f.write)
        return True
    except Exception as e:
        print(f"Failed to download {remote_file}: {e}")
        return False

def perform_yolo_detection(video_path, model_path, output_dir, enabled_classes=None, confidence=0.5):
    """Perform YOLO detection on a video file"""
    if not YOLO_AVAILABLE:
        print(f"Skipping detection for {video_path} - YOLO not available")
        return None
    
    try:
        # Load YOLO model
        model = YOLO(model_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video with detections
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_dir, f"{video_name}_detected.mp4")
        output_json_path = os.path.join(output_dir, f"{video_name}_detections.json")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Detection results storage
        all_detections = []
        frame_count = 0
        detection_summary = {}
        
        # Colors for different classes
        bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
                      (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
        
        custom_colors = {
            'crane-': (255, 165, 0),    # Orange
            'jcb': (255, 20, 147),      # Deep pink
            'numplate': (0, 255, 0)     # Green
        }
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = model(frame, verbose=False)
            detections = results[0].boxes
            
            frame_detections = []
            
            if detections is not None:
                for i in range(len(detections)):
                    classidx = int(detections[i].cls.item())
                    
                    # Filter by enabled classes if specified
                    if enabled_classes and classidx not in enabled_classes:
                        continue
                    
                    classname = model.names[classidx]
                    conf = detections[i].conf.item()
                    
                    if conf > confidence:
                        # Get bounding box coordinates
                        xyxy_tensor = detections[i].xyxy.cpu()
                        xyxy = xyxy_tensor.numpy().squeeze()
                        xmin, ymin, xmax, ymax = xyxy.astype(int)
                        
                        # Store detection data
                        detection_data = {
                            'frame': frame_count,
                            'class_id': classidx,
                            'class_name': classname,
                            'confidence': float(conf),
                            'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)]
                        }
                        frame_detections.append(detection_data)
                        
                        # Update summary
                        if classname not in detection_summary:
                            detection_summary[classname] = 0
                        detection_summary[classname] += 1
                        
                        # Draw bounding box
                        if classname.lower() in custom_colors:
                            color = custom_colors[classname.lower()]
                        else:
                            color = bbox_colors[classidx % len(bbox_colors)]
                        
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        
                        # Draw label
                        label = f'{classname}: {int(conf*100)}%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_ymin = max(ymin, labelSize[1] + 10)
                        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), 
                                    (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                        cv2.putText(frame, label, (xmin, label_ymin-7), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Add frame info
            cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f'Detections: {len(frame_detections)}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Write frame to output video
            out.write(frame)
            
            # Store frame detections
            if frame_detections:
                all_detections.extend(frame_detections)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Clean up
        cap.release()
        out.release()
        
        # Save detection results to JSON
        detection_results = {
            'video_file': os.path.basename(video_path),
            'total_frames': total_frames,
            'fps': fps,
            'resolution': [width, height],
            'model_used': model_path,
            'confidence_threshold': confidence,
            'detection_summary': detection_summary,
            'total_detections': len(all_detections),
            'detections': all_detections
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(detection_results, f, indent=2)
        
        print(f"✓ Detection complete: {len(all_detections)} detections found")
        print(f"  Output video: {output_video_path}")
        print(f"  Detection data: {output_json_path}")
        print(f"  Summary: {detection_summary}")
        
        return {
            'video_path': output_video_path,
            'json_path': output_json_path,
            'summary': detection_summary,
            'total_detections': len(all_detections)
        }
        
    except Exception as e:
        print(f"Error during detection on {video_path}: {e}")
        return None

def detection_worker(detection_queue, model_path, output_dir, enabled_classes, confidence):
    """Worker thread for processing detections"""
    while True:
        video_path = detection_queue.get()
        if video_path is None:  # Poison pill
            break
        
        print(f"\n🔍 Starting detection on: {os.path.basename(video_path)}")
        result = perform_yolo_detection(video_path, model_path, output_dir, enabled_classes, confidence)
        detection_queue.task_done()

def parse_classes_argument(classes_str, total_classes):
    """Parse the --classes argument to get enabled class indices."""
    if not classes_str:
        return None  # All classes enabled by default
    
    enabled = []
    parts = classes_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            start, end = map(int, part.split('-'))
            enabled.extend(range(start, end + 1))
        else:
            enabled.append(int(part))
    
    return list(set(enabled))

def filter_files_by_date(files, hours_ago=24):
    """Filter files by modification time"""
    cutoff_time = datetime.now() - timedelta(hours=hours_ago)
    filtered_files = []
    
    for file_info in files:
        if 'segment_' in file_info['name']:
            match = re.search(r'(\d{8}_\d{6})', file_info['name'])
            if match:
                timestamp_str = match.group(1)
                try:
                    file_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    if file_time >= cutoff_time:
                        filtered_files.append(file_info)
                except ValueError:
                    filtered_files.append(file_info)
            else:
                filtered_files.append(file_info)
        else:
            filtered_files.append(file_info)
    
    return filtered_files

def main():
    parser = argparse.ArgumentParser(description='Download videos and perform YOLO detection')
    
    # FTP arguments
    parser.add_argument('--host', required=True, help='FTP server IP address')
    parser.add_argument('--username', default='anonymous', help='FTP username (default: anonymous for no password)')
    parser.add_argument('--password', help='FTP password (optional for anonymous login)')
    parser.add_argument('--port', type=int, default=21, help='FTP port')
    parser.add_argument('--remote-dir', default='recordings', help='Remote directory')
    
    # Download arguments
    parser.add_argument('--local-dir', default='downloaded_videos', help='Local download directory')
    parser.add_argument('--hours', type=int, default=24, help='Download files from last N hours')
    parser.add_argument('--pattern', help='Download only files matching pattern')
    parser.add_argument('--latest', type=int, help='Download only the latest N files')
    parser.add_argument('--skip-existing', action='store_true', help='Skip files that already exist locally')
    
    # YOLO detection arguments
    parser.add_argument('--model', default='best.pt', help='YOLO model path')
    parser.add_argument('--output-dir', default='detection_results', help='Detection output directory')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--classes', help='Comma-separated class indices to detect (e.g., "17,19,27,42,51,55,80")')
    parser.add_argument('--no-detection', action='store_true', help='Download only, skip detection')
    parser.add_argument('--detection-only', help='Skip download, run detection on existing files in this directory')
    
    # Control arguments
    parser.add_argument('--list-only', action='store_true', help='Only list files, do not download')
    parser.add_argument('--parallel-detection', action='store_true', help='Run detection in parallel with downloads')
    
    args = parser.parse_args()
    
    # Validate YOLO model if detection is enabled
    if not args.no_detection and not args.detection_only:
        if not YOLO_AVAILABLE:
            print("Error: ultralytics not available. Install with: pip install ultralytics")
            sys.exit(1)
        
        if not os.path.exists(args.model):
            print(f"Error: YOLO model not found: {args.model}")
            sys.exit(1)
    
    # Parse enabled classes
    enabled_classes = None
    if args.classes and not args.no_detection:
        try:
            # Load model to get total classes
            if YOLO_AVAILABLE and os.path.exists(args.model):
                temp_model = YOLO(args.model)
                total_classes = len(temp_model.names)
                enabled_classes = parse_classes_argument(args.classes, total_classes)
                print(f"Enabled classes: {enabled_classes}")
                del temp_model
        except Exception as e:
            print(f"Warning: Could not parse classes: {e}")
    
    # Detection-only mode
    if args.detection_only:
        if not os.path.exists(args.detection_only):
            print(f"Error: Directory not found: {args.detection_only}")
            sys.exit(1)
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        video_files = []
        for filename in os.listdir(args.detection_only):
            if filename.endswith(('.mp4', '.h264', '.avi', '.mov')):
                video_files.append(os.path.join(args.detection_only, filename))
        
        print(f"Found {len(video_files)} video files for detection")
        
        for video_path in video_files:
            print(f"\n🔍 Processing: {os.path.basename(video_path)}")
            perform_yolo_detection(video_path, args.model, args.output_dir, enabled_classes, args.confidence)
        
        return
    
    # Connect to FTP server
    print(f"🚀 Download and Detect - Connecting to {args.host}:{args.port}")
    ftp = connect_ftp(args.host, args.username, args.password, args.port)
    
    if not ftp:
        sys.exit(1)
    
    # List and filter video files
    video_files = list_video_files(ftp, args.remote_dir)
    
    if not video_files:
        print("No video files found")
        ftp.quit()
        sys.exit(0)
    
    # Apply filters
    if args.hours:
        video_files = filter_files_by_date(video_files, args.hours)
    
    if args.pattern:
        pattern = re.compile(args.pattern)
        video_files = [f for f in video_files if pattern.search(f['name'])]
    
    if args.latest:
        video_files = sorted(video_files, key=lambda x: x['name'], reverse=True)[:args.latest]
    
    video_files = sorted(video_files, key=lambda x: x['name'])
    
    # Display file list
    total_size = sum(f['size'] for f in video_files)
    print(f"\nFound {len(video_files)} video files (Total: {total_size/1024/1024:.1f} MB)")
    
    if args.list_only:
        for i, file_info in enumerate(video_files, 1):
            size_mb = file_info['size'] / 1024 / 1024
            print(f"{i:3d}. {file_info['name']} ({size_mb:.1f} MB)")
        ftp.quit()
        return
    
    # Create directories
    os.makedirs(args.local_dir, exist_ok=True)
    if not args.no_detection:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup detection worker if parallel detection is enabled
    detection_queue = Queue()
    detection_thread = None
    
    if args.parallel_detection and not args.no_detection:
        detection_thread = threading.Thread(
            target=detection_worker,
            args=(detection_queue, args.model, args.output_dir, enabled_classes, args.confidence)
        )
        detection_thread.start()
    
    # Download and process files
    print(f"\nDownloading to {args.local_dir}/")
    if not args.no_detection:
        print(f"Detection results will be saved to {args.output_dir}/")
    
    downloaded_files = []
    success_count = 0
    
    for i, file_info in enumerate(video_files, 1):
        filename = file_info['name']
        local_path = os.path.join(args.local_dir, filename)
        
        # Skip if file already exists
        if args.skip_existing and os.path.exists(local_path):
            print(f"{i:3d}/{len(video_files)}: {filename} - Already exists, skipping")
            if not args.no_detection:
                downloaded_files.append(local_path)
            success_count += 1
            continue
        
        print(f"{i:3d}/{len(video_files)}: {filename} - Downloading...", end=' ')
        
        if download_file(ftp, filename, local_path):
            print("✓")
            success_count += 1
            downloaded_files.append(local_path)
            
            # Add to detection queue if parallel processing
            if args.parallel_detection and not args.no_detection:
                detection_queue.put(local_path)
        else:
            print("✗")
    
    ftp.quit()
    
    print(f"\nDownload complete: {success_count}/{len(video_files)} files")
    
    # Process detections
    if not args.no_detection and downloaded_files:
        if args.parallel_detection:
            # Wait for parallel detections to complete
            detection_queue.join()
            detection_queue.put(None)  # Poison pill
            detection_thread.join()
        else:
            # Sequential detection
            print(f"\nStarting YOLO detection on {len(downloaded_files)} files...")
            for i, video_path in enumerate(downloaded_files, 1):
                print(f"\n🔍 [{i}/{len(downloaded_files)}] Processing: {os.path.basename(video_path)}")
                perform_yolo_detection(video_path, args.model, args.output_dir, enabled_classes, args.confidence)
    
    print(f"\n✅ All tasks completed!")
    print(f"📁 Downloads: {args.local_dir}")
    if not args.no_detection:
        print(f"🔍 Detections: {args.output_dir}")

if __name__ == "__main__":
    main()
