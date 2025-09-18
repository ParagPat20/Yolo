#!/usr/bin/env python3

"""
FTP Video Downloader Script
Downloads video files from Raspberry Pi FTP server
Usage: python3 download_videos.py [options]
"""

import ftplib
import os
import sys
import argparse
from datetime import datetime, timedelta
import re

def connect_ftp(host, username, password, port=21):
    """Connect to FTP server"""
    try:
        ftp = ftplib.FTP()
        ftp.connect(host, port)
        ftp.login(username, password)
        return ftp
    except Exception as e:
        print(f"Failed to connect to FTP server: {e}")
        return None

def list_video_files(ftp, remote_dir="recordings"):
    """List all video files on the server"""
    try:
        ftp.cwd(remote_dir)
        files = []
        ftp.retrlines('LIST', files.append)
        
        video_files = []
        for file_info in files:
            parts = file_info.split()
            if len(parts) >= 9:
                filename = parts[-1]
                if filename.endswith(('.mp4', '.h264', '.avi', '.mov')):
                    # Extract file size and date
                    size = int(parts[4])
                    date_str = ' '.join(parts[5:8])
                    video_files.append({
                        'name': filename,
                        'size': size,
                        'date': date_str,
                        'info': file_info
                    })
        
        return video_files
    except Exception as e:
        print(f"Failed to list files: {e}")
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

def filter_files_by_date(files, hours_ago=24):
    """Filter files by modification time"""
    cutoff_time = datetime.now() - timedelta(hours=hours_ago)
    filtered_files = []
    
    for file_info in files:
        # This is a simple filter - in practice, you might want to parse the timestamp from filename
        if 'segment_' in file_info['name']:
            # Extract timestamp from filename like segment_1_20240918_143052.mp4
            match = re.search(r'(\d{8}_\d{6})', file_info['name'])
            if match:
                timestamp_str = match.group(1)
                try:
                    file_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    if file_time >= cutoff_time:
                        filtered_files.append(file_info)
                except ValueError:
                    # If timestamp parsing fails, include the file
                    filtered_files.append(file_info)
            else:
                filtered_files.append(file_info)
        else:
            filtered_files.append(file_info)
    
    return filtered_files

def main():
    parser = argparse.ArgumentParser(description='Download videos from Raspberry Pi FTP server')
    parser.add_argument('--host', required=True, help='FTP server IP address')
    parser.add_argument('--username', default='videouser', help='FTP username (default: videouser)')
    parser.add_argument('--password', required=True, help='FTP password')
    parser.add_argument('--port', type=int, default=21, help='FTP port (default: 21)')
    parser.add_argument('--remote-dir', default='recordings', help='Remote directory (default: recordings)')
    parser.add_argument('--local-dir', default='downloaded_videos', help='Local download directory')
    parser.add_argument('--hours', type=int, default=24, help='Download files from last N hours (default: 24)')
    parser.add_argument('--pattern', help='Download only files matching pattern (regex)')
    parser.add_argument('--list-only', action='store_true', help='Only list files, do not download')
    parser.add_argument('--latest', type=int, help='Download only the latest N files')
    
    args = parser.parse_args()
    
    print(f"Connecting to FTP server {args.host}:{args.port}")
    ftp = connect_ftp(args.host, args.username, args.password, args.port)
    
    if not ftp:
        sys.exit(1)
    
    print(f"Connected successfully!")
    print(f"Listing files in /{args.remote_dir}/")
    
    # List video files
    video_files = list_video_files(ftp, args.remote_dir)
    
    if not video_files:
        print("No video files found")
        ftp.quit()
        sys.exit(0)
    
    # Apply filters
    if args.hours:
        video_files = filter_files_by_date(video_files, args.hours)
        print(f"Found {len(video_files)} files from last {args.hours} hours")
    
    if args.pattern:
        pattern = re.compile(args.pattern)
        video_files = [f for f in video_files if pattern.search(f['name'])]
        print(f"Found {len(video_files)} files matching pattern '{args.pattern}'")
    
    if args.latest:
        # Sort by filename (which includes timestamp) and take latest N
        video_files = sorted(video_files, key=lambda x: x['name'], reverse=True)[:args.latest]
        print(f"Selected latest {len(video_files)} files")
    
    # Sort files by name for consistent ordering
    video_files = sorted(video_files, key=lambda x: x['name'])
    
    # Display file list
    total_size = sum(f['size'] for f in video_files)
    print(f"\nFound {len(video_files)} video files (Total: {total_size/1024/1024:.1f} MB):")
    print("-" * 80)
    
    for i, file_info in enumerate(video_files, 1):
        size_mb = file_info['size'] / 1024 / 1024
        print(f"{i:3d}. {file_info['name']} ({size_mb:.1f} MB)")
    
    if args.list_only:
        print("\nList-only mode - not downloading files")
        ftp.quit()
        return
    
    # Create local directory
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Download files
    print(f"\nDownloading to {args.local_dir}/")
    print("-" * 80)
    
    success_count = 0
    for i, file_info in enumerate(video_files, 1):
        filename = file_info['name']
        local_path = os.path.join(args.local_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(local_path):
            print(f"{i:3d}/{len(video_files)}: {filename} - Already exists, skipping")
            success_count += 1
            continue
        
        print(f"{i:3d}/{len(video_files)}: {filename} - Downloading...", end=' ')
        
        if download_file(ftp, filename, local_path):
            print("✓")
            success_count += 1
        else:
            print("✗")
    
    ftp.quit()
    
    print(f"\nDownload complete: {success_count}/{len(video_files)} files")
    print(f"Files saved to: {args.local_dir}")

if __name__ == "__main__":
    main()
