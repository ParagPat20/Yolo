#!/usr/bin/env python3

"""
Simple FTP Video Downloader - No Authentication Required
Downloads video files from Raspberry Pi simple FTP server
Usage: python3 download_simple.py --host PI_IP_ADDRESS
"""

import ftplib
import os
import sys
import argparse
from datetime import datetime, timedelta
import re

def connect_simple_ftp(host, port=21):
    """Connect to simple FTP server (anonymous login)"""
    try:
        ftp = ftplib.FTP()
        ftp.connect(host, port)
        ftp.login()  # Anonymous login - no credentials needed
        print(f"✅ Connected to {host} (anonymous)")
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
        print(f"❌ Failed to download {remote_file}: {e}")
        return False

def filter_files_by_pattern(files, pattern):
    """Filter files by name pattern"""
    if not pattern:
        return files
    
    try:
        regex = re.compile(pattern, re.IGNORECASE)
        return [f for f in files if regex.search(f['name'])]
    except re.error:
        print(f"❌ Invalid regex pattern: {pattern}")
        return files

def main():
    parser = argparse.ArgumentParser(description='Download videos from simple FTP server')
    parser.add_argument('--host', required=True, help='FTP server IP address (e.g., 192.168.1.100)')
    parser.add_argument('--port', type=int, default=21, help='FTP port (default: 21)')
    parser.add_argument('--remote-dir', default='recordings', help='Remote directory to browse (default: recordings)')
    parser.add_argument('--local-dir', default='downloaded_videos', help='Local download directory')
    parser.add_argument('--pattern', help='Download only files matching pattern (e.g., "segment_.*mp4")')
    parser.add_argument('--latest', type=int, help='Download only the latest N files')
    parser.add_argument('--list-only', action='store_true', help='Only list files, do not download')
    parser.add_argument('--skip-existing', action='store_true', help='Skip files that already exist locally')
    
    args = parser.parse_args()
    
    print("🚀 Simple FTP Video Downloader")
    print("=" * 50)
    print(f"📡 Connecting to: {args.host}:{args.port}")
    
    # Connect to FTP server
    ftp = connect_simple_ftp(args.host, args.port)
    if not ftp:
        sys.exit(1)
    
    # List video files
    print(f"🔍 Searching for video files...")
    video_files = list_video_files(ftp, args.remote_dir)
    
    if not video_files:
        print("❌ No video files found")
        # Try listing root directory contents
        print("\n📋 Available files/directories:")
        try:
            ftp.cwd("/")
            files = []
            ftp.retrlines('LIST', files.append)
            for file_info in files[:10]:  # Show first 10 items
                print(f"   {file_info}")
            if len(files) > 10:
                print(f"   ... and {len(files) - 10} more items")
        except:
            pass
        ftp.quit()
        sys.exit(0)
    
    print(f"✅ Found {len(video_files)} video files")
    
    # Apply filters
    if args.pattern:
        video_files = filter_files_by_pattern(video_files, args.pattern)
        print(f"🔍 After pattern filter: {len(video_files)} files")
    
    if args.latest:
        # Sort by filename (which usually includes timestamp) and take latest N
        video_files = sorted(video_files, key=lambda x: x['name'], reverse=True)[:args.latest]
        print(f"📊 Selected latest {len(video_files)} files")
    
    # Sort files by name for consistent ordering
    video_files = sorted(video_files, key=lambda x: x['name'])
    
    # Display file list
    total_size = sum(f['size'] for f in video_files)
    print(f"\n📋 Video Files (Total: {total_size/1024/1024:.1f} MB):")
    print("-" * 80)
    
    for i, file_info in enumerate(video_files, 1):
        size_mb = file_info['size'] / 1024 / 1024
        print(f"{i:3d}. {file_info['name']:<40} ({size_mb:>6.1f} MB)")
    
    if args.list_only:
        print(f"\n📋 List-only mode - found {len(video_files)} video files")
        ftp.quit()
        return
    
    # Create local directory
    os.makedirs(args.local_dir, exist_ok=True)
    print(f"\n📁 Download directory: {args.local_dir}")
    
    # Download files
    print(f"\n⬇️  Starting downloads...")
    print("-" * 80)
    
    success_count = 0
    skipped_count = 0
    
    for i, file_info in enumerate(video_files, 1):
        filename = file_info['name']
        local_path = os.path.join(args.local_dir, filename)
        
        # Skip if file already exists
        if args.skip_existing and os.path.exists(local_path):
            print(f"{i:3d}/{len(video_files)}: {filename:<40} - ⏭️  Already exists")
            skipped_count += 1
            continue
        
        print(f"{i:3d}/{len(video_files)}: {filename:<40} - ⬇️  ", end='')
        
        if download_file(ftp, filename, local_path):
            # Verify download
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                downloaded_size = os.path.getsize(local_path) / 1024 / 1024
                print(f"✅ ({downloaded_size:.1f} MB)")
                success_count += 1
            else:
                print("❌ File empty or corrupted")
        else:
            print("❌ Download failed")
    
    ftp.quit()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Download Summary:")
    print(f"✅ Successfully downloaded: {success_count} files")
    if skipped_count > 0:
        print(f"⏭️  Skipped (already exist): {skipped_count} files")
    print(f"❌ Failed: {len(video_files) - success_count - skipped_count} files")
    print(f"📁 Files saved to: {os.path.abspath(args.local_dir)}")
    
    if success_count > 0:
        print(f"\n🎉 Download complete! {success_count} videos ready for processing.")

if __name__ == "__main__":
    main()
