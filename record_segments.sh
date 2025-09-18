#!/bin/bash

# Raspberry Pi Video Recording Script
# Records 1-minute segments continuously
# Usage: ./record_segments.sh [output_directory] [camera_index]

# Configuration
OUTPUT_DIR=${1:-"recordings"}
CAMERA_INDEX=${2:-0}
SEGMENT_DURATION=60  # 1 minute in seconds
RESOLUTION="1280x720"
FPS=60
VIDEO_FORMAT="mp4"
CODEC="libx264"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting continuous video recording..."
echo "Output directory: $OUTPUT_DIR"
echo "Camera index: $CAMERA_INDEX"
echo "Segment duration: ${SEGMENT_DURATION} seconds"
echo "Resolution: $RESOLUTION"
echo "Press Ctrl+C to stop recording"

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "Stopping recording..."
    if [ ! -z "$FFMPEG_PID" ]; then
        kill $FFMPEG_PID 2>/dev/null
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Counter for file naming
SEGMENT_COUNT=1

while true; do
    # Generate timestamp for filename
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    FILENAME="${OUTPUT_DIR}/segment_${SEGMENT_COUNT}_${TIMESTAMP}.${VIDEO_FORMAT}"
    
    echo "Recording segment $SEGMENT_COUNT: $FILENAME"
    
    # Record using ffmpeg (works well on Raspberry Pi)
    ffmpeg -f v4l2 -i /dev/video${CAMERA_INDEX} \
           -t $SEGMENT_DURATION \
           -r $FPS \
           -s $RESOLUTION \
           -c:v $CODEC \
           -preset ultrafast \
           -crf 23 \
           -y "$FILENAME" \
           -loglevel quiet &
    
    FFMPEG_PID=$!
    
    # Wait for the recording to finish
    wait $FFMPEG_PID
    
    # Check if recording was successful
    if [ $? -eq 0 ]; then
        FILE_SIZE=$(stat -c%s "$FILENAME" 2>/dev/null || echo "0")
        if [ "$FILE_SIZE" -gt 1000 ]; then
            echo "✓ Segment $SEGMENT_COUNT completed (Size: $(( FILE_SIZE / 1024 ))KB)"
        else
            echo "⚠ Warning: Segment $SEGMENT_COUNT seems too small"
        fi
    else
        echo "✗ Error recording segment $SEGMENT_COUNT"
    fi
    
    # Increment counter
    SEGMENT_COUNT=$((SEGMENT_COUNT + 1))
    
    # Optional: Remove old segments to save space (keep last 100 segments)
    TOTAL_FILES=$(ls -1 "$OUTPUT_DIR"/segment_*.${VIDEO_FORMAT} 2>/dev/null | wc -l)
    if [ "$TOTAL_FILES" -gt 100 ]; then
        OLDEST_FILE=$(ls -1t "$OUTPUT_DIR"/segment_*.${VIDEO_FORMAT} | tail -1)
        rm "$OLDEST_FILE"
        echo "Removed old file: $OLDEST_FILE"
    fi
    
    # Small delay before next segment
    sleep 1
done
