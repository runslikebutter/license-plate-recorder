# License Plate Recorder - SSH Deployment Guide

This guide covers building the Docker image locally and deploying it to the Jetson Nano Orin via SSH transfer.

## 🎯 Overview

Since the Jetson at `monarch@192.168.1.87` has Docker but not Docker Compose, we'll:
1. **Build** the Docker image on your local system
2. **Transfer** the image to the Jetson via SSH
3. **Deploy** using simple Docker run commands (no Compose needed)

## 🚀 Quick Start

### 1. Build and Transfer (Run on your local machine)

```bash
# Build image and transfer to Jetson
cd license-plate-recorder
./scripts/build-for-jetson.sh

# This will:
# - Build the Docker image for ARM64
# - Save it to a tar file
# - Transfer it to the Jetson via SSH
# - Load it on the Jetson
# - Create run scripts
```

### 2. Setup on Jetson (Run on the Jetson)

```bash
# SSH to the Jetson
ssh monarch@192.168.1.87

# Run the setup script (it was transferred automatically)
/mnt/data/license-plate-recorder/run.sh
```

### 3. Use the System

```bash
# Process a video file (using convenient symlinks)
~/run-video.sh input_videos/sample.mp4

# Process RTSP stream
~/run-rtsp.sh rtsp://camera-ip:554/stream

# Batch process all videos
~/run-batch.sh

# Or access via the data partition directly
cd /mnt/data/license-plate-recorder
./run-video.sh input_videos/sample.mp4
```

## 📋 Detailed Steps

### Step 1: Local Build and Transfer

On your local machine (Mac/PC):

```bash
cd /path/to/license-plate-recorder

# Build and transfer everything
./scripts/build-for-jetson.sh
```

**What this script does:**
- ✅ Builds Docker image using Jetson base image
- ✅ Saves image to tar file
- ✅ Transfers via SCP to Jetson
- ✅ Loads image on Jetson
- ✅ Creates deployment scripts
- ✅ Cleans up temporary files

### Step 2: Jetson Setup

SSH to your Jetson:

```bash
ssh monarch@192.168.1.87
```

Run the setup:

```bash
# Setup project directory and scripts (deployment script is automatically run by build-for-jetson.sh)
# Or run manually if needed:
/mnt/data/license-plate-recorder/jetson-deploy.sh

# Check that everything is ready
docker images | grep license-plate-recorder
```

### Step 3: Add Your Content

```bash
# Add video files for testing (using symlink for convenience)
cd ~/license-plate-recorder
cp /path/to/your/videos/*.mp4 input_videos/

# Or directly to the data partition
cp /path/to/your/videos/*.mp4 /mnt/data/license-plate-recorder/input_videos/

# Or test with RTSP stream (no files needed)
```

## 🎮 Usage Examples

### Video File Processing

```bash
# Process specific video with preview (using symlinks from anywhere)
~/run-video.sh input_videos/test.mp4

# Or from project directory
cd /mnt/data/license-plate-recorder
./run-video.sh input_videos/test.mp4

# Process video without display (headless)
# Edit run-video.sh and remove --preview flag
```

### RTSP Stream Processing

```bash
# Live camera stream (using symlinks)
~/run-rtsp.sh rtsp://192.168.1.100:554/h264

# With authentication
~/run-rtsp.sh rtsp://username:password@camera-ip:554/stream
```

### Batch Processing

```bash
# Process all video files in input_videos/ (using symlinks)
~/run-batch.sh

# Results will be in recordings/ directory
ls ~/license-plate-recorder/recordings/
# Or: ls /mnt/data/license-plate-recorder/recordings/
```

## 🔧 Configuration

### Edit Settings

```bash
# Edit main configuration (using symlink)
cd ~/license-plate-recorder
nano config.yaml

# Or directly on data partition
nano /mnt/data/license-plate-recorder/config.yaml

# Edit detection zones
nano detection_zone.json

# Or use interactive zone editing:
~/run-video.sh input_videos/sample.mp4
# Press 'e' to enter zone editing mode
```

### Key Configuration Options

**config.yaml:**
```yaml
detection:
  confidence_threshold: 0.3      # Lower = more detections
  crop_position: "center"        # Auto-adjusted based on zone

recording:
  pre_detection_seconds: 3       # Record 3s before plate detected
  post_detection_seconds: 5      # Record 5s after plate detected

performance:
  fps_target: 15                 # Target FPS for processing
```

**detection_zone.json:**
```json
{
  "top": 35,                     # Skip 35% from top
  "bottom": 5,                   # Skip 5% from bottom
  "left": 5,                     # Skip 5% from left
  "right": 5,                    # Skip 5% from right
  "enabled": true
}
```

## 🔍 Monitoring and Troubleshooting

### System Resources

```bash
# Monitor Jetson system
sudo jtop

# Check Docker containers
docker ps
docker logs <container-id>

# Check disk space
df -h
```

### Performance Optimization

```bash
# Enable maximum performance
sudo jetson_clocks

# Monitor GPU usage
sudo tegrastats

# Check memory usage
free -h
```

### Common Issues

**Out of Memory:**
```bash
# Check memory usage
free -h

# Reduce batch size or add swap
sudo systemctl enable nvzramconfig
sudo systemctl start nvzramconfig
```

**Permission Issues:**
```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
chmod +x ~/license-plate-recorder/*.sh
```

**Network Issues (RTSP):**
```bash
# Test RTSP stream manually
ffplay rtsp://camera-ip:554/stream

# Check network connectivity
ping camera-ip
```

## 🔄 Updates and Maintenance

### Update the System

When you make changes to the code:

```bash
# On your local machine:
./scripts/build-for-jetson.sh

# The image will be automatically transferred and updated
```

### Backup Configuration

```bash
# On Jetson
cd ~/license-plate-recorder
tar -czf backup-$(date +%Y%m%d).tar.gz \
  config.yaml detection_zone.json recordings/

# Transfer backup to safe location
scp backup-*.tar.gz your-backup-server:/backups/
```

### Clean Up

```bash
# Remove old Docker images
docker system prune -f

# Clean old recordings (keep last 30 days)
find /mnt/data/license-plate-recorder/recordings/ -name "*.mp4" -mtime +30 -delete
```

## 📊 Output Structure

```
/mnt/data/license-plate-recorder/  # Main project directory (on data partition)
├── input_videos/                  # Your video files
├── recordings/                    # Processed outputs
│   ├── 2024-01-15/
│   │   ├── 14-30-25_ABC123_recording.mp4
│   │   └── 14-35-10_XYZ789_recording.mp4
│   └── detection_log.txt
├── logs/                          # Application logs
├── config.yaml                   # Configuration
├── detection_zone.json            # Zone settings
├── run-video.sh                  # Video processing script
├── run-rtsp.sh                   # RTSP processing script
└── run-batch.sh                  # Batch processing script

~/                                # Home directory (convenient symlinks)
├── license-plate-recorder -> /mnt/data/license-plate-recorder/
├── run-video.sh -> /mnt/data/license-plate-recorder/run-video.sh
├── run-rtsp.sh -> /mnt/data/license-plate-recorder/run-rtsp.sh
└── run-batch.sh -> /mnt/data/license-plate-recorder/run-batch.sh
```

## 🎛️ Interactive Controls

When running with preview (`--preview` flag):

- **`q`** - Quit application
- **`e`** - Toggle zone editing mode
- **`z`** - Toggle detection zone on/off
- **`h`** - Show help overlay
- **Mouse** - Edit zone boundaries (when in edit mode)

## 🆘 Getting Help

### Debug Mode

```bash
# Run with debug logging
docker run --rm -it \
    --runtime=nvidia \
    --volume="$(pwd)/input_videos:/app/input_videos" \
    --volume="$(pwd)/recordings:/app/recordings" \
    license-plate-recorder:jetson-arm64 \
    python3 main.py --input /app/input_videos/test.mp4 --preview --debug
```

### Test Basic Functionality

```bash
# Test Docker image
docker run --rm license-plate-recorder:jetson-arm64 python3 --version

# Test main application
docker run --rm license-plate-recorder:jetson-arm64 python3 main.py --help

# Test dependencies
docker run --rm license-plate-recorder:jetson-arm64 python3 -c "import cv2, torch, ultralytics; print('All OK')"
```

This approach eliminates the need for Docker Compose while maintaining all the functionality of the license plate detection system!