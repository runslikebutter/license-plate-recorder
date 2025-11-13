# License Plate Recorder - Docker Deployment for Jetson Nano Orin

This guide covers deploying the License Plate Detection and Recording System on NVIDIA Jetson Nano Orin using Docker.

## 🎯 Features

- **Real-time License Plate Detection** using Fast-ALPR (YOLO v9 + OCR)
- **Advanced Tracking** with ByteTracker and OCR confidence aggregation
- **Event-triggered Recording** with configurable pre/post buffers
- **Interactive Detection Zones** with persistent configuration
- **Multi-input Support**: RTSP streams, video files, directory batch processing
- **GPU Acceleration** optimized for Jetson hardware
- **Containerized Deployment** for easy installation and updates

## 🔧 Prerequisites

### Hardware Requirements
- **NVIDIA Jetson Nano Orin** (minimum 4GB model recommended)
- **microSD card** (64GB+ recommended for recordings)
- **USB camera** or **IP camera with RTSP support**

### Software Requirements
- **JetPack 5.x** (comes with Ubuntu 20.04)
- **Docker** and **Docker Compose**
- **NVIDIA Container Runtime**

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd license-plate-recorder

# Run the automated setup script
chmod +x scripts/setup-jetson.sh
./scripts/setup-jetson.sh
```

### 2. Prepare Input Data

```bash
# Place your video files for testing
cp /path/to/your/video.mp4 input_videos/

# Or create a sample RTSP configuration
echo "rtsp://your-camera-ip:554/stream" > input_videos/rtsp_stream.txt
```

### 3. Run the Container

```bash
# Process video file with preview (requires display)
docker-compose run --rm license-plate-recorder \
  python3 main.py --input /app/input_videos/video.mp4 --preview

# Process RTSP stream (headless)
docker-compose run --rm license-plate-recorder \
  python3 main.py --input rtsp://camera-ip:554/stream

# Batch process directory
docker-compose run --rm license-plate-recorder \
  python3 main.py --input /app/input_videos --batch
```

## 📋 Configuration

### Environment Configuration

Edit `config.yaml` to customize detection settings:

```yaml
detection:
  model_detector: "yolo-v9-t-640-license-plate-end2end"
  model_ocr: "cct-s-v1-global-model"
  confidence_threshold: 0.3
  crop_position: "center"  # Auto-adjusted based on detection zone

recording:
  pre_detection_seconds: 3
  post_detection_seconds: 5
  max_recording_seconds: 30
  output_dir: "recordings"

tracking:
  confidence_threshold: 0.3
  match_threshold: 0.8
  track_buffer: 30
```

### Detection Zone Configuration

The system automatically creates and manages `detection_zone.json`:

```json
{
  "top": 35,
  "bottom": 5,
  "left": 5,
  "right": 5,
  "enabled": true
}
```

## 🎮 Usage Examples

### Interactive Mode with Zone Editing

```bash
# Run with preview and zone editing capability
docker-compose run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  license-plate-recorder \
  python3 main.py --input /app/input_videos/sample.mp4 --preview
```

**Controls:**
- `q` - Quit application
- `e` - Toggle zone editing mode
- `z` - Toggle detection zone on/off
- `h` - Toggle help display

### Production RTSP Stream Processing

```bash
# Continuous RTSP processing (headless)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop processing
docker-compose down
```

### Batch Processing

```bash
# Process all videos in input directory
docker-compose run --rm license-plate-recorder \
  python3 main.py --input /app/input_videos --batch
```

## 🔍 Monitoring

### System Resources

```bash
# Monitor Jetson system resources
sudo jtop

# Monitor container resources
docker stats

# View application logs
docker-compose logs -f license-plate-recorder
```

### Performance Optimization

For optimal performance on Jetson Nano Orin:

```bash
# Set maximum performance mode
sudo jetson_clocks

# Monitor GPU utilization
nvidia-smi

# Check memory usage
free -h
```

## 📊 Output Structure

```
recordings/
├── 2024-01-15/
│   ├── 14-30-25_ABC123_recording.mp4
│   ├── 14-35-10_XYZ789_recording.mp4
│   └── detections.log
└── 2024-01-16/
    ├── 09-15-30_DEF456_recording.mp4
    └── detections.log

logs/
├── license_plate_recorder.log
└── detection_performance.log
```

## 🛠️ Troubleshooting

### Common Issues

**Container fails to start:**
```bash
# Check NVIDIA runtime
docker info | grep nvidia

# Test GPU access
docker run --rm --runtime=nvidia nvidia/cuda:11.0-base nvidia-smi
```

**Out of memory errors:**
```bash
# Reduce batch size or enable swap
sudo systemctl enable nvzramconfig
sudo systemctl start nvzramconfig

# Monitor memory usage
sudo jtop
```

**Poor detection performance:**
```bash
# Check GPU utilization
nvidia-smi

# Verify model downloads
docker-compose exec license-plate-recorder ls -la /home/platerecorder/.cache/
```

### Debug Mode

```bash
# Run in debug mode with shell access
docker-compose run --rm -it --entrypoint /bin/bash license-plate-recorder

# Inside container, test components
python3 -c "import cv2, torch, ultralytics; print('All imports successful')"
python3 main.py --help
```

## 🔄 Updates and Maintenance

### Update the Container

```bash
# Pull latest base image
docker-compose pull

# Rebuild with updates
docker-compose build --no-cache

# Clean old images
docker system prune -f
```

### Backup Configuration

```bash
# Backup important files
tar -czf backup-$(date +%Y%m%d).tar.gz \
  config.yaml detection_zone.json recordings/
```

## 📝 Advanced Configuration

### Custom Model Paths

Mount custom models:

```yaml
# docker-compose.override.yml
services:
  license-plate-recorder:
    volumes:
      - ./custom_models:/app/models:ro
```

### Network Configuration

For multiple camera streams:

```yaml
# docker-compose.yml
services:
  license-plate-recorder:
    ports:
      - "8080:8080"  # Web interface (future)
    environment:
      - RTSP_STREAM_1=rtsp://camera1/stream
      - RTSP_STREAM_2=rtsp://camera2/stream
```

## 🆘 Support

For issues and questions:
1. Check the logs: `docker-compose logs -f`
2. Verify system resources: `sudo jtop`
3. Test basic functionality: `python3 main.py --help`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.