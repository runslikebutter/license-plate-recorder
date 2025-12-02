# License Plate Recorder

An automated license plate detection and recording system built for edge deployment. Detects plates in real-time from RTSP streams or video files, tracks them across frames, and saves event-triggered recordings with pre/post buffers.

## Features

- **Real-time Detection**: YOLO v9-based license plate detection via Fast-ALPR
- **OCR Integration**: Automatic plate text recognition with confidence scoring
- **Smart Tracking**: ByteTrack-powered tracking with OCR aggregation across frames
- **Event Recording**: Triggered recordings with configurable pre/post detection buffers
- **Multi-input**: Supports RTSP streams, video files, or batch directory processing
- **Jetson Optimized**: Hardware acceleration support for NVIDIA Jetson devices

### Deploying docker

you will need s3 bucket access data at runtime to upload recordings.

```bash
docker pull ghcr.io/runslikebutter/license-plate-recorder:latest
```

```bash
docker run -d \
--log-driver=journald \
--net=host \
--ipc=host \
--runtime=nvidia \
--name lp-recorder \
--user 1200:1200 \
-e DEVICE_HOSTNAME=$(hostname) \
-e S3_ACCESS_KEY_ID="your_key_id_here" \
-e S3_SECRET_ACCESS_KEY="your_secret_key_here" \
-v /mnt/data/recorder:/mnt/data/recorder \
ghcr.io/runslikebutter/license-plate-recorder:release \
--input rtsp://172.16.1.114:8554/live/camera1
```

### Installation

```bash
# Clone and setup virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash

# Process RTSP stream (headless)
python main.py --input rtsp://camera-ip:554/stream

# Batch process directory
python main.py --input /path/to/videos/ --batch

# Custom output directory
python main.py --input video.mp4 --preview --output-dir ./my_recordings
```

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
detection:
  confidence_threshold: 0.3    # Min confidence for detections
  crop_position: "center"      # Frame crop: left/center/right

recording:
  pre_plate_buffer: 3          # Seconds before detection
  post_plate_buffer: 5         # Seconds after detection
  max_duration: 30             # Max recording length

tracking:
  min_detections_for_recording: 3  # Detections needed before saving
```

## Project Structure

```
license-plate-recorder/
├── main.py                 # Main application entry point
├── config.yaml             # Configuration file
├── src/
│   ├── capture/            # Video input handlers (RTSP, file, batch)
│   ├── detection/          # License plate detection (Fast-ALPR)
│   ├── tracking/           # Plate tracking (ByteTrack + OCR aggregation)
│   ├── recording/          # Event-triggered recording with buffers
│   └── utils/              # Visualization, zones, logging
├── recordings/             # Output directory (organized by date)
├── input_videos/           # Place test videos here
├── scripts/                # Deployment scripts
└── docs/                   # Detailed documentation
```

## Troubleshooting

**No detections appearing:**
- Check confidence threshold in `config.yaml`
- Verify detection zone isn't excluding your target area (press `z` to disable)
- Ensure lighting conditions are good

**RTSP connection fails:**
- Verify stream URL is accessible: `ffplay rtsp://your-stream-url`
- Check GStreamer installation: `gst-inspect-1.0 rtspsrc`
- Review `license_plate_recorder.log` for errors

**Out of memory:**
- Reduce `pre_plate_buffer` duration in config
- Lower video resolution at source
- Enable swap on Jetson devices
