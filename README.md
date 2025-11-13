# License Plate Recorder

An automated license plate detection and recording system built for edge deployment. Detects plates in real-time from RTSP streams or video files, tracks them across frames, and saves event-triggered recordings with pre/post buffers.

## Features

- **Real-time Detection**: YOLO v9-based license plate detection via Fast-ALPR
- **OCR Integration**: Automatic plate text recognition with confidence scoring
- **Smart Tracking**: ByteTrack-powered tracking with OCR aggregation across frames
- **Event Recording**: Triggered recordings with configurable pre/post detection buffers
- **Detection Zones**: Interactive zone editor to focus on specific areas (reduce false positives)
- **Multi-input**: Supports RTSP streams, video files, or batch directory processing
- **Jetson Optimized**: Hardware acceleration support for NVIDIA Jetson devices

## Quick Start

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
# Process a video file with preview
python main.py --input video.mp4 --preview

# Process RTSP stream (headless)
python main.py --input rtsp://camera-ip:554/stream

# Batch process directory
python main.py --input /path/to/videos/ --batch

# Custom output directory
python main.py --input video.mp4 --preview --output-dir ./my_recordings
```

### Interactive Controls (Preview Mode)

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `e` | Toggle zone editing mode |
| `z` | Enable/disable detection zone |
| `h` | Toggle help overlay |
| `s` | Save screenshot |
| `d` | Toggle debug information |
| `Space` | Pause/Resume |

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

Detection zones are saved in `detection_zone.json` and can be edited interactively by pressing `e` in preview mode.

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

## How It Works

1. **Capture**: Reads frames from RTSP stream, video file, or directory
2. **Detect**: Runs Fast-ALPR (YOLO v9 + OCR) on center-cropped frames
3. **Filter**: Applies detection zone to ignore irrelevant areas
4. **Track**: Matches detections across frames, aggregates OCR readings
5. **Record**: Triggers recording when confidence threshold met, saves with pre/post buffers

All frames are continuously buffered in memory, so when a plate is detected, the system can "rewind" and include footage from before the detection event.

## Deployment

### Development/Testing
Run locally with Python virtual environment (see Quick Start above).

### Production (Jetson)
See `docs/JETSON-DEPLOYMENT.md` for detailed instructions on:
- Docker deployment
- Hardware acceleration setup
- Systemd service configuration
- SSH deployment scripts

Use `deploy-jetson.sh` for automated Jetson deployment or build Docker containers with hardware decoder support.

## Output

Recordings are saved to `recordings/` with filenames in the format:
```
YYYYMMDD_HHMMSS_PLATETEXT.mp4
```

Example: `20240115_143025_ABC123.mp4`

## Requirements

- Python 3.8+
- OpenCV with GStreamer support (for RTSP)
- PyTorch 2.0+ (CPU or CUDA)
- Fast-ALPR with ONNX runtime

For Jetson-specific requirements, see `requirements-jetson.txt` and `required_libs/README.md`.

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

## Notes

This is a working prototype that's been tested with various RTSP cameras and video formats. The detection zone feature is particularly useful for fixed-camera installations where you want to ignore sky, adjacent lanes, or background traffic.

The OCR aggregation in the tracker means plate text gets more accurate over time as the vehicle moves through the frame - usually reliable after 3-5 detections.
