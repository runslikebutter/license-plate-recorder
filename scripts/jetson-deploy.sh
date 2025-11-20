#!/bin/bash

# Deployment script for Jetson Nano Orin (runs on the Jetson itself)
# This script sets up the environment and provides easy run commands

set -e

IMAGE_NAME="license-plate-recorder:jetson-arm64"
PROJECT_DIR="/mnt/data/license-plate-recorder"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}ℹ️  INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅ SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠️  WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}❌ ERROR:${NC} $1"
}

# Check if running on Jetson
check_jetson() {
    log_info "Checking Jetson environment..."

    if [ -f /etc/nv_tegra_release ]; then
        log_success "Running on NVIDIA Jetson device"
        cat /etc/nv_tegra_release
    else
        log_warning "Not running on a Jetson device"
    fi

    # Check available memory
    local total_mem=$(grep MemTotal /proc/meminfo | awk '{printf "%.1f", $2/1024/1024}')
    log_info "Available memory: ${total_mem}GB"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed on this Jetson"
        echo "Please install Docker first:"
        echo "  curl -fsSL https://get.docker.com -o get-docker.sh"
        echo "  sh get-docker.sh"
        echo "  sudo usermod -aG docker \$USER"
        echo "  newgrp docker"
        exit 1
    fi

    # Check NVIDIA runtime
    if docker info 2>/dev/null | grep -q nvidia; then
        log_success "NVIDIA Container Runtime is available"
    else
        log_warning "NVIDIA Container Runtime not detected"
        echo "To enable GPU acceleration, install nvidia-docker2:"
        echo "  sudo apt update"
        echo "  sudo apt install nvidia-docker2"
        echo "  sudo systemctl restart docker"
    fi
}

# Setup project directory
setup_project() {
    log_info "Setting up project directory..."

    # Create project directory structure
    mkdir -p "$PROJECT_DIR"/{input_videos,recordings,logs,src,model_cache}
    cd "$PROJECT_DIR"

    # Create default config if it doesn't exist
    if [ ! -f config.yaml ]; then
        log_info "Creating default config.yaml..."
        cat > config.yaml << 'EOF'
# License Plate Recorder Configuration
detection:
  model_detector: "yolo-v9-t-640-license-plate-end2end"
  model_ocr: "cct-s-v1-global-model"
  confidence_threshold: 0.3
  crop_position: "full"
  detection_input_size: 640
  ocr_crop_size: [128, 64]
  enable_ocr: true

recording:
  pre_detection_seconds: 3
  post_detection_seconds: 5
  max_recording_seconds: 30
  output_dir: "recordings"
  video_codec: "mp4v"

tracking:
  confidence_threshold: 0.3
  match_threshold: 0.8
  track_buffer: 30
  ocr_confidence_threshold: 0.7
  min_detections_for_ocr: 2

logging:
  level: "INFO"
  file: "logs/license_plate_recorder.log"
  max_file_size: 10485760
  backup_count: 5

performance:
  fps_target: 15
  memory_cleanup_interval: 300
  stats_log_interval: 60
EOF
        log_success "Created config.yaml"
    fi

    # Create default detection zone if it doesn't exist
    if [ ! -f detection_zone.json ]; then
        log_info "Creating default detection_zone.json..."
        cat > detection_zone.json << 'EOF'
{
  "top": 35,
  "bottom": 5,
  "left": 5,
  "right": 5,
  "enabled": true
}
EOF
        log_success "Created detection_zone.json"
    fi

    log_success "Project directory setup complete: $PROJECT_DIR"
}

# Create run scripts
create_run_scripts() {
    log_info "Creating run scripts..."

    # Main run script
    cat > "$PROJECT_DIR/run-video.sh" << EOF
#!/bin/bash

# Run License Plate Recorder with video file
# Usage: ./run-video.sh path/to/video.mp4

set -e

IMAGE_NAME="$IMAGE_NAME"
VIDEO_FILE="\${1:-input_videos/sample.mp4}"

if [ ! -f "\$VIDEO_FILE" ] && [[ "\$VIDEO_FILE" != /* ]]; then
    VIDEO_FILE="\$(pwd)/\$VIDEO_FILE"
fi

echo "🚀 Running License Plate Recorder with video..."
echo "Video: \$VIDEO_FILE"
echo "Working directory: \$(pwd)"
echo ""

# Check if video file exists
if [[ "\$VIDEO_FILE" == input_videos/* ]] && [ ! -f "\$VIDEO_FILE" ]; then
    echo "❌ Video file not found: \$VIDEO_FILE"
    echo "Please place video files in the input_videos/ directory"
    exit 1
fi

docker run --rm -it \\
    --runtime=nvidia \\
    --volume="\$(pwd)/src:/app/src" \\
    --volume="\$(pwd)/main.py:/app/main.py" \\
    --volume="\$(pwd)/input_videos:/app/input_videos" \\
    --volume="\$(pwd)/recordings:/app/recordings" \\
    --volume="\$(pwd)/logs:/app/logs" \\
    --volume="\$(pwd)/config.yaml:/app/config.yaml" \\
    --volume="\$(pwd)/detection_zone.json:/app/detection_zone.json" \\
    --volume="\$(pwd)/model_cache:/home/platerecorder/.cache" \\
    "\$IMAGE_NAME" \\
    python3 main.py --input "/app/\$VIDEO_FILE"

echo ""
echo "✅ Processing complete"
echo "📁 Check recordings/ directory for output videos"
EOF

    # RTSP stream script
    cat > "$PROJECT_DIR/run-rtsp.sh" << EOF
#!/bin/bash

# Run License Plate Recorder with RTSP stream
# Usage: ./run-rtsp.sh rtsp://camera-ip:554/stream

set -e

IMAGE_NAME="$IMAGE_NAME"
RTSP_URL="\${1}"

if [ -z "\$RTSP_URL" ]; then
    echo "❌ Please provide RTSP URL"
    echo "Usage: ./run-rtsp.sh rtsp://camera-ip:554/stream"
    exit 1
fi

echo "🚀 Running License Plate Recorder with RTSP stream..."
echo "RTSP URL: \$RTSP_URL"
echo "Working directory: \$(pwd)"
echo "Press Ctrl+C to stop"
echo ""

docker run --rm -it \\
    --runtime=nvidia \\
    --network=host \\
    --volume="\$(pwd)/src:/app/src" \\
    --volume="\$(pwd)/main.py:/app/main.py" \\
    --volume="\$(pwd)/recordings:/app/recordings" \\
    --volume="\$(pwd)/logs:/app/logs" \\
    --volume="\$(pwd)/config.yaml:/app/config.yaml" \\
    --volume="\$(pwd)/detection_zone.json:/app/detection_zone.json" \\
    --volume="\$(pwd)/model_cache:/home/platerecorder/.cache" \\
    "\$IMAGE_NAME" \\
    python3 main.py --input "\$RTSP_URL" --preview

echo ""
echo "✅ Processing stopped"
EOF

    # Batch processing script
    cat > "$PROJECT_DIR/run-batch.sh" << EOF
#!/bin/bash

# Run License Plate Recorder in batch mode (all videos in input_videos/)
# Usage: ./run-batch.sh

set -e

IMAGE_NAME="$IMAGE_NAME"

echo "🚀 Running License Plate Recorder in batch mode..."
echo "Processing all videos in input_videos/"
echo "Working directory: \$(pwd)"
echo ""

# Check if input_videos directory has any video files
if ! ls input_videos/*.{mp4,avi,mov,mkv} 1> /dev/null 2>&1; then
    echo "❌ No video files found in input_videos/ directory"
    echo "Please add video files (.mp4, .avi, .mov, .mkv) to process"
    exit 1
fi

docker run --rm -it \\
    --runtime=nvidia \\
    --volume="\$(pwd)/src:/app/src" \\
    --volume="\$(pwd)/main.py:/app/main.py" \\
    --volume="\$(pwd)/input_videos:/app/input_videos" \\
    --volume="\$(pwd)/recordings:/app/recordings" \\
    --volume="\$(pwd)/logs:/app/logs" \\
    --volume="\$(pwd)/config.yaml:/app/config.yaml" \\
    --volume="\$(pwd)/detection_zone.json:/app/detection_zone.json" \\
    --volume="\$(pwd)/model_cache:/home/platerecorder/.cache" \\
    "\$IMAGE_NAME" \\
    python3 main.py --input /app/input_videos --batch

echo ""
echo "✅ Batch processing complete"
echo "📁 Check recordings/ directory for output videos"
EOF

    # Make scripts executable
    chmod +x "$PROJECT_DIR"/{run-video.sh,run-rtsp.sh,run-batch.sh}

    # Remove any existing symlinks and create fresh ones
    rm -f "$HOME/run-video.sh" "$HOME/run-rtsp.sh" "$HOME/run-batch.sh" "$HOME/license-plate-recorder"

    # Create convenient symlinks in home directory for easy access
    ln -sf "$PROJECT_DIR/run-video.sh" "$HOME/run-video.sh"
    ln -sf "$PROJECT_DIR/run-rtsp.sh" "$HOME/run-rtsp.sh"
    ln -sf "$PROJECT_DIR/run-batch.sh" "$HOME/run-batch.sh"
    ln -sf "$PROJECT_DIR" "$HOME/license-plate-recorder"

    log_success "Run scripts created and made executable with home directory symlinks"
}

# Check Docker image
check_image() {
    log_info "Checking Docker image..."

    if docker images | grep -q "${IMAGE_NAME}"; then
        log_success "Docker image found: ${IMAGE_NAME}"
        docker images | grep "${IMAGE_NAME%:*}"
        return 0
    else
        log_error "Docker image not found: ${IMAGE_NAME}"
        echo ""
        echo "Please build and transfer the image first:"
        echo "1. On your build machine, run: ./scripts/build-for-jetson.sh"
        echo "2. Or manually load an image: docker load -i image.tar"
        return 1
    fi
}

# Show usage information
show_usage() {
    echo ""
    log_info "Setup complete! Here's how to use the system:"
    echo ""
    echo "📁 Project directory: $PROJECT_DIR"
    echo ""
    echo "🎮 Available commands (can run from anywhere):"
    echo ""
    echo "  📹 Process video file:"
    echo "     ~/run-video.sh input_videos/your-video.mp4"
    echo "     # or: cd $PROJECT_DIR && ./run-video.sh input_videos/your-video.mp4"
    echo ""
    echo "  📡 Process RTSP stream:"
    echo "     ~/run-rtsp.sh rtsp://192.168.1.100:554/stream"
    echo ""
    echo "  📦 Batch process all videos:"
    echo "     ~/run-batch.sh"
    echo ""
    echo "🔧 Interactive controls (when preview is shown):"
    echo "     q - Quit"
    echo "     e - Edit detection zones"
    echo "     z - Toggle detection zone on/off"
    echo "     h - Show help"
    echo ""
    echo "📊 Monitor system:"
    echo "     sudo jtop           # Jetson system monitor"
    echo "     docker ps           # Running containers"
    echo "     docker logs <id>    # Container logs"
    echo ""
    echo "📁 File locations:"
    echo "     $PROJECT_DIR/input_videos/       # Place your video files here"
    echo "     $PROJECT_DIR/recordings/         # Processed output videos"
    echo "     $PROJECT_DIR/logs/              # Application logs"
    echo "     $PROJECT_DIR/model_cache/        # Cached AI models"
    echo "     $PROJECT_DIR/config.yaml        # System configuration"
    echo "     $PROJECT_DIR/detection_zone.json # Detection zone settings"
    echo ""
    echo "🔗 Convenient access:"
    echo "     ~/license-plate-recorder/        # Symlink to project directory"
    echo "     ~/run-video.sh                   # Symlink to video processing script"
    echo "     ~/run-rtsp.sh                    # Symlink to RTSP stream script"
    echo "     ~/run-batch.sh                   # Symlink to batch processing script"
}

# Main function
main() {
    echo -e "${BLUE}"
    echo "🚀 License Plate Recorder - Jetson Setup"
    echo "========================================"
    echo -e "${NC}"

    check_jetson
    setup_project
    create_run_scripts

    if check_image; then
        show_usage
        log_success "Setup complete and ready to run!"
    else
        echo ""
        log_warning "Setup complete but Docker image not found"
        echo "Transfer the Docker image to continue"
    fi
}

# Handle script arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "check")
        check_image
        ;;
    "usage")
        show_usage
        ;;
    "help")
        echo "Usage: $0 [setup|check|usage|help]"
        echo ""
        echo "Commands:"
        echo "  setup    - Complete setup (default)"
        echo "  check    - Check if Docker image exists"
        echo "  usage    - Show usage information"
        echo "  help     - Show this help"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac
