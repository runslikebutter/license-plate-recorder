#!/bin/bash
# License Plate Recorder Jetson Deployment Script
# Based on stream processing pipeline approach

set -e

PROJECT_NAME="license-plate-recorder"
IMAGE_NAME="license-plate-recorder-jetson"
TAG="latest"
CONTAINER_NAME="lpr-jetson"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on Jetson
check_jetson() {
    if [ -f /etc/nv_tegra_release ]; then
        echo_info "Detected NVIDIA Jetson device"
        cat /etc/nv_tegra_release
    else
        echo_warn "Not running on Jetson device - proceeding anyway"
    fi
}

# Check Docker and NVIDIA runtime
check_requirements() {
    echo_info "Checking requirements..."

    if ! command -v docker &> /dev/null; then
        echo_error "Docker not found. Please install Docker first."
        exit 1
    fi

    # Check for NVIDIA runtime
    if docker info | grep -q nvidia; then
        echo_info "NVIDIA Docker runtime detected"
    else
        echo_warn "NVIDIA Docker runtime not detected. You may need to install nvidia-docker2"
    fi
}

# Download required wheels (placeholder - user needs to do this manually)
check_wheels() {
    echo_info "Checking required ARM64 wheels..."

    if [ ! -d "required_libs" ] || [ -z "$(ls -A required_libs 2>/dev/null | grep -v README.md)" ]; then
        echo_warn "No ARM64 wheels found in required_libs/"
        echo_warn "Please download the required wheels:"
        echo_warn "- torch-*-cp310-cp310-linux_aarch64.whl"
        echo_warn "- torchvision-*-cp310-cp310-linux_aarch64.whl"
        echo_warn "- onnxruntime_gpu-*-cp310-cp310-linux_aarch64.whl"
        echo_warn "See required_libs/README.md for download instructions"
        echo_warn "Continuing without wheels (will use pip install)..."
    else
        echo_info "Found ARM64 wheels in required_libs/"
        ls -la required_libs/*.whl 2>/dev/null || true
    fi
}

# Build Docker image
build_image() {
    echo_info "Building Docker image: ${IMAGE_NAME}:${TAG}"

    docker build \
        -f Dockerfile.jetson \
        -t ${IMAGE_NAME}:${TAG} \
        --progress=plain \
        .

    echo_info "Build completed successfully"
}

# Stop and remove existing container
cleanup_container() {
    if docker ps -a | grep -q ${CONTAINER_NAME}; then
        echo_info "Stopping and removing existing container: ${CONTAINER_NAME}"
        docker stop ${CONTAINER_NAME} || true
        docker rm ${CONTAINER_NAME} || true
    fi
}

# Run the container
run_container() {
    echo_info "Starting container: ${CONTAINER_NAME}"

    # Create host directories if they don't exist
    mkdir -p ./recordings ./input_videos ./logs

    docker run -d \
        --name ${CONTAINER_NAME} \
        --runtime nvidia \
        --gpus all \
        --restart unless-stopped \
        --privileged \
        -v $(pwd)/recordings:/app/recordings \
        -v $(pwd)/input_videos:/app/input_videos \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/config.yaml:/app/config.yaml \
        -v $(pwd)/detection_zone.json:/app/detection_zone.json \
        -p 8080:8080 \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        ${IMAGE_NAME}:${TAG}

    echo_info "Container started successfully"
    echo_info "Logs: docker logs -f ${CONTAINER_NAME}"
    echo_info "Stop: docker stop ${CONTAINER_NAME}"
}

# Show logs
show_logs() {
    echo_info "Showing container logs (Ctrl+C to exit):"
    docker logs -f ${CONTAINER_NAME}
}

# Main deployment function
deploy() {
    echo_info "Starting Jetson deployment for ${PROJECT_NAME}"

    check_jetson
    check_requirements
    check_wheels
    build_image
    cleanup_container
    run_container

    echo_info "Deployment completed!"
    echo_info "Container is running. Use './deploy-jetson.sh logs' to view logs"
}

# Handle command line arguments
case "${1:-deploy}" in
    "build")
        check_requirements
        check_wheels
        build_image
        ;;
    "run")
        cleanup_container
        run_container
        ;;
    "logs")
        show_logs
        ;;
    "stop")
        echo_info "Stopping container: ${CONTAINER_NAME}"
        docker stop ${CONTAINER_NAME}
        ;;
    "clean")
        echo_info "Cleaning up container and image"
        docker stop ${CONTAINER_NAME} || true
        docker rm ${CONTAINER_NAME} || true
        docker rmi ${IMAGE_NAME}:${TAG} || true
        ;;
    "deploy"|*)
        deploy
        ;;
esac