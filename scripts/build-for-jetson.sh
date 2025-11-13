#!/bin/bash

# Build Docker image for Jetson Nano Orin on local system
# This script uses Docker Buildx for cross-platform builds

set -e

# Configuration
IMAGE_NAME="license-plate-recorder"
IMAGE_TAG="jetson-arm64"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
JETSON_HOST="monarch@192.168.1.87"
JETSON_IMAGE_PATH="/mnt/data/license-plate-recorder/${IMAGE_NAME}-${IMAGE_TAG}.tar"

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

# Check Docker
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    log_success "Prerequisites checked"
}

# Build the image (will be compatible since Jetson runs on ARM64 and we're building for the target)
build_image() {
    log_info "Building Docker image for Jetson..."

    cd "$(dirname "$0")/.."

    # Simple docker build (the Jetson base image will handle ARM64 compatibility)
    if docker build \
        --tag "${FULL_IMAGE_NAME}" \
        --file Dockerfile \
        .; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Save image to tar file
save_image() {
    log_info "Saving Docker image to tar file..."

    local tar_file="/tmp/${IMAGE_NAME}-${IMAGE_TAG}.tar"

    if docker save -o "${tar_file}" "${FULL_IMAGE_NAME}"; then
        log_success "Image saved to ${tar_file}"
        log_info "Image size: $(du -h "${tar_file}" | cut -f1)"
        echo "${tar_file}"
    else
        log_error "Failed to save Docker image"
        exit 1
    fi
}

# Transfer image to Jetson
transfer_image() {
    local tar_file="$1"

    log_info "Transferring image to Jetson (${JETSON_HOST})..."

    # Test SSH connectivity
    if ! ssh -o ConnectTimeout=10 "${JETSON_HOST}" "echo 'SSH connection successful'" &> /dev/null; then
        log_error "Cannot connect to Jetson via SSH. Please check:"
        echo "  - SSH key is set up: ssh-copy-id ${JETSON_HOST}"
        echo "  - Jetson is accessible: ping 192.168.1.87"
        echo "  - SSH service is running on Jetson"
        exit 1
    fi

    # Transfer the tar file
    log_info "Uploading image (this may take several minutes)..."
    if scp "${tar_file}" "${JETSON_HOST}:${JETSON_IMAGE_PATH}"; then
        log_success "Image transferred successfully"
    else
        log_error "Failed to transfer image"
        exit 1
    fi

    # Load image on Jetson
    log_info "Loading image on Jetson..."
    if ssh "${JETSON_HOST}" "docker load -i ${JETSON_IMAGE_PATH}"; then
        log_success "Image loaded on Jetson"

        # Clean up the tar file on Jetson
        ssh "${JETSON_HOST}" "rm ${JETSON_IMAGE_PATH}"
        log_success "Cleaned up temporary file on Jetson"
    else
        log_error "Failed to load image on Jetson"
        exit 1
    fi

    # Transfer deployment script and source code to Jetson
    log_info "Transferring deployment script and source code to Jetson..."
    if scp "$(dirname "$0")/jetson-deploy.sh" "${JETSON_HOST}:/mnt/data/license-plate-recorder/jetson-deploy.sh"; then
        ssh "${JETSON_HOST}" "chmod +x /mnt/data/license-plate-recorder/jetson-deploy.sh"
        log_success "Deployment script transferred and made executable"
    else
        log_error "Failed to transfer deployment script"
        exit 1
    fi

    # Transfer source code
    log_info "Transferring source code to Jetson..."
    if scp -r src/ main.py "${JETSON_HOST}:/mnt/data/license-plate-recorder/"; then
        log_success "Source code transferred successfully"
    else
        log_error "Failed to transfer source code"
        exit 1
    fi

    # Clean up local tar file
    rm "${tar_file}"
    log_success "Cleaned up local temporary file"

    # Run deployment setup on Jetson
    log_info "Running deployment setup on Jetson..."
    if ssh "${JETSON_HOST}" "/mnt/data/license-plate-recorder/jetson-deploy.sh setup"; then
        log_success "Deployment setup completed on Jetson"
    else
        log_warning "Deployment setup encountered issues (this is normal if already set up)"
    fi
}

# Verify image on Jetson
verify_image() {
    log_info "Verifying image on Jetson..."

    if ssh "${JETSON_HOST}" "docker images | grep -q ${IMAGE_NAME}"; then
        log_success "Image verified on Jetson"

        # Show image info
        log_info "Image details on Jetson:"
        ssh "${JETSON_HOST}" "docker images ${IMAGE_NAME}"
    else
        log_error "Image verification failed"
        exit 1
    fi
}


# Main function
main() {
    echo -e "${BLUE}"
    echo "🚀 Building License Plate Recorder for Jetson Nano Orin"
    echo "====================================================="
    echo -e "${NC}"

    check_prerequisites
    build_image
    local tar_file=$(save_image)
    transfer_image "$tar_file"
    verify_image

    echo ""
    log_success "Build and deployment complete!"
    echo ""
    echo "📋 Next steps on Jetson (${JETSON_HOST}):"
    echo ""
    echo "1. SSH to Jetson:"
    echo "   ssh ${JETSON_HOST}"
    echo ""
    echo "2. Add your video files:"
    echo "   cp /path/to/your/videos/*.mp4 ~/license-plate-recorder/input_videos/"
    echo ""
    echo "3. Run the system (using convenient symlinks):"
    echo "   ~/run-video.sh input_videos/your-video.mp4"
    echo "   ~/run-rtsp.sh rtsp://camera-ip:554/stream"
    echo "   ~/run-batch.sh  # Process all files in input_videos/"
    echo ""
    echo "4. Or use the project directory directly:"
    echo "   cd /mnt/data/license-plate-recorder"
    echo "   ./run-video.sh input_videos/your-video.mp4"
    echo ""
    echo "🎮 Controls when running with preview:"
    echo "   q - Quit"
    echo "   e - Edit detection zones"
    echo "   z - Toggle detection zone"
    echo "   h - Show help"
    echo ""
}

# Handle script arguments
case "${1:-build}" in
    "build")
        main
        ;;
    "transfer-only")
        if [ -f "/tmp/${IMAGE_NAME}-${IMAGE_TAG}.tar" ]; then
            transfer_image "/tmp/${IMAGE_NAME}-${IMAGE_TAG}.tar"
        else
            log_error "No image file found. Run full build first."
            exit 1
        fi
        ;;
    "verify")
        verify_image
        ;;
    "help")
        echo "Usage: $0 [build|transfer-only|verify|help]"
        echo ""
        echo "Commands:"
        echo "  build         - Complete build and transfer (default)"
        echo "  transfer-only - Transfer existing image file"
        echo "  verify        - Verify image on Jetson"
        echo "  help          - Show this help"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac