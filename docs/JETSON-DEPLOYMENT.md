# License Plate Recorder - Jetson Deployment Guide

This guide shows how to deploy the License Plate Recorder on NVIDIA Jetson devices using Docker, following the same pattern as the stream processing pipeline.

## Prerequisites

### Hardware Requirements
- NVIDIA Jetson Nano, Xavier NX, or Orin series
- At least 8GB storage space
- Camera or RTSP stream source

### Software Requirements
- JetPack 6.0+ (Ubuntu 22.04)
- Docker with NVIDIA runtime support
- Python 3.10

## Quick Start

### 1. Prepare ARM64 Wheels (Required)

Download the required ARM64 wheels to the `required_libs/` directory:

```bash
# Create directory
mkdir -p required_libs

# Download PyTorch for Jetson (example URLs - check NVIDIA forums for latest)
cd required_libs
wget https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl
wget https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
wget https://nvidia.box.com/shared/static/48dtuob7meiw6ebgfsfqakc9vse62sg4.whl -O onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl
cd ..
```

See `required_libs/README.md` for detailed download instructions.

### 2. Deploy to Jetson

```bash
# Single command deployment
./deploy-jetson.sh

# Or step by step:
./deploy-jetson.sh build    # Build image only
./deploy-jetson.sh run      # Run container only
./deploy-jetson.sh logs     # View logs
./deploy-jetson.sh stop     # Stop container
./deploy-jetson.sh clean    # Clean up everything
```

### 3. Monitor the System

```bash
# View real-time logs
./deploy-jetson.sh logs

# Check container status
docker ps

# Monitor Jetson resources
sudo jtop
```

## Architecture

The deployment follows the stream processing pipeline pattern:

```
┌─────────────────────────────────────────┐
│ NVIDIA Jetson Device                    │
├─────────────────────────────────────────┤
│ Docker Container (l4t-jetpack:r36.3.0) │
├─────────────────────────────────────────┤
│ • TensorRT Optimized Models             │
│ • CUDA Accelerated Processing           │
│ • ARM64 Optimized Libraries             │
│ • GStreamer Pipeline Support            │
└─────────────────────────────────────────┘
```

## Key Features

### Hardware Acceleration
- **TensorRT**: GPU-accelerated inference
- **CUDA**: Parallel processing on Jetson GPU
- **GStreamer**: Hardware-accelerated video decode
- **NVENC**: Hardware video encoding (if available)

### Optimizations
- ARM64-specific wheels for maximum performance
- Memory management optimized for Jetson
- Multi-threaded processing pipeline
- Efficient video I/O with minimal CPU usage

### Container Benefits
- **Isolation**: Clean separation from host system
- **Reproducibility**: Consistent environment across devices
- **Portability**: Same container works on all Jetson models
- **Updates**: Easy version management and rollbacks

## Configuration

### Volume Mounts
```bash
./recordings     → /app/recordings     # Output videos and data
./input_videos   → /app/input_videos   # Input video files
./logs          → /app/logs           # Application logs
./config.yaml   → /app/config.yaml    # Configuration file
```

### Environment Variables
```bash
NVIDIA_VISIBLE_DEVICES=all           # Use all GPUs
NVIDIA_DRIVER_CAPABILITIES=all       # Full CUDA capabilities
PYTHONPATH=/app:/app/src            # Python module paths
YOLO_CONFIG_DIR=/app/.config/Ultralytics
```

## Performance Tuning

### Jetson Power Mode
```bash
# Maximum performance (high power consumption)
sudo nvpmodel -m 0
sudo jetson_clocks

# Balanced mode
sudo nvpmodel -m 1
```

### Memory Management
```bash
# Increase swap if needed (for large models)
sudo systemctl disable nvzramconfig
sudo fallocate -l 4G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
```

### Docker Optimization
```bash
# Increase Docker daemon memory limits
sudo systemctl edit docker
# Add:
# [Service]
# LimitMEMLOCK=infinity
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check NVIDIA runtime
docker info | grep nvidia

# Test GPU access
docker run --rm --runtime nvidia --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi
```

#### 2. Memory Issues
```bash
# Monitor memory usage
sudo jtop

# Check Docker memory
docker stats

# Clear cache
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

#### 3. Build Failures
```bash
# Clean Docker build cache
docker system prune -a

# Check ARM64 wheels
ls -la required_libs/

# Manual wheel installation test
docker run --rm -v $(pwd)/required_libs:/wheels --runtime nvidia nvcr.io/nvidia/l4t-jetpack:r36.3.0 pip install /wheels/*.whl
```

### Performance Monitoring

```bash
# Jetson stats
sudo jtop

# Container resources
docker stats license-plate-recorder-jetson

# Application logs
tail -f logs/license_plate_recorder.log

# GPU utilization
nvidia-smi
```

## Deployment Variants

### Development Mode
```bash
# Mount source code for development
docker run -it --rm \
  --runtime nvidia --gpus all \
  -v $(pwd):/app \
  license-plate-recorder-jetson \
  bash
```

### Production Mode
```bash
# Run with automatic restart
./deploy-jetson.sh
```

### Multi-Camera Setup
```bash
# Scale for multiple cameras
docker run -d --name lpr-cam1 --runtime nvidia --gpus all \
  -v $(pwd)/recordings:/app/recordings \
  -e CAMERA_ID=1 \
  license-plate-recorder-jetson

docker run -d --name lpr-cam2 --runtime nvidia --gpus all \
  -v $(pwd)/recordings:/app/recordings \
  -e CAMERA_ID=2 \
  license-plate-recorder-jetson
```

## Next Steps

1. **Configure your camera streams** in `config.yaml`
2. **Set up detection zones** using the zone editor
3. **Monitor performance** with `jtop` and Docker stats
4. **Scale horizontally** for multiple camera streams
5. **Integrate with your existing infrastructure**

For support and updates, see the main project documentation.