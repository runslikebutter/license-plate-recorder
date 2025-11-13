# Required ARM64 Wheels for Jetson Deployment

This directory should contain the following ARM64 wheel files for Jetson compatibility:

## Required Wheels:
1. **PyTorch for Jetson**: `torch-2.3.0-cp310-cp310-linux_aarch64.whl`
   - Download from: https://forums.developer.nvidia.com/t/pytorch-for-jetson/
   - Or build from: https://github.com/pytorch/pytorch

2. **TorchVision for Jetson**: `torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl`
   - Download from: https://forums.developer.nvidia.com/t/pytorch-for-jetson/
   - Must match PyTorch version

3. **ONNX Runtime GPU for Jetson**: `onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl`
   - Download from: https://onnxruntime.ai/docs/install/#jetson-tx1tx2nanoxaviernx
   - Or from NVIDIA NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch

## Download Instructions:

### Option 1: Use NVIDIA's pre-built wheels
```bash
# PyTorch 2.3.0 for Jetson (Python 3.10)
wget https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl

# TorchVision (compatible with PyTorch 2.3.0)
wget https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl

# ONNX Runtime GPU
wget https://nvidia.box.com/shared/static/48dtuob7meiw6ebgfsfqakc9vse62sg4.whl -O onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl
```

### Option 2: Copy from working Jetson installation
If you have a working Jetson with these packages installed:
```bash
# Find the installed wheels
pip list --format=freeze | grep -E "(torch|onnxruntime)"

# Copy from pip cache or site-packages
find /usr/local/lib/python3.10/dist-packages/ -name "*.whl"
```

## Notes:
- These wheels are specifically for Jetson Orin/Nano with JetPack 6.0 (Ubuntu 22.04, Python 3.10)
- Make sure wheel versions are compatible with each other
- The Dockerfile will automatically install any .whl files found in this directory