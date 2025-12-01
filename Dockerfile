FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

RUN apt-get update && \
    apt-get install -y \
    wget python3 python3-pip python3-virtualenv \
    libopenmpi3 libopenblas-base python3-gst-1.0 \
    python3-gi libgirepository1.0-dev gstreamer-1.0 \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    htop \
    vim \
    ffmpeg \
    python3-cairo && \
    pip3 install -U jetson-stats && \
    rm -rf /var/lib/apt/lists/*

RUN addgroup --gid 1200 monarch && \
    adduser --uid 1200 --ingroup monarch --disabled-password --gecos "" monarch && \
    mkdir -p /home/monarch/.local/bin && \
    chown -R monarch:monarch /home/monarch && \
    mkdir -p /etc/recorder && \
    mkdir -p /etc/recorder/models && \
    chown -R monarch:monarch /etc/recorder && \
    mkdir -p /app && \
    chown monarch:monarch /app

USER monarch

# Set the PATH environment variable for the monarch user
ENV PATH="/home/monarch/.local/bin:${PATH}"

WORKDIR /app

COPY --chown=monarch:monarch required_libs/ required_libs/
COPY --chown=monarch:monarch requirements.txt .
COPY --chown=monarch:monarch setup.py .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir pycuda

COPY --chown=monarch:monarch src/ src/
COPY --chown=monarch:monarch src/config.yaml /etc/recorder/config.yaml
COPY --chown=monarch:monarch models/car_lpd_11nv6.engine /etc/recorder/models/
COPY --chown=monarch:monarch models/fast_plate_ocr /etc/recorder/models/

RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/.config/Ultralytics
ENV YOLO_CONFIG_DIR=/app/.config/Ultralytics

# Use ENTRYPOINT + CMD to allow passing arguments
# Default to config at /etc/recorder/config.yaml
ENTRYPOINT ["python3", "src/main.py", "--config", "/etc/recorder/config.yaml"]
CMD []
