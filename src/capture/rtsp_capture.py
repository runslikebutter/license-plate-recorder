import cv2
import time
import numpy as np
from typing import Optional
from utils.logger import setup_logger
from utils.config import Config


class RTSPCapture:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

        self.rtsp_url = config.stream['rtsp_url']
        self.reconnect_delay = config.stream['reconnect_delay']
        self.connection_timeout = config.stream['connection_timeout']

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.source_fps = 15
        self.frame_interval = 1.0 / self.source_fps

    def connect(self) -> bool:
        try:
            # Prepare RTSP URL for FFmpeg backend
            transport = self.config.stream.get('transport', 'tcp')
            rtsp_url = self.rtsp_url

            if transport and 'rtsp_transport=' not in rtsp_url.lower():
                separator = '&' if '?' in rtsp_url else '?'
                rtsp_url = f"{rtsp_url}{separator}rtsp_transport={transport}"

            self.logger.info(
                "Connecting to %s stream via FFmpeg backend (transport=%s)",
                rtsp_url,
                transport,
            )
            self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

            if not self.cap.isOpened():
                self.logger.error("Failed to open RTSP stream")
                return False
            else:
                self.logger.info("Connected to RTSP stream")

            # Set buffer size
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Apply timeouts when supported by the OpenCV build
            timeout_ms = int(self.connection_timeout * 1000)
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)

            # Prime stream and record FPS
            self.source_fps = self._detect_source_fps()
            if self.source_fps > 0:
                self.frame_interval = 1.0 / self.source_fps
            else:
                self.frame_interval = 0.0

            ret, frame = self.cap.read()

            if ret and frame is not None:
                self.is_connected = True
                self.logger.info("Connected to RTSP stream: %s", self.rtsp_url)
                return True
            else:
                self.logger.error("Failed to read from RTSP stream")
                return False

        except Exception as e:
            self.logger.error(f"Failed to connect to RTSP stream: {e}")
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read frame directly from RTSP stream (blocking)
        Blocks until next frame arrives or timeout (10s)
        """
        if not self.cap or not self.is_connected:
            return None

        try:
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                return frame
            
            self.logger.warning("Failed to read frame from RTSP stream")
            return None

        except Exception as e:
            self.logger.error(f"Error reading frame: {e}")
            self.is_connected = False
            return None

    def reconnect(self) -> bool:
        self.logger.info(
            "Attempting to reconnect in %s seconds...",
            self.reconnect_delay,
        )
        self.disconnect()
        time.sleep(self.reconnect_delay)
        return self.connect()

    def disconnect(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        self.logger.info("Disconnected from RTSP stream")

    def get_fps(self) -> int:
        if self.cap:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return int(fps) if fps > 0 else 30
        return 30

    def get_resolution(self) -> tuple:
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (1920, 1080)

    def _detect_source_fps(self) -> float:
        """Read FPS from the capture; fallback to configured default."""
        fallback_fps = float(self.config.stream.get('fallback_fps', 30.0))
        if not self.cap:
            return fallback_fps

        fps = 0.0

        for attempt in range(3):
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
            if fps > 0:
                if attempt > 0:
                    self.logger.info("Source FPS detected on retry: %s", fps)
                break
            time.sleep(0.1)

        if fps <= 0:
            self.logger.warning(
                "Unable to determine source FPS from camera; defaulting to %s",
                fallback_fps,
            )
            fps = float(fallback_fps)
        else:
            self.logger.info("Detected source FPS: %s", fps)

        return fps
