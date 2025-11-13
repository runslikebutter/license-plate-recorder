import cv2
import time
import numpy as np
from typing import Optional, Callable
from ..utils.logger import setup_logger
from ..utils.config import Config


class RTSPCapture:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

        self.rtsp_url = config.stream['rtsp_url']
        self.reconnect_delay = config.stream['reconnect_delay']
        self.connection_timeout = config.stream['connection_timeout']

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False

    def connect(self) -> bool:
        try:
            # Set up GStreamer pipeline for OpenCV
            if self.config.gstreamer.get('use_hardware_decoding', False):
                # For Jetson with hardware decoding
                gst_pipeline = (
                    f"rtspsrc location={self.rtsp_url} latency=0 ! "
                    "rtph264depay ! h264parse ! nvv4l2decoder ! "
                    "nvvidconv ! video/x-raw,format=BGRx ! "
                    "videoconvert ! video/x-raw,format=BGR ! appsink"
                )
            else:
                # Software decoding
                gst_pipeline = (
                    f"rtspsrc location={self.rtsp_url} latency=0 ! "
                    "rtph264depay ! h264parse ! avdec_h264 ! "
                    "videoconvert ! video/x-raw,format=BGR ! appsink"
                )

            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

            # Set buffer size
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Test connection
            ret, _ = self.cap.read()
            if ret:
                self.is_connected = True
                self.logger.info(f"Connected to RTSP stream: {self.rtsp_url}")
                return True
            else:
                self.logger.error("Failed to read from RTSP stream")
                return False

        except Exception as e:
            self.logger.error(f"Failed to connect to RTSP stream: {e}")
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        if not self.cap or not self.is_connected:
            return None

        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                self.logger.warning("Failed to read frame")
                self.is_connected = False
                return None

        except Exception as e:
            self.logger.error(f"Error reading frame: {e}")
            self.is_connected = False
            return None

    def reconnect(self) -> bool:
        self.logger.info(f"Attempting to reconnect in {self.reconnect_delay} seconds...")
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