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

        # Reconnection settings with exponential backoff
        self.initial_reconnect_delay = config.stream.get(
            'initial_reconnect_delay', 2.0
        )
        self.max_reconnect_delay = config.stream.get(
            'max_reconnect_delay', 60.0
        )
        self.backoff_multiplier = config.stream.get(
            'backoff_multiplier', 2.0
        )
        self.max_reconnect_attempts = config.stream.get(
            'max_reconnect_attempts', 999999
        )
        self.current_reconnect_delay = self.initial_reconnect_delay
        self.reconnect_attempt = 0

        # Timeout settings
        self.connection_timeout = config.stream.get('connection_timeout', 5)
        self.read_timeout = config.stream.get('read_timeout', 5)

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.source_fps = 15
        self.frame_interval = 1.0 / self.source_fps
        self.consecutive_failures = 0

    def connect(self) -> bool:
        """Connect to RTSP stream with fast failure timeouts"""
        try:
            # Prepare RTSP URL for FFmpeg backend
            transport = self.config.stream.get('transport', 'tcp')
            rtsp_url = self.rtsp_url

            if transport and 'rtsp_transport=' not in rtsp_url.lower():
                separator = '&' if '?' in rtsp_url else '?'
                rtsp_url = f"{rtsp_url}{separator}rtsp_transport={transport}"

            self.logger.info(
                "Connecting to RTSP stream via FFmpeg backend "
                "(transport=%s, timeout=%ds)",
                transport,
                self.connection_timeout
            )
            self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

            if not self.cap.isOpened():
                self.logger.error("Failed to open RTSP stream")
                return False

            # Set buffer size for minimal latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Apply fast failure timeouts when supported by OpenCV
            connection_timeout_ms = int(self.connection_timeout * 1000)
            read_timeout_ms = int(self.read_timeout * 1000)

            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                self.cap.set(
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    connection_timeout_ms
                )
                timeout_val = self.connection_timeout
                self.logger.debug(f"Set connection timeout: {timeout_val}s")
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                self.cap.set(
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    read_timeout_ms
                )
                self.logger.debug(f"Set read timeout: {self.read_timeout}s")

            # Prime stream and record FPS
            self.source_fps = self._detect_source_fps()
            if self.source_fps > 0:
                self.frame_interval = 1.0 / self.source_fps
            else:
                self.frame_interval = 0.0

            ret, frame = self.cap.read()

            if ret and frame is not None:
                self.is_connected = True
                self.consecutive_failures = 0
                # Reset reconnection delay on successful connection
                self.current_reconnect_delay = self.initial_reconnect_delay
                self.reconnect_attempt = 0
                self.logger.info(
                    "Successfully connected to RTSP stream: %s",
                    self.rtsp_url
                )
                return True
            else:
                self.logger.error(
                    "Failed to read initial frame from RTSP stream"
                )
                return False

        except Exception as e:
            self.logger.error(f"Exception during RTSP connection: {e}")
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read frame directly from RTSP stream with fast failure detection
        Returns None on failure - caller should trigger reconnection
        """
        if not self.cap or not self.is_connected:
            self.logger.warning("Attempted to read frame while not connected")
            return None

        try:
            ret, frame = self.cap.read()

            if ret and frame is not None and frame.size > 0:
                # Successful read - reset failure counter
                self.consecutive_failures = 0
                return frame

            # Failed to read frame
            self.consecutive_failures += 1
            self.logger.warning(
                "Failed to read frame from RTSP stream "
                "(consecutive failures: %d)",
                self.consecutive_failures
            )

            # Mark as disconnected after first failure for reconnection
            if self.consecutive_failures >= 1:
                self.is_connected = False

            return None

        except Exception as e:
            self.consecutive_failures += 1
            self.logger.error(
                "Exception reading frame: %s (consecutive failures: %d)",
                e,
                self.consecutive_failures
            )
            self.is_connected = False
            return None

    def reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff
        Returns True if reconnection successful, False otherwise
        """
        self.reconnect_attempt += 1

        # Check if we've exceeded max attempts
        if self.reconnect_attempt > self.max_reconnect_attempts:
            self.logger.error(
                "Maximum reconnection attempts (%d) exceeded. Giving up.",
                self.max_reconnect_attempts
            )
            return False

        # Log reconnection attempt with backoff info
        max_attempts_str = (
            str(self.max_reconnect_attempts)
            if self.max_reconnect_attempts < 999999 else "∞"
        )
        self.logger.info(
            "Reconnection attempt %d/%s - waiting %.1fs before retry...",
            self.reconnect_attempt,
            max_attempts_str,
            self.current_reconnect_delay
        )

        # Disconnect cleanly
        self.disconnect()

        # Wait with exponential backoff
        time.sleep(self.current_reconnect_delay)

        # Attempt to reconnect
        success = self.connect()

        if success:
            self.logger.info(
                "Reconnection successful after %d attempt(s)",
                self.reconnect_attempt
            )
            # Reset on successful connection (done in connect())
            return True
        else:
            # Increase backoff delay for next attempt (exponential backoff)
            self.current_reconnect_delay = min(
                self.current_reconnect_delay * self.backoff_multiplier,
                self.max_reconnect_delay
            )
            self.logger.warning(
                "Reconnection attempt %d failed. Next delay: %.1fs",
                self.reconnect_attempt,
                self.current_reconnect_delay
            )
            return False

    def reset_reconnection_state(self):
        """Reset reconnection state (useful after successful connection)"""
        self.reconnect_attempt = 0
        self.current_reconnect_delay = self.initial_reconnect_delay
        self.consecutive_failures = 0

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
