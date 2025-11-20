"""
Video file capture handler for license plate detection
Handles single video file processing with proper frame rate control
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from utils.logger import setup_logger


class FileCapture:
    """Captures frames from a video file with proper timing control"""

    def __init__(self, file_path: str, maintain_fps: bool = True):
        """
        Initialize file capture

        Args:
            file_path: Path to video file
            maintain_fps: Whether to maintain original file FPS timing
        """
        self.file_path = Path(file_path)
        self.maintain_fps = maintain_fps
        self.logger = setup_logger(self.__class__.__name__)

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame = 0
        self.width = 0
        self.height = 0

        # Timing control
        self.last_frame_time = 0.0
        self.frame_interval = 1.0 / 30.0  # Default to 30 FPS

    def connect(self) -> bool:
        """Connect to video file"""
        try:
            if not self.file_path.exists():
                self.logger.error(f"Video file not found: {self.file_path}")
                return False

            self.cap = cv2.VideoCapture(str(self.file_path))

            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video file: {self.file_path}")
                return False

            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0  # Default fallback
                self.logger.warning("Invalid FPS detected, using 30 FPS")

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.frame_interval = 1.0 / self.fps
            self.current_frame = 0
            self.last_frame_time = time.time()
            self.is_connected = True

            self.logger.info(
                f"Connected to video file: {self.file_path.name} "
                f"({self.width}x{self.height} @ {self.fps:.1f}fps, {self.total_frames} frames)"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error opening video file: {e}")
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame from video file"""
        if not self.cap or not self.is_connected:
            return None

        try:
            # Check if we've reached the end
            if self.current_frame >= self.total_frames:
                self.logger.info("Reached end of video file")
                self.is_connected = False
                return None

            # Timing control for FPS maintenance
            if self.maintain_fps:
                current_time = time.time()
                time_since_last = current_time - self.last_frame_time

                if time_since_last < self.frame_interval:
                    # Sleep to maintain proper frame rate
                    sleep_time = self.frame_interval - time_since_last
                    time.sleep(sleep_time)

                self.last_frame_time = time.time()

            # Read frame
            ret, frame = self.cap.read()

            if ret:
                self.current_frame += 1
                return frame
            else:
                self.logger.info("Reached end of video file")
                self.is_connected = False
                return None

        except Exception as e:
            self.logger.error(f"Error reading frame: {e}")
            self.is_connected = False
            return None

    def seek_frame(self, frame_number: int) -> bool:
        """Seek to specific frame number"""
        if not self.cap or not self.is_connected:
            return False

        try:
            if 0 <= frame_number < self.total_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                self.current_frame = frame_number
                return True
            else:
                self.logger.warning(f"Frame number {frame_number} out of range (0-{self.total_frames-1})")
                return False

        except Exception as e:
            self.logger.error(f"Error seeking to frame {frame_number}: {e}")
            return False

    def seek_time(self, time_seconds: float) -> bool:
        """Seek to specific time in seconds"""
        frame_number = int(time_seconds * self.fps)
        return self.seek_frame(frame_number)

    def get_progress(self) -> Tuple[int, int, float]:
        """Get current progress information"""
        progress_percent = (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0
        return self.current_frame, self.total_frames, progress_percent

    def get_time_info(self) -> Tuple[float, float]:
        """Get current time and total duration in seconds"""
        current_time = self.current_frame / self.fps if self.fps > 0 else 0
        total_time = self.total_frames / self.fps if self.fps > 0 else 0
        return current_time, total_time

    def disconnect(self):
        """Disconnect from video file"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        self.logger.info(f"Disconnected from video file: {self.file_path.name}")

    def get_fps(self) -> float:
        """Get video FPS"""
        return self.fps

    def get_resolution(self) -> Tuple[int, int]:
        """Get video resolution"""
        return (self.width, self.height)

    def get_total_frames(self) -> int:
        """Get total number of frames"""
        return self.total_frames

    def get_current_frame_number(self) -> int:
        """Get current frame number"""
        return self.current_frame

    def is_end_of_file(self) -> bool:
        """Check if reached end of file"""
        return self.current_frame >= self.total_frames

    def reset(self) -> bool:
        """Reset to beginning of file"""
        return self.seek_frame(0)

    def get_file_info(self) -> dict:
        """Get comprehensive file information"""
        current_time, total_time = self.get_time_info()
        current_frame, total_frames, progress = self.get_progress()

        return {
            'file_path': str(self.file_path),
            'file_name': self.file_path.name,
            'resolution': (self.width, self.height),
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration_seconds': total_time,
            'current_frame': self.current_frame,
            'current_time': current_time,
            'progress_percent': progress,
            'is_connected': self.is_connected,
            'end_of_file': self.is_end_of_file()
        }