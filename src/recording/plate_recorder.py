"""
License plate recording system with event-triggered recording and buffers
Manages pre/post event recording based on license plate detections
"""

import cv2
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from .circular_buffer import CircularBuffer
from ..utils.logger import setup_logger


class RecordingState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    POST_RECORDING = "post_recording"


class PlateRecorder:
    """Manages license plate triggered recording with pre/post buffers"""

    def __init__(self,
                 output_dir: str,
                 pre_plate_duration: float = 3.0,
                 post_plate_duration: float = 5.0,
                 max_recording_duration: float = 30.0,
                 fps: float = 30.0,
                 video_codec: str = 'mp4v'):
        """
        Initialize the plate recorder

        Args:
            output_dir: Directory to save recordings
            pre_plate_duration: Seconds to record before plate detection
            post_plate_duration: Seconds to record after plate detection
            max_recording_duration: Maximum recording length
            fps: Video frame rate
            video_codec: Video codec for output files
        """
        self.output_dir = Path(output_dir)
        self.pre_plate_duration = pre_plate_duration
        self.post_plate_duration = post_plate_duration
        self.max_recording_duration = max_recording_duration
        self.fps = fps
        self.video_codec = video_codec

        self.logger = setup_logger(self.__class__.__name__)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Recording state
        self.state = RecordingState.IDLE
        self.current_recording_start_time = None
        self.current_recording_end_time = None
        self.plate_detection_time = None
        self.triggered_plate_text = None
        self.current_video_writer = None
        self.current_output_file = None

        # Circular buffer for pre-recording (reduced for memory efficiency)
        buffer_duration = pre_plate_duration + 1.0  # Minimal extra buffer
        self.frame_buffer = CircularBuffer(buffer_duration, int(fps))

        # Recording metrics
        self.total_recordings = 0
        self.frames_written = 0

        self.logger.info(
            f"PlateRecorder initialized: pre={pre_plate_duration}s, "
            f"post={post_plate_duration}s, max={max_recording_duration}s"
        )

    def add_frame(self, frame, timestamp: Optional[float] = None):
        """Add frame to buffer (always call this for every frame)"""
        if timestamp is None:
            timestamp = time.time()

        # Always add to circular buffer for pre-recording capability
        self.frame_buffer.add_frame(frame, timestamp)

        # If we're actively recording, write to video file
        if self.state == RecordingState.RECORDING and self.current_video_writer:
            self.current_video_writer.write(frame)
            self.frames_written += 1

        # Check if we should stop recording
        self._check_recording_completion(timestamp)

    def trigger_recording(self, plate_text: str, trigger_time: Optional[float] = None) -> bool:
        """
        Trigger a new recording based on license plate detection

        Args:
            plate_text: The detected license plate text
            trigger_time: Time when plate was detected

        Returns:
            True if recording was started successfully
        """
        if trigger_time is None:
            trigger_time = time.time()

        # Don't start new recording if already recording
        if self.state != RecordingState.IDLE:
            self.logger.debug(f"Cannot trigger recording - already in state: {self.state.value}")
            return False

        self.logger.info(f"Triggering recording for plate: '{plate_text}'")

        # Set recording state
        self.state = RecordingState.RECORDING
        self.plate_detection_time = trigger_time
        self.triggered_plate_text = plate_text
        self.current_recording_start_time = trigger_time
        self.current_recording_end_time = trigger_time + self.post_plate_duration
        self.frames_written = 0

        # Generate output filename
        self.current_output_file = self._generate_output_filename(plate_text, trigger_time)

        # Initialize video writer
        if not self._initialize_video_writer():
            self.logger.error("Failed to initialize video writer")
            self._reset_recording_state()
            return False

        # Write pre-recording frames
        pre_recording_start = trigger_time - self.pre_plate_duration
        pre_frames = self.frame_buffer.get_frames_since(pre_recording_start)

        self.logger.info(
            f"Writing {len(pre_frames)} pre-recording frames "
            f"(from {self.pre_plate_duration}s before detection)"
        )

        for frame, frame_time in pre_frames:
            if frame_time >= pre_recording_start:
                self.current_video_writer.write(frame)
                self.frames_written += 1

        self.total_recordings += 1
        return True

    def stop_recording(self) -> Optional[str]:
        """
        Stop current recording and return output file path

        Returns:
            Path to saved recording file, or None if no recording was active
        """
        if self.state == RecordingState.IDLE:
            return None

        output_file = self.current_output_file
        plate_text = self.triggered_plate_text
        frames_written = self.frames_written

        # Clean up video writer
        if self.current_video_writer:
            self.current_video_writer.release()
            self.current_video_writer = None

        # Reset state
        self._reset_recording_state()

        if output_file and frames_written > 0:
            # Verify file was created
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                duration = frames_written / self.fps
                self.logger.info(
                    f"Recording completed: '{plate_text}' -> {output_file} "
                    f"({frames_written} frames, {duration:.1f}s, {file_size} bytes)"
                )
                return output_file
            else:
                self.logger.error(f"Recording file not found: {output_file}")
        else:
            self.logger.warning("Recording stopped but no frames were written")

        return None

    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self.state == RecordingState.RECORDING

    def get_state(self) -> RecordingState:
        """Get current recording state"""
        return self.state

    def get_recording_info(self) -> Dict[str, Any]:
        """Get information about current recording"""
        return {
            'state': self.state.value,
            'is_recording': self.is_recording(),
            'triggered_plate': self.triggered_plate_text,
            'frames_written': self.frames_written,
            'recording_duration': self._get_current_recording_duration(),
            'time_remaining': self._get_time_remaining(),
            'output_file': self.current_output_file,
            'total_recordings': self.total_recordings
        }

    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer information"""
        return self.frame_buffer.get_buffer_info()

    def _check_recording_completion(self, current_time: float):
        """Check if recording should be stopped"""
        if self.state != RecordingState.RECORDING:
            return

        # Check maximum recording duration
        if (self.current_recording_start_time and
            current_time - self.current_recording_start_time >= self.max_recording_duration):
            self.logger.info("Stopping recording - maximum duration reached")
            self.stop_recording()
            return

        # Check post-recording duration
        if (self.current_recording_end_time and
            current_time >= self.current_recording_end_time):
            self.logger.info("Stopping recording - post-detection duration complete")
            self.stop_recording()
            return

    def _generate_output_filename(self, plate_text: str, timestamp: float) -> str:
        """Generate output filename for recording"""
        # Clean plate text for filename
        clean_plate = "".join(c for c in plate_text if c.isalnum() or c in '-_').upper()
        if not clean_plate:
            clean_plate = "UNKNOWN"

        # Generate timestamp
        dt = datetime.fromtimestamp(timestamp)
        time_str = dt.strftime("%Y%m%d_%H%M%S")

        # Create filename
        filename = f"{time_str}_{clean_plate}.mp4"
        output_path = self.output_dir / filename

        # Handle duplicate filenames
        counter = 1
        while output_path.exists():
            filename = f"{time_str}_{clean_plate}_{counter:02d}.mp4"
            output_path = self.output_dir / filename
            counter += 1

        return str(output_path)

    def _initialize_video_writer(self) -> bool:
        """Initialize OpenCV video writer"""
        if not self.current_output_file:
            return False

        try:
            # Get first frame from buffer to determine dimensions
            frames = self.frame_buffer.get_frames()
            if not frames:
                self.logger.error("No frames in buffer to determine video dimensions")
                return False

            sample_frame = frames[-1][0]  # Get most recent frame
            height, width = sample_frame.shape[:2]

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            self.current_video_writer = cv2.VideoWriter(
                self.current_output_file,
                fourcc,
                self.fps,
                (width, height)
            )

            if not self.current_video_writer.isOpened():
                self.logger.error("Failed to open video writer")
                return False

            self.logger.debug(f"Video writer initialized: {width}x{height} @ {self.fps}fps")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing video writer: {e}")
            return False

    def _reset_recording_state(self):
        """Reset recording state to idle"""
        self.state = RecordingState.IDLE
        self.current_recording_start_time = None
        self.current_recording_end_time = None
        self.plate_detection_time = None
        self.triggered_plate_text = None
        self.current_output_file = None
        self.frames_written = 0

    def _get_current_recording_duration(self) -> Optional[float]:
        """Get duration of current recording"""
        if self.current_recording_start_time:
            return time.time() - self.current_recording_start_time
        return None

    def _get_time_remaining(self) -> Optional[float]:
        """Get time remaining in current recording"""
        if self.current_recording_end_time:
            return max(0, self.current_recording_end_time - time.time())
        return None

    def cleanup(self):
        """Clean up resources"""
        if self.current_video_writer:
            self.current_video_writer.release()
            self.current_video_writer = None

        self.frame_buffer.clear()
        self._reset_recording_state()

    def get_statistics(self) -> Dict[str, Any]:
        """Get recording statistics"""
        return {
            'total_recordings': self.total_recordings,
            'current_state': self.state.value,
            'buffer_info': self.get_buffer_info(),
            'settings': {
                'pre_plate_duration': self.pre_plate_duration,
                'post_plate_duration': self.post_plate_duration,
                'max_recording_duration': self.max_recording_duration,
                'fps': self.fps,
                'output_dir': str(self.output_dir)
            }
        }
