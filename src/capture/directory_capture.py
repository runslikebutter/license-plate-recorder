"""
Directory batch processing for license plate detection
Handles processing multiple video files in a directory
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from capture.file_capture import FileCapture
from utils.logger import setup_logger


class DirectoryCapture:
    """Batch processes video files from a directory"""

    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv', '.flv', '.webm'}

    def __init__(self, directory_path: str, recursive: bool = False, maintain_fps: bool = True):
        """
        Initialize directory capture

        Args:
            directory_path: Path to directory containing video files
            recursive: Whether to search subdirectories recursively
            maintain_fps: Whether to maintain original file FPS timing
        """
        self.directory_path = Path(directory_path)
        self.recursive = recursive
        self.maintain_fps = maintain_fps
        self.logger = setup_logger(self.__class__.__name__)

        # File processing state
        self.video_files: List[Path] = []
        self.current_file_index = 0
        self.current_capture: Optional[FileCapture] = None
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0

        # Statistics
        self.start_time = 0.0
        self.total_frames_processed = 0

    def initialize(self) -> bool:
        """Initialize and scan for video files"""
        try:
            if not self.directory_path.exists():
                self.logger.error(f"Directory not found: {self.directory_path}")
                return False

            if not self.directory_path.is_dir():
                self.logger.error(f"Path is not a directory: {self.directory_path}")
                return False

            # Scan for video files
            self._scan_video_files()

            if not self.video_files:
                self.logger.error(f"No video files found in: {self.directory_path}")
                return False

            self.total_files = len(self.video_files)
            self.current_file_index = 0
            self.processed_files = 0
            self.failed_files = 0
            self.start_time = time.time()

            self.logger.info(
                f"Found {self.total_files} video files in {self.directory_path} "
                f"(recursive: {self.recursive})"
            )

            # Log first few files
            for i, file_path in enumerate(self.video_files[:5]):
                self.logger.info(f"  {i+1}. {file_path.name}")
            if len(self.video_files) > 5:
                self.logger.info(f"  ... and {len(self.video_files) - 5} more files")

            return True

        except Exception as e:
            self.logger.error(f"Error initializing directory capture: {e}")
            return False

    def _scan_video_files(self):
        """Scan directory for video files"""
        self.video_files = []

        try:
            if self.recursive:
                # Recursive search
                for file_path in self.directory_path.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                        self.video_files.append(file_path)
            else:
                # Non-recursive search
                for file_path in self.directory_path.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                        self.video_files.append(file_path)

            # Sort files for consistent processing order
            self.video_files.sort()

        except Exception as e:
            self.logger.error(f"Error scanning directory: {e}")
            self.video_files = []

    def connect_next_file(self) -> bool:
        """Connect to the next video file in the directory"""
        # Clean up current capture
        if self.current_capture:
            self.current_capture.disconnect()
            self.current_capture = None

        # Check if we have more files
        if self.current_file_index >= len(self.video_files):
            self.logger.info("Finished processing all video files")
            return False

        # Get next file
        current_file = self.video_files[self.current_file_index]
        self.logger.info(
            f"Processing file {self.current_file_index + 1}/{self.total_files}: {current_file.name}"
        )

        # Create capture for current file
        self.current_capture = FileCapture(str(current_file), self.maintain_fps)

        if self.current_capture.connect():
            file_info = self.current_capture.get_file_info()
            self.logger.info(
                f"  Resolution: {file_info['resolution'][0]}x{file_info['resolution'][1]} "
                f"@ {file_info['fps']:.1f}fps, {file_info['total_frames']} frames "
                f"({file_info['duration_seconds']:.1f}s)"
            )
            return True
        else:
            self.logger.error(f"Failed to connect to file: {current_file}")
            self.failed_files += 1
            self.current_file_index += 1
            return self.connect_next_file()  # Try next file

    def read_frame(self) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Read next frame from current file

        Returns:
            Tuple of (frame, file_info) or None if no more frames
        """
        if not self.current_capture:
            return None

        frame = self.current_capture.read_frame()

        if frame is not None:
            self.total_frames_processed += 1
            # Return frame and current file info
            file_info = self.current_capture.get_file_info()
            return frame, file_info

        else:
            # Current file is finished, move to next
            self.logger.info(f"Finished processing: {self.get_current_file_name()}")
            self.processed_files += 1
            self.current_file_index += 1

            # Try to connect to next file
            if self.connect_next_file():
                return self.read_frame()  # Recursive call to get first frame of next file
            else:
                return None  # No more files

    def get_current_file_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current file being processed"""
        if self.current_capture:
            return self.current_capture.get_file_info()
        return None

    def get_current_file_name(self) -> str:
        """Get name of current file being processed"""
        if self.current_file_index < len(self.video_files):
            return self.video_files[self.current_file_index].name
        return "Unknown"

    def get_current_file_path(self) -> Optional[Path]:
        """Get path of current file being processed"""
        if self.current_file_index < len(self.video_files):
            return self.video_files[self.current_file_index]
        return None

    def get_batch_progress(self) -> Dict[str, Any]:
        """Get progress information for the entire batch"""
        elapsed_time = time.time() - self.start_time if self.start_time > 0 else 0
        files_remaining = self.total_files - self.processed_files - self.failed_files

        # Estimate remaining time
        if self.processed_files > 0 and elapsed_time > 0:
            avg_time_per_file = elapsed_time / self.processed_files
            estimated_remaining = avg_time_per_file * files_remaining
        else:
            estimated_remaining = 0

        progress_percent = ((self.processed_files + self.failed_files) / self.total_files * 100) if self.total_files > 0 else 0

        return {
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'current_file_index': self.current_file_index,
            'files_remaining': files_remaining,
            'progress_percent': progress_percent,
            'elapsed_time': elapsed_time,
            'estimated_remaining': estimated_remaining,
            'total_frames_processed': self.total_frames_processed,
            'current_file': self.get_current_file_name()
        }

    def get_file_list(self) -> List[str]:
        """Get list of all video files found"""
        return [str(file_path) for file_path in self.video_files]

    def skip_current_file(self) -> bool:
        """Skip current file and move to next"""
        if self.current_capture:
            self.logger.info(f"Skipping file: {self.get_current_file_name()}")
            self.current_capture.disconnect()
            self.current_capture = None

        self.failed_files += 1
        self.current_file_index += 1

        return self.connect_next_file()

    def seek_to_file(self, file_index: int) -> bool:
        """Seek to specific file by index"""
        if 0 <= file_index < len(self.video_files):
            if self.current_capture:
                self.current_capture.disconnect()
                self.current_capture = None

            self.current_file_index = file_index
            return self.connect_next_file()
        else:
            self.logger.warning(f"File index {file_index} out of range (0-{len(self.video_files)-1})")
            return False

    def is_finished(self) -> bool:
        """Check if all files have been processed"""
        return self.current_file_index >= len(self.video_files)

    def disconnect(self):
        """Disconnect and clean up"""
        if self.current_capture:
            self.current_capture.disconnect()
            self.current_capture = None

        self.logger.info("Disconnected from directory capture")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        batch_progress = self.get_batch_progress()

        return {
            'directory_path': str(self.directory_path),
            'recursive_search': self.recursive,
            'supported_extensions': list(self.SUPPORTED_EXTENSIONS),
            'batch_progress': batch_progress,
            'processing_stats': {
                'start_time': self.start_time,
                'total_frames_processed': self.total_frames_processed,
                'avg_frames_per_file': self.total_frames_processed / max(1, self.processed_files)
            }
        }