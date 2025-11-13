import numpy as np
from collections import deque
from threading import Lock
from typing import List, Tuple, Optional
import time


class CircularBuffer:
    def __init__(self, duration_seconds: float, fps: int = 30):
        self.duration = duration_seconds
        self.fps = fps
        self.max_frames = int(duration_seconds * fps)
        self.buffer = deque(maxlen=self.max_frames)
        self.lock = Lock()

    def add_frame(self, frame: np.ndarray, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()

        with self.lock:
            self.buffer.append((frame, timestamp))

    def get_frames(self) -> List[Tuple[np.ndarray, float]]:
        with self.lock:
            return list(self.buffer)

    def get_frames_since(self, start_time: float) -> List[Tuple[np.ndarray, float]]:
        with self.lock:
            return [(f, t) for f, t in self.buffer if t >= start_time]

    def get_frames_in_range(self, start_time: float, end_time: float) -> List[Tuple[np.ndarray, float]]:
        """Get frames within a specific time range"""
        with self.lock:
            return [(f, t) for f, t in self.buffer if start_time <= t <= end_time]

    def clear(self):
        with self.lock:
            self.buffer.clear()

    def is_full(self) -> bool:
        return len(self.buffer) == self.max_frames

    def __len__(self) -> int:
        return len(self.buffer)

    def get_oldest_timestamp(self) -> Optional[float]:
        with self.lock:
            if self.buffer:
                return self.buffer[0][1]
        return None

    def get_newest_timestamp(self) -> Optional[float]:
        with self.lock:
            if self.buffer:
                return self.buffer[-1][1]
        return None

    def get_buffer_info(self) -> dict:
        """Get information about buffer state"""
        with self.lock:
            oldest_ts = self.buffer[0][1] if self.buffer else None
            newest_ts = self.buffer[-1][1] if self.buffer else None

            return {
                'buffer_frames': len(self.buffer),
                'max_buffer_frames': self.max_frames,
                'duration_seconds': self.duration,
                'fps': self.fps,
                'is_full': self.is_full(),
                'oldest_timestamp': oldest_ts,
                'newest_timestamp': newest_ts
            }