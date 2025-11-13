"""
Base detector interface for license plate detection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Detection:
    """Represents a single detection result"""
    class_id: int
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    class_name: str
    track_id: Optional[int] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None

    def __post_init__(self):
        """Ensure bbox is properly formatted"""
        if isinstance(self.bbox, np.ndarray):
            self.bbox = self.bbox.tolist()


class Detector(ABC):
    """Abstract base class for all detectors"""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame

        Args:
            frame: Input image frame

        Returns:
            List of Detection objects
        """
        pass

    @abstractmethod
    def get_performance_stats(self) -> dict:
        """
        Get detector performance statistics

        Returns:
            Dictionary with performance metrics
        """
        pass