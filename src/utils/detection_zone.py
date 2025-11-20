import numpy as np
from typing import List, Tuple, Dict, Optional
from .zone_config import ZoneConfigManager


class DetectionZone:
    def __init__(self, top_percent: float = 0, bottom_percent: float = 0,
                 left_percent: float = 0, right_percent: float = 0,
                 config_manager: Optional[ZoneConfigManager] = None):
        """
        Initialize detection zone with percentage padding from edges.

        Args:
            top_percent: Percentage to skip from top (0-50)
            bottom_percent: Percentage to skip from bottom (0-50)
            left_percent: Percentage to skip from left (0-50)
            right_percent: Percentage to skip from right (0-50)
            config_manager: Optional zone configuration manager
        """
        # Initialize config manager
        self.config_manager = config_manager or ZoneConfigManager()

        # Check if this is a default initialization (no parameters provided)
        # In this case, load from saved config
        if (top_percent == 0 and bottom_percent == 0 and
            left_percent == 0 and right_percent == 0):
            saved_config = self.config_manager.load_zone_config()
            if saved_config.get('enabled', True):  # Only load if zone was enabled
                top_percent = saved_config.get('top', 0)
                bottom_percent = saved_config.get('bottom', 0)
                left_percent = saved_config.get('left', 0)
                right_percent = saved_config.get('right', 0)

        # Validate percentages (0-95% to ensure minimum zone coverage)
        self.top_percent = max(0, min(95, top_percent))
        self.bottom_percent = max(0, min(95, bottom_percent))
        self.left_percent = max(0, min(95, left_percent))
        self.right_percent = max(0, min(95, right_percent))

        # Ensure total padding doesn't exceed 100%
        if self.top_percent + self.bottom_percent >= 100:
            self.top_percent = 40
            self.bottom_percent = 40
        if self.left_percent + self.right_percent >= 100:
            self.left_percent = 40
            self.right_percent = 40

        # Cache for zone boundaries
        self._zone_rect = None
        self._frame_size = None

    def calculate_zone(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """
        Calculate the detection zone rectangle.

        Args:
            frame_width: Width of the video frame
            frame_height: Height of the video frame

        Returns:
            Tuple of (x1, y1, x2, y2) defining the detection zone
        """
        # Calculate pixel offsets
        top_offset = int(frame_height * self.top_percent / 100)
        bottom_offset = int(frame_height * self.bottom_percent / 100)
        left_offset = int(frame_width * self.left_percent / 100)
        right_offset = int(frame_width * self.right_percent / 100)

        # Calculate zone boundaries
        x1 = left_offset
        y1 = top_offset
        x2 = frame_width - right_offset
        y2 = frame_height - bottom_offset

        # Ensure valid rectangle
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)

        # Cache the result
        self._zone_rect = (x1, y1, x2, y2)
        self._frame_size = (frame_width, frame_height)

        return self._zone_rect

    def is_detection_in_zone(self, bbox: List[float], frame_width: int, frame_height: int) -> bool:
        """
        Check if a detection bounding box is within the detection zone.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            frame_width: Width of the video frame
            frame_height: Height of the video frame

        Returns:
            True if the center of the bbox is within the detection zone
        """
        # Update zone if frame size changed
        if self._frame_size != (frame_width, frame_height):
            self.calculate_zone(frame_width, frame_height)

        if not self._zone_rect:
            return True  # No zone defined, accept all

        # Get detection center point
        det_center_x = (bbox[0] + bbox[2]) / 2
        det_center_y = (bbox[1] + bbox[3]) / 2

        # Check if center is within zone
        zone_x1, zone_y1, zone_x2, zone_y2 = self._zone_rect

        return (zone_x1 <= det_center_x <= zone_x2 and
                zone_y1 <= det_center_y <= zone_y2)

    def filter_detections(self, detections: List, frame_width: int, frame_height: int) -> List:
        """
        Filter detections to only include those within the detection zone.

        Args:
            detections: List of detection objects with bbox attribute
            frame_width: Width of the video frame
            frame_height: Height of the video frame

        Returns:
            Filtered list of detections within the zone
        """
        if not detections:
            return []

        # If no zone is configured (all percentages are 0), return all detections
        if all(p == 0 for p in [self.top_percent, self.bottom_percent,
                                self.left_percent, self.right_percent]):
            return detections

        # Filter detections
        filtered = []
        for det in detections:
            if self.is_detection_in_zone(det.bbox, frame_width, frame_height):
                filtered.append(det)

        return filtered

    def get_zone_info(self, frame_width: int, frame_height: int) -> Dict:
        """
        Get information about the detection zone.

        Returns:
            Dictionary with zone information
        """
        if self._frame_size != (frame_width, frame_height):
            self.calculate_zone(frame_width, frame_height)

        if not self._zone_rect:
            return {
                'enabled': False,
                'rect': None,
                'percentages': {
                    'top': 0, 'bottom': 0, 'left': 0, 'right': 0
                }
            }

        x1, y1, x2, y2 = self._zone_rect

        return {
            'enabled': True,
            'rect': self._zone_rect,
            'size': (x2 - x1, y2 - y1),
            'percentages': {
                'top': self.top_percent,
                'bottom': self.bottom_percent,
                'left': self.left_percent,
                'right': self.right_percent
            },
            'coverage': ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height) * 100
        }

    def is_enabled(self) -> bool:
        """Check if detection zone is enabled (any padding > 0)"""
        return any(p > 0 for p in [self.top_percent, self.bottom_percent,
                                   self.left_percent, self.right_percent])

    def update_zone_percentages(self, top: float, bottom: float, left: float, right: float, save: bool = True):
        """
        Update zone percentages and optionally save to config

        Args:
            top: Top percentage padding
            bottom: Bottom percentage padding
            left: Left percentage padding
            right: Right percentage padding
            save: Whether to save changes to config file
        """
        # Validate percentages (0-95% to ensure minimum zone coverage)
        self.top_percent = max(0, min(95, top))
        self.bottom_percent = max(0, min(95, bottom))
        self.left_percent = max(0, min(95, left))
        self.right_percent = max(0, min(95, right))

        # Ensure total padding doesn't exceed 100%
        if self.top_percent + self.bottom_percent >= 100:
            self.top_percent = 40
            self.bottom_percent = 40
        if self.left_percent + self.right_percent >= 100:
            self.left_percent = 40
            self.right_percent = 40

        # Clear cache
        self._zone_rect = None
        self._frame_size = None

        # Save to config file if requested
        if save and self.config_manager:
            zone_config = {
                'top': self.top_percent,
                'bottom': self.bottom_percent,
                'left': self.left_percent,
                'right': self.right_percent,
                'enabled': self.is_enabled()
            }
            self.config_manager.save_zone_config(zone_config)

    def update_zone_from_pixels(self, top_y: int, bottom_y: int, left_x: int, right_x: int,
                               frame_width: int, frame_height: int, save: bool = True):
        """
        Update zone from pixel coordinates

        Args:
            top_y: Top boundary in pixels from top of frame
            bottom_y: Bottom boundary in pixels from top of frame
            left_x: Left boundary in pixels from left of frame
            right_x: Right boundary in pixels from left of frame
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            save: Whether to save changes to config file
        """
        if self.config_manager:
            zone_config = self.config_manager.update_zone_from_pixels(
                top_y, bottom_y, left_x, right_x, frame_width, frame_height
            )
            # Update our percentages from the saved config
            self.top_percent = zone_config['top']
            self.bottom_percent = zone_config['bottom']
            self.left_percent = zone_config['left']
            self.right_percent = zone_config['right']
        else:
            # Calculate percentages manually
            top_percent = (top_y / frame_height) * 100
            bottom_percent = ((frame_height - bottom_y) / frame_height) * 100
            left_percent = (left_x / frame_width) * 100
            right_percent = ((frame_width - right_x) / frame_width) * 100

            self.update_zone_percentages(top_percent, bottom_percent, left_percent, right_percent, save)

        # Clear cache
        self._zone_rect = None
        self._frame_size = None

    def disable_zone(self, save: bool = True):
        """
        Disable detection zone (use full frame)

        Args:
            save: Whether to save changes to config file
        """
        self.update_zone_percentages(0, 0, 0, 0, save)

    def reset_to_defaults(self, save: bool = True):
        """
        Reset zone to default values

        Args:
            save: Whether to save changes to config file
        """
        if self.config_manager:
            default_config = self.config_manager.reset_to_defaults()
            self.top_percent = default_config['top']
            self.bottom_percent = default_config['bottom']
            self.left_percent = default_config['left']
            self.right_percent = default_config['right']
        else:
            self.update_zone_percentages(35, 5, 5, 5, save)

        # Clear cache
        self._zone_rect = None
        self._frame_size = None

    @classmethod
    def from_config(cls, config: Dict) -> 'DetectionZone':
        """
        Create DetectionZone from configuration dictionary.

        Args:
            config: Configuration dictionary with detection_zone settings

        Returns:
            DetectionZone instance
        """
        zone_config = config.get('detection', {}).get('detection_zone', {})

        return cls(
            top_percent=zone_config.get('top', 0),
            bottom_percent=zone_config.get('bottom', 0),
            left_percent=zone_config.get('left', 0),
            right_percent=zone_config.get('right', 0)
        )

    def get_optimal_crop_position(self, frame_width: int, frame_height: int) -> str:
        """
        Determine optimal crop position based on detection zone location.

        Args:
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels

        Returns:
            'left', 'center', or 'right' based on zone position
        """
        if not self.is_enabled():
            return 'full'  # Default to using full frame when zone is disabled

        # Calculate the zone boundaries
        zone_rect = self.calculate_zone(frame_width, frame_height)
        if not zone_rect:
            return 'center'

        left_x, top_y, right_x, bottom_y = zone_rect

        # Calculate zone center horizontal position as percentage of frame width
        zone_center_x = (left_x + right_x) / 2
        zone_center_percent = (zone_center_x / frame_width) * 100

        # Determine crop position based on where the zone center is
        # Left third: 0-33%, Center third: 33-67%, Right third: 67-100%
        if zone_center_percent < 33:
            return 'left'
        elif zone_center_percent > 67:
            return 'right'
        else:
            return 'center'