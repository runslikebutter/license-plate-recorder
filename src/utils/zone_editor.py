"""
Interactive detection zone editor with improved mouse handling
Handles mouse interactions for adjusting detection zone boundaries
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Callable
from .logger import setup_logger


class ZoneEditor:
    """Interactive detection zone editor with mouse controls"""

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.logger = setup_logger(self.__class__.__name__)

        # Zone boundaries (in pixels)
        self.top_y = 0
        self.bottom_y = frame_height
        self.left_x = 0
        self.right_x = frame_width

        # Mouse state
        self.dragging = False
        self.drag_handle = None
        self.drag_start_pos = (0, 0)
        self.drag_start_rect = None
        self.hover_handle = None

        # Handle properties
        self.corner_handle_size = 15  # Larger corner handles
        self.edge_handle_size = 10   # Smaller edge handles
        self.edge_handle_length = 30 # Length of edge handles

        # Editing mode
        self.editing_enabled = False

        # Callback for zone updates
        self.on_zone_changed: Optional[Callable] = None

        # Minimum zone size
        self.min_zone_size = 100

    def set_zone_from_percentages(self, top_percent: float, bottom_percent: float,
                                 left_percent: float, right_percent: float):
        """Set zone boundaries from percentage values"""
        self.top_y = int((top_percent / 100) * self.frame_height)
        self.bottom_y = int(self.frame_height - (bottom_percent / 100) * self.frame_height)
        self.left_x = int((left_percent / 100) * self.frame_width)
        self.right_x = int(self.frame_width - (right_percent / 100) * self.frame_width)

        self._validate_boundaries()

    def get_zone_percentages(self) -> Tuple[float, float, float, float]:
        """Get current zone as percentages (top, bottom, left, right)"""
        top_percent = (self.top_y / self.frame_height) * 100
        bottom_percent = ((self.frame_height - self.bottom_y) / self.frame_height) * 100
        left_percent = (self.left_x / self.frame_width) * 100
        right_percent = ((self.frame_width - self.right_x) / self.frame_width) * 100

        return top_percent, bottom_percent, left_percent, right_percent

    def get_zone_rect(self) -> Tuple[int, int, int, int]:
        """Get zone as rectangle (x1, y1, x2, y2)"""
        return self.left_x, self.top_y, self.right_x, self.bottom_y

    def set_editing_mode(self, enabled: bool):
        """Enable or disable editing mode"""
        self.editing_enabled = enabled
        if not enabled:
            self.dragging = False
            self.drag_handle = None
            self.hover_handle = None

    def _validate_boundaries(self):
        """Ensure zone boundaries are within frame and properly ordered"""
        # Ensure minimum zone size
        zone_width = self.right_x - self.left_x
        zone_height = self.bottom_y - self.top_y

        if zone_width < self.min_zone_size:
            center_x = (self.left_x + self.right_x) / 2
            self.left_x = int(center_x - self.min_zone_size / 2)
            self.right_x = int(center_x + self.min_zone_size / 2)

        if zone_height < self.min_zone_size:
            center_y = (self.top_y + self.bottom_y) / 2
            self.top_y = int(center_y - self.min_zone_size / 2)
            self.bottom_y = int(center_y + self.min_zone_size / 2)

        # Clamp to frame boundaries
        self.top_y = max(0, min(self.frame_height - self.min_zone_size, self.top_y))
        self.bottom_y = max(self.top_y + self.min_zone_size, min(self.frame_height, self.bottom_y))
        self.left_x = max(0, min(self.frame_width - self.min_zone_size, self.left_x))
        self.right_x = max(self.left_x + self.min_zone_size, min(self.frame_width, self.right_x))

    def _get_handle_at_position(self, x: int, y: int) -> Optional[str]:
        """Determine which handle is at the given position"""
        if not self.editing_enabled:
            return None

        # Check corner handles first (higher priority)
        corners = [
            ('top-left', self.left_x, self.top_y),
            ('top-right', self.right_x, self.top_y),
            ('bottom-left', self.left_x, self.bottom_y),
            ('bottom-right', self.right_x, self.bottom_y)
        ]

        for handle_name, handle_x, handle_y in corners:
            if (abs(x - handle_x) <= self.corner_handle_size and
                abs(y - handle_y) <= self.corner_handle_size):
                return handle_name

        # Check edge handles
        # Top edge
        if (abs(y - self.top_y) <= self.edge_handle_size and
            self.left_x + self.edge_handle_length <= x <= self.right_x - self.edge_handle_length):
            return 'top'

        # Bottom edge
        if (abs(y - self.bottom_y) <= self.edge_handle_size and
            self.left_x + self.edge_handle_length <= x <= self.right_x - self.edge_handle_length):
            return 'bottom'

        # Left edge
        if (abs(x - self.left_x) <= self.edge_handle_size and
            self.top_y + self.edge_handle_length <= y <= self.bottom_y - self.edge_handle_length):
            return 'left'

        # Right edge
        if (abs(x - self.right_x) <= self.edge_handle_size and
            self.top_y + self.edge_handle_length <= y <= self.bottom_y - self.edge_handle_length):
            return 'right'

        # Check if inside zone for move operation
        if (self.left_x + self.corner_handle_size < x < self.right_x - self.corner_handle_size and
            self.top_y + self.corner_handle_size < y < self.bottom_y - self.corner_handle_size):
            return 'move'

        return None

    def handle_mouse_event(self, event: int, x: int, y: int, flags: int):
        """Handle OpenCV mouse events"""
        if not self.editing_enabled:
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # Update hover state
            if not self.dragging:
                self.hover_handle = self._get_handle_at_position(x, y)

            # Handle dragging
            if self.dragging and self.drag_handle and self.drag_start_rect:
                dx = x - self.drag_start_pos[0]
                dy = y - self.drag_start_pos[1]

                # Apply the drag operation based on handle type
                self._apply_drag_operation(dx, dy)

                # Notify about zone change
                if self.on_zone_changed:
                    self.on_zone_changed()

        elif event == cv2.EVENT_LBUTTONDOWN:
            handle = self._get_handle_at_position(x, y)
            if handle:
                self.dragging = True
                self.drag_handle = handle
                self.drag_start_pos = (x, y)
                self.drag_start_rect = (self.left_x, self.top_y, self.right_x, self.bottom_y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_handle = None
            self.drag_start_pos = (0, 0)
            self.drag_start_rect = None

    def _apply_drag_operation(self, dx: int, dy: int):
        """Apply drag operation based on current handle"""
        if not self.drag_start_rect:
            return

        start_left, start_top, start_right, start_bottom = self.drag_start_rect

        if self.drag_handle == 'top-left':
            self.left_x = start_left + dx
            self.top_y = start_top + dy
        elif self.drag_handle == 'top-right':
            self.right_x = start_right + dx
            self.top_y = start_top + dy
        elif self.drag_handle == 'bottom-left':
            self.left_x = start_left + dx
            self.bottom_y = start_bottom + dy
        elif self.drag_handle == 'bottom-right':
            self.right_x = start_right + dx
            self.bottom_y = start_bottom + dy
        elif self.drag_handle == 'top':
            self.top_y = start_top + dy
        elif self.drag_handle == 'bottom':
            self.bottom_y = start_bottom + dy
        elif self.drag_handle == 'left':
            self.left_x = start_left + dx
        elif self.drag_handle == 'right':
            self.right_x = start_right + dx
        elif self.drag_handle == 'move':
            # Move entire zone
            zone_width = start_right - start_left
            zone_height = start_bottom - start_top

            new_left = start_left + dx
            new_top = start_top + dy

            # Constrain to frame boundaries
            new_left = max(0, min(self.frame_width - zone_width, new_left))
            new_top = max(0, min(self.frame_height - zone_height, new_top))

            self.left_x = new_left
            self.top_y = new_top
            self.right_x = new_left + zone_width
            self.bottom_y = new_top + zone_height

        # Validate boundaries after any change
        self._validate_boundaries()

    def draw_edit_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw editing overlay on frame"""
        if not self.editing_enabled:
            return frame

        # Colors
        zone_color = (0, 255, 255)      # Yellow for zone border
        handle_color = (255, 255, 0)    # Cyan for handles
        hover_color = (0, 255, 0)       # Green for hovered handle
        active_color = (0, 0, 255)      # Red for active (dragging) handle

        # Draw zone rectangle
        cv2.rectangle(frame, (self.left_x, self.top_y),
                     (self.right_x, self.bottom_y), zone_color, 2)

        # Draw corner handles
        corners = [
            ('top-left', self.left_x, self.top_y),
            ('top-right', self.right_x, self.top_y),
            ('bottom-left', self.left_x, self.bottom_y),
            ('bottom-right', self.right_x, self.bottom_y)
        ]

        for handle_name, handle_x, handle_y in corners:
            color = handle_color
            if self.drag_handle == handle_name:
                color = active_color
            elif self.hover_handle == handle_name:
                color = hover_color

            cv2.rectangle(frame,
                         (handle_x - self.corner_handle_size, handle_y - self.corner_handle_size),
                         (handle_x + self.corner_handle_size, handle_y + self.corner_handle_size),
                         color, -1)
            # Draw border
            cv2.rectangle(frame,
                         (handle_x - self.corner_handle_size, handle_y - self.corner_handle_size),
                         (handle_x + self.corner_handle_size, handle_y + self.corner_handle_size),
                         (0, 0, 0), 2)

        # Draw edge handles
        edges = [
            ('top', (self.left_x + self.right_x) // 2, self.top_y),
            ('bottom', (self.left_x + self.right_x) // 2, self.bottom_y),
            ('left', self.left_x, (self.top_y + self.bottom_y) // 2),
            ('right', self.right_x, (self.top_y + self.bottom_y) // 2)
        ]

        for handle_name, handle_x, handle_y in edges:
            color = handle_color
            if self.drag_handle == handle_name:
                color = active_color
            elif self.hover_handle == handle_name:
                color = hover_color

            cv2.circle(frame, (handle_x, handle_y), self.edge_handle_size, color, -1)
            cv2.circle(frame, (handle_x, handle_y), self.edge_handle_size, (0, 0, 0), 2)

        # Draw move handle (center of zone)
        center_x = (self.left_x + self.right_x) // 2
        center_y = (self.top_y + self.bottom_y) // 2

        move_color = handle_color
        if self.drag_handle == 'move':
            move_color = active_color
        elif self.hover_handle == 'move':
            move_color = hover_color

        # Draw cross for move handle
        cross_size = 8
        cv2.line(frame, (center_x - cross_size, center_y), (center_x + cross_size, center_y), move_color, 3)
        cv2.line(frame, (center_x, center_y - cross_size), (center_x, center_y + cross_size), move_color, 3)

        # Draw editing instructions
        instructions = [
            "Zone Edit Mode - Active",
            "Drag corners/edges to resize zone",
            "Drag center cross to move zone",
            "Press 'E' to save and exit"
        ]

        y_offset = 30
        for i, instruction in enumerate(instructions):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            thickness = 2 if i == 0 else 1
            cv2.putText(frame, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            y_offset += 25

        # Show current percentages and handle info
        top_p, bottom_p, left_p, right_p = self.get_zone_percentages()
        percentage_text = f"Zone: T:{top_p:.1f}% B:{bottom_p:.1f}% L:{left_p:.1f}% R:{right_p:.1f}%"

        # Add handle info if hovering or dragging
        handle_info = ""
        if self.drag_handle:
            handle_info = f" | Dragging: {self.drag_handle}"
        elif self.hover_handle:
            handle_info = f" | Hover: {self.hover_handle}"

        # Add boundary warnings for user guidance
        boundary_warnings = []
        if self.right_x >= self.frame_width:
            boundary_warnings.append("At right edge")
        if self.left_x <= 0:
            boundary_warnings.append("At left edge")

        warning_text = " | " + " | ".join(boundary_warnings) if boundary_warnings else ""

        status_text = percentage_text + handle_info + warning_text
        cv2.putText(frame, status_text, (10, y_offset + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame