import cv2
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime


class Visualizer:
    def __init__(self):
        # Color scheme
        self.colors = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'blue': (255, 0, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'gray': (128, 128, 128),
            'orange': (0, 165, 255),
            'purple': (128, 0, 128),
            'cyan': (255, 255, 0)
        }

        # Font settings - increased for better visibility
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.9  # Increased from 0.6
        self.font_scale_small = 0.7  # For secondary text
        self.thickness = 2
        self.thickness_bold = 3  # For important text

    def draw_detections(self, frame: np.ndarray, detections: List,
                       color: Tuple[int, int, int] = None) -> np.ndarray:
        """Draw bounding boxes and labels for license plate detections"""
        if color is None:
            color = self.colors['green']

        for det in detections:
            # Draw bounding box
            x1, y1, x2, y2 = [int(x) for x in det.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label text with OCR results
            if hasattr(det, 'ocr_text') and det.ocr_text and det.ocr_text != 'Unknown':
                if hasattr(det, 'ocr_confidence') and det.ocr_confidence:
                    label = f"Plate: {det.ocr_text} ({det.ocr_confidence:.2f})"
                else:
                    label = f"Plate: {det.ocr_text}"
            else:
                label = f"License Plate: {det.confidence:.2f}"

            # Add track ID if available
            if hasattr(det, 'track_id') and det.track_id is not None:
                label = f"ID:{det.track_id} {label}"

            label_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)

            # Background for label
            cv2.rectangle(frame,
                         (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1),
                         color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 2),
                       self.font, self.font_scale, self.colors['white'], self.thickness)

        return frame

    def draw_status_bar(self, frame: np.ndarray, state: str,
                       frame_count: int, detection_count: int,
                       recordings_saved: int, is_recording: bool) -> np.ndarray:
        """Draw status bar at the top of the frame"""
        height, width = frame.shape[:2]
        bar_height = 60  # Increased from 40

        # Create semi-transparent overlay for status bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_height), self.colors['black'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Recording indicator - larger and more visible
        if is_recording:
            # Larger red recording dot
            cv2.circle(frame, (30, 30), 12, self.colors['red'], -1)
            cv2.putText(frame, "REC", (50, 38),
                       self.font, 0.8, self.colors['red'], self.thickness_bold)

        # State indicator with color coding
        state_colors = {
            'idle': self.colors['gray'],
            'recording': self.colors['red'],
            'post_recording': self.colors['yellow']
        }
        state_color = state_colors.get(state, self.colors['white'])
        state_text = f"State: {state.upper()}"
        cv2.putText(frame, state_text, (150, 38),
                   self.font, 0.8, state_color, self.thickness)

        # Frame counter
        frame_text = f"Frame: {frame_count}"
        cv2.putText(frame, frame_text, (350, 38),
                   self.font, 0.8, self.colors['white'], self.thickness)

        # Detection counter
        det_text = f"Plates: {detection_count}"
        cv2.putText(frame, det_text, (530, 38),
                   self.font, 0.8, self.colors['green'], self.thickness)

        # Recordings saved
        rec_text = f"Saved: {recordings_saved}"
        cv2.putText(frame, rec_text, (720, 38),
                   self.font, 0.8, self.colors['blue'], self.thickness)

        # Timestamp - larger
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (width - 150, 38),
                   self.font, 0.8, self.colors['white'], self.thickness)

        return frame

    def draw_state_border(self, frame: np.ndarray, state: str,
                         is_recording: bool) -> np.ndarray:
        """Draw colored border around frame based on state"""
        height, width = frame.shape[:2]
        border_thickness = 5

        if is_recording:
            color = self.colors['red']
        elif state == 'post_recording':
            color = self.colors['yellow']
        elif state == 'idle':
            color = self.colors['gray']
        else:
            color = self.colors['green']

        # Draw border
        cv2.rectangle(frame, (0, 0), (width-1, height-1), color, border_thickness)

        return frame

    def draw_info_panel(self, frame: np.ndarray, info_dict: dict,
                       position: str = 'bottom_left') -> np.ndarray:
        """Draw information panel with key-value pairs"""
        height, width = frame.shape[:2]

        # Panel settings - increased sizes
        panel_width = 350  # Increased from 250
        line_height = 35   # Increased from 25
        panel_height = len(info_dict) * line_height + 30  # Increased padding
        padding = 15  # Increased from 10

        # Determine position
        if position == 'bottom_left':
            x1, y1 = padding, height - panel_height - padding
        elif position == 'bottom_right':
            x1, y1 = width - panel_width - padding, height - panel_height - padding
        elif position == 'top_left':
            x1, y1 = padding, 50  # Below status bar
        else:  # top_right
            x1, y1 = width - panel_width - padding, 50

        x2, y2 = x1 + panel_width, y1 + panel_height

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.colors['black'], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Draw border
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['white'], 1)

        # Draw info text - larger font
        y_offset = y1 + 25
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (x1 + 10, y_offset),
                       self.font, 0.8, self.colors['white'], self.thickness)
            y_offset += line_height

        return frame

    def draw_controls_help(self, frame: np.ndarray) -> np.ndarray:
        """Draw keyboard controls help"""
        height, width = frame.shape[:2]

        controls = [
            "Controls:",
            "Q - Quit",
            "S - Screenshot",
            "D - Toggle Debug",
            "Space - Pause",
            "H - Hide Help",
            "E - Edit Zone",
            "Z - Toggle Zone",
            "T - Track Settings"
        ]

        # Position at bottom right - adjusted for larger text
        x = width - 250  # Increased from 150
        y_start = height - len(controls) * 30 - 30  # Increased spacing

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y_start - 5),
                     (width - 10, height - 10),
                     self.colors['black'], -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        # Draw control text - larger font
        for i, control in enumerate(controls):
            y = y_start + i * 30  # Increased spacing
            color = self.colors['yellow'] if i == 0 else self.colors['white']
            thickness = self.thickness if i == 0 else 1
            cv2.putText(frame, control, (x, y),
                       self.font, self.font_scale_small, color, thickness)

        return frame

    def draw_detection_zone(self, frame: np.ndarray, zone_rect: Optional[Tuple[int, int, int, int]],
                           zone_info: Optional[dict] = None) -> np.ndarray:
        """Draw detection zone boundary"""
        if not zone_rect:
            return frame

        x1, y1, x2, y2 = zone_rect
        height, width = frame.shape[:2]

        # Create overlay for areas outside detection zone
        overlay = frame.copy()

        # Darken areas outside the zone
        # Top area
        if y1 > 0:
            cv2.rectangle(overlay, (0, 0), (width, y1), self.colors['black'], -1)
        # Bottom area
        if y2 < height:
            cv2.rectangle(overlay, (0, y2), (width, height), self.colors['black'], -1)
        # Left area
        if x1 > 0:
            cv2.rectangle(overlay, (0, y1), (x1, y2), self.colors['black'], -1)
        # Right area
        if x2 < width:
            cv2.rectangle(overlay, (x2, y1), (width, y2), self.colors['black'], -1)

        # Apply overlay with transparency
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Draw zone border
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['yellow'], 2)

        # Draw corner markers
        corner_length = 20
        corner_thickness = 3

        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), self.colors['yellow'], corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), self.colors['yellow'], corner_thickness)

        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), self.colors['yellow'], corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), self.colors['yellow'], corner_thickness)

        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), self.colors['yellow'], corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), self.colors['yellow'], corner_thickness)

        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), self.colors['yellow'], corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), self.colors['yellow'], corner_thickness)

        # Add zone label
        if zone_info and zone_info.get('enabled'):
            coverage = zone_info.get('coverage', 0)

            label = f"Detection Zone ({coverage:.0f}% coverage)"
            label_size, _ = cv2.getTextSize(label, self.font, self.font_scale_small, self.thickness)

            # Position label at top of zone
            label_x = x1 + 10
            label_y = y1 - 10 if y1 > 30 else y1 + 30

            # Background for label
            cv2.rectangle(frame,
                         (label_x - 5, label_y - label_size[1] - 5),
                         (label_x + label_size[0] + 5, label_y + 5),
                         self.colors['black'], -1)

            # Draw label
            cv2.putText(frame, label, (label_x, label_y),
                       self.font, self.font_scale_small, self.colors['yellow'], self.thickness)

        return frame

    def draw_crop_area(self, frame: np.ndarray, crop_info: dict) -> np.ndarray:
        """Draw the detection crop area visualization"""
        if not crop_info:
            return frame

        crop_position = crop_info.get('crop_position', 'center')
        mode = crop_info.get('mode', 'crop')

        if mode == 'pad':
            height, width = crop_info.get('original_shape', frame.shape[:2])
            x1, y1 = 0, 0
            x2, y2 = width, height
        else:
            crop_size = crop_info.get('crop_size', 0)
            if crop_size <= 0:
                return frame
            start_x = crop_info.get('start_x', 0)
            start_y = crop_info.get('start_y', 0)
            x1, y1 = start_x, start_y
            x2, y2 = start_x + crop_size, start_y + crop_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['blue'], 2)

        # Add crop area label
        crop_label = f"Detection Area ({crop_position})"
        label_y = y1 - 10 if y1 > 20 else y2 + 25
        cv2.putText(frame, crop_label, (x1, label_y),
                    self.font, 0.6, self.colors['blue'], 2)

        return frame

    def draw_buffer_indicator(self, frame: np.ndarray, buffer_size: int,
                             max_buffer_size: int) -> np.ndarray:
        """Draw circular buffer fill indicator"""
        height, width = frame.shape[:2]

        # Position and size - increased for visibility
        center_x = width - 80  # Adjusted position
        center_y = 140  # Moved down due to larger status bar
        radius = 45  # Increased from 30

        # Calculate fill percentage
        fill_percent = min(buffer_size / max_buffer_size, 1.0) if max_buffer_size > 0 else 0

        # Draw background circle
        cv2.circle(frame, (center_x, center_y), radius, self.colors['gray'], 2)

        # Draw filled arc
        if fill_percent > 0:
            angle = int(360 * fill_percent)
            cv2.ellipse(frame, (center_x, center_y), (radius, radius),
                       -90, 0, angle, self.colors['green'], -1)

        # Draw center circle
        cv2.circle(frame, (center_x, center_y), radius - 8, self.colors['black'], -1)

        # Draw text - larger
        text = f"{buffer_size}"
        text_size, _ = cv2.getTextSize(text, self.font, self.font_scale_small, self.thickness)
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        cv2.putText(frame, text, (text_x, text_y),
                   self.font, self.font_scale_small, self.colors['white'], self.thickness)

        # Label - larger
        cv2.putText(frame, "Buffer", (center_x - 35, center_y + radius + 20),
                   self.font, self.font_scale_small, self.colors['white'], self.thickness)

        return frame

    def draw_plate_tracks_info(self, frame: np.ndarray, tracks_info: List[dict]) -> np.ndarray:
        """Draw information about active license plate tracks"""
        if not tracks_info:
            return frame

        height, width = frame.shape[:2]

        # Panel settings
        panel_width = 400
        line_height = 25
        panel_height = min(len(tracks_info) * line_height + 40, height // 2)

        # Position at top right
        x1 = width - panel_width - 10
        y1 = 70  # Below status bar
        x2 = x1 + panel_width
        y2 = y1 + panel_height

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.colors['black'], -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        # Draw border
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['cyan'], 1)

        # Title
        cv2.putText(frame, "Active Plate Tracks", (x1 + 10, y1 + 25),
                   self.font, 0.7, self.colors['cyan'], 2)

        # Track info
        y_offset = y1 + 50
        for i, track in enumerate(tracks_info[:8]):  # Show max 8 tracks
            track_id = track.get('track_id', 'Unknown')
            plate_text = track.get('best_plate', 'Unknown')
            confidence = track.get('best_confidence', 0.0)
            detections = track.get('detection_count', 0)

            # Truncate long plate text
            if len(plate_text) > 12:
                plate_text = plate_text[:12] + "..."

            track_text = f"ID{track_id}: {plate_text} ({confidence:.2f}) x{detections}"

            color = self.colors['green'] if confidence > 0.5 else self.colors['yellow']
            cv2.putText(frame, track_text, (x1 + 10, y_offset),
                       self.font, 0.5, color, 1)
            y_offset += line_height

            if y_offset >= y2 - 10:
                break

        return frame

    def create_preview_frame(self, frame: np.ndarray, state: str,
                           detections: List, frame_count: int,
                           detection_count: int, recordings_saved: int,
                           is_recording: bool, buffer_size: int,
                           max_buffer_size: int, fps: float,
                           zone_rect: Optional[Tuple[int, int, int, int]] = None,
                           zone_info: Optional[dict] = None,
                           crop_info: Optional[dict] = None,
                           tracks_info: Optional[List[dict]] = None,
                           show_debug: bool = False,
                           show_help: bool = True) -> np.ndarray:
        """Create complete preview frame with all visualizations"""
        # Make a copy to avoid modifying original
        preview = frame.copy()

        # Draw detection zone first (so it's in background)
        if zone_rect:
            zone_info_with_debug = zone_info.copy() if zone_info else {}
            preview = self.draw_detection_zone(preview, zone_rect, zone_info_with_debug)

        # Draw crop area if available
        if crop_info:
            preview = self.draw_crop_area(preview, crop_info)

        # Draw detections
        if detections:
            preview = self.draw_detections(preview, detections)

        # Draw status bar
        preview = self.draw_status_bar(preview, state, frame_count,
                                      detection_count, recordings_saved,
                                      is_recording)

        # Draw state border
        preview = self.draw_state_border(preview, state, is_recording)

        # Draw buffer indicator
        preview = self.draw_buffer_indicator(preview, buffer_size, max_buffer_size)

        # Draw active tracks info
        if tracks_info:
            preview = self.draw_plate_tracks_info(preview, tracks_info)

        # Draw debug info if enabled
        if show_debug:
            debug_info = {
                'FPS': f"{fps:.1f}",
                'Resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                'Buffer': f"{buffer_size}/{max_buffer_size}",
                'State': state,
                'Tracks': str(len(tracks_info)) if tracks_info else "0"
            }
            if zone_info and zone_info.get('enabled'):
                debug_info['Zone'] = f"{zone_info.get('coverage', 0):.0f}%"
            preview = self.draw_info_panel(preview, debug_info, 'bottom_left')

        # Draw controls help if enabled
        if show_help:
            preview = self.draw_controls_help(preview)

        return preview