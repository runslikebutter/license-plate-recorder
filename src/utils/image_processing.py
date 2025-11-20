"""
Image processing utilities for license plate detection
Extracted and adapted from zalpr.py for multi-resolution processing
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Any
from .logger import setup_logger

logger = setup_logger(__name__)


def extract_center_square_crop(frame: np.ndarray, crop_position: str = 'center') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract square from frame at full resolution

    Args:
        frame: Input frame
        crop_position: 'left', 'center', 'right', or 'full' - crop alignment strategy

    Returns:
        square_crop, crop_info dict
    """
    h, w = frame.shape[:2]

    square_mode = 'crop'
    pad_x = 0
    pad_y = 0

    if crop_position == 'full':
        # Letterbox the entire frame into a square canvas to avoid losing content
        crop_size = max(h, w)
        square_crop = np.zeros((crop_size, crop_size, 3), dtype=frame.dtype)
        pad_y = (crop_size - h) // 2
        pad_x = (crop_size - w) // 2
        square_crop[pad_y:pad_y + h, pad_x:pad_x + w] = frame
        start_x = 0
        start_y = 0
        square_mode = 'pad'
    else:
        # Calculate crop size (square) limited by the shorter frame dimension
        crop_size = min(h, w)

        # Vertical position (centered)
        start_y = (h - crop_size) // 2

        # Horizontal position based on crop_position
        if crop_position == 'left':
            start_x = 0
        elif crop_position == 'right':
            start_x = w - crop_size
        else:  # 'center' or any other value defaults to center
            start_x = (w - crop_size) // 2

        square_crop = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

    crop_info = {
        'start_x': start_x,
        'start_y': start_y,
        'crop_size': crop_size,
        'original_shape': (h, w),
        'crop_position': crop_position,
        'mode': square_mode,
        'pad_x': pad_x,
        'pad_y': pad_y,
    }

    return square_crop, crop_info


def process_frame_for_detection(square_crop: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float]:
    """
    Resize square crop to target size for detection

    Returns:
        detection_frame, scale_factor
    """
    detection_frame = cv2.resize(square_crop, (target_size, target_size),
                                interpolation=cv2.INTER_AREA)
    scale_factor = square_crop.shape[0] / target_size

    return detection_frame, scale_factor


def extract_plate_crop(square_crop: np.ndarray, bbox: List[float],
                      target_height: int = 64, target_width: int = 128) -> np.ndarray:
    """
    Extract and resize license plate crop from full-resolution square crop

    Args:
        square_crop: Full resolution square crop
        bbox: Detection bounding box [x1, y1, x2, y2] in square coordinates
        target_height: Target height for OCR crop (64)
        target_width: Target width for OCR crop (128)

    Returns:
        Resized plate crop for OCR processing
    """
    x1, y1, x2, y2 = map(int, bbox)
    height, width = square_crop.shape[:2]

    # Add padding around the detection (10%)
    pad_width = int((x2 - x1) * 0.1)
    pad_height = int((y2 - y1) * 0.1)

    # Expand bbox with padding and clamp to image bounds
    x1 = max(0, x1 - pad_width)
    y1 = max(0, y1 - pad_height)
    x2 = min(width, x2 + pad_width)
    y2 = min(height, y2 + pad_height)

    # Extract plate region
    plate_crop = square_crop[y1:y2, x1:x2]

    if plate_crop.size == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Resize to target OCR dimensions
    return cv2.resize(plate_crop, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def scale_detections_from_640_to_square(alpr_results: List[Any], scale_factor: float) -> Tuple[np.ndarray, List[Any]]:
    """
    Convert ALPR detection results from 640x640 coordinates to full-res square coordinates

    Returns:
        detections array (x1, y1, x2, y2, score) and OCR results
    """
    if not alpr_results:
        return np.empty((0, 5)), []

    detections = []
    ocr_results = []

    for result in alpr_results:
        try:
            bbox = result.detection.bounding_box
            confidence = result.detection.confidence

            # Scale from 640x640 back to full-res square coordinates
            square_x1 = bbox.x1 * scale_factor
            square_y1 = bbox.y1 * scale_factor
            square_x2 = bbox.x2 * scale_factor
            square_y2 = bbox.y2 * scale_factor

            detections.append([square_x1, square_y1, square_x2, square_y2, confidence])
            ocr_results.append(result.ocr)

        except Exception as e:
            logger.error(f"Error processing detection: {e}")
            continue

    return np.array(detections) if detections else np.empty((0, 5)), ocr_results


def transform_bbox_to_original(bbox: List[float], crop_info: Dict[str, Any]) -> List[float]:
    """Transform bbox from square coordinates to original frame coordinates"""
    x1, y1, x2, y2 = bbox
    height, width = crop_info.get('original_shape', (0, 0))
    mode = crop_info.get('mode', 'crop')

    if mode == 'pad':
        pad_x = crop_info.get('pad_x', 0)
        pad_y = crop_info.get('pad_y', 0)
        orig_x1 = x1 - pad_x
        orig_y1 = y1 - pad_y
        orig_x2 = x2 - pad_x
        orig_y2 = y2 - pad_y
    else:
        start_x = crop_info.get('start_x', 0)
        start_y = crop_info.get('start_y', 0)
        orig_x1 = x1 + start_x
        orig_y1 = y1 + start_y
        orig_x2 = x2 + start_x
        orig_y2 = y2 + start_y

    # Clamp to original frame bounds
    orig_x1 = max(0, min(width, orig_x1))
    orig_y1 = max(0, min(height, orig_y1))
    orig_x2 = max(0, min(width, orig_x2))
    orig_y2 = max(0, min(height, orig_y2))

    return [orig_x1, orig_y1, orig_x2, orig_y2]


def save_debug_crop(crop: np.ndarray, frame_num: int, track_id: int,
                   ocr_confidence: float, output_dir: str) -> None:
    """
    Save crop for debugging with OCR confidence in filename

    Args:
        crop: Plate crop image
        frame_num: Frame number
        track_id: Track ID
        ocr_confidence: OCR confidence score
        output_dir: Output directory path
    """
    if output_dir is None:
        return

    try:
        import os
        os.makedirs(output_dir, exist_ok=True)
        filename = f"frame_{frame_num:06d}_track_{track_id}_{ocr_confidence:.3f}.jpg"
        file_path = os.path.join(output_dir, filename)
        cv2.imwrite(file_path, crop)
        logger.debug(f"Saved debug crop: {filename}")
    except Exception as e:
        logger.warning(f"Failed to save crop: {e}")


def validate_bbox(bbox: List[float], frame_width: int, frame_height: int) -> bool:
    """
    Validate that a bounding box is within frame boundaries and has valid dimensions

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        frame_width: Frame width
        frame_height: Frame height

    Returns:
        True if bbox is valid
    """
    x1, y1, x2, y2 = bbox

    # Check bounds
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        return False

    # Check dimensions
    if x2 <= x1 or y2 <= y1:
        return False

    # Check minimum size (at least 10x10 pixels)
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return False

    return True


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def draw_detection_box(frame: np.ndarray, bbox: List[float], label: str,
                      confidence: float, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw detection box with label on frame

    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        label: Text label to display
        confidence: Detection confidence
        color: Box color in BGR format

    Returns:
        Frame with detection box drawn
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Prepare label text
    label_text = f"{label}: {confidence:.2f}"

    # Get text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

    # Draw label background
    cv2.rectangle(frame, (x1, y1 - text_height - baseline - 4),
                 (x1 + text_width, y1), color, -1)

    # Draw label text
    cv2.putText(frame, label_text, (x1, y1 - baseline - 2),
               font, font_scale, (255, 255, 255), thickness)

    return frame