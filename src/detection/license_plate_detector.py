"""
License plate detector using Fast-ALPR
Integrates the proven detection and OCR pipeline from zalpr.py
"""

import time
import numpy as np
from typing import List, Optional
from fast_alpr import ALPR

from .detector import Detector, Detection
from ..utils.logger import setup_logger
from ..utils.image_processing import (
    extract_center_square_crop,
    process_frame_for_detection,
    scale_detections_from_640_to_square,
    transform_bbox_to_original,
    extract_plate_crop,
    validate_bbox
)


class LicensePlateDetector(Detector):
    """License plate detector using Fast-ALPR with multi-resolution processing"""

    def __init__(self,
                 detector_model: str = "yolo-v9-t-640-license-plate-end2end",
                 ocr_model: str = "cct-s-v1-global-model",
                 confidence_threshold: float = 0.3,
                 crop_position: str = "center",
                 detection_input_size: int = 640,
                 ocr_crop_size: tuple = (128, 64),
                 enable_ocr: bool = True):
        """
        Initialize the license plate detector

        Args:
            detector_model: Fast-ALPR detector model name
            ocr_model: Fast-ALPR OCR model name
            confidence_threshold: Minimum confidence threshold for detections
            crop_position: Position for square crop ('left', 'center', 'right')
            detection_input_size: Input size for detection model (typically 640)
            ocr_crop_size: Size for OCR crops (width, height)
            enable_ocr: Whether to run OCR on detected plates
        """
        self.detector_model = detector_model
        self.ocr_model = ocr_model
        self.confidence_threshold = confidence_threshold
        self.crop_position = crop_position
        self.detection_input_size = detection_input_size
        self.ocr_crop_size = ocr_crop_size
        self.enable_ocr = enable_ocr

        self.logger = setup_logger(self.__class__.__name__)

        # Performance tracking
        self.inference_times = []
        self.avg_inference_time = 0.0
        self.total_detections = 0
        self.frames_processed = 0

        # Initialize Fast-ALPR
        self.alpr = None
        self._initialize_alpr()

    def _initialize_alpr(self):
        """Initialize the Fast-ALPR detector"""
        try:
            self.logger.info(f"Initializing Fast-ALPR detector...")
            self.logger.info(f"Detector model: {self.detector_model}")
            self.logger.info(f"OCR model: {self.ocr_model}")

            self.alpr = ALPR(
                detector_model=self.detector_model,
                ocr_model=self.ocr_model if self.enable_ocr else None
            )

            self.logger.info("Fast-ALPR initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Fast-ALPR: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect license plates in a frame using multi-resolution processing

        Args:
            frame: Input image frame

        Returns:
            List of Detection objects with license plate information
        """
        if self.alpr is None:
            self.logger.error("ALPR detector not initialized")
            return []

        detections = []
        start_time = time.time()

        try:
            self.frames_processed += 1

            # Step 1: Extract center square crop from full resolution frame
            square_crop, crop_info = extract_center_square_crop(frame, self.crop_position)

            # Step 2: Resize to detection input size (640x640)
            detection_frame, scale_factor = process_frame_for_detection(
                square_crop, self.detection_input_size
            )

            # Step 3: Run Fast-ALPR detection
            alpr_results = self.alpr.predict(detection_frame)

            # Step 4: Scale detection coordinates back to full-res square
            plate_detections_square, ocr_results = scale_detections_from_640_to_square(
                alpr_results, scale_factor
            )

            # Step 5: Process each detection
            if len(plate_detections_square) > 0:
                for idx, detection_data in enumerate(plate_detections_square):
                    try:
                        # Extract detection info
                        square_bbox = detection_data[:4].tolist()  # [x1, y1, x2, y2] in square coords
                        confidence = float(detection_data[4])

                        # Filter by confidence threshold
                        if confidence < self.confidence_threshold:
                            continue

                        # Validate bounding box
                        if not validate_bbox(square_bbox, square_crop.shape[1], square_crop.shape[0]):
                            self.logger.debug(f"Invalid bbox detected: {square_bbox}")
                            continue

                        # Transform to original frame coordinates
                        original_bbox = transform_bbox_to_original(square_bbox, crop_info)

                        # Get OCR result if available
                        ocr_text = None
                        ocr_confidence = None
                        if self.enable_ocr and idx < len(ocr_results) and ocr_results[idx]:
                            ocr_result = ocr_results[idx]
                            if hasattr(ocr_result, 'text') and ocr_result.text:
                                ocr_text = ocr_result.text.strip().upper()
                                ocr_confidence = getattr(ocr_result, 'confidence', 0.0)

                        # Create Detection object
                        detection = Detection(
                            class_id=0,  # License plate class
                            confidence=confidence,
                            bbox=original_bbox,
                            class_name="license_plate",
                            ocr_text=ocr_text,
                            ocr_confidence=ocr_confidence
                        )

                        detections.append(detection)
                        self.total_detections += 1

                        self.logger.debug(
                            f"Detected plate: bbox={original_bbox}, conf={confidence:.3f}, "
                            f"ocr='{ocr_text}' (conf={ocr_confidence:.3f})"
                        )

                    except Exception as e:
                        self.logger.warning(f"Error processing detection {idx}: {e}")
                        continue

            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:  # Keep last 100 measurements
                self.inference_times.pop(0)
            self.avg_inference_time = sum(self.inference_times) / len(self.inference_times)

            if detections:
                self.logger.debug(
                    f"Frame {self.frames_processed}: Found {len(detections)} plates "
                    f"(processing time: {inference_time:.3f}s)"
                )

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")

        return detections

    def get_performance_stats(self) -> dict:
        """Get detector performance statistics"""
        return {
            "detector_model": self.detector_model,
            "ocr_model": self.ocr_model,
            "confidence_threshold": self.confidence_threshold,
            "crop_position": self.crop_position,
            "detection_input_size": self.detection_input_size,
            "ocr_enabled": self.enable_ocr,
            "avg_inference_time": self.avg_inference_time,
            "inference_fps": 1.0 / self.avg_inference_time if self.avg_inference_time > 0 else 0,
            "total_detections": self.total_detections,
            "frames_processed": self.frames_processed,
            "detections_per_frame": self.total_detections / self.frames_processed if self.frames_processed > 0 else 0
        }

    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"Updated confidence threshold to {self.confidence_threshold}")

    def set_crop_position(self, position: str):
        """Update crop position"""
        if position in ['left', 'center', 'right']:
            self.crop_position = position
            self.logger.info(f"Updated crop position to {self.crop_position}")
        else:
            self.logger.warning(f"Invalid crop position: {position}. Using 'center'")
            self.crop_position = 'center'

    def get_crop_info(self, frame: np.ndarray) -> dict:
        """Get information about how the frame would be cropped"""
        _, crop_info = extract_center_square_crop(frame, self.crop_position)
        return crop_info

    def enable_ocr_processing(self, enable: bool):
        """Enable or disable OCR processing"""
        if enable != self.enable_ocr:
            self.enable_ocr = enable
            self.logger.info(f"OCR processing {'enabled' if enable else 'disabled'}")
            # Note: This requires re-initializing ALPR with/without OCR model
            # For now, just log the change. Full re-init could be added if needed.

    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_times.clear()
        self.avg_inference_time = 0.0
        self.total_detections = 0
        self.frames_processed = 0
        self.logger.info("Performance statistics reset")