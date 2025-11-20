"""
License plate detector using Ultralytics YOLO with TensorRT engine
Based on stream-processing-pipeline implementation
"""

import time
import logging
import cv2
import numpy as np
import torch
from typing import List, Optional
from ultralytics import YOLO

from detection.detector import Detector, Detection
from detection.trt_ocr import TensorRTOCR
from utils.logger import setup_logger


logger = logging.getLogger(__name__)


class YOLOLPDDetector(Detector):
    """
    License plate detector using Ultralytics YOLO (TensorRT engine)
    """
    
    # Class IDs
    LPD_CLASS_ID = 0
    CAR_CLASS_ID = 1
    
    CLASS_NAMES = {0: "license_plate", 1: "car"}
    WARMUP_ITERATIONS = 3
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        max_det: int = 300,
        device: str = "cuda:0",
        enable_ocr: bool = True,
        ocr_engine_path: Optional[str] = None,
        ocr_config_path: Optional[str] = None
    ):
        """
        Initialize YOLO detector with TensorRT engine
        
        Args:
            model_path: Path to YOLO TensorRT engine file
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            input_size: Model input size
            max_det: Maximum detections per image
            device: CUDA device
            enable_ocr: Enable OCR recognition
            ocr_engine_path: Path to OCR TensorRT engine
            ocr_config_path: Path to OCR config YAML
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.max_det = max_det
        self.device = device
        self.enable_ocr = enable_ocr
        self.ocr_engine_path = ocr_engine_path
        self.ocr_config_path = ocr_config_path
        
        self.logger = setup_logger(self.__class__.__name__)
        
        # Performance tracking
        self.inference_times = []
        self.ocr_times = []
        self.total_detections = 0
        self.frames_processed = 0
        
        # Initialize YOLO
        self.model = None
        self._initialize_model()
        
        # Initialize OCR if enabled
        self.ocr = None
        if self.enable_ocr and ocr_engine_path and ocr_config_path:
            self._initialize_ocr()
    
    def _initialize_model(self):
        """Initialize Ultralytics YOLO model with TensorRT engine"""
        try:
            self.logger.info("Initializing YOLO TensorRT engine...")
            self.logger.info(f"Engine path: {self.model_path}")
            self.logger.info(f"Input size: {self.input_size}")
            self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
            self.logger.info(f"IoU threshold: {self.iou_threshold}")
            
            # Load YOLO model (handles TensorRT engines automatically)
            # Disable verbose logging to reduce overhead
            import os
            os.environ['YOLO_VERBOSE'] = 'False'
            self.model = YOLO(self.model_path, task="detect", verbose=False)
            
            # Map class names
            for key in self.CLASS_NAMES.keys():
                if key < len(self.model.names):
                    self.model.names[key] = self.CLASS_NAMES[key]
            
            # Warm up the engine with FP16
            self.logger.info(f"Warming up YOLO engine ({self.WARMUP_ITERATIONS} iterations)...")
            dummy_input = torch.zeros(
                (1, 3, self.input_size, self.input_size),
                device=self.device,
                dtype=torch.float16  # FP16 for TensorRT
            )
            
            with torch.no_grad():
                for i in range(self.WARMUP_ITERATIONS):
                    _ = self.model(
                        dummy_input,
                        verbose=False,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        max_det=self.max_det
                    )
                    torch.cuda.synchronize()
            
            self.logger.info("YOLO engine initialized and warmed up successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO engine: {e}")
            raise
    
    def _initialize_ocr(self):
        """Initialize TensorRT OCR engine"""
        try:
            self.logger.info("Initializing TensorRT OCR engine...")
            self.logger.info(f"OCR engine: {self.ocr_engine_path}")
            self.logger.info(f"OCR config: {self.ocr_config_path}")
            
            self.ocr = TensorRTOCR(
                engine_path=self.ocr_engine_path,
                config_path=self.ocr_config_path
            )
            
            self.logger.info("TensorRT OCR initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR: {e}")
            self.logger.warning("Continuing without OCR")
            self.ocr = None
            self.enable_ocr = False
    
    def preprocess(self, frame: np.ndarray) -> tuple[torch.Tensor, float, tuple, tuple]:
        """
        Preprocess frame with letterboxing
        
        Args:
            frame: Input BGR frame from OpenCV
        
        Returns:
            - Preprocessed tensor (1, 3, H, W) in RGB format, normalized
            - Scale factor
            - Padding (dw, dh)
            - Original shape (height, width)
        """
        orig_h, orig_w = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor with FP16 for TensorRT engine
        tensor = torch.from_numpy(rgb_frame).to(self.device, dtype=torch.float16)
        
        # Letterbox resize using torch operations
        letterboxed, dh, dw, r = self._letterbox(
            tensor,
            self.input_size,
            self.input_size
        )
        
        return letterboxed, r, (dw, dh), (orig_h, orig_w)
    
    def _letterbox(
        self,
        rgb_tensor: torch.Tensor,
        height: int,
        width: int
    ) -> tuple[torch.Tensor, int, int, float]:
        """
        Letterbox resize maintaining aspect ratio
        
        Args:
            rgb_tensor: Input RGB tensor (H, W, C)
            height: Target height
            width: Target width
        
        Returns:
            - Letterboxed tensor (1, 3, height, width), normalized [0, 1]
            - dh: Vertical padding
            - dw: Horizontal padding
            - r: Scale factor
        """
        orig_height, orig_width = rgb_tensor.shape[:2]
        
        # Add batch and channel dimensions if needed
        if len(rgb_tensor.shape) == 3:
            rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        
        # Calculate letterbox scaling
        r_height = height / orig_height
        r_width = width / orig_width
        r = min(r_height, r_width)
        new_height = int(orig_height * r)
        new_width = int(orig_width * r)
        
        # Resize maintaining aspect ratio
        resized = torch.nn.functional.interpolate(
            rgb_tensor,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
            antialias=True
        )
        
        # Create letterboxed tensor with padding (FP16)
        letterboxed = torch.full(
            (1, 3, height, width),
            114,
            device=self.device,
            dtype=torch.float16
        )
        
        # Calculate padding
        dh = (height - new_height) // 2
        dw = (width - new_width) // 2
        
        # Place resized image in center
        letterboxed[..., dh:dh + new_height, dw:dw + new_width] = resized
        
        # Normalize to [0, 1]
        letterboxed = letterboxed / 255.0
        letterboxed = torch.clamp(letterboxed, 0.0, 1.0)
        
        return letterboxed, dh, dw, r
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect license plates in frame
        
        Args:
            frame: Input BGR frame from OpenCV
        
        Returns:
            List of Detection objects (license plates only)
        """
        if self.model is None:
            self.logger.error("YOLO model not initialized")
            return []
        
        start_time = time.time()
        
        try:
            # Preprocess
            tensor, scale_factor, padding, orig_shape = self.preprocess(frame)
            dw, dh = padding
            
            # Run inference (YOLO handles NMS internally)
            with torch.no_grad():
                results = self.model(
                    tensor,
                    verbose=False,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_det
                )[0]
            
            # Extract results
            boxes = results.boxes.xyxy  # xyxy format
            scores = results.boxes.conf
            class_ids = results.boxes.cls
            
            # Scale boxes back to original coordinates
            if len(boxes) > 0:
                boxes = self._scale_to_original(
                    boxes, dh, dw, scale_factor,
                    orig_shape[0], orig_shape[1]
                )
            
            # Filter for license plates only (class 0)
            detections = []
            for box, score, class_id in zip(boxes, scores, class_ids):
                class_id_int = int(class_id.item())
                
                # Only keep license plates
                if class_id_int != self.LPD_CLASS_ID:
                    continue
                
                # Extract bbox coordinates
                x1 = float(box[0].item())
                y1 = float(box[1].item())
                x2 = float(box[2].item())
                y2 = float(box[3].item())
                
                # Run OCR if enabled
                ocr_text = None
                ocr_confidence = None
                
                if self.enable_ocr and self.ocr is not None:
                    try:
                        # Crop license plate region
                        crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        if crop.size > 0:
                            # Run OCR
                            ocr_start = time.time()
                            ocr_text, ocr_confidence = self.ocr.recognize(crop)
                            ocr_time = time.time() - ocr_start
                            
                            self.ocr_times.append(ocr_time)
                            if len(self.ocr_times) > 100:
                                self.ocr_times.pop(0)
                    except Exception as e:
                        self.logger.debug(f"OCR failed for detection: {e}")
                
                det = Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(score.item()),
                    class_id=class_id_int,
                    class_name=self.CLASS_NAMES.get(class_id_int, "unknown"),
                    ocr_text=ocr_text,
                    ocr_confidence=ocr_confidence
                )
                
                detections.append(det)
            
            # Update statistics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            self.frames_processed += 1
            self.total_detections += len(detections)
            
            if detections:
                self.logger.debug(
                    f"Frame {self.frames_processed}: Found {len(detections)} "
                    f"plates (processing time: {inference_time:.3f}s)"
                )
            
            # Periodic cleanup to prevent memory buildup
            if self.frames_processed % 100 == 0:
                torch.cuda.empty_cache()
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}", exc_info=True)
            return []
    
    def _scale_to_original(
        self,
        boxes: torch.Tensor,
        dh: int,
        dw: int,
        r: float,
        orig_h: int,
        orig_w: int
    ) -> torch.Tensor:
        """Scale boxes back to original coordinates"""
        if len(boxes) == 0:
            return boxes
        
        boxes = boxes.clone()
        
        # Undo letterbox padding and scale back to original
        boxes[:, [0, 2]] -= dw  # x1, x2
        boxes[:, [1, 3]] -= dh  # y1, y2
        boxes /= r  # scale
        
        # Clamp to image bounds
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)
        
        return boxes
    
    def get_statistics(self) -> dict:
        """Get detector statistics"""
        avg_time = (
            sum(self.inference_times) / len(self.inference_times)
            if self.inference_times else 0
        )
        avg_ocr_time = (
            sum(self.ocr_times) / len(self.ocr_times)
            if self.ocr_times else 0
        )
        
        return {
            'frames_processed': self.frames_processed,
            'total_detections': self.total_detections,
            'avg_inference_time': avg_time,
            'avg_ocr_time': avg_ocr_time,
            'avg_detections_per_frame': (
                self.total_detections / self.frames_processed
                if self.frames_processed > 0 else 0
            )
        }
    
    def get_performance_stats(self) -> dict:
        """Get detector performance statistics"""
        return self.get_statistics()
    
    def get_crop_info(self, frame: np.ndarray) -> dict:
        """Get information about frame preprocessing"""
        h, w = frame.shape[:2]
        return {
            'crop_type': 'letterbox',
            'input_size': self.input_size,
            'original_size': (w, h),
            'aspect_ratio': w / h
        }
    
    def cleanup(self):
        """Release resources"""
        if self.ocr is not None:
            self.ocr.cleanup()
            self.ocr = None
        
        if self.model is not None:
            self.logger.info("Cleaning up YOLO model resources")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del self.model
            self.model = None
        
        self.logger.info("YOLO detector cleaned up")

