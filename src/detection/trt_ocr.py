"""
TensorRT-based license plate OCR using PyCUDA
Based on stream-processing-pipeline implementation
"""

import logging
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Tuple

try:
    import pycuda.driver as cuda
    # Don't use pycuda.autoinit - conflicts with PyTorch CUDA context
    import tensorrt as trt
except ImportError as e:
    raise ImportError(
        "TensorRT and PyCUDA required. Install: pip install pycuda"
    ) from e

from detection.ocr_config import OCRConfig
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


class TensorRTOCREngine:
    """TensorRT inference engine for OCR with PyCUDA"""
    
    def __init__(self, engine_path: str):
        """
        Initialize TensorRT OCR engine
        
        Args:
            engine_path: Path to .trt engine file
        """
        self.engine_path = Path(engine_path)
        self.engine = None
        self.context = None
        self.d_input = None
        self.d_output = None
        self.stream = None
        self.input_shape = None
        self.output_shape = None
        
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        
        self._load_engine()
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        logger.info(f"Loading TensorRT OCR engine: {self.engine_path}")
        
        # Initialize CUDA context for PyCUDA (share with PyTorch)
        cuda.init()
        device = cuda.Device(0)
        self.cuda_ctx = device.retain_primary_context()
        self.cuda_ctx.push()  # Make context current for this thread
        
        logger.info("CUDA primary context activated for OCR")
        
        # Load engine
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output shapes
        self.input_binding = 0
        self.output_binding = 1
        
        self.input_shape = self.engine.get_tensor_shape(
            self.engine.get_tensor_name(self.input_binding)
        )
        self.output_shape = self.engine.get_tensor_shape(
            self.engine.get_tensor_name(self.output_binding)
        )
        
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Output shape: {self.output_shape}")
        
        # Get datatypes
        input_dtype = self.engine.get_tensor_dtype(
            self.engine.get_tensor_name(self.input_binding)
        )
        output_dtype = self.engine.get_tensor_dtype(
            self.engine.get_tensor_name(self.output_binding)
        )
        
        # Map TensorRT types to numpy
        dtype_map = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT8: np.int8,
            trt.DataType.UINT8: np.uint8,
            trt.DataType.INT32: np.int32,
        }
        self.input_dtype_np = dtype_map.get(input_dtype, np.float32)
        self.output_dtype_np = dtype_map.get(output_dtype, np.float32)
        
        logger.info(f"Input dtype: {input_dtype} -> {self.input_dtype_np}")
        logger.info(f"Output dtype: {output_dtype} -> {self.output_dtype_np}")
        
        # Allocate device buffers
        input_itemsize = np.dtype(self.input_dtype_np).itemsize
        output_itemsize = np.dtype(self.output_dtype_np).itemsize
        
        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * input_itemsize)
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * output_itemsize)
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        logger.info("TensorRT OCR engine loaded successfully")
    
    def infer(self, input_data: torch.Tensor) -> np.ndarray:
        """
        Run inference on input tensor
        
        Args:
            input_data: Input tensor (NHWC format, uint8 or float16 on CUDA)
        
        Returns:
            Output array (batch, seq_len, vocab_size)
        """
        # Validate input is torch tensor on CUDA
        if not input_data.is_cuda:
            raise ValueError("Input tensor must be on CUDA")
        
        if not input_data.is_contiguous():
            input_data = input_data.contiguous()
        
        # Ensure 4D: (N, H, W, C)
        if input_data.dim() == 3:
            input_data = input_data.unsqueeze(0)
        
        # Calculate byte size
        dtype_size_map = {
            torch.uint8: 1,
            torch.int8: 1,
            torch.float16: 2,
            torch.float32: 4,
        }
        bytes_per_element = dtype_size_map.get(input_data.dtype, 1)
        num_bytes = input_data.numel() * bytes_per_element
        
        # GPU-to-GPU copy
        cuda.memcpy_dtod_async(
            self.d_input,
            input_data.data_ptr(),
            num_bytes,
            self.stream
        )
        
        # Set tensor addresses
        self.context.set_tensor_address(
            self.engine.get_tensor_name(self.input_binding),
            int(self.d_input)
        )
        self.context.set_tensor_address(
            self.engine.get_tensor_name(self.output_binding),
            int(self.d_output)
        )
        
        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Allocate output host memory
        output = np.empty(self.output_shape, dtype=np.float32)
        
        # Copy output from device
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        return output
    
    def cleanup(self):
        """Clean up CUDA resources"""
        try:
            # Free device memory first
            if self.d_input is not None:
                self.d_input.free()
                self.d_input = None
            if self.d_output is not None:
                self.d_output.free()
                self.d_output = None
            
            # Delete CUDA stream
            if self.stream is not None:
                del self.stream
                self.stream = None
            
            # Delete TensorRT context and engine
            if self.context is not None:
                del self.context
                self.context = None
            if self.engine is not None:
                del self.engine
                self.engine = None
            
            # Pop CUDA context (CRITICAL: prevents "context stack not empty" error)
            if hasattr(self, 'cuda_ctx') and self.cuda_ctx is not None:
                try:
                    self.cuda_ctx.pop()
                    logger.info("CUDA context popped successfully")
                    self.cuda_ctx = None
                except Exception as ctx_err:
                    logger.warning(f"CUDA context pop failed: {ctx_err}")
                    self.cuda_ctx = None
                    
        except Exception as e:
            logger.warning(f"Error during TensorRT engine cleanup: {e}")


class TensorRTOCR:
    """License plate OCR using TensorRT"""
    
    def __init__(self, engine_path: str, config_path: str):
        """
        Initialize TensorRT OCR
        
        Args:
            engine_path: Path to TensorRT engine file
            config_path: Path to OCR config YAML
        """
        self.logger = setup_logger(self.__class__.__name__)
        
        self.logger.info("Initializing TensorRT OCR...")
        
        # Load config
        self.config = OCRConfig.from_yaml(config_path)
        self.logger.info(f"OCR config: {self.config.img_height}x{self.config.img_width}")
        self.logger.info(f"Max slots: {self.config.max_plate_slots}")
        self.logger.info(f"Alphabet size: {len(self.config.alphabet)}")
        
        # Initialize engine
        self.engine = TensorRTOCREngine(engine_path)
        
        # Warm up
        self._warmup()
        
        self.logger.info("TensorRT OCR initialized successfully")
    
    def _warmup(self):
        """Warm up the engine"""
        self.logger.info("Warming up OCR engine...")
        dummy_input = torch.randint(
            0, 256,
            (1, self.config.img_height, self.config.img_width, 3),
            dtype=torch.uint8,
            device='cuda:0'
        )
        
        for _ in range(3):
            _ = self.engine.infer(dummy_input)
        
        self.logger.info("OCR engine warmup complete")
    
    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess license plate crop for OCR
        
        Args:
            crop: BGR crop from OpenCV
        
        Returns:
            Preprocessed tensor (1, H, W, 3) uint8 RGB on CUDA
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        if self.config.keep_aspect_ratio:
            # Letterbox resize
            resized = self._letterbox_resize(
                rgb,
                (self.config.img_width, self.config.img_height)
            )
        else:
            # Direct resize
            resized = cv2.resize(
                rgb,
                (self.config.img_width, self.config.img_height),
                interpolation=cv2.INTER_CUBIC
            )
        
        # Convert to tensor (NHWC format, uint8)
        tensor = torch.from_numpy(resized).to('cuda:0', dtype=torch.uint8)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def _letterbox_resize(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Resize with letterboxing
        
        Args:
            image: RGB image
            target_size: (width, height)
        
        Returns:
            Resized and padded image
        """
        target_w, target_h = target_size
        h, w = image.shape[:2]
        
        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Create padded image
        padded = np.full(
            (target_h, target_w, 3),
            self.config.padding_color,
            dtype=np.uint8
        )
        
        # Calculate padding offsets
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    
    def postprocess(self, output: np.ndarray) -> Tuple[str, float]:
        """
        Post-process OCR output
        
        Args:
            output: Model output (1, max_slots, vocab_size)
        
        Returns:
            (text, confidence)
        """
        # Get predictions for each slot
        predictions = np.argmax(output[0], axis=1)  # [max_slots]
        confidences = np.max(output[0], axis=1)  # [max_slots]
        
        # Convert indices to characters
        text = ""
        valid_confidences = []
        
        for pred, conf in zip(predictions, confidences):
            char = self.config.alphabet[pred]
            if char == self.config.pad_char:
                break  # Stop at padding
            text += char
            valid_confidences.append(conf)
        
        # Calculate average confidence
        avg_confidence = (
            float(np.mean(valid_confidences))
            if valid_confidences else 0.0
        )
        
        return text, avg_confidence
    
    def recognize(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text from license plate crop
        
        Args:
            crop: BGR image crop from OpenCV
        
        Returns:
            (text, confidence)
        """
        try:
            # Preprocess
            tensor = self.preprocess_crop(crop)
            
            # Run inference
            output = self.engine.infer(tensor)
            
            # Post-process
            text, confidence = self.postprocess(output)
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"OCR recognition failed: {e}")
            return "", 0.0
    
    def cleanup(self):
        """Release resources"""
        if self.engine is not None:
            self.engine.cleanup()
            self.engine = None
        
        self.logger.info("TensorRT OCR cleaned up")

