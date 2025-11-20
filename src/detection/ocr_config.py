"""
OCR Configuration loader
"""

import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class OCRConfig:
    """OCR model configuration"""
    max_plate_slots: int
    alphabet: str
    pad_char: str
    img_height: int
    img_width: int
    keep_aspect_ratio: bool
    interpolation: str
    image_color_mode: str
    padding_color: int
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load config from YAML file"""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

