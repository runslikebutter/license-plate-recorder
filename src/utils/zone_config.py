"""
Detection zone configuration management
Handles saving and loading of detection zone settings
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from .logger import setup_logger


class ZoneConfigManager:
    """Manages detection zone configuration persistence"""

    def __init__(self, config_file: str = "detection_zone.json"):
        self.config_file = Path(config_file)
        self.logger = setup_logger(self.__class__.__name__)
        self.default_zone = {
            "top": 35,
            "bottom": 5,
            "left": 5,
            "right": 5,
            "enabled": True
        }

    def load_zone_config(self) -> Dict[str, Any]:
        """Load detection zone configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.logger.info(f"Loaded detection zone config from {self.config_file}")
                    # Validate config has required fields
                    for key in self.default_zone:
                        if key not in config:
                            config[key] = self.default_zone[key]
                    return config
            else:
                self.logger.info(f"No zone config file found, using defaults")
                return self.default_zone.copy()
        except Exception as e:
            self.logger.error(f"Failed to load zone config: {e}, using defaults")
            return self.default_zone.copy()

    def save_zone_config(self, zone_config: Dict[str, Any]) -> bool:
        """Save detection zone configuration to file"""
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            # Validate and clean config
            clean_config = {}
            for key in self.default_zone:
                if key in zone_config:
                    clean_config[key] = zone_config[key]
                else:
                    clean_config[key] = self.default_zone[key]

            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(clean_config, f, indent=2)

            self.logger.info(f"Saved detection zone config to {self.config_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save zone config: {e}")
            return False

    def get_zone_percentages(self, zone_config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Get zone percentages from config"""
        if zone_config is None:
            zone_config = self.load_zone_config()

        return {
            "top": float(zone_config.get("top", self.default_zone["top"])),
            "bottom": float(zone_config.get("bottom", self.default_zone["bottom"])),
            "left": float(zone_config.get("left", self.default_zone["left"])),
            "right": float(zone_config.get("right", self.default_zone["right"]))
        }

    def update_zone_from_pixels(self, top_y: int, bottom_y: int, left_x: int, right_x: int,
                               frame_width: int, frame_height: int) -> Dict[str, Any]:
        """
        Update zone configuration from pixel coordinates

        Args:
            top_y: Top boundary in pixels from top of frame
            bottom_y: Bottom boundary in pixels from top of frame
            left_x: Left boundary in pixels from left of frame
            right_x: Right boundary in pixels from left of frame
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            Updated zone configuration
        """
        # Convert pixel coordinates to percentages
        top_percent = (top_y / frame_height) * 100
        bottom_percent = ((frame_height - bottom_y) / frame_height) * 100
        left_percent = (left_x / frame_width) * 100
        right_percent = ((frame_width - right_x) / frame_width) * 100

        # Clamp values to valid ranges (0-95% to ensure minimum zone coverage)
        top_percent = max(0, min(95, top_percent))
        bottom_percent = max(0, min(95, bottom_percent))
        left_percent = max(0, min(95, left_percent))
        right_percent = max(0, min(95, right_percent))

        zone_config = {
            "top": round(top_percent, 1),
            "bottom": round(bottom_percent, 1),
            "left": round(left_percent, 1),
            "right": round(right_percent, 1),
            "enabled": True
        }

        # Save the configuration
        self.save_zone_config(zone_config)

        return zone_config

    def reset_to_defaults(self) -> Dict[str, Any]:
        """Reset zone configuration to defaults"""
        default_config = self.default_zone.copy()
        self.save_zone_config(default_config)
        self.logger.info("Reset detection zone to defaults")
        return default_config

    def disable_zone(self) -> Dict[str, Any]:
        """Disable detection zone (use full frame)"""
        disabled_config = {
            "top": 0,
            "bottom": 0,
            "left": 0,
            "right": 0,
            "enabled": False
        }
        self.save_zone_config(disabled_config)
        self.logger.info("Disabled detection zone")
        return disabled_config