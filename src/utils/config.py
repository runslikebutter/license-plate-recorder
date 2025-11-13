import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, *keys, default=None):
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    @property
    def recording(self):
        return self.config.get('recording', {})

    @property
    def detection(self):
        return self.config.get('detection', {})

    @property
    def tracking(self):
        return self.config.get('tracking', {})

    @property
    def stream(self):
        return self.config.get('stream', {})

    @property
    def zone(self):
        return self.config.get('zone', {})

    @property
    def gstreamer(self):
        return self.config.get('gstreamer', {})

    @property
    def logging(self):
        return self.config.get('logging', {})