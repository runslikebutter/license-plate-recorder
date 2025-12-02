import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    def __init__(self, config_path: str = "/etc/recorder/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._apply_env_overrides()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _apply_env_overrides(self):
        """Override config values from environment variables (for sensitive data)"""
        # S3 credentials from environment variables
        if 's3_upload' in self.config:
            if os.environ.get('S3_ACCESS_KEY_ID'):
                self.config['s3_upload']['access_key_id'] = os.environ['S3_ACCESS_KEY_ID']
            if os.environ.get('S3_SECRET_ACCESS_KEY'):
                self.config['s3_upload']['secret_access_key'] = os.environ['S3_SECRET_ACCESS_KEY']
            
            # Optional: allow bucket and endpoint override too
            if os.environ.get('S3_BUCKET'):
                self.config['s3_upload']['bucket'] = os.environ['S3_BUCKET']
            if os.environ.get('S3_ENDPOINT'):
                self.config['s3_upload']['endpoint'] = os.environ['S3_ENDPOINT']

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
    def logging(self):
        return self.config.get('logging', {})
