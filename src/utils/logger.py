import logging
import sys
from typing import Optional

_GLOBAL_LOG_LEVEL = logging.INFO


def _resolve_level(level: Optional[str]) -> int:
    if level is None:
        return _GLOBAL_LOG_LEVEL

    if isinstance(level, str):
        level_name = level.upper()
        return getattr(logging, level_name, logging.INFO)

    if isinstance(level, int):
        return level

    return logging.INFO


def setup_logger(
    name: str,
    log_file: str = None,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Create (or reuse) a named logger with shared global level.

    If a level is provided, it becomes the new global default so subsequent
    loggers inherit it automatically. This keeps class-specific loggers aligned
    with the level specified in config.yaml (e.g., DEBUG on Jetson headless).
    """
    global _GLOBAL_LOG_LEVEL

    log_level = _resolve_level(level)
    if level is not None:
        _GLOBAL_LOG_LEVEL = log_level

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    # Update existing handlers if logger already configured
    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(log_level)
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger
