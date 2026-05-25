"""
logger.py
---------
Centralized logging configuration for the application.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance with the given name.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
