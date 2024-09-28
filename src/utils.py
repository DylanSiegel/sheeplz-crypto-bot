# File: src/utils.py

import logging
import os
import sys

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Sets up logging configuration.

    Args:
        log_level (str): Logging level.
        log_file (Optional[str]): File to log messages. If None, logs to stdout.
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        handlers=handlers
    )

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a logger instance.

    Args:
        name (Optional[str]): Name of the logger. If None, returns the root logger.

    Returns:
        logging.Logger: Logger instance.
    """
    return logging.getLogger(name)
