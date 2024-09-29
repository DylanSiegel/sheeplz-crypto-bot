# File: utils/logging_config.py

import logging
import os
from typing import Dict

def setup_logging(config: Dict):
    log_level = getattr(logging, config.level.upper())
    log_format = config.format
    log_file = config.file

    logging.basicConfig(level=log_level, format=log_format)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

def get_logger(name: str):
    return logging.getLogger(name)