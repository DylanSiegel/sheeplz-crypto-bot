from loguru import logger
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

class CustomLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Remove default logger and set up our custom configuration
        logger.remove()
        
        # Add file handler
        log_file = self.log_dir / f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="INFO",
            rotation="1 day",
            compression="zip"
        )
        
        # Add console handler
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to both file and wandb if configured"""
        metrics_str = json.dumps(metrics)
        logger.info(f"Step {step} metrics: {metrics_str}")

custom_logger = CustomLogger()  # Singleton instance