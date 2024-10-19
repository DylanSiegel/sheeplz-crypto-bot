# logging_config.py
import logging
import logging.config
import os
from logging.handlers import TimedRotatingFileHandler

# Logging Configuration Dictionary
LOGGING_CONFIG = {
   'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
       'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s (%(lineno)d): %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter':'standard',
           'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'data_processing.log',
            'when':'midnight',
            'interval': 1,
            'backupCount': 30
        }
    },
    'loggers': {
        '': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        }
    }
}

# Set up logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Create loggers
def setup_logging(log_file: str = 'data_processing.log') -> None:
    """
    Configures the logging settings for the application.
    
    Args:
        log_file (str): Path to the log file. Defaults to 'data_processing.log'.
    """
    global LOGGING_CONFIG
    
    # Update log file name in the configuration
    LOGGING_CONFIG['handlers']['file']['filename'] = log_file
    
    # Apply updated configuration
    logging.config.dictConfig(LOGGING_CONFIG)

def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance.
    
    Args:
        name (str): Name of the logger.
    
    Returns:
        logging.Logger: Logger instance.
    """
    return logging.getLogger(name)