# logging_config.py

import logging

def setup_logging(log_file: str = 'data_processing.log') -> None:
    """
    Configures the logging settings for the application.
    
    Args:
        log_file (str): Path to the log file. Defaults to 'data_processing.log'.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'  # Append mode to avoid overwriting existing logs
    )
