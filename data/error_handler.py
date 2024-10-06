import logging
import traceback

class ErrorHandler:
    """
    Centralized error handling and logging.
    """

    def __init__(self, log_file: str = "error.log"):
        """
        Initializes the ErrorHandler with a specified log file.

        Args:
            log_file (str, optional): Path to the log file. Defaults to "error.log".
        """
        self.logger = logging.getLogger("DataPipelineErrorHandler")
        self.logger.setLevel(logging.ERROR)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def handle_error(self, message: str, exc_info: bool = False, symbol: str = None, timeframe: str = None):
        """
        Logs an error message with optional contextual information.

        Args:
            message (str): Error message.
            exc_info (bool, optional): If True, includes traceback. Defaults to False.
            symbol (str, optional): Trading symbol related to the error. Defaults to None.
            timeframe (str, optional): Timeframe related to the error. Defaults to None.
        """
        context = f" (Symbol: {symbol}, Timeframe: {timeframe})" if symbol and timeframe else ""
        full_message = f"{message}{context}"
        if exc_info:
            self.logger.error(full_message, exc_info=True)
        else:
            self.logger.error(full_message)
