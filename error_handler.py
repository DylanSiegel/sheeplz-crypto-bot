# File: error_handler.py
import logging

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger("ErrorHandler")

    def handle_error(self, message: str, exc_info: bool = False, **context):
        context_info = ' | '.join([f"{key}={value}" for key, value in context.items()])
        full_message = f"{message} | {context_info}" if context else message
        self.logger.error(full_message, exc_info=exc_info)
