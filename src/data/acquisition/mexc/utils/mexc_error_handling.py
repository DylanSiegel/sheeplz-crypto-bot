# acquisition/mexc/utils/mexc_error_handling.py

import logging

logger = logging.getLogger(__name__)

class MexcAPIException(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

def handle_rest_error(status_code: int, error_message: str):
    logger.error(f"MEXC API error: status_code={status_code}, message={error_message}")
    raise MexcAPIException(status_code, error_message)

def handle_websocket_error(error: Exception):
    logger.error(f"MEXC WebSocket error: {str(error)}")
    # The calling method will handle reconnection
    raise error