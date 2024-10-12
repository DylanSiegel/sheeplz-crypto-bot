# src/exceptions.py

class DataProcessingError(Exception):
    """Exception raised for errors in the data processing pipeline."""
    pass

class ModelTrainingError(Exception):
    """Exception raised for errors during model training."""
    pass

class WebSocketConnectionError(Exception):
    """Exception raised for WebSocket connection failures."""
    pass
