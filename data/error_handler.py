import logging
import traceback

# from .alerts import send_alert  # Uncomment if you want alert functionality

class ErrorHandler:
    def __init__(self, config=None):
        """
        Initializes the error handler.

        Args:
            config: Configuration settings, if needed for alerting or additional logging options.
        """
        self.logger = logging.getLogger(__name__)
        # self.config = config  # Uncomment if using config for alerting or other settings

    def handle_error(self, message: str, exc_info=False, alert=False, data=None):
        """
        Handles errors by logging the error and optionally sending alerts.

        Args:
            message (str): The error message to log.
            exc_info (bool): If True, includes the traceback in the log and prints it.
            alert (bool): If True, sends an alert notification (if implemented).
            data (Any): Optional additional data related to the error.
        """
        # 1. Log the error
        if data is not None:
            self.logger.error(f"{message} | Data: {data}", exc_info=exc_info)
        else:
            self.logger.error(message, exc_info=exc_info)  # Log the error message, with traceback if exc_info=True
        if exc_info:
            traceback.print_exc()  # Optionally print the full traceback to the console

        # 2. Optional: Send an alert (uncomment if needed)
        # if alert:  # Example alerting functionality (e.g., if config.alerts_enabled)
        #     try:
        #         send_alert(message)  # This would be your alerting function (e.g., email/SMS)
        #     except Exception as alert_error:
        #         self.logger.error(f"Failed to send alert: {alert_error}")
