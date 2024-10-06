import asyncio
import logging
import websockets

class WebSocketManager:
    """Manages the WebSocket connection."""

    def __init__(self, ws_url: str, api_key: str, api_secret: str, rate_limit: int, config=None):
        self.ws_url = ws_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limit = rate_limit
        self.config = config
        self.ws = None  # The websocket connection object
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Establishes the WebSocket connection."""
        # The connector will handle setting the listen key in the ws_url now.
        self.ws = await websockets.connect(self.ws_url)  
        return self.ws

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the WebSocket connection."""
        await self.close()

    async def send_message(self, message: str, ws=None):
        """Sends a message over the WebSocket connection."""
        try:
            if ws:
                await ws.send(message)
            elif self.ws:
                await self.ws.send(message)  # Use self.ws if no ws argument is provided
            else:
                raise ValueError("No active WebSocket connection.")

        except websockets.exceptions.ConnectionClosed:
            self.logger.error("WebSocket connection closed when sending message.")
            raise  # Re-raise the exception to be handled by the connector

        except Exception as e:
            self.logger.exception(f"Error sending message: {e}")
            raise

    async def receive_message(self, ws):
        """Receives a message from the WebSocket."""
        try:
            message = await ws.recv()
            return message
        except websockets.exceptions.ConnectionClosed:
            self.logger.error("WebSocket connection closed when receiving message.")
            raise  # Re-raise the exception

        except Exception as e:
            self.logger.exception(f"Error receiving message: {e}")
            raise

    async def close(self):
        """Closes the WebSocket connection."""
        if self.ws and not self.ws.closed:
            await self.ws.close()
            self.ws = None  # Reset the websocket object
            self.logger.info("WebSocket connection closed.")
