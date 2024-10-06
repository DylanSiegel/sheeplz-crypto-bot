# tests/test_websocket_manager.py 
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch
import time

from data.ingestion.websocket_manager import WebSocketManager 

class MockWebSocket:
    def __init__(self, messages=None):
        self.messages = messages if messages else []
        self.sent_messages = []
        self.closed = False

    async def recv(self):
        if self.messages:
            return json.dumps(self.messages.pop(0))
        else:
            await asyncio.sleep(0.1)
            return None

    async def send(self, message):
        self.sent_messages.append(message)

    async def close(self):
        self.closed = True

    async def ping(self):
        pass

    async def pong(self):
        pass

@pytest.mark.asyncio
async def test_connect_and_subscribe():
    """Tests connecting to a mock WebSocket and subscribing to a channel."""
    mock_ws = MockWebSocket()
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_ws
        manager = WebSocketManager("wss://test-ws.com", "test_api_key", "test_api_secret", 10)
        
        async with manager: 
            assert manager.websocket is not None
            assert manager.websocket.closed is False

            await manager.subscribe_to_spot("BTCUSDT")

            expected_message = json.dumps({
                "method": "SUBSCRIPTION",
                "params": ["spot@public.deals.v3.api@BTCUSDT"]
            })
            assert expected_message in mock_ws.sent_messages


@pytest.mark.asyncio
async def test_send_and_receive_messages_with_rate_limit():
    """Tests sending and receiving messages with rate limiting."""
    test_messages = [
        {"test": "message1"},
        {"test": "message2"},
        {"test": "message3"},
    ]
    mock_ws = MockWebSocket(messages=test_messages.copy())  

    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_ws
        handler = WebSocketManager("wss://test-ws.com", "test_api_key", "test_api_secret", 2)  

        async with handler:
            assert handler.websocket is not None

            start_time = time.time()

            for message in test_messages:
                await handler.send_message(json.dumps(message))

            assert len(mock_ws.sent_messages) == len(test_messages)

            received_messages = []
            for _ in range(len(test_messages)):  
                received_messages.append(json.loads(await handler.receive_message()))

            assert received_messages == test_messages

            end_time = time.time()
            elapsed_time = end_time - start_time

            assert elapsed_time >= (len(test_messages) / handler.rate_limit) 