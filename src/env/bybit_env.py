import asyncio
import aiowebsocket
import json
import hmac
import hashlib
import time
from loguru import logger
from src.data.features import EnhancedFeatureExtractor
# ... other imports ...

BYBIT_WEBSOCKET_URL = "wss://stream-testnet.bybit.com/v5/public/linear" #Change to mainnet if needed

class BybitFuturesEnv(gym.Env):
    def __init__(self, config, api_key=None, api_secret=None):
        # ... other initializations ...
        self.config = config
        self.feature_extractor = EnhancedFeatureExtractor(config.feature)
        self.api_key = api_key
        self.api_secret = api_secret
        self.websocket = None
        self.market_data_queue = asyncio.Queue()
        self.last_ping_time = 0

    async def connect_websocket(self):
        """Establishes a websocket connection to Bybit, handling authentication if needed."""
        if self.api_key and self.api_secret:
            auth_data = self._generate_auth_data()
            url = f"{BYBIT_WEBSOCKET_URL}?max_active_time=1m" # Customize alive time
            async with aiowebsocket.connect(url) as ws:
                self.websocket = ws
                await self.websocket.send(json.dumps(auth_data))
                auth_response = await self.websocket.receive()
                if not self._check_auth_success(json.loads(auth_response)):
                    logger.error("Authentication failed!")
                    return
                await self.subscribe_to_channels()
                await self._websocket_loop()

        else:
            async with aiowebsocket.connect(BYBIT_WEBSOCKET_URL) as ws:
                self.websocket = ws
                await self.subscribe_to_channels()
                await self._websocket_loop()


    def _generate_auth_data(self):
      """Generates authentication data for private channels."""
      expires = int((time.time() + 60) * 1000)  # Expires in 60 seconds
      message = f"GET/realtime{expires}".encode('utf-8')
      signature = hmac.new(self.api_secret.encode('utf-8'), message, hashlib.sha256).hexdigest()
      return {
          "op": "auth",
          "args": [self.api_key, expires, signature]
      }

    def _check_auth_success(self, response):
        """Checks the authentication response from Bybit."""
        return response.get('retCode') == 0

    async def subscribe_to_channels(self):
        """Subscribes to the necessary Bybit websocket channels."""
        await self.websocket.send(json.dumps({"op": "subscribe", "args": ["instrument_info.100ms.BTCUSDT"]})) #Subscribe to BTCUSDT
        # Add other subscriptions as needed (e.g., trades, klines)

    async def _websocket_loop(self):
        """Main loop for receiving and processing websocket messages."""
        while True:
            try:
                message = await self.websocket.receive()
                await self.process_websocket_message(message)
                if time.time() - self.last_ping_time > 20:  #Send ping every 20 seconds
                  self.last_ping_time = time.time()
                  await self.websocket.send(json.dumps({"op": "ping"}))
                  pong_response = await asyncio.wait_for(self.websocket.receive(), timeout=5) #wait for pong
                  if pong_response != '"pong"': #check for successful pong
                    logger.warning("Ping timeout, attempting reconnect...")
                    await self.websocket.close()
                    await self.connect_websocket() #Attempt to reconnect
                    break

            except asyncio.TimeoutError:
                logger.warning("Websocket receive timeout")
                await self.websocket.close()
                await self.connect_websocket()
                break

            except aiowebsocket.exceptions.WebSocketError as e:
                logger.error(f"Websocket error: {e}")
                await asyncio.sleep(5)
                await self.websocket.close()
                await self.connect_websocket()
                break
            except Exception as e:
                logger.exception(f"An unexpected error occurred: {e}")
                await asyncio.sleep(5)
                await self.websocket.close()
                await self.connect_websocket()
                break



    async def process_websocket_message(self, message):
        """Processes a single message received from the Bybit websocket."""
        try:
            data = json.loads(message)
            if data['topic'] == 'instrument_info.100ms.BTCUSDT':
                await self.market_data_queue.put(self._parse_market_data(data['data']))

        except json.JSONDecodeError:
            logger.error("Invalid JSON received from websocket")
        except KeyError as e:
            logger.error(f"Missing key in websocket message: {e}")
        except Exception as e:
            logger.exception(f"Error processing websocket message: {e}")


    def _parse_market_data(self, raw_data):
        """Parses and processes a single market data point from Bybit's websocket.

        Args:
          raw_data (dict): A dictionary containing raw market data from Bybit's websocket.

        Returns:
          dict: A dictionary containing processed market data ready for feature extraction. Returns an empty dictionary if parsing fails.
        """
        try:
            processed_data = {
                'close_price': float(raw_data['last_price']),
                'volume': float(raw_data['volume']),
                'bid_ask_spread': float(raw_data['best_ask_price']) - float(raw_data['best_bid_price']),
                'funding_rate': float(raw_data['funding_rate']),
                'market_depth_ratio': self._calculate_depth_ratio(raw_data),  # Implement this function
                'taker_buy_ratio': float(raw_data.get('taker_buy_sell_ratio', 0.5)) # Handle missing data gracefully
            }
            return processed_data
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error parsing market data: {e}, Raw Data: {raw_data}")
            return {}

    # ... rest of your BybitFuturesEnv class ...

    # ... (rest of your code) ...