# acquisition/mexc/mexc_rest_api.py

import aiohttp
import time
from typing import Dict, Any, List
from .utils.mexc_auth import generate_signature
from .utils.mexc_endpoints import MEXC_API_URL
from .utils.mexc_error_handling import handle_rest_error

class MexcRestAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = aiohttp.ClientSession()

    async def close(self):
        if not self.session.closed:
            await self.session.close()

    async def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Any:
        url = f"{MEXC_API_URL}{endpoint}"
        headers = {"X-MEXC-APIKEY": self.api_key}
        params = params or {}

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = generate_signature(self.api_secret, params)

        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == 'POST':
                async with self.session.post(url, data=params, headers=headers) as response:
                    return await self._handle_response(response)
        except aiohttp.ClientError as e:
            handle_rest_error(None, str(e))

    async def _handle_response(self, response):
        if response.status != 200:
            text = await response.text()
            handle_rest_error(response.status, text)
        return await response.json()

    async def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[Dict[str, Any]]:
        endpoint = "/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time
        }
        return await self._request("GET", endpoint, params)

    async def get_account_info(self) -> Dict[str, Any]:
        endpoint = "/api/v3/account"
        return await self._request("GET", endpoint, signed=True)

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict[str, Any]:
        endpoint = "/api/v3/order"
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity
        }
        if price:
            params["price"] = price
        return await self._request("POST", endpoint, params, signed=True)