# acquisition/mexc/mexc_order_book.py

from typing import Dict, Any
from .mexc_rest_api import MexcRestAPI

class MexcOrderBook:
    def __init__(self):
        self.rest_api = None

    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        if not self.rest_api:
            raise ValueError("REST API client not initialized. Call set_rest_api first.")
        
        endpoint = "/api/v3/depth"
        params = {
            "symbol": symbol,
            "limit": limit
        }
        return await self.rest_api._request("GET", endpoint, params)

    def set_rest_api(self, rest_api: MexcRestAPI):
        self.rest_api = rest_api