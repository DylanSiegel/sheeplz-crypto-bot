"""
Coinbase Service

This module provides integration with Coinbase's Advanced Trade API
for real-time and historical cryptocurrency market data.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd
from loguru import logger


class CoinbaseService:
    """
    Service for interacting with Coinbase Advanced Trade API
    
    Provides methods for fetching real-time prices, historical candle data,
    and available trading pairs with proper rate limiting and error handling.
    """
    
    BASE_URL = "https://api.exchange.coinbase.com"
    
    def __init__(self, rate_limit_per_second: int = 10):
        """
        Initialize the Coinbase service
        
        Args:
            rate_limit_per_second: Maximum requests per second to avoid rate limiting
        """
        self.rate_limit = rate_limit_per_second
        self.last_request_time = 0
        self.min_request_interval = 1.0 / rate_limit_per_second
        
    async def _ensure_rate_limit(self) -> None:
        """Ensure we don't exceed the rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def get_products(self) -> List[Dict]:
        """
        Get all available trading pairs from Coinbase
        
        Returns:
            List of product dictionaries with trading pair information
        """
        await self._ensure_rate_limit()
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.BASE_URL}/products")
                response.raise_for_status()
                
                products = response.json()
                logger.info(f"Retrieved {len(products)} trading pairs from Coinbase")
                return products
                
            except httpx.HTTPError as e:
                logger.error(f"Error fetching products from Coinbase: {e}")
                raise
    
    async def get_product_candles(
        self,
        product_id: str,
        timeframe: str = "300",  # 5 minutes
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        max_candles: int = 300
    ) -> pd.DataFrame:
        """
        Get historical candle data for a specific product
        
        Args:
            product_id: Trading pair ID (e.g., "BTC-USD")
            timeframe: Granularity in seconds (60, 300, 900, 3600, 21600, 86400)
            start: Start time for data range
            end: End time for data range  
            max_candles: Maximum number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        await self._ensure_rate_limit()
        
        # Set default time range if not provided
        if end is None:
            end = datetime.utcnow()
        if start is None:
            # Calculate start time based on timeframe and max_candles
            timeframe_seconds = int(timeframe)
            start = end - timedelta(seconds=timeframe_seconds * max_candles)
        
        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "granularity": timeframe
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.BASE_URL}/products/{product_id}/candles",
                    params=params
                )
                response.raise_for_status()
                
                candles = response.json()
                
                if not candles:
                    logger.warning(f"No candle data returned for {product_id}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                # Coinbase returns: [timestamp, low, high, open, close, volume]
                df = pd.DataFrame(candles, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                
                # Convert timestamp to datetime and set as index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                
                # Ensure numeric types
                for col in ['low', 'high', 'open', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                # Sort by timestamp (oldest first)
                df.sort_index(inplace=True)
                
                logger.info(f"Retrieved {len(df)} candles for {product_id} ({timeframe}s timeframe)")
                return df
                
            except httpx.HTTPError as e:
                logger.error(f"Error fetching candles for {product_id}: {e}")
                raise
    
    async def get_product_ticker(self, product_id: str) -> Dict:
        """
        Get current ticker information for a product
        
        Args:
            product_id: Trading pair ID (e.g., "BTC-USD")
            
        Returns:
            Dictionary with current price and market data
        """
        await self._ensure_rate_limit()
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.BASE_URL}/products/{product_id}/ticker")
                response.raise_for_status()
                
                ticker = response.json()
                logger.debug(f"Retrieved ticker for {product_id}: ${ticker.get('price', 'N/A')}")
                return ticker
                
            except httpx.HTTPError as e:
                logger.error(f"Error fetching ticker for {product_id}: {e}")
                raise
    
    async def get_multi_timeframe_data(
        self,
        product_id: str,
        timeframes: List[str] = ["60", "300", "900"],  # 1m, 5m, 15m
        max_candles: int = 300
    ) -> Dict[str, pd.DataFrame]:
        """
        Get candle data for multiple timeframes simultaneously
        
        Args:
            product_id: Trading pair ID (e.g., "BTC-USD")
            timeframes: List of timeframe granularities in seconds
            max_candles: Maximum candles per timeframe
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        tasks = []
        for timeframe in timeframes:
            task = self.get_product_candles(
                product_id=product_id,
                timeframe=timeframe,
                max_candles=max_candles
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        return {
            timeframe: df 
            for timeframe, df in zip(timeframes, results)
        }
    
    def get_popular_crypto_pairs(self) -> List[str]:
        """
        Get a list of popular cryptocurrency trading pairs for testing
        
        Returns:
            List of popular product IDs
        """
        return [
            "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD",
            "LINK-USD", "MATIC-USD", "AVAX-USD", "ATOM-USD", "ALGO-USD",
            "XTZ-USD", "FIL-USD", "AAVE-USD", "UNI-USD", "SUSHI-USD"
        ]
