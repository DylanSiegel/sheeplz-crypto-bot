"""
Redis Service for Pantheon Server

This module provides Redis caching and data storage functionality
for the Pantheon cryptocurrency analysis platform.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import redis
from loguru import logger


class RedisService:
    """
    Redis service for caching analysis results, market data, and session storage.
    
    Provides methods for:
    - Caching analysis results with TTL
    - Storing market data temporarily  
    - Session management
    - Key-value operations with prefixes
    """
    
    def __init__(self):
        """Initialize Redis connection with environment configuration."""
        try:
            # Get Redis configuration from environment
            self.host = os.getenv("REDIS_HOST", "localhost")
            self.port = int(os.getenv("REDIS_PORT", 6379))
            
            # Get password from environment variable (prioritize system env var)
            self.password = os.getenv("PANTHEON_REDIS_PASSWORD") or os.getenv("REDIS_PASSWORD")
            
            # Debug logging for password resolution
            logger.debug(f"PANTHEON_REDIS_PASSWORD: {'***' if os.getenv('PANTHEON_REDIS_PASSWORD') else 'None'}")
            logger.debug(f"REDIS_PASSWORD: {'***' if os.getenv('REDIS_PASSWORD') else 'None'}")
            logger.debug(f"Final password resolved: {'***' if self.password else 'None'}")
            
            if not self.password:
                logger.warning("No Redis password found in environment variables!")
            
            self.db = int(os.getenv("REDIS_DB", 0))
            self.decode_responses = os.getenv("REDIS_DECODE_RESPONSES", "true").lower() == "true"
            
            # Connection settings
            socket_connect_timeout = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", 5))
            socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", 5))
            max_connections = int(os.getenv("REDIS_CONNECTION_POOL_MAX_CONNECTIONS", 10))
            
            # Key prefixes
            self.prefix_analysis = os.getenv("REDIS_PREFIX_ANALYSIS", "pantheon:analysis")
            self.prefix_cache = os.getenv("REDIS_PREFIX_CACHE", "pantheon:cache")
            self.prefix_market = os.getenv("REDIS_PREFIX_MARKET", "pantheon:market")
            self.prefix_session = os.getenv("REDIS_PREFIX_SESSION", "pantheon:session")
            
            # TTL settings (in seconds)
            self.ttl_market_data = int(os.getenv("CACHE_TTL_MARKET_DATA", 300))  # 5 minutes
            self.ttl_analysis_results = int(os.getenv("CACHE_TTL_ANALYSIS_RESULTS", 1800))  # 30 minutes
            self.ttl_price_data = int(os.getenv("CACHE_TTL_PRICE_DATA", 60))  # 1 minute
            
            # Create connection pool
            self.connection_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=self.decode_responses,
                socket_connect_timeout=socket_connect_timeout,
                socket_timeout=socket_timeout,
                max_connections=max_connections
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis connected successfully to {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _build_key(self, prefix: str, key: str) -> str:
        """Build a prefixed key for Redis operations."""
        return f"{prefix}:{key}"
    
    # === Analysis Results Caching ===
    
    def cache_analysis_result(self, product_id: str, timeframe: str, legend_type: str, result: Dict[str, Any]) -> bool:
        """
        Cache analysis results with automatic TTL.
        
        Args:
            product_id: Trading pair (e.g., "BTC-USD")
            timeframe: Analysis timeframe (e.g., "5m", "1h")
            legend_type: Analysis engine type
            result: Analysis result dictionary
            
        Returns:
            True if cached successfully
        """
        try:
            key = self._build_key(self.prefix_analysis, f"{product_id}:{timeframe}:{legend_type}")
            
            # Add timestamp to result
            result_with_timestamp = {
                **result,
                "cached_at": datetime.utcnow().isoformat(),
                "cache_key": key
            }
            
            # Cache with TTL
            success = self.redis_client.setex(
                key, 
                self.ttl_analysis_results, 
                json.dumps(result_with_timestamp)
            )
            
            if success:
                logger.debug(f"Cached analysis result: {key}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache analysis result: {e}")
            return False
    
    def get_cached_analysis(self, product_id: str, timeframe: str, legend_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis result.
        
        Args:
            product_id: Trading pair (e.g., "BTC-USD")
            timeframe: Analysis timeframe (e.g., "5m", "1h")  
            legend_type: Analysis engine type
            
        Returns:
            Cached analysis result or None if not found/expired
        """
        try:
            key = self._build_key(self.prefix_analysis, f"{product_id}:{timeframe}:{legend_type}")
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                result = json.loads(cached_data)
                logger.debug(f"Retrieved cached analysis: {key}")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached analysis: {e}")
            return None
    
    # === Market Data Caching ===
    
    def cache_market_data(self, product_id: str, timeframe: str, candles: List[Dict]) -> bool:
        """
        Cache market candle data temporarily.
        
        Args:
            product_id: Trading pair
            timeframe: Candle timeframe
            candles: List of candle dictionaries
            
        Returns:
            True if cached successfully
        """
        try:
            key = self._build_key(self.prefix_market, f"{product_id}:{timeframe}")
            
            market_data = {
                "candles": candles,
                "cached_at": datetime.utcnow().isoformat(),
                "count": len(candles)
            }
            
            success = self.redis_client.setex(
                key,
                self.ttl_market_data,
                json.dumps(market_data)
            )
            
            if success:
                logger.debug(f"Cached {len(candles)} candles for {product_id}:{timeframe}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache market data: {e}")
            return False
    
    def get_cached_market_data(self, product_id: str, timeframe: str) -> Optional[List[Dict]]:
        """
        Retrieve cached market data.
        
        Args:
            product_id: Trading pair
            timeframe: Candle timeframe
            
        Returns:
            List of candle dictionaries or None if not found/expired
        """
        try:
            key = self._build_key(self.prefix_market, f"{product_id}:{timeframe}")
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                market_data = json.loads(cached_data)
                logger.debug(f"Retrieved {market_data['count']} cached candles for {product_id}:{timeframe}")
                return market_data["candles"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached market data: {e}")
            return None
    
    # === Price Data Caching ===
    
    def cache_price_data(self, product_id: str, price_data: Dict[str, Any]) -> bool:
        """
        Cache current price data with short TTL.
        
        Args:
            product_id: Trading pair
            price_data: Price information dictionary
            
        Returns:
            True if cached successfully
        """
        try:
            key = self._build_key(self.prefix_cache, f"price:{product_id}")
            
            price_with_timestamp = {
                **price_data,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            success = self.redis_client.setex(
                key,
                self.ttl_price_data,
                json.dumps(price_with_timestamp)
            )
            
            if success:
                logger.debug(f"Cached price data for {product_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache price data: {e}")
            return False
    
    def get_cached_price_data(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached price data.
        
        Args:
            product_id: Trading pair
            
        Returns:
            Price data dictionary or None if not found/expired
        """
        try:
            key = self._build_key(self.prefix_cache, f"price:{product_id}")
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                price_data = json.loads(cached_data)
                logger.debug(f"Retrieved cached price data for {product_id}")
                return price_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached price data: {e}")
            return None
    
    # === Generic Cache Operations ===
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a key-value pair with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if set successfully
        """
        try:
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            
            if ttl:
                success = self.redis_client.setex(key, ttl, serialized_value)
            else:
                success = self.redis_client.set(key, serialized_value)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            return False
    
    def get(self, key: str, parse_json: bool = True) -> Optional[Any]:
        """
        Get a value by key.
        
        Args:
            key: Cache key
            parse_json: Whether to parse the value as JSON
            
        Returns:
            Cached value or None if not found
        """
        try:
            value = self.redis_client.get(key)
            
            if value and parse_json:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a key.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            result = self.redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists
        """
        try:
            return self.redis_client.exists(key) > 0
            
        except Exception as e:
            logger.error(f"Failed to check key existence {key}: {e}")
            return False
    
    # === Utility Methods ===
    
    def clear_analysis_cache(self, product_id: Optional[str] = None) -> int:
        """
        Clear analysis cache for a specific product or all products.
        
        Args:
            product_id: Specific product to clear (optional)
            
        Returns:
            Number of keys deleted
        """
        try:
            if product_id:
                pattern = self._build_key(self.prefix_analysis, f"{product_id}:*")
            else:
                pattern = self._build_key(self.prefix_analysis, "*")
            
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} analysis cache entries")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to clear analysis cache: {e}")
            return 0
    
    def clear_market_cache(self) -> int:
        """
        Clear all market data cache.
        
        Returns:
            Number of keys deleted
        """
        try:
            pattern = self._build_key(self.prefix_market, "*")
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} market cache entries")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to clear market cache: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            info = self.redis_client.info()
            
            # Count keys by prefix
            analysis_keys = len(self.redis_client.keys(self._build_key(self.prefix_analysis, "*")))
            market_keys = len(self.redis_client.keys(self._build_key(self.prefix_market, "*")))
            cache_keys = len(self.redis_client.keys(self._build_key(self.prefix_cache, "*")))
            
            return {
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "total_keys": info.get("db0", {}).get("keys", 0) if "db0" in info else 0,
                "analysis_cache_keys": analysis_keys,
                "market_cache_keys": market_keys,
                "general_cache_keys": cache_keys,
                "uptime_seconds": info.get("uptime_in_seconds")
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform Redis health check.
        
        Returns:
            Health status dictionary
        """
        try:
            start_time = datetime.utcnow()
            pong = self.redis_client.ping()
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "status": "healthy" if pong else "unhealthy",
                "response_time_ms": response_time,
                "connection": {
                    "host": self.host,
                    "port": self.port,
                    "db": self.db
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection": {
                    "host": self.host,
                    "port": self.port,
                    "db": self.db
                }
            }
    
    def close(self):
        """Close Redis connection."""
        try:
            self.redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
