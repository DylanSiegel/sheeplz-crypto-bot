#!/usr/bin/env python3
"""
Redis Service Test Script

Test the Redis service functionality with the pantheon-server configuration.
"""

import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from pantheon_server.services.redis_service import RedisService
import json

def test_redis_service():
    """Test Redis service functionality."""
    print("🧪 Testing Redis Service for Pantheon Server")
    print("=" * 50)
    
    try:
        # Initialize Redis service
        print("1. Initializing Redis service...")
        redis_service = RedisService()
        print("   ✅ Redis connection established")
        
        # Health check
        print("\n2. Performing health check...")
        health = redis_service.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Response time: {health['response_time_ms']:.2f}ms")
        
        # Test basic operations
        print("\n3. Testing basic cache operations...")
        test_key = "test:pantheon:basic"
        test_value = {"message": "Hello Pantheon!", "timestamp": "2025-09-14"}
        
        # Set
        success = redis_service.set(test_key, test_value, ttl=60)
        print(f"   Set operation: {'✅' if success else '❌'}")
        
        # Get
        retrieved = redis_service.get(test_key)
        print(f"   Get operation: {'✅' if retrieved == test_value else '❌'}")
        
        # Exists
        exists = redis_service.exists(test_key)
        print(f"   Exists check: {'✅' if exists else '❌'}")
        
        # Test analysis caching
        print("\n4. Testing analysis result caching...")
        analysis_result = {
            "signal": "BULLISH",
            "confidence": 0.85,
            "trend": "UPWARD",
            "analysis": {
                "price_change": 2.5,
                "volume_ratio": 1.8,
                "momentum": 0.12
            }
        }
        
        cached = redis_service.cache_analysis_result(
            product_id="BTC-USD",
            timeframe="5m", 
            legend_type="traditional",
            result=analysis_result
        )
        print(f"   Cache analysis: {'✅' if cached else '❌'}")
        
        # Retrieve cached analysis
        retrieved_analysis = redis_service.get_cached_analysis(
            product_id="BTC-USD",
            timeframe="5m",
            legend_type="traditional"
        )
        print(f"   Retrieve analysis: {'✅' if retrieved_analysis else '❌'}")
        
        if retrieved_analysis:
            print(f"   Signal: {retrieved_analysis.get('signal')}")
            print(f"   Confidence: {retrieved_analysis.get('confidence')}")
            print(f"   Cached at: {retrieved_analysis.get('cached_at')}")
        
        # Test market data caching
        print("\n5. Testing market data caching...")
        sample_candles = [
            {"time": 1726334400, "low": 65000, "high": 65500, "open": 65200, "close": 65400, "volume": 150.5},
            {"time": 1726334700, "low": 65300, "high": 65800, "open": 65400, "close": 65700, "volume": 200.3},
            {"time": 1726335000, "low": 65600, "high": 66000, "open": 65700, "close": 65900, "volume": 180.7}
        ]
        
        market_cached = redis_service.cache_market_data("BTC-USD", "5m", sample_candles)
        print(f"   Cache market data: {'✅' if market_cached else '❌'}")
        
        retrieved_candles = redis_service.get_cached_market_data("BTC-USD", "5m")
        print(f"   Retrieve market data: {'✅' if retrieved_candles else '❌'}")
        print(f"   Candles count: {len(retrieved_candles) if retrieved_candles else 0}")
        
        # Cache statistics
        print("\n6. Cache statistics...")
        stats = redis_service.get_cache_stats()
        if stats:
            print(f"   Redis version: {stats.get('redis_version')}")
            print(f"   Total keys: {stats.get('total_keys')}")
            print(f"   Analysis cache keys: {stats.get('analysis_cache_keys')}")
            print(f"   Market cache keys: {stats.get('market_cache_keys')}")
            print(f"   Memory used: {stats.get('used_memory_human')}")
        
        # Cleanup test data
        print("\n7. Cleaning up test data...")
        deleted_basic = redis_service.delete(test_key)
        deleted_analysis = redis_service.clear_analysis_cache("BTC-USD")
        deleted_market = redis_service.clear_market_cache()
        
        print(f"   Basic test key deleted: {'✅' if deleted_basic else '❌'}")
        print(f"   Analysis cache cleared: {deleted_analysis} keys")
        print(f"   Market cache cleared: {deleted_market} keys")
        
        print("\n🎉 Redis service test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Redis service test failed: {e}")
        return False
    
    finally:
        try:
            redis_service.close()
            print("🔌 Redis connection closed")
        except:
            pass
    
    return True

if __name__ == "__main__":
    test_redis_service()
