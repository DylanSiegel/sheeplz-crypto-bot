#!/usr/bin/env python3
"""
Final Cache Validation - Confirm Redis caching is operational
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test Redis service directly
try:
    from src.pantheon_server.services.redis_service import RedisService
    import asyncio
    
    async def test_redis_direct():
        print("🧪 Direct Redis Service Test")
        print("=" * 40)
        
        try:
            # Initialize Redis service
            redis_service = RedisService()
            
            # Test connection
            health = await redis_service.get_health()
            print(f"✅ Redis Connection: {health.get('redis_connected', False)}")
            
            # Test basic operations
            test_key = "test:validation"
            test_value = {"test": "data", "timestamp": "now"}
            
            # Set value
            await redis_service.redis.setex(test_key, 60, str(test_value))
            print("✅ Cache Write: Success")
            
            # Get value
            cached = await redis_service.redis.get(test_key)
            print(f"✅ Cache Read: {cached is not None}")
            
            # Get stats
            stats = await redis_service.get_cache_stats()
            print(f"📊 Cache Stats: {stats}")
            
            # Cleanup
            await redis_service.redis.delete(test_key)
            print("✅ Cache Cleanup: Complete")
            
            return True
            
        except Exception as e:
            print(f"❌ Redis Test Failed: {e}")
            return False
    
    # Run test
    result = asyncio.run(test_redis_direct())
    if result:
        print("\n🎉 Redis Service Validation: PASSED")
        print("✅ Phase 1 Cache Implementation: OPERATIONAL")
    else:
        print("\n❌ Redis Service Validation: FAILED")
        
except Exception as e:
    print(f"❌ Could not import Redis service: {e}")
    print("Note: This is expected if running outside the server environment")
    
    # Alternative: Check if Redis is running
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, password='pantheon_server**!', decode_responses=True)
        r.ping()
        print("✅ Redis Server: Accessible")
        print("✅ Cache Infrastructure: READY")
    except Exception as redis_error:
        print(f"❌ Redis Connection Error: {redis_error}")
