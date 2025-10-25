#!/usr/bin/env python3
"""
Phase 1 Cache Implementation Test

Test the Redis caching functionality integrated into the FastAPI endpoints.
"""

import asyncio
import httpx
import json
import time
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_PRODUCT_ID = "BTC-USD"

async def test_phase_1_caching():
    """Test Phase 1 cache implementation with actual API calls"""
    print("🧪 Testing Phase 1: Basic Cache with Manual Refresh")
    print("=" * 60)
    
    async with httpx.AsyncClient() as client:
        
        # Test 1: Health Check with Redis
        print("\n1. Testing health check with Redis status...")
        try:
            response = await client.get(f"{BASE_URL}/health")
            health = response.json()
            print(f"   Status: {health.get('status')}")
            print(f"   Redis: {health.get('redis_cache')}")
            print(f"   ✅ Health check passed" if health.get('redis_cache') == 'healthy' else f"   ⚠️ Redis status: {health.get('redis_cache')}")
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            return False
        
        # Test 2: Cache Health Endpoint
        print("\n2. Testing cache health endpoint...")
        try:
            response = await client.get(f"{BASE_URL}/cache/health")
            cache_health = response.json()
            if cache_health.get('success'):
                redis_status = cache_health['redis_health']['status']
                stats = cache_health['cache_statistics']
                print(f"   Redis Status: {redis_status}")
                print(f"   Total Keys: {stats.get('total_keys', 0)}")
                print(f"   Analysis Cache Keys: {stats.get('analysis_cache_keys', 0)}")
                print(f"   ✅ Cache health check passed")
            else:
                print(f"   ❌ Cache health check failed")
        except Exception as e:
            print(f"   ❌ Cache health failed: {e}")
        
        # Test 3: Clear all cache before testing
        print("\n3. Clearing all cache before testing...")
        try:
            response = await client.delete(f"{BASE_URL}/cache/all")
            result = response.json()
            deleted = result.get('deleted_breakdown', {}).get('total', 0)
            print(f"   Cleared {deleted} cache entries")
            print("   ✅ Cache cleared")
        except Exception as e:
            print(f"   ⚠️ Cache clear failed: {e}")
        
        # Test 4: First Analysis Call (Cache Miss)
        print(f"\n4. Testing first analysis call for {TEST_PRODUCT_ID} (expecting cache miss)...")
        start_time = time.time()
        try:
            analysis_request = {
                "product_id": TEST_PRODUCT_ID,
                "legend_type": "traditional",
                "timeframes": ["5m"],
                "max_candles": 200,
                "force_refresh": False
            }
            
            response = await client.post(f"{BASE_URL}/analyze", json=analysis_request)
            result = response.json()
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            if result.get('success'):
                cache_status = result.get('cache_status')
                data_freshness = result.get('data_freshness')
                print(f"   Cache Status: {cache_status}")
                print(f"   Data Freshness: {data_freshness}")
                print(f"   Response Time: {response_time:.2f}ms")
                print(f"   ✅ First analysis call successful")
                
                # Check if we got analysis results
                results = result.get('results', {})
                if '5m' in results and 'analysis' in results['5m']:
                    signal = results['5m']['analysis'].get('signal', 'N/A')
                    confidence = results['5m']['analysis'].get('confidence', 0)
                    print(f"   Analysis Result: {signal} (confidence: {confidence})")
                
            else:
                print(f"   ❌ First analysis failed: {result}")
                return False
        except Exception as e:
            print(f"   ❌ First analysis call failed: {e}")
            return False
        
        # Test 5: Second Analysis Call (Cache Hit)
        print(f"\n5. Testing second analysis call for {TEST_PRODUCT_ID} (expecting cache hit)...")
        start_time = time.time()
        try:
            response = await client.post(f"{BASE_URL}/analyze", json=analysis_request)
            result = response.json()
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            if result.get('success'):
                cache_status = result.get('cache_status')
                cache_age = result.get('cache_age_seconds', 0)
                data_freshness = result.get('data_freshness')
                print(f"   Cache Status: {cache_status}")
                print(f"   Cache Age: {cache_age} seconds")
                print(f"   Data Freshness: {data_freshness}")
                print(f"   Response Time: {response_time:.2f}ms")
                print(f"   ✅ Second analysis call successful")
                
                if cache_status == "hit":
                    print("   🎯 Cache working correctly - faster response expected")
                else:
                    print("   ⚠️ Expected cache hit, got cache miss")
            else:
                print(f"   ❌ Second analysis failed: {result}")
        except Exception as e:
            print(f"   ❌ Second analysis call failed: {e}")
        
        # Test 6: Force Refresh
        print(f"\n6. Testing force refresh for {TEST_PRODUCT_ID}...")
        start_time = time.time()
        try:
            analysis_request["force_refresh"] = True
            response = await client.post(f"{BASE_URL}/analyze", json=analysis_request)
            result = response.json()
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            if result.get('success'):
                cache_status = result.get('cache_status')
                data_freshness = result.get('data_freshness')
                print(f"   Cache Status: {cache_status}")
                print(f"   Data Freshness: {data_freshness}")
                print(f"   Response Time: {response_time:.2f}ms")
                print(f"   ✅ Force refresh successful")
                
                if cache_status == "refreshed":
                    print("   🔄 Force refresh working correctly")
                else:
                    print("   ⚠️ Expected 'refreshed' status")
            else:
                print(f"   ❌ Force refresh failed: {result}")
        except Exception as e:
            print(f"   ❌ Force refresh failed: {e}")
        
        # Test 7: Market Overview Caching
        print("\n7. Testing market overview caching...")
        start_time = time.time()
        try:
            response = await client.get(f"{BASE_URL}/overview?popular_only=true&legend_type=traditional")
            result = response.json()
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            if result.get('success'):
                cache_status = result.get('cache_status')
                data_freshness = result.get('data_freshness')
                print(f"   Cache Status: {cache_status}")
                print(f"   Data Freshness: {data_freshness}")
                print(f"   Response Time: {response_time:.2f}ms")
                print(f"   ✅ Market overview successful")
                
                # Check overview data
                overview = result.get('overview', {})
                pairs_analyzed = overview.get('pairs_analyzed', 0)
                print(f"   Pairs Analyzed: {pairs_analyzed}")
            else:
                print(f"   ❌ Market overview failed: {result}")
        except Exception as e:
            print(f"   ❌ Market overview failed: {e}")
        
        # Test 8: Overview Cache Hit
        print("\n8. Testing market overview cache hit...")
        start_time = time.time()
        try:
            response = await client.get(f"{BASE_URL}/overview?popular_only=true&legend_type=traditional")
            result = response.json()
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            if result.get('success'):
                cache_status = result.get('cache_status')
                cache_age = result.get('cache_age_seconds', 0)
                print(f"   Cache Status: {cache_status}")
                print(f"   Cache Age: {cache_age} seconds")
                print(f"   Response Time: {response_time:.2f}ms")
                print(f"   ✅ Overview cache test successful")
                
                if cache_status == "hit":
                    print("   🎯 Overview cache working correctly")
            else:
                print(f"   ❌ Overview cache test failed: {result}")
        except Exception as e:
            print(f"   ❌ Overview cache test failed: {e}")
        
        # Test 9: Manual Cache Clear
        print(f"\n9. Testing manual cache clear for {TEST_PRODUCT_ID}...")
        try:
            response = await client.delete(f"{BASE_URL}/cache/analysis/{TEST_PRODUCT_ID}")
            result = response.json()
            
            if result.get('success'):
                deleted_keys = result.get('deleted_keys', 0)
                print(f"   Deleted Keys: {deleted_keys}")
                print(f"   ✅ Manual cache clear successful")
            else:
                print(f"   ❌ Manual cache clear failed: {result}")
        except Exception as e:
            print(f"   ❌ Manual cache clear failed: {e}")
        
        # Test 10: Final Cache Statistics
        print("\n10. Final cache statistics...")
        try:
            response = await client.get(f"{BASE_URL}/cache/health")
            cache_health = response.json()
            
            if cache_health.get('success'):
                stats = cache_health['cache_statistics']
                print(f"    Total Keys: {stats.get('total_keys', 0)}")
                print(f"    Analysis Cache Keys: {stats.get('analysis_cache_keys', 0)}")
                print(f"    General Cache Keys: {stats.get('general_cache_keys', 0)}")
                print(f"    Memory Used: {stats.get('used_memory_human', 'N/A')}")
                print("    ✅ Final statistics retrieved")
            else:
                print("    ❌ Failed to get final statistics")
        except Exception as e:
            print(f"    ❌ Final statistics failed: {e}")
        
        print("\n🎉 Phase 1 Cache Testing Complete!")
        print("\n📊 Expected Results:")
        print("   - First analysis: cache_status='miss', slower response")
        print("   - Second analysis: cache_status='hit', faster response")
        print("   - Force refresh: cache_status='refreshed', fresh data")
        print("   - Overview caching: Working for expensive operations")
        print("   - Manual cache clear: Working for targeted cache invalidation")

async def main():
    print("🚀 Starting Phase 1 Cache Implementation Test")
    print("Make sure the pantheon-server is running on http://localhost:8000")
    print("And Redis is running on localhost:6379")
    
    # Check if server is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health", timeout=5.0)
            if response.status_code == 200:
                print("✅ Server is running")
                await test_phase_1_caching()
            else:
                print(f"❌ Server not responding properly: {response.status_code}")
    except Exception as e:
        print(f"❌ Could not connect to server: {e}")
        print("   Please start the server with: python run.py dev")

if __name__ == "__main__":
    asyncio.run(main())
