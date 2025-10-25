#!/usr/bin/env python3
"""
Quick cache test to validate Redis caching functionality
"""

import requests
import time
import json

def test_quick_cache():
    """Test basic cache functionality with single analysis"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Quick Cache Test - Single Analysis")
    print("=" * 50)
    
    # Test data
    test_data = {
        "symbol": "BTC-USD",
        "timeframe": "5m",
        "legend_type": "traditional"
    }
    
    print(f"Testing with: {test_data}")
    print()
    
    try:
        # Step 1: Clear cache
        print("1. Clearing cache...")
        clear_response = requests.delete(f"{base_url}/cache/all")
        print(f"   Cache cleared: {clear_response.status_code == 200}")
        
        # Step 2: First request (cache miss)
        print("2. First request (should be cache miss)...")
        start_time = time.time()
        response1 = requests.post(f"{base_url}/analyze", json=test_data)
        time1 = (time.time() - start_time) * 1000
        
        if response1.status_code == 200:
            result1 = response1.json()
            cache_status1 = result1.get('cache_status', 'unknown')
            print(f"   ✅ Response time: {time1:.2f}ms")
            print(f"   📊 Cache status: {cache_status1}")
        else:
            print(f"   ❌ Error: {response1.status_code}")
            return False
        
        # Step 3: Second request (cache hit)
        print("3. Second request (should be cache hit)...")
        start_time = time.time()
        response2 = requests.post(f"{base_url}/analyze", json=test_data)
        time2 = (time.time() - start_time) * 1000
        
        if response2.status_code == 200:
            result2 = response2.json()
            cache_status2 = result2.get('cache_status', 'unknown')
            print(f"   ✅ Response time: {time2:.2f}ms")
            print(f"   📊 Cache status: {cache_status2}")
            
            # Calculate performance improvement
            if time1 > 0:
                improvement = ((time1 - time2) / time1) * 100
                print(f"   🚀 Performance improvement: {improvement:.1f}%")
        else:
            print(f"   ❌ Error: {response2.status_code}")
            return False
        
        # Step 4: Test cache health
        print("4. Testing cache health...")
        health_response = requests.get(f"{base_url}/cache/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"   ✅ Redis connected: {health_data.get('redis_connected')}")
            print(f"   📈 Cache stats: {health_data.get('stats', {})}")
        else:
            print(f"   ❌ Health check failed: {health_response.status_code}")
        
        print("\n🎉 Quick cache test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_quick_cache()
    if success:
        print("\n✅ All cache tests passed!")
    else:
        print("\n❌ Cache tests failed!")
