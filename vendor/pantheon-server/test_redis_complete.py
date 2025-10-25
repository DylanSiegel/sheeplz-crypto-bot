#!/usr/bin/env python3
"""
Comprehensive Redis Cache Test Suite
Tests all aspects of the secure Redis implementation
"""

import requests
import time
import os
import sys

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print('='*60)

def print_test(test_name):
    """Print test name"""
    print(f"\n📋 {test_name}...")

def print_success(message):
    """Print success message"""
    print(f"   ✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"   ❌ {message}")

def print_info(message):
    """Print info message"""
    print(f"   📊 {message}")

def test_environment_variables():
    """Test environment variable configuration"""
    print_test("Testing Environment Variables")
    
    pantheon_pass = os.getenv('PANTHEON_REDIS_PASSWORD')
    redis_pass = os.getenv('REDIS_PASSWORD')
    
    # If not found in environment, try Windows registry (like the server does)
    if not pantheon_pass:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                pantheon_pass, _ = winreg.QueryValueEx(key, "PANTHEON_REDIS_PASSWORD")
                print_info("Retrieved password from Windows registry")
        except (ImportError, OSError, FileNotFoundError):
            pass
    
    if pantheon_pass:
        print_success(f"PANTHEON_REDIS_PASSWORD: Set ({len(pantheon_pass)} chars)")
    else:
        print_error("PANTHEON_REDIS_PASSWORD: Not set")
    
    if redis_pass:
        print_info(f"REDIS_PASSWORD: Set ({len(redis_pass)} chars)")
    else:
        print_info("REDIS_PASSWORD: Not set (expected in secure mode)")
    
    return bool(pantheon_pass or redis_pass)

def test_health_endpoint():
    """Test basic health endpoint"""
    print_test("Testing Health Endpoint")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'unknown')
            print_success(f"Health endpoint responding: {status}")
            return True
        else:
            print_error(f"Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health endpoint error: {e}")
        return False

def test_cache_health():
    """Test Redis cache health"""
    print_test("Testing Redis Cache Health")
    
    try:
        response = requests.get("http://localhost:8000/cache/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            redis_health = data.get('redis_health', {})
            stats = data.get('cache_statistics', {})
            
            status = redis_health.get('status', 'unknown')
            response_time = redis_health.get('response_time_ms', 0)
            memory = stats.get('used_memory_human', 'unknown')
            total_keys = stats.get('total_keys', 0)
            
            print_success(f"Redis status: {status}")
            print_info(f"Response time: {response_time}ms")
            print_info(f"Memory usage: {memory}")
            print_info(f"Total keys: {total_keys}")
            
            return status == 'healthy'
        else:
            print_error(f"Cache health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Cache health error: {e}")
        return False

def test_cache_miss():
    """Test cache miss (first request)"""
    print_test("Testing Cache Miss (Fresh Request)")
    
    try:
        # Clear cache first to ensure miss
        requests.delete("http://localhost:8000/cache/all", timeout=5)
        
        # Make analysis request
        data = {
            'product_id': 'BTC-USD',
            'timeframes': ['5m'],
            'legend_type': 'traditional'
        }
        
        start_time = time.time()
        response = requests.post("http://localhost:8000/analyze", json=data, timeout=30)
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            cache_status = result.get('cache_status', 'unknown')
            cache_age = result.get('cache_age_seconds', 0)
            
            print_success(f"Analysis completed in {elapsed:.2f}ms")
            print_info(f"Cache status: {cache_status}")
            print_info(f"Cache age: {cache_age} seconds")
            
            return cache_status == 'miss'
        else:
            print_error(f"Analysis failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Cache miss test error: {e}")
        return False

def test_cache_hit():
    """Test cache hit (second request)"""
    print_test("Testing Cache Hit (Repeat Request)")
    
    try:
        # Make same analysis request again
        data = {
            'product_id': 'BTC-USD',
            'timeframes': ['5m'],
            'legend_type': 'traditional'
        }
        
        start_time = time.time()
        response = requests.post("http://localhost:8000/analyze", json=data, timeout=30)
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            cache_status = result.get('cache_status', 'unknown')
            cache_age = result.get('cache_age_seconds', 0)
            
            print_success(f"Analysis completed in {elapsed:.2f}ms")
            print_info(f"Cache status: {cache_status}")
            print_info(f"Cache age: {cache_age} seconds")
            
            return cache_status == 'hit'
        else:
            print_error(f"Analysis failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Cache hit test error: {e}")
        return False

def test_cache_management():
    """Test cache management operations"""
    print_test("Testing Cache Management")
    
    try:
        # Test cache clear
        response = requests.delete("http://localhost:8000/cache/all", timeout=5)
        if response.status_code == 200:
            data = response.json()
            deleted = data.get('deleted_breakdown', {})
            total = deleted.get('total', 0)
            
            print_success(f"Cache cleared successfully")
            print_info(f"Keys deleted: {total}")
            return True
        else:
            print_error(f"Cache clear failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Cache management error: {e}")
        return False

def main():
    """Run all tests"""
    print_header("PANTHEON REDIS CACHE TEST SUITE")
    print("🔒 Testing secure Redis implementation")
    print("⚡ Testing cache performance")
    print("🛠️  Testing cache management")
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Health Endpoint", test_health_endpoint),
        ("Redis Cache Health", test_cache_health),
        ("Cache Miss", test_cache_miss),
        ("Cache Hit", test_cache_hit),
        ("Cache Management", test_cache_management),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_header("TEST RESULTS SUMMARY")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}: PASSED")
            passed += 1
        else:
            print_error(f"{test_name}: FAILED")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print_header("🎉 ALL TESTS PASSED - REDIS IMPLEMENTATION READY!")
        print("✅ Security: Environment variable authentication")
        print("✅ Performance: Cache miss/hit cycle working")
        print("✅ Health: Monitoring and statistics functional")
        print("✅ Management: Cache operations working")
        sys.exit(0)
    else:
        print_header("❌ SOME TESTS FAILED - CHECK CONFIGURATION")
        sys.exit(1)

if __name__ == "__main__":
    main()
