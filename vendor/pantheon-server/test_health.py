import requests
import sys

print("🧪 TESTING REDIS HEALTH CHECK")
print("=" * 40)

try:
    # Test basic health endpoint
    print("1. Testing /health endpoint...")
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"   Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Response: {data}")
        print(f"   ✅ Health endpoint working!")
    else:
        print(f"   ❌ Health endpoint failed: {response.status_code}")
        
    # Test cache health endpoint  
    print("\n2. Testing /cache/health endpoint...")
    response = requests.get("http://localhost:8000/cache/health", timeout=5)
    print(f"   Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Redis Connected: {data.get('redis_connected', 'Unknown')}")
        redis_health = data.get('redis_health', {})
        print(f"   Redis Status: {redis_health.get('status', 'Unknown')}")
        print(f"   ✅ Cache health endpoint working!")
    else:
        print(f"   ❌ Cache health endpoint failed: {response.status_code}")
        
except Exception as e:
    print(f"❌ Error during testing: {e}")
    sys.exit(1)

print("\n✅ Test completed successfully!")
