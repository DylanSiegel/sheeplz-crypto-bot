#!/usr/bin/env python3
"""
Simple test for secure Redis configuration
"""

import requests
import time
import os

print('🧪 SECURE REDIS CACHE - SIMPLE TEST')
print('=' * 50)

# Check environment variable
redis_pass = os.getenv('PANTHEON_REDIS_PASSWORD')
print(f'Environment variable: {"✅ Set" if redis_pass else "❌ Not set"}')

try:
    # Test cache health
    print('\n1. Testing Redis connection...')
    health_response = requests.get('http://localhost:8000/cache/health', timeout=5)
    
    if health_response.status_code == 200:
        health = health_response.json()
        print(f'   Response: {health}')
        redis_connected = health.get('redis_connected', False)
        print(f'   ✅ Redis Connected: {redis_connected}')
        
        if redis_connected:
            print('\n2. Testing cache clear...')
            clear_response = requests.post('http://localhost:8000/cache/clear', timeout=5)
            if clear_response.status_code == 200:
                print('   ✅ Cache cleared successfully')
            else:
                print(f'   ❌ Cache clear failed: {clear_response.status_code}')
        else:
            print('   ❌ Redis not connected')
    else:
        print(f'   ❌ Health check failed: {health_response.status_code}')
        
except Exception as e:
    print(f'❌ Error: {e}')

print('\n' + '=' * 50)
print('Test complete!')
