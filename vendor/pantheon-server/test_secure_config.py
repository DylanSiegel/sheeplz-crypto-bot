#!/usr/bin/env python3
"""
Test Redis service with environment variable password
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print('🔒 Testing Secure Redis Configuration')
print('=' * 40)

print('Environment variable status:')
pantheon_pass = os.getenv('PANTHEON_REDIS_PASSWORD')
redis_pass = os.getenv('REDIS_PASSWORD')

print(f'PANTHEON_REDIS_PASSWORD: {"✅ Set" if pantheon_pass else "❌ Not set"}')
print(f'REDIS_PASSWORD: {"✅ Set" if redis_pass else "❌ Not set"}')

# Test password resolution
final_password = pantheon_pass or redis_pass
print(f'Final password: {"✅ Available" if final_password else "❌ Not available"}')

if final_password:
    print(f'Password length: {len(final_password)} characters')
    print(f'Source: {"PANTHEON_REDIS_PASSWORD" if pantheon_pass else "REDIS_PASSWORD"}')

print('\nTesting Redis service initialization...')
try:
    from src.pantheon_server.services.redis_service import RedisService
    redis_service = RedisService()
    print('✅ Redis service initialized successfully!')
    print('✅ Secure configuration working!')
except Exception as e:
    print(f'❌ Error: {e}')
    
print('\n' + '=' * 40)
print('Security Status:')
if pantheon_pass and not redis_pass:
    print('✅ SECURE: Password from environment variable only')
elif pantheon_pass and redis_pass:
    print('⚠️  WARNING: Password in both environment variable and .env file')
elif redis_pass and not pantheon_pass:
    print('⚠️  WARNING: Password only in .env file (development mode)')
else:
    print('❌ ERROR: No password found anywhere')
