#!/usr/bin/env python3
"""
Debug environment variable loading
"""

import os
from dotenv import load_dotenv

print("=== BEFORE load_dotenv() ===")
print(f"PANTHEON_REDIS_PASSWORD: {os.getenv('PANTHEON_REDIS_PASSWORD', 'NOT SET')}")
print(f"REDIS_PASSWORD: {os.getenv('REDIS_PASSWORD', 'NOT SET')}")

print("\n=== Calling load_dotenv() ===")
load_dotenv()

print("\n=== AFTER load_dotenv() ===")
print(f"PANTHEON_REDIS_PASSWORD: {os.getenv('PANTHEON_REDIS_PASSWORD', 'NOT SET')}")
print(f"REDIS_PASSWORD: {os.getenv('REDIS_PASSWORD', 'NOT SET')}")

# Test if we can access it directly from registry
try:
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        try:
            value, _ = winreg.QueryValueEx(key, "PANTHEON_REDIS_PASSWORD")
            print(f"\n=== Windows Registry (User) ===")
            print(f"PANTHEON_REDIS_PASSWORD: {value}")
        except FileNotFoundError:
            print("\n=== Windows Registry (User) ===")
            print("PANTHEON_REDIS_PASSWORD: NOT FOUND")
except Exception as e:
    print(f"Registry error: {e}")
