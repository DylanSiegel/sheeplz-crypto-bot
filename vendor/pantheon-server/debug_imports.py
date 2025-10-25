#!/usr/bin/env python3
"""
Debug script to isolate the import issue
"""

print("1. Starting debug script...")

try:
    print("2. Testing legends import...")
    import legends
    print("✅ legends imported successfully")
    
    print("3. Testing Pantheon creation...")
    pantheon = legends.Pantheon.create_default()
    print("✅ Pantheon created successfully")
    
    print("4. Getting available engines...")
    engines = pantheon.available_engines
    print(f"✅ Available engines: {engines}")
    
except Exception as e:
    print(f"❌ Error during legends test: {e}")
    import traceback
    traceback.print_exc()

print("5. Testing basic imports...")
try:
    import sys
    sys.path.insert(0, 'src')
    
    print("6. Testing coinbase service...")
    from pantheon_server.services.coinbase_service import CoinbaseService
    print("✅ CoinbaseService imported")
    
    print("7. Testing market analyzer...")
    # Don't create an instance, just import
    from pantheon_server.services.market_analyzer import PantheonMarketAnalyzer
    print("✅ PantheonMarketAnalyzer imported")
    
except Exception as e:
    print(f"❌ Error during import test: {e}")
    import traceback
    traceback.print_exc()

print("8. Debug script completed")
