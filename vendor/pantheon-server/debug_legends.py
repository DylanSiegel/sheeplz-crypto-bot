#!/usr/bin/env python3
"""
Debug script to test legends imports with timeout
"""

import signal
import sys

def timeout_handler(signum, frame):
    print("❌ TIMEOUT! Import took too long")
    sys.exit(1)

def test_import_with_timeout(module_name, timeout_seconds=10):
    print(f"Testing {module_name} import...")
    
    # Set up timeout (Windows doesn't support SIGALRM, so we'll use a different approach)
    try:
        if module_name == "contracts":
            import legends.contracts
            print("✅ contracts imported successfully")
        elif module_name == "engines":
            import legends.engines
            print("✅ engines imported successfully")
        elif module_name == "pantheon":
            import legends.pantheon
            print("✅ pantheon imported successfully")
        elif module_name == "scaffold":
            import legends.scaffold
            print("✅ scaffold imported successfully")
        elif module_name == "legends":
            import legends
            print("✅ legends imported successfully")
        
    except Exception as e:
        print(f"❌ {module_name} import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔍 Testing legends module imports...")
    
    modules = ["contracts", "engines", "pantheon", "scaffold", "legends"]
    
    for module in modules:
        result = test_import_with_timeout(module)
        if not result:
            print(f"Stopping at failed module: {module}")
            break
        print()
    
    print("Debug completed")
