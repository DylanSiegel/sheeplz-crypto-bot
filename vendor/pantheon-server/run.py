#!/usr/bin/env python
"""
Pantheon Server Startup Script

This script provides easy commands to start the Pantheon Server components:
- FastAPI backend server
- Streamlit web interface
- Both together in development mode
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def ensure_venv_activated():
    """Ensure we're running in the virtual environment"""
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # Not in a virtual environment
        venv_python = Path("venv/Scripts/python.exe")
        if venv_python.exists():
            print("🔄 Activating virtual environment...")
            return str(venv_python)
        else:
            print("❌ Virtual environment not found. Please run:")
            print("   python -m venv venv")
            print("   .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt")
            sys.exit(1)
    return sys.executable


def start_api():
    """Start the FastAPI server"""
    print("🚀 Starting Pantheon Server API...")
    
    python_exe = ensure_venv_activated()
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").absolute())
    
    # Ensure Redis password is available in subprocess
    redis_password = os.getenv("PANTHEON_REDIS_PASSWORD") or os.getenv("REDIS_PASSWORD")
    
    # If not found in environment, try Windows registry
    if not redis_password:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                redis_password, _ = winreg.QueryValueEx(key, "PANTHEON_REDIS_PASSWORD")
        except (ImportError, OSError, FileNotFoundError):
            pass
    
    if redis_password:
        env["PANTHEON_REDIS_PASSWORD"] = redis_password
        print("🔒 Redis password configured from environment")
    else:
        print("⚠️  Warning: No Redis password found in environment variables")
    
    cmd = [
        python_exe, "-m", "uvicorn",
        "pantheon_server.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    try:
        subprocess.run(cmd, env=env, cwd=os.getcwd())
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")


def start_ui():
    """Start the Streamlit interface"""
    print("🎨 Starting Pantheon Server UI...")
    
    python_exe = ensure_venv_activated()
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").absolute())
    
    cmd = [
        python_exe, "-m", "streamlit", "run",
        "src/pantheon_server/ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ]
    
    try:
        subprocess.run(cmd, env=env, cwd=os.getcwd())
    except KeyboardInterrupt:
        print("\n🛑 UI server stopped")


def start_dev():
    """Start both API and UI in development mode"""
    print("🏗️  Starting Pantheon Server in development mode...")
    print("📡 API will be available at: http://localhost:8000")
    print("🎨 UI will be available at: http://localhost:8501")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    print()
    
    import threading
    import time
    
    # Start API in a separate thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    # Give API time to start
    time.sleep(3)
    
    # Start UI in main thread
    start_ui()


def install_deps():
    """Install project dependencies"""
    print("📦 Installing Pantheon Server dependencies...")
    
    python_exe = ensure_venv_activated()
    
    cmd = [python_exe, "-m", "pip", "install", "-r", "requirements.txt"]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Dependencies installed successfully!")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def test_installation():
    """Test that all components can be imported"""
    print("🧪 Testing Pantheon Server installation...")
    
    python_exe = ensure_venv_activated()
    
    test_script = """
import sys
sys.path.insert(0, 'src')

try:
    import legends
    print("✅ pantheon-legends imported successfully")
except ImportError as e:
    print(f"❌ pantheon-legends import failed: {e}")
    sys.exit(1)

try:
    import fastapi
    print("✅ FastAPI imported successfully")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")
    sys.exit(1)

try:
    import streamlit
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")
    sys.exit(1)

try:
    from pantheon_server.services import CoinbaseService, PantheonMarketAnalyzer
    print("✅ Pantheon Server services imported successfully")
except ImportError as e:
    print(f"❌ Pantheon Server services import failed: {e}")
    sys.exit(1)

print("🎉 All components imported successfully!")
"""
    
    cmd = [python_exe, "-c", test_script]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Pantheon Server - Cryptocurrency Analysis Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py api          # Start only the FastAPI backend
  python run.py ui           # Start only the Streamlit UI
  python run.py dev          # Start both API and UI for development
  python run.py install     # Install project dependencies
  python run.py test        # Test installation
        """
    )
    
    parser.add_argument(
        "command",
        choices=["api", "ui", "dev", "install", "test"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    print("🏛️  Pantheon Server")
    print("   Legendary Cryptocurrency Analysis Platform")
    print()
    
    if args.command == "api":
        start_api()
    elif args.command == "ui":
        start_ui()
    elif args.command == "dev":
        start_dev()
    elif args.command == "install":
        if install_deps():
            print("\n🧪 Testing installation...")
            if test_installation():
                print("\n🎉 Pantheon Server is ready!")
                print("   Run 'python run.py dev' to start the development server")
            else:
                print("\n❌ Installation test failed")
        else:
            print("\n❌ Installation failed")
    elif args.command == "test":
        if test_installation():
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Tests failed")


if __name__ == "__main__":
    main()
