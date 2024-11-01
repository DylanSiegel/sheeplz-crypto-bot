import os
import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Setup the development environment with optimized configurations."""
    # Create virtual environment
    subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    
    # Get the path to the virtual environment's Python interpreter
    if os.name == "nt":  # Windows
        python_path = Path(".venv/Scripts/python.exe")
        pip_path = Path(".venv/Scripts/pip.exe")
    else:  # Unix-like
        python_path = Path(".venv/bin/python")
        pip_path = Path(".venv/bin/pip")
    
    # Upgrade pip
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    
    # Install PyTorch with CUDA 11.8 support (optimized for NVIDIA 3070)
    subprocess.run([
        str(pip_path),
        "install",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        "https://download.pytorch.org/whl/cu118"
    ], check=True)
    
    # Install other requirements
    subprocess.run([
        str(pip_path),
        "install",
        "-r",
        "requirements.txt"
    ], check=True)
    
    # Create necessary directories
    directories = [
        "logs",
        "data",
        "models",
        "configs",
        "tests"
    ]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("Environment setup completed successfully!")
    print("\nActivate your virtual environment:")
    if os.name == "nt":
        print(".venv\\Scripts\\activate")
    else:
        print("source .venv/bin/activate")

if __name__ == "__main__":
    setup_environment()