#!/usr/bin/env python3
"""
Streamlit UI Launcher

This script properly launches the Streamlit UI with correct module paths.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now import and run the main UI
from pantheon_server.ui.streamlit_app import main

if __name__ == "__main__":
    main()
