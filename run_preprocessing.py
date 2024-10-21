# run_preprocessing.py

import os
import sys
from datetime import datetime

# Ensure the src directory is in the PYTHONPATH
project_root = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.data_preprocessing.preprocess import main

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Starting data preprocessing at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    end_time = datetime.now()
    print(f"Data preprocessing completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {end_time - start_time}")
