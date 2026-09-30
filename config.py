"""
Global Configuration for EV Battery RUL Platform
"""
import os
import glob
from pathlib import Path
from typing import List

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Configurable Data Directory
# Priority: 1. Environment variable BATTERY_DATA_DIR
#           2. Default relative path: <PROJECT_ROOT>/data/raw
BATTERY_DATA_DIR_ENV = os.getenv("BATTERY_DATA_DIR")
if BATTERY_DATA_DIR_ENV:
    DATA_DIR = Path(BATTERY_DATA_DIR_ENV).resolve()
else:
    DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Artifact and Result Directories
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model & Training Hyperparameters
FEATURE_LEN = 200     # 1D signal resampled length
SEQ_LEN = 30          # Sliding window cycle sequence length
BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
LEARNING_RATE = 0.001

def get_battery_files() -> List[str]:
    """
    Locates matching NASA battery .mat files in DATA_DIR.
    Raises FileNotFoundError with actionable instructions if none are found.
    """
    pattern = str(DATA_DIR / "B*.mat")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No battery dataset files found matching '{pattern}'. "
            f"Please ensure NASA battery .mat files (e.g. B0005.mat) are located in '{DATA_DIR}', "
            f"or set the BATTERY_DATA_DIR environment variable. "
            f"See README.md for dataset download and extraction instructions."
        )
    return files
