"""Standalone entry point for packaging the drowsiness warning system into an executable."""

import sys
from pathlib import Path

# Add repository root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.main import main

if __name__ == "__main__":
    main()

