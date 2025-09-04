#!/usr/bin/env python3
"""
Results Analysis Script

This script analyzes results from hyperparameter tuning and extended training
to help select the best configuration for final training.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from scripts.results_analyzer import main

if __name__ == "__main__":
    main() 