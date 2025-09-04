#!/usr/bin/env python3
"""
Final Model Training Script

This script trains the best configuration from hyperparameter tuning
on the full dataset for production use.
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from scripts.final_training import main

if __name__ == "__main__":
    main() 