#!/usr/bin/env python3
"""
Hyperparameter Tuning Entry Point

This is the main entry point for hyperparameter tuning.
Run this script to start hyperparameter optimization.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the hyperparameter tuning
from src.scripts.run_hyperparameter_tuning import main

if __name__ == "__main__":
    main() 