#!/usr/bin/env python3
"""
Main Training Entry Point

This is the main entry point for training the JAX GPT Stock Predictor.
Run this script to start the training pipeline.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main training
from src.scripts.main_training import main

if __name__ == "__main__":
    main() 