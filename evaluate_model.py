#!/usr/bin/env python3
"""
Model Evaluation Script

This script evaluates trained models with comprehensive metrics
and performance analysis.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from scripts.model_evaluator import main

if __name__ == "__main__":
    main() 