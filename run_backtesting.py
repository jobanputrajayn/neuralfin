#!/usr/bin/env python3
"""
Runner script for backtesting trained models.

Usage:
    # Auto-detect best model from hyperparameter tuning strategy
    python run_backtesting.py [options]
    
    # Use specific model
    python run_backtesting.py --model-path /path/to/model [options]
    
Examples:
    # Run backtesting with auto-detected best model
    python run_backtesting.py
    
    # Run backtesting with specific model
    python run_backtesting.py --model-path ./final_model
    
    # Generate latest signals
    python run_backtesting.py --generate-signals
    
    # Custom backtesting parameters
    python run_backtesting.py --tickers AAPL MSFT GOOGL --data-period 1y --initial-cash 50000
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from scripts.backtesting import main

if __name__ == "__main__":
    main() 