#!/usr/bin/env python3
"""
Test script to demonstrate enhanced batch-level TensorBoard logging.

This script shows how the hyperparameter optimization now logs detailed batch-level metrics
within each trial, providing comprehensive monitoring of training progress.
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hyperparameter_tuning.config import HyperparameterConfig
from hyperparameter_tuning.optimization import HyperparameterOptimizer
from data.ticker_utils import get_large_cap_tickers

def test_batch_level_logging():
    """Test the enhanced batch-level TensorBoard logging."""
    
    print("🧪 Testing Enhanced Batch-Level TensorBoard Logging")
    print("=" * 60)
    
    # Create a minimal configuration for testing
    config = HyperparameterConfig(
        # Reduced trials for quick testing
        n_random_trials=2,
        n_bayesian_trials=2,
        n_fine_tune_trials=1,
        
        # Reduced epochs for quick testing
        epochs_per_trial_random=3,
        epochs_per_trial_bayesian=3,
        epochs_per_trial_fine_tune=3,
        
        # Other parameters
        data_period="1y",  # Shorter period for testing
        train_test_split_ratio=0.8,
        early_stopping_patience=2,
        early_stopping_min_delta=0.001,
        early_stopping_overfitting_threshold=0.15
    )
    
    # Get a small subset of tickers for testing
    all_tickers = get_large_cap_tickers()
    test_tickers = all_tickers[:10]  # Use only 10 tickers for quick testing
    
    print(f"📊 Configuration:")
    print(f"   - Total trials: {config.n_random_trials + config.n_bayesian_trials + config.n_fine_tune_trials}")
    print(f"   - Tickers: {len(test_tickers)}")
    print(f"   - Data period: {config.data_period}")
    print(f"   - Epochs per trial: {config.epochs_per_trial_random}")
    
    # Create optimizer
    save_dir = "./test_batch_logging_results"
    optimizer = HyperparameterOptimizer(config, test_tickers, save_dir=save_dir)
    
    print(f"\n🚀 Starting optimization with enhanced batch-level logging...")
    print(f"📁 Results will be saved to: {save_dir}")
    print(f"📊 TensorBoard logs will be in: {save_dir}/tensorboard_logs")
    
    try:
        # Run a small optimization
        best_params, best_value = optimizer.run_optimization()
        
        print(f"\n✅ Optimization completed!")
        print(f"🏆 Best accuracy: {best_value:.4f}")
        print(f"⚙️  Best parameters: {best_params}")
        
        # Print summary
        optimizer.print_summary()
        
        print(f"\n📊 TensorBoard Logging Summary:")
        print(f"   - Each trial creates its own log directory")
        print(f"   - Batch-level metrics are logged at each training step")
        print(f"   - Validation metrics are logged after each epoch")
        print(f"   - System metrics (CPU, RAM, GPU) are tracked")
        print(f"   - Progress tracking shows completion percentage")
        
        print(f"\n🔍 To view the logs, run:")
        print(f"   tensorboard --logdir {save_dir}/tensorboard_logs --port 6009")
        print(f"   Then open http://localhost:6009 in your browser")
        
        print(f"\n📈 Key metrics logged at batch level:")
        print(f"   - train/batch_loss: Loss for each batch")
        print(f"   - train/batch_time_seconds: Time per batch")
        print(f"   - system/cpu_utilization_percent: CPU usage")
        print(f"   - system/ram_usage_gb: Memory usage")
        print(f"   - progress/completion_percent: Overall progress")
        print(f"   - eval/batch_loss: Validation loss per batch")
        print(f"   - eval/batch_accuracy: Validation accuracy per batch")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_level_logging() 