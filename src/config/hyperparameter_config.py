"""
Hyperparameter configuration for the Hyper framework.

Contains the main configuration class for hyperparameter tuning.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import jax
import jax.numpy as jnp
import numpy as np
import time
import gc
import traceback
import psutil
from flax.nnx import TrainState
import optax
import numbers

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
except Exception:
    NVML_AVAILABLE = False


def get_hyperparameter_config():
    """
    Get the hyperparameter configuration as a dictionary.
    
    Returns:
        dict: Configuration dictionary with all hyperparameter settings
    """
    config = HyperparameterConfig()
    return {
        'random_trials': config.n_random_trials,
        'bayesian_trials': config.n_bayesian_trials,
        'fine_tune_trials': config.n_fine_tune_trials,
        'epochs_per_trial_random': config.epochs_per_trial_random,
        'epochs_per_trial_bayesian': config.epochs_per_trial_bayesian,
        'epochs_per_trial_fine_tune': config.epochs_per_trial_fine_tune,
        'early_stopping_patience': config.early_stopping_patience,
        'early_stopping_min_delta': config.early_stopping_min_delta,
        'early_stopping_loss_patience': config.early_stopping_loss_patience,
        'early_stopping_overfitting_threshold': config.early_stopping_overfitting_threshold,
        'selection_weights': config.selection_weights,
        'min_accuracy_threshold': config.min_accuracy_threshold,
        'min_return_threshold': config.min_return_threshold,
        'max_trial_time_hours': config.max_trial_time_hours,
        'data_period': config.data_period,
        'train_test_split_ratio': config.train_test_split_ratio,
        'prefetch_buffer_size': config.prefetch_buffer_size,
        'search_space': config.search_space,
        'scaler_mean': config.scaler_mean,
        'scaler_std': config.scaler_std,
        'num_classes': config.num_classes,
        'enable_caching': config.enable_caching,
        'cache_size': config.cache_size,
        'adaptive_batch_size': config.adaptive_batch_size,
        'target_batch_time': config.target_batch_time,
        'jax_memory_fraction': config.jax_memory_fraction,
        'jax_recompilation_threshold': config.jax_recompilation_threshold,
        'jax_compile_warmup_batches': config.jax_compile_warmup_batches,
        'jax_memory_monitoring': config.jax_memory_monitoring,
        'min_throughput': config.min_throughput,
        'min_gpu_util': config.min_gpu_util,
        'min_cpu_util': config.min_cpu_util,
        'num_batches': config.num_batches,
        'ticker_count': config.ticker_count
    }


@dataclass
class HyperparameterConfig:
    """
    Configuration for hyperparameter tuning and search space.
    Edit the search_space dictionary to easily change ranges/types for each hyperparameter.
    """
    # Search phases - Optimized for faster exploration
    n_random_trials: int = 20  # Increased for better exploration
    n_bayesian_trials: int = 40  # Increased for better optimization
    n_fine_tune_trials: int = 15  # Increased for better refinement
    # Training parameters - Shorter epochs for faster exploration
    epochs_per_trial_random: int = 10  # Reduced from 10
    epochs_per_trial_bayesian: int = 20  # Reduced from 20
    epochs_per_trial_fine_tune: int = 30  # Reduced from 30
    # Early stopping - More aggressive for faster convergence
    early_stopping_patience: int = 3  # Reduced from 8
    early_stopping_min_delta: float = 0.002  # Increased from 0.001
    early_stopping_loss_patience: int = 8  # Reduced from 3
    early_stopping_overfitting_threshold: float = 0.12  # Reduced from 0.15
    # Model selection
    selection_weights: Optional[Dict[str, float]] = None
    min_accuracy_threshold: float = 0.4
    min_return_threshold: float = 0.02
    # Resource management
    max_trial_time_hours: float = 2.0
    # Data configuration
    data_period: str = "5y"
    train_test_split_ratio: float = 0.8
    prefetch_buffer_size: int = 100
    ticker_count: int = 50  # Number of tickers to use for training
    # Centralized hyperparameter search space
    search_space: Optional[dict] = None
    # New parameters for batch size sweep (sensible defaults)
    scaler_mean: float = 0.0
    scaler_std: float = 1.0
    num_classes: int = 3
    enable_caching: bool = True
    cache_size: int = 1000
    adaptive_batch_size: bool = False  # Disabled during hyperparameter tuning
    target_batch_time: float = 0.05
    jax_memory_fraction: float = 0.8
    jax_recompilation_threshold: int = 10
    jax_compile_warmup_batches: int = 3
    jax_memory_monitoring: bool = True
    min_throughput: float = 100  # samples/sec
    min_gpu_util: float = 10     # percentage
    min_cpu_util: float = 10     # percentage
    num_batches: int = 3

    def __post_init__(self):
        if self.selection_weights is None:
            self.selection_weights = {
                'accuracy': 0.4,
                'return': 0.4,
                'efficiency': 0.2
            }
        if self.search_space is None:
            self.search_space = {
                    # Continuous parameters (min, max, distribution_type)
                'learning_rate': (1e-5, 5e-4, 'log'),      # Wide range for learning rate
                'dropout_rate': (0.05, 0.4, 'uniform'),    # Regularization range
                'weight_alpha': (0.0, 1.0, 'uniform'),     # Loss weight for accuracy
                'weight_beta': (0.0, 1.0, 'uniform'),      # Loss weight for return
                
                # Discrete parameters (list of options)
                'time_window': [5, 7, 14, 21, 30],         # Options expiry windows
                'seq_length': [30, 60, 90, 120, 180],      # Sequence length (must be >= max(time_window))
                'num_layers': [2, 3, 4, 5, 6],                # Transformer depth
                'd_model': [64, 128, 256, 512],            # Model dimension
                'num_heads': [2, 4, 8, 16],                # Attention heads
                'd_ff': [256, 512, 1024, 2048],            # Feed-forward dimension
                'batch_size': [1, 2, 4],          # Batch sizes
                
                # Enhanced features (news and text data)
                'include_news': [True, False],                     # Include news features
                'include_text': [True, False],                     # Include text embeddings
                'news_window': [7,14],                         # News window size
            }