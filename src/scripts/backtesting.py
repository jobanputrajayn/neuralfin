"""
Backtesting script for the JAX GPT Stock Predictor.

This script provides a standalone backtesting interface for trained models.
It can be used to evaluate model performance on historical data and generate
comprehensive backtesting reports.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Set JAX memory fraction
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'

# --- TensorFlow CPU Restriction (MUST be before JAX import) ---
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    print("[INFO] TensorFlow configured to use CPU only.")
except Exception as e:
    print(f"[WARN] Could not configure TensorFlow device visibility: {e}")

# Import JAX for device information
import jax
import jax.numpy as jnp

# Import Orbax for checkpointing
import orbax.checkpoint as ocp

# Import Flax NNX for model operations
from flax import nnx

# Import Rich UI components for device dashboard
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.rule import Rule
from rich.columns import Columns
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.align import Align
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.gpt_classifier import GPTClassifier
from models.backtesting import (
    precompute_historical_signals, combine_equity_curves, GPTOptionsStrategy
)
from scripts.training_pipeline import perform_backtesting, aggregate_portfolio_performance
from data.stock_data import get_stock_data, get_large_cap_tickers
from training.checkpointing import restore_checkpoint
from config.hyperparameter_config import HyperparameterConfig
from utils.gpu_utils import check_gpu_availability, get_gpu_utilization, get_gpu_memory_info

console = Console()

def create_device_dashboard():
    """Create a rich dashboard showing JAX device information"""
    
    # Get JAX device information
    jax_devices = jax.devices()
    jax_device_count = jax.device_count()
    jax_platform = jax.default_backend()
    
    # Get GPU information
    gpu_info = check_gpu_availability()
    
    # Create device information table
    device_table = Table(title="🔧 JAX Device Configuration", show_header=True, header_style="bold magenta")
    device_table.add_column("Device Type", style="cyan", no_wrap=True)
    device_table.add_column("Device ID", style="green")
    device_table.add_column("Platform", style="yellow")
    device_table.add_column("Status", style="blue")
    
    # Add JAX devices
    for i, device in enumerate(jax_devices):
        device_type = "GPU" if "gpu" in str(device).lower() else "CPU" if "cpu" in str(device).lower() else "TPU" if "tpu" in str(device).lower() else "Unknown"
        device_table.add_row(
            device_type,
            str(i),
            str(device),
            "✅ Active"
        )
    
    # Create GPU status panel
    gpu_status_panel = Panel(
        f"NVML Available: {'✅' if gpu_info['nvml_available'] else '❌'}\n"
        f"GPU Count: {gpu_info['gpu_count']}\n"
        f"JAX Platform: {jax_platform}\n"
        f"JAX Device Count: {jax_device_count}",
        title="🚀 System Status",
        border_style="green"
    )
    
    # Create memory panel
    memory_panel = Panel(
        f"JAX Memory Fraction: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.8')}\n"
        f"Default Backend: {jax.default_backend()}\n"
        f"X64 Enabled: {jax.config.read('jax_enable_x64')}\n"
        f"Platform Name: {jax.config.read('jax_platform_name')}",
        title="🧠 Memory & Configuration",
        border_style="blue"
    )
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(Panel(device_table, border_style="magenta")),
        Layout(Columns([gpu_status_panel, memory_panel]))
    )
    
    return layout

class BacktestingEngine:
    """Standalone backtesting engine for trained models."""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the backtesting engine.
        
        Args:
            model_path: Path to the trained model checkpoint (if None, auto-detect best model)
            config_path: Path to model configuration file (optional)
        """
        # Display device dashboard
        console.print("\n" + "="*60)
        console.print(Panel.fit("[bold blue]Backtesting Engine - Device Configuration[/bold blue]", border_style="blue"))
        console.print("="*60)
        
        # Show device dashboard
        device_dashboard = create_device_dashboard()
        console.print(device_dashboard)
        
        if model_path is None:
            # Auto-detect best model from hyperparameter tuning strategy
            self.model_path = self._auto_detect_best_model()
            console.print(f"[cyan]🔍 Auto-detected best model: {self.model_path}[/cyan]")
        else:
            self.model_path = Path(model_path)
        
        self.config_path = Path(config_path) if config_path else None
        
        # Load model and configuration
        self.model, self.config = self._load_model_and_config()
        
        # Initialize GPU if available and display detailed GPU status
        gpu_info = check_gpu_availability()
        if gpu_info['gpu_count'] > 0:
            console.print(f"[green]✅ GPU Available: {gpu_info['gpu_names'][0] if gpu_info['gpu_names'] else 'Unknown GPU'}[/green]")
            
            # Show GPU utilization and memory if available
            if gpu_info['nvml_available'] and gpu_info['gpu_count'] > 0:
                for i in range(min(gpu_info['gpu_count'], 2)):  # Show first 2 GPUs
                    util = get_gpu_utilization(i)
                    mem_info = get_gpu_memory_info(i)
                    
                    if mem_info:
                        mem_used_gb = mem_info['used'] / (1024**3)
                        mem_total_gb = mem_info['total'] / (1024**3)
                        mem_percent = mem_info['utilization_percent']
                        console.print(f"   [dim]GPU {i}: {util:.1f}% util, {mem_used_gb:.1f}GB/{mem_total_gb:.1f}GB ({mem_percent:.1f}%)[/dim]")
        else:
            console.print("[yellow]⚠️  GPU not available, using CPU[/yellow]")
        
        console.print("="*60 + "\n")
    
    def _auto_detect_best_model(self) -> Path:
        """
        Automatically detect the best model from the hyperparameter tuning strategy.
        
        Priority order:
        1. Final model from final training
        2. Best model from extended training
        3. Best model from hyperparameter tuning
        4. Default fallback
        
        Returns:
            Path to the best model
        """
        console.print("[cyan]🔍 Auto-detecting best model from hyperparameter tuning strategy...[/cyan]")
        
        # Priority 1: Final model from final training
        console.print("[cyan]🔍 Checking final model directories...[/cyan]")

        # Check final_model/best_model (best checkpoint)
        best_model_dir = Path("./final_model/best_model")
        if best_model_dir.exists():
            checkpoint_steps = [d for d in best_model_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if checkpoint_steps:
                latest_step = max(checkpoint_steps, key=lambda x: int(x.name))
                console.print(f"[green]✅ Found best model checkpoint: {latest_step} (step {latest_step.name})[/green]")
                return latest_step
        
        # Check final_model/final_model first (latest checkpoint)
        final_model_dir = Path("./final_model/final_model")
        if final_model_dir.exists():
            # Find the latest checkpoint step
            checkpoint_steps = [d for d in final_model_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if checkpoint_steps:
                latest_step = max(checkpoint_steps, key=lambda x: int(x.name))
                console.print(f"[green]✅ Found final model checkpoint: {latest_step} (step {latest_step.name})[/green]")
                return latest_step
        
        
        
        # Check final_model root directory (fallback)
        final_model_root = Path("./final_model")
        if final_model_root.exists() and self._is_valid_model_path(final_model_root):
            console.print(f"[yellow]⚠️  Using fallback model: {final_model_root}[/yellow]")
            return final_model_root
        
        # Priority 2: Extended training results
        extended_results_path = Path("./extended_training_results/extended_training_results.json")
        if extended_results_path.exists():
            try:
                with open(extended_results_path, 'r') as f:
                    extended_results = json.load(f)
                
                if extended_results:
                    # Find best configuration from extended training
                    best_idx = 0
                    best_accuracy = 0
                    for i, result in enumerate(extended_results):
                        if result['avg_accuracy'] > best_accuracy:
                            best_accuracy = result['avg_accuracy']
                            best_idx = i
                    
                    # Check if there's a saved model for this configuration
                    extended_model_path = Path(f"./extended_training_results/extended_model_{best_idx}")
                    if extended_model_path.exists() and self._is_valid_model_path(extended_model_path):
                        console.print(f"[green]✅ Found best extended training model: {extended_model_path}[/green]")
                        return extended_model_path
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not load extended training results: {e}[/yellow]")
        
        # Priority 3: Best model from hyperparameter tuning
        tuning_results_path = Path("./hyperparameter_tuning_results")
        if tuning_results_path.exists():
            # Look for best trial checkpoint
            checkpoints_dir = tuning_results_path / "checkpoints"
            if checkpoints_dir.exists():
                # Find the trial with highest accuracy
                study_file = tuning_results_path / "optimization_study.pkl"
                if study_file.exists():
                    try:
                        import optuna
                        with open(study_file, 'rb') as f:
                            study = optuna.load_study(study_file=f)
                        
                        best_trial = study.best_trial
                        best_trial_path = checkpoints_dir / str(best_trial.number)
                        
                        if best_trial_path.exists() and self._is_valid_model_path(best_trial_path):
                            console.print(f"[green]✅ Found best hyperparameter tuning model: {best_trial_path}[/green]")
                            return best_trial_path
                    except Exception as e:
                        console.print(f"[yellow]⚠️  Could not load optimization study: {e}[/yellow]")
        
        # Priority 4: Default fallback
        default_paths = [
            Path("./jax_gpt_stock_predictor_checkpoints"),
            Path("./hyperparameter_tuning_results/checkpoints/5"),  # Common best trial
            Path("./hyperparameter_tuning_results/checkpoints/4"),
            Path("./hyperparameter_tuning_results/checkpoints/3")
        ]
        
        for path in default_paths:
            if path.exists() and self._is_valid_model_path(path):
                console.print(f"[yellow]⚠️  Using fallback model: {path}[/yellow]")
                return path
        
        # If no model found, raise error
        raise FileNotFoundError(
            "No valid model found. Please ensure you have completed the hyperparameter tuning strategy:\n"
            "1. Run hyperparameter tuning: python run_hyperparameter_tuning.py\n"
            "2. Run extended training: python run_extended_training.py\n"
            "3. Run final training: python run_final_training.py\n"
            "Or specify a model path manually with --model-path"
        )
    
    def _is_valid_model_path(self, path: Path) -> bool:
        """Check if a path contains a valid model checkpoint."""
        # First check if this is a directory with checkpoint subdirectories
        if path.is_dir():
            # Look for checkpoint step directories (numbers)
            checkpoint_steps = [d for d in path.iterdir() if d.is_dir() and d.name.isdigit()]
            if checkpoint_steps:
                # Check if any of the checkpoint steps contain actual checkpoint data
                for step_dir in checkpoint_steps:
                    if self._has_checkpoint_data(step_dir):
                        return True
        
        # Check for common checkpoint files in the path itself
        return self._has_checkpoint_data(path)
    
    def _has_checkpoint_data(self, path: Path) -> bool:
        """Check if a path contains actual checkpoint data files."""
        # Check for Orbax checkpoint files
        checkpoint_indicators = [
            path / "_CHECKPOINT_METADATA",
            path / "state",
            path / "state" / "params",
            path / "state" / "opt_state"
        ]
        
        # Also check for config files as secondary indicator
        config_indicators = [
            path / "config.json",
            path / "training_results.json"
        ]
        
        # Must have at least one checkpoint indicator
        has_checkpoint = any(indicator.exists() for indicator in checkpoint_indicators)
        
        # Should also have config files
        has_config = any(indicator.exists() for indicator in config_indicators)
        
        return has_checkpoint and has_config
    
    def _load_model_and_config(self) -> tuple:
        """Load the trained model and configuration."""
        console.print(f"[cyan]📂 Loading model from: {self.model_path}[/cyan]")
        
        # Load configuration
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            # Try to load from model directory or parent directories
            config_file = None
            
            # If we're in a checkpoint step directory, look in parent directories
            if self.model_path.name.isdigit():
                # Look in immediate parent first
                config_file = self.model_path.parent / "config.json"
                if not config_file.exists():
                    # Look in grandparent (final_model root)
                    config_file = self.model_path.parent.parent / "config.json"
            else:
                # Direct model directory
                config_file = self.model_path / "config.json"
            
            if config_file and config_file.exists():
                console.print(f"[cyan]📄 Loading config from: {config_file}[/cyan]")
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                console.print(f"[yellow]⚠️  No config file found, using default configuration[/yellow]")
                # Use default configuration
                config = self._get_default_config()
        
        # Create model instance with exact configuration from checkpoint
        model = GPTClassifier(
            num_classes=config.get('num_classes', 3),
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 4),
            d_ff=config.get('d_ff', 1024),
            dropout_rate=config.get('dropout_rate', 0.1),
            input_features=len(config.get('tickers', [])),  # Use exact number of tickers from config
            num_tickers=len(config.get('tickers', []))
        )
        
        console.print(f"[cyan]📊 Model configuration:[/cyan]")
        console.print(f"   [dim]Input features: {len(config.get('tickers', []))} (number of tickers)[/dim]")
        console.print(f"   [dim]Sequence length: {config.get('seq_length', 60)}[/dim]")
        console.print(f"   [dim]Num layers: {config.get('num_layers', 4)}[/dim]")
        console.print(f"   [dim]D model: {config.get('d_model', 256)}[/dim]")
        console.print(f"   [dim]Num heads: {config.get('num_heads', 8)}[/dim]")
        console.print(f"   [dim]D FF: {config.get('d_ff', 1024)}[/dim]")
        console.print(f"   [dim]Dropout rate: {config.get('dropout_rate', 0.1)}[/dim]")
        
        # Create optimizer for checkpoint restoration
        import optax
        tx = optax.adam(config.get('learning_rate', 1e-4))
        
        # Create checkpoint manager
        # For Orbax, the checkpoint manager should point to the parent directory
        # and the step will be specified during restoration
        checkpoint_parent_dir = self.model_path.parent if self.model_path.name.isdigit() else self.model_path
        step_number = int(self.model_path.name) if self.model_path.name.isdigit() else None
        
        console.print(f"[cyan]📂 Checkpoint parent directory: {checkpoint_parent_dir}[/cyan]")
        if step_number:
            console.print(f"[cyan]📂 Checkpoint step: {step_number}[/cyan]")
        
        ckpt_mgr = ocp.CheckpointManager(
            str(checkpoint_parent_dir.resolve()),
            options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
        )
        
        # Define dummy input shape for checkpoint restoration
        seq_length = config.get('seq_length', 60)  # Use exact sequence length from config
        num_tickers = len(config.get('tickers', []))  # Use exact number of tickers from config
        dummy_input_shape = (1, seq_length, num_tickers)  # (batch_size, seq_length, num_features)
        
        console.print(f"[cyan]📐 Dummy input shape: {dummy_input_shape}[/cyan]")
        
        # Load model state
        try:
            # Create dummy optimizer for tx_def parameter (not used in function but required by signature)
            import optax
            tx_def = optax.adam(config.get('learning_rate', 1e-4))
            
            # Get checkpoint directory from manager
            checkpoint_dir = Path(ckpt_mgr.directory)
            
            if step_number is not None:
                # Restore from specific step
                params, state, scaler_mean, scaler_std, step, _, _ = restore_checkpoint(
                    checkpoint_dir, model, step=step_number
                )
            else:
                # Restore latest step
                params, state, scaler_mean, scaler_std, step, _, _ = restore_checkpoint(
                    checkpoint_dir, model
                )
            
            # Merge params and state to create restored model
            if params is not None:
                if state is not None:
                    restored_model = nnx.merge(params, state)
                else:
                    # If no state, merge with original model's state
                    _, original_state = nnx.split(model)
                    restored_model = nnx.merge(params, original_state)
            else:
                restored_model = None
            
            # Check if state is None
            if restored_model is None:
                console.print("[red]❌ Model state is None after checkpoint restoration[/red]")
                console.print("[red]   This usually means no valid checkpoints were found[/red]")
                console.print(f"[red]   Checkpoint directory: {self.model_path}[/red]")
                
                # List available checkpoints
                try:
                    steps = ckpt_mgr.all_steps()
                    console.print(f"[yellow]Available checkpoint steps: {steps}[/yellow]")
                    if steps:
                        console.print(f"[yellow]Latest step: {max(steps)}[/yellow]")
                    else:
                        console.print("[red]No checkpoint steps found![/red]")
                except Exception as e:
                    console.print(f"[red]Error listing checkpoints: {e}[/red]")
                
                raise ValueError("Model state is None - no valid checkpoints found")
            
            # Verify that restored model has correct hyperparameters from config
            console.print(f"[cyan]🔍 Verifying model hyperparameters...[/cyan]")
            
            # Extract hyperparameters from restored model
            restored_params, _ = nnx.split(restored_model)
            
            # Get expected hyperparameters from config
            expected_num_layers = config.get('num_layers', 4)
            expected_d_model = config.get('d_model', 256)
            expected_num_heads = config.get('num_heads', 8)
            expected_d_ff = config.get('d_ff', 1024)
            expected_num_classes = config.get('num_classes', 3)
            expected_dropout_rate = config.get('dropout_rate', 0.1)
            expected_input_features = len(config.get('tickers', []))
            
            # Verify model architecture matches config by checking actual parameters
            try:
                # Use NNX's tree_flatten to access parameters safely
                import jax.tree_util as tree_util
                
                # Flatten the parameter tree to get all parameters
                flat_params, tree_def = tree_util.tree_flatten(restored_params)
                
                # Get expected hyperparameters from config
                console.print(f"[cyan]🔍 Checking model parameters...[/cyan]")
                
                # For now, let's do a simpler verification by checking the model can be created with the expected config
                # and that the restored model has the right structure
                
                # Create a test model with the expected config to compare structures
                test_model = GPTClassifier(
                    num_classes=expected_num_classes,
                    d_model=expected_d_model,
                    num_heads=expected_num_heads,
                    num_layers=expected_num_layers,
                    d_ff=expected_d_ff,
                    dropout_rate=expected_dropout_rate,
                    input_features=expected_input_features,
                    num_tickers=len(config.get('tickers', []))
                )
                
                # Split both models to compare their parameter structures
                test_params, _ = nnx.split(test_model)
                test_flat_params, test_tree_def = tree_util.tree_flatten(test_params)
                
                # Compare the number of parameters (should be the same for same architecture)
                assert len(flat_params) == len(test_flat_params), (
                    f"Parameter count mismatch! Expected {len(test_flat_params)} parameters, got {len(flat_params)}. "
                    f"This suggests the model architecture doesn't match the config."
                )
                
                # Compare parameter shapes
                for i, (restored_param, test_param) in enumerate(zip(flat_params, test_flat_params)):
                    if hasattr(restored_param, 'shape') and hasattr(test_param, 'shape'):
                        assert restored_param.shape == test_param.shape, (
                            f"Parameter shape mismatch at index {i}! Expected {test_param.shape}, got {restored_param.shape}"
                        )
                
                console.print(f"[green]✅ Model hyperparameters verified![/green]")
                console.print(f"[dim]   Parameter count: {len(flat_params)}[/dim]")
                console.print(f"[dim]   All parameter shapes match expected configuration[/dim]")
                
            except Exception as e:
                console.print(f"[red]❌ Model hyperparameter verification failed: {e}[/red]")
                console.print(f"[red]   Expected config: layers={expected_num_layers}, d_model={expected_d_model}, "
                            f"heads={expected_num_heads}, d_ff={expected_d_ff}, classes={expected_num_classes}, "
                            f"dropout={expected_dropout_rate}, input_features={expected_input_features}[/red]")
                console.print(f"[red]   This indicates a configuration mismatch between the checkpoint and the loaded config.[/red]")
                raise ValueError(f"Model hyperparameter verification failed: {e}")
            
            console.print(f"[green]✅ Model loaded successfully![/green]")
            console.print(f"[dim]   Checkpoint step: {step}[/dim]")
            console.print(f"[dim]   Scaler mean: {scaler_mean}[/dim]")
            console.print(f"[dim]   Scaler std: {scaler_std}[/dim]")
            
        except Exception as e:
            console.print(f"[red]❌ Error loading model: {e}[/red]")
            console.print(f"[red]   Model path: {self.model_path}[/red]")
            console.print(f"[red]   Dummy input shape: {dummy_input_shape}[/red]")
            raise
        
        # Log configuration details for transparency
        console.print(f"[dim]📋 Configuration loaded:[/dim]")
        console.print(f"   [dim]Training Batch Size: {config.get('batch_size', 'N/A')} (used during training)[/dim]")
        console.print(f"   [dim]Inference Batch Size: 64 (hardcoded for JAX optimization)[/dim]")
        console.print(f"   [dim]Latest Signals Batch Size: 1 (single prediction)[/dim]")
        console.print(f"   [dim]Sequence Length: {config.get('seq_length', 'N/A')}[/dim]")
        console.print(f"   [dim]Time Window: {config.get('time_window', 'N/A')}[/dim]")
        console.print(f"   [dim]Learning Rate: {config.get('learning_rate', 'N/A')}[/dim]")
        
        return restored_model, config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for backtesting."""
        return {
            'tickers': get_large_cap_tickers()[:50],  # Use first 50 tickers
            'data_period': '2y',
            'seq_length': 60,
            'time_window': 14,
            'num_classes': 3,
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4,
            'd_ff': 1024,
            'dropout_rate': 0.1,
            'batch_size': 64,  # Default batch size (for reference only - not used in inference)
            'learning_rate': 1e-4
        }
    
    def _compute_backtest_period(self, data_period: str) -> str:
        """
        Compute appropriate backtest period based on sequence length to ensure sufficient data.
        Uses sequence length to determine minimum required data period.
        
        Args:
            data_period: Original data period
            
        Returns:
            Computed backtest period
        """
        # Get sequence length from configuration
        seq_length = self.config.get('seq_length', 60)
        console.print(f"[cyan]📏 Model sequence length: {seq_length} days[/cyan]")
        
        # Calculate minimum required days for backtesting
        # We need at least 3x sequence length to have meaningful backtesting
        min_required_days = seq_length * 6
        console.print(f"[cyan]📊 Minimum required days for backtesting: {min_required_days}[/cyan]")
        
        # Map minimum days to yfinance periods
        if min_required_days >= 365:
            required_period = '1y'
        elif min_required_days >= 180:
            required_period = '6mo'
        elif min_required_days >= 90:
            required_period = '3mo'
        elif min_required_days >= 60:
            required_period = '2mo'
        elif min_required_days >= 30:
            required_period = '1mo'
        else:
            required_period = '1mo'  # Minimum period
        
        console.print(f"[cyan]🔄 Required period for {min_required_days} days: {required_period}[/cyan]")
        
        # Try to get split date from configuration for post-split backtesting
        split_date = self.config.get('split_date', None)
        if split_date:
            try:
                from datetime import datetime
                split_date_obj = datetime.fromisoformat(split_date.replace('Z', '+00:00'))
                current_date = datetime.now()
                days_diff = (current_date - split_date_obj).days
                
                console.print(f"[cyan]📅 Split date: {split_date}, days since split: {days_diff}[/cyan]")
                
                if days_diff >= min_required_days:
                    # We have enough post-split data, use it
                    console.print(f"[green]✅ Sufficient post-split data available ({days_diff} days)[/green]")
                    
                    # Map days to period
                    if days_diff >= 365:
                        return '1y'
                    elif days_diff >= 180:
                        return '6mo'
                    elif days_diff >= 90:
                        return '3mo'
                    elif days_diff >= 60:
                        return '2mo'
                    elif days_diff >= 30:
                        return '1mo'
                    else:
                        return '1mo'
                else:
                    console.print(f"[yellow]⚠️  Insufficient post-split data ({days_diff} days < {min_required_days} required)[/yellow]")
                    console.print(f"[yellow]   Using sequence-length-based period: {required_period}[/yellow]")
                    return required_period
                    
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not parse split date: {e}[/yellow]")
                console.print(f"[yellow]   Using sequence-length-based period: {required_period}[/yellow]")
                return required_period
        
        # No split date or insufficient post-split data, use sequence-length-based period
        console.print(f"[cyan]📈 Using sequence-length-based period: {required_period}[/cyan]")
        return required_period
    
    def run_backtest(self, 
                    tickers: Optional[List[str]] = None,
                    data_period: str = '2y',
                    initial_cash: float = 10000,
                    commission_rate: float = 0.001,
                    plot_results: bool = True,
                    output_dir: str = './backtesting_results') -> Dict[str, Any]:
        """
        Run comprehensive backtesting on the trained model.
        
        Args:
            tickers: List of tickers to backtest (if None, uses config tickers)
            data_period: Data period for backtesting
            initial_cash: Initial cash per ticker
            commission_rate: Commission rate for trades
            plot_results: Whether to generate plots
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing backtesting results
        """
        # CRITICAL: Always use the exact same tickers in the exact same order as training
        training_tickers = self.config.get('tickers', [])
        if not training_tickers:
            raise ValueError("No tickers found in model configuration. Model may be corrupted.")
        
        if tickers is not None:
            # Validate that provided tickers match training tickers exactly
            if len(tickers) != len(training_tickers):
                raise ValueError(
                    f"Ticker count mismatch! Model was trained on {len(training_tickers)} tickers, "
                    f"but {len(tickers)} tickers provided. Model is sensitive to ticker sequence."
                )
            
            # Check if tickers match exactly (order matters!)
            if tickers != training_tickers:
                console.print(f"[yellow]⚠️  Ticker order mismatch detected![/yellow]")
                console.print(f"[yellow]   Training tickers: {training_tickers[:5]}...[/yellow]")
                console.print(f"[yellow]   Provided tickers: {tickers[:5]}...[/yellow]")
                console.print(f"[yellow]   Using training tickers to ensure model compatibility.[/yellow]")
                tickers = training_tickers
        else:
            # Use training tickers by default
            tickers = training_tickers
        
        console.print(f"[green]✅ Using training tickers: {len(tickers)} tickers in exact training order[/green]")
        
        # Check for split date in configuration for post-split backtesting
        split_date = self.config.get('split_date', None)
        if split_date:
            console.print(f"[cyan]📅 Found split date: {split_date} - Using post-split data for backtesting[/cyan]")
        else:
            console.print(f"[yellow]⚠️  No split date found in configuration - Using computed period[/yellow]")
        
        # Compute appropriate backtest period
        computed_period = self._compute_backtest_period(data_period)
        if computed_period != data_period:
            console.print(f"[cyan]🔄 Computed backtest period: {computed_period} (from {data_period})[/cyan]")
            data_period = computed_period
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(Panel.fit(
            f"[bold green]Backtesting Configuration[/bold green]\n"
            f"Tickers: {len(tickers)} tickers (exact training order)\n"
            f"Data Period: {data_period}\n"
            f"Split Date: {split_date if split_date else 'Not specified'}\n"
            f"Initial Cash: ${initial_cash:,.0f}\n"
            f"Commission Rate: {commission_rate:.3f}",
            title="🚀 Backtesting Engine"
        ))
        
        # Download data
        console.print(f"\n[cyan]📊 Downloading data for {len(tickers)} tickers...[/cyan]")
        all_tickers_data = get_stock_data(tickers, period=data_period)
        
        if all_tickers_data.empty:
            raise ValueError("No data downloaded. Check ticker symbols and data period.")
        
        console.print(f"[green]✅ Downloaded {len(all_tickers_data)} days of data[/green]")
        
        # Precompute historical signals
        console.print(f"\n[cyan]🧠 Precomputing historical signals...[/cyan]")
        historical_signals = precompute_historical_signals(
            all_tickers_data, 
            self.model,
            self.config.get('scaler_mean', 0.0),
            self.config.get('scaler_std', 1.0),
            self.config.get('seq_length', 60),
            len(tickers),
            backtest_start_date=split_date,  # Use split date from configuration
            include_news=self.config.get('include_news', False),
            include_text=self.config.get('include_text', False),
            news_window=self.config.get('news_window', 7),
            text_model=self.config.get('text_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            tickers=tickers,
            time_window=self.config.get('time_window', 1),
            batch_size=self.config.get('batch_size', 1)
        )
        
        if historical_signals.empty:
            raise ValueError("No historical signals generated. Check model and data.")
        
        console.print(f"[green]✅ Generated signals for {len(historical_signals)} predictions[/green]")
        
        # Perform backtesting
        console.print(f"\n[cyan]📈 Running backtests...[/cyan]")
        equity_curves = perform_backtesting(
            historical_signals,
            tickers,
            all_tickers_data,
            self.config.get('seq_length', 60),
            self.config.get('time_window', 14),
            initial_cash,
            commission_rate,
            str(output_dir / "backtest_report.txt")
        )
        
        # Aggregate portfolio performance
        console.print(f"\n[cyan]📊 Aggregating portfolio performance...[/cyan]")
        portfolio_results = aggregate_portfolio_performance(
            equity_curves,
            initial_cash,
            plot_results,
            tickers,
            str(output_dir / "portfolio_report.txt")
        )
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'backtest_params': {
                'tickers': tickers,
                'data_period': data_period,
                'split_date': split_date,
                'initial_cash': initial_cash,
                'commission_rate': commission_rate
            },
            'equity_curves': equity_curves,
            'portfolio_results': portfolio_results,
            'historical_signals_count': len(historical_signals)
        }
        
        # Save to JSON
        results_file = output_dir / "backtest_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"[green]✅ Results saved to: {results_file}[/green]")
        
        return results
    
    def generate_signals(self, 
                        tickers: Optional[List[str]] = None,
                        data_period: str = '1mo',
                        output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Generate latest trading signals for the given tickers.
        
        Args:
            tickers: List of tickers to generate signals for
            data_period: Data period for signal generation
            output_file: File to save signals (optional)
            
        Returns:
            DataFrame with latest signals
        """
        # CRITICAL: Always use the exact same tickers in the exact same order as training
        training_tickers = self.config.get('tickers', [])
        if not training_tickers:
            raise ValueError("No tickers found in model configuration. Model may be corrupted.")
        
        if tickers is not None:
            # Validate that provided tickers match training tickers exactly
            if len(tickers) != len(training_tickers):
                raise ValueError(
                    f"Ticker count mismatch! Model was trained on {len(training_tickers)} tickers, "
                    f"but {len(tickers)} tickers provided. Model is sensitive to ticker sequence."
                )
            
            # Check if tickers match exactly (order matters!)
            if tickers != training_tickers:
                console.print(f"[yellow]⚠️  Ticker order mismatch detected![/yellow]")
                console.print(f"[yellow]   Training tickers: {training_tickers[:5]}...[/yellow]")
                console.print(f"[yellow]   Provided tickers: {tickers[:5]}...[/yellow]")
                console.print(f"[yellow]   Using training tickers to ensure model compatibility.[/yellow]")
                tickers = training_tickers
        else:
            # Use training tickers by default
            tickers = training_tickers
        
        console.print(f"[green]✅ Using training tickers: {len(tickers)} tickers in exact training order[/green]")
        
        # Compute appropriate data period for signal generation
        # We need enough data for the sequence length plus some buffer
        seq_length = self.config.get('seq_length', 60)
        min_required_days = seq_length + 30  # Add 30 days buffer for safety
        
        # Map required days to yfinance periods
        if min_required_days >= 365:
            required_period = '1y'
        elif min_required_days >= 180:
            required_period = '6mo'
        elif min_required_days >= 90:
            required_period = '3mo'
        elif min_required_days >= 60:
            required_period = '2mo'
        else:
            required_period = '1mo'
        
        if required_period != data_period:
            console.print(f"[cyan]🔄 Adjusting data period for signal generation: {required_period} (from {data_period})[/cyan]")
            console.print(f"[cyan]   Need at least {min_required_days} days for sequence length {seq_length}[/cyan]")
            data_period = required_period
        
        console.print(f"[cyan]📊 Generating latest signals for {len(tickers)} tickers...[/cyan]")
        
        # Download data with sufficient history - implement computed fallback strategy
        # Compute fallback periods based on sequence length requirements
        fallback_periods = []
        
        # Start with the computed required period
        fallback_periods.append(data_period)
        
        # Add computed fallbacks based on sequence length
        if seq_length > 180:
            fallback_periods.extend(['6mo', '1y', '2y'])
        elif seq_length > 90:
            fallback_periods.extend(['6mo', '1y'])
        elif seq_length > 60:
            fallback_periods.extend(['3mo', '6mo'])
        else:
            fallback_periods.extend(['2mo', '3mo'])
        
        # Remove duplicates while preserving order
        periods_to_try = list(dict.fromkeys(fallback_periods))
        
        all_tickers_data = None
        actual_period_used = None
        
        for attempt, period in enumerate(periods_to_try):
            console.print(f"[cyan]📥 Attempt {attempt + 1}: Downloading data for period {period}...[/cyan]")
            
            all_tickers_data = get_stock_data(tickers, period=period)
            
            if all_tickers_data.empty:
                console.print(f"[yellow]⚠️  No data returned for period {period}[/yellow]")
                continue
            
            actual_days = len(all_tickers_data)
            console.print(f"[cyan]📊 Downloaded {actual_days} days of data for period {period}[/cyan]")
            
            if actual_days >= seq_length:
                actual_period_used = period
                console.print(f"[green]✅ Sufficient data obtained: {actual_days} days >= {seq_length} required[/green]")
                break
            else:
                console.print(f"[yellow]⚠️  Insufficient data: {actual_days} days < {seq_length} required[/yellow]")
                if attempt < len(periods_to_try) - 1:
                    console.print(f"[cyan]🔄 Trying longer period...[/cyan]")
        
        if all_tickers_data is None or all_tickers_data.empty:
            raise ValueError("No data downloaded after trying all periods. Check ticker symbols.")
        
        if len(all_tickers_data) < seq_length:
            raise ValueError(f"Not enough data after trying all periods. Got {len(all_tickers_data)} days, need {seq_length} days.")
        
        console.print(f"[green]✅ Final data: {len(all_tickers_data)} days using period {actual_period_used}[/green]")
        
        # Precompute historical signals
        console.print(f"[cyan]🧠 Precomputing signals...[/cyan]")
        historical_signals = precompute_historical_signals(
            all_tickers_data, 
            self.model,
            self.config.get('scaler_mean', 0.0),
            self.config.get('scaler_std', 1.0),
            self.config.get('seq_length', 60),
            len(tickers),
            backtest_start_date=None,  # For full-period signal generation, no specific start date
            include_news=self.config.get('include_news', False),
            include_text=self.config.get('include_text', False),
            news_window=self.config.get('news_window', 7),
            text_model=self.config.get('text_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            tickers=tickers,
            latest_only=True,
            time_window=self.config.get('time_window', 1),
            batch_size=self.config.get('batch_size', 1)
        )
        
        if historical_signals.empty:
            raise ValueError("No signals generated. Check model and data.")
        
        console.print(f"[green]✅ Generated signals for {len(historical_signals)} predictions[/green]")
        
        # Get latest signals for each ticker
        latest_signals = []
        for ticker in tickers:
            ticker_signals = historical_signals.xs(ticker, level='Ticker')
            if not ticker_signals.empty:
                latest_signal = ticker_signals.iloc[-1]
                latest_signals.append({
                    'Ticker': ticker,
                    'Date': latest_signal.name,
                    'Predicted_Action': latest_signal['Predicted_Action']
                })
        
        if not latest_signals:
            raise ValueError("No latest signals found.")
        
        # Create DataFrame
        latest_signals_df = pd.DataFrame(latest_signals)
        latest_signals_df = latest_signals_df.set_index(['Ticker', 'Date'])
        
        console.print(f"[green]✅ Generated latest signals for {len(latest_signals_df)} tickers[/green]")
        
        # Display latest signals with ticker names
        console.print(f"\n[bold cyan]📊 Latest Trading Signals[/bold cyan]")
        signals_display = latest_signals_df.reset_index()
        for _, row in signals_display.iterrows():
            ticker = row['Ticker']
            action = row['Predicted_Action']
            date = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
            console.print(f"   [bold]{ticker}[/bold]: {action} (as of {date})")
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as CSV
            csv_file = output_path.with_suffix('.csv')
            latest_signals_df.reset_index().to_csv(csv_file, index=False)
            console.print(f"[green]✅ Signals saved to: {csv_file}[/green]")
            
            # Save as JSON
            json_file = output_path.with_suffix('.json')
            with open(json_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'configuration': self.config,
                    'signals': latest_signals_df.reset_index().to_dict('records')
                }, f, indent=2, default=str)
            console.print(f"[green]✅ Signals saved to: {json_file}[/green]")
        
        return latest_signals_df
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a summary of backtesting results."""
        console.print("\n[bold cyan]📊 Backtesting Results Summary[/bold cyan]")
        
        # Create summary table
        table = Table(title="Backtesting Performance")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Add basic metrics
        table.add_row("Tickers Tested", str(len(results['backtest_params']['tickers'])))
        table.add_row("Data Period", results['backtest_params']['data_period'])
        table.add_row("Initial Cash", f"${results['backtest_params']['initial_cash']:,.0f}")
        table.add_row("Signals Generated", str(results['historical_signals_count']))
        
        # Add portfolio metrics if available
        if 'portfolio_results' in results and results['portfolio_results']:
            portfolio_stats = results['portfolio_results']
            if isinstance(portfolio_stats, dict):
                for key, value in portfolio_stats.items():
                    if isinstance(value, (int, float)):
                        if 'return' in key.lower() or 'sharpe' in key.lower():
                            table.add_row(key.replace('_', ' ').title(), f"{value:.4f}")
                        else:
                            table.add_row(key.replace('_', ' ').title(), f"{value:.2f}")
        
        console.print(table)
        
        # Print top performing tickers
        if 'equity_curves' in results and results['equity_curves']:
            console.print("\n[bold yellow]🏆 Top Performing Tickers[/bold yellow]")
            
            performance_data = []
            for ticker, equity_curve in results['equity_curves'].items():
                if isinstance(equity_curve, pd.DataFrame) and 'Equity' in equity_curve.columns:
                    final_value = equity_curve['Equity'].iloc[-1]
                    initial_value = equity_curve['Equity'].iloc[0]
                    total_return = (final_value - initial_value) / initial_value
                    performance_data.append((ticker, total_return, final_value))
            
            # Sort by total return
            performance_data.sort(key=lambda x: x[1], reverse=True)
            
            perf_table = Table(title="Performance by Ticker")
            perf_table.add_column("Ticker", style="cyan")
            perf_table.add_column("Total Return", style="green")
            perf_table.add_column("Final Value", style="blue")
            
            for ticker, return_pct, final_val in performance_data[:10]:
                perf_table.add_row(
                    ticker,
                    f"{return_pct:.2%}",
                    f"${final_val:,.2f}"
                )
            
            console.print(perf_table)


def main():
    """Main function for standalone backtesting."""
    
    # Print startup banner
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold blue]🚀 JAX GPT Stock Predictor - Backtesting Engine[/bold blue]\n"
        "[dim]Comprehensive Model Evaluation and Signal Generation[/dim]",
        border_style="blue"
    ))
    console.print("="*60)
    
    parser = argparse.ArgumentParser(
        description="Run backtesting on trained model",
        epilog="""
IMPORTANT: The model is sensitive to ticker sequence and order. 
The backtesting engine automatically uses the exact same tickers in the exact same order 
as used during model training to ensure compatibility.

Examples:
  # Auto-detect best model and use training tickers
  python run_backtesting.py
  
  # Use specific model with training tickers
  python run_backtesting.py --model-path ./final_model
  
  # Generate latest signals with training tickers
  python run_backtesting.py --generate-signals
        """
    )
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to trained model checkpoint (if None, auto-detect best model)")
    parser.add_argument("--config-path", type=str, default=None,
                       help="Path to model configuration file")
    parser.add_argument("--tickers", type=str, nargs='+', default=None,
                       help="Tickers to backtest (WARNING: must match training tickers exactly)")
    parser.add_argument("--data-period", type=str, default="auto",
                       help="Data period for backtesting (default: auto - computed from split date or training period)")
    parser.add_argument("--initial-cash", type=float, default=10000,
                       help="Initial cash per ticker (default: 10000)")
    parser.add_argument("--commission-rate", type=float, default=0.001,
                       help="Commission rate for trades (default: 0.001)")
    parser.add_argument("--output-dir", type=str, default="./backtesting_results",
                       help="Output directory for results (default: ./backtesting_results)")
    parser.add_argument("--generate-signals", action="store_true",
                       help="Generate latest trading signals")
    parser.add_argument("--signals-period", type=str, default="1m",
                       help="Data period for signal generation (default: 1m)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Disable plot generation")
    
    args = parser.parse_args()
    
    try:
        # Initialize backtesting engine
        engine = BacktestingEngine(args.model_path, args.config_path)
        
        # Show ticker information
        training_tickers = engine.config.get('tickers', [])
        if training_tickers:
            console.print(f"[cyan]📊 Model was trained on {len(training_tickers)} tickers[/cyan]")
            console.print(f"[dim]Training tickers: {', '.join(training_tickers[:10])}{'...' if len(training_tickers) > 10 else ''}[/dim]")
        
        if args.generate_signals:
            # Generate latest signals
            signals = engine.generate_signals(
                tickers=args.tickers,
                data_period=args.signals_period,
                output_file=f"{args.output_dir}/latest_signals.csv"
            )
        else:
            # Handle auto data period
            data_period = args.data_period
            if data_period == "auto":
                # Use training data period as base for computation
                training_data_period = engine.config.get('data_period', '1y')
                data_period = engine._compute_backtest_period(training_data_period)
                console.print(f"[cyan]🔄 Auto-computed backtest period: {data_period} (from training period: {training_data_period})[/cyan]")
            
            # Run backtesting
            results = engine.run_backtest(
                tickers=args.tickers,
                data_period=data_period,
                initial_cash=args.initial_cash,
                commission_rate=args.commission_rate,
                plot_results=not args.no_plots,
                output_dir=args.output_dir
            )
            
            # Print results summary
            engine.print_results_summary(results)
        
        console.print(f"\n[bold green]🎉 Backtesting completed successfully![/bold green]")
        console.print(f"[dim]Results saved to: {args.output_dir}[/dim]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Backtesting failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 