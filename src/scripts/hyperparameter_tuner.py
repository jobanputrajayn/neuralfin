#!/usr/bin/env python3
"""
Hyperparameter Tuning Runner with TensorBoard Integration

Runs the complete hyperparameter optimization process with comprehensive logging.
"""

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'

# --- TensorFlow CPU Restriction (MUST be before JAX import) ---
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    print("[INFO] TensorFlow configured to use CPU only.")
except Exception as e:
    print(f"[WARN] Could not configure TensorFlow device visibility: {e}")

# Now import JAX and the rest
import jax
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.rule import Rule
from rich.columns import Columns
from rich.table import Table
from rich.live import Live

# Import from our refactored modules
from ..config.hyperparameter_config import HyperparameterConfig
from ..data.stock_data import get_large_cap_tickers
from ..hyperparameter_tuning.optimization import HyperparameterOptimizer
from ..utils.gpu_utils import check_gpu_availability, get_gpu_utilization, get_gpu_memory_info

# Initialize Rich console
console = Console()

def print_startup_banner():
    """Print beautiful startup banner"""
    banner_text = Text()
    banner_text.append("🚀 ", style="bold blue")
    banner_text.append("Hyperparameter Tuning with TensorBoard", style="bold cyan")
    banner_text.append(" 🚀", style="bold blue")
    
    subtitle = Text("Comprehensive ML Model Optimization with Real-time Monitoring", style="dim")
    
    layout = Layout()
    layout.split_column(
        Layout(Panel(banner_text, border_style="blue")),
        Layout(Panel(subtitle, border_style="dim"))
    )
    
    console.print(layout)
    console.print(Rule(style="blue"))

def print_configuration_summary(config, tickers):
    """Print configuration summary with Rich formatting"""
    
    # Get device information
    jax_devices = jax.devices()
    jax_device_count = jax.device_count()
    jax_platform = jax.default_backend()
    gpu_info = check_gpu_availability()
    
    # Create configuration panels
    search_panel = Panel(
        f"Random Trials: {config.n_random_trials}\n"
        f"Bayesian Trials: {config.n_bayesian_trials}\n"
        f"Fine-tune Trials: {config.n_fine_tune_trials}\n"
        f"Total Trials: {config.n_random_trials + config.n_bayesian_trials + config.n_fine_tune_trials}",
        title="🎯 Search Strategy",
        border_style="green"
    )
    
    training_panel = Panel(
        f"Random Epochs: {config.epochs_per_trial_random}\n"
        f"Bayesian Epochs: {config.epochs_per_trial_bayesian}\n"
        f"Fine-tune Epochs: {config.epochs_per_trial_fine_tune}",
        title="🏋️ Training Configuration",
        border_style="yellow"
    )
    
    data_panel = Panel(
        f"Data Period: {config.data_period}\n"
        f"Train/Test Split: {config.train_test_split_ratio:.1%}\n"
        f"Ticker Count: {'All Available' if config.ticker_count <= 0 else config.ticker_count}\n"
        f"Tickers: {len(tickers)}",
        title="📊 Data Configuration",
        border_style="magenta"
    )

    resource_panel = Panel(
        f"Max Trial Time: {config.max_trial_time_hours:.1f} hours\n"
        f"Min Throughput: {config.min_throughput:.1f} samples/s\n"
        f"Min GPU Util: {config.min_gpu_util:.1f}%\n"
        f"Min CPU Util: {config.min_cpu_util:.1f}%",
        title="⚙️ Resource Configuration",
        border_style="cyan"
    )

    caching_panel = Panel(
        f"Enable Caching: {config.enable_caching}\n"
        f"Cache Size: {config.cache_size}\n"
        f"Adaptive Batch Size: {config.adaptive_batch_size}\n"
        f"Target Batch Time: {config.target_batch_time:.2f}s",
        title="📦 Caching & Batching",
        border_style="purple"
    )

    jax_panel = Panel(
        f"JAX Memory Fraction: {config.jax_memory_fraction:.2f}\n"
        f"JAX Recomp. Threshold: {config.jax_recompilation_threshold}\n"
        f"JAX Compile Warmup: {config.jax_compile_warmup_batches}\n"
        f"JAX Memory Monitoring: {config.jax_memory_monitoring}",
        title="🧠 JAX Configuration",
        border_style="orange1"
    )
    
    # Create device panel
    device_info = []
    device_info.append(f"JAX Platform: {jax_platform}")
    device_info.append(f"Device Count: {jax_device_count}")
    device_info.append(f"NVML Available: {'✅' if gpu_info['nvml_available'] else '❌'}")
    
    if gpu_info['gpu_count'] > 0:
        device_info.append(f"GPU Count: {gpu_info['gpu_count']}")
        for i in range(min(gpu_info['gpu_count'], 2)):  # Show first 2 GPUs
            if i < len(gpu_info['gpu_names']):
                gpu_name = gpu_info['gpu_names'][i]
                device_info.append(f"GPU {i}: {gpu_name[:20]}...")
                
                # Get current utilization and memory
                util = get_gpu_utilization(i)
                mem_info = get_gpu_memory_info(i)
                
                if mem_info:
                    mem_used_gb = mem_info['used'] / (1024**3)
                    mem_total_gb = mem_info['total'] / (1024**3)
                    mem_percent = mem_info['utilization_percent']
                    device_info.append(f"  Memory: {mem_used_gb:.1f}GB/{mem_total_gb:.1f}GB ({mem_percent:.1f}%)")
                    device_info.append(f"  Utilization: {util:.1f}%")
    else:
        device_info.append("No GPUs detected")
    
    device_panel = Panel(
        "\n".join(device_info),
        title="🔧 Device Status",
        border_style="red"
    )
    
    # Display in columns (3x2 layout)
    top_row = Columns([search_panel, training_panel, data_panel])
    bottom_row = Columns([resource_panel, caching_panel, jax_panel])
    device_row = Columns([device_panel])
    
    console.print(top_row)
    console.print(bottom_row)
    console.print(device_row)
    console.print(Rule(style="dim"))

def main(trials=60, random_trials=15, bayesian_trials=35, fine_tune_trials=10, 
         epochs_random=10, epochs_bayesian=20, epochs_fine_tune=30,
         data_period="5y", tickers=None, ticker_count=50, output_dir="./hyperparameter_tuning_results",
         prefetch_buffer_size=5):
    """Main function to run hyperparameter tuning with TensorBoard"""
    print_startup_banner()
    
    # Configuration for production (comprehensive trials)
    config = HyperparameterConfig(
        n_random_trials=random_trials,     # Initial random exploration
        n_bayesian_trials=bayesian_trials,   # Bayesian optimization
        n_fine_tune_trials=fine_tune_trials,  # Fine-tuning around best results
        epochs_per_trial_random=epochs_random,     # Quick exploration
        epochs_per_trial_bayesian=epochs_bayesian,   # Deeper optimization
        epochs_per_trial_fine_tune=epochs_fine_tune,  # Thorough fine-tuning
        data_period=data_period,       # Use 5 years of data for production
        ticker_count=ticker_count,     # Use the ticker count from command line
        train_test_split_ratio=0.8,
        prefetch_buffer_size=prefetch_buffer_size,
        jax_memory_fraction=0.8,
        search_space={
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
            'batch_size': [1,2,4],          # Batch sizes
            
            # Enhanced features (news and text data)
            'include_news': [True, False],                     # Include news features
            'include_text': [True, False],                     # Include text embeddings
            'news_window': [7,14],                         # News window size
        }
    )
    
    # Get tickers (production subset)
    if tickers is None:
        all_tickers = get_large_cap_tickers()
        if ticker_count <= 0:
            tickers = all_tickers  # Use all available tickers
        else:
            tickers = all_tickers[:ticker_count]  # Use configurable ticker count
    
    # Print configuration summary
    print_configuration_summary(config, tickers)
    
    # Create tuner
    save_dir = output_dir
    
    # Create and run the hyperparameter optimizer
    optimizer = HyperparameterOptimizer(config, tickers, save_dir=save_dir)
    
    # --- TensorBoard Instructions ---
    tensorboard_log_dir = Path(save_dir) / "tensorboard_logs"
    console.print("\n[bold blue]📊 Monitoring with TensorBoard[/bold blue]")
    console.print("To monitor training progress, run the following command in a separate terminal:")
    console.print(f"   [bold cyan]./hyper/start_tensorboard.sh[/bold cyan]")
    console.print(f"[dim]   (TensorBoard will log data to {tensorboard_log_dir})[/dim]")

    console.print("\n[bold green]🎯 Starting hyperparameter optimization...[/bold green]")
    console.print(Rule(style="green"))
    
    try:
        # Run optimization
        best_params, best_value = optimizer.run_optimization()
        
        # Print results
        optimizer.print_summary()
        
        console.print("\n[bold green]✅ Hyperparameter optimization completed![/bold green]")
        console.print(Rule(style="green"))
        
        # Final instructions
        console.print("[bold yellow]📊 To view results, make sure TensorBoard is running:[/bold yellow]")
        console.print(f"   [bold cyan]./hyper/start_tensorboard.sh[/bold cyan]")
        
        console.print("\n[bold cyan]📁 Results saved to:[/bold cyan]")
        console.print(f"   [dim]📂 Checkpoints:[/dim] {save_dir}/checkpoints/")
        console.print(f"   [dim]📊 TensorBoard logs:[/dim] {tensorboard_log_dir}")
        console.print(f"   [dim]📈 Plots:[/dim] {save_dir}/optimization_history.png")
        console.print(f"   [dim]📋 Summary:[/dim] {save_dir}/trial_results.json")
        
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠️  Optimization interrupted by user[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Optimization failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning with TensorBoard")
    parser.add_argument("--trials", type=int, default=60, help="Total number of trials")
    parser.add_argument("--random_trials", type=int, default=15, help="Number of random trials")
    parser.add_argument("--bayesian_trials", type=int, default=35, help="Number of Bayesian trials")
    parser.add_argument("--fine_tune_trials", type=int, default=10, help="Number of fine-tune trials")
    parser.add_argument("--epochs_random", type=int, default=10, help="Number of epochs for random trials")
    parser.add_argument("--epochs_bayesian", type=int, default=20, help="Number of epochs for Bayesian trials")
    parser.add_argument("--epochs_fine_tune", type=int, default=30, help="Number of epochs for fine-tune trials")
    parser.add_argument("--data_period", type=str, default="5y", help="Data period for production")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers")
    parser.add_argument("--ticker_count", type=int, default=50, help="Number of tickers to use (use -1 for all available tickers)")
    parser.add_argument("--output_dir", type=str, default="./hyperparameter_tuning_results", help="Output directory")
    parser.add_argument("--prefetch_buffer_size", type=int, default=5, help="Prefetch buffer size for data loading")
    args = parser.parse_args()

    if args.tickers:
        args.tickers = [t.strip() for t in args.tickers.split(',')]

    main(args.trials, args.random_trials, args.bayesian_trials, args.fine_tune_trials, 
         args.epochs_random, args.epochs_bayesian, args.epochs_fine_tune,
         args.data_period, args.tickers, args.ticker_count, args.output_dir,
         args.prefetch_buffer_size) 