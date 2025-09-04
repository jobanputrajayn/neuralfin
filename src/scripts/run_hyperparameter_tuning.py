#!/usr/bin/env python3
"""
Hyperparameter Tuning Entry Point

This is the main entry point for hyperparameter tuning.
It provides a simple interface to run hyperparameter optimization.
"""

import os
import sys
import argparse
from pathlib import Path
import time
from datetime import datetime

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
import jax.profiler as profiler

# Import Rich UI components
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

# Import from our refactored modules
from .hyperparameter_tuner import main as run_hyperparameter_tuning
from ..config.hyperparameter_config import HyperparameterConfig
from ..data.stock_data import get_large_cap_tickers
from ..utils.gpu_utils import check_gpu_availability, get_gpu_utilization, get_gpu_memory_info

# Initialize Rich console
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
    
    # Create detailed GPU info if available
    if gpu_info['gpu_count'] > 0 and gpu_info['nvml_available']:
        gpu_details = []
        for i in range(gpu_info['gpu_count']):
            if i < len(gpu_info['gpu_names']):
                gpu_name = gpu_info['gpu_names'][i]
                gpu_details.append(f"GPU {i}: {gpu_name}")
                
                # Get current utilization and memory
                util = get_gpu_utilization(i)
                mem_info = get_gpu_memory_info(i)
                
                if mem_info:
                    mem_used_gb = mem_info['used'] / (1024**3)
                    mem_total_gb = mem_info['total'] / (1024**3)
                    mem_percent = mem_info['utilization_percent']
                    gpu_details.append(f"  Memory: {mem_used_gb:.1f}GB/{mem_total_gb:.1f}GB ({mem_percent:.1f}%)")
                    gpu_details.append(f"  Utilization: {util:.1f}%")
                gpu_details.append("")
        
        gpu_details_panel = Panel(
            "\n".join(gpu_details),
            title="📊 GPU Details",
            border_style="orange1"
        )
    else:
        gpu_details_panel = Panel(
            "No GPU monitoring available\n(NVML not available or no GPUs detected)",
            title="📊 GPU Details",
            border_style="red"
        )
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(device_table),
        Layout(Columns([gpu_status_panel, memory_panel, gpu_details_panel]))
    )
    
    return layout

def print_device_dashboard():
    """Print the device dashboard"""
    console.print("\n[bold cyan]🔧 JAX Device Dashboard[/bold cyan]")
    console.print(Rule(style="cyan"))
    
    dashboard = create_device_dashboard()
    console.print(dashboard)
    console.print(Rule(style="cyan"))

def create_live_device_dashboard():
    """Create a live updating device dashboard for real-time monitoring"""
    
    def generate_dashboard():
        # Get current JAX device information
        jax_devices = jax.devices()
        jax_device_count = jax.device_count()
        jax_platform = jax.default_backend()
        
        # Get current GPU information
        gpu_info = check_gpu_availability()
        
        # Create device status table
        device_table = Table(title="🔧 Live JAX Device Status", show_header=True, header_style="bold magenta")
        device_table.add_column("Device Type", style="cyan", no_wrap=True)
        device_table.add_column("Device ID", style="green")
        device_table.add_column("Platform", style="yellow")
        device_table.add_column("Status", style="blue")
        device_table.add_column("Utilization", style="orange1")
        
        # Add JAX devices with real-time info
        for i, device in enumerate(jax_devices):
            device_type = "GPU" if "gpu" in str(device).lower() else "CPU" if "cpu" in str(device).lower() else "TPU" if "tpu" in str(device).lower() else "Unknown"
            
            # Get real-time utilization if GPU
            if device_type == "GPU" and gpu_info['nvml_available'] and i < gpu_info['gpu_count']:
                util = get_gpu_utilization(i)
                utilization_str = f"{util:.1f}%"
            else:
                utilization_str = "N/A"
            
            device_table.add_row(
                device_type,
                str(i),
                str(device),
                "✅ Active",
                utilization_str
            )
        
        # Create real-time GPU memory panel
        if gpu_info['gpu_count'] > 0 and gpu_info['nvml_available']:
            gpu_memory_info = []
            for i in range(gpu_info['gpu_count']):
                mem_info = get_gpu_memory_info(i)
                util = get_gpu_utilization(i)
                
                if mem_info:
                    mem_used_gb = mem_info['used'] / (1024**3)
                    mem_total_gb = mem_info['total'] / (1024**3)
                    mem_percent = mem_info['utilization_percent']
                    gpu_memory_info.append(f"GPU {i}: {mem_used_gb:.1f}GB/{mem_total_gb:.1f}GB ({mem_percent:.1f}%) | {util:.1f}% util")
            
            memory_panel = Panel(
                "\n".join(gpu_memory_info),
                title="📊 Real-time GPU Memory & Utilization",
                border_style="orange1"
            )
        else:
            memory_panel = Panel(
                "No GPU monitoring available",
                title="📊 Real-time GPU Memory & Utilization",
                border_style="red"
            )
        
        # Create system info panel
        system_panel = Panel(
            f"JAX Platform: {jax_platform}\n"
            f"Device Count: {jax_device_count}\n"
            f"Memory Fraction: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.8')}\n"
            f"Update Time: {datetime.now().strftime('%H:%M:%S')}",
            title="🚀 System Info",
            border_style="green"
        )
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(device_table),
            Layout(Columns([system_panel, memory_panel]))
        )
        
        return layout
    
    return generate_dashboard

def run_live_device_dashboard():
    """Run a live updating device dashboard"""
    console.print("\n[bold cyan]🔧 Live JAX Device Dashboard[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop monitoring[/dim]")
    console.print(Rule(style="cyan"))
    
    dashboard_generator = create_live_device_dashboard()
    
    try:
        with Live(dashboard_generator(), refresh_per_second=2, screen=True) as live:
            while True:
                live.update(dashboard_generator())
                time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Device monitoring stopped[/bold yellow]")

def print_usage():
    """Print usage information"""
    print("""
🎯 Hyperparameter Tuning for JAX GPT Stock Predictor

Usage:
  python -m src.scripts.run_hyperparameter_tuning [options]

Options:
  --trials TOTAL_TRIALS        Total number of trials (default: 60)
  --random-trials RANDOM       Number of random trials (default: 15)
  --bayesian-trials BAYESIAN   Number of Bayesian trials (default: 35)
  --fine-tune-trials FINE      Number of fine-tune trials (default: 10)
  --data-period PERIOD         Data period (default: 5y)
  --tickers TICKER1,TICKER2    Comma-separated list of tickers
  --ticker-count COUNT         Number of tickers when auto-selecting (default: 50)
  --output-dir DIR             Output directory (default: ./hyperparameter_tuning_results)
  --epochs-random EPOCHS       Epochs per random trial (default: 10)
  --epochs-bayesian EPOCHS     Epochs per Bayesian trial (default: 20)
  --epochs-fine-tune EPOCHS    Epochs per fine-tune trial (default: 30)
  --prefetch-buffer-size BUFFER_SIZE
                              Prefetch buffer size for data loading (default: 5)
  --live-dashboard             Show live updating device dashboard during tuning
  --device-dashboard-only      Show only the device dashboard and exit
  --help                       Show this help message

Device Dashboard Options:
  --device-dashboard-only      Quick device status check
  --live-dashboard            Real-time device monitoring during tuning

Examples:
  python -m src.scripts.run_hyperparameter_tuning
  python -m src.scripts.run_hyperparameter_tuning --trials 100
  python -m src.scripts.run_hyperparameter_tuning --tickers AAPL,MSFT,GOOGL
  python -m src.scripts.run_hyperparameter_tuning --ticker-count 100
  python -m src.scripts.run_hyperparameter_tuning --ticker-count -1  # Use all available tickers
  python -m src.scripts.run_hyperparameter_tuning --output-dir ./my_results
  python -m src.scripts.run_hyperparameter_tuning --device-dashboard-only  # Check devices only
  python -m src.scripts.run_hyperparameter_tuning --live-dashboard  # Real-time monitoring
""")

def main():
    """Main entry point for hyperparameter tuning"""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning for JAX GPT Stock Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.scripts.run_hyperparameter_tuning
  python -m src.scripts.run_hyperparameter_tuning --trials 100
  python -m src.scripts.run_hyperparameter_tuning --tickers AAPL,MSFT,GOOGL
  python -m src.scripts.run_hyperparameter_tuning --ticker-count 100
  python -m src.scripts.run_hyperparameter_tuning --ticker-count -1  # Use all available tickers
        """
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=60,
        help='Total number of trials (default: 60)'
    )
    
    parser.add_argument(
        '--random-trials',
        type=int,
        default=15,
        help='Number of random trials (default: 15)'
    )
    
    parser.add_argument(
        '--bayesian-trials',
        type=int,
        default=35,
        help='Number of Bayesian trials (default: 35)'
    )
    
    parser.add_argument(
        '--fine-tune-trials',
        type=int,
        default=10,
        help='Number of fine-tune trials (default: 10)'
    )
    
    parser.add_argument(
        '--data-period',
        type=str,
        default='5y',
        help='Data period to use (default: 5y)'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        help='Comma-separated list of stock tickers (default: auto-select)'
    )
    
    parser.add_argument(
        '--ticker-count',
        type=int,
        default=50,
        help='Number of tickers to use when auto-selecting (default: 50, use -1 for all available tickers)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./hyperparameter_tuning_results',
        help='Output directory for results (default: ./hyperparameter_tuning_results)'
    )
    
    parser.add_argument(
        '--epochs-random',
        type=int,
        default=10,
        help='Epochs per random trial (default: 10)'
    )
    
    parser.add_argument(
        '--epochs-bayesian',
        type=int,
        default=20,
        help='Epochs per Bayesian trial (default: 20)'
    )
    
    parser.add_argument(
        '--epochs-fine-tune',
        type=int,
        default=30,
        help='Epochs per fine-tune trial (default: 30)'
    )
    
    parser.add_argument(
        '--prefetch-buffer-size',
        type=int,
        default=5,
        help='Prefetch buffer size for data loading (default: 5)'
    )
    
    parser.add_argument(
        '--live-dashboard',
        action='store_true',
        help='Show live updating device dashboard during tuning'
    )
    
    parser.add_argument(
        '--device-dashboard-only',
        action='store_true',
        help='Show only the device dashboard and exit'
    )
    
    args = parser.parse_args()
    
    # Handle device dashboard only mode
    if args.device_dashboard_only:
        print_device_dashboard()
        return
    
    # Validate arguments
    if args.trials != (args.random_trials + args.bayesian_trials + args.fine_tune_trials):
        print(f"[WARN] Total trials ({args.trials}) doesn't match sum of individual trials ({args.random_trials + args.bayesian_trials + args.fine_tune_trials})")
        print(f"[INFO] Adjusting total trials to {args.random_trials + args.bayesian_trials + args.fine_tune_trials}")
        args.trials = args.random_trials + args.bayesian_trials + args.fine_tune_trials
    
    # Parse tickers
    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    else:
        all_tickers = get_large_cap_tickers()
        if args.ticker_count <= 0:
            tickers = all_tickers  # Use all available tickers
        else:
            tickers = all_tickers[:args.ticker_count]  # Use configurable ticker count
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print device dashboard
    print_device_dashboard()
    
    # Handle live dashboard mode
    if args.live_dashboard:
        console.print("\n[bold yellow]Starting live device monitoring...[/bold yellow]")
        console.print("[dim]This will run in the background during hyperparameter tuning[/dim]")
        
        # Start live dashboard in a separate thread
        import threading
        dashboard_thread = threading.Thread(target=run_live_device_dashboard, daemon=True)
        dashboard_thread.start()
    
    print(f"🎯 Starting hyperparameter tuning with {args.trials} total trials")
    print(f"📊 Using {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"📁 Results will be saved to: {output_dir}")
    profiler.start_server(6090)
    
    try:
        # Run hyperparameter tuning
        run_hyperparameter_tuning(
            trials=args.trials,
            random_trials=args.random_trials,
            bayesian_trials=args.bayesian_trials,
            fine_tune_trials=args.fine_tune_trials,
            epochs_random=args.epochs_random,
            epochs_bayesian=args.epochs_bayesian,
            epochs_fine_tune=args.epochs_fine_tune,
            data_period=args.data_period,
            tickers=tickers,
            ticker_count=args.ticker_count,
            output_dir=str(output_dir),
            prefetch_buffer_size=args.prefetch_buffer_size
        )
        
    except KeyboardInterrupt:
        print("\n⚠️  Hyperparameter tuning interrupted by user")
    except Exception as e:
        print(f"\n❌ Hyperparameter tuning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 