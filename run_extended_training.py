#!/usr/bin/env python3
"""
Runner script for extended training of best hyperparameter configurations.

Usage:
    python run_extended_training.py [--top-n 5] [--epochs 100] [--results-dir ./hyperparameter_tuning_results]
"""

import os
import sys
import argparse
from pathlib import Path

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

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

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

from scripts.extended_training import ExtendedTrainer
from utils.gpu_utils import check_gpu_availability, get_gpu_utilization, get_gpu_memory_info

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

def main():
    parser = argparse.ArgumentParser(description="Extended training for best hyperparameter configurations")
    parser.add_argument("--top-n", type=int, default=5, 
                       help="Number of top configurations to train (default: 5)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs for extended training (default: 100)")
    parser.add_argument("--results-dir", type=str, default="./hyperparameter_tuning_results",
                       help="Directory containing hyperparameter tuning results (default: ./hyperparameter_tuning_results)")
    
    args = parser.parse_args()
    
    # Print device dashboard before starting training
    print_device_dashboard()
    
    try:
        trainer = ExtendedTrainer(results_dir=args.results_dir)
        trainer.run_extended_training(n_top=args.top_n, epochs=args.epochs)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have run hyperparameter tuning first and the results directory exists.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"Error during extended training: {e}")
        traceback.print_exception(e)
        sys.exit(1)

if __name__ == "__main__":
    main() 