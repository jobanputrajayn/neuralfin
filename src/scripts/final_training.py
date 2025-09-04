"""
Final Model Training Script

This script trains the best configuration from hyperparameter tuning
on the full dataset for production use.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from flax import nnx

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
import optax
from flax.nnx import TrainState

from etils import epath

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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.gpt_classifier import GPTClassifier
from models.constants import NUM_CLASSES
from data.stock_data import get_stock_data, get_large_cap_tickers
from data.sequence_generator import StockSequenceGenerator
from training.training_functions import train_step, evaluate_model, _init_gpu_optimization

from config.hyperparameter_config import HyperparameterConfig
from utils.gpu_utils import check_gpu_availability, get_gpu_utilization, get_gpu_memory_info
from utils.system_utils import get_system_info

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

class FinalTrainer:
    """Final model trainer for production-ready models."""
    
    def __init__(self, config_path: str, output_dir: str = "./final_model"):
        """
        Initialize the final trainer.
        
        Args:
            config_path: Path to the best configuration JSON file
            output_dir: Directory to save the final model
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Display device dashboard
        console.print("\n" + "="*60)
        console.print(Panel.fit("[bold blue]Final Model Training - Device Configuration[/bold blue]", border_style="blue"))
        console.print("="*60)
        
        # Show device dashboard
        device_dashboard = create_device_dashboard()
        console.print(device_dashboard)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize GPU optimization
        _init_gpu_optimization()
        
        # Get system info and display detailed GPU status
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
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the best configuration from hyperparameter tuning."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            full_config = json.load(f)
        
        # Extract the nested config if it exists
        if 'config' in full_config:
            config = full_config['config']
            # Add metadata from the full config
            config['source'] = full_config.get('source', 'unknown')
            config['accuracy'] = full_config.get('accuracy', 0.0)
            config['timestamp'] = full_config.get('timestamp', '')
            config['analysis_metadata'] = full_config.get('analysis_metadata', {})
            # Add tickers and data_period if available
            if 'tickers' in full_config:
                config['tickers'] = full_config['tickers']
            if 'data_period' in full_config:
                config['data_period'] = full_config['data_period']
        else:
            config = full_config
        
        console.print(f"[green]✅ Loaded configuration from: {self.config_path}[/green]")
        return config
    
    def _create_model(self) -> GPTClassifier:
        """Create the model with the best configuration."""
        # Get tickers for input_features
        tickers = self.config.get('tickers', get_large_cap_tickers()[:50])
        
        # Calculate enhanced input features based on configuration
        features_per_ticker = 1  # Base price feature
        
        if self.config.get('include_news', False):
            features_per_ticker += 10  # News features per ticker
        
        if self.config.get('include_text', False):
            features_per_ticker += 384  # Text features per ticker
        
        enhanced_input_features = len(tickers) * features_per_ticker
        
        console.print(f"[green]✅ Created model with configuration:[/green]")
        console.print(f"   [dim]Layers: {self.config['num_layers']}[/dim]")
        console.print(f"   [dim]Model Dim: {self.config['d_model']}[/dim]")
        console.print(f"   [dim]Heads: {self.config['num_heads']}[/dim]")
        console.print(f"   [dim]FF Dim: {self.config['d_ff']}[/dim]")
        console.print(f"   [dim]Dropout: {self.config['dropout_rate']:.2f}[/dim]")
        console.print(f"   [dim]Input Features: {len(tickers)} tickers × {features_per_ticker} features = {enhanced_input_features}[/dim]")
        console.print(f"   [dim]Enhanced Features: news={self.config.get('include_news', False)}, text={self.config.get('include_text', False)}[/dim]")
        
        model = GPTClassifier(
            num_classes=NUM_CLASSES,
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            d_ff=self.config['d_ff'],
            dropout_rate=self.config['dropout_rate'],
            input_features=enhanced_input_features,
            num_tickers=len(tickers),
            rngs=nnx.Rngs(0)
        )
        
        return model
    
    def _prepare_data(self) -> tuple:
        """Prepare training and validation data."""
        console.print(f"\n[cyan]📊 Preparing data...[/cyan]")
        
        # Get tickers from configuration
        config_tickers = self.config.get('tickers', [])
        if config_tickers:
            tickers = config_tickers
            console.print(f"[green]✅ Using {len(tickers)} tickers from configuration: {', '.join(tickers)}[/green]")
        else:
            # Fallback to default tickers
            tickers = get_large_cap_tickers()[:50]
            console.print(f"[yellow]⚠️  No tickers in configuration, using default {len(tickers)} tickers[/yellow]")
            console.print(f"[dim]Default tickers: {', '.join(tickers[:5])}...[/dim]")
        
        data_period = self.config.get('data_period', '5y')
        seq_length = self.config['seq_length']
        time_window = self.config['time_window']
        batch_size = self.config['batch_size']
        train_test_split = self.config.get('train_test_split_ratio', 0.8)
        
        console.print(f"   [dim]Tickers: {len(tickers)}[/dim]")
        console.print(f"   [dim]Data Period: {data_period}[/dim]")
        console.print(f"   [dim]Sequence Length: {seq_length}[/dim]")
        console.print(f"   [dim]Time Window: {time_window}[/dim]")
        console.print(f"   [dim]Batch Size: {batch_size}[/dim]")
        
        # Download data
        all_tickers_data = get_stock_data(tickers, period=data_period)
        
        if all_tickers_data.empty:
            raise ValueError("No data downloaded. Check ticker symbols and data period.")
        
        # Calculate scaler parameters
        close_prices = all_tickers_data.xs('Close', level=1, axis=1)[tickers].values
        scaler_mean = np.nanmean(close_prices)
        scaler_std = np.nanstd(close_prices)
        
        # Generate sequence indices
        max_valid_idx = len(close_prices) - seq_length - time_window
        all_sequence_indices = list(range(max_valid_idx))
        
        # Split into train/validation
        split_idx = int(len(all_sequence_indices) * train_test_split)
        train_indices = all_sequence_indices[:split_idx]
        val_indices = all_sequence_indices[split_idx:]
        
        # Extract enhanced features configuration from config
        include_news = self.config.get('include_news', False)
        include_text = self.config.get('include_text', False)
        news_window = self.config.get('news_window', 7)
        text_model = self.config.get('text_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Create data generators
        train_generator = StockSequenceGenerator(
            sequence_indices_to_use=train_indices,
            all_ohlcv_data=all_tickers_data,
            seq_length=seq_length,
            time_window=time_window,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            batch_size=batch_size,
            shuffle_indices=True,
            tickers=tickers,
            include_news=include_news,
            include_text=include_text,
            news_window=news_window,
            text_model=text_model
        )
        
        val_generator = StockSequenceGenerator(
            sequence_indices_to_use=val_indices,
            all_ohlcv_data=all_tickers_data,
            seq_length=seq_length,
            time_window=time_window,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            batch_size=batch_size,
            shuffle_indices=False,
            tickers=tickers,
            include_news=include_news,
            include_text=include_text,
            news_window=news_window,
            text_model=text_model
        )
        
        # Calculate alpha weights for class balancing
        all_labels = []
        for _, labels, _, _ in train_generator:
            valid_labels = labels[labels != -1]
            all_labels.extend(valid_labels.flatten().tolist())
        
        if not all_labels:
            alpha_weights = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        else:
            class_counts = np.bincount(all_labels, minlength=NUM_CLASSES)
            total_labels = class_counts.sum()
            if total_labels == 0:
                alpha_weights = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
            else:
                epsilon = np.finfo(np.float32).eps
                inverse_frequencies = total_labels / (class_counts + epsilon)
                alpha_weights = inverse_frequencies / np.sum(inverse_frequencies)
        
        console.print(f"[green]✅ Data prepared successfully[/green]")
        console.print(f"   [dim]Training sequences: {len(train_generator)}[/dim]")
        console.print(f"   [dim]Validation sequences: {len(val_generator)}[/dim]")
        console.print(f"   [dim]Scaler mean: {scaler_mean:.4f}[/dim]")
        console.print(f"   [dim]Scaler std: {scaler_std:.4f}[/dim]")
        
        return train_generator, val_generator, alpha_weights, scaler_mean, scaler_std
    
    def train_model(self, epochs: int = 200, patience: int = 15) -> Dict[str, Any]:
        """
        Train the final model with the best configuration.
        
        Args:
            epochs: Number of training epochs
            patience: Early stopping patience
            
        Returns:
            Dictionary containing training results
        """
        console.print(Panel.fit(
            f"[bold green]Final Model Training[/bold green]\n"
            f"Epochs: {epochs}\n"
            f"Patience: {patience}\n"
            f"Output: {self.output_dir}",
            title="🚀 Production Training"
        ))
        
        # Create model
        model = self._create_model()
        
        # Prepare data
        train_generator, val_generator, alpha_weights, scaler_mean, scaler_std = self._prepare_data()

        train_generator = PrefetchGenerator(train_generator)
        val_generator = PrefetchGenerator(val_generator)
        
        # Training loop
        best_val_accuracy = 0.0
        patience_counter = 0
        training_history = []
        validation_history = []
        val_accuracy = 0.0  # Initialize validation accuracy
        val_loss = 0.0      # Initialize validation loss
        # Save best model
        best_model_path = self.output_dir / "best_model"
        
        console.print(f"\n[cyan]\U0001f3cb\ufe0f  Starting training for {epochs} epochs...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Training...", total=epochs)
            
            for epoch in range(epochs):
                # Training step
                total_loss = 0.0
                num_batches = 0
                
                for batch in train_generator:
                    if batch is None:
                        continue
                    
                    x, y, r, padding_mask = batch
                    batch_tuple = (x, y, r, padding_mask)
                    try:
                        model, loss = train_step(model, batch_tuple, alpha_weights)
                    except Exception as e:
                        import traceback
                        console.print(f"[red]Exception in train_step: {e}[/red]")
                        traceback.print_exc()
                        raise
                    total_loss += float(loss)  # Convert JAX array to Python scalar
                    num_batches += 1
                
                avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
                training_history.append(avg_train_loss)  # This is already a Python scalar
                
                # Validation step (every 5 epochs)
                if epoch % 5 == 0:
                    val_metrics = evaluate_model(model, val_generator, model, alpha_weights)
                    # Convert JAX arrays to Python scalars for JSON serialization
                    val_accuracy = float(val_metrics['accuracy']) if 'accuracy' in val_metrics else float(val_metrics.get('overall_accuracy', 0.0))
                    val_loss = float(val_metrics['loss'])
                    validation_history.append(val_accuracy)
                    
                    # Early stopping
                    if val_accuracy > best_val_accuracy + 0.001:
                        best_val_accuracy = val_accuracy
                        patience_counter = 0
                        
                        from training.checkpointing import save_checkpoint
                        save_checkpoint(best_model_path, model, None, scaler_mean, scaler_std, epoch, self.config['learning_rate'])
                        
                        console.print(f"[green]\u2705 New best model saved (epoch {epoch}): {val_accuracy:.4f}[/green]")
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        console.print(f"[yellow]\u26a0\ufe0f  Early stopping at epoch {epoch}[/yellow]")
                        break
                
                progress.update(task, advance=1)
                
                # Print progress every 10 epochs
                if epoch % 10 == 0:
                    console.print(f"[dim]Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.4f}[/dim]")
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        from training.checkpointing import save_checkpoint
        save_checkpoint(final_model_path, model, None, scaler_mean, scaler_std, epoch, self.config['learning_rate'])
        
        # Save configuration
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Generate training report
        results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'training_results': {
                'final_epoch': epoch,
                'best_val_accuracy': best_val_accuracy,
                'final_val_accuracy': val_accuracy,
                'final_val_loss': val_loss,
                'training_history': training_history,
                'validation_history': validation_history
            },
            'model_paths': {
                'best_model': str(best_model_path),
                'final_model': str(final_model_path),
                'config': str(config_file)
            }
        }
        
        results_file = self.output_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"\n[bold green]\U0001f389 Final training completed![/bold green]")
        console.print(f"[dim]Best validation accuracy: {best_val_accuracy:.4f}[/dim]")
        console.print(f"[dim]Final validation accuracy: {val_accuracy:.4f}[/dim]")
        console.print(f"[dim]Results saved to: {self.output_dir}[/dim]")
        
        return results
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a summary of training results."""
        console.print("\n[bold cyan]📊 Final Training Results Summary[/bold cyan]")
        
        # Create summary table
        table = Table(title="Training Performance")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        training_results = results['training_results']
        
        table.add_row("Final Epoch", str(training_results['final_epoch']))
        table.add_row("Best Val Accuracy", f"{training_results['best_val_accuracy']:.4f}")
        table.add_row("Final Val Accuracy", f"{training_results['final_val_accuracy']:.4f}")
        table.add_row("Final Val Loss", f"{training_results['final_val_loss']:.4f}")
        
        console.print(table)
        
        # Print model paths
        console.print("\n[bold yellow]📁 Model Files[/bold yellow]")
        model_paths = results['model_paths']
        for name, path in model_paths.items():
            console.print(f"   [dim]{name}:[/dim] {path}")


def main():
    """Main function for final model training."""
    
    # Print startup banner
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold blue]🚀 JAX GPT Stock Predictor - Final Model Training[/bold blue]\n"
        "[dim]Production-Ready Model Training with Best Configuration[/dim]",
        border_style="blue"
    ))
    console.print("="*60)
    
    parser = argparse.ArgumentParser(description="Train final model with best configuration")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to best configuration JSON file")
    parser.add_argument("--output-dir", type=str, default="./final_model",
                       help="Output directory for final model (default: ./final_model)")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of training epochs (default: 200)")
    parser.add_argument("--patience", type=int, default=15,
                       help="Early stopping patience (default: 15)")
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = FinalTrainer(args.config, args.output_dir)
        
        # Train model
        results = trainer.train_model(epochs=args.epochs, patience=args.patience)
        
        # Print results summary
        trainer.print_results_summary(results)
        
        console.print(f"\n[bold green]🎉 Final model training completed successfully![/bold green]")
        console.print(f"[dim]Model saved to: {args.output_dir}[/dim]")
        
        # Instructions for next steps
        console.print("\n[bold yellow]🚀 Next Steps:[/bold yellow]")
        console.print(f"   [bold cyan]Backtest:[/bold cyan] python run_backtesting.py --model-path {args.output_dir}")
        console.print(f"   [bold cyan]Generate Signals:[/bold cyan] python run_backtesting.py --model-path {args.output_dir} --generate-signals")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Final training failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 