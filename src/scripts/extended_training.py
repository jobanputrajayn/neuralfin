"""
Extended training script for the best hyperparameter configurations.

This script takes the best configurations from hyperparameter tuning
and trains them for extended periods with cross-validation.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import optuna
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.rule import Rule
from rich.columns import Columns
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

# Add src to path for proper imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Now import using absolute paths
from models.gpt_classifier import GPTClassifier
from data.stock_data import get_stock_data, get_large_cap_tickers
from data.sequence_generator import StockSequenceGenerator
from training.training_functions import train_step, evaluate_model, _init_gpu_optimization

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

class ExtendedTrainer:
    """Extended training for best hyperparameter configurations."""
    
    def __init__(self, results_dir: str = "./hyperparameter_tuning_results"):
        self.results_dir = Path(results_dir)
        self.config = HyperparameterConfig()
        self.tickers = get_large_cap_tickers()
        
        # Print device dashboard at initialization
        print_device_dashboard()
        
        # Initialize GPU optimization
        _init_gpu_optimization()
        
        # Load optimization results
        self.study = self._load_study()
        self.trial_results = self._load_trial_results()
        
        # Extract configuration from step 1 results
        self._extract_step1_config()
        
    def _load_study(self) -> optuna.Study:
        """Load the optimization study."""
        study_file = self.results_dir / "optimization_study.pkl"
        if not study_file.exists():
            raise FileNotFoundError(f"Study file not found: {study_file}")
        
        with open(study_file, 'rb') as f:
            return pickle.load(f)
    
    def _load_trial_results(self) -> List[Dict]:
        """Load trial results."""
        results_file = self.results_dir / "trial_results.json"
        if not results_file.exists():
            return []
        
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def get_top_configurations(self, n_top: int = 5) -> List[Tuple[Dict, float]]:
        """Get top n configurations from optimization."""
        # Sort trials by validation accuracy
        sorted_trials = sorted(self.trial_results, 
                             key=lambda x: x['metrics']['validation_accuracy'], 
                             reverse=True)
        
        top_configs = []
        for trial in sorted_trials[:n_top]:
            config = trial['params']
            accuracy = trial['metrics']['validation_accuracy']
            top_configs.append((config, accuracy))
        
        return top_configs
    
    def train_extended(self, config: Dict[str, Any], 
                      epochs: int = 100,
                      cross_validation_folds: int = 3) -> Dict[str, Any]:
        """Train a configuration for extended epochs with cross-validation."""
        
        console.print(f"\n[bold cyan]🚀 Extended Training for Configuration[/bold cyan]")
        console.print(f"Epochs: {epochs}, Cross-validation folds: {cross_validation_folds}")
        
        # Try to find artifacts to get the correct tickers
        artifacts = self._find_configuration_artifacts(config)
        if artifacts is not None:
            tickers_to_use = artifacts['tickers']
            console.print(f"[dim]Using {len(tickers_to_use)} tickers from artifacts[/dim]")
        else:
            tickers_to_use = self.tickers
            console.print(f"[dim]Using {len(tickers_to_use)} default tickers[/dim]")
        
        # Calculate enhanced input features based on configuration
        features_per_ticker = 1  # Base price feature
        
        if config.get('include_news', False):
            features_per_ticker += 10  # News features per ticker
        
        if config.get('include_text', False):
            features_per_ticker += 384  # Text features per ticker
        
        enhanced_input_features = len(tickers_to_use) * features_per_ticker
        
        console.print(f"[dim]Model input features: {len(tickers_to_use)} tickers × {features_per_ticker} features = {enhanced_input_features}[/dim]")
        console.print(f"🔍 Model expects input_features: {enhanced_input_features}")
        
        # Create model with configuration
        model = GPTClassifier(
            num_classes=self.config.num_classes,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            dropout_rate=config['dropout_rate'],
            input_features=enhanced_input_features,
            num_tickers=len(tickers_to_use),
            rngs=nnx.Rngs(0)
        )
        
        # Cross-validation results
        cv_results = []
        
        for fold in range(cross_validation_folds):
            console.print(f"\n[bold yellow]Fold {fold + 1}/{cross_validation_folds}[/bold yellow]")
            
            # Prepare data with different train/test splits for each fold
            train_generator, test_generator, alpha_weights = self._prepare_data_fold(
                config, fold, cross_validation_folds
            )
            
            # Train for extended epochs
            fold_result = self._train_fold(
                model, config, train_generator, test_generator, 
                alpha_weights, epochs, enhanced_input_features
            )
            
            cv_results.append(fold_result)
        
        # Aggregate cross-validation results
        avg_accuracy = np.mean([r['final_accuracy'] for r in cv_results])
        avg_return = np.mean([r['final_return'] for r in cv_results])
        std_accuracy = np.std([r['final_accuracy'] for r in cv_results])
        
        return {
            'config': config,
            'cv_results': cv_results,
            'avg_accuracy': avg_accuracy,
            'avg_return': avg_return,
            'std_accuracy': std_accuracy,
            'epochs_trained': epochs
        }
    
    def _prepare_data_fold(self, config: Dict[str, Any], 
                          fold: int, n_folds: int) -> Tuple[StockSequenceGenerator, StockSequenceGenerator, np.ndarray]:
        """Prepare data for a specific cross-validation fold using saved artifacts."""
        # Try to find artifacts for this configuration
        artifacts = self._find_configuration_artifacts(config)
        
        # Determine tickers to use (same logic as train_extended)
        if artifacts is not None:
            tickers_to_use = artifacts['tickers']
        else:
            tickers_to_use = self.tickers
        
        if artifacts is not None:
            console.print(f"[dim]Using saved artifacts from trial {artifacts.get('trial_number', 'unknown')}[/dim]")
            return self._prepare_data_with_artifacts(artifacts, config, fold, n_folds, tickers_to_use)
        else:
            console.print("[yellow]Warning: No saved artifacts found. Generating new data split.[/yellow]")
            return self._prepare_data_new_split(config, fold, n_folds)
    
    def _find_configuration_artifacts(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find saved artifacts for a given configuration."""
        for trial_result in self.trial_results:
            if trial_result.get('artifacts') is None:
                continue
                
            trial_params = trial_result['params']
            # Check if this trial used the same configuration
            if (trial_params.get('seq_length') == config.get('seq_length') and
                trial_params.get('time_window') == config.get('time_window') and
                trial_params.get('batch_size') == config.get('batch_size')):
                # Add trial number to artifacts for reference
                artifacts = trial_result['artifacts'].copy()
                artifacts['trial_number'] = trial_result.get('trial_number', 'unknown')
                return artifacts
        
        return None
    
    def _prepare_data_with_artifacts(self, artifacts: Dict[str, Any], config: Dict[str, Any], 
                                   fold: int, n_folds: int, tickers_to_use: List[str]) -> Tuple[StockSequenceGenerator, StockSequenceGenerator, np.ndarray]:
        """Prepare data using saved artifacts for consistency."""
        # Load all OHLCV data
        all_ohlcv_data = get_stock_data(artifacts['tickers'], period=artifacts['data_period'])
        
        # Use saved indices and scalers
        train_indices = artifacts['train_indices']
        test_indices = artifacts['test_indices']
        scaler_mean = artifacts['scaler_mean']
        scaler_std = artifacts['scaler_std']
        alpha_weights = np.array(artifacts['alpha_weights'])
        
        # For cross-validation, we'll split the original train indices
        # This ensures we don't use test data for training
        fold_size = len(train_indices) // n_folds
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(train_indices)
        
        cv_train_indices = train_indices[:val_start] + train_indices[val_end:]
        cv_val_indices = train_indices[val_start:val_end]
        
        # Debug: Print the split information
        console.print(f"[dim]CV Fold {fold + 1}: Train={len(cv_train_indices)} sequences, Val={len(cv_val_indices)} sequences[/dim]")
        console.print(f"[dim]Original train size: {len(train_indices)}, Fold size: {fold_size}[/dim]")
        
        # Extract enhanced features configuration from config (use config first, then artifacts as fallback)
        include_news = config.get('include_news', artifacts.get('include_news', False))
        include_text = config.get('include_text', artifacts.get('include_text', False))
        news_window = config.get('news_window', artifacts.get('news_window', 7))
        text_model = config.get('text_model', artifacts.get('text_model', 'sentence-transformers/all-MiniLM-L6-v2'))
        
        console.print(f"[dim]Enhanced features: news={include_news}, text={include_text}, window={news_window}[/dim]")
        
        # Create generators using saved artifacts with enhanced features
        # Use the same tickers that were used to create the model
        train_generator = StockSequenceGenerator(
            sequence_indices_to_use=cv_train_indices,
            all_ohlcv_data=all_ohlcv_data,
            seq_length=artifacts['seq_length'],
            time_window=artifacts['time_window'],
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            batch_size=config['batch_size'],
            shuffle_indices=True,
            tickers=tickers_to_use,  # Use the same tickers as the model
            include_news=include_news,
            include_text=include_text,
            news_window=news_window,
            text_model=text_model
        )
        
        test_generator = StockSequenceGenerator(
            sequence_indices_to_use=cv_val_indices,
            all_ohlcv_data=all_ohlcv_data,
            seq_length=artifacts['seq_length'],
            time_window=artifacts['time_window'],
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            batch_size=config['batch_size'],
            shuffle_indices=False,
            tickers=tickers_to_use,  # Use the same tickers as the model
            include_news=include_news,
            include_text=include_text,
            news_window=news_window,
            text_model=text_model
        )
        
        return train_generator, test_generator, alpha_weights
    
    def _prepare_data_new_split(self, config: Dict[str, Any], 
                               fold: int, n_folds: int) -> Tuple[StockSequenceGenerator, StockSequenceGenerator, np.ndarray]:
        """Prepare data with new split when artifacts are not available."""
        # Try to find any artifacts to get the correct ticker count
        artifacts = self._find_configuration_artifacts(config)
        if artifacts is not None:
            tickers_to_use = artifacts['tickers']
            console.print(f"[dim]Using {len(tickers_to_use)} tickers from available artifacts[/dim]")
        else:
            # Use a reasonable subset of tickers (first 50) to avoid memory issues
            tickers_to_use = self.tickers[:50]
            console.print(f"[dim]Using {len(tickers_to_use)} tickers (subset)[/dim]")
        
        # Load all OHLCV data
        all_ohlcv_data = get_stock_data(tickers_to_use, period=self.config.data_period)
        num_sequences = len(all_ohlcv_data) - config['seq_length'] - config['time_window']
        indices = list(range(num_sequences))
        fold_size = num_sequences // n_folds
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else num_sequences
        train_indices = indices[:val_start] + indices[val_end:]
        val_indices = indices[val_start:val_end]

        # Calculate scaler mean/std from training data only
        close_prices = all_ohlcv_data.xs('Close', level=1, axis=1)[tickers_to_use]
        scaler_mean = close_prices.iloc[train_indices].values.mean()
        scaler_std = close_prices.iloc[train_indices].values.std()
        if scaler_std == 0:
            scaler_std = 1.0

        # Extract enhanced features configuration from config
        include_news = config.get('include_news', False)
        include_text = config.get('include_text', False)
        news_window = config.get('news_window', 7)
        text_model = config.get('text_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        train_generator = StockSequenceGenerator(
            sequence_indices_to_use=train_indices,
            all_ohlcv_data=all_ohlcv_data,
            seq_length=config['seq_length'],
            time_window=config['time_window'],
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            batch_size=config['batch_size'],
            shuffle_indices=True,
            tickers=tickers_to_use,
            include_news=include_news,
            include_text=include_text,
            news_window=news_window,
            text_model=text_model
        )
        test_generator = StockSequenceGenerator(
            sequence_indices_to_use=val_indices,
            all_ohlcv_data=all_ohlcv_data,
            seq_length=config['seq_length'],
            time_window=config['time_window'],
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            batch_size=config['batch_size'],
            shuffle_indices=False,
            tickers=tickers_to_use,
            include_news=include_news,
            include_text=include_text,
            news_window=news_window,
            text_model=text_model
        )
        # Calculate alpha weights
        alpha_weights = np.full(self.config.num_classes, 1.0 / self.config.num_classes)
        return train_generator, test_generator, alpha_weights
    
    def _train_fold(self, model: GPTClassifier, config: Dict[str, Any],
                   train_generator: StockSequenceGenerator,
                   test_generator: StockSequenceGenerator,
                   alpha_weights: np.ndarray, epochs: int, enhanced_input_features: int = None) -> Dict[str, Any]:
        """Train a single fold."""
        
        train_generator = PrefetchGenerator(train_generator)
        test_generator = PrefetchGenerator(test_generator)

        # Training loop with progress tracking
        best_accuracy = 0.0
        best_return = 0.0
        patience_counter = 0
        patience = 10
        
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
                    
                    # Debug: Print the batch shapes to understand the data format
                    if num_batches < 5:  # Only print for first batch
                        console.print(f"🔍 Batch shapes - x: {x.shape}, y: {y.shape}, r: {r.shape}, padding_mask: {padding_mask.shape}")
                        console.print(f"🔍 Expected input features: {enhanced_input_features}")
                        console.print(f"🔍 Model input_features: {model.input_features}")
                    
                    batch_tuple = (x, y, r, padding_mask)
                    try:
                        model, loss = train_step(model, batch_tuple, alpha_weights)
                    except Exception as e:
                        import traceback
                        console.print(f"[red]Exception in train_step: {e}[/red]")
                        traceback.print_exc()
                        raise
                    total_loss += float(loss)
                    num_batches += 1
                
                # Evaluation
                if epoch % 5 == 0:  # Evaluate every 5 epochs
                    metrics = evaluate_model(model, test_generator, model, alpha_weights)
                    accuracy = metrics['accuracy']
                    returns = metrics['simulated_return']
                    
                    # Early stopping
                    if accuracy > best_accuracy + 0.001:
                        best_accuracy = accuracy
                        best_return = returns
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        console.print(f"Early stopping at epoch {epoch}")
                        break
                
                progress.update(task, advance=1)
        
        return {
            'final_accuracy': best_accuracy,
            'final_return': best_return,
            'epochs_completed': epoch + 1
        }
    
    def run_extended_training(self, n_top: int = 5, epochs: int = 100):
        """Run extended training for top configurations."""
        
        console.print(Panel.fit(
            f"[bold green]Extended Training Pipeline[/bold green]\n"
            f"Top configurations: {n_top}\n"
            f"Epochs per config: {epochs}",
            title="🚀 Extended Training"
        ))
        
        # Get top configurations
        top_configs = self.get_top_configurations(n_top)
        
        if not top_configs:
            console.print("[red]No configurations found! Run hyperparameter tuning first.[/red]")
            return
        
        # Train each configuration
        extended_results = []
        
        for i, (config, original_accuracy) in enumerate(top_configs):
            console.print(f"\n[bold cyan]Configuration {i+1}/{len(top_configs)}[/bold cyan]")
            console.print(f"Original accuracy: {original_accuracy:.4f}")
            
            result = self.train_extended(config, epochs)
            extended_results.append(result)
        
        # Save results
        self._save_extended_results(extended_results)
        
        # Print summary
        self._print_summary(extended_results)
    
    def _save_extended_results(self, results: List[Dict]):
        """Save extended training results and artifacts."""
        # Create results directory
        results_dir = Path("./extended_training_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save results
        results_file = results_dir / "extended_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save artifacts for the best configuration
        if results:
            best_result = max(results, key=lambda x: x['avg_accuracy'])
            artifacts_file = results_dir / "best_configuration_artifacts.json"
            
            # Create artifacts for the best configuration (similar to step 1 structure)
            best_artifacts = {
                'config': best_result['config'],
                'avg_accuracy': best_result['avg_accuracy'],
                'avg_return': best_result['avg_return'],
                'std_accuracy': best_result['std_accuracy'],
                'epochs_trained': best_result['epochs_trained'],
                'cv_results': best_result['cv_results'],
                'timestamp': datetime.now().isoformat(),
                'data_period': self.config.data_period,
                'tickers': self.tickers  # Include full ticker list like step 1
            }
            
            with open(artifacts_file, 'w') as f:
                json.dump(best_artifacts, f, indent=2, default=str)
            
            console.print(f"[dim]Extended training artifacts saved to: {artifacts_file}[/dim]")
        
        console.print(f"[dim]Extended training results saved to: {results_file}[/dim]")
    
    def _print_summary(self, results: List[Dict]):
        """Print training summary."""
        table = Table(title="Extended Training Results")
        table.add_column("Config", style="cyan")
        table.add_column("Avg Accuracy", style="green")
        table.add_column("Std Accuracy", style="yellow")
        table.add_column("Avg Return", style="blue")
        table.add_column("Epochs", style="magenta")
        
        for i, result in enumerate(results):
            table.add_row(
                f"Config {i+1}",
                f"{result['avg_accuracy']:.4f}",
                f"{result['std_accuracy']:.4f}",
                f"{result['avg_return']:.4f}",
                str(result['epochs_trained'])
            )
        
        console.print(table)
        
        # Find best configuration
        best_idx = np.argmax([r['avg_accuracy'] for r in results])
        best_result = results[best_idx]
        
        console.print(f"\n[bold green]Best Configuration: Config {best_idx + 1}[/bold green]")
        console.print(f"Average Accuracy: {best_result['avg_accuracy']:.4f} ± {best_result['std_accuracy']:.4f}")
        console.print(f"Average Return: {best_result['avg_return']:.4f}")

    def _extract_step1_config(self):
        """Extract configuration from step 1 results to ensure consistency."""
        if self.trial_results:
            # Get the first trial to extract configuration
            first_trial = self.trial_results[0]
            if 'artifacts' in first_trial:
                artifacts = first_trial['artifacts']
                
                # Use the same data period and tickers from step 1
                if 'data_period' in artifacts:
                    self.config.data_period = artifacts['data_period']
                    console.print(f"[green]✅ Using data period from step 1: {self.config.data_period}[/green]")
                
                if 'tickers' in artifacts:
                    self.tickers = artifacts['tickers']
                    console.print(f"[green]✅ Using {len(self.tickers)} tickers from step 1[/green]")
                    
                    # Also update the ticker count in config
                    self.config.ticker_count = len(self.tickers)
                    console.print(f"[green]✅ Updated ticker count to: {self.config.ticker_count}[/green]")


def main():
    """Main function."""
    trainer = ExtendedTrainer()
    trainer.run_extended_training(n_top=5, epochs=100)


if __name__ == "__main__":
    main() 