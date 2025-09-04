"""
Model Evaluator Script

This script provides comprehensive evaluation of trained models
with detailed performance metrics and analysis.
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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
from flax import nnx
import optax

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.gpt_classifier import GPTClassifier
from models.constants import NUM_CLASSES
from data.stock_data import get_stock_data, get_large_cap_tickers
from data.sequence_generator import StockSequenceGenerator
from training.training_functions import evaluate_model, _init_gpu_optimization

from utils.gpu_utils import check_gpu_availability

console = Console()

class ModelEvaluator:
    """Comprehensive model evaluator for trained models."""
    
    def __init__(self, model_path: str):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = Path(model_path)
        
        # Load model and configuration
        self.state, self.model, self.config = self._load_model()
        
        # Initialize GPU optimization
        _init_gpu_optimization()
        
        # Get system info
        gpu_info = check_gpu_availability()
        if gpu_info['available']:
            console.print(f"[green]✅ GPU Available: {gpu_info['device_name']}[/green]")
        else:
            console.print("[yellow]⚠️  GPU not available, using CPU[/yellow]")
    
    def _load_model(self) -> tuple:
        """Load the trained model and configuration."""
        console.print(f"[cyan]📂 Loading model from: {self.model_path}[/cyan]")
        
        # Try to load configuration
        config_file = self.model_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            # Try to load from training results
            results_file = self.model_path / "training_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                config = results.get('configuration', {})
            else:
                # Use default configuration
                config = self._get_default_config()
        
        # Create model instance
        model = GPTClassifier(
            num_classes=config.get('num_classes', NUM_CLASSES),
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 4),
            d_ff=config.get('d_ff', 1024),
            dropout_rate=config.get('dropout_rate', 0.1),
            input_features=len(config.get('tickers', get_large_cap_tickers()[:50])),
            num_tickers=len(config.get('tickers', get_large_cap_tickers()[:50])),
            rngs=nnx.Rngs(0)
        )
        
        # Load model state using unified checkpointing
        from training.checkpointing import restore_checkpoint
        
        # Create model kwargs for restoration
        model_kwargs = {
            'num_classes': config.get('num_classes', NUM_CLASSES),
            'd_model': config.get('d_model', 256),
            'num_heads': config.get('num_heads', 8),
            'num_layers': config.get('num_layers', 4),
            'd_ff': config.get('d_ff', 1024),
            'dropout_rate': config.get('dropout_rate', 0.1),
            'input_features': len(config.get('tickers', get_large_cap_tickers()[:50])),
            'rngs': nnx.Rngs(0)
        }
        
        params, state, scaler_mean, scaler_std, restored_step, restored_lr, opt_state = restore_checkpoint(
            self.model_path, GPTClassifier, model_kwargs,
            learning_rate=1e-3,
            create_optimizer_fn=lambda m, lr: nnx.Optimizer(m, optax.adam(lr))
        )
        
        if state is None:
            raise FileNotFoundError(f"Failed to restore checkpoint from {self.model_path}")
        
        # Merge state with model
        from flax.nnx import merge
        model = merge(model, state)
        
        console.print("[green]\u2705 Model loaded successfully![green]")
        return state, model, config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for evaluation."""
        return {
            'tickers': get_large_cap_tickers()[:50],
            'data_period': '2y',
            'seq_length': 60,
            'time_window': 14,
            'num_classes': NUM_CLASSES,
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4,
            'd_ff': 1024,
            'dropout_rate': 0.1,
            'batch_size': 64,
            'learning_rate': 1e-4
        }
    
    def evaluate_on_test_data(self, 
                            tickers: Optional[List[str]] = None,
                            data_period: str = '1y',
                            test_split: float = 0.2) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            tickers: List of tickers to evaluate on
            data_period: Data period for evaluation
            test_split: Test split ratio
            
        Returns:
            Dictionary containing evaluation results
        """
        tickers = tickers or self.config.get('tickers', get_large_cap_tickers()[:50])
        
        console.print(f"\n[cyan]📊 Evaluating on test data...[/cyan]")
        console.print(f"   [dim]Tickers: {len(tickers)}[/dim]")
        console.print(f"   [dim]Data Period: {data_period}[/dim]")
        console.print(f"   [dim]Test Split: {test_split:.1%}[/dim]")
        
        # Create test data generator
        test_generator = StockSequenceGenerator(
            tickers=tickers,
            time_window=self.config.get('time_window', 14),
            seq_length=self.config.get('seq_length', 60),
            batch_size=self.config.get('batch_size', 64),
            train_test_split=1.0 - test_split,  # Use last portion as test
            data_period=data_period,
            is_training=False
        )
        
        # Calculate alpha weights
        alpha_weights = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        
        # Evaluate model
        metrics = evaluate_model(self.state, test_generator, self.model, alpha_weights)
        
        console.print(f"[green]✅ Test evaluation completed[/green]")
        
        return metrics
    
    def evaluate_on_validation_data(self, 
                                  tickers: Optional[List[str]] = None,
                                  data_period: str = '1y') -> Dict[str, Any]:
        """
        Evaluate model on validation data.
        
        Args:
            tickers: List of tickers to evaluate on
            data_period: Data period for evaluation
            
        Returns:
            Dictionary containing validation results
        """
        tickers = tickers or self.config.get('tickers', get_large_cap_tickers()[:50])
        
        console.print(f"\n[cyan]📊 Evaluating on validation data...[/cyan]")
        
        # Create validation data generator (middle portion of data)
        val_generator = StockSequenceGenerator(
            tickers=tickers,
            time_window=self.config.get('time_window', 14),
            seq_length=self.config.get('seq_length', 60),
            batch_size=self.config.get('batch_size', 64),
            train_test_split=0.8,  # Use middle portion as validation
            data_period=data_period,
            is_training=False
        )
        
        # Calculate alpha weights
        alpha_weights = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        
        # Evaluate model
        metrics = evaluate_model(self.state, val_generator, self.model, alpha_weights)
        
        console.print(f"[green]✅ Validation evaluation completed[/green]")
        
        return metrics
    
    def evaluate_on_recent_data(self, 
                              tickers: Optional[List[str]] = None,
                              data_period: str = '3mo') -> Dict[str, Any]:
        """
        Evaluate model on recent data.
        
        Args:
            tickers: List of tickers to evaluate on
            data_period: Data period for evaluation
            
        Returns:
            Dictionary containing recent data results
        """
        tickers = tickers or self.config.get('tickers', get_large_cap_tickers()[:50])
        
        console.print(f"\n[cyan]📊 Evaluating on recent data...[/cyan]")
        console.print(f"   [dim]Data Period: {data_period}[/dim]")
        
        # Create recent data generator
        recent_generator = StockSequenceGenerator(
            tickers=tickers,
            time_window=self.config.get('time_window', 14),
            seq_length=self.config.get('seq_length', 60),
            batch_size=self.config.get('batch_size', 64),
            train_test_split=0.0,  # Use all recent data
            data_period=data_period,
            is_training=False
        )
        
        # Calculate alpha weights
        alpha_weights = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        
        # Evaluate model
        metrics = evaluate_model(self.state, recent_generator, self.model, alpha_weights)
        
        console.print(f"[green]✅ Recent data evaluation completed[/green]")
        
        return metrics
    
    def generate_performance_report(self, 
                                  output_dir: str = "./evaluation_results") -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing all evaluation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(Panel.fit(
            f"[bold green]Comprehensive Model Evaluation[/bold green]\n"
            f"Model: {self.model_path}\n"
            f"Output: {output_dir}",
            title="🔍 Model Evaluation"
        ))
        
        # Run evaluations on different datasets
        test_metrics = self.evaluate_on_test_data()
        val_metrics = self.evaluate_on_validation_data()
        recent_metrics = self.evaluate_on_recent_data()
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'configuration': self.config,
            'evaluations': {
                'test_data': test_metrics,
                'validation_data': val_metrics,
                'recent_data': recent_metrics
            }
        }
        
        # Save results
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        self.print_evaluation_summary(results)
        
        console.print(f"\n[bold green]✅ Evaluation completed![/bold green]")
        console.print(f"[dim]Results saved to: {results_file}[/dim]")
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results."""
        console.print("\n[bold cyan]📊 Evaluation Results Summary[/bold cyan]")
        
        # Create summary table
        table = Table(title="Model Performance Across Datasets")
        table.add_column("Dataset", style="cyan")
        table.add_column("Accuracy", style="green")
        table.add_column("Loss", style="yellow")
        table.add_column("Simulated Return", style="blue")
        
        evaluations = results['evaluations']
        
        for dataset_name, metrics in evaluations.items():
            table.add_row(
                dataset_name.replace('_', ' ').title(),
                f"{metrics.get('accuracy', 0):.4f}",
                f"{metrics.get('loss', 0):.4f}",
                f"{metrics.get('simulated_return', 0):.4f}"
            )
        
        console.print(table)
        
        # Performance analysis
        console.print(f"\n[bold yellow]📈 Performance Analysis[/bold yellow]")
        
        test_acc = evaluations['test_data'].get('accuracy', 0)
        val_acc = evaluations['validation_data'].get('accuracy', 0)
        recent_acc = evaluations['recent_data'].get('accuracy', 0)
        
        # Check for overfitting
        if abs(test_acc - val_acc) > 0.05:
            console.print(f"   [red]⚠️  Potential overfitting detected (test: {test_acc:.4f}, val: {val_acc:.4f})[/red]")
        else:
            console.print(f"   [green]✅ Good generalization (test: {test_acc:.4f}, val: {val_acc:.4f})[/green]")
        
        # Check recent performance
        if recent_acc < min(test_acc, val_acc) - 0.05:
            console.print(f"   [red]⚠️  Recent performance degradation detected ({recent_acc:.4f})[/red]")
        else:
            console.print(f"   [green]✅ Recent performance is stable ({recent_acc:.4f})[/green]")
        
        # Model complexity analysis
        console.print(f"\n[bold yellow]🧠 Model Complexity Analysis[/bold yellow]")
        
        config = results['configuration']
        total_params = (
            config.get('num_layers', 4) * 
            config.get('d_model', 256) * 
            config.get('d_ff', 1024) * 4  # Rough estimate
        )
        
        console.print(f"   [dim]Layers: {config.get('num_layers', 4)}[/dim]")
        console.print(f"   [dim]Model Dimension: {config.get('d_model', 256)}[/dim]")
        console.print(f"   [dim]Attention Heads: {config.get('num_heads', 8)}[/dim]")
        console.print(f"   [dim]FF Dimension: {config.get('d_ff', 1024)}[/dim]")
        console.print(f"   [dim]Estimated Parameters: ~{total_params:,}[/dim]")
    
    def generate_plots(self, results: Dict[str, Any], output_dir: str = "./evaluation_plots"):
        """Generate evaluation plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"\n[cyan]📊 Generating evaluation plots...[/cyan]")
        
        evaluations = results['evaluations']
        
        # Create performance comparison plot
        plt.figure(figsize=(12, 8))
        
        # Accuracy comparison
        plt.subplot(2, 2, 1)
        datasets = list(evaluations.keys())
        accuracies = [evaluations[d].get('accuracy', 0) for d in datasets]
        plt.bar(datasets, accuracies)
        plt.title('Accuracy Across Datasets')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Loss comparison
        plt.subplot(2, 2, 2)
        losses = [evaluations[d].get('loss', 0) for d in datasets]
        plt.bar(datasets, losses)
        plt.title('Loss Across Datasets')
        plt.ylabel('Loss')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Return comparison
        plt.subplot(2, 2, 3)
        returns = [evaluations[d].get('simulated_return', 0) for d in datasets]
        plt.bar(datasets, returns)
        plt.title('Simulated Returns Across Datasets')
        plt.ylabel('Return')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Model configuration
        plt.subplot(2, 2, 4)
        config = results['configuration']
        config_items = ['num_layers', 'd_model', 'num_heads', 'd_ff']
        config_values = [config.get(item, 0) for item in config_items]
        plt.bar(config_items, config_values)
        plt.title('Model Configuration')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / "evaluation_summary.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]✅ Evaluation plots saved to: {plot_file}[/green]")


def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained model with comprehensive metrics")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--tickers", type=str, nargs='+', default=None,
                       help="Tickers to evaluate on (default: use config tickers)")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Output directory for evaluation results (default: ./evaluation_results)")
    parser.add_argument("--plots-dir", type=str, default="./evaluation_plots",
                       help="Directory for evaluation plots (default: ./evaluation_plots)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Disable plot generation")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.model_path)
        
        # Generate performance report
        results = evaluator.generate_performance_report(args.output_dir)
        
        # Generate plots
        if not args.no_plots:
            evaluator.generate_plots(results, args.plots_dir)
        
        console.print(f"\n[bold green]🎉 Model evaluation completed successfully![/bold green]")
        console.print(f"[dim]Results saved to: {args.output_dir}[/dim]")
        
        # Instructions for next steps
        console.print("\n[bold yellow]🚀 Next Steps:[/bold yellow]")
        console.print(f"   [bold cyan]Backtest Model:[/bold cyan] python run_backtesting.py --model-path {args.model_path}")
        console.print(f"   [bold cyan]Generate Signals:[/bold cyan] python run_backtesting.py --model-path {args.model_path} --generate-signals")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Model evaluation failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 