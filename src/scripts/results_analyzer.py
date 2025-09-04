"""
Results Analyzer Script

This script analyzes results from hyperparameter tuning and extended training
to help select the best configuration for final training.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config.hyperparameter_config import HyperparameterConfig

console = Console()

class ResultsAnalyzer:
    """Analyzer for hyperparameter tuning and extended training results."""
    
    def __init__(self, hyperparameter_dir: str = "./hyperparameter_tuning_results", 
                 extended_dir: str = "./extended_training_results"):
        """
        Initialize the results analyzer.
        
        Args:
            hyperparameter_dir: Directory containing hyperparameter tuning results
            extended_dir: Directory containing extended training results
        """
        self.hyperparameter_dir = Path(hyperparameter_dir)
        self.extended_dir = Path(extended_dir)
        self.hyperparameter_results = None
        self.extended_results = None
        
        # Load results
        self._load_results()
    
    def _load_results(self):
        """Load hyperparameter tuning and extended training results."""
        # Load hyperparameter tuning results
        study_file = self.hyperparameter_dir / "optimization_study.pkl"
        if study_file.exists():
            with open(study_file, 'rb') as f:
                self.hyperparameter_results = pickle.load(f)
            console.print(f"[green]✅ Loaded hyperparameter tuning results from: {study_file}[/green]")
        
        # Load extended training results
        extended_file = self.extended_dir / "extended_training_results.json"
        if extended_file.exists():
            with open(extended_file, 'r') as f:
                self.extended_results = json.load(f)
            console.print(f"[green]✅ Loaded extended training results from: {extended_file}[/green]")
        
        if not self.hyperparameter_results and not self.extended_results:
            raise FileNotFoundError(f"No results found in {self.hyperparameter_dir} or {self.extended_dir}")
    
    def analyze_hyperparameter_results(self) -> Dict[str, Any]:
        """Analyze hyperparameter tuning results."""
        if not self.hyperparameter_results:
            console.print("[yellow]⚠️  No hyperparameter tuning results found[/yellow]")
            return {}
        
        console.print("\n[bold cyan]📊 Hyperparameter Tuning Analysis[/bold cyan]")
        
        # Get best trial
        best_trial = self.hyperparameter_results.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        # Get all trials
        trials = self.hyperparameter_results.trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Calculate statistics
        values = [t.value for t in completed_trials if t.value is not None]
        mean_value = np.mean(values) if values else 0
        std_value = np.std(values) if values else 0
        
        # Create summary table
        table = Table(title="Hyperparameter Tuning Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Trials", str(len(trials)))
        table.add_row("Completed Trials", str(len(completed_trials)))
        table.add_row("Best Accuracy", f"{best_value:.4f}")
        table.add_row("Mean Accuracy", f"{mean_value:.4f}")
        table.add_row("Std Accuracy", f"{std_value:.4f}")
        table.add_row("Improvement", f"{best_value - mean_value:.4f}")
        
        console.print(table)
        
        # Show best parameters
        console.print("\n[bold yellow]🏆 Best Configuration[/bold yellow]")
        param_table = Table(title="Best Hyperparameters")
        param_table.add_column("Parameter", style="cyan")
        param_table.add_column("Value", style="green")
        
        for param, value in best_params.items():
            if isinstance(value, float):
                param_table.add_row(param, f"{value:.4f}")
            else:
                param_table.add_row(param, str(value))
        
        console.print(param_table)
        
        # Get parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.hyperparameter_results)
            
            console.print("\n[bold yellow]📈 Parameter Importance[/bold yellow]")
            importance_table = Table(title="Hyperparameter Importance")
            importance_table.add_column("Parameter", style="cyan")
            importance_table.add_column("Importance", style="green")
            
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                importance_table.add_row(param, f"{imp:.4f}")
            
            console.print(importance_table)
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not calculate parameter importance: {e}[/yellow]")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'total_trials': len(trials),
            'completed_trials': len(completed_trials),
            'mean_value': mean_value,
            'std_value': std_value,
            'parameter_importance': importance if 'importance' in locals() else {}
        }
    
    def analyze_extended_results(self) -> Dict[str, Any]:
        """Analyze extended training results."""
        if not self.extended_results:
            console.print("[yellow]⚠️  No extended training results found[/yellow]")
            return {}
        
        console.print("\n[bold cyan]📊 Extended Training Analysis[/bold cyan]")
        
        # Create summary table
        table = Table(title="Extended Training Summary")
        table.add_column("Config", style="cyan")
        table.add_column("Avg Accuracy", style="green")
        table.add_column("Std Accuracy", style="yellow")
        table.add_column("Avg Return", style="blue")
        table.add_column("Epochs", style="magenta")
        
        best_config_idx = 0
        best_accuracy = 0
        
        for i, result in enumerate(self.extended_results):
            avg_acc = result['avg_accuracy']
            std_acc = result['std_accuracy']
            avg_ret = result['avg_return']
            epochs = result['epochs_trained']
            
            table.add_row(
                f"Config {i+1}",
                f"{avg_acc:.4f}",
                f"{std_acc:.4f}",
                f"{avg_ret:.4f}",
                str(epochs)
            )
            
            if avg_acc > best_accuracy:
                best_accuracy = avg_acc
                best_config_idx = i
        
        console.print(table)
        
        # Show best configuration details
        best_result = self.extended_results[best_config_idx]
        console.print(f"\n[bold green]🏆 Best Configuration: Config {best_config_idx + 1}[/bold green]")
        console.print(f"   [dim]Average Accuracy: {best_result['avg_accuracy']:.4f} ± {best_result['std_accuracy']:.4f}[/dim]")
        console.print(f"   [dim]Average Return: {best_result['avg_return']:.4f}[/dim]")
        console.print(f"   [dim]Epochs Trained: {best_result['epochs_trained']}[/dim]")
        
        # Show cross-validation details
        console.print(f"\n[bold yellow]📊 Cross-Validation Details[/bold yellow]")
        cv_table = Table(title=f"Config {best_config_idx + 1} Cross-Validation")
        cv_table.add_column("Fold", style="cyan")
        cv_table.add_column("Accuracy", style="green")
        cv_table.add_column("Return", style="blue")
        cv_table.add_column("Epochs", style="magenta")
        
        for i, cv_result in enumerate(best_result['cv_results']):
            cv_table.add_row(
                f"Fold {i+1}",
                f"{cv_result['final_accuracy']:.4f}",
                f"{cv_result['final_return']:.4f}",
                str(cv_result['epochs_completed'])
            )
        
        console.print(cv_table)
        
        return {
            'best_config': best_result['config'],
            'best_accuracy': best_result['avg_accuracy'],
            'best_std': best_result['std_accuracy'],
            'best_return': best_result['avg_return'],
            'best_config_idx': best_config_idx,
            'total_configs': len(self.extended_results)
        }
    
    def compare_results(self) -> Dict[str, Any]:
        """Compare hyperparameter tuning vs extended training results."""
        console.print("\n[bold cyan]📊 Results Comparison[/bold cyan]")
        
        hp_results = self.analyze_hyperparameter_results()
        ext_results = self.analyze_extended_results()
        
        if not hp_results or not ext_results:
            console.print("[yellow]⚠️  Cannot compare results - missing data[/yellow]")
            return {}
        
        # Create comparison table
        table = Table(title="Hyperparameter Tuning vs Extended Training")
        table.add_column("Metric", style="cyan")
        table.add_column("Hyperparameter Tuning", style="green")
        table.add_column("Extended Training", style="blue")
        table.add_column("Improvement", style="yellow")
        
        hp_acc = hp_results['best_value']
        ext_acc = ext_results['best_accuracy']
        improvement = ext_acc - hp_acc
        
        table.add_row("Best Accuracy", f"{hp_acc:.4f}", f"{ext_acc:.4f}", f"{improvement:+.4f}")
        table.add_row("Std Accuracy", "N/A", f"{ext_results['best_std']:.4f}", "N/A")
        table.add_row("Avg Return", "N/A", f"{ext_results['best_return']:.4f}", "N/A")
        
        console.print(table)
        
        # Recommendations
        console.print(f"\n[bold yellow]💡 Recommendations[/bold yellow]")
        
        if improvement > 0.01:
            console.print(f"   [green]✅ Extended training improved accuracy by {improvement:.4f}[/green]")
            console.print(f"   [green]✅ Use extended training configuration for final model[/green]")
        elif improvement > 0:
            console.print(f"   [yellow]⚠️  Extended training improved accuracy by {improvement:.4f}[/yellow]")
            console.print(f"   [yellow]⚠️  Consider using extended training configuration[/yellow]")
        else:
            console.print(f"   [red]❌ Extended training did not improve accuracy[/red]")
            console.print(f"   [red]❌ Consider using hyperparameter tuning configuration[/red]")
        
        return {
            'hyperparameter_results': hp_results,
            'extended_results': ext_results,
            'improvement': improvement,
            'recommendation': 'extended' if improvement > 0.01 else 'hyperparameter'
        }
    
    def save_best_config(self, output_file: str = "best_config.json"):
        """Save the best configuration to a JSON file."""
        comparison = self.compare_results()
        
        if not comparison:
            console.print("[red]❌ Cannot save best config - no comparison data[/red]")
            return
        
        # Determine best configuration
        if comparison['recommendation'] == 'extended':
            best_config = comparison['extended_results']['best_config']
            source = 'extended_training'
            accuracy = comparison['extended_results']['best_accuracy']
            
            # Get tickers from extended training artifacts
            tickers = []
            try:
                # Look for the best configuration artifacts file
                best_config_idx = comparison['extended_results']['best_config_idx']
                artifacts_file = self.extended_dir / f"best_configuration_artifacts.json"
                if artifacts_file.exists():
                    with open(artifacts_file, 'r') as f:
                        artifacts = json.load(f)
                        tickers = artifacts.get('tickers', [])
                        console.print(f"[green]✅ Found {len(tickers)} tickers from extended training artifacts[/green]")
                else:
                    console.print("[yellow]⚠️  Extended training artifacts not found, tickers will be auto-selected[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not load tickers from artifacts: {e}[/yellow]")
        else:
            best_config = comparison['hyperparameter_results']['best_params']
            source = 'hyperparameter_tuning'
            accuracy = comparison['hyperparameter_results']['best_value']
            
            # Get tickers from step 1 results (hyperparameter tuning)
            tickers = []
            try:
                # Load trial results to get tickers from step 1
                trial_results_file = self.hyperparameter_dir / "trial_results.json"
                if trial_results_file.exists():
                    with open(trial_results_file, 'r') as f:
                        trials = json.load(f)
                        if trials and 'artifacts' in trials[0]:
                            tickers = trials[0]['artifacts'].get('tickers', [])
                            console.print(f"[green]✅ Found {len(tickers)} tickers from step 1 results[/green]")
                        else:
                            console.print("[yellow]⚠️  No artifacts found in step 1 results[/yellow]")
                else:
                    console.print("[yellow]⚠️  Step 1 trial results not found[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not load tickers from step 1: {e}[/yellow]")
        
        # Extract split date and data period from step 1 results
        split_date = None
        data_period = None

        # If the best source is extended training, try to get artifacts from there first
        if source == 'extended_training':
            try:
                artifacts_file = self.extended_dir / "best_configuration_artifacts.json"
                if artifacts_file.exists():
                    with open(artifacts_file, 'r') as f:
                        artifacts = json.load(f)
                        split_date = artifacts.get('split_date')
                        data_period = artifacts.get('data_period')
                        console.print(f"[green]✅ Found split date from extended training: {split_date}[/green]")
                        console.print(f"[green]✅ Found data period from extended training: {data_period}[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not load extended training artifacts: {e}[/yellow]")

        # If not found in extended training artifacts, fall back to hyperparameter results
        if not data_period:
            try:
                if self.hyperparameter_results:
                    # Get split date from the first trial's artifacts
                    study_file = self.hyperparameter_dir / "optimization_study.pkl"
                    if study_file.exists():
                        with open(study_file, 'rb') as f:
                            study = pickle.load(f)
                            if study.trials:
                                best_trial = study.best_trial
                                if hasattr(best_trial, 'user_attrs') and 'artifacts' in best_trial.user_attrs:
                                    artifacts = best_trial.user_attrs['artifacts']
                                    split_date = artifacts.get('split_date')
                                    data_period = artifacts.get('data_period')
                                    console.print(f"[green]✅ Found split date from step 1: {split_date}[/green]")
                                    console.print(f"[green]✅ Found data period from step 1: {data_period}[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not extract split date from step 1: {e}[/yellow]")
        
        # Add metadata
        config_with_metadata = {
            'config': best_config,
            'source': source,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
            'analysis_metadata': {
                'hyperparameter_trials': comparison['hyperparameter_results'].get('total_trials', 0),
                'extended_configs': comparison['extended_results'].get('total_configs', 0),
                'improvement': comparison['improvement']
            }
        }
        
        # Add tickers if available
        if tickers:
            config_with_metadata['tickers'] = tickers
        
        # Add data period and split date if available
        if data_period:
            config_with_metadata['data_period'] = data_period
        else:
            config_with_metadata['data_period'] = '1y'  # Default to step 1's period
        
        if split_date:
            config_with_metadata['split_date'] = split_date
        
        # Save to file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(config_with_metadata, f, indent=2)
        
        console.print(f"\n[bold green]✅ Best configuration saved to: {output_path}[/bold green]")
        console.print(f"   [dim]Source: {source}[/dim]")
        console.print(f"   [dim]Accuracy: {accuracy:.4f}[/dim]")
        if tickers:
            console.print(f"   [dim]Tickers: {len(tickers)} stocks[/dim]")
        if split_date:
            console.print(f"   [dim]Split Date: {split_date}[/dim]")
        if data_period:
            console.print(f"   [dim]Data Period: {data_period}[/dim]")
        
        return output_path
    
    def generate_plots(self, output_dir: str = "./analysis_plots"):
        """Generate analysis plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"\n[cyan]📊 Generating analysis plots...[/cyan]")
        
        # Hyperparameter optimization history
        if self.hyperparameter_results:
            plt.figure(figsize=(12, 8))
            
            # Optimization history
            plt.subplot(2, 2, 1)
            values = [t.value for t in self.hyperparameter_results.trials if t.value is not None]
            plt.plot(values)
            plt.title('Hyperparameter Optimization History')
            plt.xlabel('Trial')
            plt.ylabel('Accuracy')
            plt.grid(True)
            
            # Parameter importance
            try:
                importance = optuna.importance.get_param_importances(self.hyperparameter_results)
                plt.subplot(2, 2, 2)
                params = list(importance.keys())
                values = list(importance.values())
                plt.barh(params, values)
                plt.title('Parameter Importance')
                plt.xlabel('Importance')
                plt.grid(True)
            except:
                pass
            
            # Extended training comparison
            if self.extended_results:
                plt.subplot(2, 2, 3)
                configs = [f"Config {i+1}" for i in range(len(self.extended_results))]
                accuracies = [r['avg_accuracy'] for r in self.extended_results]
                stds = [r['std_accuracy'] for r in self.extended_results]
                
                plt.bar(configs, accuracies, yerr=stds, capsize=5)
                plt.title('Extended Training Results')
                plt.xlabel('Configuration')
                plt.ylabel('Accuracy')
                plt.xticks(rotation=45)
                plt.grid(True)
            
            plt.tight_layout()
            plot_file = output_dir / "analysis_summary.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            console.print(f"[green]✅ Analysis plots saved to: {plot_file}[/green]")


def main():
    """Main function for results analysis."""
    parser = argparse.ArgumentParser(description="Analyze hyperparameter tuning and extended training results")
    parser.add_argument("--hyperparameter-dir", type=str, default="./hyperparameter_tuning_results",
                       help="Directory containing hyperparameter tuning results (default: ./hyperparameter_tuning_results)")
    parser.add_argument("--extended-dir", type=str, default="./extended_training_results",
                       help="Directory containing extended training results (default: ./extended_training_results)")
    parser.add_argument("--output-config", type=str, default="best_config.json",
                       help="Output file for best configuration (default: best_config.json)")
    parser.add_argument("--plots-dir", type=str, default="./analysis_plots",
                       help="Directory for analysis plots (default: ./analysis_plots)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Disable plot generation")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ResultsAnalyzer(args.hyperparameter_dir, args.extended_dir)
        
        # Run analysis
        comparison = analyzer.compare_results()
        
        # Save best configuration
        config_file = analyzer.save_best_config(args.output_config)
        
        # Generate plots
        if not args.no_plots:
            analyzer.generate_plots(args.plots_dir)
        
        console.print(f"\n[bold green]🎉 Analysis completed successfully![/bold green]")
        console.print(f"[dim]Best configuration saved to: {config_file}[/dim]")
        
        # Instructions for next steps
        console.print("\n[bold yellow]🚀 Next Steps:[/bold yellow]")
        console.print(f"   [bold cyan]Train Final Model:[/bold cyan] python run_final_training.py --config {config_file}")
        console.print(f"   [bold cyan]Backtest Model:[/bold cyan] python run_backtesting.py --model-path ./final_model")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Analysis failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 