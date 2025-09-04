"""
Hyperparameter optimization utilities for the JAX GPT Stock Predictor.

Contains optimization algorithms, trial management, and result analysis.
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances
import jax
import jax.profiler as profiler
import jax.numpy as jnp
from flax.nnx import TrainState, value_and_grad
import optax
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.rule import Rule
from rich.columns import Columns
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import io
import orbax.checkpoint as ocp
import optuna.importance
from flax import nnx

# Import from refactored modules
from ..models.gpt_classifier import GPTClassifier
from ..models.constants import NUM_CLASSES
from ..data.stock_data import get_stock_data, get_large_cap_tickers
from ..data.sequence_generator import StockSequenceGenerator, PrefetchGenerator
from ..training.training_functions import train_step, evaluate_model, _init_gpu_optimization

from ..config.hyperparameter_config import HyperparameterConfig
from ..utils.gpu_utils import check_gpu_availability, get_gpu_utilization, NVML_AVAILABLE
from ..utils.system_utils import get_system_info

# TensorBoard imports
try:
    import tensorflow as tf
    from tensorboard.plugins.scalar import summary_v2 as scalar_summary
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    tf = None  # Set tf to None when not available
    print("Warning: TensorBoard not available. Install tensorflow and tensorboard for logging.")

# Initialize Rich console
console = Console()

class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.001, overfitting_threshold=0.15):
        self.patience = patience
        self.min_delta = min_delta
        self.overfitting_threshold = overfitting_threshold
        self.best_value = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        self.history = []
        self.train_history = []  # Track training metrics for overfitting detection

    def __call__(self, epoch, value, train_value=None, *args, **kwargs):
        self.history.append(value)
        if train_value is not None:
            self.train_history.append(train_value)
        
        if self.best_value is None or value > self.best_value + self.min_delta:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        
        # Enhanced overfitting detection
        if len(self.history) > self.patience:
            # Method 1: Simple range check (original)
            recent = self.history[-self.patience:]
            range_check = max(recent) - min(recent) > self.overfitting_threshold
            
            # Method 2: Trend analysis - check if validation is declining while training improves
            trend_check = False
            if len(self.train_history) > self.patience and len(self.history) > self.patience:
                recent_val = np.array(self.history[-self.patience:])
                recent_train = np.array(self.train_history[-self.patience:])
                
                # Calculate proper linear regression trends
                x = np.arange(len(recent_val))
                x_mean = np.mean(x)
                
                # Validation trend calculation
                val_mean = np.mean(recent_val)
                val_numerator = np.sum((x - x_mean) * (recent_val - val_mean))
                val_denominator = np.sum((x - x_mean) ** 2)
                val_trend = val_numerator / val_denominator if val_denominator != 0 else 0
                
                # Training trend calculation
                train_mean = np.mean(recent_train)
                train_numerator = np.sum((x - x_mean) * (recent_train - train_mean))
                train_denominator = np.sum((x - x_mean) ** 2)
                train_trend = train_numerator / train_denominator if train_denominator != 0 else 0
                
                # Validation: ensure trend calculations are valid
                assert not np.isnan(val_trend), "Validation trend calculation produced NaN"
                assert not np.isnan(train_trend), "Training trend calculation produced NaN"
                
                # Overfitting: validation declining while training improving
                trend_check = val_trend < -0.001 and train_trend > 0.001
            
            # Method 3: Rolling variance check - high variance suggests instability
            variance_check = False
            if len(self.history) >= 10:
                recent_10 = np.array(self.history[-10:])
                variance = np.var(recent_10)
                # Normalize variance using coefficient of variation (relative variance)
                mean_val = np.mean(recent_10)
                normalized_variance = variance / (mean_val ** 2) if mean_val != 0 else 0
                variance_check = normalized_variance > 0.1  # 10% relative variance threshold
            
            # Combine detection methods
            self.early_stop = range_check or trend_check or variance_check
            
            if self.early_stop:
                reason = []
                if range_check:
                    reason.append("range")
                if trend_check:
                    reason.append("trend")
                if variance_check:
                    reason.append("variance")
                print(f"Early stopping triggered by: {', '.join(reason)}")
        
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

class HyperparameterOptimizer:
    """Main hyperparameter optimization class with 3-phase approach as specified in PROJECT_INTENT.md"""
    
    def __init__(self, config: HyperparameterConfig, tickers: List[str], save_dir: str = "./hyperparameter_tuning_results"):
        self.config = config
        self.tickers = tickers
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU optimization at the start
        _init_gpu_optimization()
        
        # Setup Optuna study with TPE sampler for Bayesian optimization
        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            storage="sqlite:///optuna_study.db",  # <-- Add this line
            study_name="my_study",                # Optional: name your study
            load_if_exists=True                   # Optional: resume if exists
        )
        
        # Results storage
        self.trial_results = []
        self.best_trial = None
        self.optimization_history = []
        
        # Performance tracking
        self.start_time = None
        self.trial_times = []
        self.memory_usage = []
        
        # Phase tracking
        self.current_phase = 1 # Start directly at Phase 1
        self.phase_results = {
            'phase_1': {'trials': 0, 'best_accuracy': 0.0, 'best_return': 0.0},
            'phase_2': {'trials': 0, 'best_accuracy': 0.0, 'best_return': 0.0},
            'phase_3': {'trials': 0, 'best_accuracy': 0.0, 'best_return': 0.0}
        }
    
    def _get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB with precise calculation."""
        mem_bytes = psutil.virtual_memory().used
        return mem_bytes / (1024**3)  # Convert bytes to GB

    def _plot_confusion_matrix_to_image(self, confusion_matrix):
        """Renders a confusion matrix matplotlib figure to a PNG image tensor for TensorBoard."""
        class_names = ['HOLD', 'BUY_CALL', 'BUY_PUT']
        figure = plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        
        # Convert PNG buffer to TF image
        if tf is not None:
            image = tf.image.decode_png(buf.getvalue())
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image
        return None

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization"""
        try:
            # Sample hyperparameters
            params = self._sample_hyperparameters(trial)
            
            # Create model with sampled parameters
            model = self._create_model(params)
            
            # Train and evaluate
            metrics, artifacts = self._train_and_evaluate(trial, model, params)
            
            # Store results
            self._store_trial_results(trial, params, metrics, artifacts)
            
            return metrics['validation_accuracy']
            
        except Exception as e:
            console.print(f"[bold red]Trial {trial.number} failed: {e}[/bold red]")
            import traceback
            traceback.print_exception(e)
            return 0.0
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters from the search space"""
        params = {}
        
        # Process each parameter in the search space
        if self.config.search_space is not None:
            for param_name, param_config in self.config.search_space.items():
                # Check if it's a continuous parameter (tuple with 3 values)
                if isinstance(param_config, tuple) and len(param_config) == 3:
                    min_val, max_val, dist_type = param_config
                    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                        if dist_type == 'log':
                            params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=True)
                        elif dist_type == 'uniform':
                            params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                        else:
                            params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                
                # Check if it's a discrete parameter (list of options)
                elif isinstance(param_config, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
        
        return params
    
    def _create_model(self, params: Dict[str, Any]) -> GPTClassifier:
        """Create model with given hyperparameters"""
        # Calculate enhanced input features
        # Base features: 1 price feature per ticker
        features_per_ticker = 1
        
        if params.get('include_news', False):
            features_per_ticker += 10  # News features per ticker
        
        if params.get('include_text', False):
            features_per_ticker += 384  # Text features per ticker
        
        # Total input features = num_tickers * features_per_ticker
        enhanced_input_features = len(self.tickers) * features_per_ticker
        
        return GPTClassifier(
            num_layers=params['num_layers'],
            d_model=params['d_model'],
            num_heads=params['num_heads'],
            d_ff=params['d_ff'],
            num_classes=params.get('num_classes', 3),
            dropout_rate=params.get('dropout_rate', 0.1),
            input_features=enhanced_input_features,
            num_tickers=len(self.tickers),
            rngs=nnx.Rngs(0),
            learning_rate=params.get('learning_rate', 1e-4)
        )
    
    def _train_and_evaluate(self, trial: optuna.Trial, model: GPTClassifier, hyperparams: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Train and evaluate model with given parameters, with TensorBoard and early stopping"""
        # Create a fresh model for each trial
        model = self._create_model(hyperparams)
        train_generator, test_generator, alpha_weights, artifacts = self._prepare_data_generators(hyperparams)
        
        train_generator = PrefetchGenerator(train_generator)
        test_generator = PrefetchGenerator(test_generator)
        
        epochs = self._get_epochs_for_trial(trial)
        best_accuracy = 0.0
        best_state = None
        best_opt_state = None
        best_epoch = 0
        best_confusion_matrix = None
        scaler_mean = artifacts.get('scaler_mean', 0.0)
        scaler_std = artifacts.get('scaler_std', 1.0)
        early_stopper = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            overfitting_threshold=self.config.early_stopping_overfitting_threshold
        )
        tb_writer = None
        log_dir = str(Path(__file__).parent.parent.parent / "tensorboard_logs")
        os.makedirs(log_dir, exist_ok=True)
        run_name = f"trial_{trial.number}_{int(time.time())}"
        run_log_dir = str(Path(log_dir) / run_name)
        if TENSORBOARD_AVAILABLE and tf is not None:
            tb_writer = tf.summary.create_file_writer(run_log_dir)
            with tb_writer.as_default():
                for k, v in hyperparams.items():
                    tf.summary.text(f"hyperparam/{k}", str(v), step=0)
                tf.summary.text("trial/metadata", f"Trial {trial.number} - Phase {self.current_phase}", step=0)
                tf.summary.scalar("trial/phase", self.current_phase, step=0)
                tf.summary.scalar("trial/number", trial.number, step=0)
        # Initialize variables
        best_accuracy = 0.0
        best_model = None  # Store best model for checkpointing
        best_epoch = 0
        global_step = 0
        val_loss = 0.0  # Initialize val_loss to prevent UnboundLocalError
        total_batches_per_epoch = len(train_generator)
        early_stopper = EarlyStopping(patience=hyperparams.get('patience', 8))
        
        try:
            for epoch in range(epochs):
                gc.collect()
                jax.clear_caches()

                model.train()

                train_loss_sum = 0.0
                batch_count_actual = 0
                epoch_start_time = time.time()
                if tb_writer and tf is not None:
                    with tb_writer.as_default():
                        tf.summary.scalar("epoch/current", epoch, step=global_step)
                        tf.summary.scalar("epoch/total", epochs, step=global_step)
                for batch_idx, batch in enumerate(train_generator):
                    batch_start_time = time.time()
                    batch_count_actual += 1
                    # Use NNX-idiomatic train_step for each batch
                    model, loss = train_step(model, batch, alpha_weights)
                    train_loss_sum += loss
                    batch_time = time.time() - batch_start_time
                    batch_loss = loss
                    cpu_util = psutil.cpu_percent()
                    mem = self._get_memory_usage_gb()
                    gpu_util = 0.0
                    if NVML_AVAILABLE:
                        try:
                            gpu_util = get_gpu_utilization()
                        except Exception:
                            gpu_util = 0.0
                    if tb_writer and tf is not None:
                        with tb_writer.as_default():
                            tf.summary.scalar("train/batch_loss", batch_loss, step=global_step)
                            tf.summary.scalar("train/batch_time_seconds", batch_time, step=global_step)
                            tf.summary.scalar("train/batch_idx", batch_idx, step=global_step)
                            tf.summary.scalar("train/batches_per_epoch", total_batches_per_epoch, step=global_step)
                            tf.summary.scalar("system/cpu_utilization_percent", cpu_util, step=global_step)
                            tf.summary.scalar("system/ram_usage_gb", mem, step=global_step)
                            tf.summary.scalar("system/gpu_utilization_percent", gpu_util, step=global_step)
                            tf.summary.scalar("progress/epoch", epoch, step=global_step)
                            tf.summary.scalar("progress/batch_in_epoch", batch_idx, step=global_step)
                            tf.summary.scalar("progress/global_step", global_step, step=global_step)
                            tf.summary.scalar("progress/completion_percent", (epoch * total_batches_per_epoch + batch_idx) / (epochs * total_batches_per_epoch) * 100, step=global_step)
                            tf.summary.scalar("trial/current_epoch", epoch, step=global_step)
                            tf.summary.scalar("trial/current_batch", batch_idx, step=global_step)
                            tb_writer.flush()
                    global_step += 1
                epoch_time = time.time() - epoch_start_time
                train_loss = train_loss_sum / batch_count_actual if batch_count_actual > 0 else 0.0
                
                # Validation - use the model directly
                gc.collect()
                jax.clear_caches()
                model.eval()
                
                # Run validation on test generator
                val_metrics = evaluate_model(
                    model,  # state (not used in NNX)
                    test_generator,
                    model,
                    alpha_weights,
                    progress=None,
                    step_progress_task=None,
                    tb_writer=tb_writer,
                    global_step=global_step
                )
                val_acc = val_metrics.get('accuracy', 0.0)
                val_loss = val_metrics.get('loss', 0.0)
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    # Save best model for checkpointing
                    best_model = model
                    best_epoch = epoch
                    best_confusion_matrix = val_metrics.get('confusion_matrix')
                if tb_writer and tf is not None:
                    with tb_writer.as_default():
                        tf.summary.scalar("train/epoch_loss", train_loss, step=global_step)
                        tf.summary.scalar("train/epoch_time_seconds", epoch_time, step=global_step)
                        tf.summary.scalar("train/batches_processed", batch_count_actual, step=global_step)
                        tf.summary.scalar("val/accuracy", val_acc, step=global_step)
                        tf.summary.scalar("val/loss", val_loss, step=global_step)
                        tf.summary.scalar("performance/best_accuracy_so_far", best_accuracy, step=global_step)
                        tf.summary.scalar("performance/epochs_since_best", epoch - best_epoch if epoch >= best_epoch else 0, step=global_step)
                        tf.summary.scalar("trial/epoch", epoch, step=global_step)
                        tf.summary.scalar("trial/best_epoch", best_epoch, step=global_step)
                        tf.summary.scalar("trial/best_accuracy", best_accuracy, step=global_step)
                        tb_writer.flush()
                if early_stopper(epoch, val_acc, train_loss):
                    console.print(f"[bold yellow]Stopping early at epoch {epoch+1}[/bold yellow]")
                    if tb_writer and tf is not None:
                        with tb_writer.as_default():
                            tf.summary.scalar("training/early_stopped", 1, step=global_step)
                            tf.summary.text("training/stop_reason", "Early stopping triggered", step=global_step)
                    break
        except Exception as e:
            console.print(f"[bold red]Error during training: {e}[/bold red]")
            profiler.save_device_memory_profile("error.prof")
            import traceback
            traceback.print_exc()
            if tb_writer and tf is not None:
                with tb_writer.as_default():
                    tf.summary.scalar("training/error_occurred", 1, step=global_step)
                    tf.summary.text("training/error_message", str(e), step=global_step)
        # Save best model checkpoint (if needed)
        checkpoint_path = None
        if best_model is not None:
            ckpt_dir = (self.save_dir / "checkpoints" / str(trial.number)).absolute()
            from ..training.checkpointing import save_checkpoint
            # Save model and scalers (no optimizer state needed for simplified approach)
            checkpoint_path = save_checkpoint(
                ckpt_dir, best_model, None, scaler_mean, scaler_std, global_step, hyperparams['learning_rate']
            )
            if tb_writer and tf is not None:
                with tb_writer.as_default():
                    tf.summary.text("checkpoint/path", checkpoint_path, step=global_step)
                    tf.summary.scalar("checkpoint/saved", 1, step=global_step)
        # Log the best confusion matrix to TensorBoard at trial.number
        if tb_writer and tf is not None and best_confusion_matrix is not None:
            with tb_writer.as_default():
                cm_image = self._plot_confusion_matrix_to_image(best_confusion_matrix)
                if cm_image is not None:
                    tf.summary.image(f"Val-Confusion-Matrix-Best", cm_image, step=trial.number)
        if tb_writer:
            tb_writer.close()
        training_loss_avg = 0.0
        if batch_count_actual > 0:
            training_loss_avg = train_loss_sum / float(batch_count_actual)
        return {
            'validation_accuracy': best_accuracy,
            'final_loss': val_loss,
            'training_loss': training_loss_avg,
            'best_epoch': best_epoch,
            'checkpoint_path': checkpoint_path
        }, artifacts
    
    def _prepare_data_generators(self, params: Dict[str, Any]) -> Tuple[StockSequenceGenerator, StockSequenceGenerator, jnp.ndarray, Dict[str, Any]]:
        """Prepare data generators for training and return artifacts for later reuse"""
        # Load data
        all_tickers_data = get_stock_data(self.tickers, period=self.config.data_period)
        
        if all_tickers_data.empty:
            raise ValueError("No data available for training")
        
        # Calculate scalers
        close_prices = all_tickers_data.xs('Close', level=1, axis=1)
        
        # Generate sequence indices
        seq_length = params.get('seq_length', 60)
        time_window = params.get('time_window', 14)
        max_valid_idx = len(close_prices) - seq_length - time_window
        
        # Correctly generate a single list of unique indices for the universe model
        all_sequence_indices = list(range(max_valid_idx))
        
        # Split into train/test
        split_idx = int(len(all_sequence_indices) * self.config.train_test_split_ratio)
        train_indices = all_sequence_indices[:split_idx]
        test_indices = all_sequence_indices[split_idx:]
        
        # Calculate scalers ONLY on training data to prevent data leakage
        train_close_prices = close_prices.iloc[:split_idx]
        scaler_mean = train_close_prices.values.mean()
        scaler_std = train_close_prices.values.std()
        
        # Get split date for backtesting
        try:
            split_date_idx = test_indices[0] + seq_length if test_indices else len(close_prices) - 1
            split_date = close_prices.index[split_date_idx]
        except IndexError:
            split_date = close_prices.index[-1]
        
        console.print(f"[dim]Train/test split at: {split_date.date()}[/dim]")
        
        console.print(f"[dim]Generated {len(train_indices):,} training sequences and {len(test_indices):,} testing sequences[/dim]")
        
        # Calculate alpha weights for Focal Loss based on training data distribution
        console.print("[dim]Calculating class frequencies for Focal Loss alpha weights...[/dim]")
        all_train_labels_arr = []
        temp_train_generator = StockSequenceGenerator(
            sequence_indices_to_use=train_indices,
            all_ohlcv_data=all_tickers_data,
            seq_length=seq_length,
            time_window=time_window,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            batch_size=params.get('batch_size', 64), # Use batch_size from params
            shuffle_indices=False,
            tickers=self.tickers,
            enable_caching=False, # Disable caching for alpha weight calculation
            adaptive_batch_size=False, # No adaptive sizing for this temp generator
            jax_memory_monitoring=False
        )
        
        # Collect all labels from the temporary training generator
        for _, labels, _, _ in temp_train_generator:
            # Explicitly filter out padding (-1) before extending the list
            valid_labels = labels[labels != -1] # Assuming labels is a jnp.ndarray or np.ndarray
            all_train_labels_arr.append(valid_labels.flatten())
        
        all_train_labels = np.array([])
        if all_train_labels_arr:
            all_train_labels = np.concatenate(all_train_labels_arr)
            
        if all_train_labels.size == 0:
            console.print("[yellow]Warning: No labels generated from training data. Using default equal alpha weights.[/yellow]")
            class_counts_array = np.array([0,0,0])
        else:
            # np.bincount handles counts for non-negative integers
            class_counts_array = np.bincount(all_train_labels, minlength=NUM_CLASSES)
            
        console.print(f"[dim]Class counts (HOLD, BUY_CALL, BUY_PUT): {class_counts_array}[/dim]")
        
        total_labels = class_counts_array.sum()
        if total_labels == 0:
            alpha_weights_calculated = jnp.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=jnp.float32)
        else:
            # Correct inverse frequency weighting with proper normalization to sum to 1.0
            # Calculate inverse frequencies: total_labels / (class_counts + epsilon)
            epsilon = np.finfo(np.float32).eps
            inverse_frequencies = total_labels / (class_counts_array + epsilon)
            # Normalize to sum to 1.0
            alpha_weights_calculated = inverse_frequencies / jnp.sum(inverse_frequencies)
            
        alpha_weights_calculated = jnp.array(alpha_weights_calculated, dtype=jnp.float32)
        console.print(f"[dim]Dynamically computed alpha weights for Focal Loss: {alpha_weights_calculated.tolist()}[/dim]")
        console.print(f"[dim]Alpha weights sum: {jnp.sum(alpha_weights_calculated):.6f}[/dim]")
        
        # Mathematical validation: ensure alpha weights sum to 1.0
        assert abs(jnp.sum(alpha_weights_calculated) - 1.0) < 1e-6, "Alpha weights must sum to 1.0"

        # Create generators with enhanced features support
        # Get enhanced features configuration from params or use defaults
        include_news = params.get('include_news', False)
        include_text = params.get('include_text', False)
        news_window = params.get('news_window', 7)
        text_model = params.get('text_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        train_generator = StockSequenceGenerator(
            sequence_indices_to_use=train_indices,
            all_ohlcv_data=all_tickers_data,
            seq_length=seq_length,
            time_window=time_window,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            batch_size=params.get('batch_size', 64),
            shuffle_indices=True,
            tickers=self.tickers,
            # Enhanced features
            include_news=include_news,
            include_text=include_text,
            news_window=news_window,
            text_model=text_model
        )
        
        test_generator = StockSequenceGenerator(
            sequence_indices_to_use=test_indices,
            all_ohlcv_data=all_tickers_data,
            seq_length=seq_length,
            time_window=time_window,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            batch_size=params.get('batch_size', 64),
            shuffle_indices=False,
            tickers=self.tickers,
            # Enhanced features
            include_news=include_news,
            include_text=include_text,
            news_window=news_window,
            text_model=text_model
        )

        # Prepare artifacts for saving
        artifacts = {
            'train_indices': train_indices,
            'test_indices': test_indices,
            'scaler_mean': float(scaler_mean),
            'scaler_std': float(scaler_std),
            'alpha_weights': alpha_weights_calculated.tolist(),
            'class_counts': class_counts_array.tolist(),
            'split_date': split_date.isoformat(),
            'data_period': self.config.data_period,
            'tickers': self.tickers,
            'seq_length': seq_length,
            'time_window': time_window,
            'train_test_split_ratio': self.config.train_test_split_ratio
        }

        return train_generator, test_generator, alpha_weights_calculated, artifacts
    
    def _get_epochs_for_trial(self, trial: optuna.Trial) -> int:
        """Get number of epochs for a trial based on the optimization phase"""
        # Phase 1: Random search (quick exploration)
        if self.current_phase == 1:
            return self.config.epochs_per_trial_random  # 10 epochs
        
        # Phase 2: Bayesian optimization (medium exploration)
        elif self.current_phase == 2:
            return self.config.epochs_per_trial_bayesian  # 20 epochs
        
        # Phase 3: Fine-tuning (long refinement)
        elif self.current_phase == 3:
            return self.config.epochs_per_trial_fine_tune  # 30 epochs
        
        # Default fallback
        return self.config.epochs_per_trial_random
    
    def _store_trial_results(self, trial: optuna.Trial, params: Dict[str, Any], metrics: Dict[str, float], artifacts: Dict[str, Any] = None):
        """Store trial results including data artifacts"""
        result = {
            'trial_number': trial.number,
            'params': params,
            'metrics': metrics,
            'artifacts': artifacts,  # Include data artifacts
            'timestamp': datetime.now().isoformat()
        }
        
        self.trial_results.append(result)
        self.optimization_history.append({
            'trial': trial.number,
            'value': metrics['validation_accuracy'],
            'params': params
        })
    
    def run_optimization(self):
        """Run the complete hyperparameter optimization"""
        console.print("\n[bold green]🚀 Starting Hyperparameter Optimization[/bold green]")
        console.print(Rule(style="green"))
        
        self.start_time = time.time()
        
        # Update phase tracking
        self.current_phase = 1
        
        # Calculate total trials for remaining phases
        total_trials = (self.config.n_random_trials + 
                       self.config.n_bayesian_trials + 
                       self.config.n_fine_tune_trials)
        
        console.print(f"\n[bold cyan]🎯 Phase 1-3: Hyperparameter Optimization ({total_trials} trials)[/bold cyan]")
        
        # Run optimization with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Optimizing hyperparameters...", total=total_trials)
            
            # Phase 1: Random Search
            console.print(f"\n[bold cyan]🎲 Phase 1: Random Search ({self.config.n_random_trials} trials)[/bold cyan]")
            for i in range(self.config.n_random_trials):
                self.current_phase = 1
                self.study.optimize(self.objective, n_trials=1, timeout=None)
                progress.update(task, advance=1)
                if self.study.best_trial:
                    console.print(f"[dim]Best accuracy so far: {self.study.best_value:.4f}[/dim]")
            
            # Phase 2: Bayesian Optimization
            console.print(f"\n[bold cyan]🧠 Phase 2: Bayesian Optimization ({self.config.n_bayesian_trials} trials)[/bold cyan]")
            for i in range(self.config.n_bayesian_trials):
                self.current_phase = 2
                self.study.optimize(self.objective, n_trials=1, timeout=None)
                progress.update(task, advance=1)
                if self.study.best_trial:
                    console.print(f"[dim]Best accuracy so far: {self.study.best_value:.4f}[/dim]")
            
            # Phase 3: Fine-tuning
            console.print(f"\n[bold cyan]🎯 Phase 3: Fine-tuning ({self.config.n_fine_tune_trials} trials)[/bold cyan]")
            for i in range(self.config.n_fine_tune_trials):
                self.current_phase = 3
                self.study.optimize(self.objective, n_trials=1, timeout=None)
                progress.update(task, advance=1)
                if self.study.best_trial:
                    console.print(f"[dim]Best accuracy so far: {self.study.best_value:.4f}[/dim]")
        
        # Store results
        self._save_results()
        # Log hyperparameter importance to TensorBoard
        if TENSORBOARD_AVAILABLE and tf is not None:
            try:
                importance = optuna.importance.get_param_importances(self.study)
                log_dir = str(self.save_dir / "tensorboard_logs/hparam_importance")
                tb_writer = tf.summary.create_file_writer(log_dir)
                with tb_writer.as_default():
                    # Log as text
                    importance_str = "\n".join([f"{k}: {v:.4f}" for k, v in importance.items()])
                    tf.summary.text("Hyperparameter Importance", importance_str, step=0)
                    # Log as bar plot
                    if importance:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.bar(list(importance.keys()), list(importance.values()))
                        ax.set_title('Hyperparameter Importance')
                        ax.set_ylabel('Importance')
                        plt.tight_layout()
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        plt.close(fig)
                        buf.seek(0)
                        image = tf.image.decode_png(buf.getvalue())
                        image = tf.expand_dims(image, 0)
                        tf.summary.image('Hyperparameter Importance', image, step=0)
                tb_writer.close()
            except Exception as e:
                console.print(f"[dim]Could not log hyperparameter importance to TensorBoard: {e}[/dim]")
        
        console.print("[bold green]✅ Hyperparameter optimization completed![/bold green]")
        console.print(Rule(style="green"))
        
        return self.study.best_params, self.study.best_value
    
    def _make_json_serializable(self, obj):
        """Recursively convert objects like ArrayImpl and numpy arrays to lists for JSON serialization."""
        # Avoid import errors if not always present
        try:
            import numpy as np
        except ImportError:
            np = None
        # JAX ArrayImpl detection
        if obj.__class__.__name__ == "ArrayImpl":
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        if np and isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._make_json_serializable(v) for v in obj)
        return obj

    def _save_results(self):
        """Save optimization results"""
        # Save study
        study_file = self.save_dir / "optimization_study.pkl"
        with open(study_file, 'wb') as f:
            pickle.dump(self.study, f)
        
        # Save trial results
        results_file = self.save_dir / "trial_results.json"
        with open(results_file, 'w') as f:
            json.dump(self._make_json_serializable(self.trial_results), f, indent=2)
        
        # Save optimization history
        history_file = self.save_dir / "optimization_history.json"
        with open(history_file, 'w') as f:
            json.dump(self._make_json_serializable(self.optimization_history), f, indent=2)
        
        # Generate plots
        self._generate_plots()
        
        console.print(f"[dim]Results saved to: {self.save_dir}[/dim]")
    
    def _generate_plots(self):
        """Generate optimization plots"""
        # Optimization history
        fig, ax = plt.subplots(figsize=(10, 6))
        values = [h['value'] for h in self.optimization_history]
        trials = [h['trial'] for h in self.optimization_history]
        ax.plot(trials, values, 'b-', alpha=0.7)
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Optimization History')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / "optimization_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Parameter importance (if available)
        try:
            fig = plot_param_importances(self.study)
            fig.write_html(str(self.save_dir / "parameter_importance.html"))
        except:
            console.print("[dim]Could not generate parameter importance plot[/dim]")
    
    def get_best_configuration(self) -> Tuple[Dict[str, Any], float]:
        """Get the best hyperparameter configuration"""
        if self.study.best_trial is None:
            raise ValueError("No optimization has been run yet")
        
        return self.study.best_params, self.study.best_value

    def print_summary(self):
        """Print optimization summary"""
        if self.study.best_trial is None:
            console.print("[bold yellow]No optimization results available[/bold yellow]")
            return
        console.print("\n[bold cyan]📊 Optimization Summary[/bold cyan]")
        
        # Best trial info
        best_trial_result = None
        for result in self.trial_results:
            if result['trial_number'] == self.study.best_trial.number:
                best_trial_result = result
                break
        checkpoint_path = best_trial_result['metrics'].get('checkpoint_path', 'N/A') if best_trial_result else 'N/A'
        best_panel = Panel(
            f"Best Trial: {self.study.best_trial.number}\n"
            f"Best Accuracy: {self.study.best_value:.4f}\n"
            f"Total Trials: {len(self.study.trials)}\n"
            f"Checkpoint Path: {checkpoint_path}",
            title="🏆 Best Results",
            border_style="green"
        )
        # Best parameters
        params_text = "\n".join([f"{k}: {v}" for k, v in self.study.best_params.items()])
        params_panel = Panel(
            params_text,
            title="⚙️ Best Parameters",
            border_style="blue"
        )
        # Performance info
        total_time = time.time() - self.start_time if self.start_time else 0
        num_trials = len(self.study.trials)
        avg_trial_time = (total_time / max(num_trials, 1)) / 60  # Convert to minutes
        trials_per_hour = num_trials / max(total_time / 3600, 1e-6)  # Avoid division by zero
        perf_panel = Panel(
            f"Total Time: {total_time/3600:.2f} hours\n"
            f"Avg Trial Time: {avg_trial_time:.1f} minutes\n"
            f"Trials per Hour: {trials_per_hour:.1f}",
            title="⏱️ Performance",
            border_style="yellow"
        )
        columns = Columns([best_panel, params_panel, perf_panel])
        console.print(columns)
        console.print(Rule(style="dim"))
        # Print table of all trials
        table = Table(title="All Trials Summary", show_lines=True)
        table.add_column("Trial #", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("Best Epoch", justify="right")
        table.add_column("Checkpoint Path", justify="left")
        for result in self.trial_results:
            metrics = result.get('metrics', {})
            table.add_row(
                str(result.get('trial_number', 'N/A')),
                f"{metrics.get('validation_accuracy', 0.0):.4f}",
                f"{metrics.get('final_loss', 0.0):.4f}",
                str(metrics.get('best_epoch', 'N/A')),
                metrics.get('checkpoint_path', 'N/A') or 'N/A'
            )
        console.print(table)
        console.print(Rule(style="dim"))
