"""
Training functions for JAX GPT stock predictor.
Optimized for minimal data transfers and maximum GPU utilization.

This module provides a unified, NNX-idiomatic training step that works with GPTClassifier models.
The training step performs forward pass, loss calculation, backpropagation, and parameter update using NNX.

Usage Examples:

# NNX mode (idiomatic)
model = GPTClassifier(...)
model, loss = train_step(
    model, 
    batch, 
    alpha_weights,
    call_loss_weight=0.2,
    put_loss_weight=0.2,
    focal_loss_weight=1.5
)

# Using helper functions
train_step_fn = create_train_step_function('nnx')
model, loss = train_step_fn(model, batch, alpha_weights, ...)

# Complete training loop
model, metrics = train_with_unified_step(
    model, 
    train_generator, 
    alpha_weights,
    num_epochs=5
)
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from typing import Any, Dict, Tuple, Optional
from collections import defaultdict
import time
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import psutil
from flax import nnx
from flax.nnx import softmax, one_hot, TrainState, Optimizer, jit, value_and_grad, scan, Any, OfType
from jax import lax
# Import metrics from the correct location
Average = nnx.metrics.Average
Metric = nnx.metrics.Metric
Accuracy = nnx.metrics.Accuracy
Welford = nnx.metrics.Welford
MultiMetric = nnx.metrics.MultiMetric

# Import constants from models module using absolute imports
try:
    from models.constants import ACTION_BUY_CALL, ACTION_BUY_PUT, NUM_CLASSES
    from utils.gpu_utils import get_gpu_manager, get_batch_optimizer, get_gpu_utilization, get_gpu_memory_info
    from utils.system_utils import validate_numerical_inputs
except ImportError:
    # Fallback for relative imports if absolute fails
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.constants import ACTION_BUY_CALL, ACTION_BUY_PUT, NUM_CLASSES
    from utils.gpu_utils import get_gpu_manager, get_batch_optimizer, get_gpu_utilization, get_gpu_memory_info
    from utils.system_utils import validate_numerical_inputs


# Global GPU manager and batch optimizer
gpu_manager = None
batch_optimizer = None

def create_train_step_function(mode='nnx'):
    """
    Create a training step function for the specified mode.
    
    Args:
        mode: 'nnx' (NNX idiomatic) - only mode supported now
        
    Returns:
        A function that can be used for training steps
    """
    if mode == 'nnx':
        def train_step_nnx(model, batch, alpha_weights, **kwargs):
            """NNX mode training step."""
            return train_step(model, batch, alpha_weights, **kwargs)
        return train_step_nnx
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'nnx' (only mode supported)")

def focal_loss(logits, y_true, alpha_weights, gamma=2.0):
    """
    Compute focal loss for classification.
    
    Args:
        logits: Model output logits (batch_size, num_features, num_classes)
        y_true: True labels (batch_size, num_features)
        alpha_weights: Class weights for focal loss
        gamma: Focal loss gamma parameter
        
    Returns:
        Focal loss value
    """
    # Reshape for loss calculation
    batch_size, num_features, num_classes = logits.shape
    logits_flat = logits.reshape(-1, num_classes)  # (batch_size * num_features, num_classes)
    y_true_flat = y_true.reshape(-1)  # (batch_size * num_features,)
    
    # Classification loss (Focal Loss for better handling of class imbalance)
    probabilities = softmax(logits_flat)
    gamma = jnp.array(gamma, dtype=jnp.float32)  # Focal loss gamma parameter
    one_hot_y_true = one_hot(y_true_flat, num_classes=num_classes)
    epsilon = jnp.array(1e-7, dtype=jnp.float32)
    probabilities = jnp.clip(probabilities, epsilon, jnp.array(1.0, dtype=jnp.float32) - epsilon)
    
    # Improved numerical stability for cross entropy
    log_probs = jnp.log(probabilities + epsilon)
    cross_entropy = -jnp.sum(one_hot_y_true * log_probs, axis=-1)
    
    p_t = jnp.sum(one_hot_y_true * probabilities, axis=-1)
    modulating_factor = jnp.power(jnp.array(1.0, dtype=jnp.float32) - p_t, gamma)
    alpha_t = jnp.take(alpha_weights, y_true_flat)
    focal_loss_value = alpha_t * modulating_factor * cross_entropy
    
    # Return mean focal loss
    return jnp.mean(focal_loss_value)

def _init_gpu_optimization():
    """Initialize GPU optimization components."""
    global gpu_manager, batch_optimizer
    if gpu_manager is None:
        gpu_manager = get_gpu_manager()
        batch_optimizer = get_batch_optimizer()


@jit
def train_step(model, batch, alpha_weights, call_loss_weight: float = 0.2, put_loss_weight: float = 0.2, focal_loss_weight: float = 1.5):
    """
    NNX-idiomatic training step that works with GPTClassifier model instances.
    Performs forward pass, loss calculation, backpropagation, and parameter update using NNX.
    Optimized to minimize data transfers and maximize GPU utilization.
    
    Args:
        model: The GPTClassifier model (NNX module)
        batch: Tuple of (x, y_true, actual_returns, padding_mask) - all should be JAX arrays on GPU
        alpha_weights: Class weights for focal loss (JAX array on GPU)
        call_loss_weight: Weight for call option loss
        put_loss_weight: Weight for put option loss
        focal_loss_weight: Weight for focal loss
        
    Returns:
        Tuple of (model, loss)
    """
    def loss_fn(model, logits):
        x, y_true, actual_returns, padding_mask = batch
        batch_size, num_tickers, num_classes = logits.shape
        logits_flat = logits.reshape(-1, num_classes)
        y_true_flat = y_true.reshape(-1)
        actual_returns_flat = actual_returns.reshape(-1)
        expanded_padding_mask = jnp.repeat(jnp.expand_dims(padding_mask, axis=1), num_tickers, axis=1).reshape(-1)
        valid_mask = jnp.logical_and((y_true_flat != jnp.array(-1, dtype=jnp.int32)), expanded_padding_mask)
        probabilities = softmax(logits_flat)
        gamma = jnp.array(2.0, dtype=jnp.float32)
        one_hot_y_true = one_hot(y_true_flat, num_classes=model.num_classes)
        epsilon = jnp.array(1e-7, dtype=jnp.float32)
        probabilities = jnp.clip(probabilities, epsilon, jnp.array(1.0, dtype=jnp.float32) - epsilon)
        log_probs = jnp.log(probabilities + epsilon)
        cross_entropy = -jnp.sum(one_hot_y_true * log_probs, axis=-1)
        p_t = jnp.sum(one_hot_y_true * probabilities, axis=-1)
        modulating_factor = jnp.power(jnp.array(1.0, dtype=jnp.float32) - p_t, gamma)
        alpha_t = jnp.take(alpha_weights, y_true_flat)
        focal_loss = alpha_t * modulating_factor * cross_entropy
        call_loss_condition = (y_true_flat == jnp.array(ACTION_BUY_CALL, dtype=jnp.int32))
        put_loss_condition = (y_true_flat == jnp.array(ACTION_BUY_PUT, dtype=jnp.int32))
        actual_returns_clipped = jnp.clip(actual_returns_flat, -1.0, 1.0)
        call_loss = jnp.where(call_loss_condition,
                              (jnp.array(1.0, dtype=jnp.float32) - probabilities[:, ACTION_BUY_CALL]) * actual_returns_clipped,
                              jnp.array(0.0, dtype=jnp.float32))
        put_loss = jnp.where(put_loss_condition,
                             (jnp.array(1.0, dtype=jnp.float32) - probabilities[:, ACTION_BUY_PUT]) * actual_returns_clipped,
                             jnp.array(0.0, dtype=jnp.float32))
        call_loss_weight_jnp = jnp.array(call_loss_weight, dtype=jnp.float32)
        put_loss_weight_jnp = jnp.array(put_loss_weight, dtype=jnp.float32)
        focal_loss_weight_jnp = jnp.array(focal_loss_weight, dtype=jnp.float32)
        combined_loss = (call_loss_weight_jnp * call_loss + 
                        put_loss_weight_jnp * put_loss + 
                        focal_loss_weight_jnp * focal_loss) * valid_mask
        valid_count = jnp.sum(valid_mask.astype(jnp.int32))
        mean_loss = jnp.where(valid_count > 0,
                             jnp.sum(combined_loss) / (valid_count + jnp.array(1e-8, dtype=jnp.float32)),
                             jnp.array(0.0, dtype=jnp.float32))
        mean_loss = jnp.where(jnp.logical_or(jnp.isnan(mean_loss), jnp.isinf(mean_loss)),
                             jnp.array(0.0, dtype=jnp.float32),
                             mean_loss)
        return mean_loss, logits

    # Forward pass
    x, y_true, actual_returns, padding_mask = batch
    model.train()
    logits = model(x, padding_mask=padding_mask)
    
    # Compute loss and gradients using NNX pattern
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(model, logits)
    
    # Update model using its own optimizer logic
    model = model.apply_updates(grads)
    
    return model, loss


def _evaluate_batch_metrics(logits, batch_y, batch_actual_returns, padding_mask):
    """
    Computes evaluation metrics for a single batch given logits and labels.
    Ignores padded samples (y == -1) in all metrics.
    All inputs should be JAX arrays on GPU.
    """
    # logits: (batch_size, num_tickers, num_classes)
    batch_size, num_tickers, num_classes = logits.shape
    logits_flat = logits.reshape(-1, num_classes)  # (batch_size * num_tickers, num_classes)
    y_pred_flat = jnp.argmax(logits_flat, axis=-1)  # (batch_size * num_tickers,)
    y_true_flat = batch_y.reshape(-1)  # (batch_size * num_tickers,)
    actual_returns_flat = batch_actual_returns.reshape(-1)  # (batch_size * num_tickers,)

    # Mask for valid (non-padded) samples
    expanded_padding_mask = jnp.repeat(jnp.expand_dims(padding_mask, axis=1), num_tickers, axis=1).reshape(-1)
    valid_mask = jnp.logical_and((y_true_flat != jnp.array(-1, dtype=jnp.int32)), expanded_padding_mask)
    
    # Apply mask to predictions and labels before calculating metrics
    y_pred_masked = jnp.where(valid_mask, y_pred_flat, jnp.array(-1, dtype=y_pred_flat.dtype))
    y_true_masked = jnp.where(valid_mask, y_true_flat, jnp.array(-1, dtype=y_true_flat.dtype))
    actual_returns_masked = jnp.where(valid_mask, actual_returns_flat, jnp.array(0.0, dtype=jnp.float32))

    correct_predictions_count = jnp.sum(jnp.where(valid_mask, (y_pred_flat == y_true_flat).astype(jnp.int32), jnp.array(0, dtype=jnp.int32)))
    num_samples_in_batch = jnp.sum(valid_mask.astype(jnp.int32))

    # Simulated return for the batch, now using masked actual_returns
    # Calculate returns based on whether our predictions match the actual actions
    # If prediction matches actual action, we get the return; otherwise we get negative return
    simulated_return_batch = jnp.sum(
        jnp.where(valid_mask, 
                  # If we predict the same action as the actual action, we get the return
                  jnp.where(y_pred_flat == y_true_flat,
                           actual_returns_masked,  # Correct prediction gets positive return
                           -actual_returns_masked),  # Wrong prediction gets negative return
                  jnp.array(0.0, dtype=jnp.float32))
    )

    # Actual returns for true BUY_CALL positions, using masked actual_returns
    actual_calls_in_batch_masked = jnp.where(valid_mask, (y_true_flat == jnp.array(ACTION_BUY_CALL, dtype=jnp.int32)), jnp.array(False, dtype=jnp.bool_))
    total_actual_call_returns_batch = jnp.sum(jnp.where(actual_calls_in_batch_masked, actual_returns_masked, jnp.array(0.0, dtype=jnp.float32)))
    num_actual_calls_batch = jnp.sum(jnp.where(actual_calls_in_batch_masked, jnp.array(1, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)))

    # Confusion matrix components for the batch
    batch_confusion_matrix_increment = jnp.zeros((NUM_CLASSES, NUM_CLASSES), dtype=jnp.int32)
    
    # Instead of boolean indexing, use jnp.where to handle the confusion matrix
    # We'll create a mask for each valid sample and increment the confusion matrix accordingly
    def update_confusion_matrix(carry, valid_sample):
        true_class, pred_class, is_valid = valid_sample
        confusion_matrix = carry
        
        # Only update if the sample is valid (not -1)
        update_mask = jnp.logical_and(is_valid, true_class >= 0)
        confusion_matrix = jnp.where(
            update_mask,
            confusion_matrix.at[true_class, pred_class].add(1),
            confusion_matrix
        )
        return confusion_matrix, None
    
    # Create samples for confusion matrix update
    valid_samples = jnp.stack([y_true_masked, y_pred_masked, valid_mask], axis=1)
    
    # Use scan to update confusion matrix for all samples
    final_confusion_matrix, _ = lax.scan(update_confusion_matrix, batch_confusion_matrix_increment, valid_samples)
    
    return (correct_predictions_count, num_samples_in_batch,
            simulated_return_batch, total_actual_call_returns_batch, num_actual_calls_batch,
            final_confusion_matrix)


def evaluate_model(state, data_generator, model, alpha_weights, progress: Any = None, step_progress_task: Any = None, tb_writer=None, global_step: int = None,transfer_to_devide=False):
    """
    Evaluates the model on the provided data generator and returns comprehensive metrics.
    Optimized to minimize data transfers and maximum GPU utilization.

    Args:
        state: The Flax train_state.
        data_generator: An instance of StockSequenceGenerator or PrefetchGenerator for evaluation data.
        model: The GPTClassifier model.
        alpha_weights: Class weights for focal loss
        progress: The Rich progress object.
        step_progress_task: Rich progress task for step-level updates during evaluation.
        tb_writer: TensorBoard writer for logging evaluation metrics.
        global_step: Global step for TensorBoard logging (if None, uses batch_idx).

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics.
    """
    # Initialize GPU optimization
    _init_gpu_optimization()
    
    # Validate alpha weights
    if not validate_numerical_inputs(np.array(alpha_weights), "alpha_weights"):
        print("Warning: Invalid alpha weights detected, using uniform weights")
        alpha_weights = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float32)
    
    all_labels = []
    all_returns = []
    total_loss = 0.0
    num_batches = 0
    total_correct_predictions = 0
    total_samples = 0
    total_simulated_return = 0.0
    total_actual_call_returns = 0.0
    total_actual_calls = 0
    aggregated_confusion_matrix = jnp.zeros((NUM_CLASSES, NUM_CLASSES), dtype=jnp.int32)

    # Pre-transfer alpha weights to device once
    alpha_weights_device = alpha_weights

    # Reset step progress bar for evaluation if provided
    if progress is not None and step_progress_task is not None:
        progress.reset(step_progress_task)
        progress.update(step_progress_task, total=len(data_generator))
        progress.update(step_progress_task, visible=True)

    # Add a jitted eval_step for SPMD/multi-device evaluation
    @nnx.jit
    def eval_step(model, batch_x, padding_mask):
        return model(batch_x, padding_mask=padding_mask)

    batch_count = len(data_generator)
    eval_start_time = time.time()
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), transient=True) as progress:
        batch_task = progress.add_task("Eval Batches", total=batch_count)
        for batch_idx, (batch_x, batch_y, batch_actual_returns, padding_mask_batch) in enumerate(data_generator):
            batch_start_time = time.time()
            
            # Validate batch data before processing
            if not validate_numerical_inputs(batch_x, f"batch_x_{batch_idx}"):
                print(f"Warning: Invalid batch_x detected in batch {batch_idx}, skipping")
                continue
                
            if not validate_numerical_inputs(batch_y, f"batch_y_{batch_idx}"):
                print(f"Warning: Invalid batch_y detected in batch {batch_idx}, skipping")
                continue
                
            if not validate_numerical_inputs(batch_actual_returns, f"batch_returns_{batch_idx}"):
                print(f"Warning: Invalid batch_returns detected in batch {batch_idx}, skipping")
                continue
            
            # Perform a forward pass and calculate loss
            model.eval()
            logits = eval_step(model, batch_x, padding_mask_batch)
            
            # Filter out masked values (-1) for loss calculation
            # Using the explicit padding_mask for primary validity check
            expanded_padding_mask_for_loss = jnp.repeat(jnp.expand_dims(padding_mask_batch, axis=1), model.num_tickers, axis=1).reshape(-1)
            valid_mask_loss = jnp.logical_and((batch_y.reshape(-1) != jnp.array(-1, dtype=jnp.int32)), expanded_padding_mask_for_loss) # Reshape batch_y_device before comparison

            if jnp.sum(valid_mask_loss.astype(jnp.int32)) == 0:
                # If no valid data in batch, skip it
                if progress is not None and step_progress_task is not None:
                    progress.update(step_progress_task, advance=1)
                continue

            # Apply mask to labels and logits (flatten for sparse_softmax_cross_entropy_with_logits)
            flat_labels = batch_y.reshape(-1)[valid_mask_loss] # Reshape then index
            flat_logits = logits.reshape(-1, logits.shape[-1])[valid_mask_loss] # Reshape then index

            # Use Optax's sparse_softmax_cross_entropy_with_integer_labels for loss calculation
            loss = optax.softmax_cross_entropy_with_integer_labels(logits=flat_logits, labels=flat_labels).mean()
            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            predictions = jnp.argmax(logits, axis=-1)

            # Aggregate evaluation metrics using _evaluate_batch_metrics
            correct_preds_batch, num_samples_batch, simulated_return_batch, \
                actual_call_returns_batch, num_actual_calls_batch, cm_increment = \
                _evaluate_batch_metrics(logits, batch_y, batch_actual_returns, padding_mask_batch)
            
            total_correct_predictions += correct_preds_batch
            total_samples += num_samples_batch
            total_simulated_return += simulated_return_batch
            total_actual_call_returns += actual_call_returns_batch
            total_actual_calls += num_actual_calls_batch
            aggregated_confusion_matrix += cm_increment

            # Calculate batch metrics
            batch_time = time.time() - batch_start_time
            batch_loss = float(loss)
            batch_accuracy = float(correct_preds_batch) / float(num_samples_batch) if num_samples_batch > 0 else 0.0
            
            # Resource monitoring
            cpu_util = psutil.cpu_percent()
            mem = psutil.virtual_memory().used / (1024**3)
            gpu_util = 0.0
            

            # Enhanced batch-level TensorBoard logging for evaluation
            if tb_writer is not None:
                try:
                    import tensorflow as tf
                    # Use global_step if provided, otherwise use batch_idx
                    current_step = global_step if global_step is not None else batch_idx
                    
                    with tb_writer.as_default():
                        # Evaluation metrics
                        tf.summary.scalar("eval/batch_loss", batch_loss, step=current_step)
                        tf.summary.scalar("eval/batch_accuracy", batch_accuracy, step=current_step)
                        tf.summary.scalar("eval/batch_time_seconds", batch_time, step=current_step)
                        tf.summary.scalar("eval/batch_idx", batch_idx, step=current_step)
                        tf.summary.scalar("eval/batches_total", batch_count, step=current_step)
                        
                        # Batch-specific metrics
                        tf.summary.scalar("eval/batch_correct_predictions", float(correct_preds_batch), step=current_step)
                        tf.summary.scalar("eval/batch_samples", float(num_samples_batch), step=current_step)
                        tf.summary.scalar("eval/batch_simulated_return", float(simulated_return_batch), step=current_step)
                        tf.summary.scalar("eval/batch_actual_call_returns", float(actual_call_returns_batch), step=current_step)
                        tf.summary.scalar("eval/batch_actual_calls", float(num_actual_calls_batch), step=current_step)
                        
                        # System metrics
                        tf.summary.scalar("system/cpu_utilization_percent", cpu_util, step=current_step)
                        tf.summary.scalar("system/ram_usage_gb", mem, step=current_step)
                        tf.summary.scalar("system/gpu_utilization_percent", gpu_util, step=current_step)
                        
                        # Progress tracking
                        tf.summary.scalar("eval/progress_percent", (batch_idx + 1) / batch_count * 100, step=current_step)
                        tf.summary.scalar("eval/batches_processed", batch_idx + 1, step=current_step)
                        
                        tb_writer.flush()
                except Exception as e:
                    print(f"Warning: TensorBoard logging failed during evaluation: {e}")
                    
            progress.update(batch_task, advance=1)

    if progress is not None and step_progress_task is not None:
        progress.update(step_progress_task, visible=False)  # Hide after evaluation

    if num_batches == 0:
        return {'overall_accuracy': 0.0, 'avg_simulated_return': 0.0, 'loss': 0.0}

    # Calculate final metrics
    eval_time = time.time() - eval_start_time
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = (total_correct_predictions / total_samples) if total_samples > 0 else 0.0
    simulated_return_per_trade = (total_simulated_return / total_samples) if total_samples > 0 else 0.0
    
    # Calculate average actual return per call, avoiding division by zero
    avg_actual_call_return = (total_actual_call_returns / total_actual_calls) if total_actual_calls > 0 else 0.0

    # Convert aggregated_confusion_matrix to numpy for plotting/logging later
    # Make sure to transfer it from JAX device to CPU
    numpy_confusion_matrix = jax.device_get(aggregated_confusion_matrix)

    # Final evaluation summary logging
    if tb_writer is not None:
        try:
            import tensorflow as tf
            current_step = global_step if global_step is not None else batch_count
            
            with tb_writer.as_default():
                # Final evaluation metrics
                tf.summary.scalar("eval/final_loss", avg_loss, step=current_step)
                tf.summary.scalar("eval/final_accuracy", accuracy, step=current_step)
                tf.summary.scalar("eval/total_time_seconds", eval_time, step=current_step)
                tf.summary.scalar("eval/batches_processed", num_batches, step=current_step)
                tf.summary.scalar("eval/total_samples", total_samples, step=current_step)
                
                # Performance metrics
                tf.summary.scalar("eval/simulated_return", float(total_simulated_return), step=current_step)
                tf.summary.scalar("eval/simulated_return_per_trade", simulated_return_per_trade, step=current_step)
                tf.summary.scalar("eval/actual_call_return", float(total_actual_call_returns), step=current_step)
                tf.summary.scalar("eval/avg_actual_call_return", avg_actual_call_return, step=current_step)
                
                # Efficiency metrics
                tf.summary.scalar("eval/samples_per_second", total_samples / eval_time if eval_time > 0 else 0, step=current_step)
                tf.summary.scalar("eval/batches_per_second", num_batches / eval_time if eval_time > 0 else 0, step=current_step)
                
                tf.summary.flush()
        except Exception as e:
            print(f"Warning: Final TensorBoard logging failed: {e}")

    metrics = {
        'loss': avg_loss,
        'accuracy': float(accuracy),
        'simulated_return': float(total_simulated_return),
        'simulated_return_per_trade': float(simulated_return_per_trade),
        'actual_call_return': float(total_actual_call_returns),
        'avg_actual_call_return': float(avg_actual_call_return),
        'confusion_matrix': numpy_confusion_matrix.tolist() # Convert to list for JSON compatibility
    }

    # Clear GPU cache after evaluation to free up memory
    gpu_manager.clear_all_buffers()
    
    return metrics



def get_gpu_performance_stats() -> Dict[str, Any]:
    """Get comprehensive GPU performance statistics."""
    _init_gpu_optimization()
    
    transfer_stats = gpu_manager.get_transfer_stats()
    optimization_stats = batch_optimizer.get_optimization_stats()
    
    return {
        'transfer_stats': transfer_stats,
        'optimization_stats': optimization_stats,
        'current_gpu_utilization': get_gpu_utilization(),
        'gpu_memory_info': get_gpu_memory_info()
    }