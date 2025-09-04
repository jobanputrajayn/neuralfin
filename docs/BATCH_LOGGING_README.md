# Enhanced Batch-Level TensorBoard Logging

This document describes the enhanced batch-level TensorBoard logging functionality that has been implemented in the hyperparameter optimization framework.

## Overview

The hyperparameter optimization now includes comprehensive batch-level logging within each trial, providing detailed monitoring of training progress at the finest granularity. This allows for better understanding of model behavior, system performance, and training dynamics.

## Key Features

### 1. **Trial-Level Organization**
- Each trial creates its own TensorBoard log directory: `tensorboard_logs/trial_{number}_{timestamp}/`
- Hyperparameters are logged at step 0 for each trial
- Trial metadata includes phase information and trial number

### 2. **Batch-Level Training Metrics**
- **Loss tracking**: `train/batch_loss` - Loss for each individual batch
- **Timing**: `train/batch_time_seconds` - Time taken per batch
- **Progress**: `train/batch_idx` - Current batch index within epoch
- **Efficiency**: `train/batches_per_epoch` - Total batches per epoch

### 3. **Batch-Level Validation Metrics**
- **Loss**: `eval/batch_loss` - Validation loss per batch
- **Accuracy**: `eval/batch_accuracy` - Validation accuracy per batch
- **Timing**: `eval/batch_time_seconds` - Validation time per batch
- **Samples**: `eval/batch_samples` - Number of samples in batch
- **Predictions**: `eval/batch_correct_predictions` - Correct predictions per batch

### 4. **System Performance Monitoring**
- **CPU**: `system/cpu_utilization_percent` - CPU usage percentage
- **Memory**: `system/ram_usage_gb` - RAM usage in GB
- **GPU**: `system/gpu_utilization_percent` - GPU utilization percentage

### 5. **Progress Tracking**
- **Global step**: `progress/global_step` - Sequential step counter across all batches
- **Epoch progress**: `progress/epoch` - Current epoch number
- **Batch progress**: `progress/batch_in_epoch` - Current batch within epoch
- **Completion**: `progress/completion_percent` - Overall completion percentage

### 6. **Epoch-Level Summaries**
- **Training metrics**: Average loss, epoch time, batches processed
- **Validation metrics**: Accuracy, loss, validation time
- **Performance tracking**: Best accuracy so far, epochs since best
- **Additional metrics**: Simulated returns, actual call returns

## Logging Structure

### Training Metrics (Batch Level)
```
train/
├── batch_loss              # Loss per batch
├── batch_time_seconds      # Time per batch
├── batch_idx              # Batch index
└── batches_per_epoch      # Total batches per epoch
```

### Validation Metrics (Batch Level)
```
eval/
├── batch_loss              # Validation loss per batch
├── batch_accuracy          # Validation accuracy per batch
├── batch_time_seconds      # Validation time per batch
├── batch_idx              # Batch index
├── batches_total          # Total validation batches
├── batch_correct_predictions  # Correct predictions per batch
├── batch_samples          # Samples per batch
├── batch_simulated_return # Simulated return per batch
├── batch_actual_call_returns  # Actual call returns per batch
├── batch_actual_calls     # Number of actual calls per batch
├── progress_percent       # Validation progress
└── batches_processed      # Batches processed so far
```

### System Metrics (Batch Level)
```
system/
├── cpu_utilization_percent  # CPU usage
├── ram_usage_gb           # RAM usage
└── gpu_utilization_percent  # GPU usage
```

### Progress Tracking (Batch Level)
```
progress/
├── epoch                  # Current epoch
├── batch_in_epoch        # Current batch in epoch
├── global_step           # Global step counter
└── completion_percent    # Overall completion percentage
```

### Trial Information (Step 0)
```
trial/
├── metadata              # Trial metadata
├── phase                 # Optimization phase
├── number                # Trial number
├── current_epoch         # Current epoch
├── current_batch         # Current batch
├── epoch                 # Epoch number
├── best_epoch           # Best epoch so far
├── best_accuracy        # Best accuracy so far
├── final_accuracy       # Final accuracy
├── final_epoch          # Final epoch
├── total_steps          # Total steps
└── completed            # Completion flag
```

### Hyperparameters (Step 0)
```
hyperparam/
├── learning_rate         # Learning rate
├── dropout_rate          # Dropout rate
├── weight_alpha          # Loss weight alpha
├── weight_beta           # Loss weight beta
├── time_window           # Time window
├── seq_length            # Sequence length
├── num_layers            # Number of layers
├── d_model               # Model dimension
├── num_heads             # Number of attention heads
├── d_ff                  # Feed-forward dimension
└── batch_size            # Batch size
```

## Usage

### 1. Running Hyperparameter Optimization
```bash
# Run the hyperparameter tuner
python src/scripts/hyperparameter_tuner.py --trials 10 --epochs_random 5

# Or use the test script
python test_batch_logging.py
```

### 2. Starting TensorBoard
```bash
# Start TensorBoard to view logs
./start_tensorboard.sh

# Or manually
tensorboard --logdir ./tensorboard_logs --port 6009
```

### 3. Viewing in Browser
Open `http://localhost:6009` in your browser to view:
- **Scalars**: Training and validation metrics over time
- **Images**: Confusion matrices
- **Text**: Hyperparameters and trial metadata
- **Custom Scalars**: Organized metric groups

## Benefits

### 1. **Detailed Monitoring**
- Track training progress at batch granularity
- Identify performance bottlenecks
- Monitor system resource usage

### 2. **Debugging Support**
- Pinpoint when issues occur during training
- Track loss spikes and accuracy drops
- Monitor memory and GPU usage patterns

### 3. **Performance Analysis**
- Analyze batch processing times
- Identify optimal batch sizes
- Monitor system efficiency

### 4. **Research Insights**
- Understand model convergence patterns
- Analyze validation performance
- Track hyperparameter effects

## Example TensorBoard Views

### Training Loss Over Time
- View `train/batch_loss` to see loss fluctuations
- Compare with `train/epoch_loss` for smoothed trends
- Monitor `train/batch_time_seconds` for performance

### Validation Performance
- Track `eval/batch_accuracy` for real-time validation
- Compare with `val/accuracy` for epoch-level trends
- Monitor `eval/batch_loss` for validation loss

### System Performance
- Monitor `system/cpu_utilization_percent` for CPU usage
- Track `system/ram_usage_gb` for memory consumption
- Watch `system/gpu_utilization_percent` for GPU efficiency

### Progress Tracking
- View `progress/completion_percent` for overall progress
- Track `progress/global_step` for sequential steps
- Monitor `trial/current_epoch` for trial progress

## Configuration

The logging behavior can be controlled through the hyperparameter configuration:

```python
config = HyperparameterConfig(
    n_random_trials=15,
    n_bayesian_trials=35,
    n_fine_tune_trials=10,
    epochs_per_trial_random=10,
    epochs_per_trial_bayesian=20,
    epochs_per_trial_fine_tune=30,
    # ... other parameters
)
```

## Troubleshooting

### 1. **No Logs Appearing**
- Ensure TensorFlow is installed: `pip install tensorflow`
- Check that `TENSORBOARD_AVAILABLE` is True
- Verify log directory permissions

### 2. **High Memory Usage**
- Reduce batch size in hyperparameters
- Monitor `system/ram_usage_gb` in TensorBoard
- Consider reducing sequence length

### 3. **Slow Training**
- Monitor `train/batch_time_seconds` for bottlenecks
- Check `system/gpu_utilization_percent` for GPU efficiency
- Consider adjusting batch size or model complexity

### 4. **TensorBoard Not Loading**
- Ensure TensorBoard is running on correct port
- Check log directory path
- Verify TensorBoard version compatibility

## Future Enhancements

1. **Custom Metrics**: Add user-defined metrics for specific use cases
2. **Alerting**: Set up alerts for performance degradation
3. **Comparison Tools**: Compare multiple trials side-by-side
4. **Export Features**: Export metrics for external analysis
5. **Real-time Monitoring**: Web-based real-time monitoring dashboard

## Conclusion

The enhanced batch-level TensorBoard logging provides unprecedented visibility into the hyperparameter optimization process, enabling better understanding of model behavior, system performance, and training dynamics. This comprehensive logging system supports both research and production use cases, making it easier to optimize models and debug issues. 