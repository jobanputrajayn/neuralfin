# Hyperparameter Tuning Strategy Guide

## Overview

This guide explains the recommended approach for hyperparameter tuning and model training in the Hyper framework. We use a **hybrid strategy** that combines fast exploration with extended training of promising configurations.

## 🎯 Recommended Approach: Hybrid Strategy

### Phase 1: Fast Hyperparameter Exploration
- **Goal**: Quickly explore the hyperparameter space to identify promising regions
- **Method**: 3-phase optimization with shorter epochs
- **Benefits**: Fast iteration, broad coverage, early identification of good configurations

### Phase 2: Extended Training of Promising Configurations  
- **Goal**: Thoroughly train the best configurations with cross-validation
- **Method**: Extended training (50-100 epochs) with k-fold cross-validation
- **Benefits**: Robust evaluation, better convergence, reliable model selection

### Phase 3: Final Model Training
- **Goal**: Train the best configuration on full dataset
- **Method**: Production training with advanced techniques
- **Benefits**: Optimal performance, production-ready model

## 📊 Comparison of Approaches

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Fast Exploration + Extended Training** | ✅ Fast iteration<br>✅ Broad coverage<br>✅ Efficient resource use<br>✅ Robust final models | ❌ More complex pipeline<br>❌ Requires careful validation split | **Recommended** |
| **Long Epochs in Tuning** | ✅ Simple pipeline<br>✅ Thorough evaluation | ❌ Very expensive<br>❌ Slow iteration<br>❌ Wasted computation | Small search spaces |
| **Short Epochs Only** | ✅ Very fast<br>✅ Cheap | ❌ May miss good configs<br>❌ Unreliable selection | Quick prototyping |

## 🚀 Implementation

### Step 1: Fast Hyperparameter Tuning

```bash
# Run the optimized hyperparameter tuning
python run_hyperparameter_tuning.py
```

**Configuration Changes Made:**
- Increased trials: 20 random + 40 Bayesian + 15 fine-tune = 75 total
- Reduced epochs: 8 + 15 + 25 = average 16 epochs per trial
- More aggressive early stopping: patience=5, min_delta=0.002
- Total computation: 75 trials × 16 epochs = ~1,200 epochs

### Step 2: Extended Training of Top Configurations

```bash
# Train top 5 configurations for 100 epochs each
python run_extended_training.py --top-n 5 --epochs 100
```

**Features:**
- Cross-validation (3-fold) for robust evaluation
- Extended training (100 epochs) for better convergence
- Comprehensive metrics and uncertainty estimation
- Automatic selection of best configuration

### Step 3: Final Model Training

```bash
# Train the best configuration on full dataset
python run_final_training.py --config best_config.json --epochs 200
```

## 📈 Expected Results

### Computational Efficiency
- **Traditional approach**: 60 trials × 50 epochs = 3,000 epochs
- **Our approach**: 75 trials × 16 epochs + 5 configs × 100 epochs = 1,700 epochs
- **Savings**: ~43% less computation with better results

### Quality Improvements
- **Better exploration**: More trials with shorter epochs
- **Robust selection**: Cross-validation prevents overfitting
- **Reliable models**: Extended training ensures convergence

## 🔧 Configuration Details

### Hyperparameter Tuning Settings
```python
# Optimized for fast exploration
n_random_trials: int = 20
n_bayesian_trials: int = 40  
n_fine_tune_trials: int = 15
epochs_per_trial_random: int = 8
epochs_per_trial_bayesian: int = 15
epochs_per_trial_fine_tune: int = 25
early_stopping_patience: int = 5
early_stopping_min_delta: float = 0.002
```

### Extended Training Settings
```python
# Robust evaluation with cross-validation
cross_validation_folds: int = 3
extended_epochs: int = 100
evaluation_frequency: int = 5  # Evaluate every 5 epochs
early_stopping_patience: int = 10
```

## 📊 Monitoring and Analysis

### TensorBoard Integration
```bash
# Monitor training progress
./start_tensorboard.sh
```

### Key Metrics to Track
- **Validation Accuracy**: Primary selection criterion
- **Simulated Returns**: Trading performance metric
- **Training/Validation Loss**: Overfitting detection
- **Cross-validation Variance**: Model stability

### Results Analysis
```python
# Load and analyze results
from scripts.extended_training import ExtendedTrainer

trainer = ExtendedTrainer()
results = trainer.load_extended_results()

# Find best configuration
best_config = trainer.get_best_configuration()
print(f"Best accuracy: {best_config['avg_accuracy']:.4f} ± {best_config['std_accuracy']:.4f}")
```

## 🎯 Best Practices

### 1. Validation Strategy
- **Hyperparameter tuning**: Use single train/validation split
- **Extended training**: Use k-fold cross-validation
- **Final training**: Use full dataset with early stopping

### 2. Early Stopping
- **Tuning phase**: Aggressive early stopping (patience=5)
- **Extended training**: Conservative early stopping (patience=10)
- **Final training**: Very conservative (patience=15)

### 3. Resource Management
- **GPU memory**: Monitor usage and adjust batch sizes
- **Time budget**: Set maximum trial time limits
- **Parallelization**: Use multiple GPUs if available

### 4. Model Selection
- **Primary metric**: Validation accuracy
- **Secondary metrics**: Simulated returns, model complexity
- **Robustness**: Cross-validation variance
- **Practicality**: Training time, inference speed

## 🚨 Common Pitfalls

### 1. Overfitting to Validation Set
- **Problem**: Using same validation set for tuning and extended training
- **Solution**: Use cross-validation in extended training phase

### 2. Insufficient Exploration
- **Problem**: Too few trials or too short training
- **Solution**: Increase trials, use proper early stopping

### 3. Computational Waste
- **Problem**: Training poor configurations for many epochs
- **Solution**: Aggressive early stopping in tuning phase

### 4. Unreliable Selection
- **Problem**: Selecting based on single validation run
- **Solution**: Use cross-validation and multiple metrics

## 📝 Example Workflow

```bash
# 1. Run fast hyperparameter tuning (2-4 hours)
python run_hyperparameter_tuning.py

# 2. Run extended training of top configurations (4-8 hours)
python run_extended_training.py --top-n 5 --epochs 100

# 3. Analyze results and select best configuration
python analyze_results.py

# 4. Train final model (2-4 hours)
python run_final_training.py --config best_config.json --epochs 200

# 5. Evaluate and deploy
python evaluate_model.py --model final_model.pkl
```

## 🎉 Expected Outcomes

With this approach, you should achieve:

1. **Faster iteration**: 43% reduction in computation time
2. **Better exploration**: 75 trials vs 60 trials
3. **Robust selection**: Cross-validation prevents overfitting
4. **Reliable models**: Extended training ensures convergence
5. **Production ready**: Final models are thoroughly tested

## 🔄 Iterative Improvement

After initial results:

1. **Analyze hyperparameter importance**: Focus search on important parameters
2. **Expand search space**: Add new hyperparameters or widen ranges
3. **Adjust training strategy**: Modify epochs, early stopping, etc.
4. **Repeat**: Run the full pipeline with improvements

This iterative approach ensures continuous improvement while maintaining computational efficiency. 