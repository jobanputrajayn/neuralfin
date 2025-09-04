# Artifact System Implementation

## Overview

This document describes the implementation of an artifact system that ensures data consistency between hyperparameter tuning and extended training phases by saving and reusing data splits, scalers, and other preprocessing artifacts.

## Problem Statement

Previously, the extended training script would regenerate data splits and scalers independently, which could lead to:
- **Data leakage**: Different train/test splits between phases
- **Inconsistent preprocessing**: Different scaler parameters
- **Non-reproducible results**: Results that can't be directly compared

## Solution: Artifact System

### 1. Hyperparameter Tuning Phase

#### Artifact Generation
During hyperparameter tuning, the `_prepare_data_generators` method now returns artifacts:

```python
def _prepare_data_generators(self, params: Dict[str, Any]) -> Tuple[StockSequenceGenerator, StockSequenceGenerator, jnp.ndarray, Dict[str, Any]]:
    # ... data preparation logic ...
    
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
```

#### Artifact Storage
Artifacts are stored with each trial result:

```python
def _store_trial_results(self, trial: optuna.Trial, params: Dict[str, Any], metrics: Dict[str, float], artifacts: Dict[str, Any] = None):
    result = {
        'trial_number': trial.number,
        'params': params,
        'metrics': metrics,
        'artifacts': artifacts,  # Include data artifacts
        'timestamp': datetime.now().isoformat()
    }
    self.trial_results.append(result)
```

### 2. Extended Training Phase

#### Artifact Loading
The extended training script now attempts to load artifacts from hyperparameter tuning:

```python
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
            return trial_result['artifacts']
    
    return None
```

#### Data Preparation with Artifacts
When artifacts are found, they are used for consistent data preparation:

```python
def _prepare_data_with_artifacts(self, artifacts: Dict[str, Any], config: Dict[str, Any], 
                               fold: int, n_folds: int) -> Tuple[StockSequenceGenerator, StockSequenceGenerator, np.ndarray]:
    # Load all OHLCV data
    all_ohlcv_data = get_stock_data(artifacts['tickers'], period=artifacts['data_period'])
    
    # Use saved indices and scalers
    train_indices = artifacts['train_indices']
    test_indices = artifacts['test_indices']
    scaler_mean = artifacts['scaler_mean']
    scaler_std = artifacts['scaler_std']
    alpha_weights = np.array(artifacts['alpha_weights'])
    
    # For cross-validation, split the original train indices
    fold_size = len(train_indices) // n_folds
    val_start = fold * fold_size
    val_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(train_indices)
    
    cv_train_indices = train_indices[:val_start] + train_indices[val_end:]
    cv_val_indices = train_indices[val_start:val_end]
    
    # Create generators using saved artifacts
    train_generator = StockSequenceGenerator(
        sequence_indices_to_use=cv_train_indices,
        all_ohlcv_data=all_ohlcv_data,
        seq_length=artifacts['seq_length'],
        time_window=artifacts['time_window'],
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        batch_size=config['batch_size'],
        shuffle_indices=True,
        tickers=artifacts['tickers']
    )
    
    # ... similar for test_generator ...
    
    return train_generator, test_generator, alpha_weights
```

#### Fallback Mechanism
If no artifacts are found, the system falls back to generating new splits:

```python
def _prepare_data_fold(self, config: Dict[str, Any], fold: int, n_folds: int):
    artifacts = self._find_configuration_artifacts(config)
    
    if artifacts is not None:
        console.print(f"[dim]Using saved artifacts from trial {artifacts.get('trial_number', 'unknown')}[/dim]")
        return self._prepare_data_with_artifacts(artifacts, config, fold, n_folds)
    else:
        console.print("[yellow]Warning: No saved artifacts found. Generating new data split.[/yellow]")
        return self._prepare_data_new_split(config, fold, n_folds)
```

### 3. Artifact Propagation to Final Training

Extended training also saves artifacts for the best configuration:

```python
def _save_extended_results(self, results: List[Dict]):
    # Save artifacts for the best configuration
    if results:
        best_result = max(results, key=lambda x: x['avg_accuracy'])
        artifacts_file = results_dir / "best_configuration_artifacts.json"
        
        best_artifacts = {
            'config': best_result['config'],
            'avg_accuracy': best_result['avg_accuracy'],
            'avg_return': best_result['avg_return'],
            'std_accuracy': best_result['std_accuracy'],
            'epochs_trained': best_result['epochs_trained'],
            'cv_results': best_result['cv_results'],
            'timestamp': datetime.now().isoformat(),
            'data_period': self.config.data_period,
            'tickers': self.tickers
        }
        
        with open(artifacts_file, 'w') as f:
            json.dump(best_artifacts, f, indent=2, default=str)
```

## Benefits

1. **Data Consistency**: Ensures the same data splits and preprocessing between phases
2. **Reproducibility**: Results can be directly compared across phases
3. **No Data Leakage**: Test data is never used for training in extended training
4. **Backward Compatibility**: Falls back gracefully when artifacts aren't available
5. **Cross-Validation Safety**: Uses only training data for CV splits

## File Structure

```
hyperparameter_tuning_results/
├── trial_results.json          # Contains artifacts for each trial
├── optimization_study.pkl
└── checkpoints/

extended_training_results/
├── extended_training_results.json
└── best_configuration_artifacts.json  # Artifacts for final training
```

## Usage

1. **Run hyperparameter tuning**: Artifacts are automatically saved
2. **Run extended training**: Automatically loads and uses artifacts when available
3. **Run final training**: Can use artifacts from extended training for consistency

## Migration Notes

- Existing trial results without artifacts will work with fallback mechanism
- New hyperparameter tuning runs will include artifacts
- Extended training will show warnings when using fallback mode

## Future Enhancements

1. **Artifact Versioning**: Add version numbers to artifacts for compatibility
2. **Artifact Validation**: Validate artifacts before use
3. **Artifact Compression**: Compress large artifacts for storage efficiency
4. **Artifact Cleanup**: Automatic cleanup of old artifacts 