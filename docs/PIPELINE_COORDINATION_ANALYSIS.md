# Pipeline Coordination Analysis

## Overview

This document analyzes the proper wiring and coordination between the four main scripts in the hyperparameter tuning strategy pipeline:

1. `run_hyperparameter_tuning.py` - Initial hyperparameter optimization
2. `run_extended_training.py` - Extended training with cross-validation
3. `run_final_training.py` - Final model training
4. `run_backtesting.py` - Model evaluation and backtesting

## 🔗 Pipeline Flow

```
run_hyperparameter_tuning.py
    ↓ (creates ./hyperparameter_tuning_results/)
    ↓ optimization_study.pkl
    ↓ trial_results.json
    ↓ checkpoints/
    ↓
run_extended_training.py
    ↓ (reads from ./hyperparameter_tuning_results/)
    ↓ (creates extended_training_results.json)
    ↓
analyze_results.py
    ↓ (reads both results, compares, saves best_config.json)
    ↓
run_final_training.py
    ↓ (reads best_config.json, creates ./final_model/)
    ↓ config.json
    ↓ best_model/
    ↓ final_model/
    ↓ training_results.json
    ↓
run_backtesting.py
    ↓ (auto-detects best model from ./final_model/)
    ↓ (uses exact training tickers from config.json)
```

## 📁 File Structure and Dependencies

### 1. Hyperparameter Tuning Output (`./hyperparameter_tuning_results/`)

**Created by:** `run_hyperparameter_tuning.py`

**Required files:**
- `optimization_study.pkl` - Optuna study object
- `trial_results.json` - Trial results and configurations
- `checkpoints/` - Model checkpoints for each trial
- `optimization_history.json` - Optimization progress
- `optimization_history.png` - Optimization plot

**Used by:** `run_extended_training.py`, `analyze_results.py`

### 2. Extended Training Output (`./extended_training_results/`)

**Created by:** `run_extended_training.py`

**Additional files:**
- `extended_training_results.json` - Cross-validation results

**Used by:** `analyze_results.py`

### 3. Analysis Output (Root directory)

**Created by:** `analyze_results.py`

**Files:**
- `best_config.json` - Best configuration for final training
- `./analysis_plots/` - Analysis visualizations

**Used by:** `run_final_training.py`

### 4. Final Training Output (`./final_model/`)

**Created by:** `run_final_training.py`

**Files:**
- `config.json` - Training configuration with tickers
- `best_model/` - Best model checkpoint
- `final_model/` - Final model checkpoint
- `training_results.json` - Training metrics

**Used by:** `run_backtesting.py`

## 🔍 Detailed Script Analysis

### 1. `run_hyperparameter_tuning.py`

**Purpose:** Initial hyperparameter optimization with 3-phase approach

**Input:** None (uses default configuration)

**Output:** `./hyperparameter_tuning_results/`
- `optimization_study.pkl` - Optuna study
- `trial_results.json` - Trial results
- `checkpoints/` - Model checkpoints

**Key Features:**
- 3-phase optimization (random → Bayesian → fine-tune)
- Saves trial results with configurations
- Creates model checkpoints for each trial

**Dependencies:** None

**Next Step:** `run_extended_training.py`

### 2. `run_extended_training.py`

**Purpose:** Extended training of top configurations with cross-validation

**Input:** `./hyperparameter_tuning_results/`
- Reads `optimization_study.pkl`
- Reads `trial_results.json`

**Output:** `./extended_training_results/extended_training_results.json`

**Key Features:**
- Loads top N configurations from hyperparameter tuning
- Trains each configuration with cross-validation
- Saves cross-validation results

**Dependencies:** 
- `./hyperparameter_tuning_results/optimization_study.pkl`
- `./hyperparameter_tuning_results/trial_results.json`

**Next Step:** `analyze_results.py`

### 3. `analyze_results.py`

**Purpose:** Compare and select best configuration

**Input:** 
- `./hyperparameter_tuning_results/` (for `optimization_study.pkl`)
- `./extended_training_results/` (for `extended_training_results.json`)

**Output:** `best_config.json`

**Key Features:**
- Compares hyperparameter tuning vs extended training results
- Selects best configuration based on accuracy improvement
- Saves best configuration with metadata

**Dependencies:**
- `./hyperparameter_tuning_results/optimization_study.pkl`
- `./extended_training_results/extended_training_results.json`

**Next Step:** `run_final_training.py`

### 4. `run_final_training.py`

**Purpose:** Train final production model

**Input:** `best_config.json`

**Output:** `./final_model/`
- `config.json` - Training configuration with tickers
- `best_model/` - Best model checkpoint
- `final_model/` - Final model checkpoint
- `training_results.json` - Training metrics

**Key Features:**
- Loads best configuration from analysis
- Trains model with full dataset
- Saves configuration with tickers for backtesting

**Dependencies:** `best_config.json`

**Next Step:** `run_backtesting.py`

### 5. `run_backtesting.py`

**Purpose:** Evaluate model performance

**Input:** Auto-detects best model from:
1. `./final_model/` (highest priority)
2. `./extended_training_results/extended_model_*`
3. `./hyperparameter_tuning_results/checkpoints/*`
4. Default fallback

**Output:** `./backtesting_results/`

**Key Features:**
- Auto-detects best model from pipeline
- Uses exact training tickers from `config.json`
- Validates ticker compatibility

**Dependencies:** Any model checkpoint with `config.json`

## ⚠️ Critical Coordination Points

### 1. Ticker Sequence Consistency

**Issue:** Model is sensitive to ticker sequence and order

**Solution:** 
- `run_final_training.py` saves tickers in `config.json`
- `run_backtesting.py` automatically uses training tickers
- Validation prevents ticker mismatches

### 2. Configuration Propagation

**Issue:** Configuration must flow through all stages

**Solution:**
- Hyperparameter tuning saves trial configurations
- Extended training uses same configurations
- Analysis selects best configuration
- Final training uses selected configuration
- Backtesting reads configuration for tickers

### 3. Model Checkpoint Compatibility

**Issue:** Model checkpoints must be compatible across stages

**Solution:**
- All stages use same model architecture
- Checkpoints include configuration metadata
- Auto-detection handles different checkpoint formats

## 🔧 Usage Examples

### Complete Pipeline

```bash
# 1. Run hyperparameter tuning
python run_hyperparameter_tuning.py

# 2. Run extended training
python run_extended_training.py

# 3. Analyze results and get best config
python analyze_results.py

# 4. Train final model
python run_final_training.py --config best_config.json

# 5. Run backtesting (auto-detects model)
python run_backtesting.py
```

### Individual Scripts

```bash
# Extended training with custom parameters
python run_extended_training.py --top-n 3 --epochs 150

# Final training with custom epochs
python run_final_training.py --config best_config.json --epochs 300

# Backtesting with specific model
python run_backtesting.py --model-path ./final_model

# Generate signals
python run_backtesting.py --generate-signals
```

## ✅ Validation Checklist

### Before Running Each Script

1. **Hyperparameter Tuning**
   - [ ] No existing results (or backup existing)
   - [ ] Sufficient disk space for checkpoints
   - [ ] GPU available for training

2. **Extended Training**
   - [ ] `./hyperparameter_tuning_results/` exists
   - [ ] `optimization_study.pkl` exists
   - [ ] `trial_results.json` exists

3. **Analysis**
   - [ ] `./hyperparameter_tuning_results/` exists
   - [ ] `./extended_training_results/extended_training_results.json` exists
   - [ ] Sufficient disk space for plots

4. **Final Training**
   - [ ] `best_config.json` exists
   - [ ] Sufficient disk space for model
   - [ ] GPU available for training

5. **Backtesting**
   - [ ] Model checkpoint exists
   - [ ] `config.json` with tickers exists
   - [ ] Sufficient disk space for results

## 🚨 Common Issues and Solutions

### 1. Missing Dependencies

**Issue:** Script fails due to missing input files

**Solution:** Run scripts in correct order, check file existence

### 2. Ticker Mismatch

**Issue:** Backtesting fails due to ticker incompatibility

**Solution:** System automatically uses training tickers, provides warnings

### 3. Model Not Found

**Issue:** Backtesting can't find model

**Solution:** Check model paths, ensure final training completed

### 4. Configuration Errors

**Issue:** Configuration doesn't propagate correctly

**Solution:** Verify `best_config.json` format, check configuration loading

## 📊 Monitoring and Debugging

### File Existence Checks

```bash
# Check hyperparameter tuning results
ls -la ./hyperparameter_tuning_results/

# Check extended training results
ls -la ./extended_training_results/extended_training_results.json

# Check best configuration
ls -la best_config.json

# Check final model
ls -la ./final_model/

# Check backtesting results
ls -la ./backtesting_results/
```

### Log Analysis

Each script provides detailed logging:
- Progress bars for long operations
- Error messages with context
- Success confirmations
- Next step instructions

### Configuration Validation

```bash
# Validate best configuration
python -c "import json; config=json.load(open('best_config.json')); print('Valid config:', 'config' in config)"

# Check model configuration
python -c "import json; config=json.load(open('./final_model/config.json')); print('Tickers:', len(config.get('tickers', [])))"
```

## 🎯 Best Practices

1. **Run Complete Pipeline:** Always run all stages in order
2. **Monitor Progress:** Watch for errors and warnings
3. **Validate Results:** Check file existence between stages
4. **Backup Results:** Keep copies of important results
5. **Use Auto-Detection:** Let backtesting auto-detect best model
6. **Check Tickers:** Ensure ticker compatibility in backtesting

## 🔄 Pipeline Automation

For automated runs, consider:

1. **Shell Script:** Create wrapper script for complete pipeline
2. **Error Handling:** Add proper error checking between stages
3. **Logging:** Implement comprehensive logging
4. **Cleanup:** Add cleanup options for intermediate files
5. **Parallelization:** Run independent stages in parallel where possible

The pipeline is well-coordinated with clear dependencies and proper file handling. The auto-detection features in backtesting make it robust and user-friendly. 