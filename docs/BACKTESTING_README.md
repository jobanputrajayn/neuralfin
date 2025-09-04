# Backtesting Framework

## Overview

The backtesting framework provides a standalone interface for evaluating trained models on historical data. It has been extracted from the training pipeline to provide a clean, dedicated backtesting experience.

**NEW**: The backtesting engine now automatically detects and uses the best model from your hyperparameter tuning strategy!

**CRITICAL**: The model is sensitive to ticker sequence and order. The backtesting engine automatically ensures compatibility by using the exact same tickers in the exact same order as used during model training.

## 🚀 Quick Start

### Basic Backtesting (Auto-detect Best Model)

```bash
# Run backtesting with auto-detected best model from hyperparameter tuning strategy
# Uses exact training tickers automatically
python run_backtesting.py
```

### Use Specific Model

```bash
# Run backtesting on a specific trained model
# Uses exact training tickers automatically
python run_backtesting.py --model-path ./jax_gpt_stock_predictor_checkpoints
```

### Generate Latest Signals

```bash
# Generate latest trading signals with auto-detected model
# Uses exact training tickers automatically
python run_backtesting.py --generate-signals
```

## 📊 Features

### 1. Automatic Model Detection
- **Smart Detection**: Automatically finds the best model from your hyperparameter tuning strategy
- **Priority Order**: 
  1. Final model from final training (`./final_model`)
  2. Best model from extended training (`./hyperparameter_tuning_results/extended_model_*`)
  3. Best model from hyperparameter tuning (`./hyperparameter_tuning_results/checkpoints/*`)
  4. Default fallback models
- **Fallback Support**: Uses best available model if final model not found

### 2. Ticker Compatibility Protection
- **Exact Match**: Always uses the exact same tickers in the exact same order as training
- **Automatic Validation**: Validates ticker count and order to prevent model incompatibility
- **Smart Fallback**: If mismatched tickers provided, automatically uses training tickers
- **Clear Warnings**: Provides clear feedback about ticker mismatches

### 3. Comprehensive Backtesting
- **Historical Performance**: Evaluate model performance on historical data
- **Portfolio Analysis**: Aggregate individual ticker performance into portfolio metrics
- **Risk Metrics**: Calculate Sharpe ratio, drawdown, and other risk measures
- **Visualization**: Generate equity curves and performance plots

### 4. Signal Generation
- **Latest Signals**: Generate current trading signals for any ticker set
- **Confidence Scores**: Include prediction confidence for each signal
- **Real-time Ready**: Designed for integration with live trading systems

### 5. Flexible Configuration
- **Custom Tickers**: Test on any subset of tickers (must match training tickers exactly)
- **Adjustable Parameters**: Modify cash, commission rates, data periods
- **Multiple Output Formats**: JSON, CSV, and text reports

## 🔧 Usage Examples

### Example 1: Auto-detect Best Model

```bash
# Uses the best model from your hyperparameter tuning strategy
# Automatically uses exact training tickers
python run_backtesting.py \
    --data-period 1y \
    --initial-cash 50000 \
    --commission-rate 0.001
```

### Example 2: Use Specific Model

```bash
# Uses exact training tickers automatically
python run_backtesting.py \
    --model-path ./jax_gpt_stock_predictor_checkpoints \
    --data-period 1y \
    --initial-cash 50000 \
    --commission-rate 0.001
```

### Example 3: Generate Latest Signals

```bash
# Uses exact training tickers automatically
python run_backtesting.py \
    --generate-signals \
    --signals-period 1m \
    --output-dir ./signals
```

### Example 4: Comprehensive Analysis

```bash
# Uses exact training tickers automatically
python run_backtesting.py \
    --data-period 2y \
    --initial-cash 100000 \
    --commission-rate 0.002 \
    --output-dir ./detailed_analysis
```

## ⚠️ Ticker Compatibility

### Why Ticker Order Matters
The model is trained on a specific sequence of tickers, and the neural network weights are optimized for that exact order. Changing the ticker sequence can significantly impact model performance.

### Automatic Protection
The backtesting engine includes several safeguards:

1. **Training Ticker Detection**: Automatically loads the exact tickers used during training
2. **Count Validation**: Ensures the same number of tickers are provided
3. **Order Validation**: Ensures tickers are in the exact same order as training
4. **Smart Fallback**: If mismatched tickers are provided, automatically uses training tickers
5. **Clear Feedback**: Provides warnings and information about ticker usage

### Example Ticker Mismatch Handling
```bash
# If you provide different tickers, the system will warn you:
python run_backtesting.py --tickers AAPL MSFT GOOGL

# Output:
# ⚠️  Ticker order mismatch detected!
#    Training tickers: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']...
#    Provided tickers: ['AAPL', 'MSFT', 'GOOGL']...
#    Using training tickers to ensure model compatibility.
# ✅ Using training tickers: 50 tickers in exact training order
```

## 📈 Output Files

### Backtesting Results
- **`backtest_results.json`**: Complete backtesting results in JSON format
- **`backtest_report.txt`**: Detailed text report of individual ticker performance
- **`portfolio_report.txt`**: Portfolio-level analysis and metrics
- **`synthetic_portfolio_equity_curve_*.png`**: Portfolio equity curve plot

### Signal Generation
- **`latest_signals.csv`**: Current trading signals with confidence scores

## 🎯 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path` | Path to trained model checkpoint (optional, auto-detects if not provided) | Auto-detect |
| `--config-path` | Path to model configuration file | Auto-detect |
| `--tickers` | Tickers to backtest (WARNING: must match training tickers exactly) | Use training tickers |
| `--data-period` | Data period for backtesting | `2y` |
| `--initial-cash` | Initial cash per ticker | `10000` |
| `--commission-rate` | Commission rate for trades | `0.001` |
| `--output-dir` | Output directory for results | `./backtesting_results` |
| `--generate-signals` | Generate latest trading signals | False |
| `--signals-period` | Data period for signal generation | `1m` |
| `--no-plots` | Disable plot generation | False |

## 🔍 Model Detection Strategy

The backtesting engine automatically detects the best model using this priority order:

### 1. Final Model (Highest Priority)
- **Path**: `./final_model/` or `./final_model/best_model/`
- **When Available**: After running `python run_final_training.py`
- **Best For**: Production-ready model with extended training

### 2. Extended Training Model
- **Path**: `./hyperparameter_tuning_results/extended_model_*`
- **When Available**: After running `python run_extended_training.py`
- **Best For**: Best configuration with cross-validation

### 3. Hyperparameter Tuning Model
- **Path**: `./hyperparameter_tuning_results/checkpoints/*`
- **When Available**: After running `python run_hyperparameter_tuning.py`
- **Best For**: Best trial from initial optimization

### 4. Default Fallback
- **Path**: `./jax_gpt_stock_predictor_checkpoints`
- **When Available**: After basic training
- **Best For**: Basic model evaluation

## 📊 Understanding Results

### Individual Ticker Performance
Each ticker's backtest provides:
- **Total Return**: Overall percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade**: Average profit/loss per trade

### Portfolio Performance
Aggregated metrics include:
- **Portfolio Return**: Overall portfolio performance
- **Portfolio Sharpe**: Risk-adjusted portfolio return
- **Correlation Analysis**: How tickers move together
- **Diversification Benefits**: Risk reduction from multiple positions

### Signal Quality
For signal generation:
- **Predicted Action**: HOLD, BUY_CALL, or BUY_PUT
- **Confidence Score**: Model's confidence in prediction (0-1)
- **Latest Price**: Current market price for context

## 🔍 Advanced Usage

### Programmatic Usage

```python
from src.scripts.backtesting import BacktestingEngine

# Initialize engine with auto-detection
engine = BacktestingEngine()  # Auto-detects best model

# Or specify a model
engine = BacktestingEngine("./jax_gpt_stock_predictor_checkpoints")

# Run backtesting (uses training tickers automatically)
results = engine.run_backtest(
    data_period='1y',
    initial_cash=50000,
    commission_rate=0.001
)

# Generate signals (uses training tickers automatically)
signals = engine.generate_signals(
    data_period='1m'
)

# Print results
engine.print_results_summary(results)
```

### Custom Analysis

```python
# Load results
import json
with open('./backtesting_results/backtest_results.json', 'r') as f:
    results = json.load(f)

# Analyze specific metrics
equity_curves = results['equity_curves']
for ticker, curve in equity_curves.items():
    final_value = curve['Equity'].iloc[-1]
    initial_value = curve['Equity'].iloc[0]
    total_return = (final_value - initial_value) / initial_value
    print(f"{ticker}: {total_return:.2%}")
```

## 🚨 Important Notes

### Model Compatibility
- **Ticker Sequence**: The model is sensitive to the exact sequence of tickers used during training
- **Automatic Protection**: The backtesting engine automatically ensures ticker compatibility
- **Configuration Match**: Ensure the model configuration matches the training configuration
- **Dependencies**: Verify that all required dependencies are installed

### Data Requirements
- **Sufficient Historical Data**: Required for meaningful backtesting
- **Data Quality**: Affects backtesting accuracy
- **Market Conditions**: Consider market conditions during the backtesting period

### Performance Considerations
- **Large Ticker Sets**: Require more computation time
- **GPU Acceleration**: Used when available
- **Memory Usage**: Scales with data size and model complexity

## 🔧 Troubleshooting

### Common Issues

1. **Ticker Mismatch Error**
   ```
   Error: Ticker count mismatch! Model was trained on 50 tickers, but 30 tickers provided
   Solution: The system will automatically use training tickers. No action needed.
   ```

2. **No Model Found Error**
   ```
   Error: No valid model found. Please ensure you have completed the hyperparameter tuning strategy
   Solution: Run the complete pipeline: hyperparameter tuning → extended training → final training
   ```

3. **Model Loading Errors**
   ```
   Error: Model checkpoint not found
   Solution: Verify the model path and ensure the checkpoint exists
   ```

4. **Data Download Issues**
   ```
   Error: No data downloaded
   Solution: Check internet connection and ticker symbols
   ```

### Performance Optimization

- **GPU Usage**: Ensure GPU is available for faster processing
- **Memory Management**: Monitor memory usage with large ticker sets
- **Batch Processing**: The system automatically optimizes batch sizes

## 📚 Integration

### With Training Pipeline
After training a model:
```bash
# Train model
python run_training.py

# Backtest the trained model
python run_backtesting.py --model-path ./jax_gpt_stock_predictor_checkpoints
```

### With Live Trading
For production use:
```bash
# Generate daily signals
python run_backtesting.py \
    --model-path ./production_model \
    --generate-signals \
    --signals-period 1d \
    --output-dir ./daily_signals
```

## 🎉 Expected Outcomes

With proper backtesting, you should achieve:

1. **Performance Insights**: Understand model strengths and weaknesses
2. **Risk Assessment**: Quantify potential losses and volatility
3. **Strategy Validation**: Verify trading strategy effectiveness
4. **Production Readiness**: Ensure model is ready for live trading

This backtesting framework provides the tools needed to thoroughly evaluate your trained models before deploying them in production. 