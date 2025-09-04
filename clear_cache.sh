#!/bin/bash

# Clear Cache Script for Hyperparameter Tuning Project
# This script removes all cached data, logs, and checkpoints from the complete pipeline

echo "🧹 Clearing caches, logs, and checkpoints from complete pipeline..."

# Clear data cache
if [ -d "data_cache" ]; then
    echo "📁 Removing data cache..."
    rm -rf data_cache
    echo "✅ Data cache cleared"
else
    echo "ℹ️  Data cache directory not found"
fi

# Clear news cache
if [ -d "news_cache" ]; then
    echo "📁 Removing news cache..."
    rm -rf news_cache
    echo "✅ News cache cleared"
else
    echo "ℹ️  News cache directory not found"
fi

# Clear hyperparameter tuning results
if [ -d "hyperparameter_tuning_results" ]; then
    echo "📁 Removing hyperparameter tuning results..."
    rm -rf hyperparameter_tuning_results
    echo "✅ Hyperparameter tuning results cleared"
else
    echo "ℹ️  Hyperparameter tuning results directory not found"
fi

# Clear extended training results
if [ -d "extended_training_results" ]; then
    echo "📁 Removing extended training results..."
    rm -rf extended_training_results
    echo "✅ Extended training results cleared"
else
    echo "ℹ️  Extended training results directory not found"
fi

# Clear backtesting results
if [ -d "backtesting_results" ]; then
    echo "📁 Removing backtesting results..."
    rm -rf backtesting_results
    echo "✅ Backtesting results cleared"
else
    echo "ℹ️  Backtesting results directory not found"
fi

# Clear analysis plots
if [ -d "analysis_plots" ]; then
    echo "📁 Removing analysis plots..."
    rm -rf analysis_plots
    echo "✅ Analysis plots cleared"
else
    echo "ℹ️  Analysis plots directory not found"
fi

# Clear final model
if [ -d "final_model" ]; then
    echo "📁 Removing final model..."
    rm -rf final_model
    echo "✅ Final model cleared"
else
    echo "ℹ️  Final model directory not found"
fi

# Clear best configuration file
if [ -f "best_config.json" ]; then
    echo "📄 Removing best configuration file..."
    rm best_config.json
    echo "✅ Best configuration file removed"
else
    echo "ℹ️  Best configuration file not found"
fi

# Clear logs
if [ -f "hyperparameter_tuning.log" ]; then
    echo "📄 Removing log file..."
    rm hyperparameter_tuning.log
    echo "✅ Log file removed"
else
    echo "ℹ️  Log file not found"
fi

# Clear Python cache
if [ -d "__pycache__" ]; then
    echo "🐍 Removing Python cache..."
    rm -rf __pycache__
    echo "✅ Python cache cleared"
else
    echo "ℹ️  Python cache directory not found"
fi

# Clear any .pyc files
echo "🔍 Removing .pyc files..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clear TensorBoard logs
if [ -d "tensorboard_logs" ]; then
    echo "📁 Removing TensorBoard logs..."
    rm -rf tensorboard_logs
    echo "✅ TensorBoard logs cleared"
else
    echo "ℹ️  TensorBoard logs directory not found"
fi

# Clear any other potential log files
echo "🔍 Removing other log files..."
find . -name "*.log" -type f -delete 2>/dev/null || true

# Clear any temporary files
echo "🔍 Removing temporary files..."
find . -name "*.tmp" -type f -delete 2>/dev/null || true
find . -name "*.temp" -type f -delete 2>/dev/null || true

# Clear PNG images
echo "🖼️  Removing all .png files..."
find . -name "*.png" -type f -delete 2>/dev/null || true

echo "🎉 All caches, logs, and checkpoints from complete pipeline cleared!"
echo "💡 You can now start fresh with the complete hyperparameter tuning pipeline" 