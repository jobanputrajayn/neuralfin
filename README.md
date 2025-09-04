# NEURALFIN - JAX GPT Stock Predictor

*Codename: "NeuralFin" - A neural network journey into financial markets*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![JAX](https://img.shields.io/badge/JAX-0.4+-orange.svg)](https://jax.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Educational](https://img.shields.io/badge/Type-Educational%20Project-purple.svg)](https://github.com/jobanputrajyan/neuralfin)
[![Learning Project](https://img.shields.io/badge/Type-Learning%20Project-blue.svg)](https://github.com/jobanputrajyan/neuralfin)
[![Personal Project](https://img.shields.io/badge/Type-Personal%20Project-green.svg)](https://github.com/jobanputrajyan/neuralfin)

## Project Overview

This is my personal learning project where I dove deep into neural networks and machine learning. I built this to understand how modern AI actually works under the hood, especially when it comes to financial data and stock prediction.

**What I was learning**: I wanted to get hands-on experience with neural networks, so I picked a challenging problem - predicting stock movements using both price data and news sentiment. It's been an amazing way to learn about transformer architectures, attention mechanisms, and how to actually implement these concepts in code.

**How I built it**: I basically "vibe coded" my way through this - experimenting, trying things out, and learning as I went. It's not the most polished approach, but it let me explore different ideas and really understand what I was building. The code shows my problem-solving process and how I figured out how to piece together complex ML frameworks.

**Current state**: Let's be real - the model's accuracy is basically random right now. But honestly, that's kind of expected when you're learning and trying to predict something as chaotic as stock markets. The real value is in everything I learned building it.

**What I actually built**: Even though the predictions aren't great, I'm pretty proud of what I accomplished:
- Built a complete neural network from scratch using JAX and Flax
- Integrated multiple data sources (stock prices + news sentiment)
- Set up proper hyperparameter tuning and evaluation pipelines
- Organized everything into a maintainable codebase
- Added comprehensive testing and monitoring

---

A machine learning project that uses JAX and GPT-style neural networks to predict stock price movements based on historical OHLC data and news sentiment analysis.

## 🚀 Quick Start

### Get Started Immediately
1. **Install dependencies**: `./install_requirements.sh`
2. **Configure API keys**: Edit `src/config/av_key.py` and `src/config/polygon_key.py`
3. **Run pipeline**: Choose your preferred method:
   - **Testing**: `./run_complete_pipeline_subset.sh`
   - **Production**: `./run_hyperparameter_tuning_background.sh`
   - **Interactive**: `./run_complete_pipeline.sh`

### Learn More
1. **New to the project?** Start with the [Model Architecture](docs/model_architecture_diagram.md) to understand the system
2. **Setting up training?** Follow the [Pipeline Coordination](docs/PIPELINE_COORDINATION_ANALYSIS.md) guide
3. **Optimizing hyperparameters?** Read the [Hyperparameter Strategy](docs/HYPERPARAMETER_STRATEGY.md)
4. **Evaluating models?** Use the [Backtesting Framework](docs/BACKTESTING_README.md)

## 📊 Key Features

### Advanced Neural Network Architecture
- **JAX-based Neural Network**: Uses JAX for high-performance machine learning with GPU acceleration
- **GPT-style Architecture**: Implements a transformer-based model for sequence prediction
- **LIF Layers**: Leaky Integrate-and-Fire layers for temporal processing
- **Multi-head Attention**: Captures complex patterns in financial data

### Advanced Hyperparameter Tuning
- **3-Phase Strategy**: Random → Bayesian → Fine-tune optimization
- **Cross-Validation**: Robust evaluation with k-fold validation
- **Resource Efficiency**: 43% reduction in computation time
- **Comprehensive Logging**: Batch-level TensorBoard monitoring

### Robust Data Pipeline
- **Multi-modal Data**: Combines stock price data (OHLC) with news sentiment analysis
- **Data Consistency**: Artifact system ensures reproducible results
- **Caching System**: Efficient data preprocessing and storage
- **API Integration**: Alpha Vantage and Polygon.io support

### Production-Ready Evaluation
- **Comprehensive Backtesting**: Historical performance evaluation with detailed metrics
- **Risk Metrics**: Sharpe ratio, drawdown, and volatility analysis
- **Signal Generation**: Real-time trading signal production
- **Model Compatibility**: Automatic ticker sequence validation

## Project Structure

```
hyper/
├── src/                          # Source code
│   ├── config/                   # Configuration files
│   │   ├── av_key.py            # Alpha Vantage API configuration
│   │   ├── polygon_key.py       # Polygon.io API configuration
│   │   ├── hyperparameter_config.py
│   │   └── news_config.py
│   ├── data/                     # Data processing modules
│   │   ├── stock_data.py
│   │   ├── news_data.py
│   │   ├── sequence_generator.py
│   │   └── unified_news_data.py
│   ├── models/                   # Model implementations
│   │   ├── gpt_classifier.py
│   │   ├── lif_layer.py
│   │   └── backtesting.py
│   ├── training/                 # Training utilities
│   │   ├── training_functions.py
│   │   └── checkpointing.py
│   ├── hyperparameter_tuning/   # Optimization framework
│   │   └── optimization.py
│   ├── scripts/                  # Main execution scripts
│   │   ├── training_pipeline.py
│   │   ├── hyperparameter_tuner.py
│   │   ├── model_evaluator.py
│   │   └── backtesting.py
│   └── utils/                    # Utility functions
│       ├── gpu_utils.py
│       ├── jax_memory_stats.py
│       └── system_utils.py
├── docs/                         # Documentation
│   ├── API_REFERENCE.md          # API reference links
│   ├── BACKTESTING_README.md     # Backtesting framework guide
│   ├── HYPERPARAMETER_STRATEGY.md # Hyperparameter tuning strategy
│   ├── PIPELINE_COORDINATION_ANALYSIS.md # Pipeline coordination guide
│   ├── BATCH_LOGGING_README.md   # TensorBoard logging guide
│   ├── ARTIFACT_SYSTEM_IMPLEMENTATION.md # Data consistency system
│   ├── model_architecture_diagram.md # Model architecture visualization
│   ├── NEWS_DATA_README.md       # News data processing guide
│   └── ATTRIBUTIONS.md           # Third-party code attributions
├── animation/                     # Visualization and animation tools
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hyper
   ```

2. **Install dependencies** (automatic GPU detection):
   ```bash
   # Automatic installation with GPU detection
   ./install_requirements.sh
   
   # Or manual installation
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   - Edit `src/config/av_key.py` and add your Alpha Vantage API key
   - Edit `src/config/polygon_key.py` and add your Polygon.io API key
   - Get your free Alpha Vantage API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Get your Polygon.io API key from [Polygon.io](https://polygon.io/)

## 🔧 Usage

### Primary Entry Points (Recommended)

#### Quick Testing
```bash
# Run complete pipeline with subset parameters (recommended for testing)
./run_complete_pipeline_subset.sh
```
**Best for**: First-time users, testing, development

#### Production Pipeline
```bash
# Run complete pipeline in background with logging
./run_hyperparameter_tuning_background.sh

# Monitor progress
./monitor_pipeline.sh
```
**Best for**: Production runs, long training sessions, server environments

#### Full Interactive Pipeline
```bash
# Run complete pipeline with full parameters and user interaction
./run_complete_pipeline.sh
```
**Best for**: Full dataset training, research, when you want to monitor each step

### Individual Components (Advanced Users)

#### Basic Training
```bash
python run_training.py
```

#### Hyperparameter Optimization
```bash
python run_hyperparameter_tuning.py
```

#### Extended Training
```bash
python run_extended_training.py
```

#### Backtesting
```bash
python run_backtesting.py
```

### Shell Scripts

#### `install_requirements.sh`
Automatically detects GPU/TPU availability and installs the appropriate JAX version:
- **GPU**: Installs JAX with CUDA12 support
- **TPU**: Installs JAX with TPU support  
- **CPU**: Installs standard JAX for CPU

#### `run_complete_pipeline_subset.sh` (Quick Testing)
Runs the complete ML pipeline with subset parameters for testing:
- **Step 1**: Hyperparameter tuning (2 tickers, minimal trials)
- **Step 2**: Extended training with cross-validation
- **Step 3**: Results analysis and best config selection
- **Step 4**: Final model training
- **Step 5**: Backtesting with historical data
- **Step 6**: Signal generation

**Options**:
- `--start-step N`: Start from specific step (1-6)
- `--no-confirm`: Run without user confirmation
- `--help`: Show usage information

#### `run_hyperparameter_tuning_background.sh` (Production Pipeline)
Runs the complete pipeline in background with comprehensive logging:
- Creates timestamped log files in `logs/` directory
- Separates stdout, stderr, and script output
- Saves process ID for monitoring
- Continues running even if terminal is closed

**Monitoring**:
- Use `./monitor_pipeline.sh` to monitor progress
- Check `logs/` directory for detailed logs

#### `run_complete_pipeline.sh` (Full Interactive Pipeline)
Runs the complete ML pipeline with full parameters and user interaction:
- **Step 1**: Hyperparameter tuning (full dataset, comprehensive trials)
- **Step 2**: Extended training with cross-validation
- **Step 3**: Results analysis and best config selection
- **Step 4**: Final model training
- **Step 5**: Backtesting with historical data
- **Step 6**: Signal generation

**Options**:
- `--start-step N`: Start from specific step (1-6)
- `--no-confirm`: Run without user confirmation
- `--help`: Show usage information

## Configuration

### API Keys
- **Alpha Vantage**: Get your free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
- **Polygon.io**: Get your API key from [Polygon.io](https://polygon.io/)

### Data Sources
- **Stock Data**: Alpha Vantage API for historical OHLC data
- **News Data**: Polygon.io API for financial news and sentiment analysis

### Model Configuration
Key hyperparameters can be configured in `src/config/hyperparameter_config.py`:
- Sequence length
- Model dimensions
- Learning rate
- Training epochs
- Batch size

## 🔧 Development Workflow

### 1. Initial Setup
- Configure API keys (see main README)
- Install dependencies using `./install_requirements.sh` (automatic GPU detection)
- Review [Pipeline Coordination](docs/PIPELINE_COORDINATION_ANALYSIS.md) for understanding the workflow

### 2. Quick Testing
- Use `./run_complete_pipeline_subset.sh` for quick testing with subset parameters
- Use `./run_hyperparameter_tuning_background.sh` for production background processing
- Use `./run_complete_pipeline.sh` for full interactive pipeline

### 3. Model Training
- Start with [Hyperparameter Strategy](docs/HYPERPARAMETER_STRATEGY.md) for optimization
- Use [TensorBoard Logging](docs/BATCH_LOGGING_README.md) for monitoring
- Follow [Data Consistency](docs/ARTIFACT_SYSTEM_IMPLEMENTATION.md) for reproducible results

### 4. Model Evaluation
- Run [Backtesting Framework](docs/BACKTESTING_README.md) for performance evaluation
- Analyze results using the comprehensive metrics provided

### 5. Data Integration
- Set up [News Data Processing](docs/NEWS_DATA_README.md) for sentiment analysis
- Use [Polygon News Downloader](docs/POLYGON_NEWS_DOWNLOADER_README.md) for data acquisition

## 🎯 Best Practices

### Training
- Always run the complete pipeline in order
- Use cross-validation for robust model selection
- Monitor training with TensorBoard logging
- Ensure data consistency with artifact system

### Evaluation
- Use the backtesting framework for comprehensive evaluation
- Validate model compatibility with training tickers
- Analyze multiple performance metrics
- Test on different market conditions

### Development
- Follow the pipeline coordination guidelines
- Use the artifact system for reproducible results
- Monitor system resources during training
- Document any custom configurations

## Dependencies

- **JAX**: High-performance machine learning framework
- **Flax**: Neural network library for JAX
- **Optuna**: Hyperparameter optimization
- **Alpha Vantage**: Stock market data API
- **Polygon.io**: Financial news and data API
- **TensorBoard**: Training visualization
- **Rich**: Terminal output formatting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📁 Repository Information

**Repository**: `neuralfin`  
**GitHub URL**: `https://github.com/jobanputrajyan/neuralfin`  
**Clone Command**: `git clone https://github.com/jobanputrajyan/neuralfin.git`

### Repository Structure
- `src/` - Core source code and modules
- `docs/` - Comprehensive documentation
- `animation/` - Visualization and animation tools
- `logs/` - Training logs and TensorBoard outputs
- `data/` - Data cache and storage (gitignored)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It is not intended for actual trading or investment decisions. Always do your own research and consult with financial professionals before making investment decisions.

## 🔍 Troubleshooting

### Common Issues
- **Ticker Mismatch**: Ensure exact ticker sequence compatibility
- **Data Inconsistency**: Use artifact system for reproducible splits
- **Memory Issues**: Monitor GPU/CPU usage with TensorBoard
- **API Limits**: Check API key configuration and usage limits

### Getting Help
- Check the relevant documentation file for your specific issue
- Review the [Pipeline Coordination](docs/PIPELINE_COORDINATION_ANALYSIS.md) for workflow issues
- Use [TensorBoard Logging](docs/BATCH_LOGGING_README.md) for training monitoring
- Consult [Backtesting Framework](docs/BACKTESTING_README.md) for evaluation problems

## 📈 Performance Optimization

### Training Efficiency
- Use the 3-phase hyperparameter strategy
- Implement early stopping with appropriate patience
- Monitor batch processing times
- Optimize batch sizes for your hardware

### Data Processing
- Leverage caching for repeated data access
- Use appropriate data periods for your use case
- Monitor memory usage during preprocessing
- Implement efficient data loading patterns

### Model Evaluation
- Use cross-validation for robust evaluation
- Test on multiple time periods
- Validate on different market conditions
- Monitor model performance over time

## 🔄 Continuous Improvement

### Iterative Development
1. **Analyze Results**: Use backtesting and logging to understand performance
2. **Identify Bottlenecks**: Monitor system resources and training efficiency
3. **Optimize Configuration**: Adjust hyperparameters based on results
4. **Validate Changes**: Ensure improvements with comprehensive testing
5. **Document Updates**: Keep documentation current with changes

### Research Integration
- Experiment with new architectures
- Test different data sources
- Explore advanced optimization techniques
- Contribute improvements back to the project

## 📚 Documentation

Comprehensive documentation is available in the `docs/` folder:

### Core Guides
- **[API Reference](docs/API_REFERENCE.md)**: Links to JAX, Flax, and related library documentation
- **[Model Architecture](docs/model_architecture_diagram.md)**: Visual guide to the GPT-style neural network architecture

### Training & Optimization
- **[Hyperparameter Strategy](docs/HYPERPARAMETER_STRATEGY.md)**: Advanced 3-phase hyperparameter tuning approach
- **[Pipeline Coordination](docs/PIPELINE_COORDINATION_ANALYSIS.md)**: Complete ML pipeline workflow and dependencies
- **[TensorBoard Logging](docs/BATCH_LOGGING_README.md)**: Enhanced batch-level monitoring and visualization

### Data & Evaluation
- **[Backtesting Framework](docs/BACKTESTING_README.md)**: Comprehensive backtesting and model evaluation
- **[Data Consistency](docs/ARTIFACT_SYSTEM_IMPLEMENTATION.md)**: Ensuring reproducible results across pipeline stages
- **[News Data Processing](docs/NEWS_DATA_README.md)**: Financial news integration and sentiment analysis
- **[Polygon News Downloader](docs/POLYGON_NEWS_DOWNLOADER_README.md)**: News data acquisition tools
- **[Polygon News Integration](docs/POLYGON_NEWS_README.md)**: News data processing pipeline
- **[Preprocessing Cache Flow](docs/PREPROCESSING_CACHE_FLOW.md)**: Data preprocessing and caching system
- **[Third-Party Attributions](docs/ATTRIBUTIONS.md)**: Proper attribution for third-party code and libraries

## Support

For questions and support, please open an issue on GitHub.
