# Third-Party Code Attributions

This document provides proper attribution for third-party code and libraries used in this project.

## LIF (Leaky Integrate-and-Fire) Layer Implementation

### Source
The LIF layer implementation in `src/models/lif_layer.py` is based on the **snnax** library.

### Original Implementation
- **Library**: [snnax](https://github.com/neuromorphs/snnax) - A JAX-based library for spiking neural networks
- **License**: MIT License
- **Authors**: The snnax development team

### Adaptations Made
The original snnax implementation has been adapted for use with Flax NNX:

1. **State Management**: Converted from snnax's state management to NNX Variables
2. **Module Structure**: Adapted to inherit from `nnx.Module` instead of snnax modules
3. **Parameter Handling**: Converted snnax parameters to NNX `Param` objects
4. **API Compatibility**: Maintained the same mathematical operations and neuron dynamics

### Classes Adapted
- `SimpleLIF`: Simple leaky integrate-and-fire neurons
- `LIF`: LIF neurons with synaptic currents
- `LIFSoftReset`: LIF with additive reset mechanism
- `AdaptiveLIF`: Adaptive exponential LIF neurons

### Surrogate Gradient Functions
The surrogate gradient functions are directly based on snnax implementations:
- `superspike_surrogate()`: SuperSpike surrogate gradient
- `sigmoid_surrogate()`: Sigmoidal surrogate gradient  
- `piecewise_surrogate()`: Piecewise linear surrogate gradient

### Mathematical Formulations
The core mathematical formulations for neuron dynamics remain unchanged from snnax:
- Membrane potential updates
- Synaptic current dynamics
- Spike generation and reset mechanisms
- Adaptive threshold dynamics

### Attribution Notice
```python
# Surrogate gradient functions (from snnax)
# Based on snnax implementation but adapted for NNX
# Original source: https://github.com/neuromorphs/snnax
```

## External Libraries and Dependencies

### Core JAX Ecosystem
- **JAX**: [Google JAX](https://github.com/google/jax) - High-performance machine learning
- **Flax**: [Google Flax](https://github.com/google/flax) - Neural network library for JAX
- **Optax**: [Google Optax](https://github.com/deepmind/optax) - Optimization library for JAX

### Data Processing
- **Pandas**: [Pandas](https://github.com/pandas-dev/pandas) - Data manipulation and analysis
- **NumPy**: [NumPy](https://github.com/numpy/numpy) - Numerical computing
- **Alpha Vantage**: [Alpha Vantage API](https://www.alphavantage.co/) - Stock market data
- **Polygon.io**: [Polygon.io API](https://polygon.io/) - Financial data and news

### Machine Learning
- **Optuna**: [Optuna](https://github.com/optuna/optuna) - Hyperparameter optimization
- **Scikit-learn**: [Scikit-learn](https://github.com/scikit-learn/scikit-learn) - Machine learning utilities
- **TensorBoard**: [TensorBoard](https://github.com/tensorflow/tensorboard) - Training visualization

### Visualization
- **Matplotlib**: [Matplotlib](https://github.com/matplotlib/matplotlib) - Plotting library
- **Rich**: [Rich](https://github.com/Textualize/rich) - Terminal output formatting

### Development Tools
- **Pytest**: [Pytest](https://github.com/pytest-dev/pytest) - Testing framework
- **Black**: [Black](https://github.com/psf/black) - Code formatting
- **Manim**: [Manim](https://github.com/ManimCommunity/manim) - Animation library

## License Compliance

### MIT License Dependencies
Most dependencies use the MIT License, which is compatible with this project's MIT License:
- JAX, Flax, Optax (Google)
- snnax (MIT License)
- Pandas, NumPy, Scikit-learn
- Matplotlib, Rich, Pytest

### Commercial API Services
- **Alpha Vantage**: Free tier available, commercial use requires paid subscription
- **Polygon.io**: Commercial service with various pricing tiers

## Code References

### Direct Code Usage
1. **LIF Layer Implementation**: Adapted from snnax library
   - File: `src/models/lif_layer.py`
   - Original: snnax LIF implementations
   - License: MIT (compatible)

### Algorithmic References
1. **Transformer Architecture**: Based on "Attention Is All You Need" (Vaswani et al., 2017)
2. **Positional Encoding**: Standard sinusoidal encoding from original transformer paper
3. **Multi-Head Attention**: Standard implementation from transformer literature

### Data Processing Patterns
1. **Time Series Preprocessing**: Common financial data preprocessing techniques
2. **News Sentiment Analysis**: Standard text processing and sentiment scoring methods
3. **Backtesting Framework**: Standard financial backtesting methodologies

## Acknowledgments

### Research Community
- **JAX Team**: For the excellent JAX ecosystem and documentation
- **Flax Team**: For the neural network library and NNX framework
- **snnax Contributors**: For the spiking neural network implementations
- **Optuna Team**: For the hyperparameter optimization framework

### Financial Data Providers
- **Alpha Vantage**: For providing free stock market data API
- **Polygon.io**: For comprehensive financial data and news services

### Open Source Community
- All contributors to the open-source libraries used in this project
- The broader machine learning and financial technology communities

## License Information

This project is licensed under the MIT License. All third-party code used in this project is either:
1. Licensed under compatible open-source licenses (MIT, Apache 2.0, BSD)
2. Used in accordance with their respective terms of service
3. Properly attributed as required by their licenses

## Contact

For questions about third-party code usage or licensing, please open an issue on the project repository.

---

**Note**: This attribution file is maintained to ensure proper credit is given to all third-party code and libraries used in this project. If you notice any missing attributions or have questions about specific code usage, please contact the project maintainers.
