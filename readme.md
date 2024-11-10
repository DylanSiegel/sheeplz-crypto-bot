# n-LNN: A Deep Reinforcement Learning Platform for Trading BTC/USDT Derivatives Futures

This repository contains the code for n-LNN, a deep reinforcement learning (DRL) platform designed for automated trading of Bitcoin/USDT perpetual futures contracts. n-LNN utilizes a novel recurrent neural network architecture with unique properties to handle the complex and volatile nature of the cryptocurrency derivatives market.

## Key Features

* **Novel n-LNN Architecture:**  A custom recurrent neural network (RNN) with Hypersphere Normalization, Spherical Linear Interpolation (SLERP), Eigen Learning Rates, and Scaling Factors for enhanced stability and efficient learning.
* **Advanced DRL Algorithms:**  Implementation of Asynchronous Proximal Policy Optimization (APPO) and support for other DRL algorithms.
* **Deep Exploration Techniques:**  Integration of intrinsic motivation and curiosity-driven exploration to encourage the agent to discover profitable trading strategies.
* **Hierarchical Reinforcement Learning (HRL) Ready:** Designed for future implementation of HRL to handle different levels of decision-making (portfolio allocation, trade execution, risk management).
* **Multi-Agent Reinforcement Learning (MARL) Ready:**  Architecture allows for future development of MARL systems with agents employing diverse strategies.
* **Comprehensive Backtesting and Evaluation:** Tools for evaluating agent performance using metrics like Sharpe ratio, maximum drawdown, and win/loss ratio.
* **Python 12, PyTorch, and Gymnasium:** Built using modern and efficient tools for deep learning and reinforcement learning.

# n-LNN: Neural Liquid Market State Analysis Platform for Cryptocurrency Trading

This repository implements a novel Neural Liquid Neural Network (n-LNN) architecture for analyzing and visualizing cryptocurrency market states, with a focus on BTC/USDT trading pairs. The platform uses advanced dimensionality reduction and visualization techniques to identify market patterns and potential trading opportunities.

## Architecture Overview

### Core Components

1. **HypersphericalEncoder**
   - Custom neural network architecture for encoding market states
   - GRU-based sequence processing with bidirectional layers
   - Projects market data onto a normalized hypersphere
   - Temperature-scaled normalization for stable representations

2. **Market State Processing**
   - Efficient data handling with caching mechanisms
   - Separate scaling for price and technical indicators
   - Support for both standard and min-max scaling
   - Automated batch processing with CUDA optimization

3. **Visualization Engine**
   - Interactive 3D and 2D projections using both t-SNE and PCA
   - Multiple synchronized views (3D, Top, Front, Side)
   - Real-time hover information for detailed state analysis
   - Support for various color-coding schemes based on market metrics

## Technical Details

### Hardware Requirements
- CUDA-capable GPU (tested on NVIDIA 3070)
- 32GB RAM recommended
- Modern multi-core CPU (tested on AMD Ryzen 9 7900X)

### Software Dependencies
```
Python 3.12+
PyTorch 2.0+
plotly
scikit-learn
pandas
numpy
```

### Project Structure
```
.
├── ai_module.py        # Neural network architecture
├── dataset.py          # Data handling and preprocessing
├── visualization.py    # Visualization components
└── main.py            # Main execution script
```

## Features

### Market State Encoding
- Sequence-based encoding of market states
- Bidirectional GRU with hyperspherical projections
- Automatic mixed precision (AMP) support
- Efficient caching mechanism for frequently accessed states

### Data Processing
- Real-time market data preprocessing
- Technical indicator calculation
- Separate normalization for price and indicator features
- Efficient batch processing with GPU acceleration

### Visualization Capabilities
- Interactive 3D market state visualization
- Multiple 2D projections for different perspectives
- Color coding based on various market metrics
- Detailed hover information for state analysis
- Support for both t-SNE and PCA dimensionality reduction

## Usage

### Basic Setup
```python
from market_visualizer import MarketVisualizer

# Initialize visualizer
viz = MarketVisualizer(
    sequence_length=128,
    hidden_size=256,
    scaling_method="minmax",
    batch_size=512,
    use_amp=True
)

# Generate visualizations
viz.visualize_hypersphere(
    n_samples=10000,
    projection_method='tsne',
    color_feature="close",
    hover_features=['open', 'high', 'low', 'volume']
)
```

### Custom Visualization
```python
# Create t-SNE visualization
fig_tsne = viz.visualize_hypersphere(
    projection_method='tsne',
    perplexity=50,
    color_feature='rsi_14',
    create_subplots=True
)

# Save visualization
fig_tsne.write_html("market_visualization_tsne.html")
```

## Performance Optimizations

- CUDA optimizations for GPU acceleration
- Automatic Mixed Precision (AMP) training
- Efficient data caching mechanism
- Multi-worker data loading
- Memory-efficient data processing

## Future Development

- [ ] Integration of clustering algorithms for market regime detection
- [ ] Real-time market data streaming support
- [ ] Enhanced visualization features with pattern recognition
- [ ] Integration with trading execution systems
- [ ] Support for additional technical indicators
- [ ] Multi-timeframe analysis capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.