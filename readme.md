# n-LNN: A Deep Reinforcement Learning Platform for Trading BTC/USDT Derivatives Futures

This repository contains the code for n-LNN, a deep reinforcement learning (DRL) platform designed for automated trading of Bitcoin/USDT perpetual futures contracts on the Bybit testnet. n-LNN utilizes a novel recurrent neural network architecture with unique properties to handle the complex and volatile nature of the cryptocurrency derivatives market.

## Key Features

* **Novel n-LNN Architecture:**  A custom recurrent neural network (RNN) with Hypersphere Normalization, Spherical Linear Interpolation (SLERP), Eigen Learning Rates, and Scaling Factors for enhanced stability and efficient learning.
* **Advanced DRL Algorithms:**  Implementation of Asynchronous Proximal Policy Optimization (APPO) and support for other DRL algorithms.
* **Deep Exploration Techniques:**  Integration of intrinsic motivation and curiosity-driven exploration to encourage the agent to discover profitable trading strategies.
* **Hierarchical Reinforcement Learning (HRL) Ready:** Designed for future implementation of HRL to handle different levels of decision-making (portfolio allocation, trade execution, risk management).
* **Multi-Agent Reinforcement Learning (MARL) Ready:**  Architecture allows for future development of MARL systems with agents employing diverse strategies.
* **Bybit Testnet Integration:** Connects to the Bybit testnet API for realistic simulated trading.
* **Comprehensive Backtesting and Evaluation:** Tools for evaluating agent performance using metrics like Sharpe ratio, maximum drawdown, and win/loss ratio.
* **Python 12, PyTorch, and Gymnasium:** Built using modern and efficient tools for deep learning and reinforcement learning.


snippet of data 1 min kline- 

open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ignore
2020-01-01 00:00:00+00:00,7189.43,7190.52,7177.0,7182.44,246.092,2020-01-01 00:00:59.999000+00:00,1767430.16121,336,46.63,334813.1982,0
2020-01-01 00:01:00+00:00,7182.43,7182.44,7178.75,7179.01,70.909,2020-01-01 00:01:59.999000+00:00,509145.78482,140,32.597,234063.27884,0
2020-01-01 00:02:00+00:00,7179.01,7179.01,7175.25,7177.93,99.42,2020-01-01 00:02:59.999000+00:00,713539.55348,148,16.311,117066.92118,0
2020-01-01 00:03:00+00:00,7177.77,7182.6,7177.0,7181.11,69.33,2020-01-01 00:03:59.999000+00:00,497793.35929,104,43.723,313920.02981,0
2020-01-01 00:04:00+00:00,7179.1,7179.1,7172.94,7175.25,97.368,2020-01-01 00:04:59.999000+00:00,698627.39382,193,36.616,262734.68999,0

Let me provide a clear, detailed explanation of how we'll design the data pipeline specifically for the n-LNN trading bot, incorporating the hypersphere considerations:

1. **Overall Pipeline Architecture**

A. **Data Flow Stages:**
1. Raw Data Input (1-minute BTC/USDT klines)
2. Initial Preprocessing
3. Feature Engineering
4. Normalization & Scaling
5. Sequence Creation
6. Output Preparation for n-LNN

B. **Key Components:**
- Configuration Manager
- Data Preprocessor
- Feature Generator
- Normalizer/Scaler
- Sequence Builder
- Output Formatter

2. **Detailed Component Breakdown**

A. **Configuration Manager**
- Sequence parameters (length: 60 for 1-hour lookback)
- Prediction horizon (5 minutes forward)
- Feature specifications
- Scaling parameters
- Hardware optimization settings
- n-LNN specific parameters (for coordination)

B. **Data Preprocessor**
- Handles raw OHLCV data
- Cleans outliers
- Validates OHLC relationships
- Handles missing values
- Memory-efficient processing using chunks

C. **Feature Generator**
- Price features (OHLCV)
- Technical indicators:
  * Multiple RSIs (6, 14, 24)
  * EMAs (5, 10, 20, 60, 120)
  * Bollinger Bands
  * Volatility metrics
- Time features:
  * Hour, day, session markers
- All calculations Numba-accelerated

D. **Normalizer/Scaler (Crucial for n-LNN)**
- Primary scaling approach for n-LNN coordination
- Options:
  1. MinMax scaling to [-1, 1] (for tanh activation)
  2. Robust scaling for outlier resistance
  3. Standardization for normal distribution
- Separate scaling for different feature groups
- Scale persistence for consistency

E. **Sequence Builder**
- Creates fixed-length sequences
- Handles temporal alignment
- Incorporates prediction horizon
- Efficient memory management
- Proper train/validation splitting

F. **Output Formatter**
- Prepares data in n-LNN compatible format
- Creates PyTorch datasets/dataloaders
- Manages batch sizes
- Handles GPU memory pinning

3. **n-LNN Specific Considerations**

A. **Normalization Coordination:**
- Pipeline scales features appropriately for n-LNN
- Avoids redundant normalization
- Coordinates with n-LNN's hypersphere operations

B. **Feature Scaling Strategy:**
- Initial scaling in pipeline ([-1, 1] or [0, 1])
- Preserves relationships for n-LNN's hypersphere projections
- Consistent across all features

C. **Data Structure:**
- Sequences shaped for n-LNN's recurrent processing
- Proper temporal alignment for state updates
- Efficient batch processing format

4. **Implementation Benefits**

A. **Performance:**
- Hardware-optimized processing
- Efficient memory usage
- Accelerated computations
- GPU-ready data format

B. **Reliability:**
- Robust data validation
- Consistent scaling
- Error handling
- State preservation

C. **Flexibility:**
- Configurable parameters
- Adaptable feature sets
- Multiple scaling options
- Extensible design

5. **Usage Flow**

```python
# Example usage flow:
1. Initialize pipeline with config
2. Load raw kline data
3. Preprocess and clean
4. Generate features
5. Apply coordinated scaling
6. Create sequences
7. Format for n-LNN
8. Get train/val loaders
```

6. **Key Distinctions**

A. **Pipeline Responsibilities:**
- Data cleaning and preparation
- Feature engineering
- Initial scaling/normalization
- Sequence creation
- Batch preparation

B. **n-LNN Responsibilities:**
- Hypersphere projections
- SLERP/NLERP operations
- Weight normalization
- State updates
- Learning dynamics

This pipeline design ensures that data is properly prepared for the n-LNN model while maintaining clear separation of concerns and optimal performance. Would you like me to elaborate on any specific component or aspect?

src/
├── config/
│   ├── config.py
│   └── __init__.py
├── data/
│   ├── data_preprocessor.py
│   ├── feature_transformer.py
│   ├── market_indicators.py
│   └── __init__.py
│   └── raw/
│       └── btc_usdt_1m_klines.parquet
├── utils/
│   ├── logging.py
│   ├── system_monitor.py
│   └── __init__.py
├── pipeline/
│   ├── __init__.py
│   ├── normalizers.py
│   ├── augmenters.py
│   ├── datasets.py
│   ├── handlers.py
│   ├── metrics.py
│   ├── pipeline.py
│   └── feature_transformer.py
└── main.py
