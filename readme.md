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

DATA FOLDER

src/
├── data/
│   ├── raw/
│   │   └── btc_usdt_1m_klines.csv           # Original data
│   │
│   ├── processed/
│   │   ├── btc_1m_normalized.parquet        # Standard normalized features
│   │   ├── btc_1m_hypersphere.parquet       # Hyperspherical features
│   │   └── btc_1m_tensors/                  # Additional transformations
│   │
│   ├── data_preprocessor.py                 # (formerly processor.py)
│   │   - Primary data loading
│   │   - Data cleaning
│   │   - Base feature creation
│   │
│   ├── market_indicators.py                 # (formerly indicators.py)
│   │   - Technical analysis indicators
│   │   - Market microstructure features
│   │   - Volume profiles
│   │
│   ├── feature_transformer.py               # (formerly features.py)
│   │   - Feature normalization
│   │   - Hyperspherical transformations
│   │   - Tensor preparation
│   │
│   └── __init__.py

snippet of data 1 min kline- 
open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ignore
2020-01-01 00:00:00+00:00,7189.43,7190.52,7177.0,7182.44,246.092,2020-01-01 00:00:59.999000+00:00,1767430.16121,336,46.63,334813.1982,0
2020-01-01 00:01:00+00:00,7182.43,7182.44,7178.75,7179.01,70.909,2020-01-01 00:01:59.999000+00:00,509145.78482,140,32.597,234063.27884,0
2020-01-01 00:02:00+00:00,7179.01,7179.01,7175.25,7177.93,99.42,2020-01-01 00:02:59.999000+00:00,713539.55348,148,16.311,117066.92118,0
2020-01-01 00:03:00+00:00,7177.77,7182.6,7177.0,7181.11,69.33,2020-01-01 00:03:59.999000+00:00,497793.35929,104,43.723,313920.02981,0
2020-01-01 00:04:00+00:00,7179.1,7179.1,7172.94,7175.25,97.368,2020-01-01 00:04:59.999000+00:00,698627.39382,193,36.616,262734.68999,0
