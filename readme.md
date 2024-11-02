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

data/
│
├── raw/
│   ├── btc_1min_data.csv               # Raw 1-minute timeframe data
│   ├── btc_15min_data.csv              # Raw 15-minute timeframe data
│   ├── btc_30min_data.csv              # Raw 30-minute timeframe data
│   ├── btc_4hr_data.csv                # Raw 4-hour timeframe data
│   ├── btc_daily_data.csv              # Raw daily timeframe data
│   └── btc_weekly_data.csv             # Raw weekly timeframe data
│
├── processed/
│   ├── 1min_processed.csv              # Preprocessed and feature-engineered 1-minute data
│   ├── 15min_processed.csv             # Preprocessed and feature-engineered 15-minute data
│   ├── 30min_processed.csv             # Preprocessed and feature-engineered 30-minute data
│   ├── 4hr_processed.csv               # Preprocessed and feature-engineered 4-hour data
│   ├── daily_processed.csv             # Preprocessed and feature-engineered daily data
│   └── weekly_processed.csv            # Preprocessed and feature-engineered weekly data
│
├── feature_engineering/
│   ├── technical_indicators.py         # Script for adding technical indicators
│   ├── multi_timeframe_features.py     # Script for multi-timeframe features
│   ├── frequency_features.py           # Script for Fourier Transform and other frequency features
│   └── normalization.py                # Script for normalizing features
│
├── preprocessing/
│   ├── data_cleaning.py                # Handles missing values, outliers, and basic cleaning
│   ├── time_sync.py                    # Aligns timeframes, fills gaps, and resamples data
│   └── stationarization.py             # Applies log transformations, differencing, etc., for stationarity
│
├── loaders/
│   ├── load_raw_data.py                # Functions to load raw data from various sources
│   ├── load_processed_data.py          # Functions to load processed data ready for model training
│   └── batch_generator.py              # Generates data batches for efficient training in n-LNN
│
└── utils/
    ├── config.py                       # Central configuration file for data paths and preprocessing parameters
    └── logger.py                       # Logging configurations for tracking data processing steps
