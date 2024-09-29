# AI-Powered Crypto Trading Bot

## Overview

This repository contains an AI-powered cryptocurrency trading bot built using deep reinforcement learning (DRL). The bot serves as a research platform for developing and testing advanced trading strategies. It emphasizes flexibility in reward function design, data ingestion, and multi-agent experimentation.  This project uses Hydra for configuration management and PyTorch for model training.

## Key Features

* **Dynamic Reward Function Framework:** Easily iterate and test various reward strategies to optimize trading performance based on desired metrics (e.g., profit, Sharpe ratio, custom metrics).
* **Flexible Data Ingestion:** Supports multiple data sources, including live market data from exchanges (e.g., Binance), historical data from CSV files, and (potentially) synthetic data generation.
* **Modular Gym Environment:**  A configurable trading environment built using OpenAI Gym, adaptable to different scenarios, asset pairs, and data inputs.
* **Online Learning (Potential):**  Designed for online learning to adapt to evolving market dynamics (implementation in progress).
* **Multi-Agent Framework (Potential):**  Facilitates comparative testing of multiple agents (implementation in progress).
* **Experiment Tracking (with MLflow or WandB):** Log experiments, track metrics, and compare configurations.
* **Hydra Configuration Management:** Uses Hydra for flexible and organized configuration management.
* **PyTorch for Model Training:** Leverages PyTorch for building and training DRL models.

Disclaimer
This project is for educational and research purposes only. Trading cryptocurrencies involves significant risk. Do not use for live trading without a thorough understanding of the risks involved. The developers are not responsible for any financial losses.