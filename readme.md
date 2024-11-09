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