# Real-Time Cryptocurrency Market Data Machine Learning System

## Overview

This project implements a real-time machine learning system designed to process cryptocurrency market data from Binance Futures WebSocket streams. It leverages a Liquid Neural Network (LNN) model to analyze and predict market trends in real-time, utilizing GPU acceleration for optimized performance.

## Features

- **Real-Time Data Processing:** Connects to Binance Futures WebSocket streams to receive live market data.
- **Liquid Neural Network (LNN):** A custom LNN model tailored for time-series data analysis.
- **GPU Acceleration:** Optimized to leverage NVIDIA 3070 GPU for accelerated computations.
- **Robust Error Handling:** Comprehensive mechanisms to handle and recover from various error scenarios.
- **Scalable Architecture:** Designed to handle multiple symbols and high-frequency data efficiently.
- **Configuration Management:** Centralized configuration using YAML for easy adjustments.
- **Enhanced Logging:** Structured and detailed logging for monitoring and debugging.
- **Unit Testing:** Comprehensive test suite to ensure reliability and correctness.
- **Documentation:** Detailed documentation for setup, usage, and development.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Logging](#logging)
- [Model Management](#model-management)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- **Python Version:** 3.12.6
- **Operating System:** Windows 11
- **Hardware:**
  - NVIDIA GeForce RTX 3070 8GB GPU
  - AMD Ryzen 9 7900X 12-Core Processor (4.7 GHz, 24 logical processors)

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/crypto_ml_system.git
   cd crypto_ml_system
