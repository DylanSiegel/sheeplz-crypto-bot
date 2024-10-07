# Sheeplz Crypto Bot

A high-frequency crypto data processing bot for BTCUSDT, designed to integrate with a Meta Graph Network (MGN).

## Features

- Connects to MEXC WebSocket for real-time high-frequency data.
- Subscribes to multiple data streams:
  - Order Book Updates
  - Trade Streams
  - Best Bid/Ask (Book Ticker)
  - Kline (Candlestick) Streams
- Processes data in real-time, calculating technical indicators:
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Fibonacci Retracement Levels
- Forwards processed data directly to the Meta Graph Network (MGN) (mocked for now).


sheeplz-crypto-bot/
├── configs/
│   └── .env
├── data/
│   ├── __init__.py
│   ├── error_handler.py
│   ├── indicator_calculations.py
│   ├── data_processor.py
│   └── mexc_websocket_connector.py
├── main.py
├── requirements.txt
└── README.md
