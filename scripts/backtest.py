# scripts/backtest.py

import argparse
import pandas as pd
from src.data_acquisition import BinanceDataProvider
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.trading import TradingExecutor
from models.evaluator import Evaluator
from src.rewards import ProfitReward, SharpeRatioReward
from src.utils import setup_logging, get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest Trading Strategies")
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Candlestick timeframe')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--reward_type', type=str, choices=['profit', 'sharpe'], default='profit', help='Type of reward function')
    return parser.parse_args()

def main():
    # Setup logging
    setup_logging()
    logger = get_logger()

    args = parse_args()

    # Initialize Data Provider
    data_provider = BinanceDataProvider(api_key='YOUR_API_KEY', api_secret='YOUR_API_SECRET')
    logger.info(f"Fetching historical data for {args.symbol} from {args.start_date} to {args.end_date}")
    df = data_provider.get_data(symbol=args.symbol, timeframe=args.timeframe, start_date=args.start_date, end_date=args.end_date)

    # Feature Engineering
    feature_engineer = FeatureEngineer()
    df = feature_engineer.add_technical_indicators(df)
    df = feature_engineer.add_custom_features(df)
    df = df.dropna()

    # Feature Selection
    feature_selector = FeatureSelector(threshold=0.01, max_features=10)
    target = (df['close'].shift(-1) > df['close']).astype(int).fillna(0).astype(int)
    X_selected = feature_selector.fit_transform(df[['SMA_20', 'EMA', 'RSI', 'MACD', 'ATR', 'pct_change', 'volatility']], target)

    # Initialize Reward Function
    if args.reward_type == 'profit':
        reward_function = ProfitReward()
    else:
        reward_function = SharpeRatioReward()

    # Initialize Trading Executor
    trading_executor = TradingExecutor(initial_balance=10000.0, transaction_fee=0.001)
    trade_history = trading_executor.execute_backtest(df, X_selected, target, reward_function)

    # Evaluate Performance
    evaluator = Evaluator(trade_history)
    evaluator.summary()
    evaluator.plot_equity_curve()
    evaluator.plot_drawdown()

if __name__ == "__main__":
    main()
