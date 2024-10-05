import asyncio
import logging
import os
from crypto_gmn import CryptoGMN
from config import load_config

async def main():
    # Optionally set the environment variable before running
    # os.environ["ENVIRONMENT"] = "PRODUCTION"

    config = load_config()
    
    async with CryptoGMN(config) as gmn:
        # Example batch of new data points
        new_data = [
            {'c': 150.0, 'v': 1000},
            {'c': 155.0, 'v': 1200},
            {'c': 160.0, 'v': 1500},
            {'c': 158.0, 'v': 1100},
            {'c': 162.0, 'v': 1300},
        ]

        # Update the graph with new data
        await gmn.update_graph(new_data)

        # Retrieve specific indicator data
        rsi_data = gmn.get_data('1m', 'rsi')
        macd_data = gmn.get_data('1m', 'macd')
        fibonacci_data = gmn.get_data('1m', 'fibonacci')

        print(f"RSI Data for 1m: {rsi_data}")
        print(f"MACD Data for 1m: {macd_data}")
        print(f"Fibonacci Data for 1m: {fibonacci_data}")

        # Retrieve all market data
        all_data = gmn.get_all_data()
        print(f"All Market Data: {all_data}")

        # Dynamically add a new indicator (e.g., EMA)
        ema_func = gmn.indicator_factory.create_ema(timeperiod=50)
        gmn.add_indicator('1m', 'ema', ema_func)

        # Update with new data to trigger EMA calculation
        new_data = [
            {'c': 165.0, 'v': 1400},
            {'c': 170.0, 'v': 1600},
        ]
        await gmn.update_graph(new_data)

        ema_data = gmn.get_data('1m', 'ema')
        print(f"EMA Data for 1m: {ema_data}")

        # Fetch real-time data from Binance
        await gmn.fetch_real_time_data(exchange='binance', symbol='BTCUSDT')

        # Access cached indicator data
        cached_rsi = gmn.get_cached_indicator('1m', 'rsi', window=10)
        print(f"Cached RSI Data for 1m (last 10): {cached_rsi}")

if __name__ == "__main__":
    asyncio.run(main())