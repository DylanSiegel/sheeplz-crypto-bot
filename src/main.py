# src/main.py

import asyncio
from data.acquisition.mexc_data_provider import MexcDataProvider
from data.preprocessing.feature_engineering import FeatureEngineer
from data.storage.data_loader import create_data_loader
from utils.config_manager import ConfigManager

async def main():
    config = ConfigManager("config/base_config.yaml")
    
    # Initialize data provider
    data_provider = MexcDataProvider(
        api_key=config.get_exchange_credentials()['api_key'],
        api_secret=config.get_exchange_credentials()['api_secret']
    )

    # Fetch historical data
    symbol = config.get('exchange.symbol')
    timeframe = config.get('exchange.timeframe')
    start_date = config.get('data.start_date')
    end_date = config.get('data.end_date')

    historical_data = await data_provider.get_historical_data(symbol, timeframe, start_date, end_date)

    # Process features
    feature_engineer = FeatureEngineer()
    processed_data = feature_engineer.process_features(historical_data)

    # Create data loader
    features = processed_data[feature_engineer.get_feature_names()]
    targets = processed_data['close']  # Assuming 'close' price as the target
    data_loader = create_data_loader(features, targets, batch_size=32)

    # Print some information
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Features: {feature_engineer.get_feature_names()}")

    # Fetch real-time data
    real_time_data = await data_provider.get_real_time_data(symbol)
    print(f"Real-time data for {symbol}: {real_time_data}")

    # Get account balance
    balance = await data_provider.get_account_balance()
    print(f"Account balance: {balance}")

    # Close connections
    await data_provider.close()

if __name__ == "__main__":
    asyncio.run(main())