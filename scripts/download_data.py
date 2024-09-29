# scripts/download_data.py

import argparse
import pandas as pd
from src.data.data_acquisition import BinanceDataProvider  
from src.features.feature_engineer import FeatureEngineer  # Corrected import
from src.utils.utils import setup_logging, get_logger  # Corrected import

def parse_args():
    parser = argparse.ArgumentParser(description="Download Historical Data")
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Candlestick timeframe')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data/raw/', help='Output directory for raw data')
    return parser.parse_args()

def main():
    # Setup logging
    setup_logging()
    logger = get_logger()

    args = parse_args()

    # Initialize Data Provider
    data_provider = BinanceDataProvider(api_key='YOUR_API_KEY', api_secret='YOUR_API_SECRET')

    # Fetch data
    logger.info(f"Downloading data for {args.symbol} from {args.start_date} to {args.end_date}")
    df = data_provider.get_data(symbol=args.symbol, timeframe=args.timeframe, start_date=args.start_date, end_date=args.end_date)

    # Save raw data
    output_file = f"{args.output}{args.symbol.replace('/', '_')}_{args.timeframe}_{args.start_date}_{args.end_date}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Raw data saved to {output_file}")

    # Feature Engineering
    feature_engineer = FeatureEngineer()
    df = feature_engineer.add_technical_indicators(df)
    df = feature_engineer.add_custom_features(df)
    df = df.dropna()

    # Save processed data
    processed_file = f"data/processed/{args.symbol.replace('/', '_')}_{args.timeframe}_{args.start_date}_{args.end_date}_processed.csv"
    df.to_csv(processed_file, index=False)
    logger.info(f"Processed data saved to {processed_file}")

if __name__ == "__main__":
    main()
