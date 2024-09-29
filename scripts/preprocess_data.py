# scripts/preprocess_data.py

import argparse
import pandas as pd
from src.features.feature_engineer import FeatureEngineer  # Corrected import
from src.features.feature_selector import FeatureSelector  # Corrected import
from src.utils.utils import setup_logging, get_logger  # Corrected import

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Raw Data")
    parser.add_argument('--input', type=str, required=True, help='Input raw data CSV file')
    parser.add_argument('--output', type=str, default='data/processed/', help='Output directory for processed data')
    parser.add_argument('--reward_type', type=str, choices=['profit', 'sharpe'], default='profit', help='Type of reward function for feature selection')
    return parser.parse_args()

def main():
    # Setup logging
    setup_logging()
    logger = get_logger()

    args = parse_args()

    # Load raw data
    logger.info(f"Loading raw data from {args.input}")
    df = pd.read_csv(args.input)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Feature Engineering
    feature_engineer = FeatureEngineer()
    df = feature_engineer.add_technical_indicators(df)
    df = feature_engineer.add_custom_features(df)
    df = df.dropna()
    logger.info("Feature engineering completed.")

    # Feature Selection
    feature_selector = FeatureSelector(threshold=0.01, max_features=10)
    
    # Define target variable based on reward type
    if args.reward_type == 'profit':
        target = (df['close'].shift(-1) > df['close']).astype(int).fillna(0).astype(int)
    else:
        # For Sharpe ratio, define a continuous target based on returns
        target = df['close'].pct_change().fillna(0)

    X = df[['SMA_20', 'EMA', 'RSI', 'MACD', 'ATR', 'pct_change', 'volatility']]
    X_selected = feature_selector.fit_transform(X, target)
    logger.info(f"Selected features: {X_selected.columns.tolist()}")

    # Save processed data
    processed_file = f"{args.output}{args.input.split('/')[-1].replace('.csv', '_processed.csv')}"
    processed_df = X_selected.copy()
    processed_df['target'] = target
    processed_df.to_csv(processed_file, index=False)
    logger.info(f"Processed data saved to {processed_file}")

if __name__ == "__main__":
    main()
