# File: tests/test_feature_engineering.py

import unittest
import pandas as pd
from src.feature_engineering import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):

    def setUp(self):
        # Sample data
        data = {
            'timestamp': pd.date_range(start='2021-01-01', periods=5, freq='H'),
            'open': [100, 102, 101, 103, 104],
            'high': [105, 106, 103, 107, 108],
            'low': [99, 101, 100, 102, 103],
            'close': [104, 105, 102, 106, 107],
            'volume': [1000, 1500, 1200, 1300, 1400]
        }
        self.df = pd.DataFrame(data)

    def test_add_technical_indicators(self):
        feature_engineer = FeatureEngineer()
        df = feature_engineer.add_technical_indicators(self.df)
        self.assertIn('SMA_20', df.columns)
        self.assertIn('RSI', df.columns)
        self.assertIn('MACD', df.columns)
        self.assertIn('ATR', df.columns)
        self.assertTrue(df['SMA_20'].isnull().all())

    def test_add_custom_features(self):
        feature_engineer = FeatureEngineer()
        df = feature_engineer.add_custom_features(self.df)
        self.assertIn('pct_change', df.columns)
        self.assertIn('volatility', df.columns)
        self.assertEqual(df['pct_change'].iloc[1], 0.02)
        self.assertEqual(df['volatility'].iloc[2], 0.02)

if __name__ == '__main__':
    unittest.main()
