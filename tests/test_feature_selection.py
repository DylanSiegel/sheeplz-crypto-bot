# File: tests/test_feature_selection.py

import unittest
import pandas as pd
from src.feature_selection import FeatureSelector

class TestFeatureSelector(unittest.TestCase):

    def setUp(self):
        # Sample data
        self.X = pd.DataFrame({
            'SMA_20': [95, 96, 97, 98, 99],
            'EMA': [90, 91, 92, 93, 94],
            'RSI': [30, 32, 31, 33, 34],
            'MACD': [1, 1.2, 1.1, 1.3, 1.4],
            'ATR': [0.5, 0.55, 0.6, 0.65, 0.7],
            'pct_change': [0.01, 0.02, -0.01, 0.03, 0.04],
            'volatility': [0.02, 0.025, 0.03, 0.035, 0.04]
        })
        self.y = pd.Series([0, 1, 0, 1, 0])

    def test_selectfrommodel(self):
        selector = FeatureSelector(method="SelectFromModel", threshold=0.1)
        X_selected = selector.fit_transform(self.X, self.y)
        self.assertTrue(len(X_selected.columns) <= 10)

    def test_rfe(self):
        selector = FeatureSelector(method="RFE", max_features=3)
        X_selected = selector.fit_transform(self.X, self.y)
        self.assertEqual(len(X_selected.columns), 3)

    def test_invalid_method(self):
        selector = FeatureSelector(method="InvalidMethod")
        with self.assertRaises(ValueError):
            selector.fit_transform(self.X, self.y)

if __name__ == '__main__':
    unittest.main()
