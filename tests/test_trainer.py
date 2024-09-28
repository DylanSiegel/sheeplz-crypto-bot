# File: tests/test_trainer.py

import unittest
from unittest.mock import MagicMock
import pandas as pd
from models.trainer import train_model
from models.model import TradingModel
from src.feature_selection import FeatureSelector

class TestTrainer(unittest.TestCase):

    def setUp(self):
        # Sample data
        self.train_df = pd.DataFrame({
            'SMA_20': [95, 96, 97, 98, 99],
            'EMA': [90, 91, 92, 93, 94],
            'RSI': [30, 32, 31, 33, 34],
            'MACD': [1, 1.2, 1.1, 1.3, 1.4],
            'ATR': [0.5, 0.55, 0.6, 0.65, 0.7],
            'pct_change': [0.01, 0.02, -0.01, 0.03, 0.04],
            'volatility': [0.02, 0.025, 0.03, 0.035, 0.04]
        })
        self.target_df = pd.Series([0, 1, 0, 1, 0])

    def test_train_model(self):
        # Mock config
        config = {
            'feature_selection': {
                'threshold': 0.01,
                'max_features': 10
            },
            'model': {
                'batch_size': 2,
                'hidden_size': 64,
                'output_size': 3,
                'learning_rate': 0.001,
                'epochs': 1,
                'model_save_path': './models/checkpoints/'
            }
        }

        # Mock FeatureSelector
        selector = FeatureSelector(threshold=0.01, max_features=10)
        selector.fit_transform = MagicMock(return_value=self.train_df)
        selector.get_selected_features = MagicMock(return_value=self.train_df.columns.tolist())

        with unittest.mock.patch('models.trainer.FeatureSelector', return_value=selector):
            # Mock TradingDataset and DataLoader
            with unittest.mock.patch('models.trainer.TradingDataset') as mock_dataset:
                mock_dataset.return_value = MagicMock()
                with unittest.mock.patch('torch.save') as mock_save:
                    # Mock model
                    with unittest.mock.patch('models.trainer.TradingModel', return_value=MagicMock()):
                        # Run training
                        train_model(config, self.train_df, self.target_df)
                        mock_save.assert_called()

if __name__ == '__main__':
    unittest.main()
